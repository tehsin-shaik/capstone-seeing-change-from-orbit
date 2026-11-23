"""Tests for dinov3/train_dinov3_lora.py on a tiny random DINOv3 and made-up chips.

No real weights, no Hugging Face login, no real data and no GPU are needed. Run from the code folder:
    python -m pytest -q tests/
They check that the LoRA code is wired correctly and writes the same outputs as the frozen run. The scores
mean nothing: the model is random and the chips are synthetic.
Skipped automatically if PyTorch, transformers or peft is not installed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

CODE = Path(__file__).resolve().parents[1]
for p in (CODE, CODE / "dinov3"):
    sys.path.insert(0, str(p))

import train_dinov3 as T  # noqa: E402
import train_dinov3_lora as L  # noqa: E402
from chipio import DEFAULT_CHANNELS  # noqa: E402

REPO_DINO_CV = CODE.parents[1] / "results" / "mangrove_segmentation" / "dinov3" / "cv_results.csv"
SIZE = 32          # chip size in pixels; x2 upsampling -> 64 px -> a 4 x 4 token grid


def tiny_backbone():
    """The random 4-block, 64-wide DINOv3 that train_dinov3.py uses for --smoke."""
    bb, *_ = T.load_backbone("unused", True, "cpu")
    return bb


def make_chip(path, rng, n_cols):
    lab = np.zeros((SIZE, SIZE), np.uint8); lab[:, :n_cols] = 1
    m = lab == 1
    img = rng.normal(0.05, 0.01, (8, SIZE, SIZE)).astype(np.float32)
    for name, hi, lo in (("B08", 0.30, 0.04), ("NDVI", 0.70, 0.05), ("MNDWI", -0.40, 0.30)):
        img[DEFAULT_CHANNELS.index(name)] = np.where(m, hi, lo) + rng.normal(0, 0.02, m.shape)
    np.savez_compressed(path, image=img, label_v4019=lab, label_v4112=lab, cloud=np.zeros(m.shape, bool),
                        nodata=np.zeros(m.shape, bool), channels=np.array(DEFAULT_CHANNELS))


@pytest.fixture
def trainval(tmp_path, monkeypatch):
    """5 sites = 5 folds, 3 dates each, far apart; one hazy scene. Same layout as Train_Val_Dataset."""
    root = tmp_path / "Train_Val_Dataset"; (root / "chips").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows, qa = [], []
    for k in range(1, 6):
        for j, date in enumerate(["2020-02-01", "2020-05-01", "2020-11-01"]):
            cid = f"s{k}_{date}_r00000_c00000"
            make_chip(root / "chips" / f"{cid}.npz", rng, n_cols=8 + 2 * k + j)
            x0 = 100_000 * k
            rows.append({"chip_id": cid, "site": f"s{k}", "date": date, "year": 2020, "split": "train", "fold": k,
                         "bounds_epsg32640": f"{x0},0,{x0 + 2560},2560", "path": f"chips/{cid}.npz"})
            qa.append({"site": f"s{k}", "date": date, "water_b04_p5": 0.09 if (k, j) == (2, 1) else 0.01})
    pd.DataFrame(rows).to_csv(root / "catalog.csv", index=False)
    qa_path = tmp_path / "QA_download.csv"; pd.DataFrame(qa).to_csv(qa_path, index=False)
    monkeypatch.setattr(T.DP, "qa_csv", lambda: qa_path)      # where the frozen script looks for the haze table
    return root, qa_path


def lora_args(root, qa, out, *extra):
    return L.parse(["--smoke", "--force", "--cpu", "--out", str(out), "--train-val", str(root), "--qa", str(qa),
                    "--epochs", "2", "--patience", "2", *extra])


# ------------------------------------------------------------------------------------------------ LoRA wiring
def test_default_targets_rank_and_trainable_count():
    model, info = L.add_lora(tiny_backbone())                      # defaults: r=8, alpha=16, q+v, all 4 blocks
    assert (info["lora_r"], info["lora_alpha"], info["lora_scaling"]) == (8, 16, 2.0)
    assert info["lora_targets"] == ["q_proj", "v_proj"] and info["lora_modules"] == 8 and info["lora_blocks"] == 4
    assert info["trainable_backbone_params"] == 8 * 8 * (64 + 64)  # modules x rank x (inputs + outputs)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)      # every original weight is frozen


def test_rank_alpha_targets_and_layers_are_configurable():
    _, info = L.add_lora(tiny_backbone(), r=4, alpha=4, targets=["q_proj", "k_proj", "v_proj", "o_proj"], last_n=2)
    assert (info["lora_r"], info["lora_alpha"], info["lora_scaling"]) == (4, 4, 1.0)
    assert info["lora_modules"] == 4 * 2 and info["lora_blocks"] == 2
    assert info["trainable_backbone_params"] == 8 * 4 * (64 + 64)
    names = L.resolve_targets(tiny_backbone(), ["q_proj"], last_n=1)
    assert len(names) == 1 and L.layer_index(names[0]) == 3       # the last of blocks 0..3
    with pytest.raises(SystemExit):
        L.resolve_targets(tiny_backbone(), ["no_such_layer"])
    with pytest.raises(SystemExit):
        L.resolve_targets(tiny_backbone(), ["q_proj"], last_n=9)


def test_lora_model_starts_identical_to_the_frozen_model():
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        frozen = T.patch_tokens(tiny_backbone().eval(), x, 16, False)
        model, _ = L.add_lora(tiny_backbone())
        lora = T.patch_tokens(model.eval(), x, 16, False)
    assert frozen.shape == (2, 64, 4, 4)
    assert torch.allclose(frozen, lora, atol=1e-6)                 # B starts at zero: no change yet


@pytest.mark.parametrize("grad_checkpoint", [False, True])
def test_training_changes_only_lora_and_the_decoder(grad_checkpoint):
    torch.manual_seed(0)
    model, _ = L.add_lora(tiny_backbone(), grad_checkpoint=grad_checkpoint)
    head = T.FusionHead(64)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW([{"params": head.parameters(), "lr": 1e-3},
                             {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-2}])
    model.train(); head.train()
    for _ in range(3):
        x, spec = torch.randn(2, 3, 64, 64), torch.randn(2, 8, SIZE, SIZE)
        y = (torch.rand(2, SIZE, SIZE) > 0.7).float()
        opt.zero_grad()
        T.loss_fn(head(T.patch_tokens(model, x, 16, False), spec, y.shape[-2:]), y, torch.ones_like(y)).backward()
        assert all(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
        opt.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all("lora_" in n for n in changed)          # the pretrained weights did not move
    assert any("lora_B" in n for n in changed)


# ------------------------------------------------------------------------------------------------ whole run
def test_cpu_is_refused_without_force(trainval, tmp_path):
    root, qa = trainval
    a = L.parse(["--smoke", "--cpu", "--out", str(tmp_path / "o"), "--train-val", str(root), "--qa", str(qa)])
    with pytest.raises(SystemExit):
        L.run(a)


def test_weight_updates_follow_batch_not_gpu_batch(trainval, tmp_path):
    root, qa = trainval
    a = lora_args(root, qa, tmp_path / "o")
    cat = T.load_cat(root); data = T.load_all(cat, root)
    run_ = T.Runner(a, data, "cpu")
    ids = list(cat.chip_id)[:10]
    mean, std = T.fold_stats(ids, data)
    steps = {}
    for gpu_batch in (4, 1):
        model, _ = L.add_lora(tiny_backbone()); head = T.FusionHead(64)

        class Counting(torch.optim.AdamW):
            n = 0

            def step(self, *args, **kw):
                Counting.n += 1
                return super().step(*args, **kw)

        opt = Counting(list(head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=1e-3)
        loss = L.train_epoch(run_, head, model, opt, torch.amp.GradScaler("cuda", enabled=False), ids, mean, std,
                             np.random.default_rng(0), batch=4, gpu_batch=gpu_batch, amp=False)
        assert np.isfinite(loss)
        steps[gpu_batch] = Counting.n
    assert steps == {4: 3, 1: 3}                                   # 10 chips, 4 per update -> 3 updates either way


def test_lora_run_writes_the_same_outputs_as_the_frozen_run(trainval, tmp_path):
    root, qa = trainval
    out = tmp_path / "runs"
    frozen = T.run(T.parse(["--smoke", "--cpu", "--out", str(out), "--train-val", str(root),
                            "--cache-dir", str(tmp_path / "cache"), "--epochs", "2"]))
    lora = L.run(lora_args(root, qa, out, "--batch", "4", "--gpu-batch", "2"))
    assert (frozen.name, lora.name) == ("SMOKE_frozen_smoke", "SMOKE_lora_smoke")

    def files(d):
        return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}

    assert files(lora) - files(frozen) == {"lora_vs_frozen.csv"}   # everything the frozen run writes, plus one table
    assert files(frozen) <= files(lora)
    for f in ("cv_results.csv", "fold_fits.csv", "history.csv", "per_chip.csv", "score_catalog.csv"):
        assert list(pd.read_csv(lora / f).columns) == list(pd.read_csv(frozen / f).columns), f
    if REPO_DINO_CV.exists():                                      # and the same as the real results in the repo
        assert list(pd.read_csv(lora / "cv_results.csv").columns[:12]) == list(pd.read_csv(REPO_DINO_CV).columns[:12])

    # same folds, same training / validation / held-out chips, same scored pixels
    same = ["variant", "fold", "held_site", "held_chips", "train_chips", "inner_fold", "inner_chips"]
    pd.testing.assert_frame_equal(pd.read_csv(lora / "fold_fits.csv")[same], pd.read_csv(frozen / "fold_fits.csv")[same])
    cv_l, cv_f = pd.read_csv(lora / "cv_results.csv"), pd.read_csv(frozen / "cv_results.csv")
    pd.testing.assert_frame_equal(cv_l[["fold", "held_site", "chips", "mangrove_px"]],
                                  cv_f[["fold", "held_site", "chips", "mangrove_px"]])
    assert len(cv_l) == 5 and set(cv_l.model) == {"DINOv3_LoRA"} and set(cv_f.model) == {"DINOv3"}
    assert cv_l.iou.between(0, 1).all()
    fits = pd.read_csv(lora / "fold_fits.csv").set_index("fold")
    assert fits.loc[1, "train_chips"] == 8                         # folds 2, 4, 5 (9 chips) minus the hazy s2 scene

    cfg_l, cfg_f = json.load(open(lora / "config.json")), json.load(open(frozen / "config.json"))
    assert set(cfg_f) <= set(cfg_l)
    assert (cfg_l["mode"], cfg_l["lora_r"], cfg_l["lora_alpha"], cfg_l["batch"], cfg_l["gpu_batch"]) == ("lora", 8, 16, 4, 2)
    assert cfg_l["lora_targets"] == ["q_proj", "v_proj"] and cfg_l["mixed_precision"] is False

    ck = torch.load(lora / "dinov3_fold1.pt", weights_only=False)
    assert ck["lora"] and all("lora_" in n for n in ck["lora"]) and ck["lora_config"]["lora_r"] == 8
    assert set(torch.load(frozen / "dinov3_fold1.pt", weights_only=False)) <= set(ck)

    md = (lora / "RESULTS.md").read_text(encoding="utf-8")
    assert "## LoRA settings" in md and "LoRA r=8, alpha=16 on q_proj, v_proj in all blocks" in md
    assert "## LoRA against the frozen run" in md and "on attention q/v" not in md
    assert len(pd.read_csv(lora / "lora_vs_frozen.csv")) == 5


def test_custom_settings_get_their_own_run_folder(trainval, tmp_path):
    root, qa = trainval
    out = L.run(lora_args(root, qa, tmp_path / "runs", "--folds", "1", "--epochs", "1", "--lora-r", "4",
                          "--lora-targets", "q_proj", "k_proj", "v_proj", "o_proj", "--lora-last-n", "2"))
    assert out.name == "SMOKE_lora_smoke_r4_a8_k-o-q-v_last2"
    cfg = json.load(open(out / "config.json"))
    assert (cfg["lora_r"], cfg["lora_alpha"], cfg["lora_modules"], cfg["lora_last_n"]) == (4, 8, 8, 2)
    assert "No frozen run found" in (out / "RESULTS.md").read_text(encoding="utf-8")


def test_bench_runs_without_data_or_weights():
    res = L.bench(L.parse(["--bench", "--cpu", "--bench-arch", "vits", "--bench-chip", str(SIZE), "--batch", "2",
                           "--gpu-batch", "1", "--bench-steps", "1"]))
    assert res["tokens"] == 16 and res["seconds_per_update"] > 0 and res["lora_modules"] == 12 * 2


def test_rgb_only_decoders_do_not_use_the_other_five_channels():
    """--no-spectral and --rgb-branch: changing NIR, SWIR and the indices must not change the output."""
    torch.manual_seed(0)
    tok, spec = torch.randn(2, 64, 4, 4), torch.randn(2, 8, SIZE, SIZE)
    other = spec.clone(); other[:, 3:] += 5.0                     # B08, B11, B12, NDVI, MNDWI
    rgb = spec.clone(); rgb[:, 0] += 5.0                          # B02
    for flags in (["--no-spectral"], ["--rgb-branch"]):
        head = L.make_head(64, L.parse(flags)).eval()
        with torch.no_grad():
            assert torch.equal(head(tok, spec, (SIZE, SIZE)), head(tok, other, (SIZE, SIZE))), flags
    with torch.no_grad():
        assert not torch.equal(head(tok, spec, (SIZE, SIZE)), head(tok, rgb, (SIZE, SIZE)))      # the RGB branch is live
        default = L.make_head(64, L.parse([])).eval()
        assert not torch.equal(default(tok, spec, (SIZE, SIZE)), default(tok, other, (SIZE, SIZE)))
    assert head.spec[0][0].in_channels == 3 and type(default) is T.FusionHead


def test_rgb_branch_keeps_the_rest_of_the_decoder_start_and_gets_its_own_folder(trainval, tmp_path):
    torch.manual_seed(3); default = L.make_head(64, L.parse([]))
    torch.manual_seed(3); rgb = L.make_head(64, L.parse(["--rgb-branch"]))
    for part in ("proj", "fuse"):                                 # same random start outside the branch
        a, b = getattr(default, part).state_dict(), getattr(rgb, part).state_dict()
        assert all(torch.equal(a[n], b[n]) for n in a)
    root, qa = trainval
    out = L.run(lora_args(root, qa, tmp_path / "runs", "--folds", "1", "--epochs", "1", "--rgb-branch"))
    assert out.name == "SMOKE_lora_smoke_rgbbranch" and json.load(open(out / "config.json"))["rgb_branch"] is True
    assert "over B04, B03, B02 only" in (out / "RESULTS.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        L.parse(["--rgb-branch", "--no-spectral"])
