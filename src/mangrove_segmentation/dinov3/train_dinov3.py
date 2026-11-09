"""DINOv3 route for mangrove segmentation: a frozen foundation model plus light tuning.

Same protocol as the baselines and the U-Net, so the numbers are directly comparable:
  * Train_Val_Dataset only, 2020 chips, label GMW v4.0.19 (10 m). The Test_Dataset is never read.
  * 5 leave-one-site-out folds. Hazy scenes are dropped from training (variant b, dark-water red p5 > 0.05).
  * A 1-chip (2,560 m) buffer between training chips and held-out chips (Eastern Mangroves / Jubail touch).
  * The inner validation fold (the same one B2 used: first of 3, 4, 5, 2, 1 that isn't held out) drives early
    stopping and the decision threshold. The held-out fold is scored with code/evaluate.py, exactly like B1/B2/U-Net.

Two modes
  frozen  Stage A (CPU-friendly). DINOv3 is frozen. Its patch features are extracted ONCE per chip and cached
          (float16) in 6_Work/dinov3_cache, then only a small fusion decoder is trained. Fast on a CPU.
  lora    Stage B (GPU). LoRA adapters (r=8) on the attention q/v projections, trained end to end with the same
          decoder. Refuses to run on a CPU unless --force.

How the 10 m Sentinel-2 chip is shown to DINOv3 (an RGB model)
  RGB = B04, B03, B02, stretched per chip (2-98 %) to [0, 1], normalised with the model's own mean/std, and upsampled
  x2 (256 -> 512 px) so the 16-px patches give a 32 x 32 token grid (one token = 80 m on the ground).
  DINOv3 can't see near-infrared or SWIR, so the decoder also gets a light conv branch over all 8 spectral channels
  at full 10 m resolution (switch it off with --no-spectral to measure what it adds).

Examples (from the SDP folder, in the Anaconda Prompt with the 'mangrove' environment active)
  python 2_Code/dinov3/train_dinov3.py --mode frozen                       # SAT ViT-L, all 5 folds
  python 2_Code/dinov3/train_dinov3.py --mode frozen --no-spectral         # ablation: DINO RGB only
  python 2_Code/dinov3/train_dinov3.py --model-id facebook/dinov3-vits16-pretrain-lvd1689m   # small model, low RAM
  python 2_Code/dinov3/train_dinov3.py --mode lora --epochs 30             # needs an NVIDIA GPU
  python 2_Code/dinov3/train_dinov3.py --smoke --out <temp folder>         # random tiny model: tests the code only
Needs a Hugging Face login with the DINOv3 licence accepted (see HOW_TO_GET_DINOV3.md), except --smoke.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dino_paths as DP  # noqa: E402  (also puts 2_Code on sys.path)
from chipio import chip_path, get_label, load_chip, load_mask, save_mask  # noqa: E402
import evaluate as EV  # noqa: E402

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"
SMALL_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
HAZE_P5, BUFFER_M, INNER_PREF = 0.05, 2560, [3, 4, 5, 2, 1]
REFL, INDEX_EPS = slice(0, 6), 0.01
VARIANT = "b_no_haze"
BAR_B2 = 0.704

DEFAULTS = {
    "frozen": dict(epochs=40, patience=10, batch=8, lr_head=1e-3, lr_lora=0.0),
    "lora": dict(epochs=30, patience=8, batch=4, lr_head=1e-3, lr_lora=1e-4),
}


# ============================================================================ data (same rules as train_unet.py)
def seed_all(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load_cat(tv: Path) -> pd.DataFrame:
    cat = pd.read_csv(tv / "catalog.csv")
    cat = cat[cat["year"] == 2020].copy()
    qa = pd.read_csv(DP.qa_csv())[["site", "date", "water_b04_p5"]]
    cat = cat.merge(qa, on=["site", "date"], how="left")
    cat["hazy"] = cat["water_b04_p5"] > HAZE_P5
    b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
    cat[["x0", "y0", "x1", "y1"]] = b.values
    cat["fold"] = cat["fold"].astype(int)
    return cat.reset_index(drop=True)


def drop_near(train: pd.DataFrame, other: pd.DataFrame, buf=BUFFER_M) -> pd.DataFrame:
    if train.empty or other.empty:
        return train
    keep = []
    for _, r in train.iterrows():
        dx = np.maximum(0, np.maximum(other.x0.values - r.x1, r.x0 - other.x1.values))
        dy = np.maximum(0, np.maximum(other.y0.values - r.y1, r.y0 - other.y1.values))
        keep.append(np.maximum(dx, dy).min() >= buf)
    return train[np.array(keep)]


def load_all(cat: pd.DataFrame, tv: Path) -> dict:
    out = {}
    for _, r in cat.iterrows():
        c = load_chip(chip_path(tv, r))
        truth, labelled = get_label(c, "v4019")
        out[r.chip_id] = (c["image"].astype(np.float32), truth.astype(np.uint8), (c["valid"] & labelled))
    return out


def fold_stats(ids, data):
    s = np.zeros(8); ss = np.zeros(8); n = 0
    for i in ids:
        img, _, use = data[i]
        x = img[:, use].astype(np.float64)
        s += x.sum(1); ss += (x ** 2).sum(1); n += x.shape[1]
    mean = s / max(n, 1)
    return mean.astype(np.float32), np.sqrt(np.maximum(ss / max(n, 1) - mean ** 2, 1e-12)).astype(np.float32)


def recompute_indices(img):
    b03, b04, b08, b11 = img[1], img[2], img[3], img[4]
    img[6] = np.clip((b08 - b04) / np.maximum(b08 + b04, INDEX_EPS), -1, 1)
    img[7] = np.clip((b03 - b11) / np.maximum(b03 + b11, INDEX_EPS), -1, 1)
    return img


def normalise(img, mean, std):
    return np.nan_to_num((img - mean[:, None, None]) / std[:, None, None], nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================ DINOv3 backbone
def read_norm(model_id: str):
    """The model's own image mean/std from its preprocessor_config.json (falls back to ImageNet values)."""
    try:
        from huggingface_hub import hf_hub_download
        cfg = json.load(open(hf_hub_download(model_id, "preprocessor_config.json")))
        return tuple(cfg["image_mean"]), tuple(cfg["image_std"])
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"could not read the preprocessor config of {model_id} ({e}); using ImageNet mean/std")
        return IMAGENET_MEAN, IMAGENET_STD


def load_backbone(model_id: str, smoke: bool, device: str):
    """Returns (model, mean tensor, std tensor, patch size, feature channels)."""
    if smoke:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        cfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128,
                              patch_size=16, image_size=512, num_register_tokens=4)
        torch.manual_seed(0)
        model, (mean, std) = DINOv3ViTModel(cfg), (IMAGENET_MEAN, IMAGENET_STD)
    else:
        from transformers import AutoModel
        try:
            model = AutoModel.from_pretrained(model_id)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"\nCould not load {model_id}:\n  {e}\n\nDINOv3 weights are gated. Accept the licence on "
                f"https://huggingface.co/{model_id} and log in once with `huggingface-cli login` "
                f"(see 2_Code/dinov3/HOW_TO_GET_DINOV3.md). To test the code without weights, add --smoke.") from None
        mean, std = read_norm(model_id)
    model.eval().to(device)
    return (model, torch.tensor(mean).float().view(3, 1, 1), torch.tensor(std).float().view(3, 1, 1),
            int(model.config.patch_size), int(model.config.hidden_size))


def rgb_tensor(img: np.ndarray, mean: torch.Tensor, std: torch.Tensor, scale: int) -> torch.Tensor:
    """[8,H,W] reflectance chip -> normalised RGB tensor [3, H*scale, W*scale] (B04, B03, B02, 2-98 % stretch)."""
    rgb = np.stack([img[2], img[1], img[0]]).astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        v = rgb[c][np.isfinite(rgb[c])]
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0.0, 1.0))
        out[c] = np.clip((rgb[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    t = torch.from_numpy(np.nan_to_num(out))[None]
    if scale != 1:
        t = F.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False)
    return ((t[0] - mean) / std).float()


def patch_tokens(model, x: torch.Tensor, patch: int, last4: bool) -> torch.Tensor:
    """x [B,3,H,W] -> patch features [B, C, H/patch, W/patch]. Patch tokens are the last N tokens (after CLS and
    register tokens). With last4, the last four layers are averaged."""
    o = model(pixel_values=x, output_hidden_states=last4)
    h, w = x.shape[-2] // patch, x.shape[-1] // patch
    seq = torch.stack(o.hidden_states[-4:]).mean(0) if last4 else o.last_hidden_state
    pt = seq[:, -h * w:, :]
    return pt.transpose(1, 2).reshape(x.shape[0], -1, h, w)


def cache_features(model, ids, data, mean, std, patch, scale, last4, cache: Path, device, batch=4) -> dict:
    """Extract (or load) the frozen DINOv3 features of every chip once. Files: <cache>/<chip_id>.npy (float16)."""
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (cache / f"{i}.npy").exists()]
    if todo:
        print(f"extracting DINOv3 features for {len(todo)} chips -> {cache}", flush=True)
        t0 = time.time()
        with torch.inference_mode():
            for j in range(0, len(todo), batch):
                b = todo[j:j + batch]
                x = torch.stack([rgb_tensor(data[i][0], mean, std, scale) for i in b]).to(device)
                f = patch_tokens(model, x, patch, last4).float().cpu().numpy().astype(np.float16)
                for i, ff in zip(b, f):
                    np.save(cache / f"{i}.npy", ff)
                done = j + len(b)
                if done % 20 < batch or done == len(todo):
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)} chips, {el/done:.1f} s/chip, about {el/done*(len(todo)-done)/60:.0f} min left",
                          flush=True)
    return {i: np.load(cache / f"{i}.npy") for i in ids}


# ============================================================================ decoder
def conv3(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class FusionHead(nn.Module):
    """DINOv3 tokens (1x1 projection, upsampled to 10 m) + a light conv branch over the 8 spectral channels."""
    def __init__(self, c_tok: int, spectral: bool = True, d: int = 64, s: int = 24, f: int = 32):
        super().__init__()
        self.spectral = spectral
        self.proj = nn.Sequential(nn.Conv2d(c_tok, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        if spectral:
            self.spec = nn.Sequential(conv3(8, s), conv3(s, s))
        self.fuse = nn.Sequential(conv3(d + (s if spectral else 0), f), conv3(f, f), nn.Conv2d(f, 1, 1))

    def forward(self, tok, spec, size):
        f = F.interpolate(self.proj(tok), size=size, mode="bilinear", align_corners=False)
        if self.spectral:
            f = torch.cat([f, self.spec(spec)], 1)
        return self.fuse(f)[:, 0]


def loss_fn(logits, y, m, pos_weight=4.0):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logits) * m
    inter = (p * y * m).sum()
    return bce + 1 - (2 * inter + 1) / (p.sum() + (y * m).sum() + 1)


# ============================================================================ training
class Runner:
    """Holds the backbone/features for one run, and builds batches for either mode."""

    def __init__(self, a, data, device):
        self.a, self.data, self.device = a, data, device
        self.backbone, self.mean_rgb, self.std_rgb, self.patch, self.c_tok = load_backbone(a.model_id, a.smoke, device)
        self.feats = None

    def prepare_frozen(self, ids):
        tag = "smoke_random" if self.a.smoke else self.a.model_id.split("/")[-1]
        cache = Path(self.a.cache_dir) if self.a.cache_dir else DP.DINO_CACHE
        cache = cache / f"{tag}_x{self.a.scale}{'_last4' if self.a.last4 else ''}"
        self.feats = cache_features(self.backbone, ids, self.data, self.mean_rgb, self.std_rgb, self.patch,
                                    self.a.scale, self.a.last4, cache, self.device)

    def fresh_lora(self):
        """A new backbone with LoRA adapters for each fold (the frozen weights are reloaded so folds don't leak)."""
        from peft import LoraConfig, get_peft_model
        bb, *_ = load_backbone(self.a.model_id, self.a.smoke, self.device)
        cfg = LoraConfig(r=self.a.lora_r, lora_alpha=2 * self.a.lora_r, lora_dropout=0.05,
                         target_modules=["q_proj", "v_proj"])
        bb = get_peft_model(bb, cfg)
        bb.print_trainable_parameters()
        return bb

    def batch(self, ids, mean, std, rng, augment, lora_bb=None):
        toks, specs, ys, ms, rgbs = [], [], [], [], []
        for i in ids:
            img, y, use = self.data[i]
            im = img.copy()
            if augment:
                im[REFL] += rng.uniform(-0.02, 0.02)
                im = recompute_indices(im)
            tok = None if lora_bb is not None else torch.from_numpy(self.feats[i].astype(np.float32))
            rgb = rgb_tensor(img, self.mean_rgb, self.std_rgb, self.a.scale) if lora_bb is not None else None
            sp = torch.from_numpy(normalise(im, mean, std)).float()
            yy, mm = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(use.astype(np.float32))
            if augment and self.a.aug == "flip":           # flip everything together (features flipped as a grid)
                for dim, flip in ((-1, rng.random() < 0.5), (-2, rng.random() < 0.5)):
                    if flip:
                        sp, yy, mm = sp.flip(dim), yy.flip(dim), mm.flip(dim)
                        tok = tok.flip(dim) if tok is not None else None
                        rgb = rgb.flip(dim) if rgb is not None else None
            toks.append(tok); rgbs.append(rgb); specs.append(sp); ys.append(yy); ms.append(mm)
        dev = self.device
        spec = torch.stack(specs).to(dev); y = torch.stack(ys).to(dev); m = torch.stack(ms).to(dev)
        if lora_bb is not None:
            tok = patch_tokens(lora_bb, torch.stack(rgbs).to(dev), self.patch, self.a.last4)
        else:
            tok = torch.stack(toks).to(dev)
        return tok, spec, y, m

    @torch.no_grad()
    def predict(self, head, ids, mean, std, lora_bb=None, bs=4):
        head.eval()
        if lora_bb is not None:
            lora_bb.eval()
        out = {}
        for j in range(0, len(ids), bs):
            b = ids[j:j + bs]
            tok, spec, _, _ = self.batch(b, mean, std, None, False, lora_bb)
            p = torch.sigmoid(head(tok, spec, spec.shape[-2:])).float().cpu().numpy()
            out.update(dict(zip(b, p)))
        return out


def iou_at(probs, ids, data, t):
    tp = fp = fn = 0
    for i in ids:
        _, y, use = data[i]
        pr = probs[i] > t; yy = y.astype(bool)
        tp += np.sum(pr & yy & use); fp += np.sum(pr & ~yy & use); fn += np.sum(~pr & yy & use)
    return tp / max(tp + fp + fn, 1)


def run(a) -> Path:
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    if a.mode == "lora" and device == "cpu" and not a.force:
        raise SystemExit("LoRA fine-tuning on a CPU would take days. Use an NVIDIA GPU (or Colab), "
                         "or run --mode frozen. Add --force only to test the code path.")
    cfg = {**DEFAULTS[a.mode]}
    for k in ("epochs", "patience", "batch"):
        if getattr(a, k):
            cfg[k] = getattr(a, k)
    tv = Path(a.train_val) if a.train_val else DP.TRAIN_VAL
    name = a.run_name or (f"{'SMOKE_' if a.smoke else ''}{a.mode}_{'smoke' if a.smoke else a.model_id.split('/')[-1]}"
                          f"{'_nospec' if a.no_spectral else ''}")
    out = Path(a.out) / name if a.out else DP.DINO_RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | mode {a.mode} | model {'random tiny (smoke)' if a.smoke else a.model_id}\n"
          f"Train/Val: {tv}\nresults : {out}", flush=True)

    cat = load_cat(tv)
    folds = [int(k) for k in (a.folds or sorted(cat.fold.unique()))]
    rng0 = np.random.default_rng(0)
    if a.max_chips:   # smoke/dry-run: keep a few chips per fold
        cat = pd.concat([g.sample(min(len(g), a.max_chips), random_state=0) for _, g in cat.groupby("fold")],
                        ignore_index=True)
    data = load_all(cat, tv)
    run_ = Runner(a, data, device)
    if a.mode == "frozen":
        run_.prepare_frozen(list(cat.chip_id))
        for p in run_.backbone.parameters():
            p.requires_grad_(False)

    fits, hist = [], []
    for k in folds:
        seed_all(k)
        t0 = time.time()
        held = cat[cat.fold == k]
        train = drop_near(cat[(cat.fold != k) & (~cat.hazy)], held)
        inner = next(f for f in INNER_PREF if f != k and f in set(train.fold))
        iv = train[train.fold == inner]
        tr = drop_near(train[train.fold != inner], iv)
        if tr.empty:
            tr = train[train.fold != inner]
        tr_ids, iv_ids = list(tr.chip_id), list(iv.chip_id)
        mean, std = fold_stats(tr_ids, data)
        lora_bb = run_.fresh_lora() if a.mode == "lora" else None
        head = FusionHead(run_.c_tok, spectral=not a.no_spectral).to(device)
        groups = [{"params": head.parameters(), "lr": cfg["lr_head"]}]
        if lora_bb is not None:
            groups.append({"params": [p for p in lora_bb.parameters() if p.requires_grad], "lr": cfg["lr_lora"]})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
        rng = np.random.default_rng(k)
        best, best_ep, bad, best_state, ep = -1.0, 0, 0, None, 0
        for ep in range(1, cfg["epochs"] + 1):
            te = time.time(); head.train()
            if lora_bb is not None:
                lora_bb.train()
            ids = np.array(tr_ids); rng.shuffle(ids); losses = []
            for j in range(0, len(ids), cfg["batch"]):
                tok, spec, y, m = run_.batch(ids[j:j + cfg["batch"]], mean, std, rng, True, lora_bb)
                opt.zero_grad()
                loss = loss_fn(head(tok, spec, y.shape[-2:]), y, m)
                loss.backward(); opt.step(); losses.append(loss.item())
            sched.step()
            probs = run_.predict(head, iv_ids, mean, std, lora_bb)
            viou = iou_at(probs, iv_ids, data, 0.5)
            hist.append({"fold": k, "epoch": ep, "train_loss": float(np.mean(losses)), "inner_val_iou@0.5": float(viou),
                         "seconds": round(time.time() - te, 1)})
            print(f"fold {k} ep {ep:3d} loss {np.mean(losses):.4f} innerIoU {viou:.4f} {time.time()-te:.0f}s", flush=True)
            if viou > best:
                best, best_ep, bad = viou, ep, 0
                best_state = {"head": {n: v.detach().clone() for n, v in head.state_dict().items()},
                              "lora": ({n: v.detach().clone() for n, v in lora_bb.state_dict().items() if "lora_" in n}
                                       if lora_bb is not None else None)}
            else:
                bad += 1
            if bad >= cfg["patience"]:
                break
        head.load_state_dict(best_state["head"])
        if lora_bb is not None and best_state["lora"]:
            lora_bb.load_state_dict(best_state["lora"], strict=False)
        torch.save({"head": best_state["head"], "lora": best_state["lora"], "mean": mean, "std": std,
                    "model_id": a.model_id, "spectral": not a.no_spectral}, out / f"dinov3_fold{k}.pt")
        probs = run_.predict(head, iv_ids, mean, std, lora_bb)
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        ious = [iou_at(probs, iv_ids, data, t) for t in grid]
        thr = float(grid[int(np.argmax(ious))])
        hp = run_.predict(head, list(held.chip_id), mean, std, lora_bb)
        for cid, p in hp.items():
            save_mask(out / "preds" / f"DINO_{VARIANT}", cid, p > thr)
        rec = {"variant": VARIANT, "fold": k, "held_site": ",".join(sorted(held.site.unique())),
               "held_chips": len(held), "train_chips": len(tr_ids), "inner_fold": inner, "inner_chips": len(iv_ids),
               "best_epoch": best_ep, "epochs_run": ep, "inner_iou@0.5": round(best, 4), "threshold": thr,
               "inner_iou@thr": round(float(max(ious)), 4), "minutes": round((time.time() - t0) / 60, 1)}
        fits.append(rec); print(rec, flush=True)
        pd.DataFrame(fits).to_csv(out / "fold_fits.csv", index=False)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        del lora_bb
    json.dump({"mode": a.mode, "model_id": "random tiny (smoke)" if a.smoke else a.model_id, "smoke": a.smoke,
               "spectral_branch": not a.no_spectral, "scale": a.scale, "last4": a.last4, "aug": a.aug,
               "lora_r": a.lora_r if a.mode == "lora" else None, "device": device, **cfg},
              open(out / "config.json", "w"), indent=1)
    score(out, tv, cat, a)
    return out


# ============================================================================ scoring, figures, report
def score(out: Path, tv: Path, cat: pd.DataFrame, a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fits, hist = pd.read_csv(out / "fold_fits.csv"), pd.read_csv(out / "history.csv")
    cfgj = json.load(open(out / "config.json"))
    pred = out / "preds" / f"DINO_{VARIANT}"
    done = {p.stem for p in pred.glob("*")}
    sc = cat.assign(split="f" + cat.fold.astype(str))
    sc = sc[sc.chip_id.isin(done)]          # only chips that were predicted (all of them in a full run)
    scp = out / "score_catalog.csv"; sc.to_csv(scp, index=False)
    rows, chips = [], []
    for k in sorted(fits.fold):
        res, pc = EV.evaluate(pred, scp, tv, split=f"f{k}", label="v4019", reps=1000, seed=0)
        r = res["pooled"]
        rows.append({"model": "DINOv3", "variant": VARIANT, "fold": int(k),
                     "held_site": fits.query("fold==@k").held_site.iloc[0], "chips": r["n_chips"],
                     "mangrove_px": r["mangrove_px"], **{x: round(r[x], 4) for x in EV.METRICS},
                     "iou_ci95_lo": round(r["iou_ci95"][0], 4), "iou_ci95_hi": round(r["iou_ci95"][1], 4),
                     **{f"iou_{s}": round(d["iou"], 4) for s, d in res["per_site"].items()}})
        chips.append(pc.assign(model="DINOv3", variant=VARIANT, fold=int(k)))
    cv = pd.DataFrame(rows); cv.to_csv(out / "cv_results.csv", index=False)
    pc = pd.concat(chips); pc.to_csv(out / "per_chip.csv", index=False)
    d = cv.set_index("fold")

    # comparison models (whatever exists)
    comp = {}
    bfile = DP.BASELINES / "cv_results.csv"
    if bfile.exists():
        b = pd.read_csv(bfile).query("variant==@VARIANT")
        comp["B1 NDVI threshold"] = b.query("model=='B1'").set_index("fold")
        comp["B2 random forest"] = b.query("model=='B2'").set_index("fold")
    if DP.unet_cv_results().exists():
        comp["U-Net"] = pd.read_csv(DP.unet_cv_results()).set_index("fold")
    label = f"DINOv3 {cfgj['mode']}{' (RGB only)' if not cfgj['spectral_branch'] else ''}"

    def msd(s):
        return f"{s.mean():.3f} ± {s.std():.3f}" if len(s) > 1 else f"{s.mean():.3f}"

    head_rows = [{"Model": label + (" [SMOKE TEST]" if a.smoke else ""),
                  **{m: msd(d[m]) for m in ("iou", "f1", "precision", "recall")}}]
    for n, df in comp.items():
        df2 = df.loc[df.index.intersection(d.index)]
        head_rows.append({"Model": n, **{m: msd(df2[m]) for m in ("iou", "f1", "precision", "recall")}})
    head = pd.DataFrame(head_rows)
    per = []
    b2 = comp.get("B2 random forest")
    for k in d.index:
        row = {"Fold": k, "Held-out site": d.loc[k, "held_site"].replace("_", " "),
               "DINOv3 IoU (95% CI)": f"{d.loc[k,'iou']:.3f} ({d.loc[k,'iou_ci95_lo']:.3f}–{d.loc[k,'iou_ci95_hi']:.3f})"}
        if b2 is not None and k in b2.index:
            ov = not (d.loc[k, "iou_ci95_lo"] > b2.loc[k, "iou_ci95_hi"] or b2.loc[k, "iou_ci95_lo"] > d.loc[k, "iou_ci95_hi"])
            row.update({"B2 IoU": f"{b2.loc[k,'iou']:.3f}", "DINOv3 − B2": f"{d.loc[k,'iou']-b2.loc[k,'iou']:+.3f}",
                        "CIs overlap": "yes" if ov else "no"})
        for n in ("U-Net", "B1 NDVI threshold"):
            if n in comp and k in comp[n].index:
                row[f"{n.split(' (')[0].split(' ')[0]} IoU"] = f"{comp[n].loc[k,'iou']:.3f}"
        per.append(row)
    per = pd.DataFrame(per)

    figs = out / "figs"; figs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 4))
    x = np.arange(len(d)); models = list(comp.items()) + [(label, d)]
    colors = ["#2f6aa3", "#c9571c", "#2e8b57", "#7b3fa0"]
    w = 0.8 / len(models)
    for i, (n, df) in enumerate(models):
        vals = df.reindex(d.index)["iou"].values
        lo = vals - df.reindex(d.index)["iou_ci95_lo"].values; hi = df.reindex(d.index)["iou_ci95_hi"].values - vals
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, yerr=[lo, hi], capsize=2,
               color=colors[i % len(colors)] if n != label else "#7b3fa0", label=n)
    ax.set_xticks(x); ax.set_xticklabels([f"F{k}\n{d.loc[k,'held_site'].replace('_',' ')[:22]}" for k in d.index], fontsize=8)
    ax.axhline(BAR_B2, color="#c9571c", ls="--", lw=0.8); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_ylabel("IoU on held-out site (GMW v4.0.19)")
    ax.set_title(("SMOKE TEST (random weights, meaningless): " if a.smoke else "") + "leave-one-site-out IoU per fold")
    plt.tight_layout(); plt.savefig(figs / "iou_per_fold.png", dpi=130); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, g in hist.groupby("fold"):
        axs[0].plot(g.epoch, g.train_loss, label=f"F{k}"); axs[1].plot(g.epoch, g["inner_val_iou@0.5"], label=f"F{k}")
    axs[0].set_title("Training loss (BCE + Dice)"); axs[1].set_title("Inner-validation IoU at 0.5"); axs[1].set_ylim(0, 1)
    for ax_ in axs:
        ax_.set_xlabel("epoch"); ax_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(figs / "training_curves.png", dpi=130); plt.close()

    ex = pc.sort_values("iou").iloc[[len(pc) // 4, len(pc) // 2, 3 * len(pc) // 4]] if len(pc) >= 3 else pc
    b2pred = DP.BASELINES / "preds" / f"B2_{VARIANT}"
    ncol = 4 if b2pred.exists() else 3
    fig, axs = plt.subplots(len(ex), ncol, figsize=(2.8 * ncol, 2.8 * len(ex)), squeeze=False)
    for i, (_, r) in enumerate(ex.iterrows()):
        crow = cat[cat.chip_id == r.chip_id].iloc[0]
        chip = load_chip(chip_path(tv, crow)); img = chip["image"]
        rgb = np.nan_to_num(np.clip(np.stack([img[2], img[1], img[0]], -1) / 0.25, 0, 1))
        truth, _ = get_label(chip, "v4019")
        panels = [(rgb, f"{crow.site} {crow.date}"), (truth, "GMW v4.0.19 label")]
        if ncol == 4:
            try:
                panels.append((load_mask(b2pred, r.chip_id), "B2 random forest"))
            except Exception:  # noqa: BLE001
                panels.append((np.zeros_like(truth), "B2 (missing)"))
        panels.append((load_mask(pred, r.chip_id), f"DINOv3 (IoU {r.iou:.2f})"))
        for j, (im, t) in enumerate(panels):
            axs[i, j].imshow(im, cmap=None if j == 0 else "Greens", vmin=0, vmax=1, interpolation="nearest")
            axs[i, j].set_title(t, fontsize=8); axs[i, j].axis("off")
    if a.smoke:
        fig.suptitle("SMOKE TEST: random weights, predictions are meaningless", color="#c0392b")
    plt.tight_layout(); plt.savefig(figs / "examples.png", dpi=110); plt.close()

    mu = d.iou.mean()
    verdict = ("SMOKE TEST: random weights, the numbers mean nothing. This run only proves the code works."
               if a.smoke else (f"**Yes: beats the random forest** (DINOv3 {mu:.3f} vs B2 {BAR_B2})" if mu > BAR_B2
                                else f"**No: does not beat the random forest yet** (DINOv3 {mu:.3f} vs B2 {BAR_B2})"))
    md = f"""# DINOv3 ({cfgj['mode']}), leave-one-site-out CV{'  [SMOKE TEST]' if a.smoke else ''}

Train_Val_Dataset only (Test_Dataset never read). 2020 chips, label GMW v4.0.19 (10 m), hazy scenes dropped from training
(variant b), the same 5 folds, 1-chip buffer and scoring code as B1, B2 and the U-Net.

**Verdict:** {verdict}

## Headline (mean ± sd over folds)

{head.to_markdown(index=False)}

## Per fold

{per.to_markdown(index=False)}

## Setup

- **Backbone:** {cfgj['model_id']}, {'frozen' if cfgj['mode'] == 'frozen' else f"LoRA r={cfgj['lora_r']} on attention q/v"};
  RGB = B04/B03/B02 stretched 2–98 % per chip, upsampled x{cfgj['scale']} (token = {160 // cfgj['scale']} m){', last 4 layers averaged' if cfgj['last4'] else ''}.
- **Decoder:** 1x1 projection of the DINOv3 tokens to 64 channels, upsampled to 10 m,
  {'+ a light conv branch over the 8 spectral channels (B02–B12, NDVI, MNDWI)' if cfgj['spectral_branch'] else 'RGB features only (no spectral branch)'}, then 2 conv layers (32 channels).
- **Training:** BCE (pos_weight 4) + Dice, AdamW, cosine schedule, up to {cfgj['epochs']} epochs, early stopping (patience {cfgj['patience']})
  on inner-validation IoU, batch {cfgj['batch']}, augmentation: brightness ±0.02{' + flips' if cfgj['aug'] == 'flip' else ''}. Device: {cfgj['device']}.
- **Threshold:** chosen on the inner-validation fold (grid 0.05–0.95): {', '.join(f"F{r.fold} {r.threshold}" for r in fits.itertuples())}.
- **Runtime:** {fits.minutes.sum():.0f} min of training (feature extraction not included).

Files: `cv_results.csv`, `per_chip.csv`, `fold_fits.csv`, `history.csv`, `config.json`, `figs/`, `dinov3_fold*.pt`, `preds/`.
"""
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    print(f"\n{head.to_string(index=False)}\n\nreport: {out / 'RESULTS.md'}", flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}; small: {SMALL_MODEL}")
    ap.add_argument("--mode", choices=["frozen", "lora"], default="frozen")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--epochs", type=int); ap.add_argument("--patience", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--scale", type=int, default=2, help="upsampling of the 256 px chip before DINOv3 (2 -> 32x32 tokens)")
    ap.add_argument("--last4", action="store_true", help="average the last 4 layers instead of the last one")
    ap.add_argument("--no-spectral", action="store_true", help="ablation: DINOv3 RGB features only")
    ap.add_argument("--aug", choices=["none", "flip"], default="flip")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--cache-dir", default=None, help="default 6_Work/dinov3_cache")
    ap.add_argument("--train-val", default=None, help="default 6_Work/datasets/Train_Val_Dataset")
    ap.add_argument("--out", default=None, help="parent folder of the run; default 4_Results/DINOv3")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--max-chips", type=int, default=None, help="keep only N chips per fold (quick tests)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    ap.add_argument("--force", action="store_true", help="allow --mode lora on a CPU (code test only)")
    ap.add_argument("--smoke", action="store_true", help="tiny random DINOv3 instead of the real weights (tests code only)")
    a = ap.parse_args(argv)
    if a.smoke and not a.out:
        ap.error("--smoke needs --out <temp folder>, so smoke results never land in 4_Results")
    return a


if __name__ == "__main__":
    run(parse())
