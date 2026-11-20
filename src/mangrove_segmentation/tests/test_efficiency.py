"""Tests for the small U-Net presets and efficiency.py, on tiny made-up chips (no real data needed).

Run from the code folder:  python -m pytest -q tests/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import efficiency as eff  # noqa: E402
from chipio import DEFAULT_CHANNELS, save_mask  # noqa: E402


def make_chip(path, label, size=16):
    """Chip in prepare.py format. The image doesn't matter here, only the label."""
    img = np.full((8, size, size), 0.1, np.float32)
    np.savez_compressed(path, image=img, label_v4019=label, label_v4112=label,
                        cloud=np.zeros(label.shape, bool), nodata=np.zeros(label.shape, bool),
                        channels=np.array(DEFAULT_CHANNELS))


@pytest.fixture
def fake_run(tmp_path):
    """Two folds with two chips each, plus a finished 'gpu_b4' run whose predictions are known."""
    tv, results, work = tmp_path / "tv", tmp_path / "results", tmp_path / "work"
    (tv / "chips").mkdir(parents=True)
    rows = []
    lab = np.zeros((16, 16), np.uint8); lab[:, :8] = 1               # left half is mangrove
    preds = work / "runs" / "Unet_gpu_b4" / "preds" / "UNET_b_no_haze"
    for fold, site in ((1, "site_a"), (2, "site_b")):
        for n in range(2):
            cid = f"{site}_2020-01-01_r{n:05d}_c00000"
            make_chip(tv / "chips" / f"{cid}.npz", lab)
            rows.append(dict(chip_id=cid, site=site, date="2020-01-01", year=2020, fold=fold,
                             path=f"chips/{cid}.npz"))
            pred = lab.copy()
            if fold == 2:
                pred[:, 8:12] = 1                                        # fold 2 over-predicts: IoU 8/12
            save_mask(preds, cid, pred.astype(bool))
    pd.DataFrame(rows).to_csv(tv / "catalog.csv", index=False)
    out = results / "Unet_gpu_b4"; out.mkdir(parents=True)
    pd.DataFrame(dict(fold=[1, 2], held_site=["site_a", "site_b"], minutes=[2.0, 4.0],
                      predict_ms_per_chip=[3.0, 5.0], peak_gpu_mb=[None, None])).to_csv(out / "fold_fits.csv", index=False)
    json.dump({"base": 4, "params": 122273, "device": "cpu"}, open(out / "config.json", "w"))
    return tv, results, work, tmp_path / "tmp"


def test_score_preset_gives_the_expected_iou(fake_run):
    summary, cv = eff.score_preset("gpu_b4", *fake_run)
    assert list(cv["iou"]) == pytest.approx([1.0, 8 / 12], abs=1e-4)
    assert summary["mean_iou"] == pytest.approx((1 + 8 / 12) / 2, abs=1e-4)
    assert summary["train_min_per_fold"] == 3.0 and summary["predict_ms_per_chip"] == 4.0
    assert np.isnan(summary["peak_gpu_mb"])                                # CPU run, no GPU memory
    assert set(cv["model"]) == {"UNET_gpu_b4"} and set(cv["variant"]) == {"b_no_haze"}


def test_older_runs_without_timing_columns_still_work(fake_run):
    tv, results, work, tmp = fake_run
    f = results / "Unet_gpu_b4" / "fold_fits.csv"
    pd.read_csv(f)[["fold", "held_site", "minutes"]].to_csv(f, index=False)
    summary, _ = eff.score_preset("gpu_b4", tv, results, work, tmp)
    assert np.isnan(summary["predict_ms_per_chip"])


def test_small_presets_only_change_the_width():
    pytest.importorskip("torch")
    import train_unet as U
    sizes = []
    for name, base in (("gpu", 32), ("gpu_b16", 16), ("gpu_b8", 8), ("gpu_b4", 4)):
        cfg = U.PRESETS[name]
        assert cfg["base"] == base
        assert {k: v for k, v in cfg.items() if k != "base"} == {k: v for k, v in U.PRESETS["gpu"].items() if k != "base"}
        sizes.append(sum(p.numel() for p in U.UNet(8, base).parameters()))
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 7_764_481                       # the gpu model in results/mangrove_segmentation/unet


def test_paired_vs_committed_lines_up_with_the_real_folds():
    committed = eff.REPO_RESULTS / "unet" / "cv_results.csv"
    if not committed.exists():
        pytest.skip("committed results not present")
    full = pd.read_csv(committed).query("model == 'UNET' and variant == 'b_no_haze'")
    small = full.assign(model="UNET_gpu_b8", iou=full["iou"] - 0.02)
    folds, pairs = eff.paired_vs_committed({"gpu_b8": small})
    assert list(folds.columns) == ["held_site", "U-Net (full)", "Random forest", "U-Net gpu_b8"]
    row = pairs[(pairs.model_a == "U-Net (full)") & (pairs.model_b == "U-Net gpu_b8")].iloc[0]
    assert row.mean_diff == pytest.approx(0.02) and row.a_wins == 5


def test_paired_vs_committed_refuses_mismatched_folds():
    committed = eff.REPO_RESULTS / "unet" / "cv_results.csv"
    if not committed.exists():
        pytest.skip("committed results not present")
    full = pd.read_csv(committed).query("model == 'UNET' and variant == 'b_no_haze'")
    wrong = full.assign(held_site=list(reversed(full["held_site"])))
    with pytest.raises(ValueError, match="different folds"):
        eff.paired_vs_committed({"gpu_b8": wrong})
