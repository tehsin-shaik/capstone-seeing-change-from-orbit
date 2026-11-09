"""Tests for evaluate.py and the baselines on synthetic chips with known answers.

Run from the code/ folder:  python -m pytest -q tests/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as ev  # noqa: E402
from chipio import DEFAULT_CHANNELS, save_mask  # noqa: E402


def make_chip(path, label, valid=None, ndvi=None):
    h, w = label.shape
    img = np.zeros((len(DEFAULT_CHANNELS), h, w), np.float32)
    if ndvi is not None:
        img[DEFAULT_CHANNELS.index("NDVI")] = ndvi
        img[DEFAULT_CHANNELS.index("MNDWI")] = -0.3
    np.savez(path, image=img, label_v4019=label.astype(np.uint8), label_v4112=label.astype(np.uint8),
             valid=np.ones_like(label, bool) if valid is None else valid,
             channels=np.array(DEFAULT_CHANNELS))


def test_confusion_and_metrics_known_values():
    truth = np.zeros((10, 10), bool); truth[:, :5] = True          # 50 mangrove px
    pred = np.zeros((10, 10), bool); pred[:, :6] = True            # 50 TP + 10 FP
    tp, fp, fn, tn = ev.confusion(pred, truth, np.ones_like(truth))
    assert (tp, fp, fn, tn) == (50, 10, 0, 40)
    m = ev.metrics_from_counts(tp, fp, fn)
    assert m["iou"] == pytest.approx(50 / 60)
    assert m["precision"] == pytest.approx(50 / 60)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(100 / 110)


def test_invalid_pixels_are_ignored():
    truth = np.zeros((10, 10), bool); truth[:, :5] = True
    pred = np.zeros((10, 10), bool); pred[:, :6] = True
    valid = np.ones((10, 10), bool); valid[:, 5] = False            # hide the 10 FP column
    assert ev.confusion(pred, truth, valid) == (50, 0, 0, 40)


def test_prepare_format_nolabel_cloud_nodata_ignored(tmp_path):
    """Chips in prepare.py format: 255 = no label, cloud/nodata masks, path relative to <out>."""
    out = tmp_path / "data"; (out / "chips").mkdir(parents=True)
    lab = np.zeros((10, 10), np.uint8); lab[:, :5] = 1
    lab[:, 9] = 255                                                  # no label: 10 px
    cloud = np.zeros((10, 10), bool); cloud[:, 8] = True             # cloud: 10 px
    nodata = np.zeros((10, 10), bool); nodata[:, 7] = True           # nodata: 10 px
    img = np.zeros((8, 10, 10), np.float32); img[:, 0, 6] = np.nan   # NaN pixel
    np.savez_compressed(out / "chips" / "x.npz", image=img, label_v4019=lab,
                        label_v4112=np.full((10, 10), 255, np.uint8), cloud=cloud, nodata=nodata,
                        channels=np.array(DEFAULT_CHANNELS))
    pd.DataFrame({"chip_id": ["x"], "site": ["s"], "split": ["test"], "path": ["chips/x.npz"]}) \
        .to_csv(out / "catalog.csv", index=False)
    pred = np.ones((10, 10), bool)   # predicts mangrove everywhere
    save_mask(tmp_path / "p", "x", pred)
    for chips_arg in (out, out / "chips"):                           # both --chips forms work
        res, _ = ev.evaluate(tmp_path / "p", out / "catalog.csv", chips_arg, split="test", reps=20)
        # usable cols 0-6, minus the NaN pixel at (0,6): TP = 50, FP = 2*10 - 1 = 19
        assert (res["pooled"]["tp"], res["pooled"]["fp"], res["pooled"]["fn"]) == (50, 19, 0)
    with pytest.raises(SystemExit):   # v4112 is all 255 for this chip -> nothing to score
        ev.evaluate(tmp_path / "p", out / "catalog.csv", out, split="test", label="v4112", reps=20)


def test_bootstrap_ci_collapses_for_identical_chips():
    counts = np.tile([30, 10, 20], (8, 1))
    ci = ev.bootstrap_ci(counts, reps=200, seed=0)
    assert ci["iou"] == pytest.approx([0.5, 0.5])


def test_end_to_end_pooled_and_per_site(tmp_path):
    chips, preds = tmp_path / "chips", tmp_path / "preds"
    chips.mkdir()
    truth = np.zeros((8, 8), bool); truth[:4] = True                # 32 mangrove px per chip
    # chip a (site s1): perfect. chip b (site s1): misses half. chip c (site s2): 32 TP + 32 FP.
    make_chip(chips / "a.npz", truth); save_mask(preds, "a", truth)
    half = truth.copy(); half[2:4] = False
    make_chip(chips / "b.npz", truth); save_mask(preds, "b", half)
    make_chip(chips / "c.npz", truth); save_mask(preds, "c", np.ones((8, 8), bool))
    pd.DataFrame({"chip_id": ["a", "b", "c"], "site": ["s1", "s1", "s2"], "split": ["test"] * 3}) \
        .to_csv(tmp_path / "catalog.csv", index=False)
    res, per_chip = ev.evaluate(preds, tmp_path / "catalog.csv", chips, split="test", reps=200)
    # pooled: TP = 32 + 16 + 32 = 80, FP = 32, FN = 16 -> IoU = 80 / 128
    assert res["pooled"]["iou"] == pytest.approx(80 / 128)
    assert res["per_site"]["s1"]["iou"] == pytest.approx(48 / 64)
    assert res["per_site"]["s2"]["iou"] == pytest.approx(0.5)
    lo, hi = res["pooled"]["iou_ci95"]
    assert lo <= res["pooled"]["iou"] <= hi
    assert list(per_chip["iou"]) == pytest.approx([1.0, 0.5, 0.5])
    md = ev.markdown_table(res, "toy")
    assert "pooled" in md and "0.625" in md


def test_baselines_run_on_synthetic_data(tmp_path):
    import baseline_ndvi as b1
    import baseline_rf as b2
    chips = tmp_path / "chips"; chips.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        lab = np.zeros((16, 16), bool); lab[:, : 4 + 2 * i] = True
        ndvi = np.where(lab, 0.7, 0.05) + rng.normal(0, 0.02, lab.shape)
        make_chip(chips / f"c{i}.npz", lab, ndvi=ndvi)
        rows.append({"chip_id": f"c{i}", "site": f"s{i}", "split": split})
    cat = tmp_path / "catalog.csv"; pd.DataFrame(rows).to_csv(cat, index=False)

    m = b1.fit(cat, chips)
    assert m["val_iou"] == pytest.approx(1.0)
    b1.predict(cat, chips, m, "test", tmp_path / "p_ndvi")
    res, _ = ev.evaluate(tmp_path / "p_ndvi", cat, chips, split="test", reps=50)
    assert res["pooled"]["iou"] == pytest.approx(1.0)

    rf, meta = b2.fit(cat, chips, cap=2000, n_estimators=20)
    b2.predict(cat, chips, {"rf": rf, "meta": meta}, "test", tmp_path / "p_rf")
    res, _ = ev.evaluate(tmp_path / "p_rf", cat, chips, split="test", reps=50)
    assert res["pooled"]["iou"] > 0.95
    json.dumps(meta, default=str)
