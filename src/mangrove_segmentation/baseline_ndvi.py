"""Baseline B1: NDVI threshold with water exclusion.

    mangrove = (NDVI > t) AND (MNDWI < w)

t and w are grid-searched on VALIDATION chips only (maximising pixel-pooled IoU over
valid pixels). The chosen thresholds are saved, then applied to any split.

Usage:
    python baseline_ndvi.py fit     --catalog catalog.csv --chips chips/ --label v4019 --out models/ndvi.json
    python baseline_ndvi.py predict --catalog catalog.csv --chips chips/ --model models/ndvi.json \
        --split test --pred-out preds/ndvi
    python evaluate.py --pred preds/ndvi --catalog catalog.csv --chips chips/ --split test --out results/ndvi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chipio import channel_index, chip_path, get_label, load_catalog, load_chip, save_mask

T_GRID = np.round(np.arange(0.0, 0.81, 0.02), 3)
W_GRID = [float(x) for x in np.round(np.arange(-0.5, 0.51, 0.05), 3)] + [float("inf")]  # inf = no water rule


def indices(chip: dict) -> tuple[np.ndarray, np.ndarray]:
    """NDVI and MNDWI, taken from the chip's own channels if present, else computed from bands."""
    img = chip["image"].astype(np.float32)
    try:
        ndvi = img[channel_index(chip, "NDVI")]
    except ValueError:
        red, nir = img[channel_index(chip, "B04")], img[channel_index(chip, "B08")]
        ndvi = (nir - red) / (nir + red + 1e-6)
    try:
        mndwi = img[channel_index(chip, "MNDWI")]
    except ValueError:
        green, swir = img[channel_index(chip, "B03")], img[channel_index(chip, "B11")]
        mndwi = (green - swir) / (green + swir + 1e-6)
    return np.nan_to_num(ndvi, nan=-1.0), np.nan_to_num(mndwi, nan=1.0)


def predict_chip(chip: dict, t: float, w: float) -> np.ndarray:
    ndvi, mndwi = indices(chip)
    return (ndvi > t) & (mndwi < w) & chip["valid"]


def fit(catalog, chips_dir, label="v4019", max_px=5_000_000, seed=0) -> dict:
    cat = load_catalog(catalog)
    val = cat[cat["split"] == "val"]
    if val.empty:
        raise SystemExit("no validation chips in catalog (split == 'val')")
    nd, mw, y = [], [], []
    for _, r in val.iterrows():
        chip = load_chip(chip_path(chips_dir, r))
        ndvi, mndwi = indices(chip)
        truth, labelled = get_label(chip, label)
        v = chip["valid"] & labelled
        nd.append(ndvi[v]); mw.append(mndwi[v]); y.append(truth[v])
    nd, mw, y = np.concatenate(nd), np.concatenate(mw), np.concatenate(y)
    if len(y) == 0:
        raise SystemExit(f"validation chips have no usable GMW {label} labels")
    if len(y) > max_px:  # subsample uniformly to keep the search fast
        keep = np.random.default_rng(seed).choice(len(y), max_px, replace=False)
        nd, mw, y = nd[keep], mw[keep], y[keep]
    best = {"iou": -1.0}
    grid = []
    for w in W_GRID:
        water_ok = mw < w
        for t in T_GRID:
            p = (nd > t) & water_ok
            tp = int(np.sum(p & y)); fp = int(np.sum(p & ~y)); fn = int(np.sum(~p & y))
            iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
            grid.append((float(t), w, iou))
            if iou > best["iou"]:
                best = {"t": float(t), "w": w, "iou": iou}
    return {"model": "ndvi_threshold", "rule": "NDVI > t and MNDWI < w", "label": label,
            "t_ndvi": best["t"], "w_mndwi": best["w"], "val_iou": best["iou"],
            "val_chips": int(len(val)), "val_pixels_used": int(len(y)),
            "grid_top5": sorted(grid, key=lambda g: -g[2])[:5]}


def predict(catalog, chips_dir, model: dict, split: str, pred_out) -> int:
    cat = load_catalog(catalog)
    if split != "all":
        cat = cat[cat["split"] == split]
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(chips_dir, r))
        save_mask(pred_out, r["chip_id"], predict_chip(chip, model["t_ndvi"], model["w_mndwi"]))
    return len(cat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--catalog", required=True); f.add_argument("--chips", required=True)
    f.add_argument("--label", default="v4019", choices=["v4019", "v4112"])
    f.add_argument("--out", required=True, help="thresholds .json")
    p = sub.add_parser("predict")
    p.add_argument("--catalog", required=True); p.add_argument("--chips", required=True)
    p.add_argument("--model", required=True); p.add_argument("--split", default="test")
    p.add_argument("--pred-out", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        m = fit(a.catalog, a.chips, a.label)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(m, indent=1, default=str))
        print(f"t_ndvi={m['t_ndvi']}  w_mndwi={m['w_mndwi']}  val IoU={m['val_iou']:.3f}")
    else:
        m = json.loads(Path(a.model).read_text())
        m["w_mndwi"] = float(m["w_mndwi"])
        n = predict(a.catalog, a.chips, m, a.split, a.pred_out)
        print(f"wrote {n} masks to {a.pred_out}")


if __name__ == "__main__":
    main()
