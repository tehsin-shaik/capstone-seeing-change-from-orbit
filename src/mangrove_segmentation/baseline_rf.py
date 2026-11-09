"""Baseline B2: per-pixel random forest on all chip channels (bands + indices).

Training pixels: valid pixels from TRAIN chips, class-balanced, capped (default 200k).
Light tuning on VALIDATION chips: a few depth / leaf settings, then the probability
threshold, all chosen by pixel-pooled IoU. Fixed seed throughout.

Usage:
    python baseline_rf.py fit     --catalog catalog.csv --chips chips/ --label v4019 --out models/rf.joblib
    python baseline_rf.py predict --catalog catalog.csv --chips chips/ --model models/rf.joblib \
        --split test --pred-out preds/rf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from chipio import chip_path, get_label, load_catalog, load_chip, save_mask

SETTINGS = [{"max_depth": None, "min_samples_leaf": 1},
            {"max_depth": 20, "min_samples_leaf": 5},
            {"max_depth": 12, "min_samples_leaf": 10}]
THRESHOLDS = np.round(np.arange(0.2, 0.81, 0.05), 2)


def pixels(chip: dict) -> np.ndarray:
    img = np.nan_to_num(chip["image"].astype(np.float32), nan=0.0)
    return img.reshape(img.shape[0], -1).T  # (H*W, C)


def collect(cat, chips_dir, label, cap, rng, balanced):
    """Sample valid pixels. balanced=True → up to cap/2 per class; else uniform up to cap."""
    X, y = [], []
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(chips_dir, r))
        truth, labelled = get_label(chip, label)
        v = (chip["valid"] & labelled).ravel()
        X.append(pixels(chip)[v]); y.append(truth.ravel()[v])
    X, y = np.concatenate(X), np.concatenate(y)
    if len(y) == 0:
        raise SystemExit(f"no usable GMW {label} labels in these chips")
    if balanced:
        keep = []
        for cls in (True, False):
            idx = np.flatnonzero(y == cls)
            keep.append(rng.choice(idx, min(len(idx), cap // 2), replace=False))
        keep = np.concatenate(keep)
    else:
        keep = rng.choice(len(y), min(len(y), cap), replace=False)
    return X[keep], y[keep]


def iou(p, t):
    tp = np.sum(p & t); fp = np.sum(p & ~t); fn = np.sum(~p & t)
    return float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0


def fit(catalog, chips_dir, label="v4019", cap=200_000, val_cap=500_000, seed=0, n_estimators=200):
    rng = np.random.default_rng(seed)
    cat = load_catalog(catalog)
    tr, va = cat[cat["split"] == "train"], cat[cat["split"] == "val"]
    if tr.empty or va.empty:
        raise SystemExit("need both train and val chips in the catalog")
    Xtr, ytr = collect(tr, chips_dir, label, cap, rng, balanced=True)
    Xva, yva = collect(va, chips_dir, label, val_cap, rng, balanced=False)  # natural class mix
    best = None
    for s in SETTINGS:
        rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed, **s).fit(Xtr, ytr)
        prob = rf.predict_proba(Xva)[:, list(rf.classes_).index(True)]
        for th in THRESHOLDS:
            score = iou(prob > th, yva)
            if best is None or score > best["val_iou"]:
                best = {"settings": s, "threshold": float(th), "val_iou": score, "rf": rf}
    rf = best.pop("rf")
    meta = {"model": "random_forest", "label": label, "seed": seed, "n_estimators": n_estimators,
            "train_pixels": int(len(ytr)), "val_pixels": int(len(yva)), **best}
    return rf, meta


def predict(catalog, chips_dir, bundle, split, pred_out):
    rf, th = bundle["rf"], bundle["meta"]["threshold"]
    cat = load_catalog(catalog)
    if split != "all":
        cat = cat[cat["split"] == split]
    pos = list(rf.classes_).index(True)
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(chips_dir, r))
        v = chip["valid"]
        mask = np.zeros(v.shape, bool)
        if v.any():
            mask[v] = rf.predict_proba(pixels(chip)[v.ravel()])[:, pos] > th
        save_mask(pred_out, r["chip_id"], mask)
    return len(cat)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--catalog", required=True); f.add_argument("--chips", required=True)
    f.add_argument("--label", default="v4019", choices=["v4019", "v4112"])
    f.add_argument("--cap", type=int, default=200_000); f.add_argument("--seed", type=int, default=0)
    f.add_argument("--out", required=True, help="model .joblib (a .json summary is written beside it)")
    p = sub.add_parser("predict")
    p.add_argument("--catalog", required=True); p.add_argument("--chips", required=True)
    p.add_argument("--model", required=True); p.add_argument("--split", default="test")
    p.add_argument("--pred-out", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        rf, meta = fit(a.catalog, a.chips, a.label, a.cap, seed=a.seed)
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"rf": rf, "meta": meta}, out)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"settings={meta['settings']} threshold={meta['threshold']} val IoU={meta['val_iou']:.3f}")
    else:
        n = predict(a.catalog, a.chips, joblib.load(a.model), a.split, a.pred_out)
        print(f"wrote {n} masks to {a.pred_out}")


if __name__ == "__main__":
    main()
