"""Shared helpers for the change-map scripts.

The random forest code is not copied here. `baseline_rf.py` and `chipio.py` are imported unchanged from
src/mangrove_segmentation. The cross-validation steps follow the team's script `run_cv.py`, which is not
in this repository (see README.md). They are written out again in `train_fold` below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "mangrove_segmentation"
BASELINE_CV = HERE.parents[1] / "results" / "mangrove_segmentation" / "baselines" / "cv_results.csv"
sys.path.insert(0, str(CODE))

import baseline_rf as B2  # noqa: E402  (the team's random forest code, unchanged)
from chipio import chip_path, get_label, load_chip  # noqa: E402,F401

# run_cv.py widens the forest's threshold grid from 0.80 to 0.95 in exactly this way
B2.THRESHOLDS = np.round(np.arange(0.2, 0.951, 0.05), 2)

LABEL = "v4019"
HAZE_P5 = 0.05          # dark-water red above this = hazy scene (same rule as run_cv.py)
BUFFER_M = 2560         # one chip
INNER_PREF = [3, 4, 5, 2, 1]
SEED = 0
CFG = dict(inner_cap=100_000, inner_val_cap=300_000, inner_trees=100, final_cap=200_000, final_trees=200)
CHIP = 256
PX_HA = 0.01            # one 10 m pixel = 100 m2 = 0.01 ha
VARIANTS = {"a": "a_all_dates", "b": "b_no_haze"}


def add_path_args(ap) -> None:
    ap.add_argument("--train-val", required=True, help="the Train_Val_Dataset folder (catalog.csv, hazy_scenes.csv, chips/)")
    ap.add_argument("--out", required=True, help="folder for the outputs. Keep it outside the repository: "
                                                 "it receives fitted models and rasters as well as the tables")
    ap.add_argument("--qa", default=None, help="haze table with site, date, water_b04_p5 "
                                               "(default: hazy_scenes.csv inside --train-val)")


def ha(n) -> float:
    return round(float(n) * PX_HA, 2)


def load_full_catalog(tv, qa=None) -> pd.DataFrame:
    """Train/Val catalog, all years, with the haze flag and chip corner coordinates."""
    tv = Path(tv)
    cat = pd.read_csv(tv / "catalog.csv")
    q = pd.read_csv(qa if qa else tv / "hazy_scenes.csv")[["site", "date", "water_b04_p5"]].drop_duplicates(["site", "date"])
    cat = cat.merge(q, on=["site", "date"], how="left")
    if cat["water_b04_p5"].isna().any():
        raise SystemExit("some scenes have no haze value in the haze table")
    cat["hazy"] = cat["water_b04_p5"] > HAZE_P5
    b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
    cat[["x0", "y0", "x1", "y1"]] = b.values
    cat["fold"] = cat["fold"].astype(int)
    return cat.reset_index(drop=True)


def drop_near(train: pd.DataFrame, other: pd.DataFrame, buf=BUFFER_M) -> pd.DataFrame:
    """Drop training chips within `buf` metres of any chip in `other` (same rule as run_cv.py)."""
    if train.empty or other.empty:
        return train
    keep = []
    for _, r in train.iterrows():
        dx = np.maximum(0, np.maximum(other.x0.values - r.x1, r.x0 - other.x1.values))
        dy = np.maximum(0, np.maximum(other.y0.values - r.y1, r.y0 - other.y1.values))
        keep.append(np.maximum(dx, dy).min() >= buf)
    return train[np.array(keep)]


def counts(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
    return {"tp": tp, "fp": fp, "fn": fn,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def train_fold(cat20: pd.DataFrame, k: int, tv, work_dir, variant: str = "b") -> dict:
    """One outer fold of the baseline cross-validation for the random forest (B2).

    cat20 is the 2020 part of the catalog. Fold k is held out. Variant b drops hazy scenes from training.
    Training chips within one chip of a held-out chip are dropped. Depth, leaf size and threshold are tuned
    on an inner split (fit on three folds, validate on the first of folds 3, 4, 5, 2, 1 that is not held out).
    The forest is then refitted on all four training folds with the random generator restarted from the seed.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    present = set(int(f) for f in cat20.fold.unique())
    held = cat20[cat20.fold == k]
    pool_all = cat20[cat20.fold != k]
    pool = pool_all[~pool_all.hazy] if variant == "b" else pool_all
    train = drop_near(pool, held)
    assert not set(train["site"]) & set(held["site"])
    inner = next(f for f in INNER_PREF if f != k and f in present)
    iv = train[train.fold == inner]
    it = drop_near(train[train.fold != inner], iv)
    p = work_dir / f"inner_catalog_fold{k}_{variant}.csv"       # B2.fit reads a catalog with a split column
    pd.concat([it.assign(split="train"), iv.assign(split="val")]).to_csv(p, index=False)
    _, meta = B2.fit(p, tv, LABEL, cap=CFG["inner_cap"], val_cap=CFG["inner_val_cap"], seed=SEED,
                     n_estimators=CFG["inner_trees"])
    rng = np.random.default_rng(SEED)
    X, y = B2.collect(train, tv, LABEL, CFG["final_cap"], rng, balanced=True)
    rf = RandomForestClassifier(n_estimators=CFG["final_trees"], n_jobs=-1, random_state=SEED, **meta["settings"]).fit(X, y)
    return {"rf": rf, "threshold": meta["threshold"], "settings": meta["settings"], "held": held,
            "info": {"variant": variant, "held_fold": int(k), "held_sites": sorted(held["site"].unique()),
                     "held_chips": len(held), "hazy_chips_dropped": int(pool_all.hazy.sum()) if variant == "b" else 0,
                     "buffer_chips_dropped": len(pool) - len(train), "train_chips": len(train),
                     "train_sites": sorted(train["site"].unique()), "inner_fold": int(inner),
                     "inner_val_chips": len(iv), "inner_train_chips": len(it),
                     "inner_iou": round(meta["val_iou"], 4), "final_train_pixels": int(len(y))}}


def heldout_scores(rf, th, held: pd.DataFrame, tv) -> dict:
    """Pixel-pooled scores on the held-out 2020 chips against GMW v4.0.19: per site, and '__fold__' for all of them."""
    pos = list(rf.classes_).index(True)
    parts = {}
    for _, r in held.iterrows():
        chip = load_chip(chip_path(tv, r))
        v = chip["valid"]
        mask = np.zeros(v.shape, bool)
        if v.any():
            mask[v] = rf.predict_proba(B2.pixels(chip)[v.ravel()])[:, pos] > th
        truth, labelled = get_label(chip, LABEL)
        use = v & labelled                       # the pixels evaluate.py scores
        parts.setdefault(r["site"], []).append((mask[use], truth[use]))
    out = {s: counts(np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for s, v in parts.items()}
    allp = [x for v in parts.values() for x in v]
    out["__fold__"] = counts(np.concatenate([a for a, _ in allp]), np.concatenate([b for _, b in allp]))
    return out


def published_fold_iou(k: int, variant: str = "b"):
    """IoU of the committed B2 run for this fold (results/mangrove_segmentation/baselines), or None."""
    if not BASELINE_CV.exists():
        return None
    cv = pd.read_csv(BASELINE_CV)
    cv = cv[(cv["model"] == "B2") & (cv["variant"] == VARIANTS[variant]) & (cv["fold"] == k)]
    return float(cv["iou"].iloc[0]) if len(cv) else None
