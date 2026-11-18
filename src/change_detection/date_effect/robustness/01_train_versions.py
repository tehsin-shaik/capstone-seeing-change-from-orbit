"""Train the held-out random forests for versions B (brightness augmentation) and C (indices only), plus a check
run with no change that must reproduce version A.

The training steps are those of `train_fold` in src/change_detection/common.py: hold one fold
out, drop hazy scenes, drop chips within one chip of a held-out chip, tune depth / leaf size / threshold on an inner
split, refit on all four training folds. `baseline_rf.py` and `chipio.py` are imported unchanged. Two things are swapped
in from here, in memory only, without touching the repository:
    * for B, the chip loader returns altered copies of a chip when the catalog row is a copy
    * for C, the feature function returns NDVI and MNDWI only
What was fixed beforehand is in ../README.md.

Writes <work>/forests/<version>_fold<k>.joblib, <work>/forests/fits.csv and <work>/forests/augmentation_draws.csv.

Usage: python 01_train_versions.py --train-val <Train_Val_Dataset> --work <work folder>
           --lines <image_differences results>/correction_lines.csv --a-forests <image_differences work>/forests
"""
import argparse
import sys
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=str(Path(__file__).resolve().parents[2]), help="folder that holds common.py (default: src/change_detection)")
ap.add_argument("--train-val", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--lines", required=True)
ap.add_argument("--a-forests", required=True)
ap.add_argument("--versions", nargs="+", default=["Acheck", "B", "C"])
a = ap.parse_args()
sys.path.insert(0, a.code)
import common as C  # noqa: E402
B2 = C.B2

tv = Path(a.train_val)
if "Train_Val_Dataset" not in str(tv):
    raise SystemExit("only the Train_Val_Dataset folder may be used")
work = Path(a.work)
(work / "forests").mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
N_COPIES = 3
INDEX_EPS = 0.01

# ---------------------------------------------------------------- the real between-date lines
L = pd.read_csv(a.lines)
L = L[L["site"] != "al_nouf"]
ok = L.groupby(["site", "date"])["gain_could_be_fitted"].transform("all")
L = L[ok]
LINES = {}            # site -> (dates, gain[date, band], offset[date, band])
for site, g in L.groupby("site"):
    dates = sorted(g["date"].unique())
    G = np.array([[g[(g.date == d) & (g.band == b)]["gain_offset_gain"].iloc[0] for b in BANDS] for d in dates])
    O = np.array([[g[(g.date == d) & (g.band == b)]["gain_offset_offset"].iloc[0] for b in BANDS] for d in dates])
    LINES[site] = (dates, G, O)

STATE = {"fold": None, "held_sites": [], "draws": []}
_chip_path, _load_chip, _pixels = B2.chip_path, B2.load_chip, B2.pixels


def draw_line(fold, chip_id, copy):
    """One real between-date line (gain and offset per band), from a site the fold does not hold out."""
    rng = np.random.default_rng([C.SEED, fold, copy, zlib.crc32(chip_id.encode())])
    sites = sorted(s for s in LINES if s not in STATE["held_sites"])
    site = sites[rng.integers(len(sites))]
    dates, G, O = LINES[site]
    i, j = rng.choice(len(dates), 2, replace=False)
    gain = G[j] / G[i]
    off = O[j] - gain * O[i]
    return site, dates[i], dates[j], gain, off


def chip_path_with_copy(d, row):
    copy = int(row["aug_copy"]) if "aug_copy" in row and pd.notna(row["aug_copy"]) else 0
    return (_chip_path(d, row), copy, str(row["chip_id"]))


def load_chip_with_copy(key):
    path, copy, chip_id = key
    chip = _load_chip(path)
    if copy == 0:
        return chip
    site, d0, d1, gain, off = draw_line(STATE["fold"], chip_id.split("__aug")[0], copy)
    img = chip["image"].astype(np.float32).copy()
    refl = img[:6] * gain[:, None, None].astype(np.float32) + off[:, None, None].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.clip((refl[3] - refl[2]) / np.maximum(refl[3] + refl[2], INDEX_EPS), -1, 1)
        mndwi = np.clip((refl[1] - refl[4]) / np.maximum(refl[1] + refl[4], INDEX_EPS), -1, 1)
    img[:6], img[6], img[7] = refl, ndvi, mndwi
    chip["image"] = img                      # valid, labels and masks are left as they are
    STATE["draws"].append({"fold": STATE["fold"], "chip_id": chip_id, "copy": copy, "line_site": site, "from_date": d0, "to_date": d1,
                           **{f"gain_{b}": round(float(x), 4) for b, x in zip(BANDS, gain)},
                           **{f"offset_{b}": round(float(x), 4) for b, x in zip(BANDS, off)}})
    return chip


def with_copies(df):
    parts = [df.assign(aug_copy=0)]
    for c in range(1, N_COPIES + 1):
        parts.append(df.assign(aug_copy=c, chip_id=df["chip_id"].astype(str) + f"__aug{c}"))
    return pd.concat(parts, ignore_index=True)


def train(cat20, k, version):
    """Same steps as common.train_fold (variant b), with the two swaps."""
    present = set(int(f) for f in cat20.fold.unique())
    held = cat20[cat20.fold == k]
    pool_all = cat20[cat20.fold != k]
    pool = pool_all[~pool_all.hazy]
    train_ = C.drop_near(pool, held)
    assert not set(train_["site"]) & set(held["site"])
    inner = next(f for f in C.INNER_PREF if f != k and f in present)
    iv = train_[train_.fold == inner]
    it = C.drop_near(train_[train_.fold != inner], iv)
    STATE["fold"], STATE["held_sites"] = k, sorted(held["site"].unique())
    mult = 1
    B2.chip_path, B2.load_chip, B2.pixels = chip_path_with_copy, load_chip_with_copy, _pixels
    if version == "B":
        it, train_, mult = with_copies(it), with_copies(train_), N_COPIES + 1
    else:
        it, train_ = it.assign(aug_copy=0), train_.assign(aug_copy=0)
    if version == "C":
        B2.pixels = lambda chip: _pixels(chip)[:, 6:8]
    try:
        d = work / "forests" / "work"
        d.mkdir(exist_ok=True)
        p = d / f"inner_catalog_{version}_fold{k}.csv"
        pd.concat([it.assign(split="train"), iv.assign(split="val", aug_copy=0)]).to_csv(p, index=False)
        _, meta = B2.fit(p, tv, C.LABEL, cap=C.CFG["inner_cap"] * mult, val_cap=C.CFG["inner_val_cap"], seed=C.SEED,
                         n_estimators=C.CFG["inner_trees"])
        rng = np.random.default_rng(C.SEED)
        X, y = B2.collect(train_, tv, C.LABEL, C.CFG["final_cap"] * mult, rng, balanced=True)
        rf = RandomForestClassifier(n_estimators=C.CFG["final_trees"], n_jobs=-1, random_state=C.SEED, **meta["settings"]).fit(X, y)
    finally:
        B2.chip_path, B2.load_chip, B2.pixels = _chip_path, _load_chip, _pixels
    return {"rf": rf, "threshold": meta["threshold"], "settings": meta["settings"], "held_sites": sorted(held["site"].unique()),
            "features": "ndvi_mndwi" if version == "C" else "all", "inner_iou": meta["val_iou"], "train_pixels": int(len(y)),
            "train_rows": len(train_), "inner_fold": int(inner)}


cat = C.load_full_catalog(tv)
assert set(cat["role"]) == {"cv"}, "catalog holds rows that are not train/val"
cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
fits = []
for version in a.versions:
    for k in sorted(int(f) for f in cat20.fold.unique()):
        fit = train(cat20, k, version)
        row = {"version": version, "fold": k, "held_sites": "+".join(fit["held_sites"]), "threshold": fit["threshold"],
               "settings": str(fit["settings"]), "inner_iou": round(fit["inner_iou"], 4), "train_pixels": fit["train_pixels"],
               "training_rows_with_copies": fit["train_rows"]}
        if version == "Acheck":
            # must be the same forest as version A: same threshold and settings, same probabilities on a held-out chip
            A = joblib.load(Path(a.a_forests) / f"fold{k}.joblib")
            chip = C.load_chip(C.chip_path(tv, cat20[cat20.fold == k].iloc[0]))
            X = C.B2.pixels(chip)
            diff = float(np.max(np.abs(A["rf"].predict_proba(X) - fit["rf"].predict_proba(X))))
            row["largest_probability_difference_from_A"] = diff
            assert A["threshold"] == fit["threshold"] and A["settings"] == fit["settings"] and diff < 1e-9, \
                f"fold {k}: the new training code does not reproduce version A (difference {diff})"
        else:
            joblib.dump(fit, work / "forests" / f"{version}_fold{k}.joblib", compress=3)
        fits.append(row)
        print(row, flush=True)
pd.DataFrame(fits).to_csv(work / "forests" / "fits.csv", index=False)
if STATE["draws"]:
    D = pd.DataFrame(STATE["draws"]).drop_duplicates(["fold", "chip_id", "copy"])
    D.to_csv(work / "forests" / "augmentation_draws.csv", index=False)
    g = [c for c in D.columns if c.startswith("gain_")]
    o = [c for c in D.columns if c.startswith("offset_")]
    print(f"augmentation: {len(D)} copies. gain 5th to 95th percentile {np.percentile(D[g].values, 5):.3f} to {np.percentile(D[g].values, 95):.3f}, "
          f"offset {np.percentile(D[o].values, 5):+.4f} to {np.percentile(D[o].values, 95):+.4f}")
print("done")
