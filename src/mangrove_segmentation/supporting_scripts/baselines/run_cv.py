"""Stage 4: baselines B1 (NDVI + MNDWI threshold) and B2 (per-pixel random forest) with
leave-one-site-out cross-validation on Train_Val_Dataset only (Test_Dataset is never touched).

Reuses code/baseline_ndvi.py, code/baseline_rf.py, code/evaluate.py and code/chipio.py unchanged:
per fold it writes a temporary catalog whose `split` column tells those functions which chips
to fit / tune on, then scores the held-out fold with evaluate.evaluate().

Usage: python 2_Code/supporting_scripts/baselines/run_cv.py      (all models, both haze variants)
       python ... run_cv.py --fast                                (quick smoke test: small samples, few trees)
Paths come from 2_Code/paths.py: tables -> 4_Results/Baselines, predictions -> 6_Work/runs/Baselines/preds.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 2_Code
import paths as P
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import baseline_ndvi as B1
import baseline_rf as B2
B2.THRESHOLDS = np.round(np.arange(0.2, 0.951, 0.05), 2)   # extended from 0.8 (code/ default) to 0.95
from chipio import chip_path, load_chip, save_mask

TV = P.TRAIN_VAL
OUT = P.BASELINES
PRED = P.WORK / "runs" / "Baselines" / "preds"
TMP = P.TMP / "baselines"
LABEL = "v4019"
HAZE_P5 = 0.05          # dark-water red (5th pct B04 over water) above this = hazy scene
BUFFER_M = 2560         # one chip
INNER_PREF = [3, 4, 5, 2, 1]   # inner-validation fold for B2 tuning: first of these that is a training fold
SEED = 0
# sample sizes / forest sizes (defaults = what was run; --fast shrinks them for a smoke test)
CFG = dict(inner_cap=100_000, inner_val_cap=300_000, inner_trees=100, final_cap=200_000, final_trees=200, calib_cap=300_000)


def load_cat():
    cat = pd.read_csv(TV / "catalog.csv")
    cat = cat[cat["year"] == 2020].copy()
    qa = pd.read_csv(P.QA_CSV)[["site", "date", "water_b04_p5"]]
    cat = cat.merge(qa, on=["site", "date"], how="left")
    cat["hazy"] = cat["water_b04_p5"] > HAZE_P5
    b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
    cat[["x0", "y0", "x1", "y1"]] = b.values
    cat["fold"] = cat["fold"].astype(int)
    return cat.reset_index(drop=True)


def drop_near(train: pd.DataFrame, other: pd.DataFrame, buf=BUFFER_M) -> pd.DataFrame:
    """Drop training chips within `buf` metres of any chip in `other` (different site/fold)."""
    if train.empty or other.empty:
        return train
    keep = []
    for _, r in train.iterrows():
        dx = np.maximum(0, np.maximum(other.x0.values - r.x1, r.x0 - other.x1.values))
        dy = np.maximum(0, np.maximum(other.y0.values - r.y1, r.y0 - other.y1.values))
        keep.append(np.maximum(dx, dy).min() >= buf)
    return train[np.array(keep)]


def write_cat(df: pd.DataFrame, name: str) -> Path:
    p = TMP / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def run(cat):
    OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    present = set(int(f) for f in cat.fold.unique())
    log = []
    importances = []
    calib = []
    for variant in ("a_all_dates", "b_no_haze"):
        for k in sorted(cat.fold.unique()):
            t0 = time.time()
            held = cat[cat.fold == k]
            pool = cat[cat.fold != k]
            if variant == "b_no_haze":
                pool = pool[~pool.hazy]
            n_pool = len(pool)
            train = drop_near(pool, held)
            n_buf = n_pool - len(train)
            n_haze = int((cat[cat.fold != k].hazy).sum()) if variant == "b_no_haze" else 0

            # ---- B1: thresholds fitted on the 4 training folds (B1.fit uses split == 'val') ----
            p = write_cat(train.assign(split="val"), f"b1_{variant}_f{k}")
            m1 = B1.fit(p, TV, LABEL, seed=SEED)
            for _, r in held.iterrows():
                chip = load_chip(chip_path(TV, r))
                save_mask(PRED / f"B1_{variant}", r.chip_id, B1.predict_chip(chip, m1["t_ndvi"], m1["w_mndwi"]))

            # ---- B2: tune settings + threshold on an inner fold, then refit on all 4 folds ----
            inner = next(f for f in INNER_PREF if f != k and f in present)   # = original rule when all 5 folds exist
            iv = train[train.fold == inner]
            it = drop_near(train[train.fold != inner], iv)
            p = write_cat(pd.concat([it.assign(split="train"), iv.assign(split="val")]), f"b2_{variant}_f{k}")
            rf_in, meta = B2.fit(p, TV, LABEL, cap=CFG["inner_cap"], val_cap=CFG["inner_val_cap"], seed=SEED,
                                 n_estimators=CFG["inner_trees"])
            # calibration of the tuned (balanced-trained) model on the inner-validation fold, natural class mix
            Xv, yv = B2.collect(iv, TV, LABEL, CFG["calib_cap"], np.random.default_rng(SEED + 1), balanced=False)
            pv = rf_in.predict_proba(Xv)[:, list(rf_in.classes_).index(True)]
            bins = np.clip((pv * 10).astype(int), 0, 9)
            for b in range(10):
                sel = bins == b
                calib.append({"variant": variant, "fold": int(k), "inner_fold": inner, "bin": f"{b/10:.1f}-{(b+1)/10:.1f}",
                              "n_px": int(sel.sum()), "mean_pred": float(pv[sel].mean()) if sel.any() else np.nan,
                              "observed_mangrove_frac": float(yv[sel].mean()) if sel.any() else np.nan,
                              "inner_val_prevalence": float(yv.mean())})
            rng = np.random.default_rng(SEED)
            X, y = B2.collect(train, TV, LABEL, CFG["final_cap"], rng, balanced=True)
            rf = RandomForestClassifier(n_estimators=CFG["final_trees"], n_jobs=-1, random_state=SEED, **meta["settings"]).fit(X, y)
            pos = list(rf.classes_).index(True)
            for _, r in held.iterrows():
                chip = load_chip(chip_path(TV, r))
                v = chip["valid"]; mask = np.zeros(v.shape, bool)
                if v.any():
                    mask[v] = rf.predict_proba(B2.pixels(chip)[v.ravel()])[:, pos] > meta["threshold"]
                save_mask(PRED / f"B2_{variant}", r.chip_id, mask)
            importances.append({"variant": variant, "fold": k, **dict(zip(["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"], rf.feature_importances_.round(4)))})
            rec = {"variant": variant, "fold": int(k), "held_site": ",".join(sorted(held.site.unique())),
                   "held_chips": len(held), "train_chips": len(train), "dropped_buffer": int(n_buf),
                   "dropped_hazy_from_training": n_haze,
                   "b1_t_ndvi": m1["t_ndvi"], "b1_w_mndwi": m1["w_mndwi"], "b1_fit_iou": round(m1["val_iou"], 4),
                   "b2_inner_fold": inner, "b2_settings": json.dumps(meta["settings"]), "b2_threshold": meta["threshold"],
                   "b2_inner_iou": round(meta["val_iou"], 4), "seconds": round(time.time() - t0, 1)}
            log.append(rec); print(rec, flush=True)
    pd.DataFrame(log).to_csv(OUT / "fold_fits.csv", index=False)
    pd.DataFrame(importances).to_csv(OUT / "rf_feature_importances.csv", index=False)
    pd.DataFrame(calib).to_csv(OUT / "rf_calibration_inner.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small samples and few trees (smoke test only)")
    a = ap.parse_args()
    if a.fast:
        CFG.update(inner_cap=5_000, inner_val_cap=20_000, inner_trees=10, final_cap=10_000, final_trees=20, calib_cap=20_000)
    print(P.describe(), flush=True)
    cat = load_cat()
    OUT.mkdir(parents=True, exist_ok=True)
    cat.to_csv(OUT / "cv_catalog_2020.csv", index=False)
    run(cat)
    print("tables ->", OUT, "| predictions ->", PRED, flush=True)
