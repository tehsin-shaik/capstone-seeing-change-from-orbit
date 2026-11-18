"""Train the five held-out random forests again (same code, same seed as before), keep them on disk, and put every
non-hazy scene of every training site on its site grid with all bands.

The forest code and the cross-validation steps are imported unchanged from src/change_detection/common.py.
Keep the work folder outside the repository.

Writes
    <work>/forests/fold<k>.joblib      the fitted forest, its threshold and the sites it holds out
    <work>/forests/fold_fits.csv       held-out IoU next to the committed result (must match)
    <work>/stacks/<site>.npz           dates, image [date, 8, H, W] (B02 B03 B04 B08 B11 B12 NDVI MNDWI), valid, covered,
                                       prob (forest probability on the images as they are), gmw4019, threshold
Also checks that NDVI and MNDWI recomputed from the six bands equal the ones stored in the chips, because the corrected
images need them recomputed.

Usage: python 02_stacks_and_forests.py --train-val <Train_Val_Dataset> --work <work folder>
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shared import indices

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=str(Path(__file__).resolve().parents[2]), help="folder that holds common.py (default: src/change_detection)")
ap.add_argument("--train-val", required=True)
ap.add_argument("--work", required=True)
a = ap.parse_args()
sys.path.insert(0, a.code)
import common as C  # noqa: E402

tv = Path(a.train_val)
if "Train_Val_Dataset" not in str(tv):
    raise SystemExit("only the Train_Val_Dataset folder may be used")
work = Path(a.work)
(work / "forests").mkdir(parents=True, exist_ok=True)
(work / "stacks").mkdir(parents=True, exist_ok=True)

cat = C.load_full_catalog(tv)
assert set(cat["role"]) == {"cv"}, "catalog holds rows that are not train/val"
cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
CH = C.CHIP
fits, worst = [], 0.0
for k in sorted(int(f) for f in cat20.fold.unique()):
    fit = C.train_fold(cat20, k, tv, work / "forests" / "work", "b")
    rf, th = fit["rf"], fit["threshold"]
    pos = list(rf.classes_).index(True)
    scores = C.heldout_scores(rf, th, fit["held"], tv)
    joblib.dump({"rf": rf, "threshold": th, "held_sites": fit["info"]["held_sites"], "settings": fit["settings"]},
                work / "forests" / f"fold{k}.joblib", compress=3)
    print(f"fold {k}: threshold {th}, held-out IoU {scores['__fold__']['iou']:.4f}, committed {C.published_fold_iou(k, 'b')}", flush=True)
    for site in fit["info"]["held_sites"]:
        fits.append({"fold": k, "site": site, "threshold": th, "heldout_iou_site": round(scores[site]["iou"], 4),
                     "heldout_iou_fold": round(scores["__fold__"]["iou"], 4), "committed_b2_fold_iou": C.published_fold_iou(k, "b")})
        rows = cat[cat["site"] == site]
        r0, c0 = int(rows["row"].min()), int(rows["col"].min())
        H, W = int(rows["row"].max()) - r0 + CH, int(rows["col"].max()) - c0 + CH
        dates = sorted(d for d, g in rows.groupby("date") if not bool(g["hazy"].iloc[0]))
        n = len(dates)
        img = np.full((n, 8, H, W), np.nan, np.float32)
        prob = np.zeros((n, H, W), np.float32)
        valid = np.zeros((n, H, W), bool)
        covered = np.zeros((n, H, W), bool)
        gmw = np.full((H, W), 255, np.uint8)
        for _, r in rows.iterrows():
            chip = C.load_chip(C.chip_path(tv, r))
            assert [str(c) for c in chip["channels"]] == ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"]
            y, x = int(r["row"]) - r0, int(r["col"]) - c0
            sl = (slice(y, y + CH), slice(x, x + CH))
            if int(r["year"]) == 2020 and "label_v4019" in chip:
                lab = chip["label_v4019"]
                gmw[sl] = np.where(lab != 255, lab, gmw[sl])
            if bool(r["hazy"]):
                continue
            i = dates.index(r["date"])
            v = chip["valid"]
            p = np.zeros(v.shape, np.float32)
            if v.any():
                p[v] = rf.predict_proba(C.B2.pixels(chip)[v.ravel()])[:, pos]
            prob[i][sl], valid[i][sl], covered[i][sl] = p, v, True
            img[i][(slice(None),) + sl] = chip["image"]
            nd, mn = indices(chip["image"][:6])
            worst = max(worst, float(np.nanmax(np.abs(nd - chip["image"][6])[v], initial=0)), float(np.nanmax(np.abs(mn - chip["image"][7])[v], initial=0)))
        np.savez_compressed(work / "stacks" / f"{site}.npz", dates=np.array(dates), image=img, prob=prob, valid=valid, covered=covered,
                            gmw4019=gmw, threshold=np.float32(th), fold=k)
        print(f"  {site}: {n} scenes, grid {H}x{W}", flush=True)
pd.DataFrame(fits).to_csv(work / "forests" / "fold_fits.csv", index=False)
print(f"largest difference between recomputed and stored NDVI / MNDWI on usable pixels: {worst:.2e}")
print("done")
