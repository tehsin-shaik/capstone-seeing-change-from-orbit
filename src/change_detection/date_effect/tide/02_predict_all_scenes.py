"""Random forest predictions for every non-hazy scene of every training site, each site predicted by the
forest that never saw it (the fold that holds the site out, hazy scenes dropped from training).

The forest code and the cross-validation steps are imported unchanged from src/change_detection/common.py
(which imports baseline_rf.py and chipio.py). Keep the work folder outside the repository.

For each site one file <work>/pred/<site>.npz is written, with every scene on the site's chip grid:
    dates      list of scene dates (non-hazy only)
    prob       forest probability of mangrove            [date, H, W] float16
    valid      usable pixel (in a chip, not cloud, not nodata, all bands finite)
    covered    pixel lies in a chip that exists on that date
    b03, b08, b11, ndvi   green, near-infrared, short-wave infrared reflectance and NDVI
    gmw4019    GMW v4.0.19 label on the same grid (1 mangrove, 0 not, 255 none), taken from the 2020 chips
    threshold  the forest's tuned threshold
Also <work>/pred/fold_fits.csv with the held-out IoU of each forest next to the committed result.

Usage: python 02_predict_all_scenes.py --train-val <Train_Val_Dataset> --work <work folder>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
out = Path(a.work) / "pred"
out.mkdir(parents=True, exist_ok=True)

cat = C.load_full_catalog(tv)
assert set(cat["role"]) == {"cv"}, "catalog holds rows that are not train/val"
cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
CH = C.CHIP
fits = []
for k in sorted(int(f) for f in cat20.fold.unique()):
    fit = C.train_fold(cat20, k, tv, out / "work", "b")
    rf, th = fit["rf"], fit["threshold"]
    pos = list(rf.classes_).index(True)
    scores = C.heldout_scores(rf, th, fit["held"], tv)
    print(f"fold {k}: threshold {th}, held-out IoU {scores['__fold__']['iou']:.4f}, committed {C.published_fold_iou(k, 'b')}", flush=True)
    for site in fit["info"]["held_sites"]:
        fits.append({"fold": k, "site": site, "threshold": th, "heldout_iou_site": round(scores[site]["iou"], 4),
                     "heldout_iou_fold": round(scores["__fold__"]["iou"], 4), "committed_b2_fold_iou": C.published_fold_iou(k, "b")})
        rows = cat[cat["site"] == site]
        r0, c0 = int(rows["row"].min()), int(rows["col"].min())
        H, W = int(rows["row"].max()) - r0 + CH, int(rows["col"].max()) - c0 + CH
        dates = sorted(d for d, g in rows.groupby("date") if not bool(g["hazy"].iloc[0]))
        n = len(dates)
        prob = np.zeros((n, H, W), np.float16)
        valid = np.zeros((n, H, W), bool)
        covered = np.zeros((n, H, W), bool)
        bands = {b: np.full((n, H, W), np.nan, np.float32) for b in ("b03", "b08", "b11", "ndvi")}
        gmw = np.full((H, W), 255, np.uint8)
        for _, r in rows.iterrows():
            chip = C.load_chip(C.chip_path(tv, r))
            y, x = int(r["row"]) - r0, int(r["col"]) - c0
            sl = (slice(y, y + CH), slice(x, x + CH))
            if int(r["year"]) == 2020 and "label_v4019" in chip:
                lab = chip["label_v4019"]
                old = gmw[sl]
                if not np.all((old == 255) | (lab == 255) | (old == lab)):
                    print(f"  WARNING: GMW label differs between dates in cell {r['row']},{r['col']} of {site}")
                gmw[sl] = np.where(lab != 255, lab, old)
            if bool(r["hazy"]):
                continue
            i = dates.index(r["date"])
            v = chip["valid"]
            p = np.zeros(v.shape, np.float32)
            if v.any():
                p[v] = rf.predict_proba(C.B2.pixels(chip)[v.ravel()])[:, pos]
            prob[i][sl] = p
            valid[i][sl] = v
            covered[i][sl] = True
            names = [str(c) for c in chip["channels"]]
            for key, ch in (("b03", "B03"), ("b08", "B08"), ("b11", "B11"), ("ndvi", "NDVI")):
                bands[key][i][sl] = chip["image"][names.index(ch)]
        np.savez_compressed(out / f"{site}.npz", dates=np.array(dates), prob=prob, valid=valid, covered=covered, gmw4019=gmw,
                            threshold=np.float32(th), row0=r0, col0=c0, **bands)
        print(f"  {site}: {n} scenes, grid {H}x{W}", flush=True)
pd.DataFrame(fits).to_csv(out / "fold_fits.csv", index=False)
print("done")
