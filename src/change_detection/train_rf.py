"""Train the random forest baseline (B2) with one site held out, for change mapping at that site.

The forest is trained on the 2020 chips of the other folds, so it has never seen the site. The steps are
those of the baseline cross-validation (see common.train_fold). As a check, the held-out 2020 chips are
predicted and scored against GMW v4.0.19, and the IoU is compared with the committed B2 result for that fold.

Writes <out>/<site>/model_<variant>/rf.joblib and fit_summary.json. The model file is not for the repository.

Usage: python train_rf.py --train-val <Train_Val_Dataset> --out <output folder> [--site jubail_island] [--variant b]
"""
import argparse
import json
import time
from pathlib import Path

import joblib

from common import add_path_args, heldout_scores, load_full_catalog, published_fold_iou, train_fold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--site", default="jubail_island", help="site to hold out (its whole fold is held out)")
    ap.add_argument("--variant", default="b", choices=["a", "b"],
                    help="a = all 2020 dates in training, b = hazy scenes dropped from training (the default)")
    a = ap.parse_args()
    out = Path(a.out) / a.site / f"model_{a.variant}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cat = load_full_catalog(a.train_val, a.qa)
    cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
    if a.site not in set(cat20["site"]):
        raise SystemExit(f"site {a.site} not in the catalog; sites: {sorted(cat20['site'].unique())}")
    k = int(cat20.loc[cat20["site"] == a.site, "fold"].iloc[0])
    fit = train_fold(cat20, k, a.train_val, out, a.variant)
    rf, th = fit["rf"], fit["threshold"]
    print(fit["info"], flush=True)

    scores = heldout_scores(rf, th, fit["held"], a.train_val)
    check = {s: {"mangrove_px": c["tp"] + c["fn"], "predicted_px": c["tp"] + c["fp"],
                 **{m: round(c[m], 4) for m in ("iou", "precision", "recall")},
                 "predicted_area_over_label_area": round((c["tp"] + c["fp"]) / max(c["tp"] + c["fn"], 1), 4)}
             for s, c in scores.items()}
    check["published_fold_iou"] = published_fold_iou(k, a.variant)
    print("held-out 2020 check:", check, flush=True)

    joblib.dump({"rf": rf, "threshold": th, "settings": fit["settings"], "variant": a.variant, "site": a.site, "fold": k},
                out / "rf.joblib", compress=3)
    summary = {**fit["info"], "site": a.site, "settings": fit["settings"], "threshold": th,
               "feature_importances": dict(zip(["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"],
                                               [round(float(v), 4) for v in rf.feature_importances_])),
               "heldout_2020_check": check, "minutes": round((time.time() - t0) / 60, 2)}
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=1))
    print("saved ->", out, flush=True)


if __name__ == "__main__":
    main()
