"""How steady is each model across image dates, and how sure are we of the ~0.70?

Reads the per-chip TP/FP/FN file (one row per chip, date and model) and makes two tables:

1. Across dates: IoU pooled over the chips of each site and date, then per model the mean, standard
   deviation, min and max over those site-dates. Also lists the worst site-date for each model,
   so those images can be opened and looked at.
2. Random subsets: resample chip locations (all dates of a spot together) within each fold many times
   and recompute the mean-over-folds IoU. Per model: mean, standard deviation, 95% interval, lowest and
   highest. A model with a tighter range is steadier even if its mean is the same.

Usage (from src/mangrove_segmentation):
  python date_stability.py --per-chip ../../results/mangrove_segmentation/model_comparison/per_chip_scores.csv \\
      --out ../../results/mangrove_segmentation/date_stability
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import compare_models as CM


def by_site_date(pc: pd.DataFrame) -> pd.DataFrame:
    """Pooled IoU for every model, site and date."""
    g = pc.groupby(["model", "site", "date"])[["TP", "FP", "FN"]].sum().reset_index()
    g["iou"] = g["TP"] / (g["TP"] + g["FP"] + g["FN"]).clip(lower=1)
    g["chips"] = pc.groupby(["model", "site", "date"]).size().values
    return g


def across_dates(sd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in sd.groupby("model", sort=False):
        worst = g.loc[g["iou"].idxmin()]
        rows.append(dict(model=model, site_dates=len(g), mean=g["iou"].mean(), sd=g["iou"].std(),
                         min=g["iou"].min(), max=g["iou"].max(),
                         worst=f"{worst['site']} {worst['date']}"))
    return pd.DataFrame(rows)


def per_site_spread(sd: pd.DataFrame) -> pd.DataFrame:
    """Within each site, how much a model moves between that site's dates."""
    return (sd.groupby(["site", "model"])["iou"].agg(["mean", "std", "min", "max"]).reset_index()
            .rename(columns={"std": "sd"}))


def subset_confidence(path: Path, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    models, folds = CM.load_per_chip(path)
    rng = np.random.default_rng(seed)
    weights = [rng.multinomial(len(c), np.full(len(c), 1 / len(c)), size=reps) for c in folds]
    boot = CM.mean_fold_iou(folds, weights)
    obs = CM.mean_fold_iou(folds)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return pd.DataFrame(dict(model=models, mean_iou=obs, sd=boot.std(0), ci95_lo=lo, ci95_hi=hi,
                             lowest=boot.min(0), highest=boot.max(0), range=boot.max(0) - boot.min(0)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-chip", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args(argv)
    pc = pd.read_csv(a.per_chip)
    pc["model"] = pc["model"].map(lambda m: CM.PER_CHIP_NAMES.get(m, m))
    sd = by_site_date(pc)
    t1, t2, t3 = across_dates(sd), per_site_spread(sd), subset_confidence(a.per_chip, a.reps)
    print(CM.to_markdown(t1.round(3))); print(); print(CM.to_markdown(t3.round(3)))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        sd.round(4).to_csv(a.out / "iou_by_site_date.csv", index=False)
        t1.round(4).to_csv(a.out / "across_dates.csv", index=False)
        t2.round(4).to_csv(a.out / "per_site_spread.csv", index=False)
        t3.round(4).to_csv(a.out / "subset_confidence.csv", index=False)


if __name__ == "__main__":
    main()
