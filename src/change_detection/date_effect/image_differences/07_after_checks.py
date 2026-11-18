"""Checks added AFTER the result of the corrected-image test was seen, to understand it. They are not part of the test.

1. Are the swings after the correction the same swings as before? (correlation of each scene's deviation before and after)
2. Do the image measures of part a, taken on the images as they are, still track the swings after the correction?
3. Did the correction make solid mangrove canopy look more alike between dates? The correction is fitted on water and
   bright ground. If canopy does not follow it, the forests cannot be expected to steady.
   Solid canopy here = GMW v4.0.19 mangrove at least 3 pixels from its edge, never water. GMW is used only to look,
   not to correct.

Outputs in <out>: after_checks.csv, canopy_before_after.csv

Usage: python 07_after_checks.py --work <work> --out <results>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from shared import BANDS, CHIP, SITES, additive, cell_layout, demean, indices, load_stack, pooled, r2_within

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)
K3 = np.ones((3, 3), bool)

T = pd.read_csv(out / "scene_table_images.csv")
A = pd.read_csv(out / "scene_after_correction.csv")
A = A[A.main & (A.version == "gain_offset")][["site", "date", "dev_pct"]].rename(columns={"dev_pct": "dev_after_pct"})
T = T.merge(A, on=["site", "date"])
rows = []
dm = demean(T, ["dev_pct", "dev_after_pct"])
rows.append({"check": "correlation of the scene swing before and after the correction (site averages removed)",
             "value": round(float(np.corrcoef(dm.dev_pct, dm.dev_after_pct)[0, 1]), 3), "scenes": len(T)})
rows.append({"check": "within-site spread of the scene swing, before (%)", "value": round(float(dm.dev_pct.std()), 2), "scenes": len(T)})
rows.append({"check": "within-site spread of the scene swing, after (%)", "value": round(float(dm.dev_after_pct.std()), 2), "scenes": len(T)})
for x, name in (("deep_water_red_diff", "deep water (red)"), ("bright_ground_diff_pct", "bright stable ground"),
                ("zero_or_negative_diff_points", "zero or negative share"), ("sun_elevation", "sun elevation"), ("tide_eot20_m", "tide")):
    for y, when in (("dev_pct", "before"), ("dev_after_pct", "after")):
        r = pooled(T, x, y)
        rows.append({"check": f"r of {name} with the swing {when} the correction", "value": r["r_within"], "p_shuffle_dates": r["p_shuffle_dates"], "scenes": r["scenes"]})
for y, when in (("dev_pct", "before"), ("dev_after_pct", "after")):
    R2, _, n = r2_within(T, y, ["deep_water_red_diff", "bright_ground_diff_pct"])
    rows.append({"check": f"share of the swing {when} the correction explained by deep water (red) + bright ground", "value": round(R2, 3), "scenes": n})

# ---- solid canopy before and after
L = pd.read_csv(out / "correction_lines.csv")
can = []
for site in SITES:
    S = load_stack(work, site)
    dates = [str(d) for d in S["dates"]]
    years = np.array([int(d[:4]) for d in dates])
    n = len(dates)
    refl, valid, covered, gmw = S["image"][:, :6].astype(np.float32), S["valid"], S["covered"], S["gmw4019"]
    H, W = gmw.shape
    cells, present, use, npres = cell_layout(covered, valid)
    cov = covered & use[None]
    with np.errstate(invalid="ignore"):
        water = (refl[:, 1] > refl[:, 3]) & cov
    canopy = ndimage.binary_erosion(gmw == 1, K3, iterations=3) & use & (npres >= 2) & (water.sum(0) == 0)
    ls = L[L.site == site]
    typical = dates.index(ls["typical_date"].iloc[0])
    G = np.array([[ls[(ls.date == d) & (ls.band == b)]["gain_offset_gain"].iloc[0] for b in BANDS] for d in dates])
    O = np.array([[ls[(ls.date == d) & (ls.band == b)]["gain_offset_offset"].iloc[0] for b in BANDS] for d in dates])
    P2 = present & (present.sum(1) >= 2)[:, None]

    def site_values(arr):
        V = np.full((len(cells), n), np.nan)
        wgt = np.zeros(len(cells))
        for ci, (r, c) in enumerate(cells):
            mm = canopy[r:r + CHIP, c:c + CHIP]
            if mm.sum() < 50:
                continue
            wgt[ci] = mm.sum()
            for i in np.where(P2[ci])[0]:
                V[ci, i] = np.median(arr[i, r:r + CHIP, c:c + CHIP][mm])
        return additive(np.nan_to_num(V), P2 & (wgt > 0)[:, None], wgt)

    corr = np.empty_like(refl)
    for i in range(n):
        gm = G[typical] / G[i]
        om = O[typical] - gm * O[i]
        corr[i] = refl[i] * gm[:, None, None].astype(np.float32) + om[:, None, None].astype(np.float32)
    for when, x in (("before", refl), ("after", corr)):
        nd = np.stack([indices(x[i])[0] for i in range(n)])
        vals = {"ndvi": site_values(nd), "red": site_values(x[:, 2]), "nir": site_values(x[:, 3]), "swir1": site_values(x[:, 4])}
        row = {"site": site, "when": when, "canopy_px": int(canopy.sum())}
        for k, v in vals.items():
            row[f"{k}_spread_between_dates"] = round(float(np.nanstd(v, ddof=1)), 4)
            sy = [abs(v[i] - v[j]) for i in range(n) for j in range(i + 1, n) if years[i] == years[j] and np.isfinite(v[i]) and np.isfinite(v[j])]
            row[f"{k}_mean_same_year_difference"] = round(float(np.mean(sy)), 4) if sy else np.nan
        row["ndvi_by_date"] = " ".join(f"{d}:{v:.3f}" for d, v in zip(dates, vals["ndvi"]))
        can.append(row)
Cn = pd.DataFrame(can)
Cn.to_csv(out / "canopy_before_after.csv", index=False)
m = Cn.groupby("when")[[c for c in Cn.columns if c.endswith("dates") or c.endswith("difference")]].mean().round(4)
for c in m.columns:
    rows.append({"check": f"solid canopy, {c.replace('_', ' ')}, mean of sites, before", "value": m.loc["before", c]})
    rows.append({"check": f"solid canopy, {c.replace('_', ' ')}, mean of sites, after", "value": m.loc["after", c]})
R = pd.DataFrame(rows)
R.to_csv(out / "after_checks.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 110):
    print(R.to_string(index=False))
    print(Cn.drop(columns=["ndvi_by_date"]).to_string(index=False))
    print(Cn[["site", "when", "ndvi_by_date"]].to_string(index=False))
