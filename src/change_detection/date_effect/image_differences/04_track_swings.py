"""Part b: do the image-comparability measures track the swings in predicted mangrove area?

Same method as the tide test: each site's average is removed, all sites are put together, and the chance of a link
this strong is found by shuffling the dates within each site (sites that share a date get the same shuffle).
The swing is `dev_pct` from the tide check's scene table (percent above or below the site's average predicted area).

The main measures were chosen before looking at the result. The per-band table is written as well but is not one of them.

Outputs in <out>: scene_table_images.csv, track_pooled.csv, track_per_site.csv, track_models.csv, track_per_band.csv

Usage: python 04_track_swings.py --out <results folder> --tide-table <tide results>/scene_table.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared import demean, pooled, r2_within

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--tide-table", required=True)
a = ap.parse_args()
out = Path(a.out)

M = pd.read_csv(out / "scene_image_measures.csv")
T0 = pd.read_csv(a.tide_table)[["site", "date", "year", "dev_pct", "dev_within_year_pct", "tide_eot20_m", "water_pct", "interior_ndvi",
                                 "rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw", "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw",
                                 "cells_on_this_date"]]
T = T0.merge(M, on=["site", "date"])
assert len(T) == 40
T["newer_processing"] = (T["processing_baseline"].astype(float) > 5.05).astype(float)
T["site_year"] = T["site"] + "_" + T["year"].astype(str)
T.to_csv(out / "scene_table_images.csv", index=False)

HEAD = {"deep_water_red_diff": "deep water, red band", "deep_water_visible_diff": "deep water, visible bands",
        "bright_ground_diff_pct": "bright stable ground, all bands", "zero_or_negative_diff_points": "share of zero or negative pixels",
        "haze_measure": "team's haze measure", "newer_processing": "processing version 05.11 (2025) against 05.00"}
rows = []
for x, name in HEAD.items():
    r = pooled(T, x, "dev_pct"); r.update(measure=name, compared="all dates of a site"); rows.append(r)
    if x != "newer_processing":            # the version is the same for every scene of a year, so it cannot vary within a year
        r = pooled(T, x, "dev_within_year_pct", by="site_year"); r.update(measure=name, compared="dates of the same year only"); rows.append(r)
        r = pooled(T[T.date != "2020-05-18"], x, "dev_pct"); r.update(measure=name, compared="all dates, without the 18 May 2020 pass"); rows.append(r)
        r = pooled(T[T.site != "al_nouf"], x, "dev_pct"); r.update(measure=name, compared="all dates, without Al Nouf"); rows.append(r)
for x in ("tide_eot20_m", "water_pct", "sun_elevation", "interior_ndvi"):
    r = pooled(T, x, "dev_pct"); r.update(measure=f"for comparison: {x}", compared="all dates of a site"); rows.append(r)
for y in ("rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw", "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw"):
    for x in ("deep_water_red_diff", "bright_ground_diff_pct", "zero_or_negative_diff_points"):
        r = pooled(T[T.year == 2020], x, y); r.update(measure=HEAD[x], compared="2020 scenes only"); rows.append(r)
R = pd.DataFrame(rows)
R[["measure", "compared", "y", "scenes", "dates", "r_within", "share_explained", "spearman_within", "r_each_site_equal_weight", "p_shuffle_dates",
   "leave_one_site_out_r_min", "leave_one_site_out_r_max", "x"]].to_csv(out / "track_pooled.csv", index=False)

per = []
for site, g in T.groupby("site"):
    row = {"site": site, "scenes": len(g)}
    for x in list(HEAD)[:5]:
        gg = g.dropna(subset=[x])
        row[f"r_{x}"] = round(np.corrcoef(gg[x], gg["dev_pct"])[0, 1], 2) if len(gg) >= 3 and gg[x].std() > 0 else np.nan
    per.append(row)
pd.DataFrame(per).to_csv(out / "track_per_site.csv", index=False)

MODELS = {"deep water (red)": ["deep_water_red_diff"], "bright stable ground": ["bright_ground_diff_pct"],
          "zero or negative share": ["zero_or_negative_diff_points"], "team's haze measure": ["haze_measure"],
          "deep water (red) + bright stable ground": ["deep_water_red_diff", "bright_ground_diff_pct"],
          "deep water (red) + bright ground + zero or negative share": ["deep_water_red_diff", "bright_ground_diff_pct", "zero_or_negative_diff_points"],
          "for comparison: tide": ["tide_eot20_m"], "for comparison: sun elevation": ["sun_elevation"],
          "for comparison: interior greenness": ["interior_ndvi"],
          "deep water (red) + bright ground + sun elevation": ["deep_water_red_diff", "bright_ground_diff_pct", "sun_elevation"],
          "deep water (red) + bright ground + tide": ["deep_water_red_diff", "bright_ground_diff_pct", "tide_eot20_m"]}
Tz = T.copy()
Tz["dev_z"] = demean(T, ["dev_pct"])["dev_pct"] / T.groupby("site")["dev_pct"].transform("std")
mods = []
for name, xs in MODELS.items():
    R2, adj, n = r2_within(T, "dev_pct", xs)
    R2z, adjz, _ = r2_within(Tz, "dev_z", xs)
    R2y, adjy, ny = r2_within(T, "dev_within_year_pct", xs, by="site_year")
    mods.append({"explained_by": name, "terms": len(xs), "scenes": n, "share_of_swing": round(R2, 3), "share_adjusted": round(adj, 3),
                 "share_each_site_equal_weight": round(R2z, 3), "share_same_year_only": round(R2y, 3)})
pd.DataFrame(mods).to_csv(out / "track_models.csv", index=False)

B = pd.read_csv(out / "scene_band_values.csv").merge(T0[["site", "date", "dev_pct"]], on=["site", "date"])
bands = []
for band, g in B.groupby("band"):
    for x in ("deep_water_diff", "bright_ground_diff_pct"):
        r = pooled(g, x, "dev_pct", perm=False)
        bands.append({"band": band, "target": x, "scenes": r["scenes"], "r_within": r["r_within"]})
pd.DataFrame(bands).to_csv(out / "track_per_band.csv", index=False)

with pd.option_context("display.width", 280, "display.max_rows", 200, "display.max_colwidth", 60):
    print(R[["measure", "compared", "y", "scenes", "dates", "r_within", "share_explained", "r_each_site_equal_weight", "p_shuffle_dates",
             "leave_one_site_out_r_min", "leave_one_site_out_r_max"]].to_string(index=False))
    print(pd.DataFrame(per).to_string(index=False))
    print(pd.DataFrame(mods).to_string(index=False))
    print(pd.DataFrame(bands).pivot(index="band", columns="target", values="r_within").to_string())


# ---- shuffled-dates check for two or three measures together: the rows of measures are shuffled between the dates of a site
def perm_r2(d, y, xs, by="site", n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = d.dropna(subset=[y] + xs).sort_values([by, "date"]).reset_index(drop=True)
    dm = demean(d, [y] + xs, by)
    yv, X = dm[y].values, dm[xs].values

    def r2(Xp):
        res = yv - Xp @ np.linalg.lstsq(Xp, yv, rcond=None)[0]
        return 1 - (res ** 2).sum() / (yv ** 2).sum()

    obs = r2(X)
    alld = sorted(d["date"].unique())
    idx = {dd: i for i, dd in enumerate(alld)}
    groups = [(g.index.values, np.array([idx[v] for v in g["date"]])) for _, g in d.groupby(by)]
    hits = 0
    for _ in range(n_perm):
        key = rng.random(len(alld))
        Xp = np.empty_like(X)
        for ix, di in groups:
            Xp[ix[np.argsort(key[di])]] = X[ix]
        hits += r2(Xp) >= obs - 1e-12
    return round(float(obs), 3), round((hits + 1) / (n_perm + 1), 4), len(d)


joint = []
for name, xs in (("deep water (red) + bright stable ground", ["deep_water_red_diff", "bright_ground_diff_pct"]),
                 ("deep water (red) + bright ground + zero or negative share", ["deep_water_red_diff", "bright_ground_diff_pct", "zero_or_negative_diff_points"])):
    for label, d, y, by in (("all dates of a site", T, "dev_pct", "site"), ("dates of the same year only", T, "dev_within_year_pct", "site_year"),
                            ("all dates, without the 18 May 2020 pass", T[T.date != "2020-05-18"], "dev_pct", "site"),
                            ("all dates, without Al Nouf", T[T.site != "al_nouf"], "dev_pct", "site")):
        R2, p, n = perm_r2(d, y, xs, by)
        joint.append({"explained_by": name, "compared": label, "scenes": n, "share_of_swing": R2, "p_shuffle_dates": p})
J = pd.DataFrame(joint)
J.to_csv(out / "track_joint.csv", index=False)
with pd.option_context("display.width", 250, "display.max_colwidth", 70):
    print(J.to_string(index=False))
