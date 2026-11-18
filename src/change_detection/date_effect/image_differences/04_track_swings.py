"""Part b: do the image-comparability measures track the swings in predicted mangrove area?

Same method as the tide test. The swing of a scene (`dev_pct` in the tide test's scene table) is its predicted mangrove
area as a percent above or below the average of the dates of the SAME SITE AND THE SAME YEAR. The measures are taken
the same way (difference from the average of the same site and year), so real change between 2018, 2020 and 2025 cannot
get into the comparison. All sites and years are then put together, and the chance of a link this strong is found by
shuffling the dates inside each site and year (sites that share a date get the same shuffle).

The main measures were chosen before looking at the result. The per-band table is written as well but is not one of them.
Rows marked "old definition" repeat the test against the site's average over all its years, for comparison only.

Outputs in <out>: scene_table_images.csv, track_pooled.csv, track_per_site.csv, track_models.csv, track_joint.csv, track_per_band.csv

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
T0 = pd.read_csv(a.tide_table)[["site", "date", "year", "dev_pct", "dev_all_years_pct", "tide_eot20_m", "water_pct", "interior_ndvi",
                                 "rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw", "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw",
                                 "cells_on_this_date"]]
T = T0.merge(M, on=["site", "date"])
assert len(T) == 40
T["site_year"] = T["site"] + "_" + T["year"].astype(str)
T.to_csv(out / "scene_table_images.csv", index=False)

SAME = "dates of the same site and year"
HEAD = {"deep_water_red_diff": "deep water, red band", "deep_water_visible_diff": "deep water, visible bands",
        "bright_ground_diff_pct": "bright stable ground, all bands", "zero_or_negative_diff_points": "share of zero or negative pixels",
        "haze_measure": "team's haze measure"}
rows = []
for x, name in HEAD.items():
    r = pooled(T, x, "dev_pct"); r.update(measure=name, compared=SAME); rows.append(r)
    r = pooled(T[T.date != "2020-05-18"], x, "dev_pct"); r.update(measure=name, compared=SAME + ", without the 18 May 2020 pass"); rows.append(r)
    r = pooled(T[T.site != "al_nouf"], x, "dev_pct"); r.update(measure=name, compared=SAME + ", without Al Nouf"); rows.append(r)
# The processing version is the same for every scene of a site and year, so it cannot be compared inside a year at all.
for x in ("tide_eot20_m", "water_pct", "sun_elevation", "interior_ndvi"):
    r = pooled(T, x, "dev_pct"); r.update(measure=f"for comparison: {x}", compared=SAME); rows.append(r)
for y in ("rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw", "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw"):
    for x in ("deep_water_red_diff", "bright_ground_diff_pct", "zero_or_negative_diff_points"):
        r = pooled(T[T.year == 2020], x, y); r.update(measure=HEAD[x], compared="2020 scenes only"); rows.append(r)
# for comparison only: the old definition, each scene against the site's average over all its years
OLD = "old definition: against the site's average over all years"
for x, name in (("deep_water_red_diff_all_years", "deep water, red band"), ("bright_ground_diff_all_years_pct", "bright stable ground, all bands"),
                ("zero_or_negative_diff_all_years_points", "share of zero or negative pixels"), ("haze_measure", "team's haze measure")):
    r = pooled(T, x, "dev_all_years_pct", by="site"); r.update(measure=name, compared=OLD); rows.append(r)
R = pd.DataFrame(rows)
R[["measure", "compared", "y", "scenes", "dates", "r_within", "share_explained", "spearman_within", "r_each_group_equal_weight", "p_shuffle_dates",
   "leave_one_site_out_r_min", "leave_one_site_out_r_max", "x"]].to_csv(out / "track_pooled.csv", index=False)

per = []
for site, g in T.dropna(subset=["dev_pct"]).groupby("site"):
    row = {"site": site, "scenes": len(g)}
    for x in HEAD:
        gg = g.dropna(subset=[x])
        dm = demean(gg, [x, "dev_pct"])
        row[f"r_{x}"] = round(np.corrcoef(dm[x], dm["dev_pct"])[0, 1], 2) if len(gg) >= 3 and dm[x].std() > 0 else np.nan
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
OLD_NAMES = {"deep_water_red_diff": "deep_water_red_diff_all_years", "bright_ground_diff_pct": "bright_ground_diff_all_years_pct",
             "zero_or_negative_diff_points": "zero_or_negative_diff_all_years_points"}
Tz = T.copy()
dmv = demean(T, ["dev_pct"])["dev_pct"]
Tz["dev_z"] = dmv / dmv.groupby(T["site"]).transform("std")
mods = []
for name, xs in MODELS.items():
    R2, adj, n = r2_within(T, "dev_pct", xs)
    R2z, adjz, _ = r2_within(Tz, "dev_z", xs)
    R2o, _, no = r2_within(T, "dev_all_years_pct", [OLD_NAMES.get(x, x) for x in xs], by="site")
    mods.append({"explained_by": name, "terms": len(xs), "scenes": n, "share_of_same_year_swing": round(R2, 3), "share_adjusted": round(adj, 3),
                 "share_each_site_equal_weight": round(R2z, 3), "old_definition_share_all_years": round(R2o, 3), "old_definition_scenes": no})
pd.DataFrame(mods).to_csv(out / "track_models.csv", index=False)

B = pd.read_csv(out / "scene_band_values.csv").merge(T[["site", "date", "site_year", "dev_pct"]], on=["site", "date"])
bands = []
for band, g in B.groupby("band"):
    for x in ("deep_water_diff", "bright_ground_diff_pct"):
        r = pooled(g, x, "dev_pct", perm=False)
        bands.append({"band": band, "target": x, "scenes": r["scenes"], "r_within": r["r_within"]})
pd.DataFrame(bands).to_csv(out / "track_per_band.csv", index=False)


# ---- shuffled-dates check for two or three measures together: the rows of measures are shuffled between the dates of a site and year
def perm_r2(d, y, xs, by="site_year", n_perm=10000, seed=0):
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
JOINT = (("deep water (red) + bright stable ground", ["deep_water_red_diff", "bright_ground_diff_pct"]),
         ("deep water (red) + bright ground + zero or negative share", ["deep_water_red_diff", "bright_ground_diff_pct", "zero_or_negative_diff_points"]),
         ("deep water (red) + bright stable ground + tide", ["deep_water_red_diff", "bright_ground_diff_pct", "tide_eot20_m"]),
         ("tide alone", ["tide_eot20_m"]))
for name, xs in JOINT:
    for label, d in ((SAME, T), (SAME + ", without the 18 May 2020 pass", T[T.date != "2020-05-18"]), (SAME + ", without Al Nouf", T[T.site != "al_nouf"])):
        R2, p, n = perm_r2(d, "dev_pct", xs)
        joint.append({"explained_by": name, "compared": label, "scenes": n, "share_of_swing": R2, "p_shuffle_dates": p})
    R2, p, n = perm_r2(T, "dev_all_years_pct", [OLD_NAMES.get(x, x) for x in xs], by="site")
    joint.append({"explained_by": name, "compared": OLD, "scenes": n, "share_of_swing": R2, "p_shuffle_dates": p})
J = pd.DataFrame(joint)
J.to_csv(out / "track_joint.csv", index=False)

with pd.option_context("display.width", 280, "display.max_rows", 200, "display.max_colwidth", 70):
    print(R[["measure", "compared", "y", "scenes", "dates", "r_within", "share_explained", "r_each_group_equal_weight", "p_shuffle_dates",
             "leave_one_site_out_r_min", "leave_one_site_out_r_max"]].to_string(index=False))
    print(pd.DataFrame(per).to_string(index=False))
    print(pd.DataFrame(mods).to_string(index=False))
    print(pd.DataFrame(bands).pivot(index="band", columns="target", values="r_within").to_string())
    print(J.to_string(index=False))
