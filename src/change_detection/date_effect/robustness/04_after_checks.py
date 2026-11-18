"""Checks added AFTER the results were seen, to read version D correctly. Not part of the comparison.

The probability sum counts a little mangrove in every pixel, also where the forest is far below its threshold, so it is a
bigger number than the thresholded area. A swing in percent of a bigger number looks smaller even if the same amount of
area moved. This puts A and D side by side in hectares, and repeats the main number without the one-pair site.

Writes <out>/after_checks_D.csv.   Usage: python 04_after_checks.py --out <results folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
P = pd.read_csv(out / "same_year_pairs.csv")
P["net_ha"] = P["net_pct_of_first"] * P["area_first_ha"] / 100
key = ["site", "first_date", "second_date"]
w = P[P.version.isin(["A", "B", "C", "D"])].pivot_table(index=key, columns="version", values=["area_first_ha", "net_ha", "net_pct_of_first"])
rows = []
for site, g in w.groupby(level=0):
    rows.append({"site": site, "pairs": len(g),
                 "A_area_first_ha": g[("area_first_ha", "A")].mean(), "D_probability_sum_first_ha": g[("area_first_ha", "D")].mean(),
                 "D_sum_over_A_area": g[("area_first_ha", "D")].sum() / g[("area_first_ha", "A")].sum(),
                 "A_mean_abs_swing_ha": g[("net_ha", "A")].abs().mean(), "D_mean_abs_swing_ha": g[("net_ha", "D")].abs().mean(),
                 "B_mean_abs_swing_ha": g[("net_ha", "B")].abs().mean(), "C_mean_abs_swing_ha": g[("net_ha", "C")].abs().mean(),
                 "A_mean_abs_swing_pct": g[("net_pct_of_first", "A")].abs().mean(), "D_mean_abs_swing_pct": g[("net_pct_of_first", "D")].abs().mean(),
                 "pairs_where_D_moves_the_same_way_as_A": int((np.sign(g[("net_ha", "A")]) == np.sign(g[("net_ha", "D")])).sum())})
R = pd.DataFrame(rows)
tot = {"site": "MEAN OF THE SIX SITES", "pairs": int(R.pairs.sum()), **{c: R[c].mean() for c in R.columns if c not in ("site", "pairs", "pairs_where_D_moves_the_same_way_as_A")},
       "pairs_where_D_moves_the_same_way_as_A": int(R.pairs_where_D_moves_the_same_way_as_A.sum())}
five = R[R.site != "al_nouf"]
tot5 = {"site": "MEAN WITHOUT AL NOUF (one pair)", "pairs": int(five.pairs.sum()), **{c: five[c].mean() for c in R.columns if c not in ("site", "pairs", "pairs_where_D_moves_the_same_way_as_A")},
        "pairs_where_D_moves_the_same_way_as_A": int(five.pairs_where_D_moves_the_same_way_as_A.sum())}
R = pd.concat([R, pd.DataFrame([tot, tot5])], ignore_index=True)
R["D_swing_in_ha_as_pct_of_A_swing_in_ha"] = 100 * R.D_mean_abs_swing_ha / R.A_mean_abs_swing_ha
R.round(3).to_csv(out / "after_checks_D.csv", index=False)
ap_ = w[("net_pct_of_first", "A")].abs(); dp = w[("net_pct_of_first", "D")].abs(); bp = w[("net_pct_of_first", "B")].abs(); cp = w[("net_pct_of_first", "C")].abs()
with pd.option_context("display.width", 260, "display.float_format", "{:.2f}".format):
    print(R.to_string(index=False))
    print(f"pairs with a smaller percent swing than A: B {int((bp < ap_).sum())}, C {int((cp < ap_).sum())}, D {int((dp < ap_).sum())} of {len(w)}")
    print(f"correlation of the pair swings (signed, percent) with A: B {np.corrcoef(w[('net_pct_of_first','A')], w[('net_pct_of_first','B')])[0,1]:.2f}, "
          f"C {np.corrcoef(w[('net_pct_of_first','A')], w[('net_pct_of_first','C')])[0,1]:.2f}, D {np.corrcoef(w[('net_pct_of_first','A')], w[('net_pct_of_first','D')])[0,1]:.2f}")
