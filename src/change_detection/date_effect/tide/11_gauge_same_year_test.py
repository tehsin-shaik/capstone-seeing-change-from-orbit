"""The same-year tide test again, with tide gauge heights in place of the tide model.

The swing (dev_pct) and the test are the ones in 04_analysis.py: every value is the difference from the average of
the same site and year, and the chance figure comes from shuffling the dates inside each site and year (sites that
share a calendar date get the same shuffle).

No gauge is in the UAE, so three heights are tried, and each row of the table says which it is:
    gauge height itself        the level at a gauge in other water (Gulf of Oman or Kuwait), not at our sites
    model + gauge difference   EOT20 at the site, plus how far the gauge was above or below EOT20 at the gauge at
                               that moment. A rough stand-in for what the weather did to the level.
    model, same scenes         EOT20 at the site on the scenes that gauge covers, to compare like with like

Reads <tide out>/scene_table.csv, <out>/gauge_at_overpass.csv and <work>/model_grid.csv.
Writes <out>/gauge_same_year_test.csv and <out>/gauge_vs_model.png

Usage: python 11_gauge_same_year_test.py --work <gauge work folder> --tide-out <tide out> --out <tide out>/gauge
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--tide-out", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, tide_out, out = Path(a.work), Path(a.tide_out), Path(a.out)
RNG = np.random.default_rng(0)
BY = "site_year"

T = pd.read_csv(tide_out / "scene_table.csv")
T = T.dropna(subset=["dev_pct"]).reset_index(drop=True)          # a site and year with one date has no same-year swing
G = pd.read_csv(out / "gauge_at_overpass.csv")
G = G[G["usable"]]
M = pd.read_csv(work / "model_grid.csv", parse_dates=["time"])

# the site heights from 09 must be the ones the tide test used
ms = M[(M["kind"] == "site") & M["at_overpass"]].copy()
ms["date"] = ms["time"].dt.strftime("%Y-%m-%d")
chk = T.merge(ms[["place", "date", "eot20_m"]], left_on=["site", "date"], right_on=["place", "date"])
worst = float((chk["eot20_m"] - chk["tide_eot20_m"]).abs().max())
print("site heights against the ones in the tide test:", len(chk), "scenes, largest difference", round(worst, 4), "m")
assert len(chk) == len(T) and worst < 0.002


def demean(d, cols):
    return d[cols] - d.groupby(BY)[cols].transform("mean")


def perm_p(d, x, y, n_perm=20000):
    """Same shuffle as perm_p in 04_analysis.py."""
    d = d.dropna(subset=[x, y]).sort_values([BY, "date"])
    yd = demean(d, [y])[y].values
    obs = np.corrcoef(demean(d, [x])[x].values, yd)[0, 1]
    alld = sorted(d["date"].unique())
    idx = {dd: i for i, dd in enumerate(alld)}
    groups = [(g.index.values, np.array([idx[v] for v in g["date"]]), g[x].values) for _, g in d.groupby(BY)]
    pos = {ix: k for k, ix in enumerate(d.index)}
    hits = 0
    for _ in range(n_perm):
        key = RNG.random(len(alld))
        xp = np.empty(len(d))
        for ix, di, xv in groups:
            order = np.argsort(key[di])
            vals = xv - xv.mean()
            for k, o in enumerate(order):
                xp[pos[ix[o]]] = vals[k]
        if abs(np.corrcoef(xp, yd)[0, 1]) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def test(d, x, label, what):
    d = d.dropna(subset=[x, "dev_pct"])
    d = d[d.groupby(BY)[x].transform("size") >= 2]
    dm = demean(d, [x, "dev_pct"])
    r = np.corrcoef(dm[x], dm["dev_pct"])[0, 1]
    return dict(height=label, what_it_is=what, scenes=len(d), site_years=d[BY].nunique(), passes=d["date"].nunique(),
                years=" ".join(str(y) for y in sorted(d["year"].unique())), r_same_year=round(float(r), 3),
                share_of_same_year_swing_pct=round(100 * float(r * r), 1), chance_shuffled_dates_pct=round(100 * perm_p(d, x, "dev_pct"), 1))


rows = [test(T, "tide_eot20_m", "EOT20 model at the site, all scenes", "the test as it stands")]
for name, key in [("Muscat, Oman (UHSLC hourly)", "muscat"), ("Majis, Oman (IOC)", "majis"), ("Khawr Wudam, Oman (IOC)", "wudam"),
                  ("Salmiya, Kuwait (IOC)", "kuwait")]:
    g = G[G["gauge"] == name].set_index("date")
    T[key] = T["date"].map(g["gauge_m"])
    T[key + "_diff"] = T["date"].map(g["gauge_minus_model_m"])
    T[key + "_adj"] = T["tide_eot20_m"] + T[key + "_diff"]
    sub = T[T[key].notna()]
    rows.append(test(sub, key, f"{name}: gauge height itself", "level in other water, not at our sites"))
    rows.append(test(sub, key + "_adj", f"{name}: model at the site + gauge difference", "model plus the weather effect seen at the gauge"))
    rows.append(test(sub, "tide_eot20_m", f"{name}: model at the site, same scenes", "like-for-like comparison"))
# every scene: the model at the site plus the difference at the gauge that covers that year
T["all_adj"] = T["tide_eot20_m"] + T["muscat_diff"].fillna(T["kuwait_diff"])
rows.append(test(T, "all_adj", "Model at the site + gauge difference, all scenes (Muscat for 2018 and 2020, Kuwait for 2025)",
                 "model plus the weather effect seen at the gauge"))
R = pd.DataFrame(rows)
R.to_csv(out / "gauge_same_year_test.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 95)
print(R.drop(columns=["what_it_is"]).to_string(index=False))

# ---------------------------------------------------------------- figure
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("Muscat, Oman (UHSLC hourly)", "Muscat, Oman (2018 and 2020 dates)", "#2a78d6", "o"),
          ("Majis, Oman (IOC)", "Majis, Oman (2018 and 2020 dates)", "#eb6834", "s"),
          ("Salmiya, Kuwait (IOC)", "Salmiya, Kuwait (2025 dates)", "#1baf7a", "^")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
site_h = T.groupby("date")["tide_eot20_m"].mean()                              # one height per pass: mean of the sites imaged that day
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), sharey=True)
PANELS = [("eot20_at_gauge_m", "EOT20 tide model at the gauge position (m)", "Against the model at the gauge position: they agree"),
          ("site", "EOT20 tide model at our sites, as used in the tide test (m)", "Against the model at our sites: no relation")]
for ax, (xcol, xl, title) in zip(axes, PANELS):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    if xcol == "eot20_at_gauge_m":
        ax.plot([-1, 1], [-1, 1], color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.annotate("equal", (0.86, 0.86), xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    for name, lab, colour, mk in SERIES:
        g = G[G["gauge"] == name]
        x = g[xcol] if xcol != "site" else g["date"].map(site_h)
        ax.scatter(x, g["gauge_m"], s=56, c=colour, marker=mk, edgecolors=SURFACE, linewidths=1.4, label=lab, zorder=3)
    ax.set_xlabel(xl)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
axes[0].set_ylabel("Water level measured by the gauge at the overpass time,\nminus its 15-day average (m)")
axes[0].legend(title="Tide gauge", frameon=False, loc="upper left", fontsize=8.5, title_fontsize=8.5)
fig.suptitle("Tide gauges at the satellite overpass times.  Each point is one scene date at one gauge.", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out / "gauge_vs_model.png", dpi=170)
print("wrote gauge_vs_model.png")
