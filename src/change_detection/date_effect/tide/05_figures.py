"""Scatter plots for Test 1, from <out>/scene_table.csv.

    test1_all_sites.png   all sites together after removing each site's average:
                          the scene's predicted-mangrove deviation against (left) the water share seen in the image
                          and (right) the tide height from the EOT20 model.
    test1_per_site.png    the same two plots for each site on its own.

Colour and marker shape show the season of the scene, because season is the thing tide could be confused with.
The numbers behind every point are in scene_table.csv.

Usage: python 05_figures.py --out <results folder>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
T = pd.read_csv(out / "scene_table.csv")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEASONS = {"November": ("#2a78d6", "o"), "January to March": ("#eb6834", "s"), "May or September": ("#1baf7a", "^")}
NAMES = {"al_aryam_islands": "Al Aryam Islands", "al_nouf": "Al Nouf", "eastern_mangroves": "Eastern Mangroves",
         "ghantoot": "Ghantoot", "jubail_island": "Jubail Island", "ras_ghurab": "Ras Ghurab"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def dots(ax, d, x, y, size=52):
    for season, (col, mk) in SEASONS.items():
        g = d[d["season"] == season]
        ax.scatter(g[x], g[y], s=size, c=col, marker=mk, edgecolors=SURFACE, linewidths=1.4, label=season, zorder=3)


def fit_line(ax, x, y):
    if len(x) >= 3 and np.std(x) > 0:
        b, c = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, b * xs + c, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)


def r_of(x, y):
    return np.corrcoef(x, y)[0, 1]


# ---------------------------------------------------------------- all sites together, site averages removed
D = T.copy()
for c in ("water_pct", "tide_eot20_m", "dev_pct"):
    D[c + "_d"] = D[c] - D.groupby("site")[c].transform("mean")
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
PANELS = [("water_pct_d", "Water share of the site in the image, minus the site's average (points)", "Water seen in the image"),
          ("tide_eot20_m_d", "Tide height from the EOT20 model, minus the site's average (m)", "Tide from the tide model")]
for ax, (x, xl, title) in zip(axes, PANELS):
    style(ax)
    ax.axvline(0, color=INK2, linewidth=0.8)
    fit_line(ax, D[x].values, D["dev_pct_d"].values)
    dots(ax, D, x, "dev_pct_d")
    ax.set_xlabel(xl)
    ax.set_title(f"{title}:  r = {r_of(D[x], D['dev_pct_d']):+.2f},  {len(D)} scenes, {D['site'].nunique()} sites",
                 loc="left", fontsize=10, color=INK)
    # name the far-out points so the reader can see which site drives the picture
    far = D.reindex(D["dev_pct_d"].abs().sort_values(ascending=False).index).head(4)
    for k, (_, r) in enumerate(far.iterrows()):
        # two of the four sit close together (the 18 May 2020 pass), so one label goes above and one below
        ax.annotate(f"{NAMES[r['site']]} {r['date']}", (r[x], r["dev_pct_d"]), xytext=(7, -13 if k == 2 else 5),
                    textcoords="offset points", fontsize=7.5, color=INK2)
axes[0].set_ylabel("Predicted mangrove area, % above or below the site's average")
axes[0].legend(title="Season of the scene", frameon=False, loc="lower left", fontsize=8.5, title_fontsize=8.5)
fig.suptitle("Does water level go with the swings in predicted mangrove area?  Each point is one scene.",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out / "test1_all_sites.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- per site
sites = sorted(T["site"].unique())
fig, axes = plt.subplots(2, len(sites), figsize=(3.05 * len(sites), 6.4))
for j, s in enumerate(sites):
    g = T[T["site"] == s]
    for i, (x, xl) in enumerate((("water_pct", "Water share in the image (%)"), ("tide_eot20_m", "Model tide height (m)"))):
        ax = axes[i, j]
        style(ax)
        fit_line(ax, g[x].values, g["dev_pct"].values)
        dots(ax, g, x, "dev_pct", size=44)
        ax.set_title(f"{NAMES[s] if i == 0 else ''}\nr = {r_of(g[x], g['dev_pct']):+.2f}, {len(g)} scenes".strip(),
                     loc="left", fontsize=9, color=INK)
        ax.set_xlabel(xl, fontsize=8)
        if j == 0:
            ax.set_ylabel("Predicted mangrove area,\n% from the site's average", fontsize=8)
        ax.margins(0.12)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, title="Season of the scene", frameon=False, loc="upper right", ncol=3, fontsize=8.5, title_fontsize=8.5)
fig.suptitle("The same, site by site (each panel has its own scale)", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out / "test1_per_site.png", dpi=170)
plt.close(fig)
print("wrote test1_all_sites.png and test1_per_site.png")
