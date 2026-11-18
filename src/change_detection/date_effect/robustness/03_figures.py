"""Figures for the steadiness comparison.

    sensitivity_A.png   how far the current forest's mangrove area moves when every image is made brighter or darker
                        (left: bands multiplied, right: a flat amount added). One line per site.
    versions.png        per site, the average same-year swing of versions A, B, C, D and the held-out IoU of A, B, C.

The numbers are in sensitivity_A.csv, site_table.csv and summary.csv.

Usage: python 03_figures.py --out <results folder>
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

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SITES = ["al_aryam_islands", "al_nouf", "eastern_mangroves", "ghantoot", "jubail_island", "ras_ghurab"]
NAMES = {"al_aryam_islands": "Al Aryam Islands", "al_nouf": "Al Nouf", "eastern_mangroves": "Eastern Mangroves",
         "ghantoot": "Ghantoot", "jubail_island": "Jubail Island", "ras_ghurab": "Ras Ghurab"}
SITE_COL = dict(zip(SITES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))     # fixed order, one hue per site
SITE_MK = dict(zip(SITES, ["o", "s", "^", "D", "v", "P"]))
VERS = {"A": ("#8a8983", "s", "A  current forest"), "B": ("#2a78d6", "o", "B  brightness augmentation"),
        "C": ("#eb6834", "^", "C  indices only"), "D": ("#1baf7a", "D", "D  A with probability sum")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- sensitivity of A
S = pd.read_csv(out / "sensitivity_A.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, (kind, xl, title) in zip(axes, [("scale", "Every band multiplied: change in brightness (%)", "Brightness scaled"),
                                        ("offset", "A flat amount added to every band (reflectance)", "Brightness shifted (not asked for)")]):
    style(ax)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ends = []
    for s in SITES:
        g = S[(S.site == s) & (S.change == kind)].sort_values("amount")
        x = np.concatenate([g.amount[g.amount < 0], [0], g.amount[g.amount > 0]])
        y = np.concatenate([g.area_change_pct[g.amount < 0], [0], g.area_change_pct[g.amount > 0]])
        ax.plot(x, y, color=SITE_COL[s], linewidth=2, marker=SITE_MK[s], markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=NAMES[s])
        ends.append((y[-1], s, x[-1]))
    # direct labels at the right end, pushed apart so they do not sit on each other
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    ypos = []
    for y, s, x in ends:
        ypos.append(y if not ypos else max(y, ypos[-1] + gap))
    for (y, s, x), yy in zip(ends, ypos):
        ax.annotate(NAMES[s], (x, y), xytext=(x + (0.6 if kind == "scale" else 0.0012), yy), fontsize=8, color=INK2, va="center")
    ax.set_xlabel(xl)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.margins(x=0.22)
axes[0].set_ylabel("Change in predicted mangrove area (%)")
axes[0].legend(frameon=False, loc="lower left", fontsize=8.5, title="Site", title_fontsize=8.5)
fig.suptitle("How far the current forest's mangrove area moves when the image is made brighter or darker", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out / "sensitivity_A.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- the four versions
T = pd.read_csv(out / "site_table.csv")
H = pd.read_csv(out / "summary.csv").set_index("version")
labels = [NAMES[s] for s in SITES] + ["Mean of the six sites\n(IoU: mean of the five folds)"]
ypos = np.arange(len(labels))[::-1].astype(float)
ypos[-1] -= 0.5
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for ax, (col, hcol, vers, xl, title) in zip(axes, [
        ("mean_abs_swing_pct", "six_site_mean_abs_swing_pct", ["A", "B", "C", "D"], "Average size of the same-year swing (% of predicted mangrove area)",
         "Same-year swing: smaller is better"),
        ("iou_all_2020_chips", "iou_mean_of_5_folds_all_2020_chips", ["A", "B", "C"], "Held-out IoU, all 2020 chips of the site", "Held-out IoU: larger is better (D is A)")]):
    style(ax)
    ax.grid(axis="y", visible=False)
    for y, s in zip(ypos, SITES + [None]):
        vals = [(H.loc[v, hcol] if s is None else T[(T.version == v) & (T.site == s)][col].iloc[0]) for v in vers]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=2.2, zorder=1)
        for v, val in zip(vers, vals):
            c, mk, lab = VERS[v]
            ax.scatter([val], [y], s=74, c=c, marker=mk, edgecolors=SURFACE, linewidths=1.4, zorder=3, label=lab if y == ypos[0] else None)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel(xl)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylim(ypos.min() - 0.7, ypos.max() + 0.7)
axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
fig.suptitle("Four versions of the held-out random forest: which is steadiest across dates?", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out / "versions.png", dpi=170)
plt.close(fig)
print("wrote sensitivity_A.png and versions.png")
