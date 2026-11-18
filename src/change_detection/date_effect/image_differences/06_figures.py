"""Figures for the image-comparability check.

    track_swings.png        part b: each scene's same-year swing in predicted mangrove area against (left) how bright deep
                            water is, (middle) how bright stable ground is, (right) the swing the two together would predict.
                            Everything is the difference from the average of the same site and year.
                            Colour and marker shape show the season.
    before_after.png        part c: per site, the average same-year swing and the held-out IoU, on the images as they are
                            and after the correction (gain and offset, typical date as reference).

The numbers behind the points are in scene_table_images.csv, site_swing.csv and summary.csv.

Usage: python 06_figures.py --out <results folder>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared import NAMES, SITES, demean

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SEASONS = {"November": ("#2a78d6", "o"), "January to March": ("#eb6834", "s"), "May or September": ("#1baf7a", "^")}
BEFORE, AFTER = "#8a8983", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax, zero=True):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if zero:
        ax.axhline(0, color=INK2, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- part b
T = pd.read_csv(out / "scene_table_images.csv")
m = pd.to_datetime(T["date"]).dt.month
T["season"] = np.where(m == 11, "November", np.where(m <= 3, "January to March", "May or September"))
# every value is the difference from the average of the same site and the same year
D = T.dropna(subset=["dev_pct", "deep_water_red_diff", "bright_ground_diff_pct"]).copy()
xs = ["deep_water_red_diff", "bright_ground_diff_pct"]
dm = demean(D, ["dev_pct"] + xs)
beta = np.linalg.lstsq(dm[xs].values, dm["dev_pct"].values, rcond=None)[0]
D["dev_d"], D["dw"], D["bg"] = dm["dev_pct"], 100 * dm["deep_water_red_diff"], dm["bright_ground_diff_pct"]
D["fit"] = dm[xs].values @ beta
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True)
PANELS = [("dw", "Deep water in the red band (reflectance x 100),\ndifference from the average of the same site and year", "Deep water brightness"),
          ("bg", "Bright stable ground (%),\ndifference from the average of the same site and year", "Bright stable ground brightness"),
          ("fit", "Same-year swing predicted\nfrom the two measures together (%)", "Both together")]
for ax, (x, xl, title) in zip(axes, PANELS):
    style(ax)
    ax.axvline(0, color=INK2, linewidth=0.8)
    b, c = np.polyfit(D[x], D["dev_d"], 1)
    xx = np.array([D[x].min(), D[x].max()])
    ax.plot(xx, b * xx + c, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    for season, (col, mk) in SEASONS.items():
        g = D[D["season"] == season]
        ax.scatter(g[x], g["dev_d"], s=52, c=col, marker=mk, edgecolors=SURFACE, linewidths=1.4, label=season, zorder=3)
    r = np.corrcoef(D[x], D["dev_d"])[0, 1]
    ax.set_title(f"{title}:  r = {r:+.2f},  {len(D)} scenes", loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xl)
    far = D.reindex(D["dev_d"].abs().sort_values(ascending=False).index).head(2)      # the highest and the lowest scene
    for k, (_, rr) in enumerate(far.iterrows()):
        left = rr[x] > np.mean(ax.get_xlim())                                           # keep the label inside its panel
        ax.annotate(f"{NAMES[rr['site']]} {rr['date']}", (rr[x], rr["dev_d"]), xytext=(-7 if left else 7, 5), textcoords="offset points",
                    ha="right" if left else "left", fontsize=7.5, color=INK2)
axes[0].set_ylabel("Same-year swing: predicted mangrove area,\n% above or below the average of the same site and year")
axes[0].legend(title="Season of the scene", frameon=False, loc="lower left", fontsize=8.5, title_fontsize=8.5)
fig.suptitle("Do image differences on things that should not change go with the same-year swings?  Each point is one scene.",
             x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out / "track_swings.png", dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- part c
S = pd.read_csv(out / "site_swing.csv")
Hd = pd.read_csv(out / "summary.csv").set_index("version")
b0 = S[S.version == "as_is"].set_index("site")
b1 = S[S.version == "gain_offset"].set_index("site")
rows = [(NAMES[s], b0.loc[s, "mean_abs_swing_pct"], b1.loc[s, "mean_abs_swing_pct"], b0.loc[s, "iou_2020"], b1.loc[s, "iou_2020"]) for s in SITES]
rows.append(("Mean of the six sites\n(IoU: mean of the five folds)", Hd.loc["as_is", "six_site_mean_abs_swing_pct"], Hd.loc["gain_offset", "six_site_mean_abs_swing_pct"],
             Hd.loc["as_is", "iou_mean_over_5_folds"], Hd.loc["gain_offset", "iou_mean_over_5_folds"]))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
ypos = np.arange(len(rows))[::-1].astype(float)
ypos[-1] -= 0.5                                               # a gap before the mean row
for ax, (i0, i1, xl, title, fmt) in zip(axes, [(1, 2, "Average size of the same-year swing (% of predicted mangrove area)",
                                                "Same-year swing: smaller is better", "{:.1f}"),
                                               (3, 4, "Held-out IoU on the non-hazy 2020 chips", "Held-out IoU: larger is better", "{:.3f}")]):
    style(ax, zero=False)
    ax.grid(axis="y", visible=False)
    for y, r in zip(ypos, rows):
        v0, v1 = r[i0], r[i1]
        ax.plot([v0, v1], [y, y], color=INK2, linewidth=1.4, zorder=2)
        ax.scatter([v0], [y], s=70, c=BEFORE, marker="s", edgecolors=SURFACE, linewidths=1.4, zorder=3, label="Images as they are" if y == ypos[0] else None)
        ax.scatter([v1], [y], s=70, c=AFTER, marker="o", edgecolors=SURFACE, linewidths=1.4, zorder=4, label="After the correction" if y == ypos[0] else None)
        lo, hi = (v0, v1) if v0 <= v1 else (v1, v0)
        ax.annotate(fmt.format(lo), (lo, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK2)
        ax.annotate(fmt.format(hi), (hi, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel(xl)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.margins(x=0.18)
    ax.set_ylim(ypos.min() - 0.7, ypos.max() + 0.7)
axes[0].legend(frameon=False, loc="lower right", fontsize=8.5)
fig.suptitle("Same forests, images corrected so stable ground matches the site's typical date", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out / "before_after.png", dpi=170)
plt.close(fig)
print("wrote track_swings.png and before_after.png")
