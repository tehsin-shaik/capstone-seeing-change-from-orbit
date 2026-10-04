"""
Figures for the landmark-detection results. Run AFTER landmarks.py (reads its CSVs).

  python figures.py --data "D:/MyStuff/University/SDP/data/jubail" --out results/registration

Makes:
  fig_search_areas.png      where each filter lets the detector look
  fig_landmarks.png         matched landmarks on both dates (kept vs rejected by RANSAC)
  fig_detector_comparison.png  alignment error per detector x filter (known-warp trials)
  fig_checkerboard.png      a known misalignment before and after correction
"""
import argparse
import csv
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from landmarks import (FILTERS, DETECTORS, load_scene, to_gray, stable_mask, run_pair,
                       random_affine, warp_bands)

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"none": "#2a78d6", "open_water": "#eb6834", "landcover": "#1baf7a"}
LABEL = {"none": "No filter", "open_water": "Ignore open water", "landcover": "Built-up/bare ground only"}
KEEP, REJECT = "#1baf7a", "#e34948"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "figure.facecolor": "white"})


def true_color(zip_path):
    import zipfile
    m = [n for n in zipfile.ZipFile(zip_path).namelist() if "True_color" in n][0]
    with rasterio.open(f"zip://{os.path.abspath(zip_path)}!{m}") as s:
        return np.clip(np.moveaxis(s.read(), 0, -1), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/registration")
    args = ap.parse_args()
    zips = sorted(glob.glob(os.path.join(args.data, "*.zip")))
    ref_zip, mov_zip = zips[0], zips[-1]
    nR, nM = [os.path.basename(z)[7:17] for z in (ref_zip, mov_zip)]   # the dates
    bR, _, _ = load_scene(ref_zip)
    bM, _, _ = load_scene(mov_zip)
    tcR, tcM = true_color(ref_zip), true_color(mov_zip)
    gR, gM = to_gray(bR), to_gray(bM)
    summary = list(csv.DictReader(open(os.path.join(args.out, "summary.csv"))))
    best = min(summary, key=lambda s: (-float(s["success_rate"]), float(s["worst_err_px"])))
    bd, bf = best["detector"], best["filter"]

    # 1) search areas -------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, f in zip(axs, FILTERS):
        m = stable_mask(bR, f) > 0
        img = tcR.copy()
        img[~m] = img[~m] * 0.25 + 0.75 * np.array([0.35, 0.35, 0.35])
        ax.imshow(img)
        ax.set_title(f"{LABEL[f]}  -  searches {m.mean():.0%} of image", fontsize=10, loc="left")
        ax.axis("off")
    fig.suptitle(f"Where the corner detector is allowed to look (grey = excluded), {nR}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_search_areas.png"), dpi=150)
    plt.close(fig)

    # 2) matched landmarks on the real pair ---------------------------------------------
    r = run_pair(gR, stable_mask(bR, bf), gM, stable_mask(bM, bf), bd)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.9))
    for ax, img, pts, name in [(axs[0], tcR, r["pRef"], nR), (axs[1], tcM, r["pMov"], nM)]:
        ax.imshow(img)
        ax.scatter(pts[~r["inl"], 0], pts[~r["inl"], 1], s=14, facecolors="none", edgecolors=REJECT, linewidths=1.0,
                   label=f"rejected by RANSAC ({(~r['inl']).sum()})")
        ax.scatter(pts[r["inl"], 0], pts[r["inl"], 1], s=12, c=KEEP, edgecolors="white", linewidths=0.5,
                   label=f"kept landmark ({r['inl'].sum()})")
        ax.set_title(name, loc="left")
        ax.axis("off")
    axs[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.suptitle(f"Matched landmarks, {bd.upper()} + '{LABEL[bf]}' filter", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_landmarks.png"), dpi=150)
    plt.close(fig)

    # 3) detector comparison ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(DETECTORS))
    w = 0.26
    for i, f in enumerate(FILTERS):
        med = [float(next(s for s in summary if s["detector"] == d and s["filter"] == f)["median_err_px"]) for d in DETECTORS]
        worst = [float(next(s for s in summary if s["detector"] == d and s["filter"] == f)["worst_err_px"]) for d in DETECTORS]
        xs = x + (i - 1) * (w + 0.02)
        ax.bar(xs, med, w, color=SERIES[f], label=f"{LABEL[f]} (bar = median)", zorder=3)
        ax.scatter(xs, worst, marker="_", s=120, color=INK, linewidths=1.6, zorder=4)
    ax.scatter([], [], marker="_", s=120, color=INK, linewidths=1.6, label="worst of 20 trials")
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(DETECTORS) - 0.5, 1.03, "1 px = 10 m", color=INK2, ha="right", va="bottom", fontsize=9)
    ax.set_xticks(x, [d.upper() if d in ("orb", "sift", "fast") else d.capitalize().replace("Shitomasi", "Shi-Tomasi")
                      for d in DETECTORS])
    ax.set_ylabel("alignment error (pixels)")
    ax.set_ylim(0, 1.3)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, frameon=False, ncol=4)
    ax.set_title("Error recovering a known misalignment, 2018 vs 2025 (lower is better)", loc="left")
    worst_all = max(float(s["worst_err_px"]) for s in summary)
    if worst_all > 1.3:
        ax.text(x[DETECTORS.index("orb")] + (w + 0.02), 1.27, f"worst {worst_all:.1f} px ↑", ha="center", va="top",
                fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_detector_comparison.png"), dpi=150)
    plt.close(fig)

    # 4) checkerboard before / after on one known warp ------------------------------------
    rng = np.random.default_rng(7)
    W, p = random_affine(rng, gR.shape)
    bW, validW = warp_bands(bM, W, gR.shape)
    gW = to_gray(bW, validW)
    rr = run_pair(gR, stable_mask(bR, bf), gW, stable_mask(bW, bf, validW), bd)
    h, wd = gR.shape
    fixed = cv2.warpAffine(gW, rr["M"], (wd, h))

    def checker(a, b, n=80):
        yy, xx = np.mgrid[0:h, 0:wd]
        return np.where(((yy // n) + (xx // n)) % 2 == 0, a, b)

    y0, x0, s_ = 300, 500, 450          # zoom on the Sheikh Khalifa Bridge / Saadiyat area
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.2))
    axs[0].imshow(checker(gR, gW)[y0:y0 + s_, x0:x0 + s_], cmap="gray")
    axs[0].set_title(f"Before: shifted {p['tx']:.1f}, {p['ty']:.1f} px, rotated {p['rot_deg']:.2f}°", loc="left", fontsize=10)
    axs[1].imshow(checker(gR, fixed)[y0:y0 + s_, x0:x0 + s_], cmap="gray")
    axs[1].set_title("After alignment from the landmarks", loc="left", fontsize=10)
    for a in axs:
        a.axis("off")
    fig.suptitle(f"Checkerboard of {nR} and {nM} squares - roads should run straight across square edges",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_checkerboard.png"), dpi=150)
    plt.close(fig)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
