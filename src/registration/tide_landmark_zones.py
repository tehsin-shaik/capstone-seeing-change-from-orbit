"""
Sort the landmarks from real_raster_check.py into shoreline and dry land, for a pair of
images taken at different tides.

Inputs are the two collect.py downloads (the 7-band float files, not the 6-band copies) and the
landmarks_gee_pair.csv that real_raster_check.py wrote for that pair.

Water rule: a pixel is water if green (B03) is brighter than near-infrared (B08). This is the
same as NDWI > 0 wherever both reflectances are positive. It is used instead of NDWI > 0 because
some scenes have slightly negative reflectance over deep water, and NDWI is meaningless there.
The NDWI > 0 share (negatives set to 0, which is what real_raster_check.py sees) is printed too.

Zones, for a landmark at a given pixel:
  shoreline  within --buffer-px pixels of water in either image
  dry land   further than that from water in both images
  tidal      part of shoreline: within --buffer-px pixels of a place that is water in only one image

What it writes into --out:
  landmark_zones.csv     one row per landmark: zone, kept by RANSAC or not, offset in metres
  zone_summary.csv       per zone: matches, kept, RMSE and mean shift of the kept landmarks
  fig_tide_landmarks.png both images and the water map with the kept landmarks on it

How to run:
  python src/registration/tide_landmark_zones.py --reference <2018 download.tif> \
      --moving <2025 download.tif> --landmarks <landmarks_gee_pair.csv> --out <folder>
"""

import argparse
import csv
import os

import cv2
import numpy as np
import rasterio


def load(path):
    """Green, NIR and a true-colour picture from a collect.py download."""
    with rasterio.open(path) as r:
        names = [(d or "")[:3] for d in r.descriptions]
        band = {n: r.read(names.index(n) + 1) for n in ("B02", "B03", "B04", "B08")}
    g, n = np.nan_to_num(band["B03"], nan=0.0), np.nan_to_num(band["B08"], nan=0.0)
    water = g > n
    gc, nc = np.clip(g, 0, None), np.clip(n, 0, None)
    ndwi_water = (gc - nc) / (gc + nc + 1e-6) > 0
    rgb = np.clip(np.nan_to_num(np.dstack([band["B04"], band["B03"], band["B02"]]), nan=0.0), 0, None)
    lo, hi = np.percentile(rgb, [2, 98])
    return water, ndwi_water, np.clip((rgb - lo) / (hi - lo), 0, 1)


def near(mask, px):
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def at(mask, xy):
    h, w = mask.shape
    x = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
    return mask[y, x]


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="collect.py download of the earlier date")
    ap.add_argument("--moving", required=True, help="collect.py download of the later date")
    ap.add_argument("--landmarks", required=True, help="landmarks_gee_pair.csv from real_raster_check.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer-px", type=int, default=3, help="how close to water counts as shoreline (default 3 px = 30 m)")
    ap.add_argument("--title", default="", help="text for the figure title, e.g. the site and dates")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    w_ref, ndwi_ref, rgb_ref = load(a.reference)
    w_mov, ndwi_mov, rgb_mov = load(a.moving)
    if w_ref.shape != w_mov.shape:
        raise SystemExit("the two images are not on the same grid")
    shore_map = near(w_ref | w_mov, a.buffer_px)
    tidal_map = near(w_ref ^ w_mov, a.buffer_px)

    rows = list(csv.DictReader(open(a.landmarks, newline="")))
    cols = list(rows[0].keys())            # id, x/y px ref, x/y px mov, E/N ref, E/N mov, ransac_inlier
    val = np.array([[float(r[c]) for c in cols[1:9]] for r in rows])
    px_ref, px_mov, en_ref, en_mov = val[:, 0:2], val[:, 2:4], val[:, 4:6], val[:, 6:8]
    kept = np.array([int(r["ransac_inlier"]) == 1 for r in rows])
    d = en_ref - en_mov                    # how far the later image has to move to sit on the earlier one
    err = np.linalg.norm(d, axis=1)

    # a landmark has a position in each image; it counts as shoreline if either one is near water
    shoreline = at(shore_map, px_ref) | at(shore_map, px_mov)
    tidal = at(tidal_map, px_ref) | at(tidal_map, px_mov)
    zone = np.where(shoreline, "shoreline", "dry land")

    with open(os.path.join(a.out, "landmark_zones.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "zone", "tidal", "ransac_inlier", "offset_east_m", "offset_north_m", "offset_m"])
        for i, r in enumerate(rows):
            w.writerow([r["id"], zone[i], int(tidal[i]), int(kept[i]), round(d[i, 0], 2), round(d[i, 1], 2), round(err[i], 2)])

    # share of the image in each zone, leaving out pixels that are water in both images
    land_any = ~(w_ref & w_mov)
    groups = [("all", np.ones(len(rows), bool), 1.0),
              ("dry land", ~shoreline, float((land_any & ~shore_map).sum() / land_any.sum())),
              ("shoreline", shoreline, float((shore_map & land_any).sum() / land_any.sum())),
              ("tidal (part of shoreline)", tidal, float((tidal_map & land_any).sum() / land_any.sum()))]
    summary = []
    for name, sel, area in groups:
        k = sel & kept
        mean = d[k].mean(0) if k.any() else np.array([np.nan, np.nan])
        summary.append(dict(zone=name, matches=int(sel.sum()), kept=int(k.sum()),
                            kept_share=round(float(k.sum() / sel.sum()), 3) if sel.any() else "",
                            share_of_kept=round(float(k.sum() / kept.sum()), 3),
                            share_of_non_water_area=round(area, 3),
                            rmse_no_correction_m=round(rmse(err[k]), 2),
                            shift_east_m=round(float(mean[0]), 2), shift_north_m=round(float(mean[1]), 2),
                            shift_m=round(float(np.linalg.norm(mean)), 2)))
    with open(os.path.join(a.out, "zone_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    rng = np.random.default_rng(0)
    dk = d[kept]
    boot = np.array([np.linalg.norm(dk[rng.integers(0, len(dk), len(dk))].mean(0)) for _ in range(5000)])
    print(f"water (green > NIR): reference {w_ref.mean():.1%}, moving {w_mov.mean():.1%}; "
          f"water in only one image {np.mean(w_ref ^ w_mov):.1%}")
    print(f"water (NDWI > 0, negatives set to 0): reference {ndwi_ref.mean():.1%}, moving {ndwi_mov.mean():.1%}")
    print(f"whole-image shift {np.linalg.norm(dk.mean(0)):.2f} m (east {dk[:, 0].mean():+.2f}, north {dk[:, 1].mean():+.2f}), "
          f"bootstrap 95% range {np.percentile(boot, 2.5):.2f} to {np.percentile(boot, 97.5):.2f} m")
    for s in summary:
        print(f"  {s['zone']:26s} matches {s['matches']:4d}  kept {s['kept']:4d} ({s['kept_share']:.0%})  "
              f"share of kept {s['share_of_kept']:.0%}  area {s['share_of_non_water_area']:.0%}  "
              f"RMSE {s['rmse_no_correction_m']:.2f} m  shift {s['shift_m']:.2f} m "
              f"(E {s['shift_east_m']:+.2f}, N {s['shift_north_m']:+.2f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zmap = np.zeros(w_ref.shape + (3,))
    zmap[~w_ref & ~w_mov] = (0.88, 0.88, 0.84)     # land in both
    zmap[w_ref & w_mov] = (0.10, 0.20, 0.55)       # water in both
    zmap[w_ref & ~w_mov] = (0.95, 0.60, 0.20)      # water only in the earlier image
    zmap[~w_ref & w_mov] = (0.35, 0.75, 0.95)      # water only in the later image
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.6))
    ax[0].imshow(rgb_ref)
    ax[0].set_title(f"earlier image, water {w_ref.mean():.0%}", fontsize=10)
    ax[1].imshow(rgb_mov)
    ax[1].set_title(f"later image, water {w_mov.mean():.0%}", fontsize=10)
    ax[2].imshow(zmap)
    ax[2].scatter(*px_ref[kept & ~shoreline].T, s=7, c="black", label=f"kept, dry land ({int((kept & ~shoreline).sum())})")
    ax[2].scatter(*px_ref[kept & shoreline].T, s=9, c="red", label=f"kept, shoreline ({int((kept & shoreline).sum())})")
    ax[2].set_title("dark blue: water in both, light blue: only later, orange: only earlier", fontsize=10)
    ax[2].legend(loc="lower left", fontsize=8, framealpha=0.9)
    for x in ax:
        x.axis("off")
    if a.title:
        fig.suptitle(a.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if a.title else 1))
    fig.savefig(os.path.join(a.out, "fig_tide_landmarks.png"), dpi=65)
    plt.close(fig)


if __name__ == "__main__":
    main()
