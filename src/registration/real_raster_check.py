"""
Track A: checking the alignment on the real 6-band GeoTIFFs, not just on landmark
coordinates. (Tehsin Shaik)

Inputs are Tehsin's Earth Engine exports of Jubail Island (B2 B3 B4 B8 B11 B12,
uint16, EPSG:32640, 10 m): 2018-12-10 is the reference, 2025-12-10 is moving.

What it does:
1. Runs Khaled's landmark pipeline (SIFT + "ignore open water", from landmarks.py)
   on that export and repeats the model choice from affine_alignment.py. This
   checks whether his result (the pair is already aligned) holds on a
   separate export of the same dates.
2. Round trip on the real raster: warps the 2025 GeoTIFF by a known random
   affine, finds the warp again from SIFT landmarks, and undoes it. If the
   transform direction or the pixel/map conversion were wrong anywhere, the
   image wouldn't come back.
3. Writes the aligned 2025 raster using the transform picked in step 1.

How to run (from the repo root):
  python src/registration/real_raster_check.py \
      --reference data/raw/Jubail_2018_12_10_6band.tif --moving data/raw/Jubail_2025_12_10_6band.tif
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affine_alignment as aa  # noqa: E402
import landmarks as lm  # noqa: E402

REPO = HERE.parents[1]
# Earth Engine names bands B2, B3...; landmarks.py expects B02, B03...
BAND_KEYS = {"B2": "B02", "B3": "B03", "B4": "B04", "B8": "B08"}


def load_tif(path):
    """Read the bands landmarks.py needs, the valid-pixel mask and the pixel -> map transform."""
    with rasterio.open(path) as r:
        bands = {BAND_KEYS[d]: r.read(i + 1).astype(np.float32)
                 for i, d in enumerate(r.descriptions) if d in BAND_KEYS}
        valid = r.read_masks(1) > 0
        t = r.transform
        # shifted by half a pixel so it gives pixel centres, same as landmarks.py
        G = np.array([[t.a, t.b, t.c + 0.5 * (t.a + t.b)], [t.d, t.e, t.f + 0.5 * (t.d + t.e)]])
        return bands, valid, G, (r.height, r.width)


def sift_landmarks(ref_path, mov_path):
    """Khaled's SIFT matching between two GeoTIFFs, with the matches also in map metres."""
    bR, vR, GR, shape = load_tif(ref_path)
    bM, vM, GM, _ = load_tif(mov_path)
    res = lm.run_pair(lm.to_gray(bR, vR), lm.stable_mask(bR, "open_water", vR),
                      lm.to_gray(bM, vM), lm.stable_mask(bM, "open_water", vM), "sift")
    pR, pM = res["pRef"].astype(float), res["pMov"].astype(float)
    return dict(px_ref=pR, px_mov=pM, en_ref=aa.apply(GR, pR), en_mov=aa.apply(GM, pM),
                inl=res["inl"], M_px=res["M"], G=GR, shape=shape)


def write_landmarks(path, L, ref_name="2018", mov_name="2025"):
    # same columns as Khaled's landmarks_for_tehsin.csv so both can be read the same way
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", f"x_px_{ref_name}", f"y_px_{ref_name}", f"x_px_{mov_name}", f"y_px_{mov_name}",
                    f"easting_{ref_name}_m", f"northing_{ref_name}_m", f"easting_{mov_name}_m",
                    f"northing_{mov_name}_m", "ransac_inlier"])
        for i in range(len(L["inl"])):
            w.writerow([i, *np.round(L["px_ref"][i], 2), *np.round(L["px_mov"][i], 2),
                        *np.round(L["en_ref"][i], 2), *np.round(L["en_mov"][i], 2), int(L["inl"][i])])


def band_agreement(a_path, b_path, band=3):
    """How similar one band is in two rasters: mean absolute difference and correlation.
    Only pixels valid in both count, and a 2 px border is dropped because of resampling edges."""
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        x, y = a.read(band).astype(float), b.read(band).astype(float)
        ok = (a.read_masks(band) > 0) & (b.read_masks(band) > 0)
    ok[:2], ok[-2:], ok[:, :2], ok[:, -2:] = False, False, False, False
    return float(np.mean(np.abs(x[ok] - y[ok]))), float(np.corrcoef(x[ok], y[ok])[0, 1]), float(ok.mean())


def rgb(path, window):
    """True-colour crop, stretched 2-98% so it's visible."""
    with rasterio.open(path) as r:
        idx = [r.descriptions.index(b) + 1 for b in ("B4", "B3", "B2")]
        a = np.dstack([r.read(i, window=window).astype(float) for i in idx])
    lo, hi = np.percentile(a, [2, 98])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def checker(a, b, tile=40):
    """Checkerboard of two images. If they're aligned, roads and coastlines continue across tiles."""
    yy, xx = np.indices(a.shape[:2]) // tile
    return np.where(((yy + xx) % 2 == 0)[..., None], a, b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=str(REPO / "data/raw/Jubail_2018_12_10_6band.tif"))
    ap.add_argument("--moving", default=str(REPO / "data/raw/Jubail_2025_12_10_6band.tif"))
    ap.add_argument("--out", default=str(aa.DEFAULT_RESULTS / "real_raster"))
    ap.add_argument("--rasters", default=str(REPO / "data/processed/registration"))
    ap.add_argument("--warps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.rasters, exist_ok=True)
    report = {}

    # 1. find landmarks on the Earth Engine export and choose a transform
    L = sift_landmarks(args.reference, args.moving)
    write_landmarks(os.path.join(args.out, "landmarks_gee_pair.csv"), L)
    inl = L["inl"]
    mov, ref = L["en_mov"][inl], L["en_ref"][inl]
    baseline, fits = aa.real_pair_baseline(mov, ref, seed=args.seed)
    aa.write_csv(os.path.join(args.out, "gee_pair_baseline.csv"), baseline)
    chosen, reason = aa.select_model(baseline)
    # how much does the RANSAC transform actually move pixels? (checked on a 20 px grid)
    h, w = L["shape"]
    ys, xs = np.mgrid[0:h:20, 0:w:20]
    P = np.c_[xs.ravel(), ys.ravel()].astype(float)
    shift_px = np.linalg.norm(aa.apply(L["M_px"], P) - P, axis=1)
    report["gee_pair"] = dict(matches=int(len(inl)), ransac_inliers=int(inl.sum()), selected=chosen, reason=reason,
                              ransac_transform_shift_px_mean=float(shift_px.mean()),
                              ransac_transform_shift_px_max=float(shift_px.max()),
                              baseline={r["model"]: dict(insample_rmse_m=r["insample_rmse_m"], cv_rmse_m=r["cv_rmse_m"])
                                        for r in baseline})
    with open(os.path.join(args.out, "selected_transform_gee_pair.json"), "w") as fh:
        json.dump(dict(selected=chosen, reason=reason, crs="EPSG:32640", direction="2025 (moving) -> 2018 (reference)",
                       matrix=fits[chosen].tolist(),
                       all_models={k: dict(matrix=v.tolist(), **aa.decompose(v)) for k, v in fits.items()}), fh, indent=2)
    print(f"[1] {len(inl)} SIFT matches, {inl.sum()} RANSAC inliers on the Earth Engine pair")
    for r in baseline:
        print(f"    {r['model']:11s} in-sample {r['insample_rmse_m']:6.2f} m  held-out {r['cv_rmse_m']:6.2f} m")
    print(f"    RANSAC transform moves pixels by {shift_px.mean():.3f} px on average (max {shift_px.max():.3f})")
    print(f"    -> selected: {chosen} ({reason})")

    # 2. warp the real 2025 raster by a known amount, then try to undo it
    G3 = aa.to3(L["G"])
    grid = aa.apply(L["G"], P)
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.warps):
        W_px, params = lm.random_affine(rng, L["shape"])
        W = (G3 @ aa.to3(W_px) @ np.linalg.inv(G3))[:2]   # pixel warp -> map warp
        warped = os.path.join(args.rasters, f"knownwarp_{k}.tif")
        aa.warp_raster(args.moving, warped, W, reference_path=args.moving)
        # find the warp again from landmarks (original vs warped) and undo it
        K = sift_landmarks(args.moving, warped)
        M_est = aa.fit_affine(K["en_mov"][K["inl"]], K["en_ref"][K["inl"]])
        truth = np.linalg.inv(aa.to3(W))[:2]
        restored = os.path.join(args.rasters, f"knownwarp_{k}_restored.tif")
        aa.warp_raster(warped, restored, M_est, reference_path=args.moving)
        mad0, r0, _ = band_agreement(args.moving, warped)
        mad1, r1, cover = band_agreement(args.moving, restored)
        rows.append(dict(trial=k, rot_deg=params["rot_deg"], scale=params["scale"], shear=params["shear"],
                         tx_px=params["tx"], ty_px=params["ty"], inliers=int(K["inl"].sum()),
                         misplacement_before_m=aa.grid_rmse(aa.identity(), truth, grid),
                         recovery_error_m=aa.grid_rmse(M_est, truth, grid),
                         b4_mean_abs_diff_before=mad0, b4_corr_before=r0,
                         b4_mean_abs_diff_after=mad1, b4_corr_after=r1, valid_fraction_after=cover))
        print(f"[2] warp {k}: scene misplacement {rows[-1]['misplacement_before_m']:7.1f} m -> recovery error "
              f"{rows[-1]['recovery_error_m']:.2f} m; B4 corr {r0:.3f} -> {r1:.3f}")
    aa.write_csv(os.path.join(args.out, "known_warp_roundtrip.csv"), rows)
    rec = np.array([r["recovery_error_m"] for r in rows])
    report["known_warp_roundtrip"] = dict(trials=len(rows), recovery_error_m_median=float(np.median(rec)),
                                          recovery_error_m_max=float(rec.max()))

    # 3. sanity check: warping with the identity should give back the exact same image
    ident = os.path.join(args.rasters, "identity_check.tif")
    aa.warp_raster(args.moving, ident, aa.identity(), reference_path=args.reference)
    with rasterio.open(args.moving) as a, rasterio.open(ident) as b:
        report["identity_warp_exact"] = bool(np.array_equal(a.read(), b.read()))
        report["identity_warp_metadata_kept"] = bool(a.crs == b.crs and a.transform == b.transform
                                                     and a.descriptions == b.descriptions and a.dtypes == b.dtypes)
    os.remove(ident)
    # then write the actual aligned 2025 image
    aligned = os.path.join(args.rasters, Path(args.moving).stem + "_aligned.tif")
    aa.warp_raster(args.moving, aligned, fits[chosen], reference_path=args.reference)
    report["aligned_output"] = os.path.relpath(aligned, REPO)
    print(f"[3] identity warp reproduces the input exactly: {report['identity_warp_exact']}; "
          f"aligned raster ({chosen}) -> {report['aligned_output']}")
    with open(os.path.join(args.out, "real_raster_check.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    # checkerboard figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.windows import Window
    win = Window(500, 250, 500, 400)
    a18, a25 = rgb(args.reference, win), rgb(args.moving, win)
    aw = rgb(os.path.join(args.rasters, "knownwarp_0.tif"), win)
    ar = rgb(os.path.join(args.rasters, "knownwarp_0_restored.tif"), win)
    panels = [(checker(a18, a25), "2018 | 2025 as delivered"),
              (checker(a25, aw), f"2025 | known warp 0 ({rows[0]['misplacement_before_m']:.0f} m)"),
              (checker(a25, ar), f"2025 | warp undone from landmarks ({rows[0]['recovery_error_m']:.1f} m)")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))
    for ax, (img, title) in zip(axs, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Checkerboards on the 6-band Jubail GeoTIFFs (B4/B3/B2, 40 px tiles): edges should continue across tiles")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_real_raster_checkerboard.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
