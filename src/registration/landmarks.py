"""
Landmark detection for aligning two Sentinel-2 images (Track A, image registration).

What this script does, in plain terms
-------------------------------------
1. Loads two dates of Sentinel-2 L2A bands (B02 blue, B03 green, B04 red, B08 near-infrared)
   straight from the Copernicus Browser zip files.
2. Converts each date to grayscale (as Dr. Panos asked).
3. Optionally restricts where corners may be found, comparing three strategies:
   none; "open_water" (ignore the inside of water bodies only); and "landcover" (the textbook
   rule: ignore water + vegetation + a 20 m buffer, keep only built-up / bare ground).
4. Detects corners with several detectors (Harris, Shi-Tomasi, FAST, ORB, SIFT, AKAZE).
5. Matches corners between the two dates (ratio test + mutual check), then uses RANSAC
   to keep only the matches that agree on one affine transformation. Corners on things
   that were built or demolished between the dates have no true partner and get rejected.
6. Scores every detector x filter combination by warping one image
   with a KNOWN affine transformation and checking how accurately the landmarks recover it.
   The two downloaded dates are already aligned to about 0.1-0.3 px, so the known warp is
   the ground truth.

Outputs (in --out):
  results_synthetic.csv      one row per detector x filter x warp trial
  summary.csv                averages per detector x filter
  landmarks_matched.csv      final landmark pairs (pixel + UTM map coordinates) for the
                             alignment step (Tehsin's module)
  fig_*.png                  figures for the report / meeting

Usage:
  python landmarks.py --data "D:/MyStuff/University/SDP/data/jubail" --out results/registration
The data folder must contain the two Copernicus zips, e.g. S2_L2A_2018-12-10.zip and
S2_L2A_2025-12-10.zip. The earlier date is the reference; the later date is moved.
"""

import argparse
import glob
import os
import zipfile

import cv2
import numpy as np
import rasterio

BANDS = ["B02", "B03", "B04", "B08"]
DETECTORS = ["harris", "shitomasi", "fast", "orb", "sift", "akaze"]
MAX_KP = 3000            # same keypoint budget for every detector, so the comparison is fair
RATIO = 0.8              # Lowe ratio test
RANSAC_PX = 1.5          # a match is an inlier if it lands within 1.5 px (15 m) of the fit
NDWI_WATER = 0.0         # NDWI > 0  -> water
NDVI_VEG = 0.25          # NDVI > 0.25 -> vegetation (mangroves, parks, irrigated land)
BUFFER_PX = 2            # 2 px (20 m) buffer used by the masks


# ----------------------------------------------------------------------------- loading
def load_scene(zip_path):
    """Read the four raw bands from a Copernicus Browser zip. Returns bands, transform, crs."""
    names = zipfile.ZipFile(zip_path).namelist()
    bands, transform, crs = {}, None, None
    for b in BANDS:
        member = [n for n in names if f"_{b}_(Raw)" in n][0]
        with rasterio.open(f"zip://{os.path.abspath(zip_path)}!{member}") as src:
            bands[b] = src.read(1).astype(np.float32)
            transform, crs = src.transform, src.crs
    return bands, transform, crs


def to_gray(bands, valid=None):
    """Luminance grayscale from red/green/blue, stretched to 0-255 using the 2nd-98th percentile.
    Stretching each date on its own evens out brightness differences between the two dates."""
    g = 0.299 * bands["B04"] + 0.587 * bands["B03"] + 0.114 * bands["B02"]
    v = g[valid] if valid is not None else g
    lo, hi = np.percentile(v, [2, 98])
    return np.clip((g - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def indices(bands):
    eps = 1e-6
    ndvi = (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + eps)
    ndwi = (bands["B03"] - bands["B08"]) / (bands["B03"] + bands["B08"] + eps)
    return ndvi, ndwi


FILTERS = ["none", "open_water", "landcover"]


def stable_mask(bands, mode="landcover", valid=None):
    """Where the detector is allowed to look (255) - three strategies are compared:
      none        - everywhere (only the image border is excluded)
      open_water  - drop the INSIDE of water bodies (waves, glint, submerged sandbanks),
                    but keep shorelines, creeks and vegetation
      landcover   - the textbook rule: drop water AND vegetation AND a 20 m buffer around
                    them, keep only built-up / bare ground
    """
    ndvi, ndwi = indices(bands)
    k = np.ones((2 * BUFFER_PX + 1, 2 * BUFFER_PX + 1), np.uint8)
    if mode == "none":
        unstable = np.zeros(ndvi.shape, bool)
    elif mode == "open_water":
        water = (ndwi > NDWI_WATER).astype(np.uint8)
        unstable = cv2.erode(water, k) > 0          # shrink water so its edge (the shoreline) stays in
    elif mode == "landcover":
        unstable = (ndwi > NDWI_WATER) | (ndvi > NDVI_VEG)
        unstable = cv2.dilate(unstable.astype(np.uint8), k) > 0
    else:
        raise ValueError(mode)
    if valid is not None:
        unstable |= cv2.dilate((~valid).astype(np.uint8), k) > 0
    return np.where(unstable, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- detection
_sift = cv2.SIFT_create()


def detect(gray, method, mask=None):
    """Detect up to MAX_KP keypoints and describe them. Pure corner detectors (Harris,
    Shi-Tomasi, FAST) have no descriptor of their own, so SIFT descriptors are computed at
    their locations - that way only the choice of POINTS differs between them."""
    if method in ("harris", "shitomasi"):
        pts = cv2.goodFeaturesToTrack(gray, MAX_KP, qualityLevel=0.01, minDistance=5,
                                      mask=mask, blockSize=5,
                                      useHarrisDetector=(method == "harris"), k=0.04)
        kps = [cv2.KeyPoint(float(x), float(y), 12) for x, y in (pts.reshape(-1, 2) if pts is not None else [])]
        kps, des = _sift.compute(gray, kps)
    elif method == "fast":
        kps = cv2.FastFeatureDetector_create(threshold=15).detect(gray, mask)
        kps = sorted(kps, key=lambda k: -k.response)[:MAX_KP]
        for k in kps:
            k.size = 12
        kps, des = _sift.compute(gray, kps)
    elif method == "orb":
        kps, des = cv2.ORB_create(nfeatures=MAX_KP).detectAndCompute(gray, mask)
    elif method == "sift":
        kps, des = cv2.SIFT_create(nfeatures=MAX_KP).detectAndCompute(gray, mask)
    elif method == "akaze":
        # OpenCV 5 moved AKAZE to the contrib module; OpenCV 4.x (tested) has it in the main module
        akaze_create = getattr(cv2, "AKAZE_create", None) or getattr(getattr(cv2, "xfeatures2d", None), "AKAZE_create", None)
        if akaze_create is None:
            raise RuntimeError("AKAZE not available - install OpenCV 4.x: python -m pip install -r requirements.txt")
        kps, des = akaze_create(threshold=0.0005).detectAndCompute(gray, mask)
        order = np.argsort([-k.response for k in kps])[:MAX_KP]
        kps = [kps[i] for i in order]
        des = des[order] if des is not None else None
    else:
        raise ValueError(method)
    return list(kps or []), des


def match(kpA, desA, kpB, desB, binary):
    """Ratio test + mutual (both-directions) check. Returns matched point arrays."""
    if desA is None or desB is None or len(kpA) < 3 or len(kpB) < 3:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2)
    fwd = {}
    for pair in bf.knnMatch(desA, desB, k=2):
        if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
            fwd[pair[0].queryIdx] = pair[0].trainIdx
    back = {m.queryIdx: m.trainIdx for m in bf.match(desB, desA)}
    keep = [(a, b) for a, b in fwd.items() if back.get(b) == a]
    pA = np.float32([kpA[a].pt for a, _ in keep]).reshape(-1, 2)
    pB = np.float32([kpB[b].pt for _, b in keep]).reshape(-1, 2)
    return pA, pB


def fit_affine(pMoving, pRef):
    """RANSAC affine that maps moving-image points onto reference-image points."""
    if len(pMoving) < 3:
        return None, np.zeros(len(pMoving), bool)
    M, inl = cv2.estimateAffine2D(pMoving, pRef, method=cv2.RANSAC,
                                  ransacReprojThreshold=RANSAC_PX, maxIters=5000, confidence=0.999)
    return M, (inl.ravel().astype(bool) if inl is not None else np.zeros(len(pMoving), bool))


def transform_rmse(M_est, M_true, shape, step=20):
    """Average misplacement (px) over a grid covering the whole image."""
    if M_est is None:
        return np.nan
    h, w = shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 0)
    d = (M_est @ P) - (M_true @ P)
    return float(np.sqrt((d ** 2).sum(0).mean()))


# ----------------------------------------------------------------------------- pipeline
def run_pair(grayRef, maskRef, grayMov, maskMov, method):
    kA, dA = detect(grayRef, method, maskRef)
    kB, dB = detect(grayMov, method, maskMov)
    pRef, pMov = match(kA, dA, kB, dB, binary=(method in ("orb", "akaze")))
    M, inl = fit_affine(pMov, pRef)
    return dict(kp_ref=len(kA), kp_mov=len(kB), matches=len(pRef), inliers=int(inl.sum()),
                M=M, pRef=pRef, pMov=pMov, inl=inl, kA=kA, kB=kB)


def random_affine(rng, shape):
    """A realistic misalignment: up to 15 px (150 m) shift, 2 deg rotation, 2 % scale, 1 % shear,
    applied about the image centre."""
    h, w = shape
    ang = np.deg2rad(rng.uniform(-2, 2))
    s = rng.uniform(0.98, 1.02)
    sh = rng.uniform(-0.01, 0.01)
    tx, ty = rng.uniform(-15, 15, 2)
    A = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([[1, sh], [0, 1]])
    c = np.array([w / 2, h / 2])
    t = c - A @ c + np.array([tx, ty])
    return np.hstack([A, t[:, None]]).astype(np.float64), dict(rot_deg=np.rad2deg(ang), scale=s, shear=sh, tx=tx, ty=ty)


def warp_bands(bands, W, shape):
    h, w = shape
    out = {b: cv2.warpAffine(v, W, (w, h), flags=cv2.INTER_LINEAR, borderValue=0) for b, v in bands.items()}
    valid = cv2.warpAffine(np.ones(shape, np.uint8), W, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return out, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the two Copernicus zips")
    ap.add_argument("--out", default="results/registration")
    ap.add_argument("--trials", type=int, default=20, help="number of known-warp test pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(args.data, "*.zip")))
    assert len(zips) >= 2, "need two zips in --data"
    ref_zip, mov_zip = zips[0], zips[-1]
    ref_name, mov_name = [os.path.basename(z)[:-4] for z in (ref_zip, mov_zip)]
    print(f"reference: {ref_name}   moving: {mov_name}")

    bRef, transform, crs = load_scene(ref_zip)
    bMov, _, _ = load_scene(mov_zip)
    shape = bRef["B04"].shape
    gRef, gMov = to_gray(bRef), to_gray(bMov)
    mRef = {f: stable_mask(bRef, f) for f in FILTERS}
    mMov = {f: stable_mask(bMov, f) for f in FILTERS}
    for f in FILTERS:
        print(f"search area with filter '{f}': {mRef[f].mean() / 255:.0%} of {ref_name}, "
              f"{mMov[f].mean() / 255:.0%} of {mov_name}")

    import csv
    I = np.hstack([np.eye(2), np.zeros((2, 1))])

    # 1) The real pair as downloaded: should come out close to "no movement".
    print("\n[real pair] estimated transform (identity = already aligned)")
    real = {}
    for method in DETECTORS:
        for f in FILTERS:
            r = run_pair(gRef, mRef[f], gMov, mMov[f], method)
            real[(method, f)] = r
            print(f"  {method:9s} {f:10s} matches={r['matches']:5d} inliers={r['inliers']:5d} "
                  f"offset from identity={transform_rmse(r['M'], I, shape):.2f} px")

    # 2) Known-warp trials: reference vs (moving date warped by a known affine).
    rows = []
    rng = np.random.default_rng(args.seed)
    for t in range(args.trials):
        W, params = random_affine(rng, shape)
        bW, validW = warp_bands(bMov, W, shape)
        gW = to_gray(bW, validW)
        mW = {f: stable_mask(bW, f, validW) for f in FILTERS}
        M_true = cv2.invertAffineTransform(W)   # the fit maps warped -> reference, i.e. undoes W
        for method in DETECTORS:
            for f in FILTERS:
                r = run_pair(gRef, mRef[f], gW, mW[f], method)
                err = transform_rmse(r["M"], M_true, shape)
                rows.append(dict(trial=t, detector=method, filter=f,
                                 kp_ref=r["kp_ref"], kp_mov=r["kp_mov"], matches=r["matches"],
                                 inliers=r["inliers"],
                                 inlier_ratio=r["inliers"] / r["matches"] if r["matches"] else 0.0,
                                 rmse_px=err, success=bool(err < 1.0), **params))
        print(f"  trial {t + 1}/{args.trials} done")

    with open(os.path.join(args.out, "results_synthetic.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    summary = []
    for method in DETECTORS:
        for f in FILTERS:
            rs = [r for r in rows if r["detector"] == method and r["filter"] == f]
            errs = np.array([r["rmse_px"] for r in rs], float)
            summary.append(dict(
                detector=method, filter=f,
                keypoints=float(np.mean([r["kp_ref"] for r in rs])),
                matches=float(np.mean([r["matches"] for r in rs])),
                inliers=float(np.mean([r["inliers"] for r in rs])),
                inlier_ratio=float(np.mean([r["inlier_ratio"] for r in rs])),
                median_err_px=float(np.nanmedian(errs)) if np.isfinite(errs).any() else float("nan"),
                worst_err_px=float(np.nanmax(errs)) if np.isfinite(errs).any() else float("nan"),
                success_rate=float(np.mean([r["success"] for r in rs]))))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        for s_ in summary:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s_.items()})
    print(f"\nsummary over {args.trials} known-warp trials:")
    for s_ in summary:
        print(f"  {s_['detector']:9s} {s_['filter']:10s} inliers={s_['inliers']:6.1f} "
              f"inlier_ratio={s_['inlier_ratio']:.2f} median_err={s_['median_err_px']:.3f}px "
              f"worst={s_['worst_err_px']:.3f}px success={s_['success_rate']:.0%}")

    # 3) Export landmark pairs from the most reliable combination on the real pair.
    best = min(summary, key=lambda s_: (-s_["success_rate"], s_["worst_err_px"]))
    r = real[(best["detector"], best["filter"])]
    with open(os.path.join(args.out, "landmarks_matched.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", f"x_{ref_name}", f"y_{ref_name}", f"x_{mov_name}", f"y_{mov_name}",
                     f"easting_{ref_name}", f"northing_{ref_name}", "ransac_inlier"])
        for i, ((xr, yr), (xm, ym), ok) in enumerate(zip(r["pRef"], r["pMov"], r["inl"])):
            e, n = transform * (xr + 0.5, yr + 0.5)
            wr.writerow([i, round(float(xr), 2), round(float(yr), 2), round(float(xm), 2), round(float(ym), 2),
                         f"{e:.1f}", f"{n:.1f}", int(ok)])
    print(f"\nbest: {best['detector']} + {best['filter']} -> exported {int(r['inl'].sum())} inlier landmark pairs "
          f"to landmarks_matched.csv [{crs}]")
    return dict(bRef=bRef, bMov=bMov, gRef=gRef, gMov=gMov, mRef=mRef, mMov=mMov, real=real,
                summary=summary, rows=rows, best=(best["detector"], best["filter"]),
                names=(ref_name, mov_name), transform=transform, args=args)


if __name__ == "__main__":
    main()
