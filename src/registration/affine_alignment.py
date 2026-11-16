"""
Track A, part 2: aligning the 2025 image onto the 2018 image and measuring the
registration error. (Tehsin Shaik)

Khaled's landmarks.py finds matching points ("landmarks") in the two images and
saves them to results/registration/landmarks_for_tehsin.csv. This script takes
those matches, fits a transform that moves 2025 onto 2018, and checks how far
off the points still are afterwards.

A few things to know before reading the code:

- 2018 is the reference image and 2025 is the moving image. Every transform
  here takes a 2025 map position (easting, northing) and gives the matching
  2018 map position.
- Only the RANSAC inliers are used (ransac_inlier == 1) and only the EPSG:32640
  easting/northing columns, in metres. The pixel columns depend on how each of
  us cropped the scene, so they don't carry over between exports.
- Each landmark pair gives 2 equations (one for x, one for y), so:
      translation   2 unknowns   needs 1 landmark
      similarity    4 unknowns   needs 2 landmarks   (shift, rotation, one scale)
      affine        6 unknowns   needs 3 landmarks   (also shear, x/y scale), not on a line
  That's why the "2 landmarks" case has to be a similarity and the
  "3 landmarks" case can be a full affine.

The experiment has two scenarios:
  real        the 2018/2025 pair as downloaded (it turns out to be aligned already)
  known_warp  the 2025 landmarks are moved by a random affine that we know
              (Khaled's random_affine), so we can check whether it gets undone

Outputs (in --out):
  alignment_baseline.csv   real pair: no correction vs each model fitted on all landmarks
  alignment_trials.csv     one row per trial and method (big, not committed)
  alignment_summary.csv    summary per scenario and method
  alignment_paired.csv     3-landmark affine vs 2-landmark similarity, same trials
  selected_transform.json  the transform to use on the real 2025 image
  fig_2_vs_3_landmarks.png, fig_alignment_example.png

How to run (from the repo root):
  python src/registration/affine_alignment.py experiment
  python src/registration/affine_alignment.py warp --moving 2025.tif --reference 2018.tif \
      --out 2025_aligned.tif [--transform results/registration/selected_transform.json]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RESULTS = REPO / "results" / "registration"

PIXEL_M = 10.0               # Sentinel-2 pixel size, just to also report errors in pixels
SCENE_SHAPE = (1096, 1836)   # rows, cols of Khaled's Jubail scene (see src/registration/README.md)
MIN_SEPARATION_M = 1000.0    # fit landmarks closer than 1 km give a shaky rotation/scale, so skip them
MIN_ANGLE_DEG = 15.0         # a triangle with an angle under 15 deg is basically a line
MIN_GAIN_M = 1.0             # only apply a correction if it beats "do nothing" by at least 1 m (0.1 px)
METHODS = ["none", "similarity_2pt", "similarity_3pt", "affine_3pt"]


# ---------------------------------------------------------------- reading the landmarks

def load_landmarks(path, inliers_only=True):
    """Read Khaled's CSV. Returns ids, 2025 points and 2018 points (metres, EPSG:32640)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if inliers_only:
        rows = [r for r in rows if int(r["ransac_inlier"]) == 1]
    ids = np.array([int(r["id"]) for r in rows])
    mov = np.array([[float(r["easting_2025_m"]), float(r["northing_2025_m"])] for r in rows])
    ref = np.array([[float(r["easting_2018_m"]), float(r["northing_2018_m"])] for r in rows])
    return ids, mov, ref


def load_pixel_landmarks(path):
    """2018 pixel and map positions. Only used to work out the scene's pixel -> map transform."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    px = np.array([[float(r["x_px_2018"]), float(r["y_px_2018"])] for r in rows])
    en = np.array([[float(r["easting_2018_m"]), float(r["northing_2018_m"])] for r in rows])
    return px, en


# ---------------------------------------------------------------- transforms
# A transform is stored as a 2x3 matrix [[a, b, tx], [c, d, ty]]:
#   x' = a*x + b*y + tx
#   y' = c*x + d*y + ty

def to3(M):
    """2x3 -> 3x3 (adds the [0 0 1] row) so transforms can be chained with @."""
    return np.vstack([M, [0.0, 0.0, 1.0]])


def apply(M, P):
    """Apply a 2x3 transform to an (n, 2) array of points."""
    return P @ M[:, :2].T + M[:, 2]


def identity():
    return np.hstack([np.eye(2), np.zeros((2, 1))])


def fit_translation(src, dst):
    # best shift is just the average difference
    M = identity()
    M[:, 2] = (dst - src).mean(0)
    return M


def fit_similarity(src, dst):
    """Shift + rotation + one scale, least squares. With exactly 2 points it's exact.

    Trick: treat each point as a complex number z = E + iN. Then a similarity is
    just w = a*z + b, where |a| is the scale and angle(a) is the rotation.
    The mean is subtracted first because UTM numbers are huge (~2.7 million) and
    that loses precision otherwise.
    """
    if len(src) < 2:
        raise ValueError("a similarity transform needs at least 2 landmark pairs")
    zs = src[:, 0] + 1j * src[:, 1]
    zd = dst[:, 0] + 1j * dst[:, 1]
    zs0, zd0 = zs - zs.mean(), zd - zd.mean()
    denom = np.sum(np.abs(zs0) ** 2)
    if denom == 0:
        raise ValueError("landmarks coincide; rotation/scale undefined")
    a = np.sum(zd0 * np.conj(zs0)) / denom
    b = zd.mean() - a * zs.mean()
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def fit_affine(src, dst):
    """Full affine (6 numbers), least squares. With exactly 3 points not on a line it's exact."""
    if len(src) < 3:
        raise ValueError("a full affine transform needs at least 3 landmark pairs")
    # centre the points first, same precision reason as above
    ms, md = src.mean(0), dst.mean(0)
    X = np.hstack([src - ms, np.ones((len(src), 1))])
    if np.linalg.matrix_rank(X[:, :2]) < 2:
        # all points on one line: the stretch across the line can't be worked out
        raise ValueError("landmarks are collinear; a full affine is not determined")
    P, *_ = np.linalg.lstsq(X, dst - md, rcond=None)
    A = P[:2].T
    t = P[2] + md - A @ ms
    return np.hstack([A, t[:, None]])


FITTERS = {"translation": fit_translation, "similarity": fit_similarity, "affine": fit_affine}


def decompose(M):
    """Split a transform into readable parts: shift (m), rotation (deg), x/y scale and shear."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    sx = float(np.hypot(a, c))
    rot = float(np.degrees(np.arctan2(c, a)))
    m = (a * b + c * d) / sx
    sy = float((a * d - b * c) / sx)
    return dict(tx_m=float(M[0, 2]), ty_m=float(M[1, 2]), rot_deg=rot, scale_x=sx, scale_y=sy,
                shear=float(m / sy) if sy != 0 else float("nan"))


def displacement_at(M, P):
    """How far (m) the transform moves each point, i.e. how big the correction is."""
    return np.linalg.norm(apply(M, P) - P, axis=1)


# ---------------------------------------------------------------- is this set of landmarks usable?

def separation(P):
    return float(np.linalg.norm(P[0] - P[1]))


def min_angle_deg(P):
    """Smallest angle of the triangle made by 3 points. 0 means they're on a line."""
    a = np.linalg.norm(P[1] - P[2])
    b = np.linalg.norm(P[0] - P[2])
    c = np.linalg.norm(P[0] - P[1])
    if min(a, b, c) == 0:
        return 0.0
    # law of cosines for two angles, the third is whatever is left of 180
    cosA = np.clip((b * b + c * c - a * a) / (2 * b * c), -1, 1)
    cosB = np.clip((a * a + c * c - b * b) / (2 * a * c), -1, 1)
    A, B = np.degrees(np.arccos(cosA)), np.degrees(np.arccos(cosB))
    return float(min(A, B, 180.0 - A - B))


def triplet_ok(P, min_sep=MIN_SEPARATION_M, min_angle=MIN_ANGLE_DEG):
    """3 points are usable if they're spread out and not close to a straight line.
    Nearly-straight triplets technically fit, but a few metres of noise turns
    into hundreds of metres of error away from the points."""
    sides = [np.linalg.norm(P[i] - P[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    return min(sides) >= min_sep and min_angle_deg(P) >= min_angle


# ---------------------------------------------------------------- measuring error

def errors(M, src, dst):
    """Registration error per landmark (m): distance between the moved 2025 point and its 2018 partner."""
    return np.linalg.norm(apply(M, src) - dst, axis=1)


def stats(e, prefix=""):
    e = np.asarray(e, float)
    return {f"{prefix}n": int(e.size),
            f"{prefix}mean_m": float(e.mean()),
            f"{prefix}median_m": float(np.median(e)),
            f"{prefix}rmse_m": float(np.sqrt(np.mean(e ** 2))),
            f"{prefix}p95_m": float(np.percentile(e, 95)),
            f"{prefix}max_m": float(e.max())}


def scene_geotransform(px, en):
    """Pixel -> map transform of Khaled's 2018 crop, fitted from the landmarks.
    landmarks.py writes pixel centres, hence the +0.5. The fit gives ~10.002 x 9.998 m
    pixels, not exactly 10 m, which is normal for a Copernicus Browser export."""
    return fit_affine(px + 0.5, en)


def scene_grid(G, shape=SCENE_SHAPE, step=20):
    """Map positions of a grid of points every 20 px over the whole scene."""
    h, w = shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    return apply(G, np.c_[xs.ravel() + 0.5, ys.ravel() + 0.5])


def grid_rmse(M_est, M_true, grid):
    """How far apart (m) the estimated and true transforms put points across the whole scene.
    Same idea as Khaled's transform_rmse, but in metres instead of pixels."""
    d = apply(M_est, grid) - apply(M_true, grid)
    return float(np.sqrt((d ** 2).sum(1).mean()))


def known_warp_map(rng, G, shape=SCENE_SHAPE):
    """Get one of Khaled's random warps (made in pixels) and convert it to map coordinates:
    W_map = G * W_px * G^-1. His function is reused as is, so the same warps are tested."""
    sys.path.insert(0, str(HERE))
    from landmarks import random_affine  # noqa: E402
    W_px, params = random_affine(rng, shape)
    W_map = (to3(G) @ to3(W_px) @ np.linalg.inv(to3(G)))[:2]
    return W_map, params


# ---------------------------------------------------------------- experiments

def kfold_cv(mov, ref, model, k=10, seed=0):
    """Fit on 9/10 of the landmarks, measure on the other 1/10, repeat 10 times.
    Error on the points you fitted with is always too optimistic, so this gives
    an error for every landmark that wasn't used to fit."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(mov)), k)
    e = np.empty(len(mov))
    for f in folds:
        train = np.setdiff1d(np.arange(len(mov)), f)
        M = identity() if model == "none" else FITTERS[model](mov[train], ref[train])
        e[f] = errors(M, mov[f], ref[f])
    return e


def real_pair_baseline(mov, ref, seed=0):
    """Real pair, all landmarks: how good is 'no correction' compared to each model?"""
    rows, fits = [], {"none": identity()}
    for model in ["none", "translation", "similarity", "affine"]:
        M = fits.get(model) if model == "none" else FITTERS[model](mov, ref)
        fits[model] = M
        row = dict(model=model, n_landmarks=len(mov))
        row.update(stats(errors(M, mov, ref), "insample_"))
        row.update(stats(kfold_cv(mov, ref, model, seed=seed), "cv_"))
        row.update(decompose(M))
        row["max_correction_m"] = float(displacement_at(M, mov).max())
        rows.append(row)
    return rows, fits


def select_model(baseline, min_gain=MIN_GAIN_M):
    """Pick the model with the lowest held-out error, but only if it actually helps by
    at least min_gain. Otherwise leave the image alone ('none')."""
    none = next(r for r in baseline if r["model"] == "none")
    best = min(baseline, key=lambda r: r["cv_rmse_m"])
    gain = none["cv_rmse_m"] - best["cv_rmse_m"]
    chosen = best["model"] if gain >= min_gain else "none"
    reason = (f"best held-out model '{best['model']}' improves RMSE by {gain:.3f} m "
              f"({'>=' if gain >= min_gain else '<'} {min_gain} m threshold)")
    return chosen, reason


def draw_triplet(rng, P):
    return rng.choice(len(P), 3, replace=False)


def fit_methods(src, dst, idx):
    """Fit every method on just the sampled landmarks idx.
    The 2-landmark method uses the first two of the three."""
    out = {"none": identity()}
    try:
        out["similarity_2pt"] = fit_similarity(src[idx[:2]], dst[idx[:2]])
    except ValueError:
        out["similarity_2pt"] = None
    try:
        out["similarity_3pt"] = fit_similarity(src[idx], dst[idx])
    except ValueError:
        out["similarity_3pt"] = None
    try:
        out["affine_3pt"] = fit_affine(src[idx], dst[idx])
    except ValueError:
        out["affine_3pt"] = None
    return out


def run_trials(ids, mov, ref, G, trials=2000, seed=42):
    """The main 2 vs 3 landmark experiment.

    Each trial picks 3 random landmarks. The 2-landmark method uses the first two,
    so both methods get the same draw and are scored on the same held-out points
    (every landmark not picked). That makes it a fair side-by-side comparison.
    """
    rng = np.random.default_rng(seed)
    grid = scene_grid(G)
    rows = []
    for scenario in ["real", "known_warp"]:
        for t in range(trials):
            if scenario == "known_warp":
                # move the 2025 points by a known warp; the right answer is its inverse
                W, wparams = known_warp_map(rng, G)
                src, M_true = apply(W, mov), np.linalg.inv(to3(W))[:2]
            else:
                src, M_true, wparams = mov, None, {}
            idx = draw_triplet(rng, mov)
            P = mov[idx]                      # judge the triangle shape on the un-warped points
            ok = triplet_ok(P)
            hold = np.setdiff1d(np.arange(len(mov)), idx)
            for method, M in fit_methods(src, ref, idx).items():
                if method == "none":
                    fit_ids = ""
                else:
                    used = idx[:2] if method == "similarity_2pt" else idx
                    fit_ids = " ".join(str(ids[i]) for i in used)
                row = dict(scenario=scenario, trial=t, method=method,
                           n_fit={"none": 0, "similarity_2pt": 2}.get(method, 3),
                           fit_ids=fit_ids,
                           pair_sep_m=separation(P),
                           tri_min_side_m=min(np.linalg.norm(P[i] - P[j]) for i, j in ((0, 1), (0, 2), (1, 2))),
                           tri_min_angle_deg=min_angle_deg(P), accepted=bool(ok))
                if M is None:
                    # fit failed (e.g. points on a line)
                    row.update(dict(val_n=len(hold), val_rmse_m=np.nan))
                else:
                    row.update(stats(errors(M, src[hold], ref[hold]), "val_"))
                    row.update(decompose(M))
                    row["grid_err_vs_truth_m"] = grid_rmse(M, M_true, grid) if M_true is not None else np.nan
                row.update({f"warp_{k}": v for k, v in wparams.items()})
                rows.append(row)
    return rows


def summarise(rows):
    """One summary row per scenario, method and accepted/rejected triplets."""
    out = []
    for scenario in ["real", "known_warp"]:
        for subset in ["accepted", "rejected"]:
            for method in METHODS:
                rs = [r for r in rows if r["scenario"] == scenario and r["method"] == method
                      and r["accepted"] == (subset == "accepted")]
                v = np.array([r["val_rmse_m"] for r in rs], float)
                failed = int(np.isnan(v).sum())
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                g = np.array([r.get("grid_err_vs_truth_m", np.nan) for r in rs], float)
                g = g[np.isfinite(g)]
                worst = max(r["val_max_m"] for r in rs if np.isfinite(r["val_rmse_m"]))
                s = dict(scenario=scenario, triplets=subset, method=method, trials=len(rs), failed_fits=failed,
                         val_rmse_median_m=np.median(v), val_rmse_mean_m=v.mean(),
                         val_rmse_iqr_m=np.percentile(v, 75) - np.percentile(v, 25),
                         val_rmse_p90_m=np.percentile(v, 90), val_rmse_max_m=v.max(),
                         val_median_err_median_m=np.median([r["val_median_m"] for r in rs if np.isfinite(r["val_rmse_m"])]),
                         worst_single_landmark_m=worst,
                         val_rmse_median_px=np.median(v) / PIXEL_M, val_rmse_p90_px=np.percentile(v, 90) / PIXEL_M)
                if len(g):
                    s.update(grid_err_median_m=np.median(g), grid_err_p90_m=np.percentile(g, 90),
                             grid_err_max_m=g.max())
                out.append(s)
    return out


def paired(rows, a="affine_3pt", b="similarity_2pt", n_boot=5000, seed=0):
    """Difference a - b in held-out RMSE on the same trials (negative = a is better),
    with a bootstrap 95% confidence interval on the median."""
    rng = np.random.default_rng(seed)
    out = []
    for scenario in ["real", "known_warp"]:
        def key(m, scenario=scenario):
            return {r["trial"]: r["val_rmse_m"] for r in rows
                    if r["scenario"] == scenario and r["method"] == m and r["accepted"]}
        ra, rb = key(a), key(b)
        common = sorted(set(ra) & set(rb))
        d = np.array([ra[t] - rb[t] for t in common])
        d = d[np.isfinite(d)]
        boots = np.median(rng.choice(d, (n_boot, len(d))), axis=1)
        out.append(dict(scenario=scenario, comparison=f"{a} minus {b}", trials=len(d),
                        median_diff_m=np.median(d), ci95_low_m=np.percentile(boots, 2.5),
                        ci95_high_m=np.percentile(boots, 97.5), mean_diff_m=d.mean(),
                        frac_trials_a_better=float(np.mean(d < 0))))
    return out


# ---------------------------------------------------------------- saving results

def write_csv(path, rows):
    # rows don't all have the same keys (e.g. warp_* only for known_warp), so collect them all
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def make_figures(out, rows, ids, mov, ref, G, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"none": "#9a9893", "similarity_2pt": "#2a78d6", "similarity_3pt": "#7d5bc6", "affine_3pt": "#eb6834"}
    labels = {"none": "No correction", "similarity_2pt": "2 landmarks\n(similarity)",
              "similarity_3pt": "3 landmarks\n(similarity)", "affine_3pt": "3 landmarks\n(full affine)"}

    # Figure 1: error per method for both scenarios, plus why we reject flat triangles
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, scen, title in [(axs[0], "real", "Real 2018/2025 pair (already aligned)"),
                            (axs[1], "known_warp", "Known synthetic misalignment")]:
        data, cols, labs = [], [], []
        for m in METHODS:
            v = np.array([r["val_rmse_m"] for r in rows if r["scenario"] == scen and r["method"] == m and r["accepted"]])
            data.append(v[np.isfinite(v)])
            cols.append(colors[m])
            labs.append(labels[m])
        bp = ax.boxplot(data, patch_artist=True, showfliers=True, whis=(5, 95),
                        flierprops=dict(markersize=2, alpha=.4))
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c)
            p.set_alpha(.75)
        ax.set_xticks(range(1, len(labs) + 1), labs)
        ax.set_yscale("log")
        ax.set_ylabel("held-out RMSE per trial (m, log scale)")
        ax.axhline(PIXEL_M, color="k", lw=.8, ls="--")
        ax.text(0.55, PIXEL_M, "1 px (10 m)", va="bottom", ha="left", fontsize=8)
        ax.set_title(title)
    ax = axs[2]
    for m in ["similarity_2pt", "affine_3pt"]:
        rs = [r for r in rows if r["scenario"] == "known_warp" and r["method"] == m and np.isfinite(r["val_rmse_m"])]
        ax.scatter([r["tri_min_angle_deg"] for r in rs], [r["val_rmse_m"] for r in rs], s=4, alpha=.35,
                   color=colors[m], label=labels[m].replace("\n", " "))
    ax.axvline(MIN_ANGLE_DEG, color="k", lw=.8, ls="--")
    ax.text(MIN_ANGLE_DEG + 1, 0.95, f"rejected below {MIN_ANGLE_DEG:.0f}°", fontsize=8,
            transform=ax.get_xaxis_transform(), va="top")
    ax.set_yscale("log")
    ax.set_xlabel("smallest triangle angle of the 3 fit landmarks (deg)")
    ax.set_ylabel("held-out RMSE (m, log scale)")
    ax.set_title("Why near-collinear triplets are rejected")
    ax.legend(fontsize=8, markerscale=3)
    fig.suptitle("Registration error on landmarks NOT used for fitting: 2 vs 3 landmarks", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_2_vs_3_landmarks.png"), dpi=150)
    plt.close(fig)

    # Figure 2: one example trial, arrows show the leftover error at each held-out landmark
    rng = np.random.default_rng(seed + 1)
    W, _ = known_warp_map(rng, G)
    src = apply(W, mov)
    while True:
        # pick a nicely spread triangle so the example is easy to read
        idx = draw_triplet(rng, mov)
        if triplet_ok(mov[idx], min_sep=4000, min_angle=30):
            break
    hold = np.setdiff1d(np.arange(len(mov)), idx)
    M2 = fit_similarity(src[idx[:2]], ref[idx[:2]])
    M3 = fit_affine(src[idx], ref[idx])
    panels = [("Real pair, no correction", mov, identity(), 100, None),
              ("Known warp, before correction", src, identity(), 1, None),
              ("Known warp, after 2-landmark similarity", src, M2, 100, idx[:2]),
              ("Known warp, after 3-landmark affine", src, M3, 100, idx)]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (title, s, M, scale, fit_idx) in zip(axs.ravel(), panels):
        pts = apply(M, s)
        d = pts - ref
        e = np.linalg.norm(d[hold], axis=1)
        # arrows are scaled up (x100) otherwise you can't see a few metres on a 18 km scene
        q = ax.quiver(ref[hold, 0] / 1000, ref[hold, 1] / 1000, d[hold, 0] * scale / 1000, d[hold, 1] * scale / 1000,
                      e, angles="xy", scale_units="xy", scale=1, cmap="viridis", width=.003)
        if fit_idx is not None:
            ax.scatter(ref[fit_idx, 0] / 1000, ref[fit_idx, 1] / 1000, marker="*", s=220, c="#e34948",
                       edgecolor="k", zorder=5, label="fit landmarks")
            ax.legend(loc="lower right", fontsize=8)
        fig.colorbar(q, ax=ax, label="error at held-out landmark (m)")
        ax.set_title(f"{title}\nheld-out RMSE {np.sqrt(np.mean(e ** 2)):.1f} m  (arrows x{scale})", fontsize=10)
        ax.set_xlabel("easting (km, EPSG:32640)")
        ax.set_ylabel("northing (km)")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_alignment_example.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- warping an actual image

def warp_raster(moving_path, out_path, M, reference_path=None, resampling="bilinear"):
    """Apply transform M (2025 map -> 2018 map) to every band of a GeoTIFF.

    Instead of moving pixels by hand, the image's georeferencing is changed to
    M @ (pixel -> map) and let GDAL resample it once onto the 2018 grid (or the
    image's own grid if no reference is given). Band names, dtype, nodata and
    tags are kept, and the matrix used is saved in the REGISTRATION_TRANSFORM tag.
    """
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject

    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(moving_path) as src:
        corrected = Affine(*np.asarray(M, float).ravel()[:6]) @ src.transform
        if reference_path:
            with rasterio.open(reference_path) as ref:
                if ref.crs != src.crs:
                    raise ValueError(f"CRS mismatch: moving {src.crs} vs reference {ref.crs}; reproject first")
                dst_transform, width, height = ref.transform, ref.width, ref.height
        else:
            dst_transform, width, height = src.transform, src.width, src.height
        # Our Earth Engine exports have no nodata value, and 0 can be a real value
        # (e.g. 2018 B08 over water), so instead of using 0 for "outside the image"
        # a separate mask band is written.
        use_mask = src.nodata is None
        fill = 0 if use_mask else src.nodata
        profile = src.profile.copy()
        profile.update(transform=dst_transform, width=width, height=height)
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                out = np.full((height, width), fill, dtype=src.dtypes[b - 1])
                reproject(source=src.read(b), destination=out, src_transform=corrected, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=src.crs, src_nodata=src.nodata,
                          dst_nodata=fill, resampling=getattr(Resampling, resampling))
                dst.write(out, b)
                if src.descriptions[b - 1]:
                    dst.set_band_description(b, src.descriptions[b - 1])
            if use_mask:
                valid = np.zeros((height, width), np.uint8)
                reproject(source=np.full((src.height, src.width), 255, np.uint8), destination=valid,
                          src_transform=corrected, src_crs=src.crs, dst_transform=dst_transform,
                          dst_crs=src.crs, dst_nodata=0, resampling=Resampling.nearest)
                dst.write_mask(valid)
            dst.update_tags(**src.tags())
            dst.update_tags(REGISTRATION_TRANSFORM=json.dumps(np.asarray(M).tolist()),
                            REGISTRATION_SOURCE=os.path.basename(moving_path))
    return out_path


# ---------------------------------------------------------------- command line

def cmd_experiment(args):
    os.makedirs(args.out, exist_ok=True)
    ids, mov, ref = load_landmarks(args.landmarks)
    px, en = load_pixel_landmarks(args.landmarks)
    G = scene_geotransform(px, en)
    print(f"{len(ids)} RANSAC-inlier landmark pairs (EPSG:32640 metres)")

    # step 1: real pair with all landmarks, and decide what to apply to the real image
    baseline, fits = real_pair_baseline(mov, ref, seed=args.seed)
    write_csv(os.path.join(args.out, "alignment_baseline.csv"), baseline)
    print("\n[real pair] all inliers, in-sample vs 10-fold held-out RMSE")
    for r in baseline:
        print(f"  {r['model']:11s} in-sample {r['insample_rmse_m']:6.2f} m   held-out {r['cv_rmse_m']:6.2f} m "
              f"({r['cv_rmse_m'] / PIXEL_M:.2f} px)   max correction {r['max_correction_m']:5.2f} m")
    chosen, reason = select_model(baseline, args.min_gain)
    print(f"  -> selected for the real image: {chosen} ({reason})")
    with open(os.path.join(args.out, "selected_transform.json"), "w") as fh:
        json.dump(dict(selected=chosen, reason=reason, crs="EPSG:32640", direction="2025 (moving) -> 2018 (reference)",
                       matrix=fits[chosen].tolist(),
                       all_models={k: dict(matrix=v.tolist(), **decompose(v)) for k, v in fits.items()}), fh, indent=2)

    # step 2: 2 vs 3 landmark trials
    rows = run_trials(ids, mov, ref, G, trials=args.trials, seed=args.seed)
    write_csv(os.path.join(args.out, "alignment_trials.csv"), rows)
    summary = summarise(rows)
    write_csv(os.path.join(args.out, "alignment_summary.csv"), summary)
    pr = paired(rows, seed=args.seed)
    write_csv(os.path.join(args.out, "alignment_paired.csv"), pr)
    print(f"\n{args.trials} trials per scenario (accepted triplets only), held-out RMSE per trial:")
    for s in summary:
        if s["triplets"] == "accepted":
            g = f"  scene error vs truth median {s['grid_err_median_m']:.2f} m" if "grid_err_median_m" in s else ""
            print(f"  {s['scenario']:10s} {s['method']:15s} median {s['val_rmse_median_m']:8.2f} m  "
                  f"p90 {s['val_rmse_p90_m']:8.2f} m  max {s['val_rmse_max_m']:8.2f} m{g}")
    for p in pr:
        print(f"  {p['scenario']:10s} {p['comparison']}: median {p['median_diff_m']:+.2f} m "
              f"[95% CI {p['ci95_low_m']:+.2f}, {p['ci95_high_m']:+.2f}], 3-pt better in {p['frac_trials_a_better']:.0%}")
    if not args.no_figures:
        make_figures(args.out, rows, ids, mov, ref, G, args.seed)
        print(f"\nfigures written to {args.out}")


def cmd_warp(args):
    with open(args.transform) as fh:
        spec = json.load(fh)
    M = np.array(spec["all_models"][args.model]["matrix"] if args.model else spec["matrix"])
    warp_raster(args.moving, args.out, M, args.reference, args.resampling)
    print(f"wrote {args.out} using {args.model or spec['selected']} transform")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    e = sub.add_parser("experiment", help="run the 2-vs-3 landmark registration-error experiments")
    e.add_argument("--landmarks", default=str(DEFAULT_RESULTS / "landmarks_for_tehsin.csv"))
    e.add_argument("--out", default=str(DEFAULT_RESULTS))
    e.add_argument("--trials", type=int, default=2000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--min-gain", type=float, default=MIN_GAIN_M)
    e.add_argument("--no-figures", action="store_true")
    w = sub.add_parser("warp", help="apply a fitted transform to a multiband GeoTIFF")
    w.add_argument("--moving", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--reference", help="GeoTIFF whose grid the output should match")
    w.add_argument("--transform", default=str(DEFAULT_RESULTS / "selected_transform.json"))
    w.add_argument("--model", choices=["none", "translation", "similarity", "affine"],
                   help="override the selected model")
    w.add_argument("--resampling", default="bilinear", choices=["nearest", "bilinear", "cubic"])
    args = ap.parse_args(argv)
    if args.cmd == "warp":
        cmd_warp(args)
    else:
        # no subcommand given -> run the experiment
        if args.cmd is None:
            args = ap.parse_args(["experiment"] + (argv or sys.argv[1:]))
        cmd_experiment(args)


if __name__ == "__main__":
    main()
