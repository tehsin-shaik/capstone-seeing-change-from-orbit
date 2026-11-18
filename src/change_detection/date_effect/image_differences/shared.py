"""Shared helpers for the image differences test. Same two-way tables and the same shuffled-dates test
as the tide test (../tide/04_analysis.py), kept here so parts b and c use one copy."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CHIP = 256
PX_HA = 0.01
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
INDEX_EPS = 0.01          # as in prepare.py
SITES = ["al_aryam_islands", "al_nouf", "eastern_mangroves", "ghantoot", "jubail_island", "ras_ghurab"]
NAMES = {"al_aryam_islands": "Al Aryam Islands", "al_nouf": "Al Nouf", "eastern_mangroves": "Eastern Mangroves",
         "ghantoot": "Ghantoot", "jubail_island": "Jubail Island", "ras_ghurab": "Ras Ghurab"}


def indices(refl):
    """NDVI and MNDWI from the six bands [6, ...], with the formula of prepare.py."""
    b03, b04, b08, b11 = refl[1], refl[2], refl[3], refl[4]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.clip((b08 - b04) / np.maximum(b08 + b04, INDEX_EPS), -1, 1)
        mndwi = np.clip((b03 - b11) / np.maximum(b03 + b11, INDEX_EPS), -1, 1)
    return ndvi, mndwi


def load_stack(work, site):
    z = np.load(Path(work) / "stacks" / f"{site}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def cell_layout(covered, valid):
    """Chip cells of a site grid. Returns the cell corners, present[cell, date], the fixed pixels (cell has >= 2 dates
    and the pixel is usable on all of them) and the number of dates of each pixel's cell."""
    n, H, W = covered.shape
    cells = [(r, c) for r in range(0, H, CHIP) for c in range(0, W, CHIP)]
    present = np.array([[bool(covered[i, r, c]) for i in range(n)] for r, c in cells])
    use = np.zeros((H, W), bool)
    npres = np.zeros((H, W), np.int16)
    for (r, c), p in zip(cells, present):
        if p.sum() >= 2:
            use[r:r + CHIP, c:c + CHIP] = np.all(valid[p][:, r:r + CHIP, c:c + CHIP], axis=0)
            npres[r:r + CHIP, c:c + CHIP] = p.sum()
    return cells, present, use, npres


def largest_component(M):
    D = M.shape[1]
    label = -np.ones(D, int)
    for d in range(D):
        if label[d] >= 0 or not M[:, d].any():
            continue
        todo, label[d] = [d], d
        while todo:
            j = todo.pop()
            for k in np.where(M[M[:, j]].any(axis=0))[0]:
                if label[k] < 0:
                    label[k] = d
                    todo.append(k)
    ok = label >= 0
    if not ok.any():
        return ok
    return label == np.bincount(label[ok]).argmax()


def ipf(Y, M):
    """count = cell size x date factor. Returns the date factors scaled to mean 1 (nan where a date cannot be linked)."""
    D = Y.shape[1]
    keep = (M.sum(1) >= 2) & ((Y * M).sum(1) > 0)
    Y, M = Y[keep].astype(float), M[keep]
    g = np.full(D, np.nan)
    if not keep.any():
        return g
    link = largest_component(M)
    Y, M = Y[:, link], M[:, link]
    k2 = M.sum(1) >= 2
    Y, M = Y[k2], M[k2]
    gam = np.ones(link.sum())
    for _ in range(5000):
        alpha = (Y * M).sum(1) / (M * gam).sum(1)
        new = (Y * M).sum(0) / (M * alpha[:, None]).sum(0)
        new = new / new.mean()
        done = np.max(np.abs(new - gam)) < 1e-10
        gam = new
        if done:
            break
    g[link] = gam
    return g


def additive(V, M, w):
    """value = cell level + date shift (weighted least squares). Returns the site value per date."""
    D = V.shape[1]
    keep = (M.sum(1) >= 2) & (w > 0)
    res = np.full(D, np.nan)
    if not keep.any():
        return res
    V, M, w = np.where(M[keep], V[keep], 0.0), M[keep], w[keep].astype(float)
    link = largest_component(M)
    V, M = V[:, link], M[:, link]
    k2 = M.sum(1) >= 2
    V, M, w = V[k2], M[k2], w[k2]
    b = np.zeros(link.sum())
    for _ in range(5000):
        acell = ((V - b) * M).sum(1) / M.sum(1)
        new = ((V - acell[:, None]) * M * w[:, None]).sum(0) / (M * w[:, None]).sum(0)
        new = new - new.mean()
        done = np.max(np.abs(new - b)) < 1e-12
        b = new
        if done:
            break
    acell = ((V - b) * M).sum(1) / M.sum(1)
    res[link] = (acell * w).sum() / w.sum() + b
    return res


# ---------------------------------------------------------------- statistics (as in the tide check)
def demean(d, cols, by="site"):
    return d[cols] - d.groupby(by)[cols].transform("mean")


def perm_p(d, x, y, by="site", n_perm=20000, seed=0):
    """Chance of a within-site correlation at least this strong if x had nothing to do with y. x is shuffled between the
    dates of each site. Sites that share a calendar date get the same shuffle, because they share the same image pass."""
    rng = np.random.default_rng(seed)
    d = d.dropna(subset=[x, y]).sort_values([by, "date"]).reset_index(drop=True)
    yd = demean(d, [y], by)[y].values
    xd = demean(d, [x], by)[x].values
    obs = np.corrcoef(xd, yd)[0, 1]
    alld = sorted(d["date"].unique())
    idx = {dd: i for i, dd in enumerate(alld)}
    groups = [(g.index.values, np.array([idx[v] for v in g["date"]])) for _, g in d.groupby(by)]
    hits = 0
    for _ in range(n_perm):
        key = rng.random(len(alld))
        xp = np.empty(len(d))
        for ix, di in groups:
            xp[ix[np.argsort(key[di])]] = xd[ix]
        hits += abs(np.corrcoef(xp, yd)[0, 1]) >= abs(obs) - 1e-12
    return (hits + 1) / (n_perm + 1)


def pooled(d, x, y, by="site", perm=True):
    d = d.dropna(subset=[x, y])
    d = d[d.groupby(by)[x].transform("size") >= 2]
    dm = demean(d, [x, y], by)
    sd = d.assign(**{x: dm[x], y: dm[y]}).groupby(by)[[x, y]].transform("std").replace(0, np.nan)
    zz = (dm / sd).dropna()
    r = np.corrcoef(dm[x], dm[y])[0, 1]
    loo = [np.corrcoef(*demean(d[d["site"] != s], [x, y], by).values.T)[0, 1] for s in d["site"].unique()]
    return {"x": x, "y": y, "scenes": len(d), "dates": d["date"].nunique(), "r_within": round(r, 3), "share_explained": round(r * r, 3),
            "spearman_within": round(stats.spearmanr(dm[x], dm[y])[0], 3),
            "r_each_site_equal_weight": round(np.corrcoef(zz[x], zz[y])[0, 1], 3) if len(zz) > 3 else np.nan,
            "p_shuffle_dates": round(perm_p(d, x, y, by), 4) if perm else np.nan,
            "leave_one_site_out_r_min": round(min(loo), 3), "leave_one_site_out_r_max": round(max(loo), 3)}


def r2_within(d, y, xs, by="site"):
    d = d.dropna(subset=[y] + xs)
    dm = demean(d, [y] + xs, by)
    X = dm[xs].values
    res = dm[y].values - X @ np.linalg.lstsq(X, dm[y].values, rcond=None)[0]
    n, k, p = len(d), d[by].nunique(), len(xs)
    R2 = 1 - (res ** 2).sum() / (dm[y].values ** 2).sum()
    dfree = n - k - p
    return R2, (1 - (1 - R2) * (n - k) / dfree if dfree > 0 else np.nan), n
