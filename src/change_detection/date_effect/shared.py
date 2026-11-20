"""Shared parts of the date effect tests: the held-out forests and site grids, the two-way tables that give a scene's
same-year swing, and the shuffled-dates test.

Every comparison removes the average of the same site and the same year first (column site_year), so a scene is only
compared with the other dates of its own site and year."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHIP = 256
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
INDEX_EPS = 0.01          # as in prepare.py
SITES = ["al_aryam_islands", "al_nouf", "eastern_mangroves", "ghantoot", "jubail_island", "ras_ghurab"]
REPO = Path(__file__).resolve().parents[3]
COMMITTED_PAIRS = REPO / "results" / "change_detection" / "all_sites" / "noise_floor_all_pairs.csv"


def load_common(train_val):
    """common.py of src/change_detection (the forest code and the cross-validation steps), imported unchanged."""
    if "Train_Val_Dataset" not in str(train_val):
        raise SystemExit("only the Train_Val_Dataset folder may be used")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import common
    return common


def indices(refl):
    """NDVI and MNDWI from the six bands [6, ...], with the formula of prepare.py."""
    b03, b04, b08, b11 = refl[1], refl[2], refl[3], refl[4]
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.clip((b08 - b04) / np.maximum(b08 + b04, INDEX_EPS), -1, 1)
        mndwi = np.clip((b03 - b11) / np.maximum(b03 + b11, INDEX_EPS), -1, 1)
    return ndvi, mndwi


def forests_and_stacks(tv, work):
    """Train the five held-out forests (same code and seed as the baseline) and put every non-hazy scene of every
    training site on its site grid. Each site is predicted by the forest that never saw it.

    Writes <work>/forests/fold<k>.joblib and <work>/stacks/<site>.npz: dates, image [date, 8, H, W] (six bands, NDVI,
    MNDWI), valid, covered, prob (forest probability), gmw4019, threshold, fold. Skipped if already there."""
    import joblib
    work = Path(work)
    if all((work / "stacks" / f"{s}.npz").exists() for s in SITES):
        return
    C = load_common(tv)
    (work / "forests").mkdir(parents=True, exist_ok=True)
    (work / "stacks").mkdir(parents=True, exist_ok=True)
    cat = C.load_full_catalog(tv)
    assert set(cat["role"]) == {"cv"}, "catalog holds rows that are not train/val"
    cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
    for k in sorted(int(f) for f in cat20.fold.unique()):
        fit = C.train_fold(cat20, k, tv, work / "forests" / "work", "b")
        rf, th = fit["rf"], fit["threshold"]
        pos = list(rf.classes_).index(True)
        iou = C.heldout_scores(rf, th, fit["held"], tv)["__fold__"]["iou"]
        print(f"fold {k}: threshold {th}, held-out IoU {iou:.4f}, committed {C.published_fold_iou(k, 'b')}", flush=True)
        assert abs(iou - C.published_fold_iou(k, "b")) < 1e-4, "the forest is not the committed one"
        joblib.dump({"rf": rf, "threshold": th, "held_sites": fit["info"]["held_sites"], "settings": fit["settings"]},
                    work / "forests" / f"fold{k}.joblib", compress=3)
        for site in fit["info"]["held_sites"]:
            rows = cat[cat["site"] == site]
            r0, c0 = int(rows["row"].min()), int(rows["col"].min())
            H, W = int(rows["row"].max()) - r0 + CHIP, int(rows["col"].max()) - c0 + CHIP
            dates = sorted(d for d, g in rows.groupby("date") if not bool(g["hazy"].iloc[0]))
            n = len(dates)
            img = np.full((n, 8, H, W), np.nan, np.float32)
            prob = np.zeros((n, H, W), np.float32)
            valid = np.zeros((n, H, W), bool)
            covered = np.zeros((n, H, W), bool)
            gmw = np.full((H, W), 255, np.uint8)
            for _, r in rows.iterrows():
                chip = C.load_chip(C.chip_path(tv, r))
                assert [str(c) for c in chip["channels"]] == BANDS + ["NDVI", "MNDWI"]
                y, x = int(r["row"]) - r0, int(r["col"]) - c0
                sl = (slice(y, y + CHIP), slice(x, x + CHIP))
                if int(r["year"]) == 2020 and "label_v4019" in chip:
                    lab = chip["label_v4019"]
                    gmw[sl] = np.where(lab != 255, lab, gmw[sl])
                if bool(r["hazy"]):
                    continue
                i = dates.index(r["date"])
                v = chip["valid"]
                p = np.zeros(v.shape, np.float32)
                if v.any():
                    p[v] = rf.predict_proba(C.B2.pixels(chip)[v.ravel()])[:, pos]
                prob[i][sl], valid[i][sl], covered[i][sl] = p, v, True
                img[i][(slice(None),) + sl] = chip["image"]
            np.savez_compressed(work / "stacks" / f"{site}.npz", dates=np.array(dates), image=img, prob=prob, valid=valid,
                                covered=covered, gmw4019=gmw, threshold=np.float32(th), fold=k)
            print(f"  {site}: {n} scenes, grid {H}x{W}", flush=True)


def load_stack(work, site):
    z = np.load(Path(work) / "stacks" / f"{site}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def predict(rf, pos, feats, valid):
    """Forest probability for one image [channels, H, W] on its usable pixels."""
    X = np.nan_to_num(feats[:, valid].T.astype(np.float32), nan=0.0)
    p = np.zeros(valid.shape, np.float32)
    if valid.any():
        p[valid] = rf.predict_proba(X)[:, pos]
    return p


def check_pairs(P):
    """The same-year pairs on the images as they are must be the committed same-year table."""
    com = pd.read_csv(COMMITTED_PAIRS)
    chk = P.merge(com, on=["site", "first_date", "second_date"], suffixes=("", "_committed"))
    assert len(chk) == len(com) == len(P), "same-year pairs differ from the committed table"
    worst = float((chk["net_pct_of_first"].round(1) - chk["net_pct_of_first_committed"]).abs().max())
    print(f"same-year pairs against the committed table: {len(chk)} pairs, largest difference {worst:.2f} points")
    assert worst < 0.051


# ---------------------------------------------------------------- two-way tables (chip cell x date)
# The chips of a site are not the same on every date, so each chip cell is its own fixed area and a date's value is
# built from the cells it has. This way a date is not penalised for the cells it happens to lack.
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
    """Dates linked to each other through shared cells: a bool per date for the largest linked group."""
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
    """Y[cell, date] counts, M = cell present on date. count = cell size x date factor.
    Returns the date factors scaled to mean 1 (nan where a date cannot be linked)."""
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


# ---------------------------------------------------------------- statistics
def demean(d, cols, by="site_year"):
    return d[cols] - d.groupby(by)[cols].transform("mean")


def _shuffles(d, by, n_perm):
    """Row orders for the shuffled-dates test: the dates are shuffled inside each site and year. Sites that share a
    calendar date get the same shuffle, because they share the same image pass and the same tide."""
    rng = np.random.default_rng(0)
    alld = sorted(d["date"].unique())
    idx = {dd: i for i, dd in enumerate(alld)}
    groups = [(g.index.values, np.array([idx[v] for v in g["date"]])) for _, g in d.groupby(by)]
    for _ in range(n_perm):
        key = rng.random(len(alld))
        src = np.empty(len(d), int)
        for ix, di in groups:
            src[ix[np.argsort(key[di])]] = ix
        yield src


def link(d, x, y, by="site_year", perm=True, n_perm=20000):
    """How strongly x goes with y once the average of each group is removed: correlation, share of y explained,
    and how often shuffled dates give a correlation at least that strong."""
    d = d.dropna(subset=[x, y])
    d = d[d.groupby(by)[x].transform("size") >= 2].sort_values([by, "date"]).reset_index(drop=True)
    dm = demean(d, [x, y], by)
    xd, yd = dm[x].values, dm[y].values
    r = np.corrcoef(xd, yd)[0, 1]
    chance = np.nan
    if perm:
        hits = sum(abs(np.corrcoef(xd[src], yd)[0, 1]) >= abs(r) - 1e-12 for src in _shuffles(d, by, n_perm))
        chance = round(100 * (hits + 1) / (n_perm + 1), 1)
    return {"scenes": len(d), "r": round(float(r), 3), "share_pct": round(100 * float(r * r), 1), "chance_shuffled_dates_pct": chance}


def link_together(d, y, xs, by="site_year", n_perm=10000):
    """The same for several measures together: share of y they explain, and how often shuffled dates do as well.
    The rows of measures are shuffled together between the dates of a site and year."""
    d = d.dropna(subset=[y] + xs).sort_values([by, "date"]).reset_index(drop=True)
    dm = demean(d, [y] + xs, by)
    yv, X = dm[y].values, dm[xs].values

    def r2(Xp):
        res = yv - Xp @ np.linalg.lstsq(Xp, yv, rcond=None)[0]
        return 1 - (res ** 2).sum() / (yv ** 2).sum()

    obs = r2(X)
    hits = sum(r2(X[src]) >= obs - 1e-12 for src in _shuffles(d, by, n_perm))
    return {"scenes": len(d), "share_pct": round(100 * float(obs), 1), "chance_shuffled_dates_pct": round(100 * (hits + 1) / (n_perm + 1), 1)}
