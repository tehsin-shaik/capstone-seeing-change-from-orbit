"""Does tide explain why the random forest's mangrove predictions change with the image date?

Inputs
    <work>/pred/<site>.npz         from 02_predict_all_scenes.py (forest that held the site out, non-hazy scenes)
    <out>/scenes.csv               from 01_scene_times.py
    <out>/scene_tides.csv, scene_tides_lagged.csv   from 03_tide_model.py
    --disagreement per_site_date.csv of the model disagreement check (2020 miss / false alarm shares of all three models)

How the "fixed area" is done
    The chips of a site are not the same on every date. Each 256 x 256 chip cell is therefore its own fixed area:
    only cells with at least two non-hazy dates are used, and inside a cell only the pixels usable on every one
    of the cell's dates. A site value for a date is then built from its cells with a two-way table (cell x date),
    so a date is not penalised for the cells it happens to lack:
      * predicted mangrove area: area = cell size x date factor (fitted by iterative proportional fitting).
        The date factor, relative to the mean over the site's dates, is the scene's deviation in percent.
      * water share and interior NDVI: value = cell level + date shift (weighted least squares).
    On cells present on every date both give the plain totals. A strict version, only the cells present on every
    date of the site, is also written (strict_*), as a check.

Water rule (one rule, every date): a pixel is water if green (B03) is brighter than near-infrared (B08).
Second rule as a check: green brighter than short-wave infrared (B11), which is MNDWI > 0.

Outputs in <out>: scene_table.csv, test1_per_site.csv, test1_pooled.csv, test1_lags.csv, test2_strip.csv,
season_models.csv, season_partial.csv, chip_level.csv, summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

CHIP = 256
PX_HA = 0.01
RNG = np.random.default_rng(0)
K3 = np.ones((3, 3), bool)

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--disagreement", required=True)
ap.add_argument("--train-val", required=True, help="Train_Val_Dataset, only for hazy_scenes.csv (the dark-water brightness of each scene)")
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)


# ------------------------------------------------------------------ two-way tables
def largest_component(M):
    """Dates linked to each other through shared cells. Returns a bool per date for the largest linked group."""
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
    best = np.bincount(label[ok]).argmax()
    return label == best


def ipf(Y, M):
    """Y[cell, date] counts, M = cell present on date. Model: count = cell size x date factor.
    Returns the date factors scaled to mean 1 (nan for dates that cannot be linked) and the fitted site total per date."""
    C, D = Y.shape
    keep = (M.sum(1) >= 2) & ((Y * M).sum(1) > 0)
    Y, M = Y[keep].astype(float), M[keep]
    g = np.full(D, np.nan)
    if not keep.any():
        return g, g.copy(), 0
    link = largest_component(M)
    Y, M = Y[:, link], M[:, link]
    keep2 = M.sum(1) >= 2
    Y, M = Y[keep2], M[keep2]
    gam = np.ones(link.sum())
    for _ in range(5000):
        alpha = (Y * M).sum(1) / (M * gam).sum(1)
        new = (Y * M).sum(0) / (M * alpha[:, None]).sum(0)
        new = new / new.mean()
        done = np.max(np.abs(new - gam)) < 1e-10
        gam = new
        if done:
            break
    alpha = (Y * M).sum(1) / (M * gam).sum(1)
    g[link] = gam
    return g, g * alpha.sum(), int(keep2.sum())


def additive(V, M, w):
    """V[cell, date] values, weights w per cell. Model: value = cell level + date shift (weighted least squares).
    Returns the site value per date = weighted mean cell level + date shift."""
    C, D = V.shape
    keep = (M.sum(1) >= 2) & (w > 0)
    res = np.full(D, np.nan)
    if not keep.any():
        return res
    V, M, w = np.where(M[keep], V[keep], 0.0), M[keep], w[keep].astype(float)
    link = largest_component(M)
    V, M = V[:, link], M[:, link]
    keep2 = M.sum(1) >= 2
    V, M, w = V[keep2], M[keep2], w[keep2]
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


# ------------------------------------------------------------------ per site
scenes = pd.read_csv(out / "scenes.csv").merge(pd.read_csv(out / "scene_tides.csv"), on=["site", "date", "time_utc"])
dis = pd.read_csv(a.disagreement)
rows, strip_rows, chip_rows = [], [], []
SITES = sorted(scenes["site"].unique())

for site in SITES:
    z = np.load(work / "pred" / f"{site}.npz", allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    years = np.array([int(d[:4]) for d in dates])
    n = len(dates)
    th = float(z["threshold"])
    valid, covered = z["valid"], z["covered"]
    mask = (z["prob"].astype(np.float32) > th) & valid
    b03, b08, b11, ndvi = z["b03"], z["b08"], z["b11"], z["ndvi"]
    with np.errstate(invalid="ignore"):
        water = (b03 > b08) & valid
        water2 = (b03 > b11) & valid
    gmw = z["gmw4019"]
    H, W = gmw.shape

    # ---- cells: present dates and the pixels usable on all of them
    cells = [(r, c) for r in range(0, H, CHIP) for c in range(0, W, CHIP)]
    present = np.array([[bool(covered[i, r, c]) for i in range(n)] for r, c in cells])          # [cell, date]
    use = np.zeros((H, W), bool)             # pixel belongs to a cell with >= 2 dates and is usable on all of them
    npres = np.zeros((H, W), np.int16)       # number of dates of the pixel's cell
    for (r, c), p in zip(cells, present):
        if p.sum() >= 2:
            sl = (slice(r, r + CHIP), slice(c, c + CHIP))
            use[sl] = np.all(valid[p][(slice(None),) + sl], axis=0)
            npres[sl] = p.sum()
    cov = covered & use[None]                # pixel analysed on that date
    m, w1, w2 = mask & cov, water & cov, water2 & cov

    # ---- solid mangrove interior: GMW mangrove at least 3 pixels from its edge, never water, always predicted mangrove
    core = ndimage.binary_erosion(gmw == 1, K3, iterations=3)
    interior = core & use & (w1.sum(0) == 0) & (m.sum(0) == npres) & (npres >= 2)

    def per_cell(arr2d_or_3d, weight=None):
        """Sum over each cell: [cell] for a 2-D array, [cell, date] for a 3-D one."""
        x = np.asarray(arr2d_or_3d, dtype=np.float64)
        if x.ndim == 2:
            return x.reshape(H // CHIP, CHIP, W // CHIP, CHIP).sum((1, 3)).ravel()
        return x.reshape(n, H // CHIP, CHIP, W // CHIP, CHIP).sum((2, 4)).reshape(n, -1).T

    npx = per_cell(use)
    Ym, Yw1, Yw2 = per_cell(m), per_cell(w1), per_cell(w2)
    P = present & (present.sum(1) >= 2)[:, None] & (npx > 0)[:, None]

    g, total_px, ncell = ipf(Ym, P)
    with np.errstate(invalid="ignore", divide="ignore"):
        wat1 = additive(100 * Yw1 / npx[:, None], P, npx)
        wat2 = additive(100 * Yw2 / npx[:, None], P, npx)
        nint = per_cell(interior)
        nd = per_cell(np.where(interior[None] & cov, np.nan_to_num(ndvi), 0.0)) / np.where(nint > 0, nint, np.nan)[:, None]
        ndv = additive(np.nan_to_num(nd), P & (nint >= 50)[:, None], np.where(nint >= 50, nint, 0))

    # ---- 2020 only: the forest's own misses and false alarms against GMW v4.0.19 on the same fixed pixels
    is20 = years == 2020
    lab = (gmw != 255) & use
    miss = per_cell(m == 0) * 0
    miss = per_cell((~m) & cov & (lab & (gmw == 1))[None])
    fa = per_cell(m & (lab & (gmw == 0))[None])
    gm = per_cell(lab & (gmw == 1))
    P20 = P & is20[None, :]
    P20 = P20 & (P20.sum(1) >= 2)[:, None] & (gm > 0)[:, None]
    _, miss_tot, _ = ipf(np.where(P20, miss, 0), P20)
    _, fa_tot, _ = ipf(np.where(P20, fa, 0), P20)
    gm_tot = gm[P20.any(1)].sum()

    # ---- strict check: only the cells present on every date
    allc = present.all(1)
    strict_use = np.zeros((H, W), bool)
    for (r, c), ok in zip(cells, allc):
        if ok:
            strict_use[r:r + CHIP, c:c + CHIP] = use[r:r + CHIP, c:c + CHIP]
    s_m = (mask & strict_use[None]).sum((1, 2)).astype(float)
    s_w = (water & strict_use[None]).sum((1, 2)).astype(float)

    for i, d in enumerate(dates):
        same_year = years == years[i]
        dd = dis[(dis["site"] == site) & (dis["date"] == d)]
        rows.append({
            "site": site, "date": d,
            "cells_on_this_date": int(P[:, i].sum()), "cells_in_site_table": ncell,
            "pred_mangrove_ha": round(total_px[i] * PX_HA, 1) if np.isfinite(total_px[i]) else np.nan,
            "dev_pct": round(100 * (g[i] - 1), 2),
            "dev_within_year_pct": round(100 * (g[i] / np.nanmean(g[same_year]) - 1), 2) if same_year.sum() > 1 else np.nan,
            "water_pct": round(wat1[i], 2), "water_swir_rule_pct": round(wat2[i], 2),
            "interior_ndvi": round(ndv[i], 4), "interior_px": int(nint.sum()),
            "rf_miss_pct_of_gmw": round(100 * miss_tot[i] / gm_tot, 2) if is20[i] and gm_tot and np.isfinite(miss_tot[i]) else np.nan,
            "rf_false_alarm_pct_of_gmw": round(100 * fa_tot[i] / gm_tot, 2) if is20[i] and gm_tot and np.isfinite(fa_tot[i]) else np.nan,
            "all3_miss_pct_of_gmw": float(dd["mangrove_rejected_by_all3_pct"].iloc[0]) if len(dd) else np.nan,
            "all3_false_alarm_pct_of_gmw": float(dd["called_by_all3_as_pct_of_gmw_mangrove_area"].iloc[0]) if len(dd) else np.nan,
            "strict_cells": int(allc.sum()),
            "strict_dev_pct": round(100 * (s_m[i] / s_m.mean() - 1), 2) if allc.any() and s_m.mean() > 0 else np.nan,
            "strict_water_pct": round(100 * s_w[i] / strict_use.sum(), 2) if allc.any() else np.nan,
        })

    # ---- chip level: per cell and date, mangrove and water as a share of the cell's fixed pixels
    for ci, (r, c) in enumerate(cells):
        for i in np.where(P[ci])[0]:
            chip_rows.append({"site": site, "date": dates[i], "cell": f"{r}_{c}", "px": npx[ci],
                              "mangrove_pct": 100 * Ym[ci, i] / npx[ci], "water_pct": 100 * Yw1[ci, i] / npx[ci]})

    # ---- Test 2: the strip that is water on some dates and land on others, and the pixels whose prediction flips
    def strip_test(sel, tag):
        """sel = dates to use. Pixels need >= 2 of those dates."""
        k = (cov[sel]).sum(0)
        ok = use & (k >= 2)
        ws, ms = w1[sel].sum(0), m[sel].sum(0)
        strip = ok & (ws > 0) & (ws < k)
        flip = ok & (ms > 0) & (ms < k)
        ever = ok & (ms > 0)
        zone = ndimage.binary_dilation(strip, K3) & ok
        zone2 = ndimage.binary_dilation(strip, K3, iterations=2) & ok
        maj = ok & (2 * ms >= k) & (ms > 0)
        edge = ok & ndimage.binary_dilation(maj, K3) & ~ndimage.binary_erosion(maj, K3, border_value=1)
        nflip, nok = int(flip.sum()), int(ok.sum())
        # mangrove rate of strip pixels on their water dates and on their land dates
        wd = (w1[sel] & strip[None]).sum(); ld = (cov[sel] & ~w1[sel] & strip[None]).sum()
        res = {"site": site, "dates_used": tag, "pixels": nok, "flip_px": nflip, "strip_px": int(strip.sum()),
               "zone_px": int(zone.sum()), "flip_in_strip_px": int((flip & strip).sum()), "flip_in_zone_px": int((flip & zone).sum()),
               "flip_in_zone2_px": int((flip & zone2).sum()), "zone2_px": int(zone2.sum()),
               "ever_px": int(ever.sum()), "ever_in_zone_px": int((ever & zone).sum()),
               "edge_px": int(edge.sum()), "edge_in_zone_px": int((edge & zone).sum()),
               "edge_flip_px": int((edge & flip).sum()), "edge_flip_in_zone_px": int((edge & flip & zone).sum()),
               "strip_mangrove_on_water_dates": int((m[sel] & w1[sel] & strip[None]).sum()), "strip_water_pixel_dates": int(wd),
               "strip_mangrove_on_land_dates": int((m[sel] & ~w1[sel] & strip[None]).sum()), "strip_land_pixel_dates": int(ld)}
        # chance by moving the strip somewhere else in the site: 200 random shifts (wrap around), same shape
        if tag == "all dates" and nflip:
            sh = []
            for _ in range(200):
                dy, dx = int(RNG.integers(20, H - 20)), int(RNG.integers(20, W - 20))
                zs = np.roll(zone, (dy, dx), (0, 1)) & ok
                sh.append((flip & zs).sum() / nflip)
            res["shift_null_mean_share"] = float(np.mean(sh))
            res["shift_null_p97_5_share"] = float(np.quantile(sh, 0.975))
        return res

    strip_rows.append(strip_test(np.ones(n, bool), "all dates"))
    for y in sorted(set(years)):
        if (years == y).sum() >= 2:
            strip_rows.append(strip_test(years == y, f"within {y}"))
    print(f"{site}: {n} scenes, {ncell} cells in the table, {int(use.sum())} fixed pixels, interior {int(interior.sum())} px", flush=True)

T = scenes.merge(pd.DataFrame(rows), on=["site", "date"])
# dark-water brightness of the scene (red band, 5th percentile over water): the team's haze measure. Near 0 is clear,
# above 0.05 was dropped as hazy, and below 0 means the atmospheric correction took too much off that scene.
hz = pd.read_csv(Path(a.train_val) / "hazy_scenes.csv")[["site", "date", "water_b04_p5"]].drop_duplicates(["site", "date"])
T = T.merge(hz.rename(columns={"water_b04_p5": "dark_water_red"}), on=["site", "date"], how="left")
dt = pd.to_datetime(T["date"])
T["month"], T["doy"] = dt.dt.month, dt.dt.dayofyear
T["season"] = np.where(T["month"] == 11, "November", np.where(T["month"] <= 3, "January to March", "May or September"))
T["november"] = (T["month"] == 11).astype(float)
T["cos_doy"], T["sin_doy"] = np.cos(2 * np.pi * T["doy"] / 365.25), np.sin(2 * np.pi * T["doy"] / 365.25)
T["east_looking"] = (T["view_azimuth"] < 180).astype(float)
T.to_csv(out / "scene_table.csv", index=False)

# ------------------------------------------------------------------ Test 2 table
S = pd.DataFrame(strip_rows)
tot = S[S["dates_used"] == "all dates"].drop(columns=["site", "dates_used"]).sum(numeric_only=True)
wy = S[S["dates_used"] != "all dates"].drop(columns=["site", "dates_used"]).sum(numeric_only=True)
S = pd.concat([S, pd.DataFrame([{"site": "ALL SITES", "dates_used": "all dates", **tot.to_dict()},
                                {"site": "ALL SITES", "dates_used": "within a year, years pooled", **wy.to_dict()}])], ignore_index=True)
for c in ("shift_null_mean_share", "shift_null_p97_5_share"):
    S.loc[S["site"] == "ALL SITES", c] = np.nan
S["flips_in_or_next_to_strip_pct"] = 100 * S.flip_in_zone_px / S.flip_px
S["flips_in_strip_pct"] = 100 * S.flip_in_strip_px / S.flip_px
S["flips_within_2px_of_strip_pct"] = 100 * S.flip_in_zone2_px / S.flip_px
S["all_pixels_in_or_next_to_strip_pct"] = 100 * S.zone_px / S.pixels
S["ever_mangrove_in_or_next_to_strip_pct"] = 100 * S.ever_in_zone_px / S.ever_px
S["times_chance_all_pixels"] = S.flips_in_or_next_to_strip_pct / S.all_pixels_in_or_next_to_strip_pct
S["times_chance_ever_mangrove"] = S.flips_in_or_next_to_strip_pct / S.ever_mangrove_in_or_next_to_strip_pct
S["flip_rate_near_strip_pct"] = 100 * S.flip_in_zone_px / S.ever_in_zone_px
S["flip_rate_away_pct"] = 100 * (S.flip_px - S.flip_in_zone_px) / (S.ever_px - S.ever_in_zone_px)
S["edge_flip_rate_near_strip_pct"] = 100 * S.edge_flip_in_zone_px / S.edge_in_zone_px
S["edge_flip_rate_away_pct"] = 100 * (S.edge_flip_px - S.edge_flip_in_zone_px) / (S.edge_px - S.edge_in_zone_px)
S["strip_mangrove_rate_when_water_pct"] = 100 * S.strip_mangrove_on_water_dates / S.strip_water_pixel_dates
S["strip_mangrove_rate_when_land_pct"] = 100 * S.strip_mangrove_on_land_dates / S.strip_land_pixel_dates
S["shift_null_mean_pct"] = 100 * S["shift_null_mean_share"]
S["shift_null_p97_5_pct"] = 100 * S["shift_null_p97_5_share"]
S.drop(columns=["shift_null_mean_share", "shift_null_p97_5_share"]).round(2).to_csv(out / "test2_strip.csv", index=False)


# ------------------------------------------------------------------ Test 1 and season statistics
def demean(d, cols, by="site"):
    return d[cols] - d.groupby(by)[cols].transform("mean")


def perm_p(d, x, y, by="site", n_perm=20000):
    """Chance of a within-site correlation at least this strong if x had nothing to do with y. x is shuffled between the
    dates of each site. Sites that share a calendar date get the same shuffle, because their tide is the same tide."""
    d = d.dropna(subset=[x, y]).sort_values([by, "date"])
    yd = demean(d, [y], by)[y].values
    obs = np.corrcoef(demean(d, [x], by)[x].values, yd)[0, 1]
    alld = sorted(d["date"].unique())
    idx = {dd: i for i, dd in enumerate(alld)}
    groups = [(g.index.values, np.array([idx[v] for v in g["date"]]), g[x].values) for _, g in d.groupby(by)]
    pos = {ix: k for k, ix in enumerate(d.index)}
    hits = 0
    for _ in range(n_perm):
        key = RNG.random(len(alld))
        xp = np.empty(len(d))
        for ix, di, xv in groups:
            order = np.argsort(key[di])
            vals = xv - xv.mean()
            for k, o in enumerate(order):
                xp[pos[ix[o]]] = vals[k]
        if abs(np.corrcoef(xp, yd)[0, 1]) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def pooled(d, x, y, by="site", perm=True, min_n=2):
    d = d.dropna(subset=[x, y])
    d = d[d.groupby(by)[x].transform("size") >= min_n]
    if len(d) < 4:
        return None
    dm = demean(d, [x, y], by)
    zs = dm / d.assign(**dm).groupby(by)[[x, y]].transform("std").replace(0, np.nan)
    r = np.corrcoef(dm[x], dm[y])[0, 1]
    k = d[by].nunique()
    slope = (dm[x] * dm[y]).sum() / (dm[x] ** 2).sum()
    zz = zs.dropna()
    return {"x": x, "y": y, "grouping": by, "scenes": len(d), "groups": k, "passes": d["date"].nunique(),
            "r_within": round(r, 3), "r2_within": round(r * r, 3), "spearman_within": round(stats.spearmanr(dm[x], dm[y])[0], 3),
            "r_each_site_equal_weight": round(np.corrcoef(zz[x], zz[y])[0, 1], 3) if len(zz) > 3 else np.nan,
            "slope_y_per_unit_x": round(slope, 3),
            "p_shuffle_dates": round(perm_p(d, x, y, by), 4) if perm else np.nan,
            "leave_one_site_out_r_min": round(min(np.corrcoef(*demean(d[d["site"] != s], [x, y], by).values.T)[0, 1] for s in d["site"].unique()), 3),
            "leave_one_site_out_r_max": round(max(np.corrcoef(*demean(d[d["site"] != s], [x, y], by).values.T)[0, 1] for s in d["site"].unique()), 3)}


XS = ["water_pct", "water_swir_rule_pct", "tide_eot20_m", "tide_got410_m", "tide_position_0low_1high"]
per_site = []
for site, g in T.groupby("site"):
    r = {"site": site, "scenes": len(g), "dev_pct_min": g.dev_pct.min(), "dev_pct_max": g.dev_pct.max(), "dev_pct_sd": round(g.dev_pct.std(), 1),
         "water_pct_min": g.water_pct.min(), "water_pct_max": g.water_pct.max(),
         "tide_m_min": g.tide_eot20_m.min(), "tide_m_max": g.tide_eot20_m.max()}
    for x in XS + ["sun_elevation", "interior_ndvi", "november"]:
        gg = g.dropna(subset=[x, "dev_pct"])
        r[f"r_dev_vs_{x}"] = round(np.corrcoef(gg[x], gg["dev_pct"])[0, 1], 2) if len(gg) >= 3 and gg[x].std() > 0 else np.nan
    gg = g.dropna(subset=["water_pct", "tide_eot20_m"])
    r["r_water_vs_tide"] = round(np.corrcoef(gg.water_pct, gg.tide_eot20_m)[0, 1], 2)
    r["r_tide_vs_sun_elevation"] = round(np.corrcoef(g.tide_eot20_m, g.sun_elevation)[0, 1], 2)
    per_site.append(r)
pd.DataFrame(per_site).to_csv(out / "test1_per_site.csv", index=False)

pool = []
for y in ("dev_pct", "dev_within_year_pct", "strict_dev_pct"):
    for x in XS:
        xx = "strict_water_pct" if (y == "strict_dev_pct" and x == "water_pct") else x
        d = T[T["strict_cells"] >= 2] if y == "strict_dev_pct" else T
        by = "site_year" if y == "dev_within_year_pct" else "site"
        d = d.assign(site_year=d["site"] + "_" + d["year"].astype(str))
        res = pooled(d, xx, y, by)
        if res:
            pool.append(res)
for y in ("rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw", "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw"):
    for x in ("water_pct", "tide_eot20_m"):
        res = pooled(T[T["year"] == 2020], x, y)
        if res:
            pool.append(res)
pool.append(pooled(T, "tide_eot20_m", "water_pct"))
pool.append(pooled(T, "tide_got410_m", "water_pct"))
for x in ("water_pct", "tide_eot20_m"):
    res = pooled(T[T["cells_on_this_date"] >= 5], x, "dev_pct")
    res["note"] = "only scenes with at least 5 chip cells"
    pool.append(res)
    res = pooled(T[T["site"] != "al_nouf"], x, "dev_pct")
    res["note"] = "without Al Nouf (few chips)"
    pool.append(res)
    res = pooled(T[T["date"] != "2020-05-18"], x, "dev_pct")
    res["note"] = "without the 18 May 2020 pass"
    pool.append(res)
for x in ("sun_elevation", "november", "interior_ndvi", "east_looking", "dark_water_red"):
    pool.append(pooled(T, x, "dev_pct"))
    pool.append(pooled(T, x, "tide_eot20_m", perm=False))
    pool.append(pooled(T, x, "water_pct", perm=False))
pd.DataFrame(pool).to_csv(out / "test1_pooled.csv", index=False)

# lag check: which delay of the model tide matches the water seen in the image, and the swings
L = pd.read_csv(out / "scene_tides_lagged.csv").merge(T[["site", "date", "water_pct", "dev_pct"]], on=["site", "date"])
lag_rows = []
for hrs, g in L.groupby("hours_earlier"):
    dm = demean(g, ["tide_eot20_m", "water_pct", "dev_pct"])
    row = {"hours_earlier": hrs, "r_water_vs_tide_all_sites": round(np.corrcoef(dm.tide_eot20_m, dm.water_pct)[0, 1], 3),
           "r_dev_vs_tide_all_sites": round(np.corrcoef(dm.tide_eot20_m, dm.dev_pct)[0, 1], 3)}
    for s, gs in g.groupby("site"):
        row[f"r_water_vs_tide_{s}"] = round(np.corrcoef(gs.tide_eot20_m, gs.water_pct)[0, 1], 2)
    lag_rows.append(row)
pd.DataFrame(lag_rows).to_csv(out / "test1_lags.csv", index=False)


# ---- season: how much of the within-site swing does each thing explain, alone and after the other
def r2(d, y, xs, by="site"):
    d = d.dropna(subset=[y] + xs)
    dm = demean(d, [y] + xs, by)
    X = dm[xs].values
    beta, *_ = np.linalg.lstsq(X, dm[y].values, rcond=None)
    res = dm[y].values - X @ beta
    n, k, p = len(d), d[by].nunique(), len(xs)
    R2 = 1 - (res ** 2).sum() / (dm[y].values ** 2).sum()
    dfree = n - k - p
    return R2, 1 - (1 - R2) * (n - k) / dfree if dfree > 0 else np.nan, n, dfree


MODELS = {"water share (image)": ["water_pct"], "tide height (EOT20 model)": ["tide_eot20_m"],
          "sun elevation": ["sun_elevation"], "November or not": ["november"], "time of year (annual wave)": ["cos_doy", "sin_doy"],
          "interior greenness (NDVI)": ["interior_ndvi"], "which side the satellite looked from": ["east_looking"],
          "dark-water brightness (haze measure)": ["dark_water_red"],
          "tide + dark-water brightness": ["tide_eot20_m", "dark_water_red"],
          "water share + dark-water brightness": ["water_pct", "dark_water_red"],
          "sun elevation + dark-water brightness": ["sun_elevation", "dark_water_red"],
          "sun elevation + interior greenness": ["sun_elevation", "interior_ndvi"],
          "sun elevation + interior greenness + dark-water brightness": ["sun_elevation", "interior_ndvi", "dark_water_red"],
          "tide + sun elevation": ["tide_eot20_m", "sun_elevation"], "tide + November": ["tide_eot20_m", "november"],
          "tide + time of year": ["tide_eot20_m", "cos_doy", "sin_doy"], "tide + interior greenness": ["tide_eot20_m", "interior_ndvi"],
          "water share + sun elevation": ["water_pct", "sun_elevation"], "water share + November": ["water_pct", "november"],
          "water share + time of year": ["water_pct", "cos_doy", "sin_doy"],
          "water share + interior greenness": ["water_pct", "interior_ndvi"],
          "water share + tide": ["water_pct", "tide_eot20_m"],
          "water share + tide + sun elevation + interior greenness": ["water_pct", "tide_eot20_m", "sun_elevation", "interior_ndvi"]}
Tz = T.copy()
num = ["dev_pct", "water_pct", "tide_eot20_m", "sun_elevation", "november", "cos_doy", "sin_doy", "interior_ndvi", "east_looking"]
mods = []
for name, xs in MODELS.items():
    R2, adj, n, dfree = r2(T, "dev_pct", xs)
    # every site given the same weight: the deviation is divided by the site's own spread first
    Tz["dev_z"] = demean(T, ["dev_pct"])["dev_pct"] / T.groupby("site")["dev_pct"].transform("std")
    R2z, adjz, _, _ = r2(Tz, "dev_z", xs)
    mods.append({"explained_by": name, "terms": len(xs), "scenes": n, "r2_within_site": round(R2, 3), "r2_adjusted": round(adj, 3),
                 "r2_each_site_equal_weight": round(R2z, 3), "r2_adjusted_equal_weight": round(adjz, 3)})
pd.DataFrame(mods).to_csv(out / "season_models.csv", index=False)


def partial(d, x, y, ctrl, by="site"):
    d = d.dropna(subset=[x, y] + ctrl)
    dm = demean(d, [x, y] + ctrl, by)
    C = dm[ctrl].values
    rx = dm[x].values - C @ np.linalg.lstsq(C, dm[x].values, rcond=None)[0]
    ry = dm[y].values - C @ np.linalg.lstsq(C, dm[y].values, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    dfree = len(d) - d[by].nunique() - len(ctrl) - 1
    p = 2 * stats.t.sf(abs(r) * np.sqrt(dfree / max(1e-12, 1 - r * r)), dfree) if dfree > 0 else np.nan
    return {"x": x, "y": y, "holding_fixed": " + ".join(ctrl), "scenes": len(d), "degrees_of_freedom": dfree,
            "partial_r": round(r, 3), "p_naive_scenes_independent": round(p, 4)}


part = []
for x, ctrls in (("tide_eot20_m", [["sun_elevation"], ["november"], ["cos_doy", "sin_doy"], ["interior_ndvi"], ["dark_water_red"],
                                   ["sun_elevation", "interior_ndvi", "dark_water_red"]]),
                 ("water_pct", [["sun_elevation"], ["november"], ["cos_doy", "sin_doy"], ["interior_ndvi"], ["tide_eot20_m"], ["dark_water_red"],
                                ["sun_elevation", "interior_ndvi", "dark_water_red"]]),
                 ("dark_water_red", [["tide_eot20_m"], ["sun_elevation"], ["interior_ndvi"]]),
                 ("sun_elevation", [["tide_eot20_m"], ["water_pct"]]), ("november", [["tide_eot20_m"], ["water_pct"]]),
                 ("interior_ndvi", [["tide_eot20_m"], ["water_pct"]])):
    for c in ctrls:
        part.append(partial(T, x, "dev_pct", c))
# same season only: remove the average of each site and season, so only scenes of the same season are compared
T["site_season"] = T["site"] + "_" + T["season"]
for x in ("water_pct", "tide_eot20_m"):
    res = pooled(T, x, "dev_pct", "site_season")
    if res:
        res["note"] = "same site and same season only"
        pool.append(res)
pd.DataFrame(pool).to_csv(out / "test1_pooled.csv", index=False)
pd.DataFrame(part).to_csv(out / "season_partial.csv", index=False)

# ---- chip level: inside one scene the season is the same for every chip, but the water does not move the same amount
Cc = pd.DataFrame(chip_rows)
Cc["sd"] = Cc["site"] + "_" + Cc["date"]
Cc["sc"] = Cc["site"] + "_" + Cc["cell"]
chip_out = []
for label, d in [("all sites", Cc)] + [(s, g) for s, g in Cc.groupby("site")]:
    d = d.copy()
    for _ in range(200):                      # remove the cell average and the scene average (two-way)
        for v in ("mangrove_pct", "water_pct"):
            d[v] = d[v] - d.groupby("sc")[v].transform("mean")
            d[v] = d[v] - d.groupby("sd")[v].transform("mean")
    one = d.copy()
    r = np.corrcoef(one.mangrove_pct, one.water_pct)[0, 1]
    # only cells that hold mangrove at some date
    has = Cc.groupby("sc")["mangrove_pct"].transform("max") > 1
    hh = d[has.loc[d.index]]
    chip_out.append({"sites": label, "chip_dates": len(d), "cells": d["sc"].nunique(), "scenes": d["sd"].nunique(),
                     "r_mangrove_change_vs_water_change": round(r, 3),
                     "slope_mangrove_points_per_water_point": round((one.mangrove_pct * one.water_pct).sum() / (one.water_pct ** 2).sum(), 3),
                     "r_cells_with_mangrove_only": round(np.corrcoef(hh.mangrove_pct, hh.water_pct)[0, 1], 3) if len(hh) > 5 else np.nan})
pd.DataFrame(chip_out).to_csv(out / "chip_level.csv", index=False)

# ------------------------------------------------------------------ print
with pd.option_context("display.width", 300, "display.max_rows", 200, "display.max_columns", 60):
    print("\nSCENE TABLE\n" + T[["site", "date", "cells_on_this_date", "pred_mangrove_ha", "dev_pct", "dev_within_year_pct", "strict_cells",
                                 "strict_dev_pct", "water_pct", "water_swir_rule_pct", "strict_water_pct", "tide_eot20_m",
                                 "sun_elevation", "interior_ndvi", "dark_water_red", "rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw",
                                 "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw"]].to_string(index=False))
    print("\nTEST 1 PER SITE\n" + pd.DataFrame(per_site).to_string(index=False))
    print("\nTEST 1 POOLED\n" + pd.DataFrame(pool).to_string(index=False))
    print("\nLAGS\n" + pd.DataFrame(lag_rows).to_string(index=False))
    print("\nTEST 2\n" + S[["site", "dates_used", "pixels", "flip_px", "strip_px", "flips_in_strip_pct", "flips_in_or_next_to_strip_pct",
                             "flips_within_2px_of_strip_pct", "all_pixels_in_or_next_to_strip_pct", "ever_mangrove_in_or_next_to_strip_pct",
                             "times_chance_all_pixels", "times_chance_ever_mangrove", "shift_null_mean_pct", "shift_null_p97_5_pct",
                             "flip_rate_near_strip_pct", "flip_rate_away_pct", "edge_flip_rate_near_strip_pct", "edge_flip_rate_away_pct",
                             "strip_mangrove_rate_when_water_pct", "strip_mangrove_rate_when_land_pct"]].round(2).to_string(index=False))
    print("\nSEASON MODELS\n" + pd.DataFrame(mods).to_string(index=False))
    print("\nPARTIAL\n" + pd.DataFrame(part).to_string(index=False))
    print("\nCHIP LEVEL\n" + pd.DataFrame(chip_out).to_string(index=False))


# ------------------------------------------------------------------ extra checks on the season question
def perm_partial(d, x, y, ctrl, by="site", n_perm=5000):
    """Shuffle-the-dates test for a partial correlation (x shuffled as in perm_p, the other variables stay with their date)."""
    d = d.dropna(subset=[x, y] + ctrl).sort_values([by, "date"]).reset_index(drop=True)
    dm = demean(d, [x, y] + ctrl, by)
    C = dm[ctrl].values
    H_ = C @ np.linalg.pinv(C)
    ry = dm[y].values - H_ @ dm[y].values

    def pr(xv):
        rx = xv - H_ @ xv
        return np.corrcoef(rx, ry)[0, 1]

    obs = pr(dm[x].values)
    alld = sorted(d["date"].unique())
    idx = {dd: i for i, dd in enumerate(alld)}
    groups = [(g.index.values, np.array([idx[v] for v in g["date"]])) for _, g in d.groupby(by)]
    xv0 = dm[x].values
    hits = 0
    for _ in range(n_perm):
        key = RNG.random(len(alld))
        xp = np.empty(len(d))
        for ix, di in groups:
            xp[ix[np.argsort(key[di])]] = xv0[ix]
        hits += abs(pr(xp)) >= abs(obs) - 1e-12
    return round(float(obs), 3), round((hits + 1) / (n_perm + 1), 4)


def coefs(d, y, xs, by="site"):
    d = d.dropna(subset=[y] + xs)
    dm = demean(d, [y] + xs, by)
    beta = np.linalg.lstsq(dm[xs].values, dm[y].values, rcond=None)[0]
    return {x: (float(b), float(dm[x].std())) for x, b in zip(xs, beta)}


extra = []
for x, ctrl in (("tide_eot20_m", ["sun_elevation"]), ("water_pct", ["sun_elevation"]), ("tide_eot20_m", ["cos_doy", "sin_doy"]),
                ("water_pct", ["cos_doy", "sin_doy"]), ("sun_elevation", ["tide_eot20_m"]), ("sun_elevation", ["water_pct"]),
                ("interior_ndvi", ["tide_eot20_m"]), ("interior_ndvi", ["sun_elevation"]), ("dark_water_red", ["sun_elevation"])):
    r, p = perm_partial(T, x, "dev_pct", ctrl)
    lo = [perm_partial(T[T["site"] != s], x, "dev_pct", ctrl, n_perm=1)[0] for s in SITES]
    extra.append({"check": f"partial r of {x} with dev_pct, holding {' + '.join(ctrl)} fixed", "value": r, "p_shuffle_dates": p,
                  "leave_one_site_out_min": min(lo), "leave_one_site_out_max": max(lo), "scenes": int(T.dropna(subset=[x] + ctrl).shape[0])})
noMay = T[T["date"] != "2020-05-18"]
for x in ("sun_elevation", "interior_ndvi", "dark_water_red", "november"):
    res = pooled(noMay, x, "dev_pct")
    extra.append({"check": f"r of {x} with dev_pct without the 18 May 2020 pass", "value": res["r_within"], "p_shuffle_dates": res["p_shuffle_dates"],
                  "leave_one_site_out_min": res["leave_one_site_out_r_min"], "leave_one_site_out_max": res["leave_one_site_out_r_max"], "scenes": res["scenes"]})
for x, ctrl in (("tide_eot20_m", ["sun_elevation"]), ("water_pct", ["sun_elevation"]), ("tide_eot20_m", ["november"]), ("water_pct", ["november"])):
    r, p = perm_partial(noMay, x, "dev_pct", ctrl)
    extra.append({"check": f"partial r of {x} with dev_pct, holding {' + '.join(ctrl)} fixed, without the 18 May 2020 pass", "value": r,
                  "p_shuffle_dates": p, "scenes": len(noMay)})
for name, xs in (("the time of year (annual wave)", ["cos_doy", "sin_doy"]), ("sun elevation", ["sun_elevation"]),
                 ("dark-water brightness", ["dark_water_red"]), ("tide", ["tide_eot20_m"])):
    R2, adj, n_, _ = r2(T, "interior_ndvi", xs)
    extra.append({"check": f"share of the interior greenness swing explained by {name}", "value": round(R2, 3), "scenes": n_})
for xs in (["tide_eot20_m", "sun_elevation"], ["water_pct", "sun_elevation"], ["tide_eot20_m"], ["water_pct"], ["sun_elevation"]):
    for x, (b, sd) in coefs(T, "dev_pct", xs).items():
        extra.append({"check": f"slope of dev_pct on {x} in the model with {' + '.join(xs)} (percent of area per unit)", "value": round(b, 3),
                      "typical_within_site_spread_of_x": round(sd, 3), "area_swing_for_that_spread_pct": round(abs(b) * sd, 2), "scenes": len(T)})
extra.append({"check": "within-site spread (standard deviation) of dev_pct, all scenes", "value": round(float(demean(T, ["dev_pct"])["dev_pct"].std()), 2), "scenes": len(T)})
for y in ("rf_miss_pct_of_gmw", "rf_false_alarm_pct_of_gmw", "all3_miss_pct_of_gmw", "all3_false_alarm_pct_of_gmw"):
    for x in ("sun_elevation", "dark_water_red", "interior_ndvi"):
        res = pooled(T[T["year"] == 2020], x, y)
        extra.append({"check": f"2020 only: r of {x} with {y}", "value": res["r_within"], "p_shuffle_dates": res["p_shuffle_dates"],
                      "leave_one_site_out_min": res["leave_one_site_out_r_min"], "leave_one_site_out_max": res["leave_one_site_out_r_max"], "scenes": res["scenes"]})
E = pd.DataFrame(extra)
E.to_csv(out / "season_extra.csv", index=False)
with pd.option_context("display.width", 300, "display.max_rows", 200, "display.max_colwidth", 120):
    print("\nEXTRA\n" + E.to_string(index=False))
