"""Part a: how each scene's brightness differs from the other dates of the same site and year on things that should not change,
and the per-date straight lines (gain and offset per band) that make those things agree.

No labels and no model predictions are used. Pixels are picked by band rules only:
    deep water            water (green > near-infrared) on every date, >= 100 m from any pixel that is not always water,
                          darker half by average green
    bright stable ground  never water, NDVI < 0.15 on every date, all 8 neighbours the same, brighter half by average
                          brightness, steadier half by movement against the other such pixels of the chip cell

Outputs in <out>:
    scene_image_measures.csv   one row per scene: deep-water and bright-ground brightness as a difference from the average
                               of the same site and year, share of pixels with a zero or negative band, haze measure,
                               processing version. Columns ending in _all_years hold the old version, the difference
                               from the site's average over all its years, for comparison only.
    scene_band_values.csv      the per-band values behind it
    correction_lines.csv       per site, date and band: gain and offset (gain + offset fit, and offset-only fit),
                               which date is the typical one (the reference), and whether a gain could be fitted
    correction_check.csv       disagreement between dates on stable pixels that were NOT used for the fit, before and after
    stable_targets.csv         how many stable pixels each site has
and in <work>/invariants/<site>.npz the masks of the stable pixels.

Usage: python 03_stable_targets.py --work <work folder> --out <results folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from shared import BANDS, CHIP, SITES, additive, cell_layout, load_stack

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)
(work / "invariants").mkdir(exist_ok=True)
meta = pd.read_csv(out / "scene_meta.csv")
K3 = np.ones((3, 3), bool)
PER_CELL = 3000          # stable pixels sampled per chip cell and class for the line fit
MIN_SPREAD = 0.05        # a date needs this much reflectance between its 10th and 90th percentile target to get a gain
rng = np.random.default_rng(0)


def fit_lines(X, cls, offset_only=False, iters=40):
    """X[pixel, date] (nan where the pixel's cell has no chip on that date), cls = 0 deep water, 1 bright ground.
    Model x = gain_d * m_pixel + offset_d. The two classes get the same total weight on every date.
    Returns gain[date], offset[date], m[pixel], gain_fitted[date], kept[pixel]."""
    P, D = X.shape
    M = np.isfinite(X)
    keep = M.sum(1) >= 2
    Xz = np.where(M, X, 0.0)
    gain, off = np.ones(D), np.zeros(D)
    fitted = np.zeros(D, bool)
    m = np.zeros(P)

    def solve(keep):
        nonlocal gain, off, m, fitted
        Mk = M & keep[:, None]
        gain, off = np.ones(D), np.zeros(D)
        m = np.where(keep, Xz.sum(1) / np.maximum(M.sum(1), 1), 0.0)
        for _ in range(iters):
            for d in range(D):
                sel = Mk[:, d]
                if not sel.any():
                    continue
                w = np.zeros(P)
                for c in (0, 1):
                    s = sel & (cls == c)
                    if s.any():
                        w[s] = 1.0 / s.sum()
                w, md, xd = w[sel], m[sel], Xz[sel, d]
                spread = np.quantile(md, 0.9) - np.quantile(md, 0.1)
                mw, xw = np.average(md, weights=w), np.average(xd, weights=w)
                if offset_only or spread < MIN_SPREAD:
                    gain[d], off[d], fitted[d] = 1.0, xw - mw, False
                else:
                    g = np.sum(w * (md - mw) * (xd - xw)) / np.sum(w * (md - mw) ** 2)
                    gain[d], off[d], fitted[d] = g, xw - g * mw, True
            # fix the free scale and shift: mean gain 1 and mean offset 0 over the dates
            gbar = gain.mean()
            gain, m = gain / gbar, m * gbar
            c = off.mean() / gain.mean()
            m, off = m + c, off - gain * c
            num = (Mk * gain[None, :] * (Xz - off[None, :])).sum(1)
            den = (Mk * gain[None, :] ** 2).sum(1)
            m = np.where(keep, num / np.maximum(den, 1e-12), 0.0)

    solve(keep)
    res = np.where(M & keep[:, None], Xz - (gain[None, :] * m[:, None] + off[None, :]), np.nan)
    for c in (0, 1):                                   # drop pixels that changed (more than 3 robust spreads off on any date)
        r = res[cls == c]
        if np.isfinite(r).any():
            mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
            bad = (np.nanmax(np.abs(np.nan_to_num(res)), axis=1) > 3 * max(mad, 1e-4)) & (cls == c)
            keep = keep & ~bad
    solve(keep)
    return gain.copy(), off.copy(), m.copy(), fitted.copy(), keep


meas, bandrows, linerows, checkrows, targets = [], [], [], [], []
for site in SITES:
    S = load_stack(work, site)
    dates = [str(d) for d in S["dates"]]
    n = len(dates)
    refl, nd, valid, covered = S["image"][:, :6], S["image"][:, 6], S["valid"], S["covered"]
    H, W = valid.shape[1:]
    cells, present, use, npres = cell_layout(covered, valid)
    cov = covered & use[None]
    with np.errstate(invalid="ignore"):
        water = (refl[:, 1] > refl[:, 3]) & cov
        lowveg = (nd < 0.15) & cov
        negany = np.any(refl <= 0, axis=1) & cov
        negband = (refl <= 0) & cov[:, None]
    R = np.where(cov[:, None], np.nan_to_num(refl), 0.0)
    npr = np.maximum(npres, 1)

    # ---- deep water
    always_water = use & (npres >= 2) & (water.sum(0) == npres)
    dist = ndimage.distance_transform_edt(always_water)
    deep = always_water & (dist >= 10)
    mean_green = R[:, 1].sum(0) / npr
    dark = deep & (mean_green <= np.median(mean_green[deep])) if deep.any() else deep

    # ---- bright stable ground
    cand = ndimage.binary_erosion(use & (npres >= 2) & (water.sum(0) == 0) & (lowveg.sum(0) == npres), K3)
    bright_d = R.mean(1)                                    # [date, H, W] mean of the six bands
    mean_bright = bright_d.sum(0) / npr
    brighter = cand & (mean_bright >= np.median(mean_bright[cand])) if cand.any() else cand
    cv = np.full((H, W), np.nan)
    for (r, c), p in zip(cells, present):
        sl = (slice(r, r + CHIP), slice(c, c + CHIP))
        b = brighter[sl]
        if p.sum() < 2 or b.sum() < 100:
            continue
        v = bright_d[p][(slice(None),) + sl][:, b]          # [dates of the cell, pixels]
        ratio = v / np.median(v, axis=1, keepdims=True)
        tmp = np.full(b.shape, np.nan)
        tmp[b] = ratio.std(0) / ratio.mean(0)
        cv[sl] = tmp
    ok = np.isfinite(cv)
    stable = ok & (cv <= np.nanmedian(cv[ok])) if ok.any() else ok
    np.savez_compressed(work / "invariants" / f"{site}.npz", deep_water=dark, bright_ground=stable)
    targets.append({"site": site, "fixed_pixels": int(use.sum()), "deep_water_px": int(dark.sum()), "bright_ground_px": int(stable.sum()),
                    "cells_with_deep_water": int(sum(dark[r:r + CHIP, c:c + CHIP].sum() >= 50 for r, c in cells)),
                    "cells_with_bright_ground": int(sum(stable[r:r + CHIP, c:c + CHIP].sum() >= 50 for r, c in cells)),
                    "cells": int((present.sum(1) >= 2).sum())})

    # ---- part a: per-date value of each target, built from the chip cells (median per cell and date, then the two-way table).
    # The table is made separately for each year, so a date is only compared with the other dates of the same site and year.
    P2 = present & (present.sum(1) >= 2)[:, None]
    years = np.array([int(d[:4]) for d in dates])

    def cell_medians(mask, arr):                            # arr [date, H, W] -> median per cell and date, and the cell weights
        V = np.full((len(cells), n), np.nan)
        wgt = np.zeros(len(cells))
        for ci, (r, c) in enumerate(cells):
            mm = mask[r:r + CHIP, c:c + CHIP]
            if mm.sum() < 50:
                continue
            wgt[ci] = mm.sum()
            for i in np.where(P2[ci])[0]:
                V[ci, i] = np.median(arr[i, r:r + CHIP, c:c + CHIP][mm])
        return np.nan_to_num(V), wgt

    def cell_shares(flag):                                  # flag [date, H, W] bool -> percent of the cell's fixed pixels
        npx = use.reshape(H // CHIP, CHIP, W // CHIP, CHIP).sum((1, 3)).ravel().astype(float)
        cnt = flag.reshape(n, H // CHIP, CHIP, W // CHIP, CHIP).sum((2, 4)).reshape(n, -1).T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(100 * cnt / npx[:, None]), npx

    def same_year(V, wgt, relative=False):
        """Per date: the value built from the cells of the same year, and its difference from the mean of that year
        (in percent of that mean if relative)."""
        val, dif = np.full(n, np.nan), np.full(n, np.nan)
        for y in sorted(set(years)):
            sel = years == y
            Py = P2 & sel[None, :]
            Py = Py & (Py.sum(1) >= 2)[:, None] & (wgt > 0)[:, None]
            if sel.sum() < 2 or not Py.any():
                continue
            v = additive(V, Py, wgt)
            mean = np.nanmean(v[sel])
            val[sel] = v[sel]
            dif[sel] = 100 * (v[sel] / mean - 1) if relative else v[sel] - mean
        return val, dif

    def all_years(V, wgt, relative=False):                  # the old version: difference from the site's average over all dates
        v = additive(V, P2 & (wgt > 0)[:, None], wgt)
        return 100 * (v / np.nanmean(v) - 1) if relative else v - np.nanmean(v)

    Vd = [cell_medians(dark, refl[:, b]) for b in range(6)]
    Vb = [cell_medians(stable, refl[:, b]) for b in range(6)]
    Vn = cell_shares(negany)
    dk, dk_dev = (np.array(x) for x in zip(*[same_year(*v) for v in Vd]))                       # [band, date]
    br, br_rel = (np.array(x) for x in zip(*[same_year(*v, relative=True) for v in Vb]))
    _, neg_dif = same_year(*Vn)
    negb = np.array([same_year(*cell_shares(negband[:, b]))[1] for b in range(6)])
    dk_all = np.array([all_years(*v) for v in Vd])
    br_all = np.array([all_years(*v, relative=True) for v in Vb])
    neg_all = all_years(*Vn)
    for i, d in enumerate(dates):
        mrow = meta[(meta.site == site) & (meta.date == d)].iloc[0]
        meas.append({"site": site, "date": d,
                     "deep_water_red_diff": round(dk_dev[2, i], 5), "deep_water_visible_diff": round(np.mean(dk_dev[:3, i]), 5),
                     "deep_water_infrared_diff": round(np.mean(dk_dev[3:, i]), 5),
                     "bright_ground_diff_pct": round(np.mean(br_rel[:, i]), 3), "bright_ground_red_diff_pct": round(br_rel[2, i], 3),
                     "bright_ground_nir_diff_pct": round(br_rel[3, i], 3),
                     "zero_or_negative_diff_points": round(neg_dif[i], 3),
                     "deep_water_red_diff_all_years": round(dk_all[2, i], 5), "bright_ground_diff_all_years_pct": round(np.mean(br_all[:, i]), 3),
                     "zero_or_negative_diff_all_years_points": round(neg_all[i], 3),
                     "zero_or_negative_raw_pct": round(100 * negany[i].sum() / max(cov[i].sum(), 1), 3),
                     "haze_measure": mrow["haze_measure"],
                     "processing_baseline": mrow["processing_baseline"], "sun_elevation": mrow["sun_elevation"], "view_azimuth": mrow["view_azimuth"]})
        for b, name in enumerate(BANDS):
            bandrows.append({"site": site, "date": d, "band": name, "deep_water": round(dk[b, i], 5), "deep_water_diff": round(dk_dev[b, i], 5),
                             "bright_ground": round(br[b, i], 5), "bright_ground_diff_pct": round(br_rel[b, i], 3),
                             "zero_or_negative_diff_points": round(negb[b, i], 3)})

    # ---- the lines: sample stable pixels per cell and class, half for the fit and half kept aside to check it
    idx, cls = [], []
    for (r, c), p in zip(cells, present):
        if p.sum() < 2:
            continue
        for k, mask in enumerate((dark, stable)):
            yy, xx = np.nonzero(mask[r:r + CHIP, c:c + CHIP])
            if len(yy) == 0:
                continue
            pick = rng.choice(len(yy), min(len(yy), 2 * PER_CELL), replace=False)
            idx.append(np.column_stack([yy[pick] + r, xx[pick] + c]))
            cls.append(np.full(len(pick), k))
    idx, cls = np.concatenate(idx), np.concatenate(cls)
    half = rng.random(len(idx)) < 0.5
    covp = cov[:, idx[:, 0], idx[:, 1]].T                                   # [pixel, date]
    fits = {}
    for b, name in enumerate(BANDS):
        X = np.where(covp, refl[:, b, idx[:, 0], idx[:, 1]].T.astype(float), np.nan)
        for kind in ("gain_offset", "offset_only"):
            if kind == "gain_offset":
                g, o, m, fitted, kept = fit_lines(X[half], cls[half])
            else:
                # offset only = dark-object subtraction: the offset comes from deep water alone.
                # A date with no deep water in its chips gets no offset and is flagged.
                Xd = X[half & (cls == 0)]
                g, o, m, _, kept = fit_lines(Xd, np.zeros(len(Xd), int), offset_only=True)
                fitted = np.isfinite(Xd).any(axis=0)
                o = np.where(fitted, o, 0.0)
            fits[(name, kind)] = (g, o, m, fitted)
            # check on the pixels kept aside: spread between dates of the same pixel, before and after
            Xc = X[~half]
            for c, cname in ((0, "deep water"), (1, "bright ground")):
                xc = Xc[cls[~half] == c]
                xc = xc[np.isfinite(xc).sum(1) >= 2]
                corr = (xc - o[None, :]) / g[None, :]
                checkrows.append({"site": site, "band": name, "fit": kind, "target": cname, "pixels": len(xc),
                                  "spread_between_dates_before": round(float(np.nanmean(np.nanstd(xc, axis=1))), 5),
                                  "spread_between_dates_after": round(float(np.nanmean(np.nanstd(corr, axis=1))), 5)})
    # ---- the typical date: its lines are closest to the median line of the site's dates
    dist = np.zeros(n)
    for name in BANDS:
        g, o, m, _ = fits[(name, "gain_offset")]
        lo, hi = np.quantile(m[m != 0], [0.1, 0.9])
        for anchor in (lo, hi):
            v = g * anchor + o
            dist += np.abs(v - np.median(v)) / max(hi - lo, 1e-6)
    ref = int(np.argmin(dist))
    for b, name in enumerate(BANDS):
        for i, d in enumerate(dates):
            row = {"site": site, "date": d, "band": name, "typical_date": dates[ref], "is_typical_date": i == ref,
                   "distance_from_median_line": round(dist[i] / 12, 4)}
            for kind in ("gain_offset", "offset_only"):
                g, o, _, fitted = fits[(name, kind)]
                row[f"{kind}_gain"], row[f"{kind}_offset"] = round(g[i], 6), round(o[i], 6)
                row["gain_could_be_fitted" if kind == "gain_offset" else "offset_only_available"] = bool(fitted[i])
            linerows.append(row)
    print(f"{site}: deep water {int(dark.sum())} px, bright ground {int(stable.sum())} px, typical date {dates[ref]}, "
          f"gain fitted on {int(np.mean([fits[(b, 'gain_offset')][3].sum() for b in BANDS]))} of {n} dates", flush=True)

pd.DataFrame(meas).to_csv(out / "scene_image_measures.csv", index=False)
pd.DataFrame(bandrows).to_csv(out / "scene_band_values.csv", index=False)
pd.DataFrame(linerows).to_csv(out / "correction_lines.csv", index=False)
C = pd.DataFrame(checkrows)
C.to_csv(out / "correction_check.csv", index=False)
pd.DataFrame(targets).to_csv(out / "stable_targets.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 300):
    print(pd.DataFrame(targets).to_string(index=False))
    print(pd.DataFrame(meas).to_string(index=False))
    print(C.groupby(["fit", "target"])[["spread_between_dates_before", "spread_between_dates_after"]].mean().round(5))
    L = pd.DataFrame(linerows)
    print(L[L.band.isin(["B04", "B08"])][["site", "date", "band", "is_typical_date", "gain_offset_gain", "gain_offset_offset",
                                           "offset_only_offset", "gain_could_be_fitted", "offset_only_available"]].to_string(index=False))
