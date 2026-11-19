"""Image differences test: are the images comparable between dates, do the differences go with the same-year swings,
and does correcting them remove the swings?

Run tide_test.py first (this reads scenes.csv, tide_scene_table.csv, forests/ and stacks/ from <work>).

  a. stable targets   pixels that should not change, picked by band rules only (no labels, no model output):
                        deep water            water (green > near-infrared) on every date, at least 100 m from any pixel that
                                              is not always water, darker half by average green
                        bright stable ground  never water, NDVI < 0.15 on every date, all 8 neighbours the same, brighter
                                              half by average brightness, steadier half against the other such pixels of its cell
                      Each scene gets its deep-water and bright-ground brightness as a difference from the average of
                      the same site and year.
  b. the link         those differences against the same-year swing (dev_pct of the tide test), with the shuffled-dates test.
  c. the correction   one straight line per band and date (gain and offset) fitted on the stable pixels, each scene
                      corrected to the site's most typical date, the same forests run again, and the same-year pairs redone.
                      The number that counts was fixed before running: per site, the mean over its same-year pairs of
                      |net change as a % of the first date's predicted mangrove|, then the mean of the six sites.

Writes <work>/correction_lines.csv (robustness_test.py needs it) and <out>/summary.csv.
Rows marked "all years" use the older version, against the site's average over all its years, for comparison only.

Usage: python image_differences_test.py --work <work> --out <results>/image_differences
"""
import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage

from shared import BANDS, CHIP, SITES, additive, cell_layout, check_pairs, indices, link, link_together, load_stack, predict

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
K3 = np.ones((3, 3), bool)
PER_CELL = 3000          # stable pixels sampled per chip cell and class for the line fit
MIN_SPREAD = 0.05        # a date needs this much reflectance between its 10th and 90th percentile target to get a gain
rng = np.random.default_rng(0)


def fit_lines(X, cls, iters=40):
    """X[pixel, date] (nan where the pixel's cell has no chip on that date), cls = 0 deep water, 1 bright ground.
    Model x = gain_d * m_pixel + offset_d. The two classes get the same total weight on every date.
    Returns gain[date], offset[date], m[pixel], gain_fitted[date]."""
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
                if spread < MIN_SPREAD:
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
    return gain.copy(), off.copy(), m.copy(), fitted.copy()


# ---------------------------------------------------------------- a. stable targets, and the lines for part c
meas, linerows = [], []
for site in SITES:
    S = load_stack(work, site)
    dates = [str(d) for d in S["dates"]]
    years = np.array([int(d[:4]) for d in dates])
    n = len(dates)
    refl, nd, valid, covered = S["image"][:, :6], S["image"][:, 6], S["valid"], S["covered"]
    H, W = valid.shape[1:]
    cells, present, use, npres = cell_layout(covered, valid)
    cov = covered & use[None]
    with np.errstate(invalid="ignore"):
        water = (refl[:, 1] > refl[:, 3]) & cov
        lowveg = (nd < 0.15) & cov
        negany = np.any(refl <= 0, axis=1) & cov
    R = np.where(cov[:, None], np.nan_to_num(refl), 0.0)
    npr = np.maximum(npres, 1)

    always_water = use & (npres >= 2) & (water.sum(0) == npres)
    deep = always_water & (ndimage.distance_transform_edt(always_water) >= 10)
    mean_green = R[:, 1].sum(0) / npr
    dark = deep & (mean_green <= np.median(mean_green[deep])) if deep.any() else deep

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

    # per-date value of each target: median per cell and date, then the two-way table, made separately for each year
    P2 = present & (present.sum(1) >= 2)[:, None]

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

    def same_year(V, wgt, relative=False):
        """Per date: difference from the mean of the same year (in % of that mean if relative)."""
        dif = np.full(n, np.nan)
        for y in sorted(set(years)):
            sel = years == y
            Py = P2 & sel[None, :]
            Py = Py & (Py.sum(1) >= 2)[:, None] & (wgt > 0)[:, None]
            if sel.sum() < 2 or not Py.any():
                continue
            v = additive(V, Py, wgt)
            mean = np.nanmean(v[sel])
            dif[sel] = 100 * (v[sel] / mean - 1) if relative else v[sel] - mean
        return dif

    def all_years(V, wgt, relative=False):                  # the older version: difference from the site's average over all dates
        v = additive(V, P2 & (wgt > 0)[:, None], wgt)
        return 100 * (v / np.nanmean(v) - 1) if relative else v - np.nanmean(v)

    Vd = cell_medians(dark, refl[:, 2])                                     # deep water, red band
    Vb = [cell_medians(stable, refl[:, b]) for b in range(6)]               # bright ground, every band
    dk, dk_all = same_year(*Vd), all_years(*Vd)
    br = np.array([same_year(*v, relative=True) for v in Vb])
    br_all = np.array([all_years(*v, relative=True) for v in Vb])
    for i, d in enumerate(dates):
        meas.append({"site": site, "date": d, "deep_water_red_diff": round(dk[i], 5), "bright_ground_diff_pct": round(np.mean(br[:, i]), 3),
                     "deep_water_red_diff_all_years": round(dk_all[i], 5), "bright_ground_diff_all_years_pct": round(np.mean(br_all[:, i]), 3),
                     "zero_or_negative_pct": round(100 * negany[i].sum() / max(cov[i].sum(), 1), 3)})

    # the lines: sample stable pixels per cell and class. Half of them are used for the fit.
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
        fits[name] = fit_lines(X[half], cls[half])
    # the typical date: its lines are closest to the median line of the site's dates
    dist = np.zeros(n)
    for name in BANDS:
        g, o, m, _ = fits[name]
        lo, hi = np.quantile(m[m != 0], [0.1, 0.9])
        for anchor in (lo, hi):
            v = g * anchor + o
            dist += np.abs(v - np.median(v)) / max(hi - lo, 1e-6)
    ref = int(np.argmin(dist))
    for name in BANDS:
        g, o, _, fitted = fits[name]
        linerows += [{"site": site, "date": d, "band": name, "typical_date": dates[ref], "gain_offset_gain": round(g[i], 6),
                      "gain_offset_offset": round(o[i], 6), "gain_could_be_fitted": bool(fitted[i])} for i, d in enumerate(dates)]
    print(f"{site}: deep water {int(dark.sum())} px, bright ground {int(stable.sum())} px, typical date {dates[ref]}", flush=True)
pd.DataFrame(linerows).to_csv(work / "correction_lines.csv", index=False)
L = pd.read_csv(work / "correction_lines.csv")                              # read back, so the rounded lines are the ones used

# ---------------------------------------------------------------- b. do the differences go with the same-year swings?
M = pd.DataFrame(meas)
T = pd.read_csv(work / "tide_scene_table.csv").merge(M, on=["site", "date"])
assert len(T) == len(M) == 40
T.to_csv(work / "image_scene_table.csv", index=False)
TWO = ["deep_water_red_diff", "bright_ground_diff_pct"]
TWO_ALL = ["deep_water_red_diff_all_years", "bright_ground_diff_all_years_pct"]
SWING = "share of the same-year swing that goes with "
summary = []


def add(quantity, res, value="share_pct", unit="%"):
    summary.append({"quantity": quantity, "value": res[value], "unit": unit, "scenes": res["scenes"],
                    "chance_shuffled_dates_pct": res.get("chance_shuffled_dates_pct", np.nan)})


have = T.dropna(subset=["dev_pct"])
summary += [{"quantity": "bright stable ground: darkest scene against the average of its site and year", "value": round(have[TWO[1]].min(), 1), "unit": "%"},
            {"quantity": "bright stable ground: brightest scene against the average of its site and year", "value": round(have[TWO[1]].max(), 1), "unit": "%"}]
for _, r in T[T["date"] == "2020-11-17"].iterrows():
    summary.append({"quantity": f"pixels with a band at or below zero on 2020-11-17, {r['site']}", "value": round(r["zero_or_negative_pct"], 1), "unit": "%"})
versions = pd.read_csv(work / "scenes.csv").groupby(["site", "year"])["processing_baseline"].nunique()
summary.append({"quantity": "sites and years whose scenes have more than one processing version", "value": int((versions > 1).sum()), "unit": "site-years"})
add(SWING + "deep water (red band) and bright stable ground together", link_together(T, "dev_pct", TWO))
add(SWING + "deep water (red band) alone", link(T, TWO[0], "dev_pct"))
add(SWING + "bright stable ground alone", link(T, TWO[1], "dev_pct"))
add(SWING + "deep water, bright stable ground and tide height together", link_together(T, "dev_pct", TWO + ["tide_eot20_m"]))
add("deep water (red band) against the height of the sun, correlation", link(have, TWO[0], "sun_elevation", perm=False), "r", "r")
add("bright stable ground against the height of the sun, correlation", link(have, TWO[1], "sun_elevation", perm=False), "r", "r")
add("all years (older version): share of the swing that goes with deep water and bright stable ground together",
    link_together(T, "dev_all_years_pct", TWO_ALL, by="site"))


# ---------------------------------------------------------------- c. correct the images and run the same forests again
def same_year_pairs(site, version, dates, mask, valid, covered):
    years = [int(d[:4]) for d in dates]
    pairs = []
    for i, j in itertools.combinations(range(len(dates)), 2):
        if years[i] != years[j] or int(round((covered[i] & covered[j]).sum() / CHIP ** 2)) == 0:
            continue
        u = valid[i] & valid[j]
        gain, loss, a0 = int((~mask[i] & mask[j] & u).sum()), int((mask[i] & ~mask[j] & u).sum()), int((mask[i] & u).sum())
        pairs.append({"site": site, "version": version, "first_date": dates[i], "second_date": dates[j],
                      "net_pct_of_first": 100 * (gain - loss) / a0 if a0 else np.nan})
    return pairs


pairs, counts = [], []
for site in SITES:
    S = load_stack(work, site)
    dates = [str(d) for d in S["dates"]]
    F = joblib.load(work / "forests" / f"fold{int(S['fold'])}.joblib")
    rf, th = F["rf"], float(F["threshold"])
    assert site in F["held_sites"] and abs(th - float(S["threshold"])) < 1e-6
    pos = list(rf.classes_).index(True)
    refl, valid, covered, gmw = S["image"][:, :6], S["valid"], S["covered"], S["gmw4019"]

    def run(x, v):
        """Forest probability from six bands; NDVI and MNDWI are recomputed as prepare.py does."""
        return predict(rf, pos, np.concatenate([x, *(k[None] for k in indices(x))], axis=0), v)

    assert np.max(np.abs(run(refl[0], valid[0]) - S["prob"][0])) < 1e-6, "forest on disk does not reproduce the stored probabilities"
    ls = L[L["site"] == site]
    G = np.array([[ls[(ls.date == d) & (ls.band == b)]["gain_offset_gain"].iloc[0] for b in BANDS] for d in dates])
    O = np.array([[ls[(ls.date == d) & (ls.band == b)]["gain_offset_offset"].iloc[0] for b in BANDS] for d in dates])
    ref = dates.index(ls["typical_date"].iloc[0])
    corrected = np.zeros_like(valid)
    for i in range(len(dates)):                             # the line that takes date i to the reference date
        gm = G[ref] / G[i]
        om = O[ref] - gm * O[i]
        corrected[i] = (run(refl[i] * gm[:, None, None].astype(np.float32) + om[:, None, None].astype(np.float32), valid[i]) > th) & valid[i]
    for version, mask in (("as_is", (S["prob"] > th) & valid), ("corrected", corrected)):
        pairs += same_year_pairs(site, version, dates, mask, valid, covered)
        lab = (gmw != 255)[None] & valid & np.array([d.startswith("2020") for d in dates])[:, None, None]     # held-out IoU on the 2020 scenes
        counts.append({"site": site, "version": version, "tp": int((mask & (gmw == 1) & lab).sum()), "fp": int((mask & (gmw == 0) & lab).sum()),
                       "fn": int((~mask & (gmw == 1) & lab).sum())})
    print(f"{site}: corrected to {dates[ref]}", flush=True)

P, N = pd.DataFrame(pairs), pd.DataFrame(counts)
check_pairs(P[P.version == "as_is"])
site_swing = P.assign(swing=P["net_pct_of_first"].abs()).groupby(["version", "site"])["swing"].mean()
N["fold"] = N["site"].map(lambda s: "5" if s in ("al_aryam_islands", "al_nouf") else s)      # fold 5 pools its two sites, as the baseline table does
f = N.groupby(["version", "fold"])[["tp", "fp", "fn"]].sum()
iou = (f.tp / (f.tp + f.fp + f.fn)).groupby("version").mean()
for v, name in (("as_is", "images as they are"), ("corrected", "after the correction")):
    summary.append({"quantity": f"average same-year swing per site, {name} (mean of the six sites)", "value": round(site_swing[v].mean(), 2), "unit": "%"})
    summary.append({"quantity": f"the same without Al Nouf, {name}", "value": round(site_swing[v].drop("al_nouf").mean(), 2), "unit": "%"})
    summary.append({"quantity": f"held-out IoU on the non-hazy 2020 chips, {name} (mean of the five folds)", "value": round(iou[v], 4), "unit": "IoU"})
    summary += [{"quantity": f"average same-year swing, {name}, {s}", "value": round(site_swing[v][s], 1), "unit": "%"} for s in SITES]
summary.append({"quantity": "fall in the average same-year swing after the correction (a quarter was needed to count as helping)",
                "value": round(100 * (1 - site_swing["corrected"].mean() / site_swing["as_is"].mean()), 1), "unit": "%"})
R = pd.DataFrame(summary)
R["scenes"] = R["scenes"].astype("Int64")
R.to_csv(out / "summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 100):
    print(R.to_string(index=False))
