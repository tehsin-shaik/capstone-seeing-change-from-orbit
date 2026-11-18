"""Part c: correct each scene so its stable targets match the reference date of the site, run the same
held-out forests on the corrected images, and rerun the same-year check.

The number that counts (fixed before this was run): per site, the mean over its same-year pairs of
|net change as a percent of the first date's predicted mangrove|, and the mean of the six site numbers.
Second number: held-out IoU against GMW v4.0.19 on the non-hazy 2020 chips.

Versions
    as_is                 the images as they are (must reproduce the committed same-year table)
    gain_offset           gain and offset per band, reference = the typical date of the site      <- main result
    offset_only           dark-water offset per band, same reference                              (check)
    gain_offset, every other date as the reference                                                (check)

Outputs in <out>: same_year_pairs.csv, site_swing.csv, summary.csv, reference_choice.csv, scene_after_correction.csv
(dev_pct in scene_after_correction.csv is the scene's swing against the other dates of the same site and year)

Usage: python 05_corrected_check.py --work <work> --out <results> --committed <repo>/results/change_detection/all_sites/noise_floor_all_pairs.csv
"""
import argparse
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shared import BANDS, CHIP, SITES, cell_layout, indices, ipf, load_stack

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--committed", required=True)
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)
L = pd.read_csv(out / "correction_lines.csv")


def predict(rf, pos, refl, valid):
    """Forest probability for one scene from its six bands [6, H, W]; NDVI and MNDWI are recomputed as prepare.py does."""
    nd, mn = indices(refl)
    feats = np.concatenate([refl, nd[None], mn[None]], axis=0)
    X = np.nan_to_num(feats[:, valid].T.astype(np.float32), nan=0.0)
    p = np.zeros(valid.shape, np.float32)
    if valid.any():
        p[valid] = rf.predict_proba(X)[:, pos]
    return p


def score(site, version, ref, dates, mask, valid, covered, gmw, layout):
    """Same-year pairs, 2020 IoU counts, per-scene deviation and same-year flip share for one set of masks."""
    years = np.array([int(d[:4]) for d in dates])
    pairs = []
    for i, j in itertools.combinations(range(len(dates)), 2):
        if years[i] != years[j]:
            continue
        chips = int(round((covered[i] & covered[j]).sum() / CHIP ** 2))
        if chips == 0:
            continue
        u = valid[i] & valid[j]
        gain = int((~mask[i] & mask[j] & u).sum()); loss = int((mask[i] & ~mask[j] & u).sum())
        a0 = int((mask[i] & u).sum())
        pairs.append({"site": site, "version": version, "reference": ref, "year": int(years[i]), "first_date": dates[i], "second_date": dates[j],
                      "chips_compared": chips, "area_first_ha": round(a0 * 0.01, 2),
                      "net_pct_of_first": 100 * (gain - loss) / a0 if a0 else np.nan,
                      "changed_pct_of_first": 100 * (gain + loss) / a0 if a0 else np.nan})
    lab = gmw != 255
    tp = fp = fn = 0
    scenes = []
    cells, present, use, npres = layout
    cov = covered & use[None]
    H, W = use.shape
    m = mask & cov
    Y = m.reshape(len(dates), H // CHIP, CHIP, W // CHIP, CHIP).sum((2, 4)).reshape(len(dates), -1).T
    npx = use.reshape(H // CHIP, CHIP, W // CHIP, CHIP).sum((1, 3)).ravel()
    # the swing of a scene: its predicted area against the average of the dates of the same site and the same year
    Pall = present & (present.sum(1) >= 2)[:, None] & (npx > 0)[:, None]
    g = np.full(len(dates), np.nan)
    for y in sorted(set(years)):
        sel = years == y
        Py = Pall & sel[None, :]
        Py = Py & (Py.sum(1) >= 2)[:, None]
        if sel.sum() >= 2 and Py.any():
            g[sel] = ipf(Y, Py)[sel]
    for i, d in enumerate(dates):
        row = {"site": site, "version": version, "reference": ref, "date": d, "dev_pct": round(100 * (g[i] - 1), 2)}
        if years[i] == 2020:
            u = valid[i] & lab
            t, f, n_ = int((mask[i] & (gmw == 1) & u).sum()), int((mask[i] & (gmw == 0) & u).sum()), int((~mask[i] & (gmw == 1) & u).sum())
            tp, fp, fn = tp + t, fp + f, fn + n_
            row.update({"miss_pct_of_gmw": round(100 * n_ / max(t + n_, 1), 2), "false_alarm_pct_of_gmw": round(100 * f / max(t + n_, 1), 2),
                        "iou": round(t / max(t + f + n_, 1), 4)})
        scenes.append(row)
    # same-year flips: of the pixels called mangrove on at least one date of a year, the share not called so on all of them
    ever = flip = 0
    for y in sorted(set(years)):
        sel = years == y
        k = cov[sel].sum(0)
        ok = use & (k >= 2)
        ms = m[sel].sum(0)
        ever += int((ok & (ms > 0)).sum()); flip += int((ok & (ms > 0) & (ms < k)).sum())
    return pairs, scenes, {"tp": tp, "fp": fp, "fn": fn, "flip_share_pct": 100 * flip / max(ever, 1), "dev_sd": float(np.nanstd(100 * (g - 1), ddof=1)) if np.isfinite(g).sum() > 1 else np.nan}


pairs_all, scenes_all, site_rows = [], [], []
for site in SITES:
    S = load_stack(work, site)
    dates = [str(d) for d in S["dates"]]
    n = len(dates)
    F = joblib.load(work / "forests" / f"fold{int(S['fold'])}.joblib")
    rf, th = F["rf"], float(F["threshold"])
    assert site in F["held_sites"] and abs(th - float(S["threshold"])) < 1e-6
    pos = list(rf.classes_).index(True)
    refl, valid, covered, gmw = S["image"][:, :6], S["valid"], S["covered"], S["gmw4019"]
    layout = cell_layout(covered, valid)
    ls = L[L["site"] == site]
    G = np.array([[ls[(ls.date == d) & (ls.band == b)]["gain_offset_gain"].iloc[0] for b in BANDS] for d in dates])
    O = np.array([[ls[(ls.date == d) & (ls.band == b)]["gain_offset_offset"].iloc[0] for b in BANDS] for d in dates])
    Oo = np.array([[ls[(ls.date == d) & (ls.band == b)]["offset_only_offset"].iloc[0] for b in BANDS] for d in dates])
    Av = np.array([[bool(ls[(ls.date == d) & (ls.band == b)]["offset_only_available"].iloc[0]) for b in BANDS] for d in dates])
    typical = dates.index(ls["typical_date"].iloc[0])

    # the images as they are: the stored probabilities came from this same forest; predict one scene again to be sure
    p0 = predict(rf, pos, refl[0], valid[0])
    assert np.max(np.abs(p0 - S["prob"][0])) < 1e-6, "forest on disk does not reproduce the stored probabilities"
    mask0 = (S["prob"] > th) & valid

    def run(version, ref, masks):
        pr, sc, cnt = score(site, version, dates[ref] if ref is not None else "", dates, masks, valid, covered, gmw, layout)
        for row in pr + sc:                       # main = the images as they are, or corrected to the site's typical date
            row["main"] = ref is None or ref == typical
        pairs_all.extend(pr); scenes_all.extend(sc)
        ab = np.abs([p["net_pct_of_first"] for p in pr])
        site_rows.append({"site": site, "version": version, "reference": dates[ref] if ref is not None else "",
                          "is_typical_reference": ref == typical, "same_year_pairs": len(pr),
                          "mean_abs_swing_pct": float(np.mean(ab)), "median_abs_swing_pct": float(np.median(ab)), "max_abs_swing_pct": float(np.max(ab)),
                          "iou_2020": cnt["tp"] / max(cnt["tp"] + cnt["fp"] + cnt["fn"], 1), **cnt})

    def corrected(ref, kind):
        masks = np.zeros_like(mask0)
        for i in range(n):
            if kind == "gain_offset":
                gm = G[ref] / G[i]
                om = O[ref] - gm * O[i]
            else:
                ok = Av[i] & Av[ref]
                gm = np.ones(6)
                om = np.where(ok, Oo[ref] - Oo[i], 0.0)
            x = refl[i] * gm[:, None, None].astype(np.float32) + om[:, None, None].astype(np.float32)
            masks[i] = (predict(rf, pos, x, valid[i]) > th) & valid[i]
        return masks

    run("as_is", None, mask0)
    run("gain_offset", typical, corrected(typical, "gain_offset"))
    run("offset_only", typical, corrected(typical, "offset_only"))
    for r in range(n):
        if r != typical:
            run("gain_offset", r, corrected(r, "gain_offset"))
    print(f"{site}: done ({n} reference choices)", flush=True)

P = pd.DataFrame(pairs_all)
SR = pd.DataFrame(site_rows)
SC = pd.DataFrame(scenes_all)

# ---- the images as they are must reproduce the committed same-year table
com = pd.read_csv(a.committed)
chk = P[P.version == "as_is"].merge(com, on=["site", "first_date", "second_date"], suffixes=("", "_committed"))
assert len(chk) == len(com) == (P.version == "as_is").sum(), "same-year pairs differ from the committed table"
worst = float((chk["net_pct_of_first"].round(1) - chk["net_pct_of_first_committed"]).abs().max())
print(f"as-is same-year pairs against the committed table: {len(chk)} pairs, largest difference {worst:.2f} points")
assert worst < 0.051

P.round(3).to_csv(out / "same_year_pairs.csv", index=False)
SC.to_csv(out / "scene_after_correction.csv", index=False)

# ---- per site: as is, main result, offset only
main = SR[(SR.version == "as_is") | SR.is_typical_reference].copy()
main.round(4).to_csv(out / "site_swing.csv", index=False)


def fold_iou(d):
    """Mean over the five folds; fold 5 pools Al Aryam and Al Nouf pixel-wise, as the baseline table does."""
    d = d.assign(fold=d["site"].map(lambda s: "5" if s in ("al_aryam_islands", "al_nouf") else s))
    f = d.groupby("fold")[["tp", "fp", "fn"]].sum()
    return float((f.tp / (f.tp + f.fp + f.fn)).mean())


head = []
for v in ("as_is", "gain_offset", "offset_only"):
    d = main[main.version == v]
    head.append({"version": v, "six_site_mean_abs_swing_pct": d.mean_abs_swing_pct.mean(), "six_site_mean_of_worst_swing_pct": d.max_abs_swing_pct.mean(),
                 "all_pairs_mean_abs_swing_pct": P[(P.version == v) & P.main]["net_pct_of_first"].abs().mean(),
                 "five_site_mean_abs_swing_pct_without_al_nouf": d[d.site != "al_nouf"].mean_abs_swing_pct.mean(),
                 "iou_mean_over_5_folds": fold_iou(d), "mean_site_spread_of_same_year_swing_pct": d.dev_sd.mean(),
                 "mean_same_year_flip_share_pct": d.flip_share_pct.mean()})
Hd = pd.DataFrame(head)
Hd.round(4).to_csv(out / "summary.csv", index=False)

# ---- does the choice of reference date matter?
go = SR[SR.version == "gain_offset"]
base = SR[SR.version == "as_is"].set_index("site")
ref = go.groupby("site").agg(reference_choices=("reference", "size"), swing_min=("mean_abs_swing_pct", "min"), swing_median=("mean_abs_swing_pct", "median"),
                             swing_max=("mean_abs_swing_pct", "max"), iou_min=("iou_2020", "min"), iou_median=("iou_2020", "median"),
                             iou_max=("iou_2020", "max")).reset_index()
ref["swing_as_is"] = ref.site.map(base.mean_abs_swing_pct)
ref["iou_as_is"] = ref.site.map(base.iou_2020)
ref["swing_typical_reference"] = ref.site.map(go[go.is_typical_reference].set_index("site").mean_abs_swing_pct)
ref["choices_that_shrink_the_swing"] = ref.site.map(lambda s: int((go[go.site == s].mean_abs_swing_pct < base.loc[s, "mean_abs_swing_pct"]).sum()))
ref.round(4).to_csv(out / "reference_choice.csv", index=False)
go.round(4).to_csv(out / "reference_choice_all.csv", index=False)

with pd.option_context("display.width", 250, "display.max_rows", 300, "display.float_format", "{:.3f}".format):
    print("\nPER SITE\n" + main[["site", "version", "reference", "same_year_pairs", "mean_abs_swing_pct", "median_abs_swing_pct", "max_abs_swing_pct",
                                  "iou_2020", "dev_sd", "flip_share_pct"]].to_string(index=False))
    print("\nHEADLINE\n" + Hd.to_string(index=False))
    print("\nREFERENCE CHOICE\n" + ref.to_string(index=False))
    pv = P[P.main].pivot_table(index=["site", "first_date", "second_date"], columns="version", values="net_pct_of_first")
    print("\nPAIRS\n" + pv[["as_is", "gain_offset", "offset_only"]].to_string())
    s20 = SC[SC.date.str.startswith("2020") & SC.main].pivot_table(index=["site", "date"], columns="version",
                                                                                              values=["miss_pct_of_gmw", "false_alarm_pct_of_gmw", "iou"])
    print("\n2020 SCENES\n" + s20.to_string())
