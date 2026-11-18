"""Score the four versions: sensitivity of A to a brightness change, the same-year swing and the held-out IoU.

    A  current forest                      (forests of the image differences test, same as the committed results)
    B  trained with brightness augmentation
    C  trained on NDVI and MNDWI only
    D  forest of A, area = sum of the predicted probabilities

Fixed beforehand: the number that counts is the mean over a site's same-year pairs of
|net change as a percent of the first date's predicted mangrove|, and the mean of the six sites. "Helps" = that mean
falls by at least a quarter against A with IoU (all held-out 2020 chips, mean of five folds) no more than 0.01 lower.

Outputs in <out>:
    sensitivity_A.csv                per site and brightness change: change in thresholded mangrove area (percent)
    sensitivity_all_versions.csv     the same for A (hard and probability-sum), B and C; B and C on every 2nd row and column
    same_year_pairs.csv              every same-year pair, every version
    site_table.csv                   per site and version: swing and IoU
    summary.csv                     six-site mean swing, IoU, and the verdict by the rule

Usage: python 02_score_versions.py --train-val <Train_Val_Dataset> --work <work folder> --images-work <image_differences work folder>
           --out <results> --committed-pairs <repo>/results/change_detection/all_sites/noise_floor_all_pairs.csv
"""
import argparse
import itertools
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=str(Path(__file__).resolve().parents[2]), help="folder that holds common.py (default: src/change_detection)")
ap.add_argument("--train-val", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--images-work", required=True, help="work folder of the image differences test (stacks/ and forests/)")
ap.add_argument("--out", required=True)
ap.add_argument("--committed-pairs", required=True)
a = ap.parse_args()
sys.path.insert(0, a.code)
import common as C  # noqa: E402

tv = Path(a.train_val)
if "Train_Val_Dataset" not in str(tv):
    raise SystemExit("only the Train_Val_Dataset folder may be used")
work, hwork, out = Path(a.work), Path(a.images_work), Path(a.out)
CHIP, EPS = 256, 0.01
SITES = ["al_aryam_islands", "al_nouf", "eastern_mangroves", "ghantoot", "jubail_island", "ras_ghurab"]
SCALES = [0.90, 0.95, 0.98, 1.02, 1.05, 1.10]
OFFSETS = [-0.02, -0.01, 0.01, 0.02]
COMMITTED_IOU = {1: 0.8206, 2: 0.5616, 3: 0.6966, 4: 0.7774, 5: 0.6631}

cat = C.load_full_catalog(tv)
assert set(cat["role"]) == {"cv"}
FOLD = {s: int(cat[cat.site == s]["fold"].iloc[0]) for s in SITES}


def forest(version, site):
    k = FOLD[site]
    F = joblib.load(hwork / "forests" / f"fold{k}.joblib") if version == "A" else joblib.load(work / "forests" / f"{version}_fold{k}.joblib")
    assert site in F["held_sites"]
    return F["rf"], float(F["threshold"]), list(F["rf"].classes_).index(True)


def with_indices(refl):
    """Eight channels from six bands, NDVI and MNDWI as in prepare.py."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.clip((refl[3] - refl[2]) / np.maximum(refl[3] + refl[2], EPS), -1, 1)
        mndwi = np.clip((refl[1] - refl[4]) / np.maximum(refl[1] + refl[4], EPS), -1, 1)
    return np.concatenate([refl, ndvi[None], mndwi[None]], axis=0)


def predict(rf, pos, version, feats, valid):
    """Probability map for one image [8, H, W]. Version C sees channels 6 and 7 only."""
    X = np.nan_to_num(feats[:, valid].T.astype(np.float32), nan=0.0)
    if version == "C":
        X = X[:, 6:8]
    p = np.zeros(valid.shape, np.float32)
    if valid.any():
        p[valid] = rf.predict_proba(X)[:, pos]
    return p


pairs, sens, ious = [], [], []
for site in SITES:
    z = np.load(hwork / "stacks" / f"{site}.npz", allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    years = np.array([int(d[:4]) for d in dates])
    n = len(dates)
    img, valid, covered = z["image"], z["valid"], z["covered"]
    probs, ths = {}, {}
    for v in ("A", "B", "C"):
        rf, th, pos = forest(v, site)
        ths[v] = th
        if v == "A":
            probs[v] = z["prob"].astype(np.float32)
            chk = predict(rf, pos, v, img[0], valid[0])
            assert np.max(np.abs(chk - probs[v][0])) < 1e-6, "forest A does not reproduce the stored probabilities"
        else:
            probs[v] = np.stack([predict(rf, pos, v, img[i], valid[i]) for i in range(n)])

    # ---- same-year pairs: hard masks for A, B, C; probability sums for D (forest of A), and for B and C as a side look
    for i, j in itertools.combinations(range(n), 2):
        if years[i] != years[j]:
            continue
        chips = int(round((covered[i] & covered[j]).sum() / CHIP ** 2))
        if chips == 0:
            continue
        u = valid[i] & valid[j]
        base = {"site": site, "year": int(years[i]), "first_date": dates[i], "second_date": dates[j], "chips_compared": chips}
        for v in ("A", "B", "C"):
            m0, m1 = (probs[v][i] > ths[v]) & u, (probs[v][j] > ths[v]) & u
            a0, a1 = int(m0.sum()), int(m1.sum())
            pairs.append({**base, "version": v, "area_first_ha": round(a0 * 0.01, 2), "net_pct_of_first": 100 * (a1 - a0) / a0 if a0 else np.nan})
            s0, s1 = float(probs[v][i][u].sum()), float(probs[v][j][u].sum())
            pairs.append({**base, "version": {"A": "D", "B": "B with probability sum", "C": "C with probability sum"}[v],
                          "area_first_ha": round(s0 * 0.01, 2), "net_pct_of_first": 100 * (s1 - s0) / s0 if s0 else np.nan})

    # ---- held-out IoU against GMW v4.0.19 on every 2020 chip of the site (hazy ones flagged)
    held = cat[(cat.site == site) & (cat.year == 2020)]
    for v in ("A", "B", "C"):
        rf, th, pos = forest(v, site)
        for _, r in held.iterrows():
            chip = C.load_chip(C.chip_path(tv, r))
            vv = chip["valid"]
            mask = (predict(rf, pos, v, chip["image"], vv) > th) & vv
            truth, labelled = C.get_label(chip, C.LABEL)
            use = vv & labelled
            ious.append({"site": site, "fold": FOLD[site], "version": v, "date": r["date"], "hazy": bool(r["hazy"]),
                         "tp": int((mask & truth & use).sum()), "fp": int((mask & ~truth & use).sum()), "fn": int((~mask & truth & use).sum())})

    # ---- sensitivity: change the brightness of every non-hazy image and see how far the predicted area moves
    refl = img[:, :6]
    for v in ("A", "B", "C"):
        rf, th, pos = forest(v, site)
        step = 1 if v == "A" else 2                      # B and C on every second row and column
        val = valid[:, ::step, ::step]
        base_p = probs[v][:, ::step, ::step]
        base_hard = np.array([((base_p[i] > th) & val[i]).sum() for i in range(n)], float)
        base_soft = np.array([base_p[i][val[i]].sum() for i in range(n)], float)
        for kind, amounts in (("scale", SCALES), ("offset", OFFSETS)):
            for amt in amounts:
                hard, soft = np.zeros(n), np.zeros(n)
                for i in range(n):
                    x = refl[i][:, ::step, ::step]
                    x = x * np.float32(amt) if kind == "scale" else x + np.float32(amt)
                    p = predict(rf, pos, v, with_indices(x), val[i])
                    hard[i], soft[i] = ((p > th) & val[i]).sum(), p[val[i]].sum()
                for name, b, c in (("thresholded area", base_hard, hard), ("probability sum", base_soft, soft)):
                    per = 100 * (c / np.where(b > 0, b, np.nan) - 1)
                    sens.append({"site": site, "version": v, "area": name, "change": kind,
                                 "amount": round(100 * (amt - 1), 1) if kind == "scale" else amt,
                                 "area_change_pct": 100 * (c.sum() / b.sum() - 1), "scene_min_pct": float(np.nanmin(per)),
                                 "scene_max_pct": float(np.nanmax(per)), "scenes": n})
    print(f"{site}: done", flush=True)

P = pd.DataFrame(pairs)
Sn = pd.DataFrame(sens)
I = pd.DataFrame(ious)

# ---- checks on A: the committed same-year table and the committed IoU
com = pd.read_csv(a.committed_pairs)
chk = P[P.version == "A"].merge(com, on=["site", "first_date", "second_date"], suffixes=("", "_committed"))
assert len(chk) == len(com) == (P.version == "A").sum()
worst = float((chk["net_pct_of_first"].round(1) - chk["net_pct_of_first_committed"]).abs().max())
print(f"A against the committed same-year table: {len(chk)} pairs, largest difference {worst:.2f} points")
assert worst < 0.051


def fold_iou(d):
    f = d.groupby("fold")[["tp", "fp", "fn"]].sum()
    return f.tp / (f.tp + f.fp + f.fn)


fa = fold_iou(I[I.version == "A"])
for k, vcom in COMMITTED_IOU.items():
    assert abs(round(float(fa[k]), 4) - vcom) < 1e-4, f"fold {k}: A gives {fa[k]:.4f}, committed {vcom}"
print("A against the committed held-out IoU per fold: identical to four decimals")

P.round(3).to_csv(out / "same_year_pairs.csv", index=False)
Sn.round(3).to_csv(out / "sensitivity_all_versions.csv", index=False)
SA = Sn[(Sn.version == "A") & (Sn["area"] == "thresholded area")].drop(columns=["version", "area"])
SA.round(2).to_csv(out / "sensitivity_A.csv", index=False)

# ---- per site and version
rows = []
for v in ("A", "B", "C", "D", "B with probability sum", "C with probability sum"):
    iv = {"D": "A", "B with probability sum": "B", "C with probability sum": "C"}.get(v, v)
    for site in SITES:
        pv = P[(P.version == v) & (P.site == site)]["net_pct_of_first"].abs()
        d = I[(I.version == iv) & (I.site == site)]
        nh = d[~d.hazy]
        rows.append({"version": v, "site": site, "same_year_pairs": len(pv), "mean_abs_swing_pct": pv.mean(), "median_abs_swing_pct": pv.median(),
                     "max_abs_swing_pct": pv.max(), "iou_all_2020_chips": d.tp.sum() / (d.tp.sum() + d.fp.sum() + d.fn.sum()),
                     "iou_non_hazy_2020_chips": nh.tp.sum() / (nh.tp.sum() + nh.fp.sum() + nh.fn.sum())})
ST = pd.DataFrame(rows)
ST.round(4).to_csv(out / "site_table.csv", index=False)

head = []
A_sw = ST[ST.version == "A"].mean_abs_swing_pct.mean()
A_iou = float(fold_iou(I[I.version == "A"]).mean())
for v in ("A", "B", "C", "D", "B with probability sum", "C with probability sum"):
    iv = {"D": "A", "B with probability sum": "B", "C with probability sum": "C"}.get(v, v)
    d = ST[ST.version == v]
    sw = d.mean_abs_swing_pct.mean()
    iou = float(fold_iou(I[I.version == iv]).mean())
    a_site = ST[ST.version == "A"].set_index("site").mean_abs_swing_pct
    head.append({"version": v, "asked_for": v in ("A", "B", "C", "D"), "six_site_mean_abs_swing_pct": sw, "change_against_A_pct": 100 * (sw / A_sw - 1),
                 "all_pairs_mean_abs_swing_pct": P[P.version == v]["net_pct_of_first"].abs().mean(),
                 "sites_with_smaller_swing_than_A": int((d.set_index("site").mean_abs_swing_pct < a_site).sum()) if v != "A" else np.nan,
                 "iou_mean_of_5_folds_all_2020_chips": iou, "iou_change_against_A": iou - A_iou,
                 "iou_mean_of_5_folds_non_hazy": float(fold_iou(I[(I.version == iv) & ~I.hazy]).mean()),
                 "helps_by_the_rule": bool(sw <= 0.75 * A_sw and iou >= A_iou - 0.01) if v != "A" else np.nan})
Hd = pd.DataFrame(head)
Hd.round(4).to_csv(out / "summary.csv", index=False)

with pd.option_context("display.width", 260, "display.max_rows", 400, "display.float_format", "{:.3f}".format):
    print("\nSENSITIVITY OF A (thresholded area, percent change)\n" + SA.pivot_table(index="site", columns=["change", "amount"], values="area_change_pct").to_string())
    print("\nSENSITIVITY, MEAN OF SITES, ALL VERSIONS\n" + Sn.groupby(["version", "area", "change", "amount"]).area_change_pct.mean().unstack(["change", "amount"]).to_string())
    print("\nPER SITE\n" + ST.pivot_table(index="site", columns="version", values="mean_abs_swing_pct").to_string())
    print(ST[ST.version.isin(["A", "B", "C"])].pivot_table(index="site", columns="version", values=["iou_all_2020_chips", "iou_non_hazy_2020_chips"]).to_string())
    print("\nHEADLINE\n" + Hd.to_string(index=False))
    print("\nPAIRS\n" + P[P.version.isin(["A", "B", "C", "D"])].pivot_table(index=["site", "first_date", "second_date"], columns="version", values="net_pct_of_first").to_string())
    print("\nFITS\n" + pd.read_csv(work / "forests" / "fits.csv").to_string(index=False))
