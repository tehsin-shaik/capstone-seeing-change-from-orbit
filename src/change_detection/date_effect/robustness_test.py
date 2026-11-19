"""Robustness test: can the forest be made steady across dates? Four versions, all on the same 2020 chips and folds.

    A  current forest (the one of the baseline results)
    B  trained with extra copies of each training chip, changed in brightness like the real between-date differences
    C  trained on NDVI and MNDWI only
    D  forest A, area = sum of the predicted probabilities instead of the thresholded area

Run tide_test.py and image_differences_test.py first (this reads forests/, stacks/ and correction_lines.csv from <work>).

Fixed before running: the number that counts is the average same-year swing per site (per site, the mean over its
same-year pairs of |net change as a % of the first date's predicted mangrove|, then the mean of the six sites).
A version helps if that falls by at least a quarter against A and the held-out IoU (all held-out 2020 chips, mean of
the five folds) is no more than 0.01 lower.

Training follows train_fold in common.py step by step: hold one fold out, drop hazy scenes, drop chips within one chip
of a held-out chip, tune on an inner split, refit. Two things are swapped in from here, in memory only: for B the chip
loader returns altered copies of a chip, for C the feature function returns NDVI and MNDWI only. The lines used for B
never come from the held-out site, and Al Nouf's lines are left out because its fit was fragile. Before B and C are
trained, the same code is run with no change and must give forest A again.

Writes <out>/summary.csv.     Usage: python robustness_test.py --train-val <Train_Val_Dataset> --work <work> --out <results>/robustness
"""
import argparse
import itertools
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from shared import BANDS, CHIP, INDEX_EPS, SITES, check_pairs, indices, load_common, predict

ap = argparse.ArgumentParser()
ap.add_argument("--train-val", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tv, work, out = Path(a.train_val), Path(a.work), Path(a.out)
C = load_common(tv)
B2 = C.B2
out.mkdir(parents=True, exist_ok=True)
N_COPIES = 3

cat = C.load_full_catalog(tv)
assert set(cat["role"]) == {"cv"}, "catalog holds rows that are not train/val"
cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
FOLDS = sorted(int(f) for f in cat20.fold.unique())

# ---------------------------------------------------------------- train B and C
L = pd.read_csv(work / "correction_lines.csv")
L = L[L["site"] != "al_nouf"]
L = L[L.groupby(["site", "date"])["gain_could_be_fitted"].transform("all")]
LINES = {}            # site -> (gain[date, band], offset[date, band]) of the real between-date lines
for site, g in L.groupby("site"):
    dates = sorted(g["date"].unique())
    LINES[site] = (np.array([[g[(g.date == d) & (g.band == b)]["gain_offset_gain"].iloc[0] for b in BANDS] for d in dates]),
                   np.array([[g[(g.date == d) & (g.band == b)]["gain_offset_offset"].iloc[0] for b in BANDS] for d in dates]))

STATE = {"fold": None, "held_sites": []}
_chip_path, _load_chip, _pixels = B2.chip_path, B2.load_chip, B2.pixels


def draw_line(fold, chip_id, copy):
    """One real between-date line (gain and offset per band), from a site the fold does not hold out."""
    rng = np.random.default_rng([C.SEED, fold, copy, zlib.crc32(chip_id.encode())])
    sites = sorted(s for s in LINES if s not in STATE["held_sites"])
    G, O = LINES[sites[rng.integers(len(sites))]]
    i, j = rng.choice(len(G), 2, replace=False)
    gain = G[j] / G[i]
    return gain, O[j] - gain * O[i]


def chip_path_with_copy(d, row):
    copy = int(row["aug_copy"]) if "aug_copy" in row and pd.notna(row["aug_copy"]) else 0
    return (_chip_path(d, row), copy, str(row["chip_id"]))


def load_chip_with_copy(key):
    path, copy, chip_id = key
    chip = _load_chip(path)
    if copy == 0:
        return chip
    gain, off = draw_line(STATE["fold"], chip_id.split("__aug")[0], copy)
    img = chip["image"].astype(np.float32).copy()
    refl = img[:6] * gain[:, None, None].astype(np.float32) + off[:, None, None].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.clip((refl[3] - refl[2]) / np.maximum(refl[3] + refl[2], INDEX_EPS), -1, 1)
        mndwi = np.clip((refl[1] - refl[4]) / np.maximum(refl[1] + refl[4], INDEX_EPS), -1, 1)
    img[:6], img[6], img[7] = refl, ndvi, mndwi
    chip["image"] = img                      # valid, labels and masks are left as they are
    return chip


def with_copies(df):
    parts = [df.assign(aug_copy=0)]
    for c in range(1, N_COPIES + 1):
        parts.append(df.assign(aug_copy=c, chip_id=df["chip_id"].astype(str) + f"__aug{c}"))
    return pd.concat(parts, ignore_index=True)


def train(k, version):
    """Same steps as common.train_fold (variant b), with the two swaps."""
    present = set(int(f) for f in cat20.fold.unique())
    held = cat20[cat20.fold == k]
    pool_all = cat20[cat20.fold != k]
    train_ = C.drop_near(pool_all[~pool_all.hazy], held)
    assert not set(train_["site"]) & set(held["site"])
    inner = next(f for f in C.INNER_PREF if f != k and f in present)
    iv = train_[train_.fold == inner]
    it = C.drop_near(train_[train_.fold != inner], iv)
    STATE["fold"], STATE["held_sites"] = k, sorted(held["site"].unique())
    mult = 1
    B2.chip_path, B2.load_chip, B2.pixels = chip_path_with_copy, load_chip_with_copy, _pixels
    if version == "B":
        it, train_, mult = with_copies(it), with_copies(train_), N_COPIES + 1
    else:
        it, train_ = it.assign(aug_copy=0), train_.assign(aug_copy=0)
    if version == "C":
        B2.pixels = lambda chip: _pixels(chip)[:, 6:8]
    try:
        d = work / "forests" / "versions_work"
        d.mkdir(exist_ok=True)
        p = d / f"inner_catalog_{version}_fold{k}.csv"
        pd.concat([it.assign(split="train"), iv.assign(split="val", aug_copy=0)]).to_csv(p, index=False)
        _, meta = B2.fit(p, tv, C.LABEL, cap=C.CFG["inner_cap"] * mult, val_cap=C.CFG["inner_val_cap"], seed=C.SEED,
                         n_estimators=C.CFG["inner_trees"])
        rng = np.random.default_rng(C.SEED)
        X, y = B2.collect(train_, tv, C.LABEL, C.CFG["final_cap"] * mult, rng, balanced=True)
        rf = RandomForestClassifier(n_estimators=C.CFG["final_trees"], n_jobs=-1, random_state=C.SEED, **meta["settings"]).fit(X, y)
    finally:
        B2.chip_path, B2.load_chip, B2.pixels = _chip_path, _load_chip, _pixels
    return {"rf": rf, "threshold": meta["threshold"], "settings": meta["settings"], "held_sites": sorted(held["site"].unique())}


if not all((work / "forests" / f"{v}_fold{k}.joblib").exists() for v in "BC" for k in FOLDS):
    for k in FOLDS:                              # with no change the training code must give forest A again
        fit, A = train(k, "A"), joblib.load(work / "forests" / f"fold{k}.joblib")
        X = B2.pixels(C.load_chip(C.chip_path(tv, cat20[cat20.fold == k].iloc[0])))
        diff = float(np.max(np.abs(A["rf"].predict_proba(X) - fit["rf"].predict_proba(X))))
        assert A["threshold"] == fit["threshold"] and A["settings"] == fit["settings"] and diff < 1e-9, f"fold {k}: forest A is not reproduced"
        print(f"fold {k}: training code gives forest A again", flush=True)
    for v in "BC":
        for k in FOLDS:
            fit = train(k, v)
            joblib.dump(fit, work / "forests" / f"{v}_fold{k}.joblib", compress=3)
            print(f"trained {v} fold {k}: threshold {fit['threshold']}, {fit['settings']}", flush=True)

# ---------------------------------------------------------------- score the versions
FOLD = {s: int(cat[cat.site == s]["fold"].iloc[0]) for s in SITES}


def forest(version, site):
    F = joblib.load(work / "forests" / (f"fold{FOLD[site]}.joblib" if version == "A" else f"{version}_fold{FOLD[site]}.joblib"))
    assert site in F["held_sites"]
    return F["rf"], float(F["threshold"]), list(F["rf"].classes_).index(True)


def prob(rf, pos, version, feats, valid):
    """Probability map for one image [8, H, W]. Version C sees NDVI and MNDWI only."""
    return predict(rf, pos, feats[6:8] if version == "C" else feats, valid)


pairs, ious, sens = [], [], []
for site in SITES:
    z = np.load(work / "stacks" / f"{site}.npz", allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    years = [int(d[:4]) for d in dates]
    n = len(dates)
    img, valid, covered = z["image"], z["valid"], z["covered"]
    F = {v: forest(v, site) for v in "ABC"}
    probs = {"A": z["prob"].astype(np.float32)}
    assert np.max(np.abs(prob(*F["A"][::2], "A", img[0], valid[0]) - probs["A"][0])) < 1e-6, "forest A does not reproduce the stored probabilities"
    for v in "BC":
        probs[v] = np.stack([prob(*F[v][::2], v, img[i], valid[i]) for i in range(n)])

    # same-year pairs: thresholded area for A, B and C, sum of probabilities for D
    for i, j in itertools.combinations(range(n), 2):
        if years[i] != years[j] or int(round((covered[i] & covered[j]).sum() / CHIP ** 2)) == 0:
            continue
        u = valid[i] & valid[j]
        base = {"site": site, "first_date": dates[i], "second_date": dates[j]}
        for v in "ABC":
            a0, a1 = int(((probs[v][i] > F[v][1]) & u).sum()), int(((probs[v][j] > F[v][1]) & u).sum())
            pairs.append({**base, "version": v, "area_first_ha": round(a0 * 0.01, 2), "net_pct_of_first": 100 * (a1 - a0) / a0 if a0 else np.nan})
        s0, s1 = float(probs["A"][i][u].sum()), float(probs["A"][j][u].sum())
        pairs.append({**base, "version": "D", "area_first_ha": round(s0 * 0.01, 2), "net_pct_of_first": 100 * (s1 - s0) / s0 if s0 else np.nan})

    # held-out IoU against GMW v4.0.19 on every 2020 chip of the site
    for v in "ABC":
        rf, th, pos = F[v]
        for _, r in cat[(cat.site == site) & (cat.year == 2020)].iterrows():
            chip = C.load_chip(C.chip_path(tv, r))
            vv = chip["valid"]
            mask = (prob(rf, pos, v, chip["image"], vv) > th) & vv
            truth, labelled = C.get_label(chip, C.LABEL)
            use = vv & labelled
            ious.append({"fold": FOLD[site], "version": v, "tp": int((mask & truth & use).sum()), "fp": int((mask & ~truth & use).sum()),
                         "fn": int((~mask & truth & use).sum())})

    # sensitivity of A: make every image 10% darker or brighter and see how far the thresholded area moves
    rf, th, pos = F["A"]
    base_area = sum(int(((probs["A"][i] > th) & valid[i]).sum()) for i in range(n))
    for amount in (0.90, 1.10):
        area = 0
        for i in range(n):
            x = img[i, :6] * np.float32(amount)
            area += int(((prob(rf, pos, "A", np.concatenate([x, *(k[None] for k in indices(x))], axis=0), valid[i]) > th) & valid[i]).sum())
        sens.append({"site": site, "amount": amount, "area_change_pct": 100 * (area / base_area - 1)})
    print(f"{site}: scored", flush=True)

P, I, Sn = pd.DataFrame(pairs), pd.DataFrame(ious), pd.DataFrame(sens)
check_pairs(P[P.version == "A"])
f = I.groupby(["version", "fold"])[["tp", "fp", "fn"]].sum()
fold_iou = f.tp / (f.tp + f.fp + f.fn)
for k in FOLDS:
    assert abs(fold_iou["A"][k] - C.published_fold_iou(k, "b")) < 1e-4, f"fold {k}: A does not give the committed IoU"
iou = fold_iou.groupby("version").mean()
iou["D"] = iou["A"]

P["swing_pct"] = P["net_pct_of_first"].abs()
P["swing_ha"] = (P["net_pct_of_first"].round(3) * P["area_first_ha"] / 100).abs()
site = P.groupby(["version", "site"]).agg(swing_pct=("swing_pct", "mean"), swing_ha=("swing_ha", "mean"), area=("area_first_ha", "sum"))
WHAT = {"A": "current forest", "B": "trained with brightness copies", "C": "trained on NDVI and MNDWI only",
        "D": "forest A, area from the sum of probabilities"}
rows = []
for v in "ABCD":
    sw, sw_a = site.loc[v, "swing_pct"].mean(), site.loc["A", "swing_pct"].mean()
    rows.append({"version": v, "what": WHAT[v], "average_same_year_swing_pct": round(sw, 2), "change_against_A_pct": round(100 * (sw / sw_a - 1), 1),
                 "heldout_iou": round(iou[v], 4), "iou_change_against_A": round(iou[v] - iou["A"], 4),
                 "helps_by_the_rule": bool(sw <= 0.75 * sw_a and iou[v] >= iou["A"] - 0.01) if v != "A" else np.nan,
                 "average_same_year_swing_ha": round(site.loc[v, "swing_ha"].mean(), 1),
                 "area_measure_over_thresholded_area_of_A": round((site.loc[v, "area"] / site.loc["A", "area"]).mean(), 3),
                 "area_change_when_images_10pct_darker_pct": round(Sn[Sn.amount == 0.90].area_change_pct.mean(), 1) if v == "A" else np.nan,
                 "area_change_when_images_10pct_brighter_pct": round(Sn[Sn.amount == 1.10].area_change_pct.mean(), 1) if v == "A" else np.nan})
R = pd.DataFrame(rows)
R.to_csv(out / "summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(R.T.to_string())
