"""Score the out-of-fold predictions written by run_cv.py with code/evaluate.py, then build
cv_results.csv, per_chip.csv, oof_pooled.csv, per_site_iou.csv, the land-cover error table and figures.
Paths come from 2_Code/paths.py (tables -> 4_Results/Baselines, predictions in 6_Work/runs/Baselines/preds).
The land-cover error table is skipped if the per-chip WorldCover files are missing (run_all.py --step landcover)."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # 2_Code
import paths as P
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluate as EV
from chipio import chip_path, get_label, load_chip, load_mask

TV = P.TRAIN_VAL
OUT = P.BASELINES
PRED = P.WORK / "runs" / "Baselines" / "preds"
TMP = P.TMP / "baselines"; TMP.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = [("B1", "NDVI + MNDWI threshold"), ("B2", "Random forest (per pixel)")]
VARIANTS = [("a_all_dates", "(a) all 2020 dates"), ("b_no_haze", "(b) hazy scenes dropped from training")]
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built-up", 60: "bare", 70: "snow", 80: "water",
      90: "herbaceous wetland", 95: "mangrove (WorldCover)", 100: "moss", 0: "no data"}

cat = pd.read_csv(OUT / "cv_catalog_2020.csv")
fits = pd.read_csv(OUT / "fold_fits.csv")
score_cat = cat.assign(split="f" + cat.fold.astype(str))
sc_path = TMP / "score_catalog.csv"; score_cat.to_csv(sc_path, index=False)
clear = score_cat.copy(); clear.loc[clear.hazy, "split"] = "hazy"
clear.loc[~clear.hazy, "split"] = "clear"
clear_path = TMP / "score_catalog_clear.csv"; clear.to_csv(clear_path, index=False)

rows, chips, pooled = [], [], []
for m, _ in MODELS:
    for v, _ in VARIANTS:
        pred = PRED / f"{m}_{v}"
        for k in sorted(cat.fold.unique()):
            res, pc = EV.evaluate(pred, sc_path, TV, split=f"f{k}", label="v4019", reps=1000, seed=0)
            r = res["pooled"]
            rows.append({"model": m, "variant": v, "fold": k, "held_site": fits.query("variant==@v and fold==@k").held_site.iloc[0],
                         "chips": r["n_chips"], "mangrove_px": r["mangrove_px"],
                         **{x: round(r[x], 4) for x in EV.METRICS},
                         "iou_ci95_lo": round(r["iou_ci95"][0], 4), "iou_ci95_hi": round(r["iou_ci95"][1], 4),
                         **{f"iou_{s}": round(d["iou"], 4) for s, d in res["per_site"].items()}})
            chips.append(pc.assign(model=m, variant=v, fold=k))
        for subset, path in (("all", sc_path), ("clear", clear_path)):
            res, _ = EV.evaluate(pred, path, TV, split="all" if subset == "all" else "clear", label="v4019", reps=1000, seed=0)
            r = res["pooled"]
            pooled.append({"model": m, "variant": v, "scored_on": "all OOF chips" if subset == "all" else "non-hazy OOF chips",
                           "chips": r["n_chips"], **{x: round(r[x], 4) for x in EV.METRICS},
                           "iou_ci95": [round(x, 4) for x in r["iou_ci95"]]})
cv = pd.DataFrame(rows); cv.to_csv(OUT / "cv_results.csv", index=False)
pc = pd.concat(chips).merge(cat[["chip_id", "date", "hazy"]], on="chip_id"); pc.to_csv(OUT / "per_chip.csv", index=False)
pooled = pd.DataFrame(pooled); pooled.to_csv(OUT / "oof_pooled.csv", index=False)

# per-site IoU from OOF predictions (fold 5 holds two sites)
site_rows = []
for (m, v), g in pc.groupby(["model", "variant"]):
    for s, gs in g.groupby("site"):
        t = gs[["tp", "fp", "fn"]].sum()
        site_rows.append({"model": m, "variant": v, "site": s, "chips": len(gs), "iou": round(t.tp / (t.tp + t.fp + t.fn), 4)})
sites = pd.DataFrame(site_rows); sites.to_csv(OUT / "per_site_iou.csv", index=False)

# ---- land-cover breakdown of errors (variant a, both models) ----
lc_rows = []
have_lc = all((TV / "landcover" / f"{c}.npz").exists() for c in cat.chip_id)
if not have_lc:
    print("Land-cover files missing in", TV / "landcover", "- skipping the land-cover error table "
          "(run: python 2_Code/run_all.py --step landcover datasets)")
for m, _ in (MODELS if have_lc else []):
    v = "a_all_dates"
    fp = {}; fn = {}; tot = {}
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(TV, r)); truth, labelled = get_label(chip, "v4019")
        use = chip["valid"] & labelled
        pred = load_mask(PRED / f"{m}_{v}", r.chip_id)
        lc = np.load(TV / "landcover" / f"{r.chip_id}.npz")["lc"]
        for c in np.unique(lc[use]):
            sel = use & (lc == c)
            fp[c] = fp.get(c, 0) + int(np.sum(pred & ~truth & sel))
            fn[c] = fn.get(c, 0) + int(np.sum(~pred & truth & sel))
            tot[c] = tot.get(c, 0) + int(np.sum(sel & ~truth))
    sfp = sum(fp.values()); sfn = sum(fn.values())
    for c in sorted(tot):
        lc_rows.append({"model": m, "class": WC.get(int(c), str(c)), "non_mangrove_px": tot[c], "false_pos_px": fp[c],
                        "share_of_all_FP_%": round(100 * fp[c] / sfp, 1) if sfp else 0,
                        "FP_rate_in_class_%": round(100 * fp[c] / tot[c], 2) if tot[c] else 0,
                        "false_neg_px": fn[c], "share_of_all_FN_%": round(100 * fn[c] / sfn, 1) if sfn else 0})
lcdf = pd.DataFrame(lc_rows); lcdf.to_csv(OUT / "landcover_errors.csv", index=False)

# ---- figures ----
fig, ax = plt.subplots(figsize=(9, 4))
labels = [f"F{k}\n{cv.query('fold==@k').held_site.iloc[0].replace('_', ' ')[:22]}" for k in sorted(cv.fold.unique())]
x = np.arange(len(labels)); w = 0.2
cols = {"B1": ["#9bb7d4", "#2f6aa3"], "B2": ["#f2b48c", "#c9571c"]}
i = 0
for m, mn in MODELS:
    for j, (v, vn) in enumerate(VARIANTS):
        vals = cv.query("model==@m and variant==@v").sort_values("fold").iou.values
        ax.bar(x + (i - 1.5) * w, vals, w, color=cols[m][j], label=f"{m} {vn}")
        i += 1
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("IoU on held-out site (GMW v4.0.19)")
ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=2); ax.set_title("Leave-one-site-out IoU per fold")
plt.tight_layout(); plt.savefig(FIG / "iou_per_fold.png", dpi=130); plt.close()

imp = pd.read_csv(OUT / "rf_feature_importances.csv")
chs = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"]
a = imp[imp.variant == "a_all_dates"][chs]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(chs, a.mean().values, yerr=a.std().values, color="#c9571c", capsize=3)
ax.set_ylabel("Mean decrease in impurity"); ax.set_title("B2 random forest: feature importance (5 folds, variant a)")
plt.tight_layout(); plt.savefig(FIG / "rf_feature_importance.png", dpi=130); plt.close()

ex = []
for k in [f for f in (2, 3, 4, 1, 5) if ((cat.fold == f) & (~cat.hazy)).any()][:3]:
    g = cat[(cat.fold == k) & (~cat.hazy)].sort_values("mangrove_pct_v4019", ascending=False)
    ex.append(g.iloc[len(g) // 4])
fig, axs = plt.subplots(len(ex), 4, figsize=(11, 2.9 * len(ex)), squeeze=False)
for i, r in enumerate(ex):
    chip = load_chip(chip_path(TV, r)); img = chip["image"]
    rgb = np.clip(np.stack([img[2], img[1], img[0]], -1) / 0.25, 0, 1); rgb = np.nan_to_num(rgb)
    truth, _ = get_label(chip, "v4019")
    p1 = load_mask(PRED / "B1_a_all_dates", r.chip_id); p2 = load_mask(PRED / "B2_a_all_dates", r.chip_id)
    row = pc.query("chip_id==@r.chip_id and variant=='a_all_dates'")
    i1 = row.query("model=='B1'").iou.iloc[0]; i2 = row.query("model=='B2'").iou.iloc[0]
    for j, (im, t) in enumerate([(rgb, f"{r.site} {r.date}\nRGB"), (truth, "GMW v4.0.19 label"),
                                 (p1, f"B1 NDVI (IoU {i1:.2f})"), (p2, f"B2 RF (IoU {i2:.2f})")]):
        axs[i, j].imshow(im, cmap=None if j == 0 else "Greens", vmin=0, vmax=1, interpolation="nearest")
        axs[i, j].set_title(t, fontsize=8); axs[i, j].axis("off")
plt.tight_layout(); plt.savefig(FIG / "examples.png", dpi=110); plt.close()
print(cv.to_string()); print(pooled.to_string()); print(lcdf.to_string()); print(sites.to_string())
