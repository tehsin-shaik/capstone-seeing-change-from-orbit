"""Score the U-Net out-of-fold predictions exactly like the baselines (evaluate.py), compare with
B1/B2 (4_Results/Baselines/cv_results.csv, variant b), and write cv_results.csv, per_chip.csv, figures and
RESULTS.md into 4_Results/Unet (cpu preset) or 4_Results/Unet_<preset>.

Usage: python 2_Code/supporting_scripts/unet/score_unet.py [--preset cpu|gpu]"""
import argparse, json, sys
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

_ap = argparse.ArgumentParser(); _ap.add_argument("--preset", default="cpu"); PRESET = _ap.parse_args().preset
TV = P.TRAIN_VAL
OUT = P.unet_dir(PRESET)
RUN = P.WORK / "runs" / OUT.name                 # weights + predictions written by train_unet.py
BASE = P.BASELINES
BASE_PRED = P.WORK / "runs" / "Baselines" / "preds"
NAME = "CPU version" if PRESET == "cpu" else f"{PRESET} preset"
FIG = OUT / "figs"; FIG.mkdir(parents=True, exist_ok=True)
V = "b_no_haze"

cat = pd.read_csv(BASE / "cv_catalog_2020.csv")
fits = pd.read_csv(OUT / "fold_fits.csv")
hist = pd.read_csv(OUT / "history.csv")
cfg = json.load(open(OUT / "config.json"))
sc = cat.assign(split="f" + cat.fold.astype(str)); scp = OUT / "score_catalog.csv"; sc.to_csv(scp, index=False)

rows, chips = [], []
pred = RUN / "preds" / f"UNET_{V}"
for k in sorted(fits.fold):
    res, pc = EV.evaluate(pred, scp, TV, split=f"f{k}", label="v4019", reps=1000, seed=0)
    r = res["pooled"]
    rows.append({"model": "UNET", "variant": V, "fold": k, "held_site": fits.query("fold==@k").held_site.iloc[0],
                 "chips": r["n_chips"], "mangrove_px": r["mangrove_px"], **{x: round(r[x], 4) for x in EV.METRICS},
                 "iou_ci95_lo": round(r["iou_ci95"][0], 4), "iou_ci95_hi": round(r["iou_ci95"][1], 4),
                 **{f"iou_{s}": round(d["iou"], 4) for s, d in res["per_site"].items()}})
    chips.append(pc.assign(model="UNET", variant=V, fold=k))
cv = pd.DataFrame(rows); cv.to_csv(OUT / "cv_results.csv", index=False)
pc = pd.concat(chips); pc.to_csv(OUT / "per_chip.csv", index=False)
res_all, _ = EV.evaluate(pred, scp.with_name("score_catalog.csv"), TV, split="all", label="v4019", reps=1000, seed=0)

base = pd.read_csv(BASE / "cv_results.csv").query("variant==@V")
b1 = base.query("model=='B1'").set_index("fold"); b2 = base.query("model=='B2'").set_index("fold")
u = cv.set_index("fold")

def msd(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"

head = pd.DataFrame([
    {"Model": f"U-Net ({NAME})", **{m: msd(u[m]) for m in ["iou", "f1", "precision", "recall"]}},
    {"Model": "B2 random forest", **{m: msd(b2[m]) for m in ["iou", "f1", "precision", "recall"]}},
    {"Model": "B1 NDVI + MNDWI threshold", **{m: msd(b1[m]) for m in ["iou", "f1", "precision", "recall"]}},
])
per = []
for k in u.index:
    overlap = not (u.loc[k, "iou_ci95_lo"] > b2.loc[k, "iou_ci95_hi"] or b2.loc[k, "iou_ci95_lo"] > u.loc[k, "iou_ci95_hi"])
    per.append({"Fold": k, "Held-out site": u.loc[k, "held_site"].replace("_", " "),
                "U-Net IoU (95% CI)": f"{u.loc[k,'iou']:.3f} ({u.loc[k,'iou_ci95_lo']:.3f}–{u.loc[k,'iou_ci95_hi']:.3f})",
                "B2 IoU (95% CI)": f"{b2.loc[k,'iou']:.3f} ({b2.loc[k,'iou_ci95_lo']:.3f}–{b2.loc[k,'iou_ci95_hi']:.3f})",
                "U-Net − B2": f"{u.loc[k,'iou']-b2.loc[k,'iou']:+.3f}", "CIs overlap": "yes" if overlap else "no",
                "B1 IoU": f"{b1.loc[k,'iou']:.3f}"})
per = pd.DataFrame(per)

# ---- figures ----
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(u)); w = 0.27
for i, (lab, df, col) in enumerate([("B1 NDVI threshold", b1, "#2f6aa3"), ("B2 random forest", b2, "#c9571c"), (f"U-Net ({NAME})", u, "#2e8b57")]):
    vals = df.loc[u.index, "iou"].values
    lo = vals - df.loc[u.index, "iou_ci95_lo"].values; hi = df.loc[u.index, "iou_ci95_hi"].values - vals
    ax.bar(x + (i - 1) * w, vals, w, yerr=[lo, hi], capsize=2, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"F{k}\n{u.loc[k,'held_site'].replace('_',' ')[:22]}" for k in u.index], fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("IoU on held-out site (GMW v4.0.19)"); ax.axhline(b2.iou.mean(), color="#c9571c", ls="--", lw=0.8)
ax.legend(fontsize=8); ax.set_title("Leave-one-site-out IoU per fold, hazy scenes dropped from training (variant b)")
plt.tight_layout(); plt.savefig(FIG / "iou_per_fold.png", dpi=130); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for k, g in hist.groupby("fold"):
    site = fits.query("fold==@k").held_site.iloc[0].replace("_", " ")
    axs[0].plot(g.epoch, g.train_loss, label=f"F{k} {site[:18]}"); axs[1].plot(g.epoch, g["inner_val_iou@0.5"], label=f"F{k}")
    be = fits.query("fold==@k").best_epoch.iloc[0]
    axs[1].scatter([be], [g.query("epoch==@be")["inner_val_iou@0.5"].iloc[0]], s=18)
axs[0].set_title("Training loss (BCE + Dice)"); axs[0].set_xlabel("epoch"); axs[0].legend(fontsize=7)
axs[1].set_title("Inner-validation IoU at threshold 0.5 (dot = best epoch)"); axs[1].set_xlabel("epoch"); axs[1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(FIG / "training_curves.png", dpi=130); plt.close()

ex = []
for k in [f for f in (2, 3, 4, 1, 5) if f in set(u.index) and ((cat.fold == f) & (~cat.hazy)).any()][:3]:
    g = cat[(cat.fold == k) & (~cat.hazy)].sort_values("mangrove_pct_v4019", ascending=False)
    ex.append(g.iloc[len(g) // 4])
bpc = pd.read_csv(BASE / "per_chip.csv").query("variant==@V")
have_b2 = (BASE_PRED / f"B2_{V}").exists()
fig, axs = plt.subplots(len(ex), 4, figsize=(11, 2.9 * len(ex)), squeeze=False)
for i, r in enumerate(ex):
    chip = load_chip(chip_path(TV, r)); img = chip["image"]
    rgb = np.nan_to_num(np.clip(np.stack([img[2], img[1], img[0]], -1) / 0.25, 0, 1))
    truth, _ = get_label(chip, "v4019")
    pu = load_mask(pred, r.chip_id); iu = pc.query("chip_id==@r.chip_id").iou.iloc[0]
    if have_b2:
        p2 = load_mask(BASE_PRED / f"B2_{V}", r.chip_id)
        q = bpc.query("model=='B2' and chip_id==@r.chip_id"); i2 = q.iou.iloc[0] if len(q) else float("nan")
    else:   # baselines not run on this machine: show an empty panel
        p2 = np.zeros_like(pu); i2 = float("nan")
    for j, (im, t) in enumerate([(rgb, f"{r.site} {r.date}\nRGB"), (truth, "GMW v4.0.19 label"),
                                 (p2, f"B2 random forest (IoU {i2:.2f})"), (pu, f"U-Net (IoU {iu:.2f})")]):
        axs[i, j].imshow(im, cmap=None if j == 0 else "Greens", vmin=0, vmax=1, interpolation="nearest")
        axs[i, j].set_title(t, fontsize=8); axs[i, j].axis("off")
plt.tight_layout(); plt.savefig(FIG / "examples.png", dpi=110); plt.close()

# ---- RESULTS.md ----
mu, m2 = u.iou.mean(), b2.iou.mean()
wins = int((u.iou > b2.loc[u.index, "iou"]).sum())
sep_better = int(sum(1 for k in u.index if u.loc[k, "iou_ci95_lo"] > b2.loc[k, "iou_ci95_hi"]))
sep_worse = int(sum(1 for k in u.index if b2.loc[k, "iou_ci95_lo"] > u.loc[k, "iou_ci95_hi"]))
verdict = (f"yes, it beats the random-forest bar ({m2:.2f})" if mu > m2 else f"no, it does not beat the random-forest bar ({m2:.2f})")
pr = res_all["pooled"]
mins = fits.minutes.sum()
md = f"""# Stage 5, M1: U-Net ({NAME}), leave-one-site-out CV

Train_Val_Dataset only (Test_Dataset never read). 2020 chips, label GMW v4.0.19 (10 m), hazy scenes dropped from training
(variant b, the same rule as the baselines). The same {len(u)} folds, 1-chip buffer and scoring code (`evaluate.py`) as B1/B2.

## Headline (mean ± sd over {len(u)} folds)

{head.to_markdown(index=False)}

**Verdict: {verdict}** (U-Net {mu:.3f} vs B2 {m2:.3f}, difference {mu-m2:+.3f}).
U-Net beats B2 on {wins} of {len(u)} folds. Its CI is entirely above B2's on {sep_better} fold(s) and entirely below on {sep_worse} fold(s).
Pooled out-of-fold U-Net IoU is {pr['iou']:.3f} (95% CI {pr['iou_ci95'][0]:.3f}–{pr['iou_ci95'][1]:.3f}).

## Per fold

{per.to_markdown(index=False)}

## How it was trained

- **Model:** U-Net with 4 levels, base width {cfg['base']}, BatchNorm, {cfg['params']:,} parameters, 8 input channels (B02 B03 B04 B08 B11 B12 NDVI MNDWI).
- **Normalisation:** per fold, from that fold's training chips only.
- **Loss:** BCE (pos_weight 4) + soft Dice, ignoring unlabelled, NaN and cloud pixels.
- **Augmentation:** random {cfg['crop']}×{cfg['crop']} crops ({cfg['crops_per_chip']} per chip per epoch), rot90 and flips, and a ±0.02 reflectance shift (NDVI/MNDWI recomputed).
- **Optimiser:** Adam, lr {cfg['lr']}, cosine schedule, at most {cfg['epochs']} epochs, early stopping (patience {cfg['patience']}) on inner-validation IoU, batch {cfg['batch']}.
- **Threshold:** chosen on the inner-validation fold (grid 0.05–0.95). Chosen per fold: {', '.join(f"F{k} {t}" for k, t in zip(fits.fold, fits.threshold))}.
- **Inner validation:** the same fold B2 used (first of 3, 4, 5, 2, 1 that is not the held-out fold).
- **Runtime:** {mins:.0f} min in total on {cfg.get('device', 'cpu')}. Best epochs: {', '.join(f"F{k} {e}/{n}" for k, e, n in zip(fits.fold, fits.best_epoch, fits.epochs_run))}.

## What limits this run (what a bigger GPU run should change)

1. **No refit on all 4 training folds.** B2 was refitted on all 4 after tuning; the U-Net trains on 3 (one is kept for early stopping). On GPU, refit on all 4 folds for the best epoch count, or average the fold models.
2. **Small and short:** base width 16, 128-pixel crops, at most {cfg['epochs']} epochs. GPU preset: width 32, full 256 crops, about 100 epochs.
3. **Few chips:** {int(fits.train_chips.min())}–{int(fits.train_chips.max())} training chips per fold after dropping hazy scenes and the buffer. More augmentation, or keeping the hazy scenes (B2 barely differed between a and b), may help.
4. **One seed per fold;** fold-to-fold spread is large (sd above), so differences under about 0.03 are within noise.

Files: `cv_results.csv`, `per_chip.csv`, `fold_fits.csv`, `history.csv`, `figs/` (iou_per_fold, training_curves, examples).
Model weights and predictions: `6_Work/runs/{OUT.name}/` (`unet_fold*.pt`, `preds/`).
"""
open(OUT / "RESULTS.md", "w", encoding="utf-8").write(md)
print(head.to_string()); print(per.to_string()); print(verdict, mu, m2)
