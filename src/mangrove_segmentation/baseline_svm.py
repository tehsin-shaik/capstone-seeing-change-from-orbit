"""Baseline B3: per-pixel support vector machine (SVM) on all chip channels (bands + indices).

Mirrors baseline_rf.py (B2): same chips, same catalog, same labels, same pixel sampling
(the sampling function is imported from baseline_rf.py, not copied), same mask output,
and the same scoring code (evaluate.py).

WHY A LINEAR SVM (the "fast option")
    A textbook SVM with an RBF kernel compares every training pixel with every other one,
    so training time grows roughly with the square of the number of pixels, and predicting
    one pixel costs one comparison per support vector. Here there are 200,000 training
    pixels and about 14 million pixels to predict (215 chips x 256 x 256), which makes a
    plain RBF SVM take hours to days.
    Default here: LinearSVC, a linear SVM whose cost grows in proportion to the number of
    pixels. It trains on the same 200k class-balanced pixels as the random forest in
    seconds and predicts a chip with one matrix product.
    Limit: a linear SVM can only draw one flat boundary through the 8 features. It cannot
    learn rules like "high NDVI AND low MNDWI" as freely as a forest can, so expect it to
    land near the NDVI rule (B1) rather than the random forest (B2).
    Optional: --kernel rbf_approx keeps the linear solver but first maps the 8 features
    through a Nystroem approximation of the RBF kernel (300 landmark pixels). That gives a
    curved boundary at a cost that still grows in proportion to the number of pixels.

WHAT IS TUNED (as for B2: on validation data only, by pixel-pooled IoU)
    * C, the SVM's regularisation strength (3 values).
    * The decision threshold. An SVM gives a signed score, not a probability: 0 is the
      boundary it learned on the 50/50 balanced sample. Real chips are only about 10%
      mangrove, so the best cut-off is usually above 0. The RF tunes a probability
      threshold for the same reason (see results/mangrove_segmentation/baselines/RESULTS.md).
    Features are standardised (mean 0, sd 1) from the training pixels only; an SVM needs
    this because reflectance (0-0.5) and the indices (-1 to 1) are on different scales.

THREE COMMANDS
    fit / predict   exactly the baseline_rf.py interface (train + val split in the catalog):
        python baseline_svm.py fit     --catalog catalog.csv --chips chips/ --label v4019 --out models/svm.joblib
        python baseline_svm.py predict --catalog catalog.csv --chips chips/ --model models/svm.joblib \
            --split test --pred-out preds/svm
        python evaluate.py --pred preds/svm --catalog catalog.csv --chips chips/ --split test --label v4019 \
            --name "B3 SVM" --out results/svm

    cv              leave-one-site-out cross-validation on the Train/Val dataset:
        python baseline_svm.py cv --catalog <Train_Val_Dataset>/catalog.csv --chips <Train_Val_Dataset> \
            --qa <QA_download.csv> --out results/svm_cv
        (with the SDP folder layout of paths.py, plain `python baseline_svm.py cv` finds everything)

    The cv protocol is the one described in results/mangrove_segmentation/baselines/RESULTS.md.
    It was first rebuilt from that description and train_unet.py, and has since been compared
    line by line with the script that ran it for B1/B2 (the team's run_cv.py, which is not in
    this repository) and aligned with it:
      * 2020 chips only, label GMW v4.0.19; folds come from the catalog's `fold` column.
      * Round k holds out fold k. Variant (a) trains on all other 2020 chips, variant (b)
        drops hazy scenes from training (dark-water red, water_b04_p5, above 0.05).
      * Training chips within one chip (2,560 m) of a held-out chip are dropped.
      * C and the threshold are tuned on an inner fold (first of 3, 4, 5, 2, 1 that is not
        held out): train on 100k balanced pixels from the other 3 folds, validate on up to
        300k pixels (natural class mix) of the inner one. Same sizes as run_cv.py used for B2.
      * The chosen C and threshold are carried over unchanged, as run_cv.py carries the
        forest's settings and threshold, and the model is refitted on 200k balanced pixels
        from all 4 training folds. The random generator is restarted from the seed before
        that draw, as in run_cv.py, so for the same fold and variant the SVM is refitted on
        exactly the pixels the random forest was.
      * If the best threshold lands on the edge of the grid, the grid is widened on that side
        and searched again (run_cv.py did this by hand for B2: 0.80 -> 0.95).
      * The held-out chips are predicted and the masks are scored by evaluate.py
        (pixel-pooled, 95% bootstrap CI over chips).
    Output: cv_results.csv with the same columns as the baselines' cv_results.csv, plus
    fold_fits.csv, per_chip.csv, cv_catalog_2020.csv, config.json and preds/SVM_<variant>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as EV  # noqa: E402
from baseline_rf import collect, iou, pixels  # noqa: E402  (same pixel sampling as B2)
from chipio import chip_path, load_catalog, load_chip, save_mask  # noqa: E402

SETTINGS = [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}]
THRESHOLDS = np.round(np.arange(-2.0, 4.01, 0.25), 2)   # cut-offs on the SVM score (0 = balanced boundary)
GRID_STEP = 0.25        # spacing of THRESHOLDS; also the spacing used when the grid is widened
WIDEN_BY = 4            # thresholds added per widening (4 x 0.25 = 1.0 score unit)
MAX_WIDENINGS = 10      # per side, so the grid can grow by at most 10 score units each way
KERNELS = ("linear", "rbf_approx")
HAZE_P5 = 0.05          # same rule as make_datasets.py / train_unet.py
BUFFER_M = 2560         # one chip
INNER_PREF = [3, 4, 5, 2, 1]
VARIANTS = {"a": "a_all_dates", "b": "b_no_haze"}


def make_model(C: float, kernel: str = "linear", seed: int = 0):
    """Standardise, (optionally) approximate an RBF kernel, then a linear SVM."""
    steps = [StandardScaler()]
    if kernel == "rbf_approx":
        steps.append(Nystroem(kernel="rbf", gamma=1.0 / 8, n_components=300, random_state=seed))
    elif kernel != "linear":
        raise ValueError(f"kernel must be one of {KERNELS}")
    # dual=False: the faster formulation when there are many more pixels than features
    steps.append(LinearSVC(C=C, dual=False, max_iter=5000, random_state=seed))
    return make_pipeline(*steps)


def tune(Xtr, ytr, Xva, yva, kernel: str, seed: int) -> dict:
    """Best C and score threshold by IoU on the validation pixels.

    The search starts on THRESHOLDS. If the best threshold is the lowest or highest value of
    the grid, the true optimum may lie outside it, so the grid is widened on that side (same
    0.25 spacing) and searched again, until the best value is inside the grid. This is what
    run_cv.py did by hand for the random forest (grid end moved from 0.80 to 0.95).
    Ties go to the smaller C, then the lower threshold, as before.
    """
    fitted = []
    for s in SETTINGS:
        model = make_model(kernel=kernel, seed=seed, **s).fit(Xtr, ytr)
        fitted.append((s, model, model.decision_function(Xva)))
    grid = [float(t) for t in THRESHOLDS]
    seen = {}                                    # (setting number, threshold) -> IoU, so nothing is scored twice
    widened = {"lo": 0, "hi": 0}
    while True:
        best = None
        for i, (s, model, score) in enumerate(fitted):
            for th in grid:
                if (i, th) not in seen:
                    seen[(i, th)] = iou(score > th, yva)
                v = seen[(i, th)]
                if best is None or v > best["val_iou"]:
                    best = {"settings": s, "threshold": th, "val_iou": v, "svm": model}
        if best["threshold"] == grid[0] and widened["lo"] < MAX_WIDENINGS:
            grid = [round(grid[0] - GRID_STEP * j, 2) for j in range(WIDEN_BY, 0, -1)] + grid
            widened["lo"] += 1
        elif best["threshold"] == grid[-1] and widened["hi"] < MAX_WIDENINGS:
            grid = grid + [round(grid[-1] + GRID_STEP * j, 2) for j in range(1, WIDEN_BY + 1)]
            widened["hi"] += 1
        else:
            break
    if widened["lo"] or widened["hi"]:
        print(f"  note: best threshold was on the edge of the grid; grid widened to [{grid[0]}, {grid[-1]}]")
    if best["threshold"] in (grid[0], grid[-1]):
        print(f"  WARNING: chosen threshold {best['threshold']} is still on the edge after widening "
              f"to [{grid[0]}, {grid[-1]}]; the SVM score may carry no signal for these pixels")
    best["threshold_grid"] = [grid[0], grid[-1]]
    # best IoU each C reached on the validation pixels: shows how much the choice of C matters
    best["val_iou_by_C"] = {str(s["C"]): round(max(seen[(i, th)] for th in grid), 4)
                            for i, (s, _, _) in enumerate(fitted)}
    return best


# ----------------------------------------------------------------- fit / predict (as baseline_rf.py)

def fit(catalog, chips_dir, label="v4019", cap=200_000, val_cap=500_000, seed=0, kernel="linear"):
    rng = np.random.default_rng(seed)
    cat = load_catalog(catalog)
    tr, va = cat[cat["split"] == "train"], cat[cat["split"] == "val"]
    if tr.empty or va.empty:
        raise SystemExit("need both train and val chips in the catalog")
    Xtr, ytr = collect(tr, chips_dir, label, cap, rng, balanced=True)
    Xva, yva = collect(va, chips_dir, label, val_cap, rng, balanced=False)  # natural class mix
    best = tune(Xtr, ytr, Xva, yva, kernel, seed)
    svm = best.pop("svm")
    meta = {"model": "svm", "kernel": kernel, "label": label, "seed": seed,
            "train_pixels": int(len(ytr)), "val_pixels": int(len(yva)), **best}
    return svm, meta


def predict_chips(svm, th, cat, chips_dir, pred_out) -> int:
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(chips_dir, r))
        v = chip["valid"]
        mask = np.zeros(v.shape, bool)
        if v.any():
            mask[v] = svm.decision_function(pixels(chip)[v.ravel()]) > th
        save_mask(pred_out, r["chip_id"], mask)
    return len(cat)


def predict(catalog, chips_dir, bundle, split, pred_out):
    cat = load_catalog(catalog)
    if split != "all":
        cat = cat[cat["split"] == split]
    return predict_chips(bundle["svm"], bundle["meta"]["threshold"], cat, chips_dir, pred_out)


# ----------------------------------------------------------------- leave-one-site-out CV

def load_cv_catalog(catalog, qa=None, folds=None) -> pd.DataFrame:
    """2020 chips of the Train/Val catalog with `fold`, `hazy` and chip corner coordinates."""
    cat = load_catalog(catalog)
    if "fold" not in cat.columns:
        if folds is None:
            raise SystemExit("the catalog has no `fold` column: pass --folds folds_site_cv.csv (site, fold)")
        f = pd.read_csv(folds)
        cat["fold"] = cat["site"].map(dict(zip(f["site"], f["fold"])))
    if "year" in cat.columns:
        cat = cat[cat["year"] == 2020]
    cat = cat[cat["fold"].notna()].copy()
    if cat.empty:
        raise SystemExit("no 2020 chips with a fold in the catalog")
    cat["fold"] = cat["fold"].astype(int)
    if qa is not None:
        q = pd.read_csv(qa)[["site", "date", "water_b04_p5"]].drop_duplicates(["site", "date"])
        cat = cat.merge(q, on=["site", "date"], how="left")
        cat["hazy"] = cat["water_b04_p5"] > HAZE_P5
    else:
        cat["hazy"] = False
    if "bounds_epsg32640" in cat.columns:
        b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
        cat[["x0", "y0", "x1", "y1"]] = b.values
    else:
        print("! no bounds_epsg32640 column: the 2,560 m buffer around held-out chips is NOT applied")
    return cat.reset_index(drop=True)


def drop_near(train: pd.DataFrame, other: pd.DataFrame, buf=BUFFER_M) -> pd.DataFrame:
    """Drop training chips closer than `buf` metres to any chip in `other` (same rule as train_unet.py)."""
    if train.empty or other.empty or "x0" not in train.columns:
        return train
    keep = []
    for _, r in train.iterrows():
        dx = np.maximum(0, np.maximum(other.x0.values - r.x1, r.x0 - other.x1.values))
        dy = np.maximum(0, np.maximum(other.y0.values - r.y1, r.y0 - other.y1.values))
        keep.append(np.maximum(dx, dy).min() >= buf)
    return train[np.array(keep, dtype=bool)]


def run_cv(catalog, chips_dir, out, qa=None, folds=None, label="v4019", variants=("a", "b"),
           cap=200_000, val_cap=300_000, seed=0, kernel="linear", reps=1000):
    """Leave-one-site-out cross-validation, aligned with the team's run_cv.py (see the module docstring).

    cap      balanced pixels for the final fit on 4 folds (run_cv.py: final_cap = 200k);
             tuning on the inner split uses cap // 2 (run_cv.py: inner_cap = 100k).
    val_cap  pixels of the inner-validation fold used for tuning (run_cv.py: inner_val_cap = 300k).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if "b" in variants and qa is None:
        raise SystemExit("variant b drops hazy scenes and needs --qa (a CSV with site, date, water_b04_p5). "
                         "Use --variants a to run without it.")
    cat = load_cv_catalog(catalog, qa, folds)
    cv_cat = out / "cv_catalog_2020.csv"
    cat.to_csv(cv_cat, index=False)                      # the exact chips used; evaluate.py reads this
    all_sites = list(dict.fromkeys(cat.sort_values("fold")["site"]))
    present = sorted(cat["fold"].unique())
    if len(present) < 3:
        raise SystemExit("need at least 3 folds (held-out, inner validation, training)")
    rows, fits, per_chip = [], [], []
    for v in variants:
        name = VARIANTS[v]
        pred_dir = out / "preds" / f"SVM_{name}"
        for k in present:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            held = cat[cat["fold"] == k]
            pool = cat[cat["fold"] != k]
            if v == "b":
                pool = pool[~pool["hazy"]]
            train = drop_near(pool, held)                              # the 4 training folds
            inner = next(f for f in INNER_PREF + present if f != k and f in present)
            iv = train[train["fold"] == inner]
            tr = drop_near(train[train["fold"] != inner], iv)          # 3 folds for tuning
            if tr.empty or iv.empty:
                raise SystemExit(f"fold {k}: no chips left for tuning (train {len(tr)}, inner {len(iv)})")
            Xtr, ytr = collect(tr, chips_dir, label, cap // 2, rng, balanced=True)
            Xva, yva = collect(iv, chips_dir, label, val_cap, rng, balanced=False)
            best = tune(Xtr, ytr, Xva, yva, kernel, seed)
            # Restart the generator before the final draw, as run_cv.py does. With the same seed,
            # sampler (baseline_rf.collect) and chip order, these are the pixels B2 was refitted on.
            rng = np.random.default_rng(seed)
            Xall, yall = collect(train, chips_dir, label, cap, rng, balanced=True)
            # C and the threshold are carried over unchanged from the inner split (as run_cv.py does for B2)
            svm = make_model(kernel=kernel, seed=seed, **best["settings"]).fit(Xall, yall)   # refit on 4 folds
            predict_chips(svm, best["threshold"], held, chips_dir, pred_dir)
            held_sites = sorted(held["site"].unique())
            res, pc = EV.evaluate(pred_dir, cv_cat, chips_dir, split="all", label=label, reps=reps, seed=0,
                                  sites=held_sites)
            p = res["pooled"]
            row = {"model": "SVM", "variant": name, "fold": int(k), "held_site": ",".join(held_sites),
                   "chips": p["n_chips"], "mangrove_px": p["mangrove_px"],
                   **{m: round(p[m], 4) for m in ("iou", "f1", "precision", "recall")},
                   "iou_ci95_lo": round(p["iou_ci95"][0], 4), "iou_ci95_hi": round(p["iou_ci95"][1], 4)}
            for s in all_sites:
                row[f"iou_{s}"] = round(res["per_site"][s]["iou"], 4) if s in res["per_site"] else ""
            rows.append(row)
            fits.append({"model": "SVM", "variant": name, "fold": int(k), "held_site": row["held_site"],
                         "kernel": kernel, "C": best["settings"]["C"], "threshold": best["threshold"],
                         "inner_fold": int(inner), "inner_iou": round(best["val_iou"], 4),
                         "threshold_grid_lo": best["threshold_grid"][0], "threshold_grid_hi": best["threshold_grid"][1],
                         "inner_iou_by_C": json.dumps(best["val_iou_by_C"]),
                         "tune_chips": len(tr), "inner_chips": len(iv), "train_chips": len(train),
                         "tune_pixels": int(len(ytr)), "inner_pixels": int(len(yva)),
                         "train_pixels": int(len(yall)),
                         "hazy_chips_dropped": int((cat["fold"] != k).sum() - len(pool)),
                         "buffer_chips_dropped": int(len(pool) - len(train)),
                         "minutes": round((time.time() - t0) / 60, 2)})
            pc.insert(0, "fold", int(k)); pc.insert(0, "variant", name); pc.insert(0, "model", "SVM")
            per_chip.append(pc)
            print(f"SVM {name} fold {k} ({row['held_site']}): IoU {row['iou']:.3f} "
                  f"[{row['iou_ci95_lo']:.3f}, {row['iou_ci95_hi']:.3f}]  C={best['settings']['C']} "
                  f"threshold={best['threshold']}", flush=True)
            # written after every fold, so an interrupted run keeps what is finished
            pd.DataFrame(rows).to_csv(out / "cv_results.csv", index=False)
            pd.DataFrame(fits).to_csv(out / "fold_fits.csv", index=False)
            pd.concat(per_chip, ignore_index=True).to_csv(out / "per_chip.csv", index=False)
    (out / "config.json").write_text(json.dumps(
        {"model": "svm", "kernel": kernel, "label": label, "seed": seed, "cap": cap, "val_cap": val_cap,
         "tune_cap": cap // 2, "settings": SETTINGS, "thresholds_start": THRESHOLDS.tolist(),
         "threshold_grid_widens_if_best_on_edge": True, "haze_p5": HAZE_P5, "buffer_m": BUFFER_M,
         "inner_pref": INNER_PREF, "variants": [VARIANTS[v] for v in variants], "bootstrap_reps": reps}, indent=1))
    cv = pd.DataFrame(rows)
    print("\nmean +/- sd across folds:")
    for name, g in cv.groupby("variant"):
        print(f"  SVM {name}: " + "  ".join(f"{m} {g[m].mean():.3f} +/- {g[m].std(ddof=0):.3f}"
                                            for m in ("iou", "f1", "precision", "recall")))
    print(f"tables -> {out}")
    return cv


def cv_defaults() -> dict:
    """Default locations in the SDP folder layout (paths.py). They only matter if the flags are not given."""
    import paths as P
    return {"catalog": P.TRAIN_VAL / "catalog.csv", "chips": P.TRAIN_VAL, "qa": P.QA_CSV,
            "out": P.RESULTS / "Baselines_SVM"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--catalog", required=True); f.add_argument("--chips", required=True)
    f.add_argument("--label", default="v4019", choices=["v4019", "v4112"])
    f.add_argument("--cap", type=int, default=200_000); f.add_argument("--seed", type=int, default=0)
    f.add_argument("--kernel", default="linear", choices=KERNELS)
    f.add_argument("--out", required=True, help="model .joblib (a .json summary is written beside it)")
    p = sub.add_parser("predict")
    p.add_argument("--catalog", required=True); p.add_argument("--chips", required=True)
    p.add_argument("--model", required=True); p.add_argument("--split", default="test")
    p.add_argument("--pred-out", required=True)
    c = sub.add_parser("cv")
    c.add_argument("--catalog", help="Train_Val_Dataset/catalog.csv (needs a `fold` column, or pass --folds)")
    c.add_argument("--chips", help="Train_Val_Dataset folder")
    c.add_argument("--qa", help="CSV with site, date, water_b04_p5 (QA_download.csv or hazy_scenes.csv)")
    c.add_argument("--folds", help="CSV with site, fold; only needed if the catalog has no `fold` column")
    c.add_argument("--out", help="results folder")
    c.add_argument("--label", default="v4019", choices=["v4019", "v4112"])
    c.add_argument("--variants", nargs="+", default=["a", "b"], choices=["a", "b"],
                   help="a = all 2020 dates, b = hazy scenes dropped from training")
    c.add_argument("--cap", type=int, default=200_000); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--kernel", default="linear", choices=KERNELS)
    a = ap.parse_args()
    if a.cmd == "fit":
        svm, meta = fit(a.catalog, a.chips, a.label, a.cap, seed=a.seed, kernel=a.kernel)
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm": svm, "meta": meta}, out)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
        print(f"settings={meta['settings']} threshold={meta['threshold']} val IoU={meta['val_iou']:.3f}")
    elif a.cmd == "predict":
        n = predict(a.catalog, a.chips, joblib.load(a.model), a.split, a.pred_out)
        print(f"wrote {n} masks to {a.pred_out}")
    else:
        d = cv_defaults() if not (a.catalog and a.chips and a.out) else {}
        catalog, chips, out = a.catalog or d["catalog"], a.chips or d["chips"], a.out or d["out"]
        qa = a.qa or (d.get("qa") if d.get("qa") and Path(d["qa"]).exists() else None)
        if not Path(catalog).exists():
            raise SystemExit(f"catalog not found: {catalog}\nPass --catalog/--chips, or build the Train/Val dataset "
                             "first (run_all.py --step datasets).")
        run_cv(catalog, chips, out, qa=qa, folds=a.folds, label=a.label, variants=tuple(a.variants),
               cap=a.cap, seed=a.seed, kernel=a.kernel)


if __name__ == "__main__":
    main()
