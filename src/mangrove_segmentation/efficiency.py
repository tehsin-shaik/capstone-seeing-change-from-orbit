"""Efficiency study: how much smaller can the U-Net get before it gets worse?

Reads the U-Net runs made by train_unet.py (one results folder per preset, e.g. 4_Results/Unet_gpu,
Unet_gpu_b16, Unet_gpu_b8, Unet_gpu_b4), scores each one's held-out predictions with evaluate.py the
same way as every other model, and puts size, speed and IoU side by side.

Per preset it reports parameters, mean IoU over the 5 folds, training minutes per fold, prediction time
per 256x256 chip and peak GPU memory (empty for CPU runs). It also writes a cv_results.csv per preset in
the usual format, so compare_models.py can test the small models against the others.

Timings only make sense if every preset ran on the same machine, so the table shows the device too.

With --out it also runs the paired fold-by-fold test from compare_models.py: every small U-Net against
the full U-Net and the random forest in results/mangrove_segmentation (the committed runs).

Usage (from src/mangrove_segmentation, after training the presets):
  python efficiency.py                                    # all presets found in 4_Results
  python efficiency.py --presets gpu gpu_b8 gpu_b4 --out ../../results/mangrove_segmentation/efficiency
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_models as CM  # noqa: E402
import evaluate as EV  # noqa: E402
import paths as P  # noqa: E402

VARIANT = "b_no_haze"
REPO_RESULTS = HERE.parents[1] / "results" / "mangrove_segmentation"
FULL_UNET_PARAMS = 7_764_481    # base 32, counted in model_size.py (the committed gpu run)
PRESETS = ["gpu", "gpu_b16", "gpu_b8", "gpu_b4"]


def score_catalog(tv: Path) -> pd.DataFrame:
    """2020 chips with split = "f<fold>", which is what evaluate.py needs to score one fold at a time."""
    cat = pd.read_csv(tv / "catalog.csv")
    cat = cat[cat["year"] == 2020].copy()
    cat["fold"] = cat["fold"].astype(int)
    cat["split"] = "f" + cat["fold"].astype(str)
    return cat


def score_preset(preset: str, tv: Path, results: Path, work: Path, tmp: Path) -> tuple[dict, pd.DataFrame]:
    """One summary row for the preset, plus its cv_results table (one row per fold)."""
    out = results / ("Unet" if preset == "cpu" else f"Unet_{preset}")
    preds = work / "runs" / out.name / "preds" / f"UNET_{VARIANT}"
    fits = pd.read_csv(out / "fold_fits.csv")
    cfg = json.load(open(out / "config.json"))

    tmp.mkdir(parents=True, exist_ok=True)
    cat_path = tmp / "efficiency_score_catalog.csv"
    score_catalog(tv).to_csv(cat_path, index=False)

    rows = []
    for k in sorted(fits["fold"]):
        res, _ = EV.evaluate(preds, cat_path, tv, split=f"f{k}", label="v4019", reps=1000, seed=0)
        r = res["pooled"]
        rows.append({"model": f"UNET_{preset}", "variant": VARIANT, "fold": int(k),
                     "held_site": fits.loc[fits["fold"] == k, "held_site"].iloc[0], "chips": r["n_chips"],
                     "mangrove_px": r["mangrove_px"], **{m: round(r[m], 4) for m in EV.METRICS},
                     "iou_ci95_lo": round(r["iou_ci95"][0], 4), "iou_ci95_hi": round(r["iou_ci95"][1], 4)})
    cv = pd.DataFrame(rows)

    # older runs (before the timing columns were added) just get blanks here
    def col(name, how="mean"):
        if name not in fits or fits[name].isna().all():
            return np.nan
        return float(getattr(fits[name], how)())

    summary = {"preset": preset, "base": cfg["base"], "params": cfg["params"], "folds": len(cv),
               "mean_iou": round(cv["iou"].mean(), 4), "iou_sd": round(cv["iou"].std(), 4),
               "train_min_per_fold": round(col("minutes"), 1),
               "predict_ms_per_chip": round(col("predict_ms_per_chip"), 1),
               "peak_gpu_mb": col("peak_gpu_mb", "max"), "device": cfg.get("device", "")}
    return summary, cv


def paired_vs_committed(cvs: dict, results: Path = REPO_RESULTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fold IoU table (committed full U-Net and RF + each small preset) and the paired test on it."""
    base = CM.load_fold_iou(results, [("U-Net (full)", "unet", "UNET"), ("Random forest", "baselines", "B2")])
    for preset, cv in cvs.items():
        col = cv.set_index("fold")
        if not col["held_site"].equals(base.loc[col.index, "held_site"]):
            raise ValueError(f"{preset} was scored on different folds than the committed runs")
        base[f"U-Net {preset}"] = col["iou"]
    base = base.dropna()
    return base, CM.pairwise(base)


def plot(table: pd.DataFrame, path: Path, full: tuple | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(table["params"], table["mean_iou"], yerr=table["iou_sd"], fmt="o-", capsize=3)
    for _, r in table.iterrows():
        ax.annotate(f"base {r['base']}", (r["params"], r["mean_iou"]), textcoords="offset points", xytext=(5, 5))
    if full is not None:                     # the committed full-size run, as the point to compare against
        ax.errorbar([FULL_UNET_PARAMS], [full[0]], yerr=[full[1]], fmt="s", color="k", capsize=3)
        ax.annotate("base 32 (committed)", (FULL_UNET_PARAMS, full[0]), textcoords="offset points",
                    xytext=(-60, 8))
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("mean IoU over 5 folds (± sd)")
    ax.set_title("U-Net size vs IoU")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--presets", nargs="+", default=PRESETS)
    ap.add_argument("--out", type=Path, help="write efficiency.csv, the plot and each preset's cv_results here")
    args = ap.parse_args(argv)

    summaries, cvs = [], {}
    for preset in args.presets:
        folder = P.RESULTS / f"Unet_{preset}"
        if not (folder / "fold_fits.csv").exists():
            print(f"skipped {preset}: no {folder / 'fold_fits.csv'} (not trained yet?)")
            continue
        summary, cv = score_preset(preset, P.TRAIN_VAL, P.RESULTS, P.WORK, P.TMP)
        summaries.append(summary)
        cvs[preset] = cv
        if args.out:
            (args.out / preset).mkdir(parents=True, exist_ok=True)
            cv.to_csv(args.out / preset / "cv_results.csv", index=False)
            # keep the training log next to the scores, so the run can be checked later
            for name in ("fold_fits.csv", "history.csv", "config.json"):
                src = folder / name
                if src.exists():
                    (args.out / preset / name).write_bytes(src.read_bytes())
    if not summaries:
        raise SystemExit("no trained presets found, run train_unet.py --preset <name> first")

    table = pd.DataFrame(summaries).sort_values("params", ascending=False)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out / "efficiency.csv", index=False)
        folds, pairs = paired_vs_committed(cvs)
        plot(table, args.out / "iou_vs_params.png", (folds["U-Net (full)"].mean(), folds["U-Net (full)"].std()))
        print(f"\nwrote {args.out / 'efficiency.csv'} and {args.out / 'iou_vs_params.png'}")
        folds.round(4).to_csv(args.out / "fold_iou.csv")
        pairs.round(4).to_csv(args.out / "pairwise.csv", index=False)
        print("\nPaired test against the committed full U-Net and RF\n")
        print(CM.to_markdown(folds.reset_index().round(3)))
        print()
        print(CM.to_markdown(pairs.round(3)))


if __name__ == "__main__":
    main()
