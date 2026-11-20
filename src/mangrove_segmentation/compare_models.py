"""Compare every mangrove model fold by fold, on the same CV folds.

All the models were scored on the same 5 leave-one-site-out folds with evaluate.py, so the fair way to
compare two of them is to look at the IoU difference on each held-out site. Checking whether the
confidence intervals overlap doesn't really work, because every interval also includes the site-to-site
spread that all models share (Jubail is hard for all of them, Eastern Mangroves is easy for all of them).

For each pair of models this prints:
  - the mean IoU difference (A - B) and how many folds A wins
  - a 95% paired bootstrap interval. With 5 folds there are only 5^5 = 3125 possible resamples, so we
    just use all of them (no random seed needed)
  - an exact two-sided Wilcoxon p-value, plus the Holm-corrected one since we test all pairs

Note: with 5 folds the lowest possible Wilcoxon p is 2/32 = 0.0625, so nothing can get under 0.05.
The win count and the interval are more useful here. "No clear difference" means the folds can't
separate the models, not that they're the same.

We call a difference clear only if A wins every fold and the interval is above 0. The models that
aren't clearly worse than the best one make up the top group, and the simplest of them is the
suggested pick (simpler model on a tie).

Per-chip version (--per-chip): with only 5 folds the test above is weak, so if we have TP/FP/FN per
chip we can do a bootstrap over chips instead. Chips from the same spot on different dates are almost
the same picture, so we resample locations (all dates of a spot together), within each fold, and
recompute the mean-over-folds IoU each time. That keeps it the same number we report everywhere else.

Usage (from src/mangrove_segmentation):
  python compare_models.py                        # print the tables
  python compare_models.py --out ../../results/mangrove_segmentation/model_comparison
  python compare_models.py --per-chip ../../results/mangrove_segmentation/model_comparison/per_chip_scores.csv
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results" / "mangrove_segmentation"
VARIANT = "b_no_haze"          # hazy scenes dropped from training: the variant every model was run with

# (name in the tables, folder under results/mangrove_segmentation, "model" column value).
# Ordered simplest first. RF vs the SVMs is a judgement call (see model_size.py), so double check
# before quoting the suggested pick.
MODELS = [
    ("NDVI rule", "baselines", "B1"),
    ("SVM (linear)", "svm/linear", "SVM"),
    ("SVM (approx. RBF)", "svm/rbf_approx", "SVM"),
    ("Random forest", "baselines", "B2"),
    ("U-Net", "unet", "UNET"),
    ("DINOv3 frozen", "dinov3", "DINOv3"),
    ("DINOv3 LoRA", "dinov3_lora", "DINOv3_LoRA"),
]


def load_fold_iou(results: Path = RESULTS, models=MODELS, variant: str = VARIANT) -> pd.DataFrame:
    """Table of IoU per fold (rows) and model (columns).

    A missing cv_results.csv just gets skipped. If a model was run on different folds we stop,
    because pairing it with the others wouldn't make sense.
    """
    columns, sites = {}, None
    for name, folder, code in models:
        f = results / folder / "cv_results.csv"
        if not f.exists():
            print(f"skipped {name}: {f} not found")
            continue
        cv = pd.read_csv(f)
        cv = cv[(cv["model"] == code) & (cv["variant"] == variant)]
        if cv.empty:
            print(f"skipped {name}: no rows for model={code}, variant={variant} in {f}")
            continue
        if cv["fold"].duplicated().any():
            raise ValueError(f"{f} has more than one row per fold for model={code}, variant={variant}")
        cv = cv.set_index("fold").sort_index()
        if sites is None:
            sites = cv["held_site"]
        elif not cv["held_site"].equals(sites):
            raise ValueError(f"{name} was scored on different folds or sites than the models before it")
        columns[name] = cv["iou"].astype(float)
    if len(columns) < 2:
        raise ValueError("need cv_results for at least two models to compare")
    table = pd.DataFrame(columns)
    table.insert(0, "held_site", sites)
    return table


def exact_bootstrap_ci(diff: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """95% interval of the mean difference using every possible resample of the folds.

    Only works because we have so few folds (n^n resamples). Past 7 folds that's millions, so we
    stop instead of hanging. Use a normal random bootstrap if that ever happens.
    """
    n = len(diff)
    if n > 7:
        raise ValueError(f"exact bootstrap over {n} folds is too large; use a random bootstrap")
    idx = np.array(list(itertools.product(range(n), repeat=n)))
    means = diff[idx].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def wilcoxon_p(diff: np.ndarray) -> float:
    """Exact two-sided Wilcoxon p-value (1.0 if both models score exactly the same)."""
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(diff, alternative="two-sided", method="exact").pvalue)


def holm(p: pd.Series) -> pd.Series:
    """Holm correction, since we test every pair at once."""
    order = p.sort_values().index
    m = len(p)
    adjusted = (p[order] * np.arange(m, 0, -1)).cummax().clip(upper=1.0)
    return adjusted.reindex(p.index)


def pairwise(table: pd.DataFrame) -> pd.DataFrame:
    """Each pair of models once. A is always the one with the higher mean IoU."""
    models = [c for c in table.columns if c != "held_site"]
    rows = []
    for a, b in itertools.combinations(models, 2):
        if table[b].mean() > table[a].mean():
            a, b = b, a
        diff = (table[a] - table[b]).to_numpy()
        lo, hi = exact_bootstrap_ci(diff)
        rows.append(dict(model_a=a, model_b=b, mean_diff=diff.mean(), ci95_lo=lo, ci95_hi=hi,
                         a_wins=int((diff > 0).sum()), folds=len(diff), wilcoxon_p=wilcoxon_p(diff)))
    out = pd.DataFrame(rows)
    out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"])
    # need both: one big fold can push the interval above 0 on its own
    out["clear_difference"] = (out["a_wins"] == out["folds"]) & (out["ci95_lo"] > 0)
    return out


def top_group(table: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    """Best model plus everything it doesn't clearly beat, simplest first."""
    models = [c for c in table.columns if c != "held_site"]
    best = table[models].mean().idxmax()
    clearly_worse = set(pairs.loc[(pairs["model_a"] == best) & pairs["clear_difference"], "model_b"])
    return [m for m in models if m not in clearly_worse]


# model names used in per_chip_scores.csv -> names used in the tables above
PER_CHIP_NAMES = {
    "random_forest": "Random forest",
    "svm_rbf": "SVM (approx. RBF)",
    "unet": "U-Net",
    "dinov3_frozen": "DINOv3 frozen",
    "dinov3_lora_40ep": "DINOv3 LoRA",
}


def location_of(chip_id: str) -> str:
    """Drop the date from a chip id, so the same spot on different dates gets one key."""
    return re.sub(r"_\d{4}-\d{2}-\d{2}_", "_", chip_id)


def load_per_chip(path: Path) -> tuple[list[str], list[np.ndarray]]:
    """Per fold, an array of shape (locations, models, 3) with summed TP, FP, FN.

    Every model has to have scored the exact same chips, otherwise the pairing is off.
    """
    pc = pd.read_csv(path)
    pc["model"] = pc["model"].map(lambda m: PER_CHIP_NAMES.get(m, m))
    models = list(dict.fromkeys(pc["model"]))
    chips = pc.groupby("model")["chip_id"].apply(frozenset)
    if chips.nunique() != 1:
        raise ValueError("models in the per-chip file don't cover the same chips")
    if pc.duplicated(["model", "chip_id"]).any():
        raise ValueError("a chip appears twice for the same model in the per-chip file")
    pc["location"] = pc["chip_id"].map(location_of)
    sums = pc.groupby(["fold", "location", "model"])[["TP", "FP", "FN"]].sum()
    folds = []
    for _, g in sums.groupby(level="fold"):
        cube = g.unstack("model").reorder_levels([1, 0], axis=1)   # columns: (model, TP/FP/FN)
        folds.append(np.stack([cube[m][["TP", "FP", "FN"]].to_numpy(float) for m in models], axis=1))
    return models, folds


def mean_fold_iou(folds: list[np.ndarray], weights: list[np.ndarray] | None = None) -> np.ndarray:
    """Mean over folds of the pooled IoU, for every model (and every bootstrap rep if weights given).

    weights[f] has shape (reps, locations): how many times each location was drawn.
    """
    per_fold = []
    for f, cube in enumerate(folds):
        if weights is None:
            tp, fp, fn = cube.sum(axis=0).T                              # each (models,)
        else:
            tot = np.einsum("rl,lmk->rmk", weights[f], cube)             # (reps, models, 3)
            tp, fp, fn = tot[..., 0], tot[..., 1], tot[..., 2]
        per_fold.append(tp / np.maximum(tp + fp + fn, 1))
    return np.mean(per_fold, axis=0)


def per_chip_pairwise(models: list[str], folds: list[np.ndarray], reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Location bootstrap within each fold. A is the model with the higher observed mean IoU."""
    rng = np.random.default_rng(seed)
    weights = [rng.multinomial(len(c), np.full(len(c), 1 / len(c)), size=reps) for c in folds]
    observed = mean_fold_iou(folds)
    boot = mean_fold_iou(folds, weights)                                  # (reps, models)
    rows = []
    for i, j in itertools.combinations(range(len(models)), 2):
        if observed[j] > observed[i]:
            i, j = j, i
        d = boot[:, i] - boot[:, j]
        lo, hi = np.quantile(d, [0.025, 0.975])
        rows.append(dict(model_a=models[i], model_b=models[j], mean_diff=observed[i] - observed[j],
                         ci95_lo=lo, ci95_hi=hi, share_a_better=(d > 0).mean(),
                         clear_difference=bool(lo > 0)))
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Quick Markdown table, so we don't need the 'tabulate' package."""
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=RESULTS, help="folder holding each model's cv_results.csv")
    ap.add_argument("--out", type=Path, help="also write fold_iou.csv and pairwise.csv here")
    ap.add_argument("--per-chip", type=Path, help="per-chip TP/FP/FN file: also run the location bootstrap")
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap reps for --per-chip (default 2000)")
    args = ap.parse_args(argv)

    table = load_fold_iou(args.results)
    pairs = pairwise(table)
    group = top_group(table, pairs)

    print("IoU per held-out fold\n")
    print(to_markdown(table.reset_index().round(3)))
    print("\nPaired comparison (A = higher mean IoU)\n")
    show = pairs.copy()
    # + 0.0 gets rid of the -0.0 that rounding sometimes leaves
    show[["mean_diff", "ci95_lo", "ci95_hi"]] = show[["mean_diff", "ci95_lo", "ci95_hi"]].round(3) + 0.0
    show[["wilcoxon_p", "wilcoxon_p_holm"]] = show[["wilcoxon_p", "wilcoxon_p_holm"]].round(4)
    print(to_markdown(show))
    print(f"\nTop group (no clear difference from the best): {', '.join(group)}")
    print(f"Simplest model in the top group: {group[0]}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.out / "fold_iou.csv")
        pairs.round(4).to_csv(args.out / "pairwise.csv", index=False)
        print(f"\nwrote {args.out / 'fold_iou.csv'} and {args.out / 'pairwise.csv'}")

    if args.per_chip:
        models, folds = load_per_chip(args.per_chip)
        n_loc = sum(len(c) for c in folds)
        print(f"\nPer-chip test: {len(models)} models, {n_loc} locations, {args.reps} location bootstrap reps\n")
        means = pd.Series(mean_fold_iou(folds), index=models).round(4)
        print(to_markdown(means.rename("mean IoU").reset_index().rename(columns={"index": "model"})))
        chip_pairs = per_chip_pairwise(models, folds, args.reps)
        show = chip_pairs.copy()
        cols = ["mean_diff", "ci95_lo", "ci95_hi", "share_a_better"]
        show[cols] = show[cols].round(3) + 0.0
        print()
        print(to_markdown(show))
        if args.out:
            chip_pairs.round(4).to_csv(args.out / "per_chip_pairwise.csv", index=False)
            print(f"\nwrote {args.out / 'per_chip_pairwise.csv'}")


if __name__ == "__main__":
    main()
