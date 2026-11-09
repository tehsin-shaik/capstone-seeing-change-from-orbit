"""Shared scoring for every model (Stage 6 of the master plan).

Scores predicted binary masks against GMW labels for the mangrove class:
IoU, F1, precision, recall — per site and pooled over pixels — with a 95%
bootstrap confidence interval obtained by resampling chips (1000 reps, seed 0).
Pixels where `valid` is False (cloud, shadow, nodata) are ignored.

Usage:
    python evaluate.py --pred preds/unet --catalog catalog.csv --chips chips/ \
        --split test --label v4019 --name unet --out results/unet
    python evaluate.py --compare results/ndvi/results.json results/unet/results.json \
        --out results/comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chipio import chip_path, get_label, load_catalog, load_chip, load_mask

METRICS = ("iou", "f1", "precision", "recall")


def confusion(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN counted over valid pixels only."""
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} != label shape {truth.shape}")
    p, t, v = pred.astype(bool), truth.astype(bool), valid.astype(bool)
    tp = int(np.sum(p & t & v))
    fp = int(np.sum(p & ~t & v))
    fn = int(np.sum(~p & t & v))
    tn = int(np.sum(~p & ~t & v))
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn) -> dict:
    """Pixel-pooled metrics. Undefined ratios (0/0) are returned as NaN."""
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec}


def bootstrap_ci(counts: np.ndarray, reps: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """counts: (n_chips, 3) array of TP, FP, FN. Resample chips with replacement."""
    n = len(counts)
    if n == 0:
        return {m: [float("nan"), float("nan")] for m in METRICS}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    sums = counts[idx].sum(axis=1)  # (reps, 3)
    m = metrics_from_counts(sums[:, 0], sums[:, 1], sums[:, 2])
    out = {}
    for k in METRICS:
        v = m[k][np.isfinite(m[k])]
        out[k] = [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))] if len(v) else [float("nan")] * 2
    return out


def summarise(df: pd.DataFrame, reps: int, seed: int) -> dict:
    counts = df[["tp", "fp", "fn"]].to_numpy(dtype=np.int64)
    tot = counts.sum(axis=0)
    point = metrics_from_counts(*tot)
    ci = bootstrap_ci(counts, reps=reps, seed=seed)
    res = {"n_chips": int(len(df)), "tp": int(tot[0]), "fp": int(tot[1]), "fn": int(tot[2]),
           "mangrove_px": int(tot[0] + tot[2])}
    for k in METRICS:
        res[k] = float(point[k])
        res[f"{k}_ci95"] = ci[k]
    return res


def evaluate(pred_dir, catalog, chips_dir, split="test", label="v4019", reps=1000, seed=0,
             sites=None) -> tuple[dict, pd.DataFrame]:
    cat = load_catalog(catalog)
    if split != "all":
        cat = cat[cat["split"] == split]
    if sites:
        cat = cat[cat["site"].isin(sites)]
    if cat.empty:
        raise SystemExit(f"no chips in catalog for split={split!r}")
    rows, skipped = [], []
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(chips_dir, r))
        truth, labelled = get_label(chip, label)
        use = chip["valid"] & labelled
        if not use.any():  # e.g. v4019 (2020 only) asked for a 2018 or 2025 chip
            skipped.append(r["chip_id"])
            continue
        pred = load_mask(pred_dir, r["chip_id"])
        tp, fp, fn, tn = confusion(pred, truth, use)
        rows.append({"chip_id": r["chip_id"], "site": r["site"], "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    if not rows:
        raise SystemExit(f"no chip in split={split!r} has usable GMW {label} labels")
    per_chip = pd.DataFrame(rows)
    pc = metrics_from_counts(per_chip.tp, per_chip.fp, per_chip.fn)
    per_chip["iou"] = pc["iou"]
    results = {"split": split, "label": label, "bootstrap_reps": reps, "seed": seed,
               "chips_skipped_no_label": skipped,
               "pooled": summarise(per_chip, reps, seed),
               "per_site": {s: summarise(g, reps, seed) for s, g in per_chip.groupby("site")}}
    return results, per_chip


def fmt(res: dict, k: str) -> str:
    v, (lo, hi) = res[k], res[f"{k}_ci95"]
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def markdown_table(results: dict, name: str = "model") -> str:
    lines = [f"### {name} — split `{results['split']}`, labels GMW {results['label']}", "",
             "| Site | Chips | Mangrove px | IoU [95% CI] | F1 [95% CI] | Precision | Recall |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for site, r in list(results["per_site"].items()) + [("**pooled**", results["pooled"])]:
        lines.append(f"| {site} | {r['n_chips']} | {r['mangrove_px']} | {fmt(r, 'iou')} | {fmt(r, 'f1')} "
                     f"| {r['precision']:.3f} | {r['recall']:.3f} |")
    return "\n".join(lines) + "\n"


def compare_table(paths: list[str]) -> str:
    lines = ["| Model | Split | Labels | IoU [95% CI] | F1 [95% CI] | Precision | Recall |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for p in paths:
        res = json.loads(Path(p).read_text())
        r = res["pooled"]
        lines.append(f"| {res.get('name', Path(p).parent.name)} | {res['split']} | {res['label']} | "
                     f"{fmt(r, 'iou')} | {fmt(r, 'f1')} | {r['precision']:.3f} | {r['recall']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="folder of predicted masks, one file per chip_id")
    ap.add_argument("--catalog", help="catalog.csv")
    ap.add_argument("--chips", help="chips folder")
    ap.add_argument("--split", default="test", help="train | val | test | external | all")
    ap.add_argument("--label", default="v4019", choices=["v4019", "v4112"])
    ap.add_argument("--name", default=None, help="model name shown in tables")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output folder (or .md file with --compare)")
    ap.add_argument("--compare", nargs="+", help="results.json files to put side by side")
    a = ap.parse_args()

    if a.compare:
        md = compare_table(a.compare)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
        print(md)
        return
    if not (a.pred and a.catalog and a.chips):
        ap.error("--pred, --catalog and --chips are required unless --compare is used")
    results, per_chip = evaluate(a.pred, a.catalog, a.chips, a.split, a.label, a.reps, a.seed)
    name = a.name or Path(a.pred).name
    results["name"] = name
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    per_chip.to_csv(out / "per_chip.csv", index=False)
    md = markdown_table(results, name)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
