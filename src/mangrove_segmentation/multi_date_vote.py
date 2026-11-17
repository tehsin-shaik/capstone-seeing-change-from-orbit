"""Does combining several 2020 dates of the same ground cell raise the IoU against GMW v4.0.19?

For each held-out site, a model's masks of the non-hazy 2020 dates of one chip cell are combined by a
per-pixel majority vote, and the combined mask is scored once per cell. The combining step never looks at
the labels. It uses the masks and, when the dates are tied, the model's own score against the threshold
its fold chose during tuning.

Models, all with hazy scenes dropped from training:
  RF    random forest (B2), retrained here per fold so that its probabilities are available
        (the training steps are those of src/change_detection/common.py)
  SVM   approximate RBF SVM (B3), retrained here per fold so that its scores are available
        (the same steps as `baseline_svm.py cv --kernel rbf_approx`)
  LoRA  DINOv3 + LoRA, from its saved masks (--lora-preds). No probabilities were saved for it.
If --rf-preds or --svm-preds point at masks saved by the original runs, the retrained masks are compared
with them and the script says whether they are identical.

Ties only happen with an even number of dates. Three rules are reported:
  prob           the mean probability or score of the dates against the fold's threshold (RF and SVM only)
  not_mangrove   a tie counts as not mangrove
  mangrove       a tie counts as mangrove

Same cells, same pixels: for every cell the scored pixels are those usable on all of its non-hazy dates and
labelled. Single dates, pairs and the full vote are scored on exactly those pixels. Each cell counts once:
when a cell has several single dates (or pairs), their counts are averaged.

Usage (from src/mangrove_segmentation, CPU is enough):
  python multi_date_vote.py --train-val <Train_Val_Dataset> --lora-preds <lora run>/preds/DINO_b_no_haze \
      --out <output folder> [--rf-preds <folder>] [--svm-preds <folder>]
Keep the output folder outside the repository.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "change_detection"))

import baseline_svm as B3  # noqa: E402
import common as CD  # noqa: E402  (src/change_detection: catalog with haze flag, random forest fold training)
from baseline_rf import collect, pixels  # noqa: E402
from chipio import chip_path, load_chip, load_mask  # noqa: E402

MODELS = ["RF", "SVM", "LoRA"]
RULES = ["prob", "not_mangrove", "mangrove"]
MAIN_RULE = {"RF": "prob", "SVM": "prob", "LoRA": "not_mangrove"}      # the rule used in the summary tables
LABEL = "v4019"


def score_maps(model, held, tv, th, saved=None):
    """chip_id -> (score map with NaN where unusable, threshold). Also checks against saved masks if given."""
    out, same = {}, True
    for _, r in held.iterrows():
        chip = load_chip(chip_path(tv, r))
        v = chip["valid"]
        s = np.full(v.shape, np.nan, np.float32)
        if v.any():
            s[v] = model(pixels(chip)[v.ravel()])
        out[r["chip_id"]] = (s, th)
        if saved:
            same &= bool(((np.nan_to_num(s, nan=-1e9) > th) == load_mask(saved, r["chip_id"])).all())
    return out, (same if saved else None)


def rf_scores(cat20, tv, work, saved=None):
    out = {}
    for k in sorted(int(f) for f in cat20.fold.unique()):
        fit = CD.train_fold(cat20, k, tv, work, "b")
        rf = fit["rf"]
        pos = list(rf.classes_).index(True)
        maps, same = score_maps(lambda X: rf.predict_proba(X)[:, pos], fit["held"], tv, fit["threshold"], saved)
        out.update(maps)
        print(f"random forest fold {k}: threshold {fit['threshold']}"
              + ("" if same is None else f"; masks identical to the saved ones: {same}"), flush=True)
    return out


def svm_scores(cat20, tv, saved=None, seed=0):
    out = {}
    present = sorted(int(f) for f in cat20.fold.unique())
    for k in present:
        held = cat20[cat20.fold == k]
        train = CD.drop_near(cat20[(cat20.fold != k) & ~cat20.hazy], held)
        inner = next(f for f in CD.INNER_PREF + present if f != k and f in present)
        iv = train[train.fold == inner]
        it = CD.drop_near(train[train.fold != inner], iv)
        rng = np.random.default_rng(seed)
        Xtr, ytr = collect(it, tv, LABEL, 100_000, rng, balanced=True)
        Xva, yva = collect(iv, tv, LABEL, 300_000, rng, balanced=False)
        best = B3.tune(Xtr, ytr, Xva, yva, "rbf_approx", seed)
        rng = np.random.default_rng(seed)
        Xall, yall = collect(train, tv, LABEL, 200_000, rng, balanced=True)
        svm = B3.make_model(kernel="rbf_approx", seed=seed, **best["settings"]).fit(Xall, yall)
        maps, same = score_maps(svm.decision_function, held, tv, best["threshold"], saved)
        out.update(maps)
        print(f"SVM fold {k}: C {best['settings']['C']}, threshold {best['threshold']}"
              + ("" if same is None else f"; masks identical to the saved ones: {same}"), flush=True)
    return out


def vote(masks, scores, th, rule):
    """Majority vote of k masks on pixels where all k are usable. No labels involved."""
    k = len(masks)
    votes = np.sum(masks, axis=0)
    out = votes * 2 > k
    tie = votes * 2 == k
    if tie.any():
        if rule == "mangrove":
            out |= tie
        elif rule == "prob":
            out |= tie & (np.mean(scores, axis=0) > th)
    return out


def cell_rows(use_cat, tv, sc, lora_preds) -> pd.DataFrame:
    rows = []
    for (site, row, col), g in use_cat.groupby(["site", "row", "col"]):
        g = g.sort_values("date")
        n = len(g)
        chips = [load_chip(chip_path(tv, r)) for _, r in g.iterrows()]
        lab = chips[0]["label_v4019"]
        assert all((c["label_v4019"] == lab).all() for c in chips)          # the label is the same on every date
        V = np.all([c["valid"] for c in chips], axis=0) & (lab != 255)        # usable on all dates, and labelled
        truth = (lab == 1)[V]
        ids = list(g["chip_id"])
        for m in MODELS:
            if m in sc:
                th = sc[m][ids[0]][1]
                score = [np.nan_to_num(sc[m][i][0], nan=-1e9)[V] for i in ids]
                masks = [s > th for s in score]
            else:
                th, score = None, None
                masks = [load_mask(lora_preds, i)[V] for i in ids]
            for k in range(1, n + 1):
                subsets = list(itertools.combinations(range(n), k))
                for rule in RULES:
                    if rule == "prob" and score is None:
                        continue
                    tp = fp = fn = 0.0
                    for sub in subsets:
                        p = vote([masks[i] for i in sub], [score[i] for i in sub] if score else None, th, rule)
                        tp += (p & truth).sum(); fp += (p & ~truth).sum(); fn += (~p & truth).sum()
                    rows.append({"fold": int(g["fold"].iloc[0]), "site": site, "cell": f"{site}_{row}_{col}", "dates_of_cell": n,
                                 "model": m, "dates_used": k, "tie_rule": rule, "subsets": len(subsets), "scored_px": int(V.sum()),
                                 "tp": tp / len(subsets), "fp": fp / len(subsets), "fn": fn / len(subsets)})
    return pd.DataFrame(rows)


def scores(d) -> pd.Series:
    tp, fp, fn = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
    return pd.Series({"cells": d["cell"].nunique(), "iou": tp / (tp + fp + fn),
                      "precision": tp / (tp + fp), "recall": tp / (tp + fn)})


def table(d, by) -> pd.DataFrame:
    t = d.groupby([by, "model", "tie_rule", "dates_used"], sort=False).apply(scores, include_groups=False).reset_index()
    t["cells"] = t["cells"].astype(int)
    return t


def mean_over_folds(t) -> pd.DataFrame:
    g = t.groupby(["model", "tie_rule", "dates_used"], sort=False)
    m = g[["iou", "precision", "recall"]].mean().round(3)
    m["folds"] = g.size()
    m["cells"] = g["cells"].sum()
    return m.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-val", required=True, help="the Train_Val_Dataset folder")
    ap.add_argument("--lora-preds", required=True, help="saved DINOv3 + LoRA masks, e.g. <lora run>/preds/DINO_b_no_haze")
    ap.add_argument("--out", required=True, help="output folder, outside the repository")
    ap.add_argument("--rf-preds", default=None, help="optional: saved random forest masks to compare the retrained ones with")
    ap.add_argument("--svm-preds", default=None, help="optional: saved RBF SVM masks to compare the retrained ones with")
    ap.add_argument("--qa", default=None, help="haze table (default: hazy_scenes.csv inside --train-val)")
    a = ap.parse_args()
    tv, out = Path(a.train_val), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cat = CD.load_full_catalog(tv, a.qa)
    cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
    use_cat = cat20[~cat20.hazy]
    sc = {"RF": rf_scores(cat20, tv, out / "work", a.rf_preds), "SVM": svm_scores(cat20, tv, a.svm_preds)}
    cells = cell_rows(use_cat, tv, sc, a.lora_preds)
    cells.to_csv(out / "per_cell.csv", index=False)

    # A. one date against all non-hazy dates of each cell, every cell (a cell with one date is unchanged)
    A = pd.concat([cells[cells["dates_used"] == 1].assign(dates_used="one"),
                   cells[cells["dates_used"] == cells["dates_of_cell"]].assign(dates_used="all")])
    # B. growth with the number of dates, on the cells that have three non-hazy dates
    B = cells[cells["dates_of_cell"] == 3]
    # C. the cells with exactly two non-hazy dates
    C = cells[cells["dates_of_cell"] == 2]
    tabs = {"one_vs_all_mean_over_folds": mean_over_folds(table(A, "fold")), "one_vs_all_per_site": table(A, "site").round(4),
            "growth_3date_cells_mean_over_folds": mean_over_folds(table(B, "fold")),
            "growth_3date_cells_per_site": table(B, "site").round(4), "two_date_cells_per_site": table(C, "site").round(4)}
    n_dates = use_cat.groupby(["site", "row", "col"]).size().reset_index(name="dates")
    tabs["cells_per_site"] = n_dates.groupby(["site", "dates"]).size().unstack(fill_value=0).add_prefix("cells_with_dates_").reset_index()
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)

    def main_rule(t):
        return t[[MAIN_RULE[m] == r for m, r in zip(t["model"], t["tie_rule"])]]
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print("\nCELLS PER SITE BY NUMBER OF NON-HAZY 2020 DATES\n" + tabs["cells_per_site"].to_string(index=False))
        print("\nONE DATE vs ALL NON-HAZY DATES, every cell, mean over folds\n" + tabs["one_vs_all_mean_over_folds"].to_string(index=False))
        print("\nper site, IoU\n" + main_rule(tabs["one_vs_all_per_site"]).pivot_table(
            index=["site", "cells"], columns=["model", "dates_used"], values="iou").round(3).to_string())
        print("\nGROWTH ON CELLS WITH THREE NON-HAZY DATES, mean over folds\n" + tabs["growth_3date_cells_mean_over_folds"].to_string(index=False))
        print("\nper site, IoU\n" + tabs["growth_3date_cells_per_site"].pivot_table(
            index=["site", "cells", "model", "tie_rule"], columns="dates_used", values="iou").round(3).to_string())
    print("tables ->", out)


if __name__ == "__main__":
    main()
