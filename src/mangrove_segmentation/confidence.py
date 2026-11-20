"""How sure are we of the IoU numbers: spread across dates and sites, bootstrap intervals, and repeat seeds.

WHY
  Each model has one mean IoU over the 5 held-out folds. This script puts a range around it in three ways:
  the spread over image dates and sites (tables A and B), a bootstrap over the held-out chips (tables C and D),
  and training again with other seeds (table E). All scores are against GMW v4.0.19 on the 2020 chips,
  hazy scenes dropped from training (variant b). Only Train_Val_Dataset is read.

WHAT IS REUSED (no model is written again here)
  * Scoring: evaluate.py, called one fold at a time the way the team's score.py calls it.
  * NDVI rule: baseline_ndvi.py, thresholds fitted on the training folds as in run_cv.py.
  * Random forest and SVM with another seed: rgb_only.py (run_rf, use_channels) and baseline_svm.run_cv.
    The seed sets which pixels are sampled and the model's own random numbers.
  * U-Net and DINOv3 + LoRA with another seed: train_unet.py (gpu preset) and dinov3/train_dinov3_lora.py
    (default settings), files not edited. Both seed fold k with the number k. Seed N here means fold k is
    seeded with k + 100 * N, so seed 0 is the committed setting.

COMMANDS
  python confidence.py scores --train-val <Train_Val_Dataset> --per-chip <per_chip_scores_with_rgb.csv>
      --linear-svm-masks <folder of masks> --out <folder outside the repo>
      adds the NDVI rule and the linear SVM to the per-chip file -> <out>/per_chip_scores_all.csv
  python confidence.py tables --per-chip <per_chip_scores_all.csv> --out results/mangrove_segmentation/confidence
      tables A to D
  python confidence.py seed --model rf|svm|unet|lora --channels all8|rgb --seed N --train-val <Train_Val_Dataset>
      --out <runs folder outside the repo>
      lora takes --channels all8 (8 channel decoder branch), nospec (branch removed) or rgbbranch (branch fed RGB)
      unet and lora: set MANGROVE_ROOT to a folder outside the repo first
  python confidence.py seed-table --runs <runs folder> --out results/mangrove_segmentation/confidence
      table E
The per-chip file has one row per held-out chip and model: chip_id, site, date, fold, model, TP, FP, FN, channels.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_RESULTS = HERE.parents[1] / "results" / "mangrove_segmentation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "change_detection"))

V = "b_no_haze"
REPS, BOOT_SEED = 2000, 0
MIXED = "rgb_into_dinov3_all8_into_decoder_branch"
# model and channels in the per-chip file -> name in the tables
MODELS = [
    ("ndvi_rule", "ndvi_mndwi", "NDVI rule"),
    ("svm_linear", "all8", "SVM linear"),
    ("svm_rbf", "all8", "SVM RBF"),
    ("random_forest", "all8", "Random forest"),
    ("unet", "all8", "U-Net"),
    ("dinov3_lora_40ep", MIXED, "DINOv3 + LoRA"),
    ("dinov3_frozen", MIXED, "DINOv3 frozen"),
    ("random_forest", "rgb", "Random forest, RGB"),
    ("svm_rbf", "rgb", "SVM RBF, RGB"),
    ("dinov3_lora_40ep_no_decoder_branch", "rgb", "DINOv3 + LoRA, RGB, branch removed"),
    ("dinov3_lora_40ep_decoder_branch_rgb", "rgb", "DINOv3 + LoRA, RGB, branch fed RGB"),
    ("dinov3_frozen_no_decoder_branch", "rgb", "DINOv3 frozen, RGB, branch removed"),
]
PAIRS = [   # table D: first minus second
    ("Random forest", "DINOv3 + LoRA"), ("Random forest", "U-Net"), ("Random forest", "SVM RBF"),
    ("SVM RBF", "SVM linear"),
    ("Random forest", "Random forest, RGB"), ("SVM RBF", "SVM RBF, RGB"),
    ("DINOv3 + LoRA", "DINOv3 + LoRA, RGB, branch removed"), ("DINOv3 + LoRA", "DINOv3 + LoRA, RGB, branch fed RGB"),
    ("DINOv3 frozen", "DINOv3 frozen, RGB, branch removed"),
]
# every site has four 2020 images; the exact day differs by site, so they are grouped by time of year
SLOT = {1: "Jan-Feb", 2: "Jan-Feb", 4: "Apr-May", 5: "Apr-May", 8: "Aug-Sep", 9: "Aug-Sep", 11: "Nov"}
# table E: name, run folder inside --runs (plus _seed<N>), where the seed 0 fold numbers are committed
SEED_ROWS = [
    ("Random forest", "rf_all8", ("baselines/cv_results.csv", "B2")),
    ("Random forest, RGB", "rf_rgb", "Random forest"),
    ("SVM RBF", "svm_rbf_all8", ("svm/rbf_approx/cv_results.csv", "SVM")),
    ("SVM RBF, RGB", "svm_rbf_rgb", "SVM, approximate RBF"),
    ("U-Net", "unet", ("unet/cv_results.csv", "UNET")),
    ("DINOv3 + LoRA", "dinov3/lora_all8", ("dinov3_lora/cv_results.csv", "DINOv3_LoRA")),
    ("DINOv3 + LoRA, RGB, branch removed", "dinov3/lora_nospec", "DINOv3 + LoRA, decoder branch removed"),
    ("DINOv3 + LoRA, RGB, branch fed RGB", "dinov3/lora_rgbbranch", "DINOv3 + LoRA, decoder branch fed RGB"),
]


# ----------------------------------------------------------------- per-chip counts from saved masks
def cv_catalog(tv: Path) -> pd.DataFrame:
    cat = pd.read_csv(tv / "catalog.csv")
    cat = cat[cat["year"] == 2020].copy()
    assert set(cat["role"]) == {"cv"}, "a site that is not a cross-validation site is in the catalog"
    cat["fold"] = cat["fold"].astype(int)
    return cat.reset_index(drop=True)


def score_masks(pred: Path, tv: Path, work: Path) -> pd.DataFrame:
    """TP, FP, FN of every held-out 2020 chip for one folder of masks."""
    import evaluate as EV
    cat = cv_catalog(tv)
    work.mkdir(parents=True, exist_ok=True)
    scp = work / "score_catalog.csv"
    cat.assign(split="f" + cat.fold.astype(str)).to_csv(scp, index=False)
    parts = []
    for k in sorted(cat.fold.unique()):
        res, pc = EV.evaluate(pred, scp, tv, split=f"f{k}", label="v4019", reps=1, seed=0)
        assert not res["chips_skipped_no_label"]
        parts.append(pc.assign(fold=int(k)))
    pc = pd.concat(parts, ignore_index=True)
    assert len(pc) == len(cat), f"{pred}: {len(pc)} chips scored, {len(cat)} expected"
    return pc.merge(cat[["chip_id", "date"]], on="chip_id", validate="one_to_one")


def fold_table(pc: pd.DataFrame) -> pd.DataFrame:
    g = pc.groupby("fold")
    t = g[["tp", "fp", "fn"]].sum()
    return pd.DataFrame({"held_site": g.site.agg(lambda s: ",".join(sorted(set(s)))), "chips": g.size(),
                         "mangrove_px": t.tp + t.fn, "iou": (t.tp / (t.tp + t.fp + t.fn)).round(4)}).reset_index()


def committed(table, tag=None) -> pd.Series:
    """Fold IoU of a committed run. `table` is a cv_results.csv and its model tag, or a model name in the RGB summary."""
    if tag is None:
        s = pd.read_csv(REPO_RESULTS / "rgb_only" / "summary.csv")
        r = s[(s.model == table) & s.channels.str.startswith("RGB (")].iloc[0]
        return pd.Series({k: float(r[f"iou_fold{k}"]) for k in range(1, 6)})
    c = pd.read_csv(REPO_RESULTS / table)
    return c[(c.model == tag) & (c.variant == V)].set_index("fold").iou.astype(float)


def scores(a) -> None:
    import baseline_ndvi as B1
    import common as CD
    from chipio import chip_path, load_chip, save_mask
    tv, out = Path(a.train_val), Path(a.out)
    work = out / "work"; work.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(a.per_chip)
    cat = CD.load_full_catalog(tv)
    cat = cat[cat["year"] == 2020].reset_index(drop=True)
    assert set(cat["role"]) == {"cv"} and set(old.chip_id) == set(cat.chip_id)
    # NDVI rule: no masks were kept, so the two thresholds are fitted again on the training folds (as run_cv.py)
    ndvi = out / "preds" / f"B1_{V}"
    for k in sorted(cat.fold.unique()):
        held = cat[cat.fold == k]
        train = CD.drop_near(cat[(cat.fold != k) & ~cat.hazy], held)
        p = work / f"b1_fold{k}.csv"
        train.assign(split="val").to_csv(p, index=False)
        m = B1.fit(p, tv, "v4019", seed=0)
        for _, r in held.iterrows():
            save_mask(ndvi, r.chip_id, B1.predict_chip(load_chip(chip_path(tv, r)), m["t_ndvi"], m["w_mndwi"]))
        print(f"NDVI rule fold {k}: NDVI > {m['t_ndvi']}, MNDWI < {m['w_mndwi']}", flush=True)
    rows = [old]
    for name, channels, pred, table, tag in [("ndvi_rule", "ndvi_mndwi", ndvi, "baselines/cv_results.csv", "B1"),
                                             ("svm_linear", "all8", Path(a.linear_svm_masks), "svm/linear/cv_results.csv", "SVM")]:
        pc = score_masks(pred, tv, work)
        both = pd.DataFrame({"this": fold_table(pc).set_index("fold").iou, "committed": committed(table, tag)})
        same = bool((both["this"] == both["committed"]).all())
        print(f"\n{name}: fold IoU against the committed table (all the same: {same})\n{both.to_string()}")
        rows.append(pd.DataFrame({"chip_id": pc.chip_id, "site": pc.site, "date": pc.date, "fold": pc.fold, "model": name,
                                  "TP": pc.tp, "FP": pc.fp, "FN": pc.fn, "IoU": pc.iou,
                                  "run_source": "matches_committed_run" if same else "rerun_differs_from_committed",
                                  "channels": channels}))
    allrows = pd.concat(rows, ignore_index=True)
    assert not allrows.duplicated(["chip_id", "model", "channels"]).any()
    allrows.to_csv(out / "per_chip_scores_all.csv", index=False, float_format="%.6f")
    print(f"\nwrote {out / 'per_chip_scores_all.csv'}: {len(old)} old rows + {len(allrows) - len(old)} new")


# ----------------------------------------------------------------- tables A to D
def load_per_chip(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    names = {(m, c): n for m, c, n in MODELS}
    d["name"] = [names.get(k) for k in zip(d.model, d.channels)]
    d = d[d.name.notna()].copy()
    d["slot"] = pd.to_datetime(d.date).dt.month.map(SLOT)
    d["place"] = d.site + d.chip_id.str.extract(r"(_r\d+_c\d+)$")[0]      # the same ground at every date
    assert d.slot.notna().all() and d.place.notna().all()
    chips = d.groupby("name").chip_id.agg(lambda s: tuple(sorted(s)))
    assert chips.nunique() == 1, "the models were not scored on the same chips"
    return d


def order(names) -> list:
    return [n for *_, n in MODELS if n in set(names)]


def pooled(d: pd.DataFrame, by: list) -> pd.DataFrame:
    """Pixel-pooled IoU of every group: rows are the groups, columns the models."""
    g = d.groupby(by + ["name"])[["TP", "FP", "FN"]].sum()
    t = (g.TP / (g.TP + g.FP + g.FN)).unstack("name")
    return t[order(t.columns)]


def spread(t: pd.DataFrame, label=str) -> pd.DataFrame:
    """Mean, sd, min and max of each model's column, and which group the min and the max are."""
    return pd.DataFrame({"mean": t.mean(), "sd": t.std(), "min": t.min(), "max": t.max(),
                         "lowest": t.idxmin().map(label), "highest": t.idxmax().map(label)}).rename_axis("model")


def bootstrap(d: pd.DataFrame, unit: str, reps=REPS, seed=BOOT_SEED):
    """Mean IoU over the folds for `reps` resamples, one row per model.

    Inside each held-out site the units are drawn with replacement, as many as the site has. Every model gets
    the same draw, so two rows can be subtracted (paired). unit 'chip_id' draws single chips; 'place' draws
    chip positions and keeps all the dates of a position together."""
    rng = np.random.default_rng(seed)
    names, folds = order(d.name), {}
    for (fold, _), g in d.groupby(["fold", "site"]):
        c = g.groupby(["name", unit])[["TP", "FP", "FN"]].sum()
        arr = np.stack([c.loc[n].sort_index().to_numpy(float) for n in names])          # models x units x 3
        idx = rng.integers(0, arr.shape[1], size=(reps, arr.shape[1]))
        folds[fold] = folds.get(fold, 0) + arr[:, idx, :].sum(axis=2)                   # models x reps x 3
    iou = np.stack([s[..., 0] / s.sum(axis=-1) for s in folds.values()])                # folds x models x reps
    return names, iou.mean(axis=0)


def interval(x) -> tuple:
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def tables(a) -> None:
    d, out = load_per_chip(a.per_chip), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chips = d[d.name == d.name.iloc[0]]

    # A: every site and date
    sd = pooled(d, ["site", "date"])
    n = chips.groupby(["site", "date"]).size().rename("chips")
    sd.round(4).join(n).reset_index().to_csv(out / "table_A_site_date_iou.csv", index=False)
    A = spread(sd, lambda k: f"{k[0]} {k[1]}")
    A.insert(0, "site_dates", sd.notna().sum())
    A.round(4).to_csv(out / "table_A.csv")
    # B: by time of year only, and by site only
    B = {}
    for by, f in (("slot", "table_B_by_date.csv"), ("site", "table_B_by_site.csv")):
        t = pooled(d, [by])
        t = t.loc[[s for s in dict.fromkeys(SLOT.values()) if s in t.index]] if by == "slot" else t
        B[by] = pd.concat([t.T.rename_axis("model"), spread(t)], axis=1).round(4)
        B[by].to_csv(out / f)
    # C and D: bootstrap over chips (asked for), and over chip positions next to it
    point = pooled(d, ["fold"]).mean()
    names, chip = bootstrap(d, "chip_id")
    _, place = bootstrap(d, "place")
    C = []
    for i, m in enumerate(names):
        (lo, hi), (plo, phi) = interval(chip[i]), interval(place[i])
        C.append({"model": m, "mean_iou_5_folds": point[m], "bootstrap_mean": chip[i].mean(), "bootstrap_sd": chip[i].std(ddof=1),
                  "ci95_lo": lo, "ci95_hi": hi, "ci95_width": hi - lo,
                  "positions_sd": place[i].std(ddof=1), "positions_ci95_lo": plo, "positions_ci95_hi": phi,
                  "positions_ci95_width": phi - plo})
    C = pd.DataFrame(C).round(4)
    C.to_csv(out / "table_C_bootstrap.csv", index=False)
    D = []
    for x, y in PAIRS:
        if x not in names or y not in names:
            print("pair skipped, not in the per-chip file:", x, "|", y)
            continue
        i, j = names.index(x), names.index(y)
        (lo, hi), (plo, phi) = interval(chip[i] - chip[j]), interval(place[i] - place[j])
        D.append({"first": x, "second": y, "first_minus_second": point[x] - point[y],
                  "bootstrap_mean": (chip[i] - chip[j]).mean(), "ci95_lo": lo, "ci95_hi": hi,
                  "includes_zero": bool(lo <= 0 <= hi),
                  "positions_ci95_lo": plo, "positions_ci95_hi": phi, "positions_includes_zero": bool(plo <= 0 <= phi)})
    D = pd.DataFrame(D).round(4)
    D.to_csv(out / "table_D_paired_bootstrap.csv", index=False)
    with pd.option_context("display.width", 250):
        print(f"{d.chip_id.nunique()} chips, {len(sd)} site and date combinations, {d.place.nunique()} chip positions, "
              f"{REPS} resamples\n")
        for title, t in (("A", A.round(3).reset_index()), ("B by date", B["slot"].round(3).reset_index()),
                         ("B by site", B["site"].round(3).reset_index()), ("C", C.round(3)), ("D", D.round(3))):
            print(f"TABLE {title}\n{t.to_markdown(index=False)}\n")
        print("site and date IoU\n", sd.round(3).join(n).to_string())
    print("wrote tables A to D in", out)


# ----------------------------------------------------------------- other seeds (table E)
def shifted_numpy(shift: int):
    """numpy as a training script sees it, with default_rng(fold) turned into default_rng(fold + shift)."""
    rnd = types.SimpleNamespace(seed=np.random.seed, default_rng=lambda s: np.random.default_rng(s + shift))

    class Numpy:
        random = rnd

        def __getattr__(self, name):
            return getattr(np, name)
    return Numpy()


def seed_run(a) -> None:
    tv, runs, s = Path(a.train_val), Path(a.out), a.seed
    if a.model == "unet":
        import train_unet as U
        assert HERE.parents[1] not in U.P.WORK.parents, "set MANGROVE_ROOT to a folder outside the repository"
        U.TV, U.QA_CSV = tv, tv / "hazy_scenes.csv"            # its docstring allows these two overrides
        first = U.seed_all
        U.seed_all = lambda k: first(k + 100 * s)
        U.np = shifted_numpy(100 * s)
        out = runs / f"unet_seed{s}"
        U.run("gpu", out)
        cv = fold_table(score_masks(U.run_dir(out) / "preds" / f"UNET_{V}", tv, out / "work"))
        cv.insert(0, "variant", V); cv.insert(0, "model", "UNET")
        cv.to_csv(out / "cv_results.csv", index=False)
    elif a.model == "lora":
        sys.path.insert(0, str(HERE / "dinov3"))
        import train_dinov3_lora as L
        first = L.T.seed_all
        L.T.seed_all = lambda k: first(k + 100 * s)
        L.np = shifted_numpy(100 * s)
        flag = {"all8": [], "nospec": ["--no-spectral"], "rgbbranch": ["--rgb-branch"]}[a.channels]
        out = L.run(L.parse([*flag, "--run-name", f"lora_{a.channels}_seed{s}", "--train-val", str(tv),
                             "--qa", str(tv / "hazy_scenes.csv"), "--out", str(runs / "dinov3"),
                             "--frozen-results", str(REPO_RESULTS / "dinov3")]))
        cv = pd.read_csv(out / "cv_results.csv")
    else:
        import rgb_only as R
        from chipio import DEFAULT_CHANNELS
        if a.channels == "rgb":
            R.use_channels(R.RGB)
        if a.model == "rf":
            import common as CD
            CD.SEED = s                 # common.train_fold reads it for the pixel samples and for the forest
            out = runs / f"rf_{a.channels}_seed{s}"
            out.mkdir(parents=True, exist_ok=True)
            cv = R.run_rf(tv, out, R.RGB if a.channels == "rgb" else list(DEFAULT_CHANNELS))
        else:
            import baseline_svm as SVM
            cv = SVM.run_cv(tv / "catalog.csv", tv, runs / f"svm_rbf_{a.channels}_seed{s}", qa=tv / "hazy_scenes.csv",
                            variants=("b",), kernel="rbf_approx", seed=s)
    print(f"\n{a.model} {a.channels} seed {s}: mean IoU {cv.iou.mean():.4f}\n{cv[['fold', 'chips', 'iou']].to_string(index=False)}")


def seed_table(a) -> None:
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, folds = [], []
    for name, folder, where in SEED_ROWS:
        com = committed(*where) if isinstance(where, tuple) else committed(where)
        per, again = {0: com}, None                                  # seed -> IoU of folds 1 to 5
        for f in sorted(runs.glob(f"{folder}_seed*/cv_results.csv")):
            s = int(f.parent.name.rsplit("_seed", 1)[1])
            cv = pd.read_csv(f)
            cv = cv[cv.variant == V].set_index("fold")
            if len(cv) != 5:
                print("not finished, left out:", f.parent)
                continue
            if s == 0:                                               # the committed setting run again: a check, not a new seed
                again = cv.iou.astype(float)
            else:
                per[s] = cv.iou.astype(float)
        t = pd.DataFrame(per)                                        # rows folds, columns seeds
        for s in t.columns:
            folds.append({"model": name, "run": f"seed {s}" + (" (committed)" if s == 0 else ""),
                          **{f"fold{k}": t.loc[k, s] for k in t.index}, "mean": t[s].mean()})
        row = {"model": name, **{f"mean_iou_seed{s}": t[s].mean() for s in t.columns}, "seeds": t.shape[1],
               "mean_over_seeds": t.mean().mean(), "sd_over_seeds": t.mean().std() if t.shape[1] > 1 else np.nan,
               "sd_over_folds": t.std().mean(),                      # sd of the 5 folds, averaged over the seeds
               "sd_over_seeds_within_a_fold": t.std(axis=1).mean() if t.shape[1] > 1 else np.nan}
        if again is not None:
            folds.append({"model": name, "run": "seed 0 run again", **{f"fold{k}": again[k] for k in com.index},
                          "mean": again.mean()})
            row.update({"seed0_again_mean": again.mean(), "seed0_again_minus_committed": again.mean() - com.mean(),
                        "seed0_again_largest_fold_change": (again - com).abs().max()})
        rows.append(row)
    E = pd.DataFrame(rows)
    per_seed = sorted(c for c in E.columns if c.startswith("mean_iou_seed"))
    E = E[["model"] + per_seed + [c for c in E.columns if c != "model" and c not in per_seed]].round(4)
    F = pd.DataFrame(folds).round(4)
    E.to_csv(out / "table_E_seeds.csv", index=False)
    F.to_csv(out / "table_E_per_fold.csv", index=False)
    with pd.option_context("display.width", 250):
        print(f"TABLE E\n{E.to_markdown(index=False)}\n\nper fold\n{F.to_markdown(index=False)}")
    print("wrote table E in", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tv_help = "the Train_Val_Dataset folder (catalog.csv, hazy_scenes.csv, chips/)"
    s = sub.add_parser("scores")
    s.add_argument("--train-val", required=True, help=tv_help)
    s.add_argument("--per-chip", required=True, help="the per-chip file to add to (it is not changed)")
    s.add_argument("--linear-svm-masks", required=True, help="masks of the linear SVM run, preds/SVM_b_no_haze")
    s.add_argument("--out", required=True, help="folder for masks and the new per-chip file; keep it outside the repository")
    t = sub.add_parser("tables")
    t.add_argument("--per-chip", required=True); t.add_argument("--out", required=True)
    r = sub.add_parser("seed")
    r.add_argument("--model", required=True, choices=["rf", "svm", "unet", "lora"])
    r.add_argument("--channels", choices=["all8", "rgb", "nospec", "rgbbranch"], default="all8")
    r.add_argument("--seed", type=int, required=True, help="0 = the committed setting")
    r.add_argument("--train-val", required=True, help=tv_help)
    r.add_argument("--out", required=True, help="runs folder; keep it outside the repository")
    e = sub.add_parser("seed-table")
    e.add_argument("--runs", required=True); e.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "seed":
        allowed = {"rf": ["all8", "rgb"], "svm": ["all8", "rgb"], "unet": ["all8"], "lora": ["all8", "nospec", "rgbbranch"]}
        if a.channels not in allowed[a.model]:
            ap.error(f"--model {a.model} takes --channels {' or '.join(allowed[a.model])}")
    {"scores": scores, "tables": tables, "seed": seed_run, "seed-table": seed_table}[a.cmd](a)


if __name__ == "__main__":
    main()
