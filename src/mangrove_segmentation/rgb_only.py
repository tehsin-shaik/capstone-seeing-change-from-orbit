"""Random forest and approximate RBF SVM on RGB only (B04, B03, B02), and the RGB-only summary table.

WHY
  The random forest and the SVM get all 8 chip channels. DINOv3 itself only takes RGB (its decoder also had a
  small conv branch over all 8). To compare like with like, every model is run again with B04, B03, B02 as its
  only input: same 5 leave-one-site-out folds, hazy scenes dropped from training (variant b), same settings
  and seeds as the committed 8-channel runs.

WHAT IS REUSED (nothing about the models is written again here)
  * Random forest: src/change_detection/common.py train_fold(). It follows the team's run_cv.py step by step
    and calls baseline_rf.py unchanged (folds, haze rule, one-chip buffer, inner fold, 100k / 300k / 200k
    pixels, 100 then 200 trees, threshold grid 0.20-0.95, seed 0).
  * SVM: baseline_svm.py run_cv() with kernel rbf_approx, as in the committed run (300 landmarks, gamma 1/8,
    C in 0.01 / 0.1 / 1, seed 0). gamma is left at 1/8 although there are now 3 features, not 8.
  * Scoring: evaluate.py, called the way the team's score.py calls it.

THE ONLY CHANGE
  baseline_rf.pixels() turns a chip into a table of pixel features. Here it is replaced by a version that keeps
  3 of the 8 columns. Which pixels are sampled does not change: a pixel is still usable only if all 8 channels
  are finite, so the RGB models are fitted and scored on the same pixels as the 8-channel ones.

COMMANDS
  python rgb_only.py baselines --train-val <Train_Val_Dataset> --out <folder outside the repo>
  python rgb_only.py baselines --train-val <Train_Val_Dataset> --out <folder> --channels all8
      the same code with all 8 channels: it has to give the committed numbers again
  python rgb_only.py summary --runs <folder> --out results/mangrove_segmentation/rgb_only
The DINOv3 rows of the summary are read from <folder>/dinov3. They come from
  dinov3/train_dinov3_lora.py --no-spectral      (LoRA, decoder branch removed)
  dinov3/train_dinov3_lora.py --rgb-branch       (LoRA, decoder branch kept and fed B04, B03, B02)
  dinov3/train_dinov3.py --mode frozen --no-spectral
each run with --train-val <Train_Val_Dataset> --out <folder>/dinov3. Only Train_Val_Dataset is read.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_RESULTS = HERE.parents[1] / "results" / "mangrove_segmentation"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "change_detection"))

RGB = ["B04", "B03", "B02"]
V = "b_no_haze"
DINO = "dinov3-vitl16-pretrain-sat493m"
ALL8, MIXED = "all 8", "RGB into DINOv3, all 8 into the decoder branch"
# summary rows: RGB-only label, run folder inside --runs, then the committed run: label, table, model tag, channels
ROWS = [
    ("Random forest", "rf_rgb", "Random forest", "baselines/cv_results.csv", "B2", ALL8),
    ("SVM, approximate RBF", "svm_rbf_rgb", "SVM, approximate RBF", "svm/rbf_approx/cv_results.csv", "SVM", ALL8),
    ("DINOv3 + LoRA, decoder branch removed", f"dinov3/lora_{DINO}_nospec",
     "DINOv3 + LoRA, decoder branch on all 8", "dinov3_lora/cv_results.csv", "DINOv3_LoRA", MIXED),
    ("DINOv3 + LoRA, decoder branch fed RGB", f"dinov3/lora_{DINO}_rgbbranch",
     "DINOv3 + LoRA, decoder branch on all 8", "dinov3_lora/cv_results.csv", "DINOv3_LoRA", MIXED),
    ("DINOv3 frozen, decoder branch removed", f"dinov3/frozen_{DINO}_nospec",
     "DINOv3 frozen, decoder branch on all 8", "dinov3/cv_results.csv", "DINOv3", MIXED),
]


def use_channels(names) -> None:
    """Make the random forest and the SVM see only these chip channels."""
    import baseline_rf as B2
    import baseline_svm as SVM
    from chipio import channel_index
    full = B2.pixels

    def pixels(chip):
        return full(chip)[:, [channel_index(chip, n) for n in names]]

    B2.pixels = pixels          # used by baseline_rf.collect (sampling) and by the forest's prediction below
    SVM.pixels = pixels         # baseline_svm.predict_chips has its own reference to the function


def run_rf(tv: Path, out: Path, channels) -> pd.DataFrame:
    import baseline_rf as B2
    import common as CD
    import evaluate as EV
    from chipio import chip_path, load_chip, save_mask
    cat = CD.load_full_catalog(tv)
    cat = cat[cat["year"] == 2020].reset_index(drop=True)
    pred = out / "preds" / f"RF_{V}"
    scp = out / "score_catalog.csv"
    cat.assign(split="f" + cat.fold.astype(str)).to_csv(scp, index=False)
    rows, fits, chips = [], [], []
    for k in sorted(int(f) for f in cat.fold.unique()):
        t0 = time.time()
        fit = CD.train_fold(cat, k, tv, out / "work", "b")
        rf, th = fit["rf"], fit["threshold"]
        pos = list(rf.classes_).index(True)
        for _, r in fit["held"].iterrows():                       # predicted as in run_cv.py
            chip = load_chip(chip_path(tv, r))
            v = chip["valid"]; mask = np.zeros(v.shape, bool)
            if v.any():
                mask[v] = rf.predict_proba(B2.pixels(chip)[v.ravel()])[:, pos] > th
            save_mask(pred, r.chip_id, mask)
        res, pc = EV.evaluate(pred, scp, tv, split=f"f{k}", label="v4019", reps=1000, seed=0)
        p = res["pooled"]
        rows.append({"model": "B2", "variant": V, "fold": k, "held_site": ",".join(fit["info"]["held_sites"]),
                     "chips": p["n_chips"], "mangrove_px": p["mangrove_px"], **{m: round(p[m], 4) for m in EV.METRICS},
                     "iou_ci95_lo": round(p["iou_ci95"][0], 4), "iou_ci95_hi": round(p["iou_ci95"][1], 4),
                     **{f"iou_{s}": round(d["iou"], 4) for s, d in res["per_site"].items()}})
        fits.append({**fit["info"], "settings": str(fit["settings"]), "threshold": th,
                     "feature_importances": str(dict(zip(channels, rf.feature_importances_.round(4).tolist()))),
                     "minutes": round((time.time() - t0) / 60, 2)})
        chips.append(pc.assign(model="B2", variant=V, fold=k))
        print(f"RF {V} fold {k} ({rows[-1]['held_site']}): IoU {rows[-1]['iou']:.4f}  {fit['settings']} threshold={th}", flush=True)
        pd.DataFrame(rows).to_csv(out / "cv_results.csv", index=False)
        pd.DataFrame(fits).to_csv(out / "fold_fits.csv", index=False)
        pd.concat(chips, ignore_index=True).to_csv(out / "per_chip.csv", index=False)
    return pd.DataFrame(rows)


def baselines(a) -> None:
    from chipio import DEFAULT_CHANNELS
    tv, tag = Path(a.train_val), a.channels
    channels = RGB if tag == "rgb" else list(DEFAULT_CHANNELS)
    if tag == "rgb":
        use_channels(RGB)
    print(f"channels: {channels}\nTrain/Val: {tv}", flush=True)
    done = {}
    if "rf" in a.models:
        out = Path(a.out) / f"rf_{tag}"; out.mkdir(parents=True, exist_ok=True)
        done["rf"] = (run_rf(tv, out, channels), "baselines/cv_results.csv", "B2")
    if "svm" in a.models:
        import baseline_svm as SVM
        out = Path(a.out) / f"svm_rbf_{tag}"
        cv = SVM.run_cv(tv / "catalog.csv", tv, out, qa=tv / "hazy_scenes.csv", variants=("b",), kernel="rbf_approx")
        done["svm"] = (cv, "svm/rbf_approx/cv_results.csv", "SVM")
    for name, (cv, table, model) in done.items():                 # next to the committed 8-channel run
        com = committed(table, model)
        cv = cv.set_index("fold")
        assert (cv.chips == com.chips).all() and (cv.mangrove_px == com.mangrove_px).all(), "not the same chips"
        both = pd.DataFrame({"this run": cv.iou, "committed (all 8)": com.iou, "difference": (cv.iou - com.iou).round(4)})
        print(f"\n{name}, channels {tag}: mean IoU {cv.iou.mean():.4f}, committed {com.iou.mean():.4f}\n{both.to_string()}")
        if tag == "all8":
            print("matches the committed run on every fold:", bool((both["difference"] == 0).all()))


def committed(table: str, model: str) -> pd.DataFrame:
    c = pd.read_csv(REPO_RESULTS / table)
    return c[(c.model == model) & (c.variant == V)].set_index("fold")


def summary(a) -> None:
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, table = [], []

    def line(label, channels, iou):
        return {"model": label, "channels": channels, **{f"iou_fold{k}": iou[k] for k in sorted(iou.index)},
                "mean": round(iou.mean(), 4), "sd": round(iou.std(), 4)}          # sample sd, as in the other tables

    for label, folder, com_label, com_table, tag, com_channels in ROWS:
        f = runs / folder / "cv_results.csv"
        if not f.exists():
            print("no run yet:", f)
            continue
        new, com = pd.read_csv(f).set_index("fold"), committed(com_table, tag)
        if len(new) != len(com) or not ((new.chips == com.chips).all() and (new.mangrove_px == com.mangrove_px).all()):
            raise SystemExit(f"{f}: not the same folds and chips as {com_table}")
        rows += [line(label, "RGB (B04, B03, B02)", new.iou), line(com_label, com_channels, com.iou)]
        table.append({"Model": label, "RGB only": f"{new.iou.mean():.3f} ± {new.iou.std():.3f}",
                      "Committed run": f"{com.iou.mean():.3f} ± {com.iou.std():.3f}",
                      "Difference": f"{new.iou.mean() - com.iou.mean():+.3f}",
                      "RGB minus committed, folds 1 to 5": ", ".join(f"{d:+.3f}" for d in (new.iou - com.iou))})
    pd.DataFrame(rows).drop_duplicates().to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(rows).drop_duplicates().to_string(index=False), "\n")
    print(pd.DataFrame(table).to_markdown(index=False))
    print("\nwrote", out / "summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("--train-val", required=True, help="the Train_Val_Dataset folder (catalog.csv, hazy_scenes.csv, chips/)")
    b.add_argument("--out", required=True, help="folder for masks and tables; keep it outside the repository")
    b.add_argument("--channels", choices=["rgb", "all8"], default="rgb", help="all8 = check against the committed runs")
    b.add_argument("--models", nargs="+", choices=["rf", "svm"], default=["rf", "svm"])
    s = sub.add_parser("summary")
    s.add_argument("--runs", required=True, help="the --out folder of the baselines command (DINOv3 runs in <runs>/dinov3)")
    s.add_argument("--out", required=True, help="where summary.csv goes")
    a = ap.parse_args()
    baselines(a) if a.cmd == "baselines" else summary(a)


if __name__ == "__main__":
    main()
