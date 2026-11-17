"""Where do three different models all disagree with the GMW v4.0.19 label?

Takes the held-out 2020 masks of three models (random forest, approximate RBF SVM, DINOv3 + LoRA) and looks
at the pixels where all three give the same answer and GMW gives the other one. Only pixels that evaluate.py
scores are used: finite, not cloud, not no data, and labelled.

Per site and per fold it reports
  * the share of GMW mangrove pixels that all three models call not mangrove
  * the share of GMW non-mangrove pixels that all three models call mangrove
  * how much of each sits within 1 pixel (8 neighbours) of a GMW mangrove edge
  * what GMW v4.1.12 (about 30 m) says about the same pixels
  * each model's IoU with and without those pixels
It also gives the same shares per scene (per_site_date.csv), lists the chips with the most such pixels
(the ones worth hand labelling first) and draws one of them.

The masks are <chip_id>.npz files with key `mask`, as written by baseline_svm.py cv and
dinov3/train_dinov3_lora.py. If --rf-preds has no masks yet, the random forest is trained for every fold with
the code in src/change_detection (the baseline cross-validation steps, hazy scenes dropped from training) and
its held-out masks are written there. Keep the mask folders and --out outside the repository.

Usage (from src/mangrove_segmentation):
  python model_disagreement.py --train-val <Train_Val_Dataset> --rf-preds <folder> --svm-preds <folder> \
      --lora-preds <folder> --out <output folder>
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chipio import chip_path, load_catalog, load_chip, load_mask, save_mask  # noqa: E402

MODELS = ["RF", "SVM", "LoRA"]
S8 = np.ones((3, 3), bool)            # "within 1 pixel" = the 8 neighbours
PX_HA = 0.01                          # one 10 m pixel = 100 m2
COUNTS = ["T", "N", "rej3", "fp3", "rej3_edge", "fp3_edge", "T_edge", "N_edge",
          "rej3_v4112_lab", "rej3_v4112_not", "fp3_v4112_lab", "fp3_v4112_mang",
          "T_v4112_lab", "T_v4112_not", "N_v4112_lab", "N_v4112_mang"] + \
         [f"{m}_{x}{s}" for m in MODELS for x in ("tp", "fp", "fn") for s in ("", "_ex")]


def make_rf_masks(tv, folder, variant="b") -> None:
    """Train the random forest for every fold and save its held-out 2020 masks (uses src/change_detection)."""
    sys.path.insert(0, str(HERE.parent / "change_detection"))
    import common as CD
    cat = CD.load_full_catalog(tv)
    cat20 = cat[cat["year"] == 2020].reset_index(drop=True)
    for k in sorted(int(f) for f in cat20.fold.unique()):
        fit = CD.train_fold(cat20, k, tv, Path(folder) / "work", variant)
        rf, th = fit["rf"], fit["threshold"]
        pos = list(rf.classes_).index(True)
        for _, r in fit["held"].iterrows():
            chip = load_chip(chip_path(tv, r))
            v = chip["valid"]
            mask = np.zeros(v.shape, bool)
            if v.any():
                mask[v] = rf.predict_proba(CD.B2.pixels(chip)[v.ravel()])[:, pos] > th
            save_mask(folder, r["chip_id"], mask)
        iou = CD.heldout_scores(rf, th, fit["held"], tv)["__fold__"]["iou"]
        print(f"random forest fold {k}: threshold {th}, held-out IoU {iou:.4f} "
              f"(committed B2 result {CD.published_fold_iou(k, variant)})", flush=True)


def chip_layers(chip, masks) -> dict:
    """The boolean layers the counts and the figure are made from."""
    lab = chip["label_v4019"]
    use = chip["valid"] & (lab != 255)
    G = lab == 1
    T, N = use & G, use & ~G
    rej3 = T & ~masks["RF"] & ~masks["SVM"] & ~masks["LoRA"]      # GMW mangrove, all three say no
    fp3 = N & masks["RF"] & masks["SVM"] & masks["LoRA"]          # GMW not mangrove, all three say yes
    inner = G & ~ndi.binary_erosion(G, S8, border_value=1)        # GMW mangrove pixels touching non-mangrove
    outer = ndi.binary_dilation(G, S8) & ~G                       # non-mangrove pixels touching GMW mangrove
    return {"use": use, "G": G, "T": T, "N": N, "rej3": rej3, "fp3": fp3, "inner": inner, "outer": outer}


def chip_counts(chip, masks) -> dict:
    L = chip_layers(chip, masks)
    T, N, rej3, fp3 = L["T"], L["N"], L["rej3"], L["fp3"]
    l12 = chip["label_v4112"]
    has12 = l12 != 255
    layers = {"T": T, "N": N, "rej3": rej3, "fp3": fp3, "rej3_edge": rej3 & L["inner"], "fp3_edge": fp3 & L["outer"],
              "T_edge": T & L["inner"], "N_edge": N & L["outer"],
              "rej3_v4112_lab": rej3 & has12, "rej3_v4112_not": rej3 & (l12 == 0),
              "fp3_v4112_lab": fp3 & has12, "fp3_v4112_mang": fp3 & (l12 == 1),
              "T_v4112_lab": T & has12, "T_v4112_not": T & (l12 == 0),
              "N_v4112_lab": N & has12, "N_v4112_mang": N & (l12 == 1)}
    d = {k: int(v.sum()) for k, v in layers.items()}
    keep = L["use"] & ~rej3 & ~fp3                                # scored pixels once the all-three pixels are left out
    for m in MODELS:
        for s, u in (("", L["use"]), ("_ex", keep)):
            d[f"{m}_tp{s}"] = int((masks[m] & L["G"] & u).sum())
            d[f"{m}_fp{s}"] = int((masks[m] & ~L["G"] & u).sum())
            d[f"{m}_fn{s}"] = int((~masks[m] & L["G"] & u).sum())
    return d


def pct(a, b, nd=1):
    return round(100 * a / b, nd) if b else float("nan")


def iou(d, m, s=""):
    den = d[f"{m}_tp{s}"] + d[f"{m}_fp{s}"] + d[f"{m}_fn{s}"]
    return round(d[f"{m}_tp{s}"] / den, 4) if den else float("nan")


def summarise(d, name, chips) -> dict:
    row = {"group": name, "chips": int(chips), "gmw_mangrove_ha": round(d["T"] * PX_HA, 1),
           "mangrove_rejected_by_all3_pct": pct(d["rej3"], d["T"]),
           "of_those_on_gmw_edge_pct": pct(d["rej3_edge"], d["rej3"]),
           "all_gmw_mangrove_on_edge_pct": pct(d["T_edge"], d["T"]),
           "non_mangrove_called_by_all3_pct": pct(d["fp3"], d["N"], 2),
           "called_by_all3_as_pct_of_gmw_mangrove_area": pct(d["fp3"], d["T"]),
           "of_those_touching_gmw_mangrove_pct": pct(d["fp3_edge"], d["fp3"]),
           "all_non_mangrove_touching_gmw_mangrove_pct": pct(d["N_edge"], d["N"]),
           "rejected_by_all3_that_v4112_calls_not_mangrove_pct": pct(d["rej3_v4112_not"], d["rej3_v4112_lab"]),
           "all_gmw_mangrove_that_v4112_calls_not_mangrove_pct": pct(d["T_v4112_not"], d["T_v4112_lab"]),
           "called_by_all3_that_v4112_calls_mangrove_pct": pct(d["fp3_v4112_mang"], d["fp3_v4112_lab"]),
           "all_non_mangrove_that_v4112_calls_mangrove_pct": pct(d["N_v4112_mang"], d["N_v4112_lab"], 2)}
    for m in MODELS:
        row[f"{m}_iou"] = iou(d, m)
        row[f"{m}_iou_left_out"] = iou(d, m, "_ex")
        row[f"{m}_misses_shared_by_all3_pct"] = pct(d["rej3"], d[f"{m}_fn"])
        row[f"{m}_false_alarms_shared_by_all3_pct"] = pct(d["fp3"], d[f"{m}_fp"])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-val", required=True, help="the Train_Val_Dataset folder")
    ap.add_argument("--rf-preds", required=True, help="random forest masks; made here if the folder has none")
    ap.add_argument("--svm-preds", required=True, help="RBF SVM masks, e.g. <svm run>/preds/SVM_b_no_haze")
    ap.add_argument("--lora-preds", required=True, help="DINOv3 + LoRA masks, e.g. <lora run>/preds/DINO_b_no_haze")
    ap.add_argument("--out", required=True, help="output folder, outside the repository")
    ap.add_argument("--top", type=int, default=15, help="how many chips to list for hand labelling")
    ap.add_argument("--figure-site", default="jubail_island", help="site whose worst chip is drawn")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    tv, out = Path(a.train_val), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folders = {"RF": Path(a.rf_preds), "SVM": Path(a.svm_preds), "LoRA": Path(a.lora_preds)}

    cat = load_catalog(tv / "catalog.csv")
    cat = cat[(cat["year"] == 2020) & cat["fold"].notna()].reset_index(drop=True)
    cat["fold"] = cat["fold"].astype(int)
    if not all((folders["RF"] / f"{c}.npz").exists() for c in cat["chip_id"]):
        make_rf_masks(tv, folders["RF"])

    rows = []
    for _, r in cat.iterrows():
        chip = load_chip(chip_path(tv, r))
        masks = {m: load_mask(folders[m], r["chip_id"]) for m in MODELS}
        rows.append({"fold": r["fold"], "site": r["site"], "chip_id": r["chip_id"], "date": r["date"],
                     "row": r["row"], "col": r["col"], "bounds_epsg32640": r["bounds_epsg32640"], **chip_counts(chip, masks)})
    chips = pd.DataFrame(rows)
    chips["all3_disagree_px"] = chips["rej3"] + chips["fp3"]
    chips.to_csv(out / "per_chip.csv", index=False)

    sites = pd.DataFrame([summarise(g[COUNTS].sum(), s, len(g)) for s, g in chips.groupby("site", sort=False)]
                         + [summarise(chips[COUNTS].sum(), "all sites pooled", len(chips))])
    folds = pd.DataFrame([summarise(g[COUNTS].sum(), f"fold {f}", len(g)) for f, g in chips.groupby("fold")])
    mean = {"group": "mean over folds", "chips": len(chips)}
    mean.update({c: round(folds[c].mean(), 4) for c in folds.columns if "_iou" in c})
    folds = pd.concat([folds, pd.DataFrame([mean])], ignore_index=True)
    sites.to_csv(out / "per_site.csv", index=False)
    folds.to_csv(out / "per_fold.csv", index=False)

    # ---- the same shares per scene. The label is the same on every date, so a label error cannot change with the date.
    g = chips.groupby(["site", "date"])
    s = g[["T", "N", "rej3", "fp3"]].sum()
    scenes = pd.DataFrame({"chips": g.size(), "gmw_mangrove_ha": (s["T"] * PX_HA).round(1),
                           "mangrove_rejected_by_all3_pct": (100 * s["rej3"] / s["T"]).round(1),
                           "non_mangrove_called_by_all3_pct": (100 * s["fp3"] / s["N"]).round(2),
                           "called_by_all3_as_pct_of_gmw_mangrove_area": (100 * s["fp3"] / s["T"]).round(1)}).reset_index()
    scenes.to_csv(out / "per_site_date.csv", index=False)

    # ---- the chips with the most all-three-disagree pixels: hand label these first
    cell = chips.groupby(["site", "row", "col"])["all3_disagree_px"].agg(["min", "count"])
    top = chips.sort_values("all3_disagree_px", ascending=False).head(a.top)
    same_cell = [cell.loc[(s_, r_, c_)] for s_, r_, c_ in zip(top["site"], top["row"], top["col"])]
    top = pd.DataFrame({
        "rank": range(1, len(top) + 1), "site": top["site"].values, "chip_id": top["chip_id"].values, "date": top["date"].values,
        "fold": top["fold"].values, "row": top["row"].values, "col": top["col"].values,
        "bounds_epsg32640": top["bounds_epsg32640"].values,
        "all3_disagree_px": top["all3_disagree_px"].values,
        "gmw_mangrove_rejected_by_all3_px": top["rej3"].values,
        "gmw_non_mangrove_called_by_all3_px": top["fp3"].values,
        "all3_disagree_pct_of_scored_px": [pct(x, t + n) for x, t, n in zip(top["all3_disagree_px"], top["T"], top["N"])],
        "gmw_mangrove_px": top["T"].values,
        # the same ground cell on its other 2020 dates: a label error should show on every date
        "dates_of_this_cell": [int(x["count"]) for x in same_cell],
        "lowest_all3_disagree_px_of_this_cell": [int(x["min"]) for x in same_cell]})
    top.to_csv(out / "top_chips_to_hand_label.csv", index=False)

    tot = chips[COUNTS].sum()
    indep = np.prod([tot[f"{m}_fn"] / tot["T"] for m in MODELS])
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print("\nPER SITE\n" + sites.iloc[:, :10].to_string(index=False))
        print("\nGMW v4.1.12 ON THE SAME PIXELS\n" + sites.iloc[:, [0, 10, 11, 12, 13]].to_string(index=False))
        print("\nIoU, with and without the all-three pixels\n"
              + sites[["group"] + [c for c in sites.columns if "_iou" in c]].to_string(index=False) + "\n"
              + folds[["group"] + [c for c in folds.columns if "_iou" in c]].to_string(index=False))
        print("\nSHARE OF EACH MODEL'S ERRORS THAT ALL THREE MAKE\n"
              + sites[["group"] + [c for c in sites.columns if "shared" in c]].to_string(index=False))
        print(f"\nif the three models' misses were unrelated, all three would miss {100 * indep:.2f}% of GMW mangrove; "
              f"observed {100 * tot['rej3'] / tot['T']:.2f}%")
        print("\nPER SCENE\n" + scenes.to_string(index=False))
        print("\nTOP CHIPS\n" + top.drop(columns=["bounds_epsg32640"]).to_string(index=False))
        print(f"distinct chip cells in the top {len(top)}: {len(set(zip(top['site'], top['row'], top['col'])))}")

    best = chips[chips["site"] == a.figure_site].sort_values("all3_disagree_px", ascending=False)
    if len(best):
        r = cat[cat["chip_id"] == best["chip_id"].iloc[0]].iloc[0]
        figure(load_chip(chip_path(tv, r)), {m: load_mask(folders[m], r["chip_id"]) for m in MODELS}, r,
               out / f"worst_chip_{a.figure_site}.png", a.dpi)


# ------------------------------------------------------------------ figure

AQUA, BLUE, ORANGE, GREY, WHITE = "#1baf7a", "#2a78d6", "#eb6834", "#e9e8e4", "#ffffff"
INK, INK2 = "#0b0b0b", "#52514e"


def figure(chip, masks, r, path, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    L = chip_layers(chip, masks)
    img = np.clip(np.moveaxis(chip["image"][[3, 2, 1]], 0, -1) / 0.40, 0, 1) ** 0.7       # B08, B04, B03
    img[~np.isfinite(img).all(axis=-1)] = 1.0

    def paint(layers):
        rgb = np.empty(L["use"].shape + (3,), np.float32)
        rgb[:] = to_rgb(GREY)
        for mask, colour in layers:
            rgb[mask] = to_rgb(colour)
        rgb[~L["use"]] = to_rgb(WHITE)
        return rgb
    gmw = paint([(L["T"], AQUA)])
    dis = paint([(L["T"] & ~L["rej3"], AQUA), (L["rej3"], ORANGE), (L["fp3"], BLUE)])

    d = date.fromisoformat(r["date"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.9), dpi=dpi, facecolor="white")
    for ax, (im, title) in zip(axes, [(img, "Sentinel-2 false colour"), (gmw, "GMW v4.0.19 label"),
                                      (dis, "All three models against GMW")]):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(title, fontsize=12, color=INK, loc="left", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c3c2b7"); sp.set_linewidth(0.8)
    ha = lambda m: f"{m.sum() * PX_HA:,.1f} ha"                                         # noqa: E731
    axes[1].legend(handles=[Patch(facecolor=AQUA, label=f"GMW mangrove  {ha(L['T'])}"),
                            Patch(facecolor=GREY, edgecolor="#c3c2b7", label="GMW not mangrove")],
                   loc="upper left", bbox_to_anchor=(0, -0.02), ncol=1, frameon=False, fontsize=10, labelcolor=INK, handlelength=1.2)
    axes[2].legend(handles=[Patch(facecolor=ORANGE, label=f"GMW mangrove, all three say not mangrove  {ha(L['rej3'])}"),
                            Patch(facecolor=BLUE, label=f"GMW not mangrove, all three say mangrove  {ha(L['fp3'])}"),
                            Patch(facecolor=AQUA, label="GMW mangrove, at least one model agrees")],
                   loc="upper left", bbox_to_anchor=(0, -0.02), ncol=1, frameon=False, fontsize=10, labelcolor=INK, handlelength=1.2)
    axes[0].text(0, -0.03, "Near-infrared shown as red, so vegetation is red." + chr(10) + "The chip is 2.56 km wide.",
                 transform=axes[0].transAxes, ha="left", va="top", fontsize=10, color=INK2, linespacing=1.5)
    fig.suptitle(f"{r['site'].replace('_', ' ').title()}, chip row {r['row']} col {r['col']}, {d.day} {d:%b %Y}: "
                 "where the random forest, the RBF SVM and DINOv3 + LoRA all disagree with GMW",
                 x=0.012, y=0.985, ha="left", fontsize=13, color=INK)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.90, bottom=0.17, wspace=0.03)
    fig.savefig(path, facecolor="white")
    print("figure ->", path)


if __name__ == "__main__":
    main()
