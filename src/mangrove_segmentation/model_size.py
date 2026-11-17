"""Model size next to score: total and trainable parameters for every mangrove model, with its IoU.

The neural networks are built from the same code that trains them (train_unet.py, dinov3/train_dinov3.py,
dinov3/train_dinov3_lora.py) and their parameters are counted directly. No data or pretrained weights are
needed: a parameter count depends only on the architecture, so DINOv3 is built with random weights.

What counts as a parameter here:
  - neural networks  every learned weight (BatchNorm running averages and the input mean/std are not counted)
  - NDVI rule        its two thresholds
  - linear SVM       one weight per input channel plus the intercept (the feature scaling folds into these)
  - random forest    per tree, 2 numbers per split (which channel, what cut-off) and 1 per leaf (the mangrove
                     fraction). The number of splits is only known after fitting on the real pixels, so
                     without a fitted model this script can only give an upper limit from the settings.
The decision threshold each model picks on validation data is left out for all of them.

Usage (from src/mangrove_segmentation):
  python model_size.py                         # print the table
  python model_size.py --rf-model rf.joblib    # count the real random forest instead of the upper limit
  python model_size.py --out model_size.csv    # also save the table
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "dinov3"))
RESULTS = HERE.parents[1] / "results" / "mangrove_segmentation"
VARIANT = "b_no_haze"          # hazy scenes dropped from training: the variant every model was run with
DINO_ARCH = "vitl"             # facebook/dinov3-vitl16-pretrain-sat493m, the backbone in dinov3/RESULTS.md


def count(module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    return total, sum(p.numel() for p in module.parameters() if p.requires_grad)


def mean_iou(folder: str, model: str):
    f = RESULTS / folder / "cv_results.csv"
    if not f.exists():
        return None
    cv = pd.read_csv(f)
    cv = cv[(cv["model"] == model) & (cv["variant"] == VARIANT)]
    return round(float(cv["iou"].mean()), 3) if len(cv) else None


def unet_row() -> dict:
    import train_unet as U
    base = U.PRESETS["gpu"]["base"]                    # the gpu preset is the run in results/.../unet
    total, train = count(U.UNet(cin=8, base=base))
    return dict(model="U-Net", total=total, trainable=train, iou=mean_iou("unet", "UNET"),
                note=f"base width {base}, counted from train_unet.UNet")


def dino_rows() -> list[dict]:
    import train_dinov3 as T
    import train_dinov3_lora as L
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    arch = L.BENCH_ARCH[DINO_ARCH]

    def backbone():
        return DINOv3ViTModel(DINOv3ViTConfig(patch_size=16, image_size=512, num_register_tokens=4, **arch))

    head_total, _ = count(T.FusionHead(arch["hidden_size"], spectral=True))
    bb_total, _ = count(backbone())
    lora_bb, info = L.add_lora(backbone())             # default settings: rank 8, query and value, all blocks
    lora_total, lora_train = count(lora_bb)
    src = "architecture built with random weights, real weights not downloaded"
    return [
        dict(model="DINOv3 frozen", total=bb_total + head_total, trainable=head_total, iou=mean_iou("dinov3", "DINOv3"),
             note=f"ViT-L backbone {bb_total:,} frozen + decoder {head_total:,}; {src}"),
        dict(model="DINOv3 + LoRA", total=lora_total + head_total, trainable=lora_train + head_total,
             iou=mean_iou("dinov3_lora", "DINOv3_LoRA"),
             note=f"LoRA r={info['lora_r']} on {'/'.join(info['lora_targets'])} adds {lora_train:,} to the frozen row"),
    ]


def rf_rows(rf_model) -> list[dict]:
    import baseline_rf as B2
    iou = mean_iou("baselines", "B2")
    if rf_model:
        import joblib
        rf = joblib.load(rf_model)
        rf = rf["rf"] if isinstance(rf, dict) else rf
        leaves = sum(int(t.tree_.n_leaves) for t in rf.estimators_)
        n = 3 * leaves - 2 * len(rf.estimators_)
        return [dict(model="Random forest", total=n, trainable=n, iou=iou,
                     note=f"counted from {Path(rf_model).name}: {len(rf.estimators_)} trees, {leaves:,} leaves")]
    sig = inspect.signature(B2.fit).parameters
    trees, pixels = sig["n_estimators"].default, sig["cap"].default
    rows = []
    for s in B2.SETTINGS:
        leaves = pixels // s["min_samples_leaf"]       # every leaf needs at least min_samples_leaf pixels
        if s["max_depth"] is not None:
            leaves = min(leaves, 2 ** s["max_depth"])  # a tree of depth d has at most 2^d leaves
        n = trees * (3 * leaves - 2)
        rows.append(dict(model=f"Random forest (depth {s['max_depth']}, leaf {s['min_samples_leaf']})",
                         total=n, trainable=n, iou=iou,
                         note=f"upper limit, not a count: {trees} trees x at most {leaves:,} leaves, {pixels:,} pixels"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rf-model", help="a fitted random forest (.joblib from baseline_rf.py fit) to count exactly")
    ap.add_argument("--out", help="save the table as CSV")
    a = ap.parse_args()
    rows = [dict(model="NDVI rule", total=2, trainable=2, iou=mean_iou("baselines", "B1"),
                 note="NDVI and MNDWI thresholds (baseline_ndvi.py)")]
    rows += rf_rows(a.rf_model)
    rows.append(dict(model="SVM (linear)", total=9, trainable=9, iou=mean_iou("svm/linear", "SVM"),
                     note="8 channel weights + intercept (baseline_svm.py)"))
    rows.append(unet_row())
    rows += dino_rows()
    table = pd.DataFrame(rows)[["model", "total", "trainable", "iou", "note"]]
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(table.to_string(index=False, formatters={"total": "{:,}".format, "trainable": "{:,}".format}))
    if a.out:
        table.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
