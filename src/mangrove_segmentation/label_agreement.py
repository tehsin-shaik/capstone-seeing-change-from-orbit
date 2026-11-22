"""Compare hand labels from different people, and score GMW and the models against them.

Each hand label is a GeoPackage named <chip_id>_<labeller>.gpkg with two layers: `mangrove` (confident canopy)
and `unsure` (mixed or edge pixels), drawn on one Train_Val chip. On each chip the scored pixels are the same for
every comparison: pixels that are cloud or no data, or that any labeller put in `unsure`, are left out. Everything
else outside `mangrove` counts as non-mangrove.

Reference labels per chip: each labeller on their own and, where a chip has two or more labellers, the area they
all marked (`agreed`) and the area any of them marked (`union`). With three or more labellers, `majority` is the
area more than half of them marked.

Compared against each reference:
  - GMW v4.0.19 and v4.1.12, read from the chip itself
  - the random forest (B2, variant b), retrained for the chip's held-out fold with the same steps as the
    baseline cross-validation (`src/change_detection/common.py: train_fold`), so it never saw the site
  - any other model, from a folder of out-of-fold predicted masks (--pred NAME=DIR, one <chip_id>.npz each)

Outputs: agreement.csv (labeller vs labeller), per_chip.csv and summary.csv (GMW and models vs each reference).
With --save-majority DIR, the majority label of each chip is also saved as <chip_id>_majority.gpkg, with the same
two layers: `mangrove` (majority, outside unsure and no-data pixels) and `unsure` (anything any labeller put there).

Usage:
    python label_agreement.py --train-val <Train_Val_Dataset> --labels <label .gpkg files> \
        --pred "U-Net base 8=<preds dir>" --out <work folder> [--save-majority <folder>]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize, shapes
from shapely.geometry import shape as to_shape
from shapely.validation import make_valid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "change_detection"))
import common as C  # noqa: E402
from chipio import chip_path, get_label, load_chip, load_mask  # noqa: E402


def read_layer(path, layer, shape, transform):
    try:
        g = gpd.read_file(path, layer=layer).to_crs(32640)
    except Exception:
        return np.zeros(shape, bool)
    geoms = [make_valid(x) for x in g.geometry if x is not None and not x.is_empty]
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize([(x, 1) for x in geoms], shape, transform=transform).astype(bool)


def save_label(path, layers, transform):
    path.unlink(missing_ok=True)
    for name, m in layers.items():
        geoms = [to_shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=transform) if v == 1]
        gpd.GeoDataFrame({"label": [name] * len(geoms)}, geometry=geoms, crs=32640).to_file(path, layer=name, driver="GPKG")


def score(pred, truth, use):
    tp = int((pred & truth & use).sum()); fp = int((pred & ~truth & use).sum()); fn = int((~pred & truth & use).sum())
    return dict(iou=tp / max(tp + fp + fn, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                pred_ha=float((pred & use).sum()) * C.PX_HA, tp=tp, fp=fp, fn=fn)


def pooled(df, keys):
    s = df.groupby(keys)[["tp", "fp", "fn"]].sum()
    s["pooled_iou"] = s.tp / (s.tp + s.fp + s.fn)
    s["precision"] = s.tp / (s.tp + s.fp)
    s["recall"] = s.tp / (s.tp + s.fn)
    s["mean_chip_iou"] = df.groupby(keys)["iou"].mean()
    s["chips"] = df.groupby(keys)["iou"].size()
    return s[["chips", "pooled_iou", "mean_chip_iou", "precision", "recall"]].round(3)


def forest(cat20, k, tv, work):
    p = work / f"rf_fold{k}.joblib"
    if p.exists():
        return joblib.load(p)
    print(f"training the random forest for held-out fold {k}", flush=True)
    f = C.train_fold(cat20, k, tv, work, variant="b")
    joblib.dump((f["rf"], f["threshold"]), p)
    return f["rf"], f["threshold"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-val", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--pred", action="append", default=[], help="NAME=DIR of out-of-fold predicted masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-majority", help="folder for one majority-label GeoPackage per chip")
    a = ap.parse_args()
    tv, out = Path(a.train_val), Path(a.out)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    cat = C.load_full_catalog(tv)
    cat20 = cat[cat.year == 2020].reset_index(drop=True)
    preds = dict(p.split("=", 1) for p in a.pred)

    by_chip = defaultdict(dict)
    for lp in map(Path, a.labels):
        cid, who = lp.stem.rsplit("_", 1)
        if not (cat20.chip_id == cid).any():
            print(f"skipping {lp.name}: not a 2020 Train_Val chip label", flush=True)
            continue
        by_chip[cid][who] = lp

    rows, pairs = [], []
    for cid, files in sorted(by_chip.items()):
        r = cat20[cat20.chip_id == cid].iloc[0]
        chip = load_chip(chip_path(tv, r))
        shape, tr = chip["valid"].shape, Affine(*chip["transform"][:6])
        mang = {w: read_layer(p, "mangrove", shape, tr) for w, p in sorted(files.items())}
        unsure = np.any([read_layer(p, "unsure", shape, tr) for p in files.values()], axis=0)
        use = chip["valid"] & ~unsure
        base = dict(chip_id=cid, site=r.site, date=r.date, fold=int(r.fold), scored_pct=100 * use.mean())

        for (w1, m1), (w2, m2) in itertools.combinations(mang.items(), 2):
            pairs.append({**base, "labeller_1": w1, "labeller_2": w2, "ha_1": float((m1 & use).sum()) * C.PX_HA,
                          "ha_2": float((m2 & use).sum()) * C.PX_HA, **score(m2, m1, use)})

        refs = dict(mang)
        if len(mang) > 1:
            refs["agreed"] = np.all(list(mang.values()), axis=0)
            refs["union"] = np.any(list(mang.values()), axis=0)
        if len(mang) > 2:
            refs["majority"] = np.sum(list(mang.values()), axis=0) * 2 > len(mang)
            if a.save_majority:
                d = Path(a.save_majority)
                d.mkdir(parents=True, exist_ok=True)
                save_label(d / f"{cid}_majority.gpkg",
                           {"mangrove": refs["majority"] & use, "unsure": unsure & chip["valid"]}, tr)

        cands = {}
        for v in ("v4019", "v4112"):
            gmw, labelled = get_label(chip, v)
            cands[f"GMW {v[1]}.{v[2]}.{v[3:]}"] = gmw & labelled
        rf, th = forest(cat20, int(r.fold), tv, work)
        mask = np.zeros(shape, bool)
        v = chip["valid"]
        if v.any():
            mask[v] = rf.predict_proba(C.B2.pixels(chip)[v.ravel()])[:, list(rf.classes_).index(True)] > th
        cands["Random forest"] = mask
        for name, d in preds.items():
            cands[name] = load_mask(d, cid).astype(bool)

        for ref, truth in refs.items():
            for name, p in cands.items():
                rows.append({**base, "reference": ref, "reference_ha": float((truth & use).sum()) * C.PX_HA,
                             "compared": name, **score(p & v, truth, use)})
        print(cid, "labellers:", ", ".join(mang), flush=True)

    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "per_chip.csv", index=False)
    s = pooled(df, ["reference", "compared"])
    s.to_csv(out / "summary.csv")
    print(s.to_string())
    if pairs:
        pa = pd.DataFrame(pairs)
        pa.round(4).to_csv(out / "agreement.csv", index=False)
        print(pa[["chip_id", "labeller_1", "labeller_2", "ha_1", "ha_2", "iou"]].round(3).to_string())


if __name__ == "__main__":
    main()
