"""Split the chips into the Train/Val dataset and the LOCKED Test dataset, exactly as done in the cloud.

Input   6_Work/catalog.csv + 6_Work/chips/   (from prepare.py)
        3_Data/folds_site_cv.csv             (site -> fold, role: cv / test_fixed / external_fixed)
        3_Data/QA_download.csv               (water brightness per site/date -> 'hazy' flag)
        6_Work/landcover/<chip_id>.npz       (optional, from landcover.py)
Output  6_Work/datasets/Train_Val_Dataset/   chips/, landcover/, catalog.csv, folds_site_cv.csv,
                                             norm_trainval_2020.json, hazy_scenes.csv, README.md
        6_Work/datasets/Test_Dataset/        fixed_test/ and external/ (chips/, landcover/, catalog.csv),
                                             README_LOCKED.md
Chips are hard-linked when the file system allows it (no extra disk space), otherwise copied.
Running it again only adds what is missing (e.g. land-cover files made later); --rebuild starts over.

Usage: python 2_Code/make_datasets.py [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

HAZE_P5 = 0.05   # dark-water red (5th percentile B04 over water) above this = hazy scene (same rule as the baselines)

GROUPS = {"cv": P.TRAIN_VAL, "test_fixed": P.TEST / "fixed_test", "external_fixed": P.TEST / "external"}

README_TV = """# Train/Val dataset (Abu Dhabi, 6 sites, 5 site folds)

- catalog.csv: one row per chip (chip_id, site, date, year, fold, mangrove %), path = chips/<chip_id>.npz
- folds_site_cv.csv: fold 1 Eastern Mangroves, 2 Jubail, 3 Ras Ghurab, 4 Ghantoot, 5 Al Aryam + Al Nouf
- norm_trainval_2020.json: per-channel mean/std from all 2020 Train/Val chips
- hazy_scenes.csv: the 'hazy' flag per site/date used by the baselines and the U-Net (variant b)
- chips/*.npz: image [8,256,256] = B02,B03,B04,B08,B11,B12,NDVI,MNDWI; label_v4019 (10 m, 2020 chips only, 255 = none);
  label_v4112 (~30 m, all years); cloud and nodata masks
- landcover/*.npz: ESA WorldCover class per pixel (key 'lc'), if landcover.py was run
Training label = label_v4019 -> use the 2020 chips. 2018/2025 chips are for change mapping.
Choose models by leave-one-site-out: train on 4 folds, score on the 5th, repeat; report the mean and per-fold scores.
"""

README_LOCKED = """# TEST DATASET: LOCKED

Do not open, plot, train on, tune on or look at scores from these chips until the final evaluation.
- fixed_test/: Abu Al Abyad, Marawah. Scored ONCE per final model.
- external/: Ras Al Khor, Al Zorah, Khor Al Beida, Ras Al Khaimah creek, Khor Kalba. Generalisation test, also scored once.
All model and threshold choices are made on Train_Val_Dataset with the 5 site folds.
"""


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def hazy_table() -> pd.DataFrame:
    qa = pd.read_csv(P.QA_CSV)[["site", "date", "water_b04_p5"]]
    qa["hazy"] = qa["water_b04_p5"] > HAZE_P5
    return qa


def norm_from(cat: pd.DataFrame, chips_dir: Path) -> dict:
    s = ss = None
    n = 0
    for p in cat.path:
        x = np.load(chips_dir.parent / p)["image"].reshape(8, -1).astype(np.float64)
        x = x[:, np.isfinite(x).all(0)]
        s = x.sum(1) if s is None else s + x.sum(1)
        ss = (x ** 2).sum(1) if ss is None else ss + (x ** 2).sum(1)
        n += x.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0))
    return {"channels": ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"],
            "mean": mean.round(6).tolist(), "std": std.round(6).tolist(),
            "source": "Train_Val_Dataset, 2020 chips, all 5 folds", "n_pixels": int(n)}


def build(rebuild: bool = False) -> None:
    if not P.CATALOG.exists():
        sys.exit(f"{P.CATALOG} not found - run prepare first (run_all.py --step prepare)")
    if rebuild and P.DATASETS.exists():
        shutil.rmtree(P.DATASETS)
    cat = pd.read_csv(P.CATALOG)
    folds = pd.read_csv(P.FOLDS)
    role, fold = dict(zip(folds.site, folds.role)), dict(zip(folds.site, folds.fold))
    cat["role"], cat["fold"] = cat.site.map(role), cat.site.map(fold)
    unknown = sorted(cat[cat.role.isna()].site.unique())
    if unknown:
        print("! sites without a role in folds_site_cv.csv (left out):", unknown)
    n_lc = 0
    for r, d in GROUPS.items():
        df = cat[cat.role == r]
        (d / "chips").mkdir(parents=True, exist_ok=True)
        (d / "landcover").mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            link_or_copy(P.WORK / row.path, d / "chips" / Path(row.path).name)
            lc = P.LANDCOVER / f"{row.chip_id}.npz"
            if lc.exists():
                link_or_copy(lc, d / "landcover" / lc.name); n_lc += 1
        out = df.copy()
        out["path"] = "chips/" + out.path.map(lambda p: Path(p).name)
        out.to_csv(d / "catalog.csv", index=False)
        print(f"{d.relative_to(P.WORK)}: {len(df)} chips")
    tv = pd.read_csv(P.TRAIN_VAL / "catalog.csv")
    folds[folds.role == "cv"].drop(columns=[c for c in ("region",) if c in folds.columns]) \
        .to_csv(P.TRAIN_VAL / "folds_site_cv.csv", index=False)
    tv20 = tv[tv.year == 2020]
    if len(tv20):
        json.dump(norm_from(tv20, P.TRAIN_VAL / "chips"), open(P.TRAIN_VAL / "norm_trainval_2020.json", "w"), indent=1)
    hz = hazy_table()
    hz[hz.site.isin(tv.site.unique())].to_csv(P.TRAIN_VAL / "hazy_scenes.csv", index=False)
    (P.TRAIN_VAL / "README.md").write_text(README_TV, encoding="utf-8")
    (P.TEST / "README_LOCKED.md").write_text(README_LOCKED, encoding="utf-8")
    print(f"Train/Val: {len(tv)} chips ({len(tv20)} from 2020 with the 10 m label) -> {P.TRAIN_VAL}")
    print(f"Test (LOCKED): -> {P.TEST}")
    print(f"land-cover files linked: {n_lc}" + ("" if n_lc else " (none yet: run_all.py --step landcover datasets)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="delete 6_Work/datasets and build it again")
    build(ap.parse_args().rebuild)
