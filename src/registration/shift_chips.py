"""Copy a Train/Val dataset with one site's chips for some dates shifted by a fitted translation.

The translation is the `translation` row of a `gee_pair_baseline.csv` written by real_raster_check.py
(tx_m, ty_m in EPSG:32640, moving -> reference). Every other chip is linked, not copied, so the new folder
can be passed as --train-val to the change detection scripts.

    python src/registration/shift_chips.py --train-val <Train_Val_Dataset> --out <new folder> \
        --site al_aryam_islands \
        --shift 2025-03-31=2018-03-08=<march>/gee_pair_baseline.csv \
        --shift 2025-11-01=2018-11-23=<november>/gee_pair_baseline.csv
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.registration import phase_cross_correlation

PIXEL_M = 10.0


def pixel_shift(tx_m, ty_m):
    """(row, col) shift that moves the moving image onto the reference. Rows run south, so north is -row."""
    return (-ty_m / PIXEL_M, tx_m / PIXEL_M)


def shift_chip(z, s):
    out = dict(z)
    out["image"] = np.stack([ndimage.shift(b, s, order=1, mode="nearest") for b in z["image"]]).astype(z["image"].dtype)
    for k in ("cloud", "nodata"):
        if k in z:
            out[k] = ndimage.shift(z[k].astype(np.uint8), s, order=0, mode="nearest").astype(bool)
    return out


def offset_m(ref, mov):
    """Leftover shift between two chips, from phase correlation on the NIR band."""
    d = phase_cross_correlation(np.nan_to_num(ref), np.nan_to_num(mov), upsample_factor=20)[0]
    return float(np.hypot(*d) * PIXEL_M)


def link_dataset(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name != "chips" and not (dst / p.name).exists():
            os.symlink(p.resolve(), dst / p.name)
    (dst / "chips").mkdir(exist_ok=True)
    for p in (src / "chips").iterdir():
        q = dst / "chips" / p.name
        if not q.exists() and not q.is_symlink():
            os.symlink(p.resolve(), q)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-val", required=True)
    ap.add_argument("--out", required=True, help="new dataset folder, outside the repository")
    ap.add_argument("--site", required=True)
    ap.add_argument("--shift", action="append", required=True, metavar="MOVING_DATE=REFERENCE_DATE=CSV")
    ap.add_argument("--report", default=None, help="where to write the JSON report (default: <out>/shift_report.json)")
    a = ap.parse_args(argv)
    src, dst = Path(a.train_val), Path(a.out)
    link_dataset(src, dst)
    report = {}
    for spec in a.shift:
        d_mov, d_ref, csv = spec.split("=", 2)
        r = pd.read_csv(csv).set_index("model").loc["translation"]
        s = pixel_shift(float(r.tx_m), float(r.ty_m))
        before, after = [], []
        for p in sorted((src / "chips").glob(f"{a.site}_{d_mov}_*.npz")):
            with np.load(p, allow_pickle=False) as f:
                z = {k: f[k] for k in f.files}
            new = shift_chip(z, s)
            q = dst / "chips" / p.name
            q.unlink()
            np.savez_compressed(q, **new)
            ref = src / "chips" / p.name.replace(d_mov, d_ref)
            if ref.exists():
                with np.load(ref) as f:
                    nir = f["image"][3]
                before.append(offset_m(nir, z["image"][3]))
                after.append(offset_m(nir, new["image"][3]))
        report[d_mov] = dict(reference=d_ref, tx_m=float(r.tx_m), ty_m=float(r.ty_m), shift_px_row_col=s,
                             chips_checked=len(before),
                             median_offset_before_m=float(np.median(before)) if before else None,
                             median_offset_after_m=float(np.median(after)) if after else None)
        print(d_mov, report[d_mov])
    with open(a.report or dst / "shift_report.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
