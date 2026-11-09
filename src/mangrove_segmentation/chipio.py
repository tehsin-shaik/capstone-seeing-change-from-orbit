"""Shared chip I/O for evaluate.py and the baselines.

Follows the chip format written by prepare.py:
    <out>/chips/<chip_id>.npz with keys
        image        float32 (8, H, W)  B02 B03 B04 B08 B11 B12 NDVI MNDWI (NaN = no data)
        label_v4019  uint8   (H, W)     1 mangrove, 0 not, 255 = no label (GMW v4.0.19, 2020 only)
        label_v4112  uint8   (H, W)     same coding (GMW v4.1.12, 2018/2020/2025)
        cloud        bool    (H, W)     SCL cloud / shadow / cirrus
        nodata       bool    (H, W)
        channels     str (8,)
    <out>/catalog.csv with chip_id, site, split, path (= "chips/<chip_id>.npz", relative to <out>)

Older/synthetic chips with a single `valid` key instead of cloud/nodata also load.
`--chips` may point at either <out> or <out>/chips.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CHANNELS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"]
LABEL_KEYS = {"v4019": "label_v4019", "v4112": "label_v4112"}
NOLABEL = 255


def load_catalog(path: str | os.PathLike) -> pd.DataFrame:
    cat = pd.read_csv(path)
    missing = {"chip_id", "site", "split"} - set(cat.columns)
    if missing:
        raise ValueError(f"catalog is missing columns: {sorted(missing)}")
    cat["chip_id"] = cat["chip_id"].astype(str)
    return cat


def chip_path(chips_dir: str | os.PathLike, row) -> Path:
    d = Path(chips_dir)
    cands = []
    if "path" in row and isinstance(row["path"], str) and row["path"]:
        p = Path(row["path"])
        cands += [p] if p.is_absolute() else [d / p, d.parent / p]
    cands += [d / f"{row['chip_id']}.npz", d / "chips" / f"{row['chip_id']}.npz"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"chip {row['chip_id']} not found; tried {[str(c) for c in cands]}")


def load_chip(path: str | os.PathLike) -> dict:
    """Load a chip and add `valid` = usable pixel (finite in every band, not cloud, not nodata)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    valid = np.all(np.isfinite(d["image"]), axis=0)
    if "valid" in d:
        valid &= d["valid"].astype(bool)
    if "cloud" in d:
        valid &= ~d["cloud"].astype(bool)
    if "nodata" in d:
        valid &= ~d["nodata"].astype(bool)
    d["valid"] = valid
    return d


def channel_index(chip: dict, name: str) -> int:
    names = [str(c) for c in chip["channels"]] if "channels" in chip else DEFAULT_CHANNELS
    return names.index(name)


def get_label(chip: dict, label: str) -> tuple[np.ndarray, np.ndarray]:
    """(truth, labelled): truth = mangrove; labelled = False where the label is 255 (no label)."""
    key = LABEL_KEYS[label]
    if key not in chip:
        raise KeyError(f"chip has no {key}; available keys: {sorted(chip)}")
    lab = chip[key]
    return lab == 1, lab != NOLABEL


def load_mask(pred_dir: str | os.PathLike, chip_id: str) -> np.ndarray:
    """Predicted binary mask for one chip: .npy, .npz (key 'mask' or first array) or .tif (band 1)."""
    base = str(Path(pred_dir) / chip_id)
    for ext in (".npy", ".npz", ".tif", ".tiff"):
        p = Path(base + ext)
        if not p.exists():
            continue
        if ext == ".npy":
            return np.load(p).astype(bool)
        if ext == ".npz":
            with np.load(p) as z:
                key = "mask" if "mask" in z.files else z.files[0]
                return z[key].astype(bool)
        import rasterio
        with rasterio.open(p) as r:
            return r.read(1).astype(bool)
    raise FileNotFoundError(f"no prediction for chip {chip_id} in {pred_dir}")


def save_mask(pred_dir: str | os.PathLike, chip_id: str, mask: np.ndarray) -> Path:
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    p = Path(pred_dir) / f"{chip_id}.npz"
    np.savez_compressed(p, mask=mask.astype(np.uint8))
    return p
