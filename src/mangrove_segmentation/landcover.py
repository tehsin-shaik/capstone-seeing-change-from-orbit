"""ESA WorldCover land cover for every chip (used by the baselines' error-by-land-cover table).

1. Reads small windows of the public WorldCover maps (2020 v100, 2021 v200) around each site from AWS
   -> 6_Work/landcover/raw/wc<year>_<site>.tif
2. Puts the map onto each chip's 10 m grid (nearest neighbour)
   -> 6_Work/landcover/<chip_id>.npz (key 'lc', uint8 WorldCover class; 95 = mangrove, 80 = water ...)
Chips from 2021 or later use the 2021 map, older chips the 2020 map. Files that exist are skipped.
Run make_datasets.py afterwards (run_all.py does this) so the files are linked into the datasets.

Usage: python 2_Code/landcover.py [--sites eastern_mangroves ghantoot]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

URL = {2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{t}_Map.tif",
       2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"}
RES = 1 / 12000      # WorldCover pixel (degrees)
PAD = 0.03           # degrees of padding around each site
RAW = P.LANDCOVER / "raw"


def wc_year(y: int) -> int:
    return 2021 if y >= 2021 else 2020


def fetch_site(feature: dict, year: int) -> Path:
    name = feature["properties"]["name"]
    out = RAW / f"wc{year}_{name}.tif"
    if out.exists():
        return out
    ring = feature["geometry"]["coordinates"][0]
    xs, ys = [c[0] for c in ring], [c[1] for c in ring]
    w, e, s, n = min(xs) - PAD, max(xs) + PAD, min(ys) - PAD, max(ys) + PAD
    w, n = math.floor(w / RES) * RES, math.ceil(n / RES) * RES       # snap to the WorldCover grid
    e, s = math.ceil(e / RES) * RES, math.floor(s / RES) * RES
    W, H = round((e - w) / RES), round((n - s) / RES)
    arr = np.zeros((H, W), np.uint8)
    for lat0 in range(int(math.floor(s / 3) * 3), int(math.floor(n / 3) * 3) + 1, 3):
        for lon0 in range(int(math.floor(w / 3) * 3), int(math.floor(e / 3) * 3) + 1, 3):
            bw, be, bs, bn = max(w, lon0), min(e, lon0 + 3), max(s, lat0), min(n, lat0 + 3)
            if bw >= be or bs >= bn:
                continue
            with rasterio.open(URL[year].format(t=f"N{lat0:02d}E{lon0:03d}")) as src:
                win = window_from_bounds(bw, bs, be, bn, src.transform).round_offsets().round_lengths()
                data = src.read(1, window=win, boundless=True, fill_value=0)
            c0, r0 = round((bw - w) / RES), round((n - bn) / RES)
            h, wd = min(data.shape[0], H - r0), min(data.shape[1], W - c0)
            np.copyto(arr[r0:r0 + h, c0:c0 + wd], data[:h, :wd], where=data[:h, :wd] > 0)
    prof = dict(driver="GTiff", height=H, width=W, count=1, dtype="uint8", crs="EPSG:4326",
                transform=from_origin(w, n, RES, RES), compress="deflate", nodata=0, tiled=True)
    tmp = out.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr, 1)
    os.replace(tmp, out)
    print(f"  WorldCover {year} {name}: {arr.shape[1]} x {arr.shape[0]} px")
    return out


def main(sites: list[str] | None = None) -> None:
    if not P.CATALOG.exists():
        sys.exit(f"{P.CATALOG} not found - run prepare first")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    RAW.mkdir(parents=True, exist_ok=True)
    cat = pd.read_csv(P.CATALOG)
    if sites:
        cat = cat[cat.site.isin(sites)]
    feats = {f["properties"]["name"]: f for f in json.load(open(P.SITES))["features"]}
    need = sorted({(wc_year(int(y)), s) for s, y in zip(cat.site, cat.year)})
    print(f"WorldCover windows: {len(need)} (site, year) pairs -> {RAW}")
    tif = {(y, s): fetch_site(feats[s], y) for y, s in need}
    made = 0
    for r in cat.itertuples():
        out = P.LANDCOVER / f"{r.chip_id}.npz"
        if out.exists():
            continue
        x0, y0, x1, y1 = map(float, r.bounds_epsg32640.split(","))
        lc = np.zeros((256, 256), np.uint8)
        with rasterio.open(tif[(wc_year(int(r.year)), r.site)]) as s:
            reproject(s.read(1), lc, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=from_bounds(x0, y0, x1, y1, 256, 256), dst_crs="EPSG:32640",
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        np.savez_compressed(out, lc=lc)
        made += 1
    print(f"land-cover chips: {made} new, {len(cat) - made} already there -> {P.LANDCOVER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", help="only these sites (default: all chips in 6_Work/catalog.csv)")
    main(ap.parse_args().sites)
