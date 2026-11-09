#!/usr/bin/env python3
"""
collect.py - download Sentinel-2 L2A surface reflectance for mangrove sites.

Source: Earth Search v1 STAC, collection "sentinel-2-c1-l2a" (ESA Collection-1,
public Cloud-Optimised GeoTIFFs on AWS; no login needed).

For every site x scene it writes one GeoTIFF:
    <out>/raw/<site>/<YYYY-MM-DD>.tif
        bands 1-6 : B02 B03 B04 B08 B11 B12 as float32 surface reflectance (0-1 scale, NaN = no data)
        band 7    : SCL scene classification (float32 copy of the integer classes, NaN = no data)
    grid: EPSG:32640 (UTM 40N), 10 m pixels, clipped to the site's bounding box.
and appends one row to <out>/raw_manifest.csv.

REFLECTANCE / OFFSET RULE (the bug that broke the Seeing Orbit chips):
    Collection-1 COGs store raw digital numbers (DN). Since processing baseline 04.00 ESA adds
    BOA_ADD_OFFSET = -1000, so reflectance = DN * 0.0001 - 0.1. Collection-1 reprocessed the
    whole archive (2018 and 2020 scenes too are baseline 05.00), so EVERY scene carries the offset.
    We never decide this by date: each asset's STAC "raster:bands" entry gives scale and offset,
    and we apply them exactly once:  reflectance = DN * scale + offset.
    Only if an asset has no raster:bands metadata do we fall back to the item's
    s2:processing_baseline (>= 04.00 -> offset -0.1, else 0). The manifest records which rule was used.

Usage:
    # search by date range (best N scenes per site, least cloudy first)
    python collect.py --sites sites.geojson --start 2020-01-01 --end 2020-12-31 \
        --max-cloud 10 --per-site 4 --out data/

    # or download an exact list of scenes (CSV with columns: site,item_id)
    python collect.py --sites sites.geojson --scenes scenes.csv --out data/

sites.geojson: a FeatureCollection in lon/lat (EPSG:4326); each feature needs a "name" property.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from shapely.geometry import box, shape

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
DST_CRS = "EPSG:32640"
RES = 10.0
# Earth Search asset keys -> band names we write
BANDS = [("blue", "B02"), ("green", "B03"), ("red", "B04"), ("nir", "B08"),
         ("swir16", "B11"), ("swir22", "B12")]
SCL_KEY = "scl"
SCL_MASK_CLASSES = (3, 8, 9, 10)  # cloud shadow, cloud medium prob, cloud high prob, thin cirrus
MANIFEST_FIELDS = ["site", "date", "item_id", "tile", "cloud_cover", "processing_baseline",
                   "offset_applied", "path", "bounds_epsg32640", "valid_fraction",
                   "masked_fraction"]

# GDAL settings for fast, anonymous reads of public COGs over HTTPS
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    AWS_NO_SIGN_REQUEST="YES",
    VSI_CACHE="TRUE",
)
# Behind a corporate/sandbox proxy with its own CA, let GDAL/curl use it too.
for _ca in (os.environ.get("CURL_CA_BUNDLE"), os.environ.get("SSL_CERT_FILE"),
            "/root/.ccr/ca-bundle.crt"):
    if _ca and os.path.exists(_ca):
        GDAL_ENV["GDAL_CURL_CA_BUNDLE"] = _ca
        GDAL_ENV["CURL_CA_BUNDLE"] = _ca
        break


# ---------------------------------------------------------------- helpers
def load_sites(path):
    with open(path) as f:
        fc = json.load(f)
    sites = {}
    for feat in fc["features"]:
        name = feat["properties"].get("name")
        if not name:
            sys.exit(f"{path}: every feature needs a 'name' property")
        sites[name] = shape(feat["geometry"])
    return sites


def site_grid(geom_ll):
    """Bounding box of a lon/lat geometry in EPSG:32640, snapped outward to the 10 m grid."""
    tr = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    minx, miny, maxx, maxy = geom_ll.bounds
    xs, ys = tr.transform([minx, minx, maxx, maxx], [miny, maxy, miny, maxy])
    x0 = np.floor(min(xs) / RES) * RES
    y0 = np.floor(min(ys) / RES) * RES
    x1 = np.ceil(max(xs) / RES) * RES
    y1 = np.ceil(max(ys) / RES) * RES
    width, height = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    return (x0, y0, x1, y1), from_origin(x0, y1, RES, RES), width, height


def scale_offset(item, asset_key):
    """Return (scale, offset, rule) for one asset, from its own metadata."""
    rb = item.assets[asset_key].extra_fields.get("raster:bands") or []
    if rb and "scale" in rb[0]:
        return float(rb[0]["scale"]), float(rb[0].get("offset", 0.0)), "stac_raster_bands"
    pb = item.properties.get("s2:processing_baseline")
    if pb is not None and float(pb) >= 4.0:
        return 1e-4, -0.1, f"fallback_baseline_{pb}"
    return 1e-4, 0.0, f"fallback_baseline_{pb}"


def read_asset(href, transform, width, height, resampling):
    """Warp one COG into the site's 10 m EPSG:32640 grid (windowed HTTP reads only)."""
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=DST_CRS, transform=transform, width=width, height=height,
                           resampling=resampling, src_nodata=0, nodata=0) as vrt:
                return vrt.read(1)


def search_items(client, geom_ll, start, end, max_cloud, limit=200):
    s = client.search(collections=[COLLECTION], intersects=geom_ll.__geo_interface__,
                      datetime=f"{start}/{end}", query={"eo:cloud_cover": {"lt": max_cloud}},
                      max_items=limit)
    return list(s.items())


def pick_items(items, geom_ll, per_site):
    """One item per acquisition date (prefer UTM-40 tiles, then fewest no-data pixels), least cloudy dates first."""
    by_date = {}
    for it in items:
        d = it.datetime.date().isoformat()
        key = (0 if str(it.properties.get("grid:code", "")).startswith("MGRS-40") else 1,
               it.properties.get("s2:nodata_pixel_percentage", 100))
        if d not in by_date or key < by_date[d][0]:
            by_date[d] = (key, it)
    chosen = sorted((v[1] for v in by_date.values()),
                    key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return chosen[:per_site] if per_site else chosen


# ---------------------------------------------------------------- main work
def process(items, site, geom_ll, out_dir, overwrite=False, min_valid=0.9):
    """items = one STAC item, or a list of same-date items (e.g. two MGRS tiles) to mosaic.
    Each pixel is taken from the first item in the list that has valid data there; every item's own
    scale/offset is applied to its own pixels, exactly once."""
    if not isinstance(items, (list, tuple)):
        items = [items]
    item = items[0]
    date = item.datetime.date().isoformat()
    path = os.path.join(out_dir, "raw", site, f"{date}.tif")
    if os.path.exists(path) and not overwrite:
        print(f"  skip (exists) {path}")
        return None
    bounds, transform, width, height = site_grid(geom_ll)

    out = np.full((7, height, width), np.nan, dtype=np.float32)
    filled = np.zeros((height, width), dtype=bool)
    scl_all = np.zeros((height, width), dtype=np.float32)
    rules = set()
    jobs = [(k, Resampling.bilinear) for k, _ in BANDS] + [(SCL_KEY, Resampling.nearest)]
    for it in items:
        with ThreadPoolExecutor(max_workers=7) as ex:
            arrays = list(ex.map(lambda j: read_asset(it.assets[j[0]].href, transform, width,
                                                      height, j[1]), jobs))
        valid = np.all(np.stack(arrays[:6]) > 0, axis=0) & (arrays[6] > 0)
        take = valid & ~filled
        for i, (key, _) in enumerate(BANDS):
            scale, offset, rule = scale_offset(it, key)
            rules.add(f"{rule}:scale={scale:g},offset={offset:g}")
            out[i][take] = arrays[i][take].astype(np.float32) * scale + offset   # applied ONCE
        out[6][take] = arrays[6][take].astype(np.float32)
        scl_all[take] = arrays[6][take]
        filled |= take

    valid_frac = float(filled.mean())
    if valid_frac < min_valid:
        print(f"  skip {'+'.join(i.id for i in items)}: only {valid_frac:.0%} of the site has data")
        return None
    masked_frac = float(np.isin(scl_all[filled], SCL_MASK_CLASSES).mean()) if filled.any() else 1.0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = dict(driver="GTiff", dtype="float32", count=7, width=width, height=height,
                   crs=DST_CRS, transform=transform, nodata=np.nan, compress="deflate",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256)
    ids = "+".join(i.id for i in items)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out)
        for i, (_, name) in enumerate(BANDS, start=1):
            dst.set_band_description(i, f"{name} surface reflectance")
        dst.set_band_description(7, "SCL scene classification")
        dst.update_tags(item_id=ids, site=site, date=date,
                        offset_rule=";".join(sorted(rules)),
                        processing_baseline="+".join(str(i.properties.get("s2:processing_baseline")) for i in items),
                        source=f"{STAC_URL} {COLLECTION}")

    return dict(site=site, date=date, item_id=ids,
                tile="+".join(i.properties.get("grid:code", "") for i in items),
                cloud_cover=round(max(i.properties.get("eo:cloud_cover", float("nan")) for i in items), 3),
                processing_baseline="+".join(str(i.properties.get("s2:processing_baseline")) for i in items),
                offset_applied=";".join(sorted(rules)), path=os.path.relpath(path, out_dir),
                bounds_epsg32640=",".join(f"{b:.0f}" for b in bounds),
                valid_fraction=round(valid_frac, 4), masked_fraction=round(masked_frac, 4))


def append_manifest(out_dir, row):
    mpath = os.path.join(out_dir, "raw_manifest.csv")
    new = not os.path.exists(mpath)
    with open(mpath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="GeoJSON FeatureCollection with a 'name' per feature")
    ap.add_argument("--scenes", help="CSV with columns site,item_id (overrides the date search)")
    ap.add_argument("--start", help="search start date YYYY-MM-DD")
    ap.add_argument("--end", help="search end date YYYY-MM-DD")
    ap.add_argument("--max-cloud", type=float, default=10, help="max scene cloud cover %% (default 10)")
    ap.add_argument("--per-site", type=int, default=4, help="max scenes per site (0 = all)")
    ap.add_argument("--only", nargs="*", help="only these site names")
    ap.add_argument("--min-valid", type=float, default=0.9, help="min fraction of site with data")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list chosen scenes, download nothing")
    a = ap.parse_args()

    sites = load_sites(a.sites)
    if a.only:
        sites = {k: v for k, v in sites.items() if k in a.only}
    client = Client.open(STAC_URL)
    t0 = time.time()

    if a.scenes:
        with open(a.scenes) as f:
            wanted = [(r["site"], r["item_id"]) for r in csv.DictReader(f)]
        plan = []
        for site, iid in wanted:
            if site not in sites:
                print(f"! unknown site {site} in {a.scenes}")
                continue
            items = list(client.search(collections=[COLLECTION], ids=[iid]).items())
            if not items:
                print(f"! item not found: {iid}")
                continue
            plan.append((site, items[0]))
        # rows with the same site and acquisition date become one mosaic (e.g. a site on a tile edge)
        grouped = {}
        for site, it in plan:
            grouped.setdefault((site, it.datetime.date().isoformat()), []).append(it)
        plan = [(site, its) for (site, _), its in grouped.items()]
    else:
        if not (a.start and a.end):
            sys.exit("give --scenes, or --start and --end")
        plan = []
        for site, geom in sites.items():
            items = pick_items(search_items(client, geom, a.start, a.end, a.max_cloud), geom, a.per_site)
            print(f"{site}: {len(items)} scene(s) chosen")
            plan += [(site, it) for it in items]

    for site, it in plan:
        its = it if isinstance(it, list) else [it]
        print(f"{site}  {its[0].datetime.date()}  {'+'.join(i.id for i in its)}  "
              f"cloud={max(i.properties.get('eo:cloud_cover', 0) for i in its):.1f}%")
    if a.dry_run:
        return

    n = 0
    for site, it in plan:
        t = time.time()
        row = process(it, site, sites[site], a.out, a.overwrite, a.min_valid)
        if row:
            append_manifest(a.out, row)
            n += 1
            print(f"  wrote {row['path']} valid={row['valid_fraction']:.0%} "
                  f"cloud/shadow={row['masked_fraction']:.1%} ({time.time() - t:.0f}s)")
    print(f"done: {n} file(s) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
