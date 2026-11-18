"""
List the low-cloud Sentinel-2 scenes of a site with how much of the site is water in each one.
The water share is used as a rough stand-in for the tide, to pick image pairs taken at
different tides.

It uses the same archive and scene picking as src/mangrove_segmentation/collect.py, but it
does not download the scenes. It reads green, near-infrared and the SCL band from the reduced
copies stored inside each file, on a 40 m grid over the site, and keeps one CSV row per scene.

Columns:
  tile_cloud_pct          cloud cover of the whole Sentinel-2 tile, from the archive
  site_cloud_shadow_pct   share of the site that SCL calls cloud or cloud shadow
  water_pct               share of the site where green is brighter than near-infrared
  water_ndwi_pct          share where NDWI > 0 after negative reflectance is set to 0
                          (what the 6-band copies and landmarks.py see)
  negative_refl_pct       share of the site with negative green or near-infrared reflectance.
                          When this is large, water_ndwi_pct is too low and should not be trusted.
  water_scl_pct           share of the site that SCL calls water
  view_azimuth            which side the satellite looked from (about 104 or about 285 here)

Sites marked as test in the sites file are refused.

How to run:
  python src/registration/tide_scene_scan.py --sites <sites.geojson> --out <scenes.csv> \
      --years 2018 2025 --only jubail_island eastern_mangroves
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mangrove_segmentation"))
import collect as c  # noqa: E402
from pystac_client import Client  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

COARSE = 40.0   # metres; the reduced copies are read instead of the full 10 m bands


def scan(site, geom, it):
    (x0, y0, x1, y1), _, w, h = c.site_grid(geom)
    cw, ch = int(np.ceil(w * c.RES / COARSE)), int(np.ceil(h * c.RES / COARSE))
    tr = from_origin(x0, y1, COARSE, COARSE)
    g = c.read_asset(it.assets["green"].href, tr, cw, ch, Resampling.average).astype(np.float32)
    n = c.read_asset(it.assets["nir"].href, tr, cw, ch, Resampling.average).astype(np.float32)
    scl = c.read_asset(it.assets[c.SCL_KEY].href, tr, cw, ch, Resampling.nearest)
    valid = (g > 0) & (n > 0) & (scl > 0)
    if not valid.any():
        return None
    sg, og, _ = c.scale_offset(it, "green")
    sn, on, _ = c.scale_offset(it, "nir")
    g, n = (g * sg + og)[valid], (n * sn + on)[valid]      # reflectance, can be slightly negative
    gc, nc = np.clip(g, 0, None), np.clip(n, 0, None)
    scl = scl[valid]
    p = it.properties

    def pct(x):
        return round(100 * float(np.mean(x)), 2)

    return dict(site=site, date=it.datetime.date().isoformat(), time_utc=it.datetime.strftime("%H:%M:%S"),
                item_id=it.id, tile_cloud_pct=round(p.get("eo:cloud_cover", float("nan")), 2),
                site_cloud_shadow_pct=pct(np.isin(scl, c.SCL_MASK_CLASSES)), site_valid=round(float(valid.mean()), 4),
                water_pct=pct(g > n), water_ndwi_pct=pct((gc - nc) / (gc + nc + 1e-6) > 0),
                negative_refl_pct=pct((g <= 0) | (n <= 0)), water_scl_pct=pct(scl == 6),
                sun_elevation=round(p.get("view:sun_elevation", float("nan")), 1),
                view_azimuth=round(p.get("view:azimuth", float("nan")), 1),
                view_incidence=round(p.get("view:incidence_angle", float("nan")), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", required=True, help="the sites.geojson that collect.py uses")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--only", nargs="+", required=True, help="site names to scan")
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2025])
    ap.add_argument("--max-cloud", type=float, default=10, help="max tile cloud cover %% in the search (default 10)")
    a = ap.parse_args()

    with open(a.sites) as f:
        split = {ft["properties"].get("name"): ft["properties"].get("proposed_split") for ft in json.load(f)["features"]}
    for name in a.only:
        if name not in split:
            sys.exit(f"unknown site {name}")
        if split[name] == "test":
            sys.exit(f"{name} is a locked test site, not scanning it")
    sites = c.load_sites(a.sites)
    client = Client.open(c.STAC_URL)

    jobs = []
    for name in a.only:
        for year in a.years:
            items = c.search_items(client, sites[name], f"{year}-01-01", f"{year}-12-31", a.max_cloud, limit=3000)
            chosen = c.pick_items(items, sites[name], 0)
            print(f"{name} {year}: {len(chosen)} dates under {a.max_cloud:g}% tile cloud", flush=True)
            jobs += [(name, sites[name], it) for it in chosen]
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for r in ex.map(lambda j: scan(*j), jobs) if r]
    rows.sort(key=lambda r: (r["site"], r["date"]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} scenes")


if __name__ == "__main__":
    main()
