"""Acquisition date and time of every non-hazy training scene.

The catalog gives the date and the Sentinel-2 item ids of each scene. The exact sensing time is not in the catalog,
so it is looked up by item id in the public archive collect.py uses (Earth Search, sentinel-2-c1-l2a). Only the
metadata record is read, no imagery. The site position (for the tide model) is the centre of the site's chip boxes.

Writes <out>/scenes.csv: site, date, year, fold, chips, time_utc (ISO), item ids, sun elevation, view azimuth,
site centre lon / lat.

Usage: python 01_scene_times.py --train-val <Train_Val_Dataset> --out <results folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
HAZE_P5 = 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--train-val", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tv = Path(a.train_val)
if "Train_Val_Dataset" not in str(tv):
    raise SystemExit("only the Train_Val_Dataset folder may be used")

cat = pd.read_csv(tv / "catalog.csv")
assert set(cat["role"]) == {"cv"}
hz = pd.read_csv(tv / "hazy_scenes.csv")[["site", "date", "water_b04_p5"]].drop_duplicates(["site", "date"])
cat = cat.merge(hz, on=["site", "date"], how="left")
cat["hazy"] = cat["water_b04_p5"] > HAZE_P5
b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
cat[["x0", "y0", "x1", "y1"]] = b.values
to_ll = Transformer.from_crs(32640, 4326, always_xy=True)

client = Client.open(STAC_URL)
rows = []
for (site, date), g in cat[~cat["hazy"]].groupby(["site", "date"]):
    assert g["item_id"].nunique() == 1
    ids = g["item_id"].iloc[0].split("+")
    items = list(client.search(collections=[COLLECTION], ids=ids).items())
    found = {it.id: it for it in items}
    missing = [i for i in ids if i not in found]
    if missing:
        raise SystemExit(f"{site} {date}: item ids not found in the archive: {missing}")
    times = pd.to_datetime([found[i].datetime for i in ids], utc=True)
    assert all(t.date().isoformat() == date for t in times), (site, date, times)
    spread = (times.max() - times.min()).total_seconds()
    t = times[0] + (times - times[0]).mean()
    s = cat[cat["site"] == site]
    lon, lat = to_ll.transform((s.x0.min() + s.x1.max()) / 2, (s.y0.min() + s.y1.max()) / 2)
    p = [found[i].properties for i in ids]
    rows.append({"site": site, "date": date, "year": int(g["year"].iloc[0]), "fold": int(g["fold"].iloc[0]), "chips": len(g),
                 "time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "tile_time_spread_s": round(spread, 1),
                 "time_in_item_id": ids[0].split("_")[2], "n_tiles": len(ids), "item_ids": "+".join(ids),
                 "satellite": ids[0][:3],
                 "sun_elevation": round(float(np.mean([q.get("view:sun_elevation", np.nan) for q in p])), 2),
                 "view_azimuth": round(float(np.mean([q.get("view:azimuth", np.nan) for q in p])), 1),
                 "view_incidence": round(float(np.mean([q.get("view:incidence_angle", np.nan) for q in p])), 2),
                 "site_lon": round(lon, 5), "site_lat": round(lat, 5)})
    print(rows[-1]["site"], rows[-1]["date"], rows[-1]["time_utc"], "spread", spread, "s", flush=True)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "scenes.csv", index=False)
print(f"wrote {out / 'scenes.csv'}: {len(rows)} scenes")
