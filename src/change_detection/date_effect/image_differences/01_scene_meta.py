"""Processing version of every non-hazy training scene.

The catalog only records the scale and offset applied to the bands (column offset_applied), not the Sentinel-2
processing version (baseline). The version is read from the public archive record of each item id (Earth Search,
sentinel-2-c1-l2a), metadata only, the same way the sensing time was read for the tide check.

Writes <out>/scene_meta.csv: site, date, offset_applied, processing_baseline, processing time, satellite, sun elevation,
view azimuth, plus the team's haze measure (water_b04_p5 from hazy_scenes.csv).

Usage: python 01_scene_meta.py --train-val <Train_Val_Dataset> --out <results folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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
cat = cat[cat["water_b04_p5"] <= HAZE_P5]

client = Client.open(STAC_URL)
rows, keys = [], set()
for (site, date), g in cat.groupby(["site", "date"]):
    ids = g["item_id"].iloc[0].split("+")
    found = {it.id: it for it in client.search(collections=[COLLECTION], ids=ids).items()}
    assert all(i in found for i in ids), (site, date)
    p = [found[i].properties for i in ids]
    keys |= {k for q in p for k in q if "process" in k or "baseline" in k or "boa" in k.lower()}
    rows.append({"site": site, "date": date, "year": int(g["year"].iloc[0]), "chips": len(g),
                 "offset_applied": "|".join(sorted(set(g["offset_applied"].astype(str)))),
                 "processing_baseline": "|".join(sorted({str(q.get("s2:processing_baseline", "")) for q in p})),
                 "processed_on": "|".join(sorted({str(q.get("s2:generation_time", q.get("processing:datetime", "")))[:10] for q in p})),
                 "satellite": ids[0][:3], "haze_measure": float(g["water_b04_p5"].iloc[0]),
                 "sun_elevation": round(float(np.mean([q.get("view:sun_elevation", np.nan) for q in p])), 2),
                 "view_azimuth": round(float(np.mean([q.get("view:azimuth", np.nan) for q in p])), 1)})
    print(rows[-1], flush=True)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "scene_meta.csv", index=False)
print("archive fields about processing:", sorted(keys))
print(pd.DataFrame(rows).groupby(["year", "processing_baseline", "offset_applied"]).size())
