"""Find the tide gauges in or near the Gulf in three open archives.

Downloads the station lists of the IOC Sea Level Station Monitoring Facility and the University of Hawaii Sea Level
Center (UHSLC), and asks the GESLA server for its records in the same area. Keeps every station in a box around the
Gulf and the Gulf of Oman and works out its distance to our six sites. No water levels are downloaded here.

Writes to <work>: lists/ (the files as downloaded), ioc_stations_gulf.csv, uhslc_stations_gulf.csv, gesla_records_gulf.csv

Usage: python 06_gauge_stations.py --scenes <tide out>/scenes.csv --work <gauge work folder>
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", required=True)
ap.add_argument("--work", required=True)
a = ap.parse_args()
work = Path(a.work)
lists = work / "lists"
lists.mkdir(parents=True, exist_ok=True)
LON0, LON1, LAT0, LAT1 = 47.0, 62.0, 21.0, 31.5            # the Gulf, the Strait of Hormuz and the Gulf of Oman
BOX = f"&latitude%3E={LAT0}&latitude%3C={LAT1}&longitude%3E={LON0}&longitude%3C={LON1}&distinct()"
SOURCES = {
    "ioc_stationlist.json": "https://www.ioc-sealevelmonitoring.org/service.php?query=stationlist&showall=all",
    "uhslc_meta.geojson": "https://uhslc.soest.hawaii.edu/data/meta.geojson",
    "gesla_records_gulf.csv": "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_gesla.csv"
                              "?latitude,longitude,record_id,agency_id,station_name,station_code,station_country_code" + BOX,
}
for name, url in SOURCES.items():
    f = lists / name
    if not f.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "tide-gauge-check/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            f.write_bytes(r.read())
        print("downloaded", name)

S = pd.read_csv(a.scenes).groupby("site")[["site_lon", "site_lat"]].first()


def km(lon, lat):
    """Distance from a point to each of our sites."""
    p1, p2 = np.radians(lat), np.radians(S["site_lat"].values)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(S["site_lon"].values - lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def place(lon, lat):
    d = km(lon, lat)
    return dict(lon=lon, lat=lat, km_to_nearest_site=round(float(d.min())), nearest_site=S.index[int(d.argmin())],
                km_to_farthest_site=round(float(d.max())))


def inbox(lon, lat):
    return LON0 <= lon <= LON1 and LAT0 <= lat <= LAT1


rows = []
for s in json.loads((lists / "ioc_stationlist.json").read_text(encoding="utf-8")):
    try:
        lon, lat = float(s["Lon"]), float(s["Lat"])
    except (TypeError, ValueError):
        continue
    if inbox(lon, lat):
        rows.append(dict(code=s.get("Code"), name=s.get("Location"), country=s.get("country"), **place(lon, lat)))
ioc = pd.DataFrame(rows).drop_duplicates("code").sort_values("code")
ioc.to_csv(work / "ioc_stations_gulf.csv", index=False)

rows = []
for f in json.loads((lists / "uhslc_meta.geojson").read_text(encoding="utf-8"))["features"]:
    lon, lat = f["geometry"]["coordinates"][:2]
    p = f["properties"]
    if inbox(lon, lat):
        fd, rq = p.get("fd_span") or {}, p.get("rq_span") or {}
        rows.append(dict(code=p.get("uhslc_id"), name=p.get("name"), country=p.get("country"), **place(lon, lat),
                         fast_delivery_first=fd.get("oldest"), fast_delivery_last=fd.get("latest"),
                         research_first=rq.get("oldest"), research_last=rq.get("latest")))
uh = pd.DataFrame(rows).sort_values("code")
uh.to_csv(work / "uhslc_stations_gulf.csv", index=False)

g = pd.read_csv(lists / "gesla_records_gulf.csv", skiprows=[1])
g.to_csv(work / "gesla_records_gulf.csv", index=False)

pd.set_option("display.width", 200)
print("IOC stations in the box:", len(ioc))
print(ioc.to_string(index=False))
print("\nUHSLC stations in the box:", len(uh))
print(uh.to_string(index=False))
print("\nGESLA records in the box:", len(g))
print(g[["record_id", "station_name", "station_country_code"]].to_string(index=False))
