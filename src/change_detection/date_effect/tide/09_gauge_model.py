"""EOT20 tide heights at the gauge positions and at our sites, every 10 minutes around each scene date.

Uses the same model call as 03_tide_model.py, so the heights at the sites are the ones the tide test used
(11 checks this). The gauge positions are typed in below from the IOC station list.

Writes to <work>: model_grid.csv (place, gauge or site, time, eot20_m) and model_places.csv

Usage: python 09_gauge_model.py --models <folder with EOT20/> --scenes <tide out>/scenes.csv --work <gauge work folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--scenes", required=True)
ap.add_argument("--work", required=True)
a = ap.parse_args()
work = Path(a.work)
GAUGES = {"maji": (56.606, 24.518), "diba": (56.269, 25.649), "wuda": (57.52, 23.82), "musc": (58.566667, 23.633333),
          "kisr": (48.094841, 29.351944)}
CUTOFF_KM = 100.0

S = pd.read_csv(a.scenes)
S["t"] = pd.to_datetime(S["time_utc"]).dt.tz_localize(None)
when = S.groupby("date")["t"].first()
grid = set()
for t in when:
    grid.update(pd.date_range((t - pd.Timedelta(days=7.5)).floor("D"), (t + pd.Timedelta(days=7.5)).ceil("D"), freq="10min"))
grid = np.array(sorted(grid), dtype="datetime64[s]")
print("times on the grid:", len(grid), flush=True)


def tide(lon, lat, times, extrapolate=True):
    t = np.asarray(times, dtype="datetime64[s]")
    kw = dict(extrapolate=True, cutoff=CUTOFF_KM) if extrapolate else dict(extrapolate=False)
    h = pyTMD.compute.tide_elevations(np.full(len(t), lon), np.full(len(t), lat), t, directory=a.models, model="EOT20",
                                      crs=4326, standard="datetime", type="trajectory", method="linear", **kw)
    return np.asarray(h, dtype=float)


places = [(k, "gauge", lon, lat) for k, (lon, lat) in GAUGES.items()]
places += [(s, "site", float(g["site_lon"].iloc[0]), float(g["site_lat"].iloc[0])) for s, g in S.groupby("site")]
rows, meta = [], []
for name, kind, lon, lat in places:
    # the exact overpass times are added to the grid so they can be read off without interpolating
    exact = S.loc[S["site"] == name, "t"].values.astype("datetime64[s]") if kind == "site" else when.values.astype("datetime64[s]")
    t = np.concatenate([grid, exact])
    h = tide(lon, lat, t)
    if not np.all(np.isfinite(h)):
        raise SystemExit(f"no tide value at {name}")
    is_water = bool(np.isfinite(tide(lon, lat, grid[:1], extrapolate=False))[0])
    meta.append(dict(place=name, kind=kind, lon=lon, lat=lat, point_is_model_water=is_water))
    rows.append(pd.DataFrame(dict(place=name, kind=kind, time=pd.to_datetime(t), eot20_m=np.round(h, 4),
                                  at_overpass=np.r_[np.zeros(len(grid), bool), np.ones(len(exact), bool)])))
    print(name, kind, "model water cell" if is_water else "nearest model water cell used", flush=True)
pd.concat(rows).to_csv(work / "model_grid.csv", index=False)
pd.DataFrame(meta).to_csv(work / "model_places.csv", index=False)
print("done")
