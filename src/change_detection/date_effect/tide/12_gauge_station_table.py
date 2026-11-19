"""One table of every tide gauge found in or near the Gulf: where it is, what it covers, which of our dates it has.

Joins the station lists from 06 with the probes from 07. A probe row only counts if a water level sensor returned
values that move (some stations send zeros or one stuck value). The IOC years come from four days a year, so they
are approximate.

Writes <out>/stations_found.csv

Usage: python 12_gauge_station_table.py --work <gauge work folder> --out <tide out>/gauge
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work = Path(a.work)
LEVEL = {"rad", "ra2", "ra3", "prs", "pr1", "pr2", "pr3", "enc", "en2", "flt", "bub", "aqu", "wls", "prt", "pwl", "bwl", "stp", "ecs"}
# which water each station is on, from the map
WATER = {"ashk": "Arabian Sea", "kisr": "the Gulf (north end)", "Mina Sulman": "the Gulf (Bahrain)"}

ioc = pd.read_csv(work / "ioc_stations_gulf.csv")
uh = pd.read_csv(work / "uhslc_stations_gulf.csv")
gesla = pd.read_csv(work / "gesla_records_gulf.csv")
P = pd.read_csv(work / "ioc_probe.csv")
Y = pd.read_csv(work / "ioc_probe_years.csv")
U = pd.read_csv(work / "uhslc_probe.csv")


def moving(d):
    d = d[(d["samples"] > 0) & d["sensor"].isin(LEVEL)]
    return d[(d["max"] - d["min"]).abs() > 0.02]


P, Y = moving(P), moving(Y)
rows = []
for _, s in ioc.iterrows():
    c = s["code"]
    dates = sorted(P[P["code"] == c]["date"].unique())
    years = sorted({d[:4] for d in Y[Y["code"] == c]["date"]} | {d[:4] for d in dates})
    n = P[P["code"] == c].groupby("date")["samples"].max()                    # samples in the two hours around the overpass
    rows.append(dict(archive="IOC", station=s["name"], code=c, country=s["country"], water=WATER.get(c, "Gulf of Oman"),
                     lon=s["lon"], lat=s["lat"], km_to_nearest_site=s["km_to_nearest_site"], km_to_farthest_site=s["km_to_farthest_site"],
                     years_with_data=" ".join(years) if years else "none found",
                     sampling=f"about every {round(120 / n.median())} min, raw" if len(n) else "",
                     scene_dates_covered=len(dates), scene_dates=" ".join(dates)))
in_gesla = set(gesla["station_code"].astype(str).str.extract(r"(\d+)")[0].astype(int))
assert in_gesla <= set(uh["code"]), "GESLA has a record here that is not a UHSLC station"
for _, s in uh.iterrows():
    spans = [(s[k + "_first"], s[k + "_last"]) for k in ("research", "fast_delivery") if isinstance(s[k + "_first"], str)]
    lo, hi = min(x[0] for x in spans), max(x[1] for x in spans)
    dates = sorted(U[(U["code"] == s["code"]) & (U["hourly_values_within_1h"] > 0)]["date"])
    rows.append(dict(archive="UHSLC, GESLA" if s["code"] in in_gesla else "UHSLC", station=s["name"], code=s["code"], country=s["country"],
                     water=WATER.get(s["name"], "Gulf of Oman"), lon=s["lon"], lat=s["lat"], km_to_nearest_site=s["km_to_nearest_site"],
                     km_to_farthest_site=s["km_to_farthest_site"], years_with_data=f"{lo[:4]} to {hi[:4]}",
                     sampling="hourly, quality checked", scene_dates_covered=len(dates), scene_dates=" ".join(dates)))
T = pd.DataFrame(rows).sort_values(["km_to_nearest_site", "archive", "code"])
T.to_csv(Path(a.out) / "stations_found.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
print(T.drop(columns=["scene_dates", "lon", "lat"]).to_string(index=False))
