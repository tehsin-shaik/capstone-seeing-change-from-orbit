"""Download the gauge records around our overpass times.

IOC gauges (raw data, 1 to 15 minutes apart): for each station below and each scene date the probe found data for,
the 15 days centred on the overpass. 15 days is one spring-neap cycle, so the average of the window is a fair zero
level. Overlapping windows are joined and fetched in 5-day pieces, kept as they came in <work>/ioc/<code>/.
UHSLC Muscat (hourly, quality checked): the whole record from 2018 to mid 2021 in <work>/uhslc/muscat_hourly.csv.

Files already there are not fetched again. The IOC server is slow, so this takes a while (about 45 MB).

Usage: python 08_gauge_download.py --scenes <tide out>/scenes.csv --work <gauge work folder>
"""
import argparse
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--only", default="")              # e.g. "kisr" to fetch just that station
a = ap.parse_args()
work = Path(a.work)
IOC = "https://www.ioc-sealevelmonitoring.org/service.php"
MUSCAT = ("https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_fast.csv?time,sea_level,uhslc_id&uhslc_id=110"
          "&time%3E=2018-01-01T00:00:00Z&time%3C=2021-06-30T00:00:00Z")
# station code: the sensors to fetch (rad = radar, prs and pr1 = pressure)
STATIONS = {"maji": ["rad"], "diba": ["rad"], "wuda": ["rad"], "musc": ["prs"], "kisr": ["pr1", "rad"]}
if a.only:
    STATIONS = {k: v for k, v in STATIONS.items() if k in a.only.split(",")}
HALF = pd.Timedelta(days=7.5)
AGENT = {"User-Agent": "tide-gauge-check/1.0"}

S = pd.read_csv(a.scenes)
S["t"] = pd.to_datetime(S["time_utc"]).dt.tz_localize(None)
when = S.groupby("date")["t"].first()
P = pd.read_csv(work / "ioc_probe.csv")
P = P[P["samples"] > 0]


def fetch(url, tries=4):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=300) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print("  retry", e, flush=True)
            time.sleep(8)
    return None


for code, sensors in STATIONS.items():
    dates = sorted(P[(P["code"] == code) & P["sensor"].isin(sensors)]["date"].unique())
    spans = []
    for d in dates:
        t0, t1 = (when[d] - HALF).floor("D"), (when[d] + HALF).ceil("D")
        if spans and t0 <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], t1)
        else:
            spans.append([t0, t1])
    folder = work / "ioc" / code
    folder.mkdir(parents=True, exist_ok=True)
    total = 0
    for t0, t1 in spans:
        t = t0
        while t < t1:
            u = min(t + pd.Timedelta(days=5), t1)
            f = folder / f"{t:%Y%m%d}_{u:%Y%m%d}.json"
            if not f.exists():
                q = [("query", "data"), ("code", code), ("timestart", f"{t:%Y-%m-%dT%H:%M:%S}"), ("timestop", f"{u:%Y-%m-%dT%H:%M:%S}"), ("format", "json")]
                q += [("includesensors[]", s) for s in sensors]
                body = fetch(IOC + "?" + urllib.parse.urlencode(q))
                if body is None:
                    print("  FAILED", code, t, flush=True)
                    t = u
                    continue
                f.write_bytes(body)
                time.sleep(1.0)
            total += f.stat().st_size
            t = u
    print(code, "scene dates", len(dates), "windows", len(spans), "MB", round(total / 1e6, 1), flush=True)

f = work / "uhslc" / "muscat_hourly.csv"
if not a.only and not f.exists():
    f.parent.mkdir(parents=True, exist_ok=True)
    body = fetch(MUSCAT)
    if body is None:
        raise SystemExit("could not download the UHSLC Muscat record")
    f.write_bytes(body)
    print("UHSLC Muscat MB", round(len(body) / 1e6, 1))
print("done")
