"""Ask the archives which Gulf gauges have data at our overpass times.

IOC: for every station from 06 and every scene date, asks for the two hours around the overpass and counts the
samples that came back per sensor. The same is then done for four days in every year from 2016 on, to see roughly
which years each station covers. UHSLC: reads the hourly record of 2018, 2020 and 2025 for each station and checks
for a value at each overpass. Only the counts are kept. Each table is skipped if it is already there.

Writes to <work>: ioc_probe.csv, ioc_probe_years.csv, uhslc_probe.csv

Usage: python 07_gauge_probe.py --scenes <tide out>/scenes.csv --work <gauge work folder>
"""
import argparse
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", required=True)
ap.add_argument("--work", required=True)
a = ap.parse_args()
work = Path(a.work)
IOC = "https://www.ioc-sealevelmonitoring.org/service.php"
UHSLC = "https://uhslc.soest.hawaii.edu/erddap/tabledap/global_hourly_fast.csv"
# the IOC names for water level sensors (radar, pressure, encoder, float and so on)
SENSORS = ["rad", "ra2", "ra3", "prs", "pr1", "pr2", "pr3", "enc", "en2", "flt", "bub", "aqu", "wls", "prt", "pwl", "bwl", "stp", "ecs"]
AGENT = {"User-Agent": "tide-gauge-check/1.0"}

S = pd.read_csv(a.scenes)
S["t"] = pd.to_datetime(S["time_utc"]).dt.tz_localize(None)
overpass = list(S.groupby("date")["t"].first())               # one time per scene date
codes = sorted(pd.read_csv(work / "ioc_stations_gulf.csv")["code"].unique())


def ioc_data(code, t0, t1):
    q = [("query", "data"), ("code", code), ("timestart", f"{t0:%Y-%m-%dT%H:%M:%S}"), ("timestop", f"{t1:%Y-%m-%dT%H:%M:%S}"), ("format", "json")]
    q += [("includesensors[]", s) for s in SENSORS]
    req = urllib.request.Request(IOC + "?" + urllib.parse.urlencode(q), headers=AGENT)
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8", "replace") or "[]")
            return d if isinstance(d, list) else []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            err = e
            time.sleep(3)
    print("  no answer for", code, t0, err)
    return None


def probe(times, name):
    f = work / name
    if f.exists():
        print(name, "is already there")
        return
    rows = []
    for c in codes:
        for t in times:
            d = ioc_data(c, t - pd.Timedelta(hours=1), t + pd.Timedelta(hours=1))
            time.sleep(0.4)
            if not d:
                rows.append(dict(code=c, date=str(t.date()), sensor="request failed" if d is None else "", samples=0))
                continue
            k = pd.DataFrame(d)
            for s, g in k.groupby("sensor"):
                v = pd.to_numeric(g["slevel"], errors="coerce")
                rows.append(dict(code=c, date=str(t.date()), sensor=s, samples=len(g), min=v.min(), max=v.max()))
        got = sorted({r["date"] for r in rows if r["code"] == c and r["samples"] > 0})
        print(c, "dates with data:", len(got), flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)


probe(overpass, "ioc_probe.csv")
now = pd.Timestamp.now()
probe([t for y in range(2016, now.year + 1) for m in (1, 4, 7, 10) if (t := pd.Timestamp(y, m, 15, 7)) < now], "ioc_probe_years.csv")

f = work / "uhslc_probe.csv"
if not f.exists():
    rows = []
    for _, s in pd.read_csv(work / "uhslc_stations_gulf.csv").iterrows():
        for y in sorted({t.year for t in overpass}):
            url = f"{UHSLC}?time,sea_level&uhslc_id={int(s['code'])}&time%3E={y}-01-01T00:00:00Z&time%3C={y + 1}-01-01T00:00:00Z"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=300) as r:
                    d = pd.read_csv(io.BytesIO(r.read()), skiprows=[1])
            except urllib.error.HTTPError:                           # the server answers with an error when there are no rows
                d = pd.DataFrame(columns=["time", "sea_level"])
            v = pd.Series(pd.to_numeric(d["sea_level"], errors="coerce").values, index=pd.to_datetime(d["time"]).dt.tz_localize(None)).dropna()
            for t in overpass:
                if t.year == y:
                    n = len(v[t - pd.Timedelta(hours=1):t + pd.Timedelta(hours=1)])
                    rows.append(dict(code=int(s["code"]), name=s["name"], date=str(t.date()), hourly_values_within_1h=n))
            time.sleep(1)
        print(s["name"], "scene dates with data:", sum(r["hourly_values_within_1h"] > 0 for r in rows if r["code"] == int(s["code"])), flush=True)
    pd.DataFrame(rows).to_csv(f, index=False)
