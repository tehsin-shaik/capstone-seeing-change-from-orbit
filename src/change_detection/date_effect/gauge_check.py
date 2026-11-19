"""Tide gauge check: is the tide model right where it can be compared with real tide gauges, and does the same-year
tide test change with gauge readings in it?

No UAE gauge is in the open archives, so the gauges are in Oman (Gulf of Oman) and Kuwait. They check the model and our
timing, not the water level at our sites. Run tide_test.py first (this reads scenes.csv and tide_scene_table.csv from <work>).

Gauge records (downloaded beforehand, not in the repository):
    <records>/ioc/<code>/*.json        IOC Sea Level Station Monitoring Facility, raw, 1 to 15 minutes apart, the 15 days
                                       around each overpass, for maji (Majis), diba (Diba), wuda (Khawr Wudam), kisr (Salmiya)
    <records>/uhslc/muscat_hourly.csv  UHSLC fast delivery, Muscat (station 110), hourly and quality checked

For each gauge and scene date: the raw record is cleaned (10-minute medians; gross outliers, spikes and stuck values
dropped), the zero level is the average of the 15 days centred on the overpass (the gauges have no common datum, and 15
days is one spring-neap cycle), and the level at the overpass is read off. EOT20 at the gauge position is treated the
same way. The same-year test is the one of tide_test.py with three other heights in place of the model:
    gauge itself               the level at the gauge, which is in other water
    model + gauge difference   EOT20 at the site plus how far the gauge was above or below EOT20 at the gauge
    model, same scenes         EOT20 at the site on the scenes that gauge covers, to compare like with like

Writes <out>/summary.csv.     Usage: python gauge_check.py --models <folder with EOT20/> --gauge-records <records> --work <work> --out <results>/tide/gauge
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD

from shared import link

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--gauge-records", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
records, work, out = Path(a.gauge_records), Path(a.work), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
HALF = pd.Timedelta(days=7.5)
MIN_COVER = 0.6                       # share of the hours in the 15 days that must have data
# name: IOC code and sensor (None = the UHSLC hourly record), lon, lat, the water it is on
GAUGES = {"Muscat, Oman (UHSLC hourly)": (None, None, 58.566667, 23.633333, "Gulf of Oman"),
          "Majis, Oman (IOC)": ("maji", "rad", 56.606, 24.518, "Gulf of Oman"),
          "Diba, Oman (IOC)": ("diba", "rad", 56.269, 25.649, "Gulf of Oman"),
          "Khawr Wudam, Oman (IOC)": ("wuda", "rad", 57.52, 23.82, "Gulf of Oman"),
          "Salmiya, Kuwait (IOC)": ("kisr", "pr1", 48.094841, 29.351944, "the Gulf, north end")}
IN_TEST = ["Muscat, Oman (UHSLC hourly)", "Majis, Oman (IOC)", "Khawr Wudam, Oman (IOC)", "Salmiya, Kuwait (IOC)"]

S = pd.read_csv(work / "scenes.csv")
S["t"] = pd.to_datetime(S["time_utc"]).dt.tz_localize(None)
when = S.groupby("date")["t"].first()                                       # one overpass time per scene date
sites = S.groupby("site")[["site_lon", "site_lat"]].first()
T = pd.read_csv(work / "tide_scene_table.csv")
site_tide = T.groupby("date")["tide_eot20_m"].mean()                        # one height per pass: mean of the sites imaged that day
T = T.dropna(subset=["dev_pct"]).reset_index(drop=True)                     # a site and year with one date has no same-year swing

# ---------------------------------------------------------------- EOT20 every 10 minutes around each scene date
grid = set()
for t in when:
    grid.update(pd.date_range((t - HALF).floor("D"), (t + HALF).ceil("D"), freq="10min"))
grid = np.array(sorted(grid), dtype="datetime64[s]")


def model(lon, lat):
    h = pyTMD.compute.tide_elevations(np.full(len(grid), lon), np.full(len(grid), lat), grid, directory=a.models, model="EOT20", crs=4326,
                                      standard="datetime", type="trajectory", method="linear", extrapolate=True, cutoff=100.0)
    h = pd.Series(np.round(np.asarray(h, dtype=float), 4), index=pd.to_datetime(grid))
    assert h.notna().all()
    return h


def km(lon, lat):
    """Distance from a point to each of our sites."""
    p1, p2 = np.radians(lat), np.radians(sites["site_lat"].values)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(sites["site_lon"].values - lon) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


# ---------------------------------------------------------------- gauge levels
def load_ioc(code, sensor):
    rows = []
    for f in sorted((records / "ioc" / code).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8", errors="replace") or "[]")
        if isinstance(d, list):
            rows += [(x.get("stime"), x.get("slevel")) for x in d if isinstance(x, dict) and x.get("sensor") == sensor]
    k = pd.DataFrame(rows, columns=["t", "v"])
    k["t"] = pd.to_datetime(k["t"], errors="coerce")
    k["v"] = pd.to_numeric(k["v"], errors="coerce")
    k = k.dropna().drop_duplicates("t").sort_values("t")
    return pd.Series(k["v"].values, index=pd.DatetimeIndex(k["t"]))


def load_uhslc():
    u = pd.read_csv(records / "uhslc" / "muscat_hourly.csv", skiprows=[1])
    v = pd.to_numeric(u["sea_level"], errors="coerce") / 1000.0                # millimetres to metres
    v[(v < -30) | (v > 30)] = np.nan
    return pd.Series(v.values, index=pd.DatetimeIndex(pd.to_datetime(u["time"]).dt.tz_localize(None))).dropna().sort_index()


def clean(s):
    """Raw levels indexed by time -> 10-minute medians with bad values removed."""
    b = s.resample("10min", offset="-5min").median().dropna()               # bins centred on :00, :10, ...
    b.index = b.index + pd.Timedelta(minutes=5)
    if len(b) < 20:
        return b
    b = b[(b - b.median()).abs() < 3.0]                                         # gross outliers (the tide range here is under 3 m)
    roll = b.rolling(9, center=True, min_periods=4).median()
    b = b[(b - roll).abs() < 0.25]                                              # spikes
    run = (b.diff().abs() > 1e-9).cumsum()
    return b[run.groupby(run).transform("size") < 18]                           # stuck for 3 hours or more


def at(series, t, tol="40min"):
    """Level at time t by straight-line interpolation, if there is data on both sides within tol."""
    lo, hi = series[:t].tail(1), series[t:].head(1)
    if len(lo) == 0 or len(hi) == 0 or (t - lo.index[0]) > pd.Timedelta(tol) or (hi.index[0] - t) > pd.Timedelta(tol):
        return np.nan
    if hi.index[0] == lo.index[0]:
        return float(lo.iloc[0])
    w = (t - lo.index[0]) / (hi.index[0] - lo.index[0])
    return float(lo.iloc[0] + w * (hi.iloc[0] - lo.iloc[0]))


def best_shift(g, m):
    """Minutes by which the gauge is later than the model (negative = earlier), where the two fit best."""
    best = (np.nan, -2.0)
    for lag in range(-360, 361, 10):
        mm = m.copy()
        mm.index = mm.index + pd.Timedelta(minutes=lag)
        j = pd.concat([g, mm], axis=1, join="inner").dropna()
        if len(j) >= 200:
            r = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
            if r > best[1]:
                best = (lag, r)
    return best[0]


jubail = model(*sites.loc["jubail_island"])
levels, rows = {}, []
for name, (code, sensor, lon, lat, water) in GAUGES.items():
    raw = load_uhslc() if code is None else load_ioc(code, sensor)
    mg = model(lon, lat)
    per = []
    for d, t in when.items():
        w = raw[t - HALF:t + HALF]
        if len(w) == 0:
            continue
        if code is None:                         # hourly and already checked: only put on the 10-minute grid with a smooth curve
            c = w[(w - w.median()).abs() < 3.0].resample("10min").asfreq().interpolate("cubic", limit=5, limit_area="inside").dropna()
        else:
            c = clean(w)
        if c.index.floor("h").nunique() / (15 * 24) < MIN_COVER or not np.isfinite(at(c, t)):
            continue
        g = c - c.mean()
        m = mg[t - HALF:t + HALF]
        m = m - m.mean()
        sj = jubail[t - HALF:t + HALF]
        gauge_now, model_now = at(g, t), at(m, t)
        per.append({"date": d, "gauge_m": round(gauge_now, 3), "model_m": round(model_now, 3), "diff_m": round(gauge_now - model_now, 3),
                    "shift_vs_site_min": best_shift(g, sj - sj.mean())})
    G = pd.DataFrame(per)
    levels[name] = G.set_index("date")
    dist = km(lon, lat)
    rows.append({"gauge": name, "water": water, "km_to_nearest_site": round(float(dist.min())), "km_to_farthest_site": round(float(dist.max())),
                 "scene_dates": len(G), "years": " ".join(sorted({d[:4] for d in G["date"]})),
                 "typical_difference_from_model_cm": round(100 * float(np.sqrt((G["diff_m"] ** 2).mean())), 1),
                 "largest_difference_from_model_cm": round(100 * float(G["diff_m"].abs().max()), 1),
                 "range_of_levels_at_overpass_m": round(float(G["gauge_m"].max() - G["gauge_m"].min()), 2),
                 "r_with_model_at_gauge": round(float(np.corrcoef(G["gauge_m"], G["model_m"])[0, 1]), 3),
                 "hours_gauge_tide_is_ahead_of_tide_at_our_sites": round(-float(G["shift_vs_site_min"].median()) / 60, 1) + 0.0,
                 "r_with_model_heights_at_our_sites": round(float(np.corrcoef(G["gauge_m"], G["date"].map(site_tide))[0, 1]), 2)})
    print(name, "usable scene dates:", len(G), flush=True)

# ---------------------------------------------------------------- the same-year test with gauge heights
tests = {}
for name in IN_TEST:
    T["gauge"] = T["date"].map(levels[name]["gauge_m"])
    T["adjusted"] = T["tide_eot20_m"] + T["date"].map(levels[name]["diff_m"])
    sub = T[T["gauge"].notna()]
    res = {}
    for key, x in (("gauge_itself", "gauge"), ("model_plus_gauge_difference", "adjusted"), ("model_same_scenes", "tide_eot20_m")):
        r = link(sub, x, "dev_pct")
        res["same_year_test_scenes"] = r["scenes"]
        res[f"share_of_swing_{key}_pct"], res[f"chance_{key}_pct"] = r["share_pct"], r["chance_shuffled_dates_pct"]
    tests[name] = res
R = pd.DataFrame([{**r, **tests.get(r["gauge"], {})} for r in rows])
# every scene: the model at the site plus the gauge difference of the gauge that covers that year
T["adjusted"] = T["tide_eot20_m"] + T["date"].map(levels[IN_TEST[0]]["diff_m"]).fillna(T["date"].map(levels[IN_TEST[-1]]["diff_m"]))
both, plain = link(T, "adjusted", "dev_pct"), link(T, "tide_eot20_m", "dev_pct")
R = pd.concat([R, pd.DataFrame([{"gauge": "all scenes (Muscat for 2018 and 2020, Kuwait for 2025)", "same_year_test_scenes": both["scenes"],
                                 "share_of_swing_model_plus_gauge_difference_pct": both["share_pct"],
                                 "chance_model_plus_gauge_difference_pct": both["chance_shuffled_dates_pct"],
                                 "share_of_swing_model_same_scenes_pct": plain["share_pct"],
                                 "chance_model_same_scenes_pct": plain["chance_shuffled_dates_pct"]}])], ignore_index=True)
for c in ("km_to_nearest_site", "km_to_farthest_site", "scene_dates", "same_year_test_scenes"):
    R[c] = R[c].astype("Int64")
R.to_csv(out / "summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(R.T.to_string())
