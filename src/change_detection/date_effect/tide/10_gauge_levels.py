"""Water level at each tide gauge at the moment of each scene, compared with the EOT20 tide model.

For each gauge and scene date:
  - the raw IOC record is cleaned: 10-minute medians, then gross outliers, spikes and stuck values are dropped.
    The UHSLC hourly record is already checked, so it is only put on the 10-minute grid with a smooth curve.
  - the zero level is the average of the 15 days centred on the overpass, because the gauges have no common datum.
    EOT20 at the gauge position is treated the same way.
  - the level at the overpass is read off the cleaned record.
  - over the 15 days: the correlation between gauge and model, the time shift that fits best, the size ratio and
    the typical difference. The same against EOT20 at Jubail Island, to see how far the gauge's tide is from the
    tide at our sites.

Nothing here uses the mangrove predictions.

Reads <work>/ioc, <work>/uhslc, <work>/model_grid.csv and <tide out>/scenes.csv, scene_tides.csv. Writes to <out>:
    gauge_at_overpass.csv   one row per gauge and scene date
    gauge_vs_model.csv      one row per gauge

Usage: python 10_gauge_levels.py --work <gauge work folder> --tide-out <tide out> --out <tide out>/gauge
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--tide-out", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, tide_out, out = Path(a.work), Path(a.tide_out), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
HALF = pd.Timedelta(days=7.5)
MIN_COVER = 0.6                       # share of the hours in the 15 days that must have data
REF_SITE = "jubail_island"
# station code, sensor, name used in the tables
GAUGES = [("maji", "rad", "Majis, Oman (IOC)"), ("diba", "rad", "Diba, Oman (IOC)"), ("wuda", "rad", "Khawr Wudam, Oman (IOC)"),
          ("musc", "prs", "Muscat, Oman (IOC)"), ("kisr", "pr1", "Salmiya, Kuwait (IOC)"), ("kisr", "rad", "Salmiya, Kuwait (IOC, radar)")]

S = pd.read_csv(tide_out / "scenes.csv")
S["t"] = pd.to_datetime(S["time_utc"]).dt.tz_localize(None)
when = S.groupby("date")["t"].first()
M = pd.read_csv(work / "model_grid.csv", parse_dates=["time"])
ST = pd.read_csv(tide_out / "scene_tides.csv")


def grid_of(place):
    return M[(M["place"] == place) & ~M["at_overpass"]].drop_duplicates("time").set_index("time")["eot20_m"].sort_index()


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
    b = b[run.groupby(run).transform("size") < 18]                              # stuck for 3 hours or more
    return b


def at(series, t, tol="40min"):
    """Level at time t by straight-line interpolation, if there is data on both sides within tol."""
    lo, hi = series[:t].tail(1), series[t:].head(1)
    if len(lo) == 0 or len(hi) == 0 or (t - lo.index[0]) > pd.Timedelta(tol) or (hi.index[0] - t) > pd.Timedelta(tol):
        return np.nan
    if hi.index[0] == lo.index[0]:
        return float(lo.iloc[0])
    w = (t - lo.index[0]) / (hi.index[0] - lo.index[0])
    return float(lo.iloc[0] + w * (hi.iloc[0] - lo.iloc[0]))


def fit(g, m):
    """Gauge g against model m, both with their average removed."""
    best = None
    for lag in range(-360, 361, 10):                                            # minutes; positive = the gauge is later than the model
        mm = m.copy()
        mm.index = mm.index + pd.Timedelta(minutes=lag)
        j = pd.concat([g, mm], axis=1, join="inner").dropna()
        if len(j) < 200:
            continue
        r = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
        if best is None or r > best[1]:
            best = (lag, r, j)
    if best is None:
        return dict(r_no_shift=np.nan, best_shift_min=np.nan, r_best_shift=np.nan, size_ratio=np.nan, rms_diff_m=np.nan)
    j0 = pd.concat([g, m], axis=1, join="inner").dropna()
    r0 = np.corrcoef(j0.iloc[:, 0], j0.iloc[:, 1])[0, 1]
    lag, r, j = best
    return dict(r_no_shift=round(float(r0), 3), best_shift_min=lag, r_best_shift=round(float(r), 3),
                size_ratio=round(float(j.iloc[:, 0].std() / j.iloc[:, 1].std()), 2),
                rms_diff_m=round(float(np.sqrt(((j0.iloc[:, 0] - j0.iloc[:, 1]) ** 2).mean())), 3))


def load_ioc(code, sensor):
    rows = []
    for f in sorted((work / "ioc" / code).glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8", errors="replace") or "[]")
        except json.JSONDecodeError:
            print("  unreadable file", f.name)
            continue
        if isinstance(d, list):
            rows += [(x.get("stime"), x.get("slevel")) for x in d if isinstance(x, dict) and x.get("sensor") == sensor]
    k = pd.DataFrame(rows, columns=["t", "v"])
    k["t"] = pd.to_datetime(k["t"], errors="coerce")
    k["v"] = pd.to_numeric(k["v"], errors="coerce")
    k = k.dropna().drop_duplicates("t").sort_values("t")
    return pd.Series(k["v"].values, index=pd.DatetimeIndex(k["t"]))


series = {}
for code, sensor, name in GAUGES:
    raw = load_ioc(code, sensor)
    print(name, sensor, "raw samples", len(raw), flush=True)
    series[(code, sensor, name)] = (raw, False)
u = pd.read_csv(work / "uhslc" / "muscat_hourly.csv", skiprows=[1])
uv = pd.to_numeric(u["sea_level"], errors="coerce") / 1000.0                  # millimetres to metres
uv[(uv < -30) | (uv > 30)] = np.nan
ut = pd.DatetimeIndex(pd.to_datetime(u["time"]).dt.tz_localize(None))
series[("musc", "uhslc_hourly", "Muscat, Oman (UHSLC hourly)")] = (pd.Series(uv.values, index=ut).dropna().sort_index(), True)

rows, summ = [], []
site_grid = grid_of(REF_SITE)
for (code, sensor, name), (raw, hourly) in series.items():
    if len(raw) == 0:
        print(name, "nothing downloaded")
        continue
    mg = grid_of(code)
    for d, t in when.items():
        w = raw[t - HALF:t + HALF]
        if len(w) == 0:
            continue
        if hourly:
            c = w[(w - w.median()).abs() < 3.0]
            c = c.resample("10min").asfreq().interpolate("cubic", limit=5, limit_area="inside").dropna()
        else:
            c = clean(w)
        cover = c.index.floor("h").nunique() / (15 * 24)
        lvl = at(c, t)
        if cover < MIN_COVER or not np.isfinite(lvl):
            rows.append(dict(gauge=name, code=code, sensor=sensor, date=d, time_utc=t, cover_15d=round(cover, 2), usable=False))
            continue
        g = c - c.mean()
        m = mg[t - HALF:t + HALF]
        m = m - m.mean()
        f = fit(g, m)
        flipped = bool(np.isfinite(f["r_no_shift"]) and f["r_no_shift"] < -0.5)
        if flipped:                                                              # a sensor that reports distance to the water, so up is down
            g = -g
            f = fit(g, m)
        sg = site_grid[t - HALF:t + HALF]
        fs = fit(g, sg - sg.mean())
        gauge_now, model_now = at(g, t), at(m, t)
        rows.append(dict(gauge=name, code=code, sensor=sensor, date=d, time_utc=t, cover_15d=round(cover, 2), usable=True,
                         sign_flipped=flipped, gauge_m=round(gauge_now, 3), eot20_at_gauge_m=round(model_now, 3),
                         gauge_minus_model_m=round(gauge_now - model_now, 3),
                         r_15d=f["r_no_shift"], best_shift_min=f["best_shift_min"], r_15d_best_shift=f["r_best_shift"],
                         size_ratio=f["size_ratio"], rms_diff_15d_m=f["rms_diff_m"],
                         r_15d_vs_site_model=fs["r_no_shift"], site_model_best_shift_min=fs["best_shift_min"],
                         r_15d_vs_site_model_best_shift=fs["r_best_shift"], size_ratio_vs_site_model=fs["size_ratio"]))
    G = pd.DataFrame([r for r in rows if r["gauge"] == name and r["usable"]])
    if len(G) == 0:
        print(name, "no usable dates")
        continue
    # one value per pass for the comparison with the site heights the tide test used (mean of the sites imaged that day)
    one = ST.merge(G[["date", "gauge_m"]], on="date").groupby("date")[["gauge_m", "tide_eot20_m"]].mean()
    summ.append(dict(gauge=name, sensor=sensor, scene_dates=len(G), dates=" ".join(G["date"]),
                     sign_flipped=bool(G["sign_flipped"].any()),
                     r_15d_median=round(float(G["r_15d"].median()), 3), r_15d_min=round(float(G["r_15d"].min()), 3),
                     best_shift_min_median=float(G["best_shift_min"].median()), size_ratio_median=round(float(G["size_ratio"].median()), 2),
                     rms_diff_15d_m_median=round(float(G["rms_diff_15d_m"].median()), 3),
                     r_overpass_gauge_vs_model_at_gauge=round(float(np.corrcoef(G["gauge_m"], G["eot20_at_gauge_m"])[0, 1]), 3) if len(G) >= 3 else np.nan,
                     rms_overpass_gauge_minus_model_m=round(float(np.sqrt((G["gauge_minus_model_m"] ** 2).mean())), 3),
                     max_abs_overpass_gauge_minus_model_m=round(float(G["gauge_minus_model_m"].abs().max()), 3),
                     range_of_gauge_at_overpass_m=round(float(G["gauge_m"].max() - G["gauge_m"].min()), 2),
                     r_overpass_gauge_vs_site_model_used=round(float(np.corrcoef(one["gauge_m"], one["tide_eot20_m"])[0, 1]), 3) if len(one) >= 3 else np.nan,
                     r_15d_vs_site_model_median=round(float(G["r_15d_vs_site_model"].median()), 3),
                     site_model_best_shift_min_median=float(G["site_model_best_shift_min"].median()),
                     r_15d_vs_site_model_best_shift_median=round(float(G["r_15d_vs_site_model_best_shift"].median()), 3),
                     size_ratio_vs_site_model_median=round(float(G["size_ratio_vs_site_model"].median()), 2)))
R = pd.DataFrame(rows)
R.to_csv(out / "gauge_at_overpass.csv", index=False)
Q = pd.DataFrame(summ)
Q.to_csv(out / "gauge_vs_model.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("not usable:\n", R[~R["usable"]][["gauge", "sensor", "date", "cover_15d"]].to_string(index=False))
print()
print(Q.drop(columns=["dates"]).T.to_string())
