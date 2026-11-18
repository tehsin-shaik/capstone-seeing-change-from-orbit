"""Predicted tide height at each site at the moment of each scene, from open global tide models.

Models (both downloaded with no login, read with the pyTMD library):
    EOT20    DGFI-TUM, 1/8 degree, 17 constituents, https://doi.org/10.17882/79489  (main)
    GOT4.10c NASA GSFC, 1/2 degree, 10 constituents                               (cross-check)
Heights are metres above the model's mean sea level. The sites sit in lagoons that the model grids do not
resolve, so where the site centre is not a model water cell the nearest model water cell is used
(pyTMD extrapolate=True, up to 100 km). The tide inside a lagoon arrives later and smaller than at the open
coast, so the script also gives the height 1, 2 and 3 hours earlier (what the lagoon would see if it lagged
the coast by that much).

Reads <out>/scenes.csv (from 01_scene_times.py). Writes:
    <out>/scene_tides.csv        one row per scene with the heights, that day's low and high, and
                                 where the height sits between them (0 = low water, 1 = high water)
    <out>/tide_at_overpass_by_month.csv   EOT20 height at 07:12 UTC on every day of 2018, 2020 and 2025
                                 at each site, averaged per month (does the satellite's fixed time of
                                 day make the tide it sees depend on the season?)

Usage: python 03_tide_model.py --models <folder with EOT20/ and GOT4.10c/> --out <results folder>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
sc = pd.read_csv(out / "scenes.csv")
CUTOFF_KM = 100.0


def tide(model, lon, lat, times):
    t = np.asarray(times, dtype="datetime64[s]")
    h = pyTMD.compute.tide_elevations(np.full(len(t), lon), np.full(len(t), lat), t, directory=a.models, model=model,
                                      crs=4326, standard="datetime", type="trajectory", method="linear",
                                      extrapolate=True, cutoff=CUTOFF_KM)
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        raise SystemExit(f"{model}: no tide value at {lon},{lat}")
    return h


def direct(model, lon, lat):
    """True if the site centre is itself a model water cell (no extrapolation needed)."""
    t = np.array(["2020-01-01T00:00:00"], dtype="datetime64[s]")
    h = pyTMD.compute.tide_elevations(np.array([lon]), np.array([lat]), t, directory=a.models, model=model, crs=4326,
                                      standard="datetime", type="trajectory", method="linear", extrapolate=False)
    return bool(np.isfinite(np.asarray(h, dtype=float))[0])


rows = []
step = np.timedelta64(10, "m")
for site, g in sc.groupby("site"):
    lon, lat = float(g["site_lon"].iloc[0]), float(g["site_lat"].iloc[0])
    t = pd.to_datetime(g["time_utc"], utc=True).dt.tz_localize(None).values.astype("datetime64[s]")
    is_direct = {m: direct(m, lon, lat) for m in ("EOT20", "GOT4.10")}
    h = {m: tide(m, lon, lat, t) for m in ("EOT20", "GOT4.10")}
    lag = {k: tide("EOT20", lon, lat, t - np.timedelta64(k, "h")) for k in (1, 2, 3)}
    # the tide curve from 13 hours before to 13 hours after each scene, every 10 minutes (one model call for the site)
    offs = np.arange(-78, 79) * step
    curves = tide("EOT20", lon, lat, np.concatenate([ti + offs for ti in t])).reshape(len(t), len(offs))
    for i, (_, r) in enumerate(g.iterrows()):
        curve = curves[i]
        lo, hi = float(curve.min()), float(curve.max())
        rows.append({"site": site, "date": r["date"], "time_utc": r["time_utc"],
                     "tide_eot20_m": round(float(h["EOT20"][i]), 3), "tide_got410_m": round(float(h["GOT4.10"][i]), 3),
                     "tide_eot20_1h_earlier_m": round(float(lag[1][i]), 3), "tide_eot20_2h_earlier_m": round(float(lag[2][i]), 3),
                     "tide_eot20_3h_earlier_m": round(float(lag[3][i]), 3),
                     "day_low_m": round(lo, 3), "day_high_m": round(hi, 3), "day_range_m": round(hi - lo, 3),
                     "tide_position_0low_1high": round((float(h["EOT20"][i]) - lo) / (hi - lo), 3),
                     "rising": bool(curve[79] > curve[77]),
                     "eot20_site_is_model_water": is_direct["EOT20"], "got410_site_is_model_water": is_direct["GOT4.10"]})
    print(site, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(out / "scene_tides.csv", index=False)

# the same EOT20 height at delays from 0 to 6 hours in half-hour steps (long table, for the lag check)
lags = []
for site, g in sc.groupby("site"):
    lon, lat = float(g["site_lon"].iloc[0]), float(g["site_lat"].iloc[0])
    t = pd.to_datetime(g["time_utc"], utc=True).dt.tz_localize(None).values.astype("datetime64[s]")
    halves = np.arange(0, 13)
    hh = tide("EOT20", lon, lat, np.concatenate([t - np.timedelta64(30 * int(k), "m") for k in halves])).reshape(len(halves), len(t))
    for k in halves:
        lags += [{"site": site, "date": d, "hours_earlier": k / 2, "tide_eot20_m": round(float(v), 3)}
                 for d, v in zip(g["date"], hh[k])]
pd.DataFrame(lags).to_csv(out / "scene_tides_lagged.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 100):
    print(res.to_string(index=False))
print("EOT20 against GOT4.10c over the scenes: r =", round(float(np.corrcoef(res.tide_eot20_m, res.tide_got410_m)[0, 1]), 3),
      " largest difference", round(float((res.tide_eot20_m - res.tide_got410_m).abs().max()), 3), "m")

# ---- what tide does a satellite that always passes at 07:12 UTC see through the year?
days = np.concatenate([np.arange(f"{y}-01-01", f"{y + 1}-01-01", dtype="datetime64[D]") for y in (2018, 2020, 2025)])
at = days.astype("datetime64[s]") + np.timedelta64(7 * 3600 + 12 * 60, "s")
mm = []
for site, g in sc.groupby("site"):
    hh = tide("EOT20", float(g["site_lon"].iloc[0]), float(g["site_lat"].iloc[0]), at)
    d = pd.DataFrame({"site": site, "month": pd.to_datetime(days).month, "h": hh})
    m = d.groupby(["site", "month"])["h"].agg(["mean", "std", "min", "max"]).round(3).reset_index()
    doy = pd.to_datetime(days).dayofyear.values
    X = np.column_stack([np.ones(len(doy)), np.cos(2 * np.pi * doy / 365.25), np.sin(2 * np.pi * doy / 365.25)])
    fit = X @ np.linalg.lstsq(X, hh, rcond=None)[0]
    m["share_of_overpass_tide_variance_that_is_seasonal"] = round(1 - np.var(hh - fit) / np.var(hh), 3)
    mm.append(m)
mm = pd.concat(mm)
mm.columns = ["site", "month", "mean_tide_m", "sd_m", "min_m", "max_m", "share_of_overpass_tide_variance_that_is_seasonal"]
mm.to_csv(out / "tide_at_overpass_by_month.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 100):
    print(mm[mm.site == "jubail_island"].to_string(index=False))
