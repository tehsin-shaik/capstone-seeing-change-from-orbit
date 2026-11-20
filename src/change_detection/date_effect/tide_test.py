"""Tide test: does the water level explain the same-year swings in the forest's predicted mangrove area?

Steps (each is skipped if its file is already in <work>):
  1. scene times     sensing time, sun height and processing version of every non-hazy scene, looked up by item id in
                     the public Sentinel-2 archive (Earth Search, metadata only) -> scenes.csv
  2. predictions     the five held-out forests on every non-hazy scene -> forests/, stacks/
  3. tide heights    EOT20 tide model at the site centre at the sensing time (pyTMD) -> scene_tides.csv
  4. the test        each scene's same-year swing against the tide height and against the water share seen in the
                     image, and the time of year for comparison -> tide_scene_table.csv and <out>/summary.csv

The swing of a scene (dev_pct) is its predicted mangrove area as a % above or below the average of the dates of the
same site and year. For one site and year only the chip cells with at least two dates are used, and in a cell only the
pixels usable on all of them; the area of a date is then built from its cells (shared.ipf). Water share is built the
same way. A pixel is water if green (B03) is brighter than near-infrared (B08), the same rule on every date.
Rows marked "all years" use the older version, against the site's average over all its years, for comparison only.

Usage: python tide_test.py --train-val <Train_Val_Dataset> --models <folder with EOT20/> --work <work> --out <results>/tide
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from shared import CHIP, SITES, additive, forests_and_stacks, ipf, link, load_stack

ap = argparse.ArgumentParser()
ap.add_argument("--train-val", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tv, work, out = Path(a.train_val), Path(a.work), Path(a.out)
if "Train_Val_Dataset" not in str(tv):
    raise SystemExit("only the Train_Val_Dataset folder may be used")
work.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)
K3 = np.ones((3, 3), bool)

# ---------------------------------------------------------------- 1. scene times
if not (work / "scenes.csv").exists():
    from pyproj import Transformer
    from pystac_client import Client
    cat = pd.read_csv(tv / "catalog.csv")
    assert set(cat["role"]) == {"cv"}
    hz = pd.read_csv(tv / "hazy_scenes.csv")[["site", "date", "water_b04_p5"]].drop_duplicates(["site", "date"])
    cat = cat.merge(hz, on=["site", "date"], how="left")
    b = cat["bounds_epsg32640"].str.split(",", expand=True).astype(float)
    cat[["x0", "y0", "x1", "y1"]] = b.values
    to_ll = Transformer.from_crs(32640, 4326, always_xy=True)
    client = Client.open("https://earth-search.aws.element84.com/v1")
    rows = []
    for (site, date), g in cat[~(cat["water_b04_p5"] > 0.05)].groupby(["site", "date"]):      # above 0.05 is a hazy scene
        ids = g["item_id"].iloc[0].split("+")
        found = {it.id: it for it in client.search(collections=["sentinel-2-c1-l2a"], ids=ids).items()}
        assert all(i in found for i in ids), (site, date)
        times = pd.to_datetime([found[i].datetime for i in ids], utc=True)
        assert all(t.date().isoformat() == date for t in times), (site, date, times)
        t = times[0] + (times - times[0]).mean()
        s = cat[cat["site"] == site]
        lon, lat = to_ll.transform((s.x0.min() + s.x1.max()) / 2, (s.y0.min() + s.y1.max()) / 2)       # centre of the site's chips
        p = [found[i].properties for i in ids]
        rows.append({"site": site, "date": date, "year": int(g["year"].iloc[0]), "time_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "sun_elevation": round(float(np.mean([q.get("view:sun_elevation", np.nan) for q in p])), 2),
                     "processing_baseline": "|".join(sorted({str(q.get("s2:processing_baseline", "")) for q in p})),
                     "site_lon": round(lon, 5), "site_lat": round(lat, 5)})
    pd.DataFrame(rows).to_csv(work / "scenes.csv", index=False)
    print("scene times:", len(rows), "scenes", flush=True)
scenes = pd.read_csv(work / "scenes.csv")

# ---------------------------------------------------------------- 2. predictions
forests_and_stacks(tv, work)

# ---------------------------------------------------------------- 3. tide heights
# The sites sit in lagoons the model grid does not resolve, so where the site centre is not a model water cell the
# nearest one is used (up to 100 km). Heights are metres above the model's mean sea level.
if not (work / "scene_tides.csv").exists():
    import pyTMD
    rows = []
    for site, g in scenes.groupby("site"):
        t = pd.to_datetime(g["time_utc"], utc=True).dt.tz_localize(None).values.astype("datetime64[s]")
        lon, lat = float(g["site_lon"].iloc[0]), float(g["site_lat"].iloc[0])
        h = pyTMD.compute.tide_elevations(np.full(len(t), lon), np.full(len(t), lat), t, directory=a.models, model="EOT20", crs=4326,
                                          standard="datetime", type="trajectory", method="linear", extrapolate=True, cutoff=100.0)
        h = np.asarray(h, dtype=float)
        assert np.all(np.isfinite(h)), f"no tide value at {site}"
        rows += [{"site": site, "date": d, "tide_eot20_m": round(float(v), 3)} for d, v in zip(g["date"], h)]
        print("tide heights:", site, flush=True)
    pd.DataFrame(rows).to_csv(work / "scene_tides.csv", index=False)
scenes = scenes.merge(pd.read_csv(work / "scene_tides.csv"), on=["site", "date"])

# ---------------------------------------------------------------- 4. the test
rows = []
strip = np.zeros(4)                         # pixels, flipping pixels, pixels near the strip, flipping pixels near the strip
for site in SITES:
    S = load_stack(work, site)
    dates = [str(d) for d in S["dates"]]
    years = np.array([int(d[:4]) for d in dates])
    n = len(dates)
    valid, covered = S["valid"], S["covered"]
    # probabilities rounded to half precision, as they were stored when the test was first run
    mask = (S["prob"].astype(np.float16).astype(np.float32) > float(S["threshold"])) & valid
    with np.errstate(invalid="ignore"):
        water = (S["image"][:, 1] > S["image"][:, 3]) & valid
    H, W = valid.shape[1:]
    cells = [(r, c) for r in range(0, H, CHIP) for c in range(0, W, CHIP)]

    def per_cell(x):
        """Sum over each chip cell: [cell] for a 2-D array, [cell, date] for a 3-D one."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            return x.reshape(H // CHIP, CHIP, W // CHIP, CHIP).sum((1, 3)).ravel()
        return x.reshape(n, H // CHIP, CHIP, W // CHIP, CHIP).sum((2, 4)).reshape(n, -1).T

    def block(sel):
        """The tables for one set of dates (one year, or all dates of the site): the cells with at least two of those
        dates, the pixels usable on every one of them, and each date's area factor and water share built from the cells."""
        present = np.array([[bool(covered[i, r, c]) and bool(sel[i]) for i in range(n)] for r, c in cells])
        use = np.zeros((H, W), bool)
        for (r, c), pr in zip(cells, present):
            if pr.sum() >= 2:
                use[r:r + CHIP, c:c + CHIP] = np.all(valid[pr][:, r:r + CHIP, c:c + CHIP], axis=0)
        cov = covered & use[None] & sel[:, None, None]
        m, w = mask & cov, water & cov
        npx = per_cell(use)
        P = present & (present.sum(1) >= 2)[:, None] & (npx > 0)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(g=ipf(per_cell(m), P), water=additive(100 * per_cell(w) / npx[:, None], P, npx), use=use, cov=cov, m=m, w=w)

    A = block(np.ones(n, bool))                                                     # all years together (older version)
    Y = {int(y): block(years == y) for y in sorted(set(years)) if (years == y).sum() >= 2}      # one set of tables per year
    for i, d in enumerate(dates):
        row = {"site": site, "date": d, "dev_all_years_pct": round(100 * (A["g"][i] - 1), 2), "water_all_years_pct": round(A["water"][i], 2)}
        if int(years[i]) in Y:
            b = Y[int(years[i])]
            row.update(dev_pct=round(100 * (b["g"][i] - 1), 2), water_pct=round(b["water"][i], 2))
        rows.append(row)

    # the strip that is water on some dates of a year and land on others, and the pixels whose prediction flips in that year
    for y in Y:
        sel = years == y
        k = A["cov"][sel].sum(0)
        ok = A["use"] & (k >= 2)
        ws, ms = A["w"][sel].sum(0), A["m"][sel].sum(0)
        flip = ok & (ms > 0) & (ms < k)
        zone = ndimage.binary_dilation(ok & (ws > 0) & (ws < k), K3) & ok          # in the strip or next to it
        strip += [ok.sum(), flip.sum(), zone.sum(), (flip & zone).sum()]
    print(f"{site}: {n} scenes, same-year tables for {sorted(Y)}", flush=True)

T = scenes.merge(pd.DataFrame(rows), on=["site", "date"])
T["november"] = (pd.to_datetime(T["date"]).dt.month == 11).astype(float)
T["site_year"] = T["site"] + "_" + T["year"].astype(str)
T.to_csv(work / "tide_scene_table.csv", index=False)

have = T.dropna(subset=["dev_pct"])
no_may = T[T["date"] != "2020-05-18"]
SWING = "share of the same-year swing that goes with "
summary = [
    {"quantity": "scenes with a same-year swing", "value": len(have), "unit": "scenes"},
    {"quantity": "sites and years they are in", "value": have["site_year"].nunique(), "unit": "site-years"},
    {"quantity": "different dates", "value": have["date"].nunique(), "unit": "dates"},
]


def add(quantity, res, value="share_pct", unit="%"):
    summary.append({"quantity": quantity, "value": res[value], "unit": unit, "scenes": res["scenes"],
                    "chance_shuffled_dates_pct": res["chance_shuffled_dates_pct"]})


add(SWING + "tide height (EOT20 model)", link(T, "tide_eot20_m", "dev_pct"))
add(SWING + "the water share seen in the image", link(T, "water_pct", "dev_pct"))
add("water share in the image against tide height, correlation", link(T, "tide_eot20_m", "water_pct", perm=False), "r", "r")
add(SWING + "November or not", link(T, "november", "dev_pct"))
add(SWING + "the height of the sun", link(T, "sun_elevation", "dev_pct"))
add(SWING + "the height of the sun, without the 18 May 2020 pass", link(no_may, "sun_elevation", "dev_pct", perm=False))
add("November or not against tide height, correlation", link(T, "november", "tide_eot20_m", perm=False), "r", "r")
summary += [
    {"quantity": "pixels that flip between dates of the same year: share in or next to the strip that is sometimes water",
     "value": round(100 * strip[3] / strip[1], 1), "unit": "%"},
    {"quantity": "all pixels: share in or next to that strip (what chance would give)", "value": round(100 * strip[2] / strip[0], 1), "unit": "%"},
]
add("all years (older version): share of the swing that goes with tide height", link(T, "tide_eot20_m", "dev_all_years_pct", by="site", perm=False))
add("all years (older version): share of the swing that goes with the water share", link(T, "water_all_years_pct", "dev_all_years_pct", by="site", perm=False))
R = pd.DataFrame(summary)
R["scenes"] = R["scenes"].astype("Int64")
R.to_csv(out / "summary.csv", index=False)
with pd.option_context("display.width", 250, "display.max_colwidth", 120):
    print(R.to_string(index=False))
