"""Predict one site's 2018 and 2025 mangrove masks, build the change map, run the sanity checks, draw the figure.

Needs the forest saved by train_rf.py for the same site (the site was held out of its training).

How a year's mask is made
  Every non-hazy scene of the year is predicted with the forest (probability > threshold = mangrove).
  Per pixel, the scenes that have a usable value there vote. More than half = mangrove.
  With two scenes a 1-1 tie is possible; ties are broken by the mean probability of the scenes
  against the same threshold. Pixels with one usable scene take that scene's answer.

Sanity checks, because model errors look like change
  noise_floor.csv        two dates of the same year compared as if they were a change map
  sensitivity.csv        the 2018 to 2025 change made in other reasonable ways
  summary.csv            class areas, how much change sits on mangrove edges, model area against the labels
  label_gain_pixels.csv  what the pixels look like where the GMW v4.1.12 label says gain

Outputs in <out>/<site>/: the four CSVs, stats.json, change_map_figure.png, and change_map.npz / .tif
(the map itself; rasters are not for the repository).

Usage: python change_map.py --train-val <Train_Val_Dataset> --out <output folder> [--site jubail_island] [--variant b]
"""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import B2, CHIP, add_path_args, chip_path, counts, ha, load_chip, load_full_catalog

S8 = np.ones((3, 3), bool)      # "within 1 pixel" = the 8 neighbours
CLASSES = "0 stable non-mangrove, 1 stable mangrove, 2 gain, 3 loss, 255 no data"


# ------------------------------------------------------------------ scenes and masks

def build_scene(rows, rf, th, shape, r0, c0, tv):
    """Site-sized arrays for one scene: probability, usable mask, labels, NDVI, three bands for display."""
    pos = list(rf.classes_).index(True)
    prob = np.full(shape, np.nan, np.float32)
    valid = np.zeros(shape, bool)
    covered = np.zeros(shape, bool)
    l12 = np.full(shape, 255, np.uint8)
    l19 = np.full(shape, 255, np.uint8)
    img = np.full((3,) + shape, np.nan, np.float32)      # B08, B04, B03
    ndvi = np.full(shape, np.nan, np.float32)
    for _, r in rows.iterrows():
        chip = load_chip(chip_path(tv, r))
        v = chip["valid"]
        p = np.full(v.shape, np.nan, np.float32)
        if v.any():
            p[v] = rf.predict_proba(B2.pixels(chip)[v.ravel()])[:, pos]
        sl = (slice(r["row"] - r0, r["row"] - r0 + CHIP), slice(r["col"] - c0, r["col"] - c0 + CHIP))
        prob[sl] = p; valid[sl] = v; covered[sl] = True
        l12[sl] = chip["label_v4112"]; l19[sl] = chip["label_v4019"]
        img[(slice(None),) + sl] = chip["image"][[3, 2, 1]]
        ndvi[sl] = chip["image"][6]
    return {"prob": prob, "valid": valid, "covered": covered, "l12": l12, "l19": l19, "img": img, "ndvi": ndvi,
            "mask": valid & (np.nan_to_num(prob, nan=0.0) > th), "chips": len(rows)}


def site_scenes(cat: pd.DataFrame, site: str, rf, th, tv):
    """All non-hazy scenes of a site on one grid. Returns (scenes by date, hazy dates skipped, shape, x_left, y_top)."""
    rows = cat[cat["site"] == site]
    r0, c0 = int(rows["row"].min()), int(rows["col"].min())
    shape = (int(rows["row"].max()) - r0 + CHIP, int(rows["col"].max()) - c0 + CHIP)
    scenes = {d: build_scene(g, rf, th, shape, r0, c0, tv) for d, g in rows.groupby("date") if not bool(g["hazy"].iloc[0])}
    skipped = sorted(set(rows["date"]) - set(scenes))
    return scenes, skipped, shape, float(rows["x0"].min()), float(rows["y1"].max())


def composite(scenes, th, tie="mean_prob"):
    """Per-pixel majority vote over scenes. Returns (mask, usable, number of tied pixels)."""
    n = sum(s["valid"].astype(np.int16) for s in scenes)
    votes = sum(s["mask"].astype(np.int16) for s in scenes)
    psum = sum(np.where(s["valid"], np.nan_to_num(s["prob"], nan=0.0), 0.0) for s in scenes)
    usable = n > 0
    mask = votes * 2 > n
    is_tie = usable & (votes * 2 == n)
    if tie == "mean_prob":
        mask |= is_tie & ((psum / np.maximum(n, 1)) > th)
    elif tie == "mangrove":
        mask |= is_tie
    elif tie != "not_mangrove":
        raise ValueError(tie)
    return mask & usable, usable, int(is_tie.sum())


def change_stats(m0, m1, use):
    """Areas in hectares over the pixels usable in both masks."""
    sm = int((m0 & m1 & use).sum()); gain = int((~m0 & m1 & use).sum())
    loss = int((m0 & ~m1 & use).sum()); sn = int((~m0 & ~m1 & use).sum())
    a0, a1 = sm + loss, sm + gain
    return {"compared_ha": ha(use.sum()), "stable_mangrove_ha": ha(sm), "gain_ha": ha(gain), "loss_ha": ha(loss),
            "stable_non_mangrove_ha": ha(sn), "area_first_ha": ha(a0), "area_second_ha": ha(a1),
            "net_ha": ha(gain - loss), "net_pct_of_first": round(100 * (gain - loss) / a0, 1) if a0 else None,
            "changed_ha": ha(gain + loss), "changed_pct_of_first": round(100 * (gain + loss) / a0, 1) if a0 else None}


def pair_stats(scenes, d0, d1):
    """Two single-date masks compared on the pixels usable on both dates."""
    u = scenes[d0]["valid"] & scenes[d1]["valid"]
    chips = int(round((scenes[d0]["covered"] & scenes[d1]["covered"]).sum() / CHIP ** 2))
    return {"first_date": d0, "second_date": d1, "chips_compared": chips,
            **change_stats(scenes[d0]["mask"], scenes[d1]["mask"], u)}


def best_shift(a, b, use, max_shift=3):
    """Shift of image b relative to a (pixels) that gives the highest correlation, with a parabola fit
    around the best whole-pixel shift for a sub-pixel estimate. A rough check only."""
    H, W = a.shape
    m = max_shift
    cc = np.full((2 * m + 1, 2 * m + 1), np.nan)
    core = (slice(m, H - m), slice(m, W - m))
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            sl = (slice(m + dy, H - m + dy), slice(m + dx, W - m + dx))
            ok = use[core] & use[sl]
            if ok.sum() > 1000:
                cc[dy + m, dx + m] = np.corrcoef(a[core][ok], b[sl][ok])[0, 1]
    if not np.isfinite(cc).any():
        return None
    iy, ix = np.unravel_index(np.nanargmax(cc), cc.shape)

    def sub(v_m, v_0, v_p):
        d = v_m - 2 * v_0 + v_p
        return 0.0 if not np.isfinite(d) or d == 0 else 0.5 * (v_m - v_p) / d
    sy = sub(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix]) if 0 < iy < 2 * m else 0.0
    sx = sub(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1]) if 0 < ix < 2 * m else 0.0
    return {"best_whole_pixel_shift_dy_dx": [int(iy - m), int(ix - m)],
            "subpixel_shift_dy_dx": [round(float(iy - m + sy), 2), round(float(ix - m + sx), 2)],
            "corr_at_best": round(float(cc[iy, ix]), 4), "corr_at_zero_shift": round(float(cc[m, m]), 4)}


def flatten(d, prefix=""):
    """Nested stats -> (key, value) rows. Lists of tables are left out (they have their own CSV)."""
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows += flatten(v, key + ".")
        elif isinstance(v, list):
            if not any(isinstance(x, dict) for x in v):
                rows.append((key, " ".join(str(x) for x in v)))
        else:
            rows.append((key, v))
    return rows


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--site", default="jubail_island")
    ap.add_argument("--variant", default="b", choices=["a", "b"])
    ap.add_argument("--model", default=None, help="rf.joblib from train_rf.py (default: <out>/<site>/model_<variant>/rf.joblib)")
    ap.add_argument("--dpi", type=int, default=150, help="figure resolution")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    tv = Path(a.train_val)
    out = Path(a.out) / a.site
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(a.model or out / f"model_{a.variant}" / "rf.joblib")
    if bundle.get("site") not in (None, a.site):
        print(f"! the model was trained with {bundle['site']} held out; make sure {a.site} is in the same fold")
    rf, th = bundle["rf"], bundle["threshold"]
    rf.set_params(n_jobs=-1)
    tag = "" if a.variant == "b" else f"_variant_{a.variant}"

    cat = load_full_catalog(tv, a.qa)
    scenes, skipped, shape, x_left, y_top = site_scenes(cat, a.site, rf, th, tv)
    for d in sorted(scenes):
        print(f"predicted {d}: {scenes[d]['chips']} chips, {ha(scenes[d]['mask'].sum())} ha mangrove", flush=True)
    by_year = {y: sorted(d for d in scenes if d.startswith(str(y))) for y in (2018, 2020, 2025)}
    if not by_year[2018] or not by_year[2025]:
        raise SystemExit(f"{a.site} has no usable scene in 2018 or in 2025")

    stats = {"site": a.site, "model": f"B2 random forest, variant {a.variant}, {a.site} held out",
             "threshold": th, "settings": bundle["settings"], "site_grid_px": list(shape),
             "site_area_ha": ha(shape[0] * shape[1]), "hazy_scenes_skipped": skipped,
             "scenes_used": {str(y): by_year[y] for y in by_year},
             "chips_per_scene": {d: scenes[d]["chips"] for d in sorted(scenes)}}

    # ---- one mask per year
    comp = {}
    for y in by_year:
        if not by_year[y]:
            continue
        m, u, nt = composite([scenes[d] for d in by_year[y]], th)
        comp[y] = {"mask": m, "usable": u}
        n = sum(scenes[d]["valid"].astype(int) for d in by_year[y])
        stats[f"mask_{y}"] = {"mangrove_ha": ha(m.sum()), "usable_ha": ha(u.sum()),
                              "px_with_2plus_scenes_ha": ha((n >= 2).sum()), "px_with_1_scene_ha": ha((n == 1).sum()),
                              "px_with_no_scene_ha": ha((n == 0).sum()), "tied_px_ha": ha(nt)}

    # ---- change map 2018 -> 2025
    m18, m25 = comp[2018]["mask"], comp[2025]["mask"]
    use = comp[2018]["usable"] & comp[2025]["usable"]
    change = np.full(shape, 255, np.uint8)
    change[use & ~m18 & ~m25] = 0
    change[use & m18 & m25] = 1
    change[use & ~m18 & m25] = 2
    change[use & m18 & ~m25] = 3
    gain, loss = change == 2, change == 3
    stats["change_2018_2025"] = change_stats(m18, m25, use)
    stats["change_2018_2025"]["no_data_ha"] = ha((~use).sum())
    stats["change_2018_2025"]["chips_compared"] = int(round(use.sum() / CHIP ** 2))
    print("change:", stats["change_2018_2025"], flush=True)

    # ---- how much the answer depends on the choices made
    sens = []

    def add(name, m0, m1, u):
        if u.any():
            sens.append({"version": name, "chips_compared": int(round(u.sum() / CHIP ** 2)), **change_stats(m0, m1, u)})

    def year_masks(tie):
        x0, u0, _ = composite([scenes[d] for d in by_year[2018]], th, tie)
        x1, u1, _ = composite([scenes[d] for d in by_year[2025]], th, tie)
        return x0, x1, u0 & u1

    add("headline: all non-hazy dates, vote, ties by mean probability", m18, m25, use)
    and18, and25, _ = year_masks("not_mangrove")
    or18, or25, _ = year_masks("mangrove")
    add("ties count as not mangrove", and18, and25, use)
    add("ties count as mangrove", or18, or25, use)
    for d0, d1 in itertools.product(by_year[2018], by_year[2025]):
        if d0[5:7] == d1[5:7]:                                  # same month in both years
            add(f"single dates, same month: {d0} vs {d1}", scenes[d0]["mask"], scenes[d1]["mask"],
                scenes[d0]["valid"] & scenes[d1]["valid"])
    n18 = sum(scenes[d]["valid"].astype(int) for d in by_year[2018])
    n25 = sum(scenes[d]["valid"].astype(int) for d in by_year[2025])
    add("headline masks, only pixels with two dates in both years", m18, m25, (n18 >= 2) & (n25 >= 2))
    pd.DataFrame(sens).to_csv(out / f"sensitivity{tag}.csv", index=False)
    stats["sensitivity"] = sens
    # change that every date agrees on: gain = not mangrove on any 2018 date and mangrove on all 2025 dates
    g_all, l_all = use & ~or18 & and25, use & and18 & ~or25
    stats["change_every_date_agrees"] = {
        "gain_ha": ha(g_all.sum()), "loss_ha": ha(l_all.sum()), "net_ha": ha(int(g_all.sum()) - int(l_all.sum())),
        "share_of_headline_gain_pct": round(100 * (g_all & gain).sum() / max(gain.sum(), 1), 1),
        "share_of_headline_loss_pct": round(100 * (l_all & loss).sum() / max(loss.sum(), 1), 1)}

    # ---- noise floor: "change" between two dates of the same year, single-date masks
    noise = []
    for y in by_year:
        for d0, d1 in itertools.combinations(by_year[y], 2):
            noise.append({"kind": "same year", "year": y, **pair_stats(scenes, d0, d1)})
    for d0, d1 in itertools.product(by_year[2018], by_year[2025]):
        noise.append({"kind": "2018 vs 2025", "year": "", **pair_stats(scenes, d0, d1)})
    noise = [r for r in noise if r["chips_compared"] > 0]
    pd.DataFrame(noise).to_csv(out / f"noise_floor{tag}.csv", index=False)
    stats["noise_floor"] = noise
    same = [r["net_pct_of_first"] for r in noise if r["kind"] == "same year" and r["net_pct_of_first"] is not None]
    stats["same_year_net_pct_range"] = [min(same), max(same)] if same else []

    # ---- how much of the gain and loss sits within 1 pixel of a mangrove edge
    def edges(m):
        inner = m & ~ndi.binary_erosion(m, S8, border_value=1)      # mangrove pixels touching non-mangrove
        outer = ndi.binary_dilation(m, S8) & ~m                     # non-mangrove pixels touching mangrove
        return inner, outer
    in18, out18 = edges(m18)
    in25, out25 = edges(m25)
    dist18 = ndi.distance_transform_edt(~m18)

    def pct(part, whole):
        return round(100 * part.sum() / max(whole.sum(), 1), 1)
    stats["edge_check"] = {
        "gain_ha": ha(gain.sum()), "loss_ha": ha(loss.sum()),
        "gain_touching_2018_mangrove_pct": pct(gain & out18, gain),
        "loss_on_2018_mangrove_edge_pct": pct(loss & in18, loss),
        "gain_on_edge_of_either_year_pct": pct(gain & (out18 | in25), gain),
        "loss_on_edge_of_either_year_pct": pct(loss & (in18 | out25), loss),
        # for comparison: how common edge pixels are anyway
        "all_2018_mangrove_on_its_edge_pct": pct(in18 & use, m18 & use),
        "all_2025_mangrove_on_its_edge_pct": pct(in25 & use, m25 & use),
        "all_2018_non_mangrove_touching_mangrove_pct": pct(out18 & use, ~m18 & use),
        "gain_not_touching_2018_mangrove_ha": ha((gain & ~out18).sum()),
        "loss_inside_2018_mangrove_ha": ha((loss & ~in18).sum()),
        "gain_by_distance_to_2018_mangrove_ha": {
            "1 px (touching)": ha((gain & (dist18 < 1.5)).sum()),
            "2 to 3 px": ha((gain & (dist18 >= 1.5) & (dist18 < 3.5)).sum()),
            "4 to 10 px": ha((gain & (dist18 >= 3.5) & (dist18 < 10.5)).sum()),
            "more than 10 px": ha((gain & (dist18 >= 10.5)).sum())}}

    # ---- model area against the GMW v4.1.12 label (about 30 m), per year, same pixels
    lab = {}
    for y in comp:
        l12 = np.full(shape, 255, np.uint8)
        same_label = True
        for d in by_year[y]:
            new = scenes[d]["l12"]
            both = (l12 != 255) & (new != 255)
            same_label &= bool((l12[both] == new[both]).all())
            l12 = np.where(l12 == 255, new, l12)
        lab[y] = l12
        u = comp[y]["usable"] & (l12 != 255)
        c = counts(comp[y]["mask"][u], l12[u] == 1)
        stats[f"label_v4112_{y}"] = {"label_same_on_all_dates_of_year": same_label, "compared_ha": ha(u.sum()),
                                     "model_ha": ha(comp[y]["mask"][u].sum()), "label_ha": ha((l12[u] == 1).sum()),
                                     "model_over_label": round(float(comp[y]["mask"][u].sum() / max((l12[u] == 1).sum(), 1)), 3),
                                     "iou": round(c["iou"], 3), "precision": round(c["precision"], 3), "recall": round(c["recall"], 3)}
    ul = use & (lab[2018] != 255) & (lab[2025] != 255)
    stats["label_v4112_change_2018_2025"] = change_stats(lab[2018] == 1, lab[2025] == 1, ul)
    lg, ll = ul & (lab[2018] != 1) & (lab[2025] == 1), ul & (lab[2018] == 1) & (lab[2025] != 1)
    stats["model_vs_label_change"] = {
        "model_gain_that_is_label_gain_pct": pct(gain & lg, gain),
        "model_gain_that_label_calls_mangrove_in_both_years_pct": pct(gain & ul & (lab[2018] == 1) & (lab[2025] == 1), gain),
        "model_gain_that_label_calls_never_mangrove_pct": pct(gain & ul & (lab[2018] != 1) & (lab[2025] != 1), gain),
        "label_gain_that_is_model_gain_pct": pct(gain & lg, lg),
        "label_gain_that_model_calls_mangrove_in_both_years_pct": pct(lg & (change == 1), lg),
        "label_gain_that_model_calls_never_mangrove_pct": pct(lg & (change == 0), lg),
        "model_loss_that_is_label_loss_pct": pct(loss & ll, loss)}

    # the scene shown for each year: the one with the most chips (the later one if equal)
    show = {y: max(by_year[y], key=lambda d: (scenes[d]["chips"], d)) for y in (2018, 2025)}
    stats["scenes_shown_in_figure"] = {str(y): show[y] for y in show}
    # what the pixels look like where the label and the model disagree about gain (on the two scenes shown)
    groups = {"label: mangrove in 2018 and 2025": ul & (lab[2018] == 1) & (lab[2025] == 1),
              "label: gain (not 2018, mangrove 2025)": lg,
              "label: gain, model says never mangrove": lg & (change == 0),
              "label: gain, model also says gain": lg & gain,
              "model: stable mangrove": change == 1,
              "model: gain": gain,
              "label: never mangrove": ul & (lab[2018] != 1) & (lab[2025] != 1)}
    look = []
    for name, g in groups.items():
        row = {"group": name, "ha": ha(g.sum())}
        for y in (2018, 2025):
            gv = g & scenes[show[y]]["valid"]
            row[f"scene_{y}"] = show[y]
            if gv.any():
                row[f"median_ndvi_{y}"] = round(float(np.median(scenes[show[y]]["ndvi"][gv])), 3)
                row[f"median_rf_prob_{y}"] = round(float(np.median(scenes[show[y]]["prob"][gv])), 3)
                row[f"pct_rf_prob_over_0.5_{y}"] = round(100 * float((scenes[show[y]]["prob"][gv] > 0.5).mean()), 1)
        look.append(row)
    pd.DataFrame(look).to_csv(out / f"label_gain_pixels{tag}.csv", index=False)
    stats["label_gain_pixels"] = look

    # the headline again with a 0.5 cut on the probability instead of the tuned threshold (context only)
    def half(d):
        return {**scenes[d], "mask": scenes[d]["valid"] & (np.nan_to_num(scenes[d]["prob"]) > 0.5)}
    h18, hu18, _ = composite([half(d) for d in by_year[2018]], 0.5)
    h25, hu25, _ = composite([half(d) for d in by_year[2025]], 0.5)
    stats["change_if_threshold_were_0.5"] = change_stats(h18, h25, hu18 & hu25)

    # 2020: the 10 m label the model was trained on (at other sites), and the three years as a series
    if 2020 in comp:
        l19 = np.full(shape, 255, np.uint8)
        for d in by_year[2020]:
            l19 = np.where(l19 == 255, scenes[d]["l19"], l19)
        u = comp[2020]["usable"] & (l19 != 255)
        c = counts(comp[2020]["mask"][u], l19[u] == 1)
        stats["label_v4019_2020"] = {"compared_ha": ha(u.sum()), "model_ha": ha(comp[2020]["mask"][u].sum()),
                                     "label_ha": ha((l19[u] == 1).sum()),
                                     "model_over_label": round(float(comp[2020]["mask"][u].sum() / max((l19[u] == 1).sum(), 1)), 3),
                                     "iou": round(c["iou"], 3), "precision": round(c["precision"], 3), "recall": round(c["recall"], 3)}
        u3 = comp[2018]["usable"] & comp[2020]["usable"] & comp[2025]["usable"]
        stats["area_series_same_pixels"] = {"compared_ha": ha(u3.sum()),
                                            **{f"model_{y}_ha": ha((comp[y]["mask"] & u3).sum()) for y in comp},
                                            **{f"label_v4112_{y}_ha": ha(((lab[y] == 1) & u3).sum()) for y in comp}}

    # ---- rough check that the images line up (near-infrared band, whole site)
    reg = {}
    pairs = [(d0, d1) for d0, d1 in itertools.product(by_year[2018], by_year[2025]) if d0[5:7] == d1[5:7]]
    pairs += [p for y in (2018, 2025) for p in itertools.combinations(by_year[y], 2)]
    for d0, d1 in pairs:
        r = best_shift(np.nan_to_num(scenes[d0]["img"][0]), np.nan_to_num(scenes[d1]["img"][0]),
                       scenes[d0]["valid"] & scenes[d1]["valid"])
        if r:
            reg[f"{d0} vs {d1}"] = r
    stats["image_shift_check_nir"] = reg

    (out / f"stats{tag}.json").write_text(json.dumps(stats, indent=1))
    pd.DataFrame(flatten(stats), columns=["item", "value"]).to_csv(out / f"summary{tag}.csv", index=False)
    np.savez_compressed(out / f"change_map{tag}.npz", change=change, mask_2018=m18, mask_2025=m25, usable=use,
                        x_left=x_left, y_top=y_top, pixel_m=10.0, crs="EPSG:32640", classes=CLASSES)
    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(out / f"change_map{tag}.tif", "w", driver="GTiff", height=shape[0], width=shape[1], count=1,
                           dtype="uint8", crs="EPSG:32640", transform=from_origin(x_left, y_top, 10.0, 10.0),
                           nodata=255, compress="deflate") as dst:
            dst.write(change, 1)
            dst.update_tags(classes=CLASSES)
    except Exception as e:                                   # the GeoTIFF is a convenience; the .npz is the record
        print("GeoTIFF not written:", e)

    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(pd.DataFrame(flatten({k: v for k, v in stats.items() if k not in ("chips_per_scene",)}),
                           columns=["item", "value"]).to_string(index=False))
        print(pd.DataFrame(sens)[["version", "chips_compared", "gain_ha", "loss_ha", "net_ha", "net_pct_of_first"]].to_string(index=False))
        print(pd.DataFrame(noise)[["kind", "first_date", "second_date", "chips_compared", "gain_ha", "loss_ha", "net_ha",
                                   "net_pct_of_first", "changed_pct_of_first"]].to_string(index=False))
    if not a.no_figure:
        figure(scenes, show, by_year, change, stats, out / f"change_map_figure{tag}.png", a.dpi)


# ------------------------------------------------------------------ figure

COLORS = {0: "#e9e8e4", 1: "#1baf7a", 2: "#2a78d6", 3: "#eb6834", 255: "#ffffff"}
INK, INK2 = "#0b0b0b", "#52514e"


def nice_date(d: str) -> str:
    x = date.fromisoformat(d)
    return f"{x.day} {x:%b %Y}"


def false_colour(img):
    """Near-infrared, red, green shown as red, green, blue. Same fixed stretch for both years."""
    rgb = np.clip(np.moveaxis(img, 0, -1) / 0.40, 0, 1) ** 0.7
    rgb[~np.isfinite(rgb).all(axis=-1)] = 1.0
    return rgb


def figure(scenes, show, by_year, change, stats, path, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    s = stats["change_2018_2025"]
    H, W = change.shape
    rgb = np.zeros(change.shape + (3,), np.float32)
    for k, hexc in COLORS.items():
        rgb[change == k] = matplotlib.colors.to_rgb(hexc)

    fig_w, side, gap = 19.5, 0.012, 0.014                       # inches, then fractions of the width
    pw = (1 - 2 * side - 2 * gap) / 3
    panel_h = pw * fig_w * H / W                                # inches, keeps pixels square
    top, bottom = 0.85, 1.35                                    # inches for the titles and for legend + notes
    fig_h = panel_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    axes = [fig.add_axes([side + i * (pw + gap), bottom / fig_h, pw, panel_h / fig_h]) for i in range(3)]
    panels = [(false_colour(scenes[show[2018]]["img"]), f"{nice_date(show[2018])}, Sentinel-2 false colour"),
              (false_colour(scenes[show[2025]]["img"]), f"{nice_date(show[2025])}, Sentinel-2 false colour"),
              (rgb, "Model change map, 2018 to 2025")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, interpolation="antialiased")
        ax.set_title(title, fontsize=13, color=INK, loc="left", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c3c2b7"); sp.set_linewidth(0.8)
        ax.add_patch(Rectangle((0.03 * W, H - 0.045 * H), 200, 0.008 * H, facecolor=INK, edgecolor="white", linewidth=0.8))
        ax.text(0.03 * W + 100, H - 0.055 * H, "2 km", ha="center", va="bottom", fontsize=9, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    handles = [Patch(facecolor=COLORS[1], label=f"Stable mangrove  {s['stable_mangrove_ha']:,.0f} ha"),
               Patch(facecolor=COLORS[2], label=f"Gain  {s['gain_ha']:,.0f} ha"),
               Patch(facecolor=COLORS[3], label=f"Loss  {s['loss_ha']:,.0f} ha"),
               Patch(facecolor=COLORS[0], edgecolor="#c3c2b7", label=f"Stable non-mangrove  {s['stable_non_mangrove_ha']:,.0f} ha"),
               Patch(facecolor=COLORS[255], edgecolor="#c3c2b7", label="No chip or no usable pixel")]
    y_leg = (bottom - 0.12) / fig_h
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - side, y_leg), ncol=3, frameon=False,
               fontsize=10.5, labelcolor=INK, handlelength=1.2, columnspacing=2.0)
    fig.text(side, y_leg - 0.02 / fig_h, "Near-infrared shown as red, so vegetation is red. Same colour stretch in both years.",
             ha="left", va="top", fontsize=10, color=INK2)
    name = stats["site"].replace("_", " ").title()
    fig.text(side, 1 - 0.14 / fig_h, f"{name}: mangrove change predicted by the random forest baseline (site held out of training)",
             ha="left", va="top", fontsize=15, color=INK)
    used = ", ".join(f"{nice_date(d)} ({scenes[d]['chips']} chips)" for y in (2018, 2025) for d in by_year[y])
    lo_hi = stats.get("same_year_net_pct_range") or []
    note = (f"Net change {s['net_ha']:+,.0f} ha ({s['net_pct_of_first']:+.1f}% of the 2018 model area). "
            f"Each year's mask is a per-pixel vote over that year's scenes: {used}.")
    if lo_hi:
        note += (chr(10) + f"Same-year check at this site: the net \"change\" between two dates of one year ranges from "
                 f"{lo_hi[0]:+.1f}% to {lo_hi[1]:+.1f}%. A real change has to be clearly larger than that. See RESULTS.md.")
    fig.text(side, 0.10 / fig_h, note, linespacing=1.5, ha="left", va="bottom", fontsize=10, color=INK2)
    fig.savefig(path, facecolor="white")
    print("figure ->", path)


if __name__ == "__main__":
    main()
