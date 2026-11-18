"""New bare or built ground near the mangroves, 2018 to 2025.

No trained model. One fixed rule on three band values, the same for every site and date:
    bare or built ground  =  NDVI < 0.15  and  MNDWI < 0  and  B11 (SWIR1 reflectance) > 0.20
that is: not green, not water, and bright in the shortwave infrared (dry sand, soil, concrete, roads).
Sand and concrete look alike in these bands, so the rule cannot tell construction from bare ground, and
building on ground that was already bare does not show.

    new bare or built ground (date A -> date B)  =  usable on both dates, NOT bare on A, bare on B

Change pairs (same month only): November 2018 -> November 2025, March 2018 -> March 2025, and "both months
agree": pixels that are not bare on either 2018 date and bare on both 2025 dates.
Noise pairs (same year, the answer should be near zero): the first and the last non-hazy date of 2018, of
2020 and of 2025. Each is also run backwards.

Mangrove outline: the GMW v4.1.12 label of 2018 stored in the 2018 chips. Distances are from that outline.
With --model-maps, the model's change map (change_map.npz written by change_map.py) is used to ask whether
its loss pixels are closer to new bare ground than its stable mangrove pixels are.

Outputs in --out: pairs.csv, rings.csv, model_loss_distance.csv and new_bare_ground_<site>.png.

Usage (from src/change_detection):
  python new_bare_ground.py --train-val <Train_Val_Dataset> --out <output folder> \
      [--model-maps <output folder of change_map.py>] [--sites ghantoot jubail_island]
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import CHIP, add_path_args, chip_path, load_chip, load_full_catalog

NDVI_MAX, MNDWI_MAX, B11_MIN = 0.15, 0.0, 0.20
RINGS_M = [100, 300, 500]
PX_M, PX_HA = 10.0, 0.01


def ha(n):
    return round(float(n) * PX_HA, 1)


def pct(a, b, nd=2):
    return round(100 * float(a) / float(b), nd) if b else float("nan")


def nice(d):
    x = date.fromisoformat(d)
    return f"{x.day} {x:%b %Y}"


def short(d):
    return f"{date.fromisoformat(d):%b %Y}"


def scene(rows, shape, r0, c0, tv) -> dict:
    """Site-sized arrays for one date, with every usable pixel put in one of four classes."""
    out = {"valid": np.zeros(shape, bool), "ndvi": np.full(shape, np.nan, np.float32),
           "mndwi": np.full(shape, np.nan, np.float32), "b11": np.full(shape, np.nan, np.float32),
           "img": np.full((3,) + shape, np.nan, np.float32), "l12": np.full(shape, 255, np.uint8)}
    for _, r in rows.iterrows():
        c = load_chip(chip_path(tv, r))
        sl = (slice(r["row"] - r0, r["row"] - r0 + CHIP), slice(r["col"] - c0, r["col"] - c0 + CHIP))
        out["valid"][sl] = c["valid"]
        out["ndvi"][sl] = c["image"][6]; out["mndwi"][sl] = c["image"][7]; out["b11"][sl] = c["image"][4]
        out["img"][(slice(None),) + sl] = c["image"][[3, 2, 1]]
        out["l12"][sl] = c["label_v4112"]
    v = out["valid"]
    with np.errstate(invalid="ignore"):
        out["water"] = v & (out["mndwi"] >= MNDWI_MAX)
        out["veg"] = v & (out["mndwi"] < MNDWI_MAX) & (out["ndvi"] >= NDVI_MAX)
        out["bare"] = v & (out["ndvi"] < NDVI_MAX) & (out["mndwi"] < MNDWI_MAX) & (out["b11"] > B11_MIN)
    out["dark"] = v & ~out["water"] & ~out["veg"] & ~out["bare"]          # not water, not green, not bright: wet or dark ground
    return out


def pair(a, b) -> dict:
    """Masks for the change from scene a to scene b."""
    use = a["valid"] & b["valid"]
    new = use & ~a["bare"] & b["bare"]
    return {"use": use, "new": new, "lost": use & a["bare"] & ~b["bare"], "bare_first": use & a["bare"],
            "new_from_veg": new & a["veg"], "new_from_water": new & a["water"], "new_from_dark": new & a["dark"]}


def month_date(dates, year, month):
    d = [x for x in dates if x.startswith(f"{year}-{month}")]
    if not d:
        raise SystemExit(f"no non-hazy scene in {year}-{month}")
    return d[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("--sites", nargs="+", default=["ghantoot", "jubail_island"])
    ap.add_argument("--model-maps", default=None, help="folder with <site>/change_map.npz from change_map.py (optional)")
    ap.add_argument("--dpi", type=int, default=95, help="figure resolution")
    a = ap.parse_args()
    tv, out = Path(a.train_val), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = load_full_catalog(tv, a.qa)
    pairs_rows, ring_rows, dist_rows = [], [], []
    for site in a.sites:
        rows = cat[(cat["site"] == site) & ~cat["hazy"]]
        r0, c0 = int(cat.loc[cat["site"] == site, "row"].min()), int(cat.loc[cat["site"] == site, "col"].min())
        shape = (int(cat.loc[cat["site"] == site, "row"].max()) - r0 + CHIP, int(cat.loc[cat["site"] == site, "col"].max()) - c0 + CHIP)
        dates = sorted(rows["date"].unique())
        n18, n25 = month_date(dates, 2018, "11"), month_date(dates, 2025, "11")
        m18, m25 = month_date(dates, 2018, "03"), month_date(dates, 2025, "03")
        same_year = [(min(d), max(d)) for y in (2018, 2025, 2020) for d in [[x for x in dates if x.startswith(str(y))]] if len(d) > 1]
        need = sorted({n18, n25, m18, m25} | {d for p in same_year for d in p})
        S = {d: scene(rows[rows["date"] == d], shape, r0, c0, tv) for d in need}

        # ---- the 2018 mangrove outline (GMW v4.1.12, 2018) and the distance of every pixel from it
        lab = np.where(S[n18]["l12"] != 255, S[n18]["l12"], S[m18]["l12"])
        mang = lab == 1
        dist_m = ndi.distance_transform_edt(~mang) * PX_M
        zones = {"inside the 2018 mangrove": mang}
        for r in RINGS_M:
            zones[f"within {r} m of it"] = ~mang & (dist_m <= r)
        zones["further than 500 m"] = ~mang & (dist_m > RINGS_M[-1])

        nov, mar = pair(S[n18], S[n25]), pair(S[m18], S[m25])
        P = {f"{short(n18)} -> {short(n25)}": ("change", nov), f"{short(m18)} -> {short(m25)}": ("change", mar)}
        both_use = nov["use"] & mar["use"]
        any18 = S[n18]["bare"] | S[m18]["bare"]
        both_new = both_use & ~any18 & S[n25]["bare"] & S[m25]["bare"]
        veg18, water18 = S[n18]["veg"] & S[m18]["veg"], S[n18]["water"] & S[m18]["water"]
        P["both months agree"] = ("change", {
            "use": both_use, "new": both_new, "bare_first": both_use & any18,
            "lost": both_use & S[n18]["bare"] & S[m18]["bare"] & ~S[n25]["bare"] & ~S[m25]["bare"],
            "new_from_veg": both_new & veg18, "new_from_water": both_new & water18, "new_from_dark": both_new & ~veg18 & ~water18})
        for d0, d1 in same_year:
            P[f"{short(d0)} -> {short(d1)}"] = ("same year", pair(S[d0], S[d1]))
            P[f"{short(d1)} -> {short(d0)} (backwards)"] = ("same year", pair(S[d1], S[d0]))

        for name, (kind, p) in P.items():
            n_use = p["use"].sum()
            pairs_rows.append({"site": site, "pair": name, "kind": kind, "chips_compared": int(round(n_use / CHIP ** 2)),
                               "compared_ha": ha(n_use), "bare_at_first_date_ha": ha(p["bare_first"].sum()),
                               "new_bare_ha": ha(p["new"].sum()), "new_bare_pct_of_area": pct(p["new"].sum(), n_use),
                               "lost_bare_ha": ha(p["lost"].sum()), "net_bare_ha": ha(int(p["new"].sum()) - int(p["lost"].sum())),
                               "new_bare_was_vegetation_ha": ha(p["new_from_veg"].sum()),
                               "new_bare_was_water_ha": ha(p["new_from_water"].sum()),
                               "new_bare_was_dark_or_wet_ground_ha": ha(p["new_from_dark"].sum())})
            for zname, z in zones.items():
                zu = z & p["use"]
                ring_rows.append({"site": site, "pair": name, "kind": kind, "zone": zname, "zone_ha": ha(zu.sum()),
                                  "new_bare_ha": ha((p["new"] & z).sum()), "new_bare_pct_of_zone": pct((p["new"] & z).sum(), zu.sum()),
                                  "of_which_was_vegetation_ha": ha((p["new_from_veg"] & z).sum()),
                                  "of_which_was_water_ha": ha((p["new_from_water"] & z).sum())})

        # ---- are the model's loss pixels closer to new bare ground than its stable mangrove pixels are?
        npz = Path(a.model_maps) / site / "change_map.npz" if a.model_maps else None
        if npz and npz.exists():
            with np.load(npz) as z:
                change = z["change"]
            assert change.shape == shape, "the change map is not on this site's chip grid"
            classes = {"model loss": change == 3, "model stable mangrove": change == 1, "model gain": change == 2}
            for name, (kind, p) in P.items():
                if "backwards" in name or not p["new"].any():
                    continue
                d_new = ndi.distance_transform_edt(~p["new"]) * PX_M
                for cname, cm in classes.items():
                    m = cm & p["use"]
                    if m.any():
                        d = d_new[m]
                        dist_rows.append({"site": site, "new_bare_from": name, "kind": kind, "pixels": cname, "ha": ha(m.sum()),
                                          "is_new_bare_itself_pct": pct((d == 0).sum(), m.sum(), 1),
                                          "within_100m_pct": pct((d <= 100).sum(), m.sum(), 1),
                                          "within_300m_pct": pct((d <= 300).sum(), m.sum(), 1),
                                          "median_distance_m": round(float(np.median(d)), 0)})
        noise = [p for kind, p in P.values() if kind == "same year"]
        figure(site, S[n18], S[n25], nov, noise, mang, (n18, n25), out / f"new_bare_ground_{site}.png", a.dpi)

    pairs = pd.DataFrame(pairs_rows); rings = pd.DataFrame(ring_rows); dist = pd.DataFrame(dist_rows)
    pairs.to_csv(out / "pairs.csv", index=False); rings.to_csv(out / "rings.csv", index=False)
    dist.to_csv(out / "model_loss_distance.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print("\nPAIRS\n" + pairs.to_string(index=False))
        print("\nRINGS\n" + rings.to_string(index=False))
        if len(dist):
            print("\nMODEL LOSS vs STABLE MANGROVE: distance to new bare ground\n" + dist.to_string(index=False))


# ------------------------------------------------------------------ figure

AQUA, BLUE, ORANGE, GREY, WHITE = "#1baf7a", "#2a78d6", "#eb6834", "#e9e8e4", "#ffffff"
INK, INK2 = "#0b0b0b", "#52514e"


def false_colour(img):
    rgb = np.clip(np.moveaxis(img, 0, -1) / 0.40, 0, 1) ** 0.7
    rgb[~np.isfinite(rgb).all(axis=-1)] = 1.0
    return rgb


def figure(site, s18, s25, p, noise, mang, dates, path, dpi=95):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    H, W = mang.shape
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = to_rgb(GREY)
    rgb[mang] = to_rgb(AQUA)
    rgb[p["new"] & ~mang] = to_rgb(ORANGE)
    rgb[p["new"] & mang] = to_rgb(BLUE)
    rgb[~p["use"]] = to_rgb(WHITE)

    fig_w, side, gap = 19.5, 0.012, 0.014                       # inches, then fractions of the width
    pw = (1 - 2 * side - 2 * gap) / 3
    panel_h = pw * fig_w * H / W                                # inches, keeps pixels square
    top, bottom = 0.85, 1.45
    fig_h = panel_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    axes = [fig.add_axes([side + i * (pw + gap), bottom / fig_h, pw, panel_h / fig_h]) for i in range(3)]
    for ax, (im, title) in zip(axes, [(false_colour(s18["img"]), f"{nice(dates[0])}, Sentinel-2 false colour"),
                                      (false_colour(s25["img"]), f"{nice(dates[1])}, Sentinel-2 false colour"),
                                      (rgb, "New bare or built ground, November 2018 to November 2025")]):
        ax.imshow(im, interpolation="antialiased")
        ax.set_title(title, fontsize=13, color=INK, loc="left", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c3c2b7"); sp.set_linewidth(0.8)
        ax.add_patch(Rectangle((0.03 * W, H - 0.045 * H), 200, 0.008 * H, facecolor=INK, edgecolor="white", linewidth=0.8))
        ax.text(0.03 * W + 100, H - 0.055 * H, "2 km", ha="center", va="bottom", fontsize=9, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    handles = [Patch(facecolor=AQUA, label=f"2018 mangrove (GMW v4.1.12 label)  {ha((mang & p['use']).sum()):,.0f} ha"),
               Patch(facecolor=ORANGE, label=f"New bare or built ground  {ha((p['new'] & ~mang).sum()):,.0f} ha"),
               Patch(facecolor=BLUE, label=f"New bare or built ground inside the 2018 mangrove  {ha((p['new'] & mang).sum()):,.1f} ha"),
               Patch(facecolor=WHITE, edgecolor="#c3c2b7", label="No chip or no usable pixel")]
    y_leg = (bottom - 0.12) / fig_h
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1 - side, y_leg), ncol=2, frameon=False,
               fontsize=10.5, labelcolor=INK, handlelength=1.2, columnspacing=2.0)
    fig.text(side, y_leg - 0.02 / fig_h, "Near-infrared shown as red, so vegetation is red. Same colour stretch in both years.",
             ha="left", va="top", fontsize=10, color=INK2)
    fig.text(side, 1 - 0.14 / fig_h, f"{site.replace('_', ' ').title()}: land that turned into bare or built ground near the mangroves",
             ha="left", va="top", fontsize=15, color=INK)
    shares = [pct(q["new"].sum(), q["use"].sum()) for q in noise]
    share = pct(p["new"].sum(), p["use"].sum())
    note = ("Rule, no trained model: bare or built ground = NDVI below 0.15, MNDWI below 0 and SWIR1 (B11) above 0.20. "
            "New = not bare in 2018, bare in 2025. Sand and concrete look the same to this rule, "
            "and building on ground that was already bare does not show." + chr(10) +
            f"New bare or built ground is {share}% of the area compared.")
    if shares:
        note += (f" The same rule on two dates of one year gives {min(shares)}% to {max(shares)}%. "
                 + ("This is inside that range, so it is not above the noise." if share <= max(shares) else
                    "This is above that range, but only part of it is real change."))
    fig.text(side, 0.10 / fig_h, note, linespacing=1.5, ha="left", va="bottom", fontsize=10, color=INK2)
    fig.savefig(path, facecolor="white")
    print("figure ->", path)


if __name__ == "__main__":
    main()
