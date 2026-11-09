#!/usr/bin/env python3
"""Stage 2: raw site/date GeoTIFFs (from collect.py) -> training chips.

Usage:
    python prepare.py --manifest data/raw_manifest.csv --sites sites/sites.geojson \
        --labels labels/ --out data/

Writes:
    <out>/chips/<chip_id>.npz   one chip (see CHIP FORMAT below)
    <out>/catalog.csv           one row per kept chip
    <out>/norm.json             per-channel mean/std from TRAIN chips only
    <out>/qa/<site>_samples.png 6 random chips per site (RGB, NDVI, both labels)
    <out>/qa/<site>_hist.png    band histograms per date (an offset shows as a shifted curve)

CHIP FORMAT (.npz, compressed):
    image         float32 [8, 256, 256]  B02 B03 B04 B08 B11 B12 NDVI MNDWI (reflectance; NaN = no data)
    label_v4019   uint8   [256, 256]     GMW v4.0.19 (10 m, 2020 only): 1 mangrove, 0 not, 255 = no label
    label_v4112   uint8   [256, 256]     GMW v4.1.12 (~30 m, 2018/2020/2025 same year as scene): same coding
    cloud         bool    [256, 256]     SCL classes 3, 8, 9, 10 (cloud shadow, cloud medium/high, cirrus)
    nodata        bool    [256, 256]     NaN in any band or SCL == 0
    transform     float64 [6]            affine (a, b, c, d, e, f) of the chip, EPSG:32640
    crs           str                    "EPSG:32640"
    channels      str     [8]            channel names
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds, Window
from rasterio.transform import Affine

CHANNELS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "MNDWI"]
CLOUD_SCL = (3, 8, 9, 10)
NOLABEL = 255
V4112_YEARS = (2018, 2020, 2025)
INDEX_EPS = 0.01  # denominator floor for NDVI / MNDWI


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="raw_manifest.csv from collect.py (paths relative to its folder)")
    p.add_argument("--sites", required=True, help="sites.geojson with name + proposed_split")
    p.add_argument("--labels", required=True, help="folder with gmw_v4019_2020_uae.tif and gmw_v4112_<year>_uae.tif")
    p.add_argument("--out", required=True)
    p.add_argument("--chip", type=int, default=256)
    p.add_argument("--max-masked", type=float, default=0.05, help="drop chips with more cloud/shadow than this")
    p.add_argument("--max-nodata", type=float, default=0.10, help="drop chips with more no-data than this")
    p.add_argument("--min-mangrove", type=float, default=0.01, help="mangrove chip = this fraction under either label")
    p.add_argument("--bg-frac", type=float, default=0.20, help="background-only chips as a share of kept chips, per site")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-override", nargs="*", default=[], help="site=split pairs, e.g. eastern_mangroves=train")
    p.add_argument("--no-qa", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- inputs

def load_splits(sites_path, overrides):
    with open(sites_path) as f:
        gj = json.load(f)
    splits = {ft["properties"]["name"]: ft["properties"].get("proposed_split", "unassigned") for ft in gj["features"]}
    for o in overrides:
        k, v = o.split("=", 1)
        splits[k] = v
    return splits


def label_paths(labels_dir, year):
    v4019 = os.path.join(labels_dir, "gmw_v4019_2020_uae.tif") if year == 2020 else None
    v4112 = os.path.join(labels_dir, f"gmw_v4112_{year}_uae.tif") if year in V4112_YEARS else None
    return (v4019 if v4019 and os.path.exists(v4019) else None,
            v4112 if v4112 and os.path.exists(v4112) else None)


def warp_label(label_path, dst_shape, dst_transform, dst_crs):
    """Nearest-neighbour warp of a (large, EPSG:4326) label raster onto the scene grid, reading only the needed window."""
    out = np.full(dst_shape, NOLABEL, dtype=np.uint8)
    if label_path is None:
        return out
    h, w = dst_shape
    dst_bounds = rasterio.transform.array_bounds(h, w, dst_transform)
    with rasterio.open(label_path) as src:
        l, b, r, t = transform_bounds(dst_crs, src.crs, *dst_bounds, densify_pts=21)
        pad = 10 * abs(src.transform.a)
        win = from_bounds(l - pad, b - pad, r + pad, t + pad, src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        arr = src.read(1, window=win)
        src_tr = src.window_transform(win)
        nodata = src.nodata
        arr = arr.astype(np.uint8)
        if nodata is not None:
            arr[arr == nodata] = NOLABEL
        arr[(arr != 1) & (arr != NOLABEL)] = 0
        reproject(arr, out, src_transform=src_tr, src_crs=src.crs, dst_transform=dst_transform,
                  dst_crs=dst_crs, resampling=Resampling.nearest, src_nodata=NOLABEL, dst_nodata=NOLABEL)
    return out


def load_scene(path):
    with rasterio.open(path) as src:
        a = src.read().astype(np.float32)  # 7 bands
        tr, crs = src.transform, src.crs
    refl = a[:6]
    scl = a[6]
    # Reflectance can be slightly negative over water (correct after the -0.1 offset), so a plain
    # ratio explodes near a zero denominator. Floor the denominator at INDEX_EPS and clip to [-1, 1].
    # Never clip reflectance itself to 0: that is what produced NDVI = 1 over water before.
    with np.errstate(invalid="ignore", divide="ignore"):
        b03, b04, b08, b11 = refl[1], refl[2], refl[3], refl[4]
        ndvi = np.clip((b08 - b04) / np.maximum(b08 + b04, INDEX_EPS), -1, 1)
        mndwi = np.clip((b03 - b11) / np.maximum(b03 + b11, INDEX_EPS), -1, 1)
    img = np.concatenate([refl, ndvi[None], mndwi[None]], axis=0)
    img[~np.isfinite(img)] = np.nan
    nodata = np.isnan(refl).any(axis=0) | ~np.isfinite(scl) | (np.nan_to_num(scl) == 0)
    cloud = np.isin(np.nan_to_num(scl, nan=0).astype(np.int16), CLOUD_SCL)
    return img, cloud, nodata, tr, crs


# ---------------------------------------------------------------- chip stats

def chip_grid(h, w, c):
    return [(r, col) for r in range(0, h - c + 1, c) for col in range(0, w - c + 1, c)]


def chip_stats(cloud, nodata, lab19, lab12, r, c, size):
    sl = (slice(r, r + size), slice(c, c + size))
    nd = nodata[sl]
    n = nd.size
    valid = ~nd
    nv = max(int(valid.sum()), 1)
    cl = cloud[sl] & valid

    def mfrac(lab):
        l = lab[sl]
        ok = valid & (l != NOLABEL)
        return float((l[ok] == 1).sum() / ok.sum()) if ok.sum() else float("nan")

    return dict(nodata_frac=float(nd.sum() / n), cloud_frac=float(cl.sum() / nv),
                mang_v4019=mfrac(lab19), mang_v4112=mfrac(lab12))


# ---------------------------------------------------------------- main

def main():
    a = parse_args()
    rng = np.random.default_rng(a.seed)
    man_dir = os.path.dirname(os.path.abspath(a.manifest))
    with open(a.manifest) as f:
        rows = list(csv.DictReader(f))
    splits = load_splits(a.sites, a.split_override)
    os.makedirs(os.path.join(a.out, "chips"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "qa"), exist_ok=True)

    by_site = defaultdict(list)
    for r in rows:
        by_site[r["site"]].append(r)

    catalog = []
    stats_all = []  # for summary
    norm_sum = np.zeros(len(CHANNELS)); norm_sq = np.zeros(len(CHANNELS)); norm_n = np.zeros(len(CHANNELS))
    hist_data = defaultdict(dict)  # site -> date -> sampled band values

    for site, srows in sorted(by_site.items()):
        split = splits.get(site, "unassigned")
        if split == "unassigned":
            print(f"WARNING: {site} has no split in {a.sites}; marked 'unassigned'")
        # pass 1: candidate chips with stats
        cands = []
        for r in srows:
            path = os.path.join(man_dir, r["path"])
            year = int(r["date"][:4])
            img, cloud, nodata, tr, crs = load_scene(path)
            p19, p12 = label_paths(a.labels, year)
            lab19 = warp_label(p19, cloud.shape, tr, crs)
            lab12 = warp_label(p12, cloud.shape, tr, crs)
            # histogram sample (valid, cloud-free pixels)
            ok = ~nodata & ~cloud
            if ok.any():
                idx = np.flatnonzero(ok)
                pick = rng.choice(idx, size=min(50000, idx.size), replace=False)
                hist_data[site][r["date"]] = img[:6].reshape(6, -1)[:, pick]
            for (rr, cc) in chip_grid(*cloud.shape, a.chip):
                s = chip_stats(cloud, nodata, lab19, lab12, rr, cc, a.chip)
                s.update(row=rr, col=cc, r=r)
                cands.append(s)
            del img, cloud, nodata, lab19, lab12

        ok = [s for s in cands if s["nodata_frac"] <= a.max_nodata and s["cloud_frac"] <= a.max_masked]
        mang = [s for s in ok if np.nanmax([s["mang_v4019"], s["mang_v4112"], -1]) >= a.min_mangrove]
        mang_ids = {id(s) for s in mang}
        bg = [s for s in ok if id(s) not in mang_ids]
        n_bg = int(round(len(mang) * a.bg_frac / (1 - a.bg_frac)))
        if bg and n_bg:
            sel = rng.choice(len(bg), size=min(n_bg, len(bg)), replace=False)
            bg = [bg[i] for i in sorted(sel)]
        else:
            bg = []
        keep = mang + bg
        print(f"{site} [{split}]: {len(cands)} candidate chips, {len(ok)} pass cloud/nodata, "
              f"{len(mang)} with mangrove, {len(bg)} background kept")
        stats_all.append((site, split, len(cands), len(ok), len(mang), len(bg)))

        # pass 2: write the kept chips, one scene at a time
        keep_by_file = defaultdict(list)
        for s in keep:
            keep_by_file[s["r"]["path"]].append(s)
        samples = []
        for fpath, ss in keep_by_file.items():
            r = ss[0]["r"]
            year = int(r["date"][:4])
            img, cloud, nodata, tr, crs = load_scene(os.path.join(man_dir, fpath))
            p19, p12 = label_paths(a.labels, year)
            lab19 = warp_label(p19, cloud.shape, tr, crs)
            lab12 = warp_label(p12, cloud.shape, tr, crs)
            for s in ss:
                rr, cc, c = s["row"], s["col"], a.chip
                sl = (slice(rr, rr + c), slice(cc, cc + c))
                ctr = tr * Affine.translation(cc, rr)
                chip_id = f"{site}_{r['date']}_r{rr:05d}_c{cc:05d}"
                im = img[:, sl[0], sl[1]].copy()
                rec = dict(image=im.astype(np.float32), label_v4019=lab19[sl].copy(), label_v4112=lab12[sl].copy(),
                           cloud=cloud[sl].copy(), nodata=nodata[sl].copy(),
                           transform=np.array([ctr.a, ctr.b, ctr.c, ctr.d, ctr.e, ctr.f]),
                           crs=np.array("EPSG:32640"), channels=np.array(CHANNELS))
                rel = os.path.join("chips", chip_id + ".npz")
                np.savez_compressed(os.path.join(a.out, rel), **rec)
                minx, maxy = ctr.c, ctr.f
                maxx, miny = minx + c * ctr.a, maxy + c * ctr.e
                catalog.append(dict(
                    chip_id=chip_id, site=site, date=r["date"], year=year, split=split, row=rr, col=cc,
                    bounds_epsg32640=f"{minx:.0f},{miny:.0f},{maxx:.0f},{maxy:.0f}",
                    cloud_pct=round(100 * s["cloud_frac"], 2), nodata_pct=round(100 * s["nodata_frac"], 2),
                    mangrove_pct_v4019="" if np.isnan(s["mang_v4019"]) else round(100 * s["mang_v4019"], 2),
                    mangrove_pct_v4112="" if np.isnan(s["mang_v4112"]) else round(100 * s["mang_v4112"], 2),
                    kind="mangrove" if id(s) in mang_ids else "background",
                    offset_applied=r.get("offset_applied", ""), item_id=r.get("item_id", ""),
                    source_file=fpath, path=rel))
                if split == "train":
                    good = ~rec["nodata"] & ~rec["cloud"]
                    v = im[:, good].astype(np.float64)
                    fin = np.isfinite(v)
                    norm_sum += np.where(fin, v, 0).sum(1)
                    norm_sq += np.where(fin, v * v, 0).sum(1)
                    norm_n += fin.sum(1)
                samples.append(rec)
            del img, cloud, nodata, lab19, lab12

        if not a.no_qa:
            qa_samples(site, samples, rng, os.path.join(a.out, "qa", f"{site}_samples.png"))
            qa_hist(site, hist_data[site], os.path.join(a.out, "qa", f"{site}_hist.png"))

    # catalog
    cat_path = os.path.join(a.out, "catalog.csv")
    if catalog:
        with open(cat_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(catalog[0].keys()))
            w.writeheader(); w.writerows(catalog)
    # norm
    if norm_n.sum() > 0:
        mean = norm_sum / np.maximum(norm_n, 1)
        std = np.sqrt(np.maximum(norm_sq / np.maximum(norm_n, 1) - mean ** 2, 0))
        norm = dict(channels=CHANNELS, mean=mean.round(6).tolist(), std=std.round(6).tolist(),
                    n_pixels=norm_n.astype(int).tolist(), computed_from="train chips, valid cloud-free pixels")
    else:
        norm = dict(channels=CHANNELS, mean=None, std=None, note="no train chips in this run")
    with open(os.path.join(a.out, "norm.json"), "w") as f:
        json.dump(norm, f, indent=1)

    # summary
    print("\nsplit summary (kept chips):")
    tot = defaultdict(lambda: [0, 0])
    for site, split, _, _, m, b in stats_all:
        tot[split][0] += m; tot[split][1] += b
    for sp, (m, b) in sorted(tot.items()):
        print(f"  {sp}: {m + b} chips ({m} mangrove, {b} background)")
    print(f"catalog: {cat_path} ({len(catalog)} rows)")


# ---------------------------------------------------------------- QA figures

def qa_samples(site, samples, rng, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not samples:
        return
    pick = rng.choice(len(samples), size=min(6, len(samples)), replace=False)
    fig, axes = plt.subplots(len(pick), 4, figsize=(10, 2.5 * len(pick)), squeeze=False)
    for i, k in enumerate(pick):
        s = samples[k]
        rgb = np.nan_to_num(np.stack([s["image"][2], s["image"][1], s["image"][0]], -1) / 0.25).clip(0, 1)
        for ax, (im, title, kw) in zip(axes[i], [
                (rgb, "RGB", {}),
                (s["image"][6], "NDVI", dict(cmap="RdYlGn", vmin=-0.3, vmax=0.8)),
                (np.ma.masked_equal(s["label_v4019"], NOLABEL), "GMW v4.0.19", dict(cmap="Greens", vmin=0, vmax=1)),
                (np.ma.masked_equal(s["label_v4112"], NOLABEL), "GMW v4.1.12", dict(cmap="Greens", vmin=0, vmax=1))]):
            ax.imshow(im, **kw); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9)
    fig.suptitle(f"{site}: 6 random kept chips (blank label = no label for that year)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def qa_hist(site, by_date, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not by_date:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    bins = np.linspace(-0.05, 0.6, 131)
    for b, ax in enumerate(axes.ravel()):
        for date, v in sorted(by_date.items()):
            x = v[b]; x = x[np.isfinite(x)]
            ax.hist(x, bins=bins, histtype="step", density=True, label=date)
        ax.set_title(CHANNELS[b], fontsize=9); ax.axvline(0, color="k", lw=0.5)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{site}: reflectance per date (valid, cloud-free pixels). Water should sit near 0.00-0.03, not 0.10+",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
