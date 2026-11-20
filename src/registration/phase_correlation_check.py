"""Check the 2018/2025 alignment with phase correlation instead of SIFT.

All registration results so far come from SIFT landmarks (landmarks.py, real_raster_check.py). This is a
second, independent check: phase correlation compares whole image tiles in the frequency domain and
finds the shift between them directly, with no corners or matching involved. If it also finds a shift
well under a pixel, the "no correction needed" result doesn't depend on SIFT.

It runs on the chips in Train_Val_Dataset, or (with --raw) on full site images downloaded with
src/mangrove_segmentation/collect.py, so every training site gets checked, not just Jubail and
Eastern Mangroves. The chips only cover mangrove areas, so small sites have very few of them; the full
site images are cut into 256 x 256 tiles the same way and give many more. For each site it pairs the same season in both years (March 2018 vs March 2025,
November 2018 vs November 2025), and for every chip location that exists on both dates it:
  - takes the near-infrared band (B08), where land, vegetation and water edges stand out
  - skips the chip if clouds or no-data cover more than 5% of it, or if it's almost all water or all land
  - fills the few bad pixels, applies a Hann window and runs phase correlation at 1/100 pixel
  - does the same on the red band (B04) and drops the chip if the two bands disagree by more than 2 m.
    Chips where the ground really changed (new canals, sand bars moving) or where there's nothing to
    lock on to give a random answer, and two bands won't agree on a random answer

The shift reported is where the 2025 image sits relative to 2018, in metres (east, north), the same
direction as the "whole-image shift" in results/registration/. Each site's number is the median over
its chips, with a bootstrap interval over chips.

Usage (from the repo root):
  python src/registration/phase_correlation_check.py --train-val <Train_Val_Dataset> \\
      --out results/registration/phase_correlation
  # full site images from collect.py (<raw>/<site>/<date>.tif), dates taken from the same catalog
  python src/registration/phase_correlation_check.py --train-val <Train_Val_Dataset> --raw <raw> \\
      --out results/registration/phase_correlation/full_sites
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation

PIXEL_M = 10.0
RED, NIR = 2, 3         # image channels are B02 B03 B04 B08 B11 B12 NDVI MNDWI
MNDWI = 7
MAX_BAD = 0.05          # skip chips with more than 5% cloud / no-data
LAND_RANGE = (0.05, 0.95)   # skip chips that are almost all water or all land (nothing to lock on to)
AGREE_M = 2.0           # NIR and red must give the same offset within this, or the chip is dropped
MIN_CHIPS = 3           # fewer chips than this and the site result is marked as too few


def offset_m(reference: np.ndarray, moving: np.ndarray, upsample: int = 100) -> tuple[float, float, float]:
    """Where `moving` sits relative to `reference`, in metres east and north, plus the fit error.

    skimage gives the shift that would move `moving` back onto `reference` in (row, col) pixels,
    so the offset is the opposite of that. Rows go down (south), hence the sign on north.
    """
    win = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))
    ref = (reference - reference.mean()) * win
    mov = (moving - moving.mean()) * win
    shift, error, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample, normalization=None)
    d_row, d_col = shift
    return float(-d_col * PIXEL_M), float(d_row * PIXEL_M), float(error)


def season_pairs(dates_2018: list[str], dates_2025: list[str]) -> list[tuple[str, str]]:
    """Same-month pairs (Mar/Mar, Nov/Nov), so tide and season differ as little as possible."""
    pairs = []
    for month in ("03", "11"):
        a = [d for d in dates_2018 if d[5:7] == month]
        b = [d for d in dates_2025 if d[5:7] == month]
        if a and b:
            pairs.append((a[0], b[0]))
    return pairs


def location(chip_id: str) -> str:
    return chip_id.split("_r", 1)[1] if "_r" in chip_id else chip_id


def load(tv: Path, path: str):
    z = np.load(tv / path)
    bad = z["cloud"].astype(bool) | z["nodata"].astype(bool) | ~np.isfinite(z["image"][NIR])
    return z["image"], bad


TILE = 256
SCL_BAD = {0, 1, 3, 8, 9, 10}   # no data, saturated, cloud shadow, clouds, cirrus


def raw_tiles(raw: Path, site: str, date: str) -> dict:
    """Cut a collect.py GeoTIFF into chip-sized tiles, in the same channel order as the chips."""
    import rasterio
    with rasterio.open(raw / site / f"{date}.tif") as src:
        a = src.read().astype(np.float32)          # B02 B03 B04 B08 B11 B12 SCL
    scl = np.nan_to_num(a[6], nan=0).astype(int)
    bad_all = np.isin(scl, list(SCL_BAD)) | ~np.isfinite(a[:6]).all(0)
    mndwi = (a[1] - a[4]) / np.maximum(a[1] + a[4], 0.01)
    img = np.concatenate([a[:6], np.zeros((1,) + a.shape[1:], np.float32), mndwi[None]])
    out = {}
    for r in range(0, a.shape[1] - TILE + 1, TILE):
        for c in range(0, a.shape[2] - TILE + 1, TILE):
            out[f"{r:05d}_c{c:05d}"] = (img[:, r:r + TILE, c:c + TILE], bad_all[r:r + TILE, c:c + TILE])
    return out


def compare(ia, bad_a, ib, bad_b):
    """Offset of one tile pair, or None if the tile is too cloudy or has nothing to lock on to."""
    bad = bad_a | bad_b
    if bad.mean() > MAX_BAD:
        return None
    land = np.nanmean(ia[MNDWI][~bad] < 0)
    if not LAND_RANGE[0] <= land <= LAND_RANGE[1]:
        return None
    res = {}
    for name, band in (("nir", NIR), ("red", RED)):
        ra, rb = ia[band].copy(), ib[band].copy()
        ra[bad] = np.nanmedian(ra[~bad]); rb[bad] = np.nanmedian(rb[~bad])
        res[name] = offset_m(ra, rb)
    (east, north, err), (e_red, n_red, _) = res["nir"], res["red"]
    return dict(east_m=round(east, 2), north_m=round(north, 2), error=round(err, 4),
                east_red_m=round(e_red, 2), north_red_m=round(n_red, 2),
                bands_agree=bool(np.hypot(east - e_red, north - n_red) <= AGREE_M),
                land_frac=round(float(land), 2), bad_frac=round(float(bad.mean()), 3))


def check_site_raw(raw: Path, cat: pd.DataFrame, site: str, log=print) -> list[dict]:
    rows = []
    s = cat[cat["site"] == site]
    d18 = sorted(s.loc[s["year"] == 2018, "date"].unique())
    d25 = sorted(s.loc[s["year"] == 2025, "date"].unique())
    for a, b in season_pairs(d18, d25):
        if not ((raw / site / f"{a}.tif").exists() and (raw / site / f"{b}.tif").exists()):
            log(f"  {site:18s} {a} vs {b}: image missing, skipped")
            continue
        A, B = raw_tiles(raw, site, a), raw_tiles(raw, site, b)
        used = tried = 0
        for loc in sorted(set(A) & set(B)):
            r = compare(*A[loc], *B[loc])
            if r is None:
                continue
            rows.append(dict(site=site, pair=f"{a} vs {b}", date_2018=a, date_2025=b, chip=loc, **r))
            used += r["bands_agree"]; tried += 1
        log(f"  {site:18s} {a} vs {b}: {used} of {tried} tiles kept (bands agree)")
    return rows


def check_site(tv: Path, cat: pd.DataFrame, site: str, log=print) -> list[dict]:
    rows = []
    s = cat[cat["site"] == site]
    d18 = sorted(s.loc[s["year"] == 2018, "date"].unique())
    d25 = sorted(s.loc[s["year"] == 2025, "date"].unique())
    for a, b in season_pairs(d18, d25):
        A = s[s["date"] == a].assign(loc=lambda t: t["chip_id"].map(location)).set_index("loc")
        B = s[s["date"] == b].assign(loc=lambda t: t["chip_id"].map(location)).set_index("loc")
        used = tried = 0
        for loc in sorted(set(A.index) & set(B.index)):
            r = compare(*load(tv, A.loc[loc, "path"]), *load(tv, B.loc[loc, "path"]))
            if r is None:
                continue
            rows.append(dict(site=site, pair=f"{a} vs {b}", date_2018=a, date_2025=b, chip=loc, **r))
            used += r["bands_agree"]; tried += 1
        log(f"  {site:18s} {a} vs {b}: {used} of {tried} chips kept (bands agree)")
    return rows


def summarise(per_chip: pd.DataFrame, reps: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Median offset per site and pair, with a bootstrap interval over chips for the size of the shift."""
    rng = np.random.default_rng(seed)
    out = []
    kept = per_chip[per_chip["bands_agree"]] if "bands_agree" in per_chip else per_chip
    for (site, pair), g in kept.groupby(["site", "pair"], sort=False):
        e, n = g["east_m"].to_numpy(), g["north_m"].to_numpy()
        me, mn = np.median(e), np.median(n)
        boot = []
        for _ in range(reps):
            i = rng.integers(0, len(g), len(g))
            boot.append(np.hypot(np.median(e[i]), np.median(n[i])))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out.append(dict(site=site, pair=pair, chips=len(g), east_m=round(me, 2), north_m=round(mn, 2),
                        shift_m=round(float(np.hypot(me, mn)), 2), shift_ci95=f"{lo:.2f} to {hi:.2f}",
                        shift_px=round(float(np.hypot(me, mn)) / PIXEL_M, 2),
                        chip_spread_m=round(float(np.median(np.hypot(e - me, n - mn))), 2),
                        enough_chips=len(g) >= MIN_CHIPS))
    return pd.DataFrame(out)


def figure(per_chip: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = list(dict.fromkeys(summary["site"]))
    fig, axes = plt.subplots(1, len(sites), figsize=(3.1 * len(sites), 3.4), sharex=True, sharey=True)
    for ax, site in zip(np.atleast_1d(axes), sites):
        sub = per_chip[(per_chip["site"] == site) & per_chip["bands_agree"]]
        for (pair, g), mk in zip(sub.groupby("pair"), ("o", "s")):
            ax.scatter(g["east_m"], g["north_m"], s=14, alpha=0.6, marker=mk, label=pair[:7] + " / " + pair[-10:-3])
        for _, r in summary[summary["site"] == site].iterrows():
            ax.plot(r["east_m"], r["north_m"], "k+", ms=12, mew=2)
        ax.add_patch(plt.Circle((0, 0), PIXEL_M, fill=False, ls="--", color="grey"))
        ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
        ax.set_title(site.replace("_", " "), fontsize=9); ax.set_aspect("equal")
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
        ax.legend(fontsize=6, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("north (m)")
    fig.supxlabel("east (m)   ·   dashed circle = 1 pixel (10 m), + = site median", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-val", type=Path, required=True, help="Train_Val_Dataset folder (catalog.csv, chips/)")
    ap.add_argument("--out", type=Path, default=Path("results/registration/phase_correlation"))
    ap.add_argument("--raw", type=Path, help="use full site images from collect.py (<raw>/<site>/<date>.tif)")
    ap.add_argument("--sites", nargs="*", help="default: every site in the catalog")
    args = ap.parse_args(argv)

    cat = pd.read_csv(args.train_val / "catalog.csv")
    cat = cat[cat["year"].isin([2018, 2025])]
    sites = args.sites or sorted(cat["site"].unique())
    print(f"phase correlation on {len(sites)} sites ({'full site images' if args.raw else 'dataset chips'})")
    rows = []
    for site in sites:
        rows += check_site_raw(args.raw, cat, site) if args.raw else check_site(args.train_val, cat, site)
    per_chip = pd.DataFrame(rows)
    summary = summarise(per_chip)
    print(summary.to_string(index=False))

    args.out.mkdir(parents=True, exist_ok=True)
    per_chip.to_csv(args.out / "per_chip.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    figure(per_chip, summary, args.out / "fig_offsets.png")
    print(f"wrote {args.out}/per_chip.csv, summary.csv and fig_offsets.png")


if __name__ == "__main__":
    main()
