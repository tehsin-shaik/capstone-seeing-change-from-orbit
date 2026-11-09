# Mangrove pipeline code

> **Running on your own PC?** Read `RUN_LOCALLY.md` in the SDP folder. In short, from the SDP folder:
> `conda env create -f 2_Code\environment.yml`, `conda activate mangrove`, `python 2_Code\run_all.py`.
> Every path comes from `paths.py` (inputs in `3_Data`, big files in `6_Work`, results in `4_Results`);
> `make_datasets.py` builds the Train/Val and locked Test datasets, `landcover.py` the WorldCover chips.

## Data collection

Two scripts that do the same job: pull Sentinel-2 surface reflectance for each site and date onto one grid
(EPSG:32640, 10 m). **Use one of them for the whole project, not both.**

| Script | Source | Login | Labels / radar | Runs where |
| --- | --- | --- | --- | --- |
| `collect.py` (default) | Earth Search v1, `sentinel-2-c1-l2a` (public COGs on AWS) | none | no (add GMW in Stage 2) | laptop, Colab, server |
| `collect_gee.py` | GEE `COPERNICUS/S2_SR_HARMONIZED` | Earth Engine account | GMW label band, optional Sentinel-1 VV/VH | Colab |

## Install

```bash
pip install pystac-client rasterio shapely pyproj numpy      # collect.py
pip install earthengine-api                                  # collect_gee.py
```

## Usage

`sites.geojson` = lon/lat FeatureCollection, one feature per site with a `name` property.

```bash
# see which scenes it would pick (nothing downloaded)
python collect.py --sites sites.geojson --start 2020-01-01 --end 2020-12-31 --max-cloud 5 --per-site 0 --out data/ --dry-run

# download an exact, hand-picked list (recommended: pick dates per season yourself)
#   scenes.csv:  site,item_id
#                eastern_mangroves,S2A_T40RBN_20201117T070153_L2A
python collect.py --sites sites.geojson --scenes scenes.csv --out data/

# or let it take the N least-cloudy dates per site
python collect.py --sites sites.geojson --start 2025-10-01 --end 2025-12-31 --per-site 3 --out data/
```

`--per-site` picks the least-cloudy dates, which can all fall in one season. For season coverage, run it once per
season window, or write `scenes.csv` by hand from the `--dry-run` list. Re-runs skip files that already exist.

GEE (in Colab, after `ee.Authenticate(); ee.Initialize(project=...)`):

```python
%run collect_gee.py --sites sites.geojson --start 2020-01-01 --end 2021-01-01 --per-site 4 --s1
```

## Outputs (`collect.py`)

```
data/raw/<site>/<YYYY-MM-DD>.tif     7 bands, float32, NaN = no data
                                     1-6: B02 B03 B04 B08 B11 B12 surface reflectance (B11/B12 bilinear from 20 m)
                                     7:   SCL (nearest); cloud/shadow = classes 3, 8, 9, 10
data/raw_manifest.csv                site, date, item_id, tile, cloud_cover, processing_baseline,
                                     offset_applied, path, bounds_epsg32640, valid_fraction, masked_fraction
```

Size: about 8 MB per site-date for a 7 x 5.7 km site (721 x 568 px). Download time: a few seconds per scene.

## The offset rule (read this)

Earth Search Collection-1 files store raw numbers (DN). Every scene, including 2018 and 2020 (reprocessed as
baseline 05.00), carries ESA's offset, and each file's STAC metadata says so: `scale 0.0001, offset -0.1`.
`collect.py` reads those two numbers per band and applies **reflectance = DN x 0.0001 - 0.1 exactly once**.
The manifest's `offset_applied` column records it.

- Do **not** subtract 0.1 again later, and do **not** clip negatives to 0. Dark water legitimately comes out
  slightly below 0 (around 14% of B04 pixels in the 2020 test scene). Clipping makes NDVI = 1 over water.
- The GEE harmonized collection is already shifted, so there it is DN / 10000 with no extra offset.
- Earlier team data that used `DN / 10000` on Earth Search C1 files was **0.1 too bright**. That matches the
  Seeing Orbit symptom: their chips read about 0.1 darker than ours, with dark pixels at 0. Check
  `so_prepare.py` before blaming their chips.

## Test run (23 Oct 2026, `test_run/`)

Eastern Mangroves AOI 54.40-54.47E, 24.43-24.48N, tile 40RBN, two November scenes:

| Date | Scene | Water B04 median | Vegetation B08 / B04 median | Vegetation NDVI median |
| --- | --- | --- | --- | --- |
| 2020-11-17 | S2A_T40RBN_20201117T070153_L2A | 0.009 | 0.254 / 0.026 | 0.82 |
| 2025-11-20 | S2A_T40RBN_20251120T065320_L2A | 0.049 | 0.238 / 0.064 | 0.57 |

Water and vegetation are both in the expected ranges. 2025 is brighter in the red band on the same month,
probably haze, so NDVI thresholds will not transfer between dates. Normalise per scene, or let models learn it.
There are 25 scenes under 2% cloud in 2020 alone for this site, so imagery is not a limit.

## prepare.py (Stage 2: raw files -> chips)

Turns the per-site/date GeoTIFFs from `collect.py` into 256 x 256 training chips that every model reads.

```bash
pip install rasterio numpy matplotlib
python prepare.py --manifest data/raw_manifest.csv --sites sites/sites.geojson --labels labels/ --out data/
```

What it does, per raw file:
1. Warps the GMW labels onto the scene's exact grid (EPSG:32640, 10 m, nearest neighbour), as two bands so the
   team can choose later: `label_v4019` (GMW v4.0.19, 10 m, 2020 scenes only) and `label_v4112` (GMW v4.1.12,
   ~30 m, same year as the scene for 2018 / 2020 / 2025). 255 = no label for that year.
2. Cloud/shadow mask from SCL classes 3, 8, 9, 10; adds NDVI and MNDWI = (B03 - B11) / (B03 + B11).
   Index denominators are floored at 0.01 and clipped to [-1, 1]; reflectance itself is never clipped.
3. Cuts non-overlapping 256 x 256 chips; drops chips > 5% cloud/shadow or > 10% no-data; keeps every chip with
   >= 1% mangrove under either label plus background-only chips up to 20% of the kept total per site (seed 42).
4. Split comes from the site's `proposed_split` in `sites.geojson` (override with `--split-override site=split`).

Outputs:

```
<out>/chips/<site>_<date>_r<row>_c<col>.npz
      image        float32 [8,256,256]  B02 B03 B04 B08 B11 B12 NDVI MNDWI (NaN = no data)
      label_v4019  uint8   [256,256]    1 mangrove, 0 not, 255 no label
      label_v4112  uint8   [256,256]    same coding
      cloud, nodata bool   [256,256]
      transform    float64 [6]          chip affine in EPSG:32640
<out>/catalog.csv   chip_id, site, date, year, split, row, col, bounds, cloud_pct, nodata_pct,
                    mangrove_pct_v4019, mangrove_pct_v4112, kind, offset_applied, item_id, source_file, path
<out>/norm.json     per-channel mean/std from TRAIN chips only (valid, cloud-free pixels)
<out>/qa/<site>_samples.png   6 random chips: RGB, NDVI, both labels
<out>/qa/<site>_hist.png      band histograms per date; water should sit at 0.00-0.03, not 0.10+
```

Loading a chip:

```python
d = np.load("data/chips/<chip_id>.npz")
x = (d["image"] - mean[:, None, None]) / std[:, None, None]   # from norm.json
y = d["label_v4112"]; valid = (y != 255) & ~d["nodata"] & ~d["cloud"]   # train only on valid pixels
```

Notes: chips are cut on a grid starting at the raster's top-left corner, so a narrow site gives few chips; the
same place appears once per date (all dates of a site share its split, so this does not leak). Tested 23 Oct on
Eastern Mangroves 2020-11-17 and 2025-11-20: 8 chips, ~3 s, ~1.3 MB per chip.

## Sentinel-1

`collect_s1.py` adds radar (VV, VH) for the Stage 7 fusion experiment, on the **exact grid of each Sentinel-2 file**
so the two stack pixel for pixel.

- Source: Microsoft Planetary Computer `sentinel-1-rtc` (terrain-corrected gamma0, linear power). URLs are signed by
  the `planetary-computer` package; no account or key needed. `pip install planetary-computer pystac-client rasterio`
- It imports the grid and GDAL settings from `collect.py`, so keep both files in the same folder.

```bash
# mode 1 (recommended): one radar scene per Sentinel-2 file, closest in time within ±6 days
python collect_s1.py --manifest data/raw_manifest.csv --out data/ [--orbit descending] [--max-gap 6]

# mode 2: every radar scene per site in a window, on collect.py's site grid
python collect_s1.py --sites sites.geojson --start 2020-11-01 --end 2020-11-30 --out data/
```

Output: `data/s1/<site>/<date>.tif`, float32, NaN = no data. Bands: 1 VV, 2 VH (linear), 3 VV_dB, 4 VH_dB.
In mode 1 the date in the file name is the **Sentinel-2** date; the radar date is in `data/s1_manifest.csv`
(`s1_datetime`, `gap_days`, `orbit_state`, `relative_orbit`, `item_id`, `valid_fraction`).
Scenes covering less than 90% of the grid are skipped (`--min-valid`). No speckle filter is applied.

**Pass direction matters.** Ascending (afternoon) and descending (early morning) passes see the coast from
different angles and at different tide states. The nearest scene can switch direction between dates
(the test picked descending for 2020 and ascending for 2025). For change detection, fix one with `--orbit`.

Test (Eastern Mangroves, 23 Oct): 2020-11-17 matched a 2020-11-14 descending pass (3.2 days earlier),
2025-11-20 matched a 2025-11-21 ascending pass (1.3 days later); about 5 MB per file, about 20 s per date.
Median backscatter: water VV −17 dB / VH −21 dB; vegetation (S2 NDVI > 0.5) VV −8 dB / VH −15 dB.

## Sites on a tile edge: same-date mosaics (added 23 Oct)

Some sites sit where several MGRS tiles meet (e.g. `al_aryam_islands`, at the corner of 39QZG, 39RZH, 40QBM and
40RBN), so no single item covers them. In `--scenes` mode, list every item of the same acquisition date on its own
row with the same site name; `collect.py` groups rows by (site, date) and mosaics them. Each pixel comes from the
first listed item that has valid data there, and each item's own scale/offset is applied to its own pixels, once.
The manifest's `item_id` and `tile` then hold the item ids / tiles joined with `+`. A single item per row works as
before. The selection used for the full download is in `../data/scripts/select_scenes.py` and `select_mosaic.py`.

## Evaluate and baselines

Every model is scored by the same script, against the same frozen split. Files: `evaluate.py`, `baseline_ndvi.py` (B1),
`baseline_rf.py` (B2), `chipio.py` (shared chip loading), `tests/test_evaluate.py`.

```bash
pip install numpy pandas scikit-learn joblib rasterio pytest

# B1: NDVI > t AND MNDWI < w. t and w are grid-searched on VALIDATION chips only.
python baseline_ndvi.py fit     --catalog data/catalog.csv --chips data --label v4019 --out models/ndvi.json
python baseline_ndvi.py predict --catalog data/catalog.csv --chips data --model models/ndvi.json --split test --pred-out preds/ndvi

# B2: per-pixel random forest, 200k class-balanced TRAIN pixels; depth/leaf and probability threshold picked on VALIDATION
python baseline_rf.py fit     --catalog data/catalog.csv --chips data --label v4019 --out models/rf.joblib
python baseline_rf.py predict --catalog data/catalog.csv --chips data --model models/rf.joblib --split test --pred-out preds/rf

# Score one model (writes results.json, results.md, per_chip.csv)
python evaluate.py --pred preds/ndvi --catalog data/catalog.csv --chips data --split test --label v4019 --name "B1 NDVI" --out results/ndvi

# Put several models side by side
python evaluate.py --compare results/ndvi/results.json results/rf/results.json --out results/comparison.md
```

**Your model's predictions:** one binary mask per chip, named `<chip_id>.npy`, `.npz` (key `mask`) or `.tif`, same
256 × 256 grid as the chip. 1 = mangrove.

**What is counted:**
- Only usable pixels count: finite in every band, not cloud, not no-data, and label ≠ 255.
- `--label v4019` (10 m, 2020 only) skips 2018 and 2025 chips automatically; they are listed in `chips_skipped_no_label`.
- `--label v4112` (~30 m, 2018/2020/2025) scores every year.
- Metrics are for the mangrove class, pixel-pooled per site and over all chips: IoU, F1, precision, recall.
- 95% CI = bootstrap over chips: 1000 resamples, seed 0, so the CI is identical on every rerun.

**Rules:**
- Choose thresholds and settings on `--split val` only.
- Run `--split test` once per final model.
- `--split external` gives the generalisation score.

`--chips` accepts either the prepare.py output folder or its `chips/` subfolder.

Tests: `python -m pytest -q tests/` (6 tests, known-answer IoU on synthetic chips, prepare.py chip format).
