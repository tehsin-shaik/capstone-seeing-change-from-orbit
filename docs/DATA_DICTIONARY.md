# Data Dictionary

What every data file, column, band and label in this project means, and where each fact comes from.

Written 15 Nov 2026 from the contents of this repository only (branch `zain-data-dictionary`).

## How to read this document

- **Source** says which file a fact was taken from. Code files are the authority: where the code and a document disagree, this dictionary reports **what the code does** and lists the disagreement in [section 11](#11-where-code-and-documents-disagree).
- **UNVERIFIED** means the repository does not contain enough to confirm the fact. The usual reason is that the data files themselves (`catalog.csv`, `norm.json`, `sites.geojson`, the label rasters, the satellite images) are kept outside Git, so only the code that reads or writes them could be checked. All UNVERIFIED items are collected in [section 12](#12-everything-marked-unverified).
- Nothing here was measured on the real dataset, because the real dataset is not in the repository.

Short names used for sources:

| Short name | File |
| --- | --- |
| `collect.py` | [src/mangrove_segmentation/collect.py](../src/mangrove_segmentation/collect.py) |
| `prepare.py` | [src/mangrove_segmentation/prepare.py](../src/mangrove_segmentation/prepare.py) |
| `chipio.py` | [src/mangrove_segmentation/chipio.py](../src/mangrove_segmentation/chipio.py) |
| `make_datasets.py` | [src/mangrove_segmentation/make_datasets.py](../src/mangrove_segmentation/make_datasets.py) |
| `paths.py` | [src/mangrove_segmentation/paths.py](../src/mangrove_segmentation/paths.py) |
| `landcover.py` | [src/mangrove_segmentation/landcover.py](../src/mangrove_segmentation/landcover.py) |
| `train_unet.py` | [src/mangrove_segmentation/train_unet.py](../src/mangrove_segmentation/train_unet.py) |
| `evaluate.py` | [src/mangrove_segmentation/evaluate.py](../src/mangrove_segmentation/evaluate.py) |
| pipeline README | [src/mangrove_segmentation/README.md](../src/mangrove_segmentation/README.md) |
| notebook 00 / 01 / 02 | [src/mangrove_segmentation/notebooks/](../src/mangrove_segmentation/notebooks/) |
| baselines RESULTS | [results/mangrove_segmentation/baselines/RESULTS.md](../results/mangrove_segmentation/baselines/RESULTS.md) |
| baselines cv_results | [results/mangrove_segmentation/baselines/cv_results.csv](../results/mangrove_segmentation/baselines/cv_results.csv) |
| data README | [data/README.md](../data/README.md) |
| registration README | [src/registration/README.md](../src/registration/README.md) |
| test-chip zip | [data/test_chips/seeing-orbit-test-chips-inputs-only.zip](../data/test_chips/seeing-orbit-test-chips-inputs-only.zip) (its `README.txt`, `manifest_test.csv`, `SOURCE_ATTRIBUTION.txt`, and the arrays inside) |
| U-Net report | [docs/reports/Seeing_Orbit_Mangrove_U-Net_Report.pdf](reports/Seeing_Orbit_Mangrove_U-Net_Report.pdf) |
| hand label | [data/hand_labels/KAB_2020_07.gpkg](../data/hand_labels/KAB_2020_07.gpkg), read with `ogrinfo` |

## 1. There are two separate datasets in this repository

This is the most important thing to know before reading further. The two are easy to mix up and they differ in almost every detail.

| | **Main pipeline** | **"Seeing Orbit" set** |
| --- | --- | --- |
| Where | Code in `src/mangrove_segmentation/`, results in `results/mangrove_segmentation/` | `data/test_chips/seeing-orbit-test-chips-inputs-only.zip` and the U-Net report PDF in `docs/reports/` |
| Area | 13 UAE sites | Eastern Mangroves only |
| Dates | 8 per site: four in 2020, plus March and November of 2018 and 2025 | 5 Nov 2018 and 1 Nov 2025 |
| Channels per chip | 8: B02, B03, B04, B08, B11, B12, NDVI, **MNDWI** | 6: blue, green, red, NIR, NDVI, **NDWI** |
| Label | GMW v4.0.19 (training) and GMW v4.1.12 (stored alongside) | GMW v4.1.12 only |
| Chip names | `<site>_<date>_r<row>_c<col>` | `patch_<5 digits>` |
| Negative reflectance | Kept (never clipped) | None present; about a third or more of the values are exactly 0 |
| Model results | Baselines, U-Net, DINOv3 in `results/` | The U-Net report |

Sources: `prepare.py`, notebook 00, test-chip zip, U-Net report. Notebook 01 calls GMW v4.1.12 "the Seeing Orbit reference".

Sections 2 to 9 describe the main pipeline. Section 10 describes the Seeing Orbit set.

## 2. Sites

The pipeline uses **13 UAE sites** (notebook 00: "104 Sentinel-2 images (13 UAE sites × 8 dates)").

| Site name in the data | Role | Fold | 2020 chips | Source |
| --- | --- | --- | --- | --- |
| `eastern_mangroves` | cross-validation | 1 | 8 | baselines cv_results; `make_datasets.py` |
| `jubail_island` | cross-validation | 2 | 80 | same |
| `ras_ghurab` | cross-validation | 3 | 50 | same |
| `ghantoot` | cross-validation | 4 | 41 | same |
| `al_aryam_islands` | cross-validation | 5 | 36 together | same |
| `al_nouf` | cross-validation | 5 | (with Al Aryam) | same |
| Abu Al Abyad | fixed test, **locked** | none | UNVERIFIED | `make_datasets.py` (text of `README_LOCKED`) |
| Marawah | fixed test, **locked** | none | UNVERIFIED | same |
| Ras Al Khor | external, **locked** | none | UNVERIFIED | same |
| Al Zorah | external, **locked** | none | UNVERIFIED | same |
| Khor Al Beida | external, **locked** | none | UNVERIFIED | same |
| Ras Al Khaimah creek | external, **locked** | none | UNVERIFIED | same |
| Khor Kalba | external, **locked** | none | UNVERIFIED | same |

- The six cross-validation sites are in Abu Dhabi and hold 215 chips from 2020 (baselines RESULTS). Al Aryam and Al Nouf are 1.5 km apart and share fold 5 (notebook 02).
- "Locked" means the chips must not be opened, plotted, trained on or tuned on until the final evaluation, and are scored once (`make_datasets.py`).
- The seven locked sites are written above as they appear in prose. Their exact names in the data (the lower-case, underscore form) are **UNVERIFIED**.
- Site boundaries live in `3_Data/sites/sites.geojson`, which is not in the repository. It is a longitude/latitude (EPSG:4326) file with one feature per site, a `name` property and a `proposed_split` property (`collect.py`, `prepare.py`). The boundaries are **UNVERIFIED**. The only box given anywhere is Eastern Mangroves: 54.40–54.47° E, 24.43–24.48° N, tile 40RBN, 721 × 568 pixels (pipeline README).
- The fold and role of each site come from `3_Data/folds_site_cv.csv` (columns `site`, `fold`, `role`, optionally `region`; roles are `cv`, `test_fixed`, `external_fixed`). That file is not in the repository; the table above is rebuilt from the sources named in it.

## 3. Satellite imagery

| Fact | Value | Source |
| --- | --- | --- |
| Satellite and product | Sentinel-2 Level-2A surface reflectance | `collect.py` |
| Where it is downloaded from | Earth Search v1 STAC, `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-c1-l2a` (public files on AWS, no login) | `collect.py` |
| Images per site | 8 dates: four in 2020, March and November of 2018 and 2025, all under 10% cloud | notebook 00 |
| Exact image list | `3_Data/scenes_selected.csv` (columns `site`, `item_id`) | `paths.py`, `collect.py`. File not in the repository, so the exact dates are **UNVERIFIED** |
| Example image ID | `S2A_T40RBN_20201117T070153_L2A` | pipeline README |
| Pixel size | 10 m | `collect.py` (`RES = 10.0`) |
| Total size | about 2.4 GB of raw images | notebook 00, `run_all.py` |

### Bands

| Position in chip | Name | What it measures | Earth Search name | Native pixel size | How it is put on the 10 m grid |
| --- | --- | --- | --- | --- | --- |
| 0 | B02 | Blue | `blue` | 10 m | bilinear |
| 1 | B03 | Green | `green` | 10 m | bilinear |
| 2 | B04 | Red | `red` | 10 m | bilinear |
| 3 | B08 | Near-infrared | `nir` | 10 m | bilinear |
| 4 | B11 | Shortwave infrared 1 | `swir16` | 20 m | bilinear |
| 5 | B12 | Shortwave infrared 2 | `swir22` | 20 m | bilinear |
| 6 | NDVI | Vegetation index | computed | | (B08 − B04) / max(B08 + B04, 0.01), clipped to −1…1 |
| 7 | MNDWI | Water index | computed | | (B03 − B11) / max(B03 + B11, 0.01), clipped to −1…1 |

Sources: band order and Earth Search names from `collect.py` (`BANDS`) and `prepare.py` (`CHANNELS`); index formulas from `prepare.py` (`load_scene`); native pixel sizes from the data README. B11 and B12 are stretched from 20 m to 10 m, which changes the grid but adds no detail (data README).

The **SCL** band (Sentinel-2's own scene classification) is also downloaded, with nearest-neighbour resampling. It is used for the cloud mask and is not a model input (`collect.py`).

### Raw image files

Written by `collect.py`, one per site and date:

- Path: `raw/<site>/<YYYY-MM-DD>.tif`
- 7 bands, 32-bit float, NaN = no data. Bands 1–6 are B02, B03, B04, B08, B11, B12 as reflectance. Band 7 is SCL.
- A pixel is no data if any of the six bands, or SCL, is 0 in the source file.
- If a site lies on the edge of several Sentinel-2 tiles, same-date images are merged: each pixel comes from the first listed image that has data there.

### `raw_manifest.csv` columns

One row per raw image. Source: `collect.py` (`MANIFEST_FIELDS` and `process`).

| Column | Meaning |
| --- | --- |
| `site` | Site name |
| `date` | Acquisition date, `YYYY-MM-DD` |
| `item_id` | Earth Search image ID. Several IDs joined with `+` for a merged image |
| `tile` | Sentinel-2 tile code(s), joined with `+` |
| `cloud_cover` | Cloud cover of the whole scene in percent, as reported by the archive (largest value if merged) |
| `processing_baseline` | ESA processing version of the image |
| `offset_applied` | Which scale and offset were applied, and where they came from (see section 4) |
| `path` | File path relative to the output folder |
| `bounds_epsg32640` | Site box as `minx,miny,maxx,maxy` in metres, EPSG:32640 |
| `valid_fraction` | Share of the site's pixels that have data (0–1) |
| `masked_fraction` | Share of the pixels with data that SCL marks as cloud or shadow (0–1) |

## 4. Reflectance: is the offset applied once or twice?

**What the code does: once.** Source: `collect.py`.

- The archive stores whole numbers (DN). Each band of each image carries its own `scale` and `offset` in its metadata. `collect.py` reads them and computes `reflectance = DN × scale + offset` a single time (the line is commented `# applied ONCE`). For these images that is `DN × 0.0001 − 0.1`.
- If an image has no such metadata, the code falls back on the processing version: 04.00 or later gets offset −0.1, older gets 0.
- The rule used is recorded per image in the `offset_applied` column and inside the GeoTIFF's tags.
- No later step changes reflectance. `prepare.py` only reads it. Values are **never clipped**, so dark water is allowed to be slightly negative (about 14% of red-band pixels in the 2020 test scene, pipeline README). Only NDVI and MNDWI are clipped, to −1…1.

**What the documents say that differs:**

- The pipeline README says earlier team data used `DN / 10000` on these files with no offset, which makes values 0.1 too bright, and separately that the Seeing Orbit chips "read about 0.1 darker than ours, with dark pixels at 0". It points to a script `so_prepare.py` for the cause. That script is not in the repository.
- The chips inside the test-chip zip do show that symptom. In all 10 chips the smallest optical value is exactly 0, between 30% and 54% of all optical values are exactly 0, and none are negative (checked directly on the arrays). That is what clipping at 0 looks like. **Whether the cause is an offset applied twice is UNVERIFIED**, because the code that made those chips is not here.
- The data README still says scale factors and the export script "need to be recorded". For the main pipeline that is out of date: `collect.py` records them.

## 5. Coordinate system (CRS)

**What the code does: EPSG:32640 (WGS 84 / UTM zone 40N), 10 m pixels, for every site.**

| Where | What it does | Source |
| --- | --- | --- |
| Download | Warps every image to EPSG:32640 at 10 m, on a grid snapped outward to whole 10 m steps | `collect.py` (`DST_CRS`, `site_grid`) |
| Chips | Each chip stores `crs = "EPSG:32640"` and its own 6-number position (`transform`). The CRS text is written as a fixed string, not read from the image | `prepare.py` |
| Catalog | Chip box stored in the column `bounds_epsg32640` | `prepare.py` |
| Land cover | Warped onto each chip's EPSG:32640 grid | `landcover.py` |
| Hand label | `KAB_2020_07.gpkg` is in EPSG:32640 | hand label (`ogrinfo`) |
| Registration images | EPSG:32640 | registration README |
| Inputs in other systems | `sites.geojson` and ESA WorldCover are longitude/latitude (EPSG:4326). The GMW label rasters are read in whatever CRS they carry (a code comment says EPSG:4326) and warped to the chip grid | `collect.py`, `landcover.py`, `prepare.py` |

**On EPSG:32639:** the number 32639 does not appear anywhere in this repository, in code, documents, notebooks or results. A possible origin: the pipeline README says the site `al_aryam_islands` sits where tiles 39QZG, 39RZH, 40QBM and 40RBN meet. Sentinel-2 tiles whose code starts with 39 are delivered in UTM zone 39N, which is EPSG:32639 (general Sentinel-2 knowledge, not stated in the repository). `collect.py` does not keep that: it warps every tile to EPSG:32640, and when choosing between tiles it prefers the zone-40 one. So any file made by this pipeline is in 32640. Whether a file made some other way is in 32639 is **UNVERIFIED**.

## 6. Chips

A chip is a 256 × 256 pixel square (2.56 km) cut from one raw image. Written by `prepare.py`.

### How chips are cut and chosen

- Squares do not overlap and start at the top-left corner of the image. Leftover strips at the right and bottom edges are not used.
- A chip is dropped if more than 5% of it is cloud or shadow, or more than 10% is no data.
- A chip counts as a mangrove chip if at least 1% of it is mangrove under **either** label.
- Chips with no mangrove are kept only up to 20% of the kept total per site, chosen at random with seed 42.
- The same place appears once per date.

### Chip file contents

Path: `chips/<chip_id>.npz` (compressed NumPy). Source: `prepare.py`.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `image` | float32, 8 × 256 × 256 | The 8 channels of section 3, in that order. Reflectance, NaN = no data |
| `label_v4019` | uint8, 256 × 256 | GMW v4.0.19 label: 1 mangrove, 0 not mangrove, 255 no label |
| `label_v4112` | uint8, 256 × 256 | GMW v4.1.12 label, same coding |
| `cloud` | bool, 256 × 256 | True where SCL is 3, 8, 9 or 10 (cloud shadow, medium cloud, high cloud, thin cirrus) |
| `nodata` | bool, 256 × 256 | True where any band is NaN or SCL is 0 |
| `transform` | float64, 6 numbers | Position of the chip on the ground (affine a, b, c, d, e, f) in EPSG:32640 |
| `crs` | text | `"EPSG:32640"` |
| `channels` | text, 8 names | Channel names, in order |

**Usable pixel.** Every model and the scoring code use only pixels that are finite in all 8 channels, not cloud, not no data, and have a label other than 255 (`chipio.py`, `evaluate.py`).

### `catalog.csv` columns

One row per kept chip. The file itself is not in the repository; the 17 columns below are exactly the ones `prepare.py` writes, in this order.

| # | Column | Meaning | How it is filled |
| --- | --- | --- | --- |
| 1 | `chip_id` | Unique chip name | `<site>_<date>_r<row>_c<col>`, row and col padded to 5 digits |
| 2 | `site` | Site name | From the manifest |
| 3 | `date` | Image date | `YYYY-MM-DD` |
| 4 | `year` | Image year | First 4 characters of `date` |
| 5 | `split` | The site's original split | `proposed_split` of the site in `sites.geojson`, or `unassigned`. Can be overridden on the command line. See the note below |
| 6 | `row` | Top edge of the chip, in pixels from the top of the raw image | Multiple of 256 |
| 7 | `col` | Left edge of the chip, in pixels from the left of the raw image | Multiple of 256 |
| 8 | `bounds_epsg32640` | Chip box, `minx,miny,maxx,maxy`, metres, EPSG:32640 | Rounded to whole metres |
| 9 | `cloud_pct` | Percent of the chip's pixels with data that are cloud or shadow | 2 decimals |
| 10 | `nodata_pct` | Percent of all the chip's pixels that are no data | 2 decimals |
| 11 | `mangrove_pct_v4019` | Percent mangrove under GMW v4.0.19, among pixels that have data and a label | 2 decimals. Empty when there is no v4.0.19 label for that year (every year except 2020) |
| 12 | `mangrove_pct_v4112` | Same for GMW v4.1.12 | Empty when there is no label for that year |
| 13 | `kind` | `mangrove` or `background` | `mangrove` if at least 1% mangrove under either label |
| 14 | `offset_applied` | Reflectance rule used for the source image | Copied from the manifest (section 4) |
| 15 | `item_id` | Earth Search image ID(s) | Copied from the manifest |
| 16 | `source_file` | Raw image the chip was cut from | Path relative to the manifest's folder |
| 17 | `path` | Chip file | `chips/<chip_id>.npz`, relative to the output folder |

Notes:

- The pipeline README lists column 8 as `bounds`. The code writes `bounds_epsg32640`.
- `chipio.py` needs only `chip_id`, `site` and `split` to load a catalog.
- **`split` is not what the reported results use.** Notebook 00 calls it "the original split". The reported results use leave-one-site-out folds instead. Which sites have which `proposed_split` is **UNVERIFIED**.
- `make_datasets.py` writes a second catalog inside each dataset folder (`Train_Val_Dataset`, `Test_Dataset/fixed_test`, `Test_Dataset/external`). It has the same 17 columns plus two more, and `path` is rewritten to `chips/<file name>`:

| Extra column | Meaning |
| --- | --- |
| `role` | `cv`, `test_fixed` or `external_fixed`, from `folds_site_cv.csv` |
| `fold` | Fold number 1–5 for cross-validation sites, from the same file |

- At run time the training code adds `water_b04_p5`, `hazy`, `x0`, `y0`, `x1`, `y1`. These are computed in memory and not saved in the catalog (`train_unet.py`).

## 7. Normalisation files

Normalising means subtracting a mean and dividing by a standard deviation per channel so that all channels are on a similar scale. There are two files, and the deep-learning code uses neither.

| File | Written by | Computed from | Keys |
| --- | --- | --- | --- |
| `norm.json` | `prepare.py` | Chips whose `split` is `train`; pixels that have data, are not cloud, and are finite | `channels` (8 names), `mean` (8 numbers), `std` (8 numbers), `n_pixels` (8 counts, one per channel), `computed_from` (text) |
| `norm_trainval_2020.json` | `make_datasets.py` | All 2020 chips of the Train/Val dataset, all 5 folds; pixels finite in all 8 channels. The cloud mask is **not** applied here | `channels`, `mean`, `std`, `n_pixels` (one count), `source` (text) |

- Values are rounded to 6 decimals. If there are no `train` chips, `norm.json` has `mean` and `std` set to null and a `note`.
- How to apply: `x = (image − mean) / std`, channel by channel (pipeline README).
- **What the models do:** the U-Net computes its own mean and standard deviation inside each fold, from that fold's training chips only (`train_unet.py`, `fold_stats`). It does not read either file. The random forest and the NDVI rule use raw values with no normalisation (`baseline_rf.py`, `baseline_ndvi.py`).
- The actual numbers in both files are **UNVERIFIED**; neither file is in the repository.

## 8. Labels and other reference data

### Global Mangrove Watch (GMW)

`prepare.py` stores two GMW versions side by side in every chip.

| | `label_v4019` | `label_v4112` |
| --- | --- | --- |
| GMW version | v4.0.19 | v4.1.12 |
| Pixel size | 10 m | about 30 m |
| Years available | 2020 only | 2018, 2020, 2025 (the label of the image's own year) |
| Expected file | `gmw_v4019_2020_uae.tif` | `gmw_v4112_<year>_uae.tif` |
| Coding | 1 mangrove, 0 not mangrove, 255 no label | same |
| How it reaches the chip | Nearest-neighbour warp onto the image grid | same |

Sources: `prepare.py` (file names, years, coding, warp), pipeline README and notebook 00 (pixel sizes).

- **Which one the code trains and scores on: v4.0.19.** It is the default label of both baselines and of `evaluate.py`, it is fixed in `train_unet.py`, and every RESULTS file states it. That is why all reported results use 2020 chips only.
- v4.1.12 is kept for the years v4.0.19 does not cover; the Train/Val README inside `make_datasets.py` says the 2018 and 2025 chips "are for change mapping".
- The two versions disagree with each other. Notebook 01 compares them at Jubail Island, and the Meeting 6 minutes record agreement of only IoU 0.47–0.83 between them.
- GMW is produced by a machine-learning classifier, not drawn by people (Meeting 6 minutes).
- **UNVERIFIED:** the label rasters are not in the repository, so their true pixel size, CRS and download source could not be checked. The repository gives no download link for v4.0.19. For v4.1.12 the test-chip zip cites `https://zenodo.org/records/21346457`. The Meeting 5 minutes mention "25 m" ground truth where the README says about 30 m.

### Hand labels

| Fact | Value | Source |
| --- | --- | --- |
| File | `data/hand_labels/KAB_2020_07.gpkg`, with `KAB_2020_07_preview.png` | repository |
| What it is | Mangrove outline drawn by hand in QGIS from a Sentinel-2 false-colour chip of Khor Al Beida, 2020, as human ground truth for checking GMW and the models | commit `69b078f` message |
| Layers | `mangrove`: 24 polygons. `unsure`: 0 polygons | hand label (`ogrinfo`) |
| Attribute | `note` (a number field), empty for all 24 polygons | same |
| CRS | EPSG:32640 | same |
| Extent | 360509.6–363070.3 E, 2830089.5–2832369.3 N, about 2,561 m × 2,280 m, the size of one chip | same |

Khor Al Beida is on the locked external list (section 2). Whether drawing a label on a locked site fits the lock rule is a question for the team, not something the repository settles.

### ESA WorldCover (land cover)

Used only to explain errors by land-cover type, never for training. Source: `landcover.py`.

- Chips from before 2021 use the 2020 map (v100); chips from 2021 or later use the 2021 map (v200).
- Stored as `landcover/<chip_id>.npz`, key `lc`, uint8 class code per pixel (95 = mangrove, 80 = water).

### Haze flag

A scene is "hazy" if the 5th percentile of red (B04) over water is above 0.05. The value is in the column `water_b04_p5` of `3_Data/QA_download.csv`, by site and date (`make_datasets.py`, `train_unet.py`). The file is not in the repository.

## 9. File and naming conventions

| Thing | Pattern | Example | Source |
| --- | --- | --- | --- |
| Site name | lower case, underscores | `eastern_mangroves` | baselines cv_results |
| Raw image | `raw/<site>/<YYYY-MM-DD>.tif` | `raw/eastern_mangroves/2020-11-17.tif` | `collect.py` |
| Chip ID | `<site>_<YYYY-MM-DD>_r<row, 5 digits>_c<col, 5 digits>` | `eastern_mangroves_2020-11-17_r00000_c00256` (format example) | `prepare.py` |
| Chip file | `chips/<chip_id>.npz` | | `prepare.py` |
| Predicted mask | `<chip_id>.npz` with key `mask` (also accepted: `.npy`, `.tif`); 1 = mangrove | | `chipio.py` |
| GMW label raster | `gmw_v4019_2020_uae.tif`, `gmw_v4112_<year>_uae.tif` | `gmw_v4112_2018_uae.tif` | `prepare.py` |
| Land-cover window | `landcover/raw/wc<year>_<site>.tif` | | `landcover.py` |
| Land cover per chip | `landcover/<chip_id>.npz` | | `landcover.py` |
| U-Net weights | `unet_fold<k>.pt` | `unet_fold1.pt` | `train_unet.py` |
| Earth Search image ID | as given by the archive | `S2A_T40RBN_20201117T070153_L2A` | pipeline README |
| Seeing Orbit chip | `patch_<5 digits>.npz` | `patch_00010.npz` | test-chip zip |
| Hand label | `<code>_<year>_<number>` | `KAB_2020_07` | see below |

**`KAB_2020_07`.** No file in the repository defines this convention. It does not follow the pipeline's chip ID pattern. From the commit message, `KAB` stands for Khor Al Beida and `2020` is the year. **What `07` means is UNVERIFIED**: it could be a chip number or the month, and nothing here says which.

**Folder names.** The code expects to live in a project folder laid out like this (`paths.py`):

| Folder the code expects | Holds | Where it is in this repository |
| --- | --- | --- |
| `2_Code/` | Scripts | `src/mangrove_segmentation/` |
| `3_Data/` | Small inputs: sites, labels, image list, folds, haze table | not present |
| `4_Results/` | Tables, figures, reports | `results/mangrove_segmentation/` (part of it) |
| `6_Work/` | Everything large and rebuildable: raw images, chips, datasets, land cover, model weights | not present, by design |

The project folder can be set with the environment variable `MANGROVE_ROOT`.

## 10. The Seeing Orbit test chips

Everything in this section is from the test-chip zip and the U-Net report. This set was made by different code, which is not in the repository.

| Fact | Value | Source |
| --- | --- | --- |
| Contents | 10 chips, `manifest_test.csv`, `README.txt`, `SHA256SUMS.txt`, `SOURCE_ATTRIBUTION.txt` | zip |
| Purpose | Held-out inputs for a fair model comparison. The reference masks are kept private | zip `README.txt` |
| Area and images | Eastern Mangroves; `S2A_40RBN_20181105_1_L2A` (5 Nov 2018) and `S2C_40RBN_20251101_0_L2A` (1 Nov 2025) | U-Net report |
| Chip arrays | `image`: float32, 6 × 256 × 256. `valid`: uint8, 256 × 256, score only where 1 | zip `README.txt`, confirmed on the arrays |
| Channel order | 0 blue, 1 green, 2 red, 3 near-infrared, 4 NDVI, 5 NDWI | zip `README.txt` |
| NDWI formula | (green − NIR) / (green + NIR) | U-Net report |
| Value ranges | Optical 0 to 1, indices −1 to 1 | zip `README.txt`, confirmed on the arrays |
| Reference label | GMW v4.1.12 | zip `SOURCE_ATTRIBUTION.txt`, U-Net report |
| Grid | 828 × 902 pixels, 10 m, WGS 84 / UTM zone 40N | U-Net report |
| Split sizes | 38 training, 8 validation, 10 test patches | U-Net report |

`manifest_test.csv` columns:

| Column | Meaning |
| --- | --- |
| `patch_id` | Chip number (10, 15, … 55) |
| `filename` | `patch_<5 digits>.npz` |
| `role` | `before` (2018) or `after` (2025) |
| `date` | `2018-11-05` or `2025-11-01` |
| `row`, `column` | Position on the analysis grid in pixels. Rows step by 128, so neighbouring chips overlap by half. All ten have column 512 |
| `split` | `test` for all ten |
| `valid_fraction` | Share of valid pixels (0.91–1.00) |

The image IDs above are written in a different style from the main pipeline's (`S2A_40RBN_20181105_1_L2A` against `S2A_T40RBN_20201117T070153_L2A`), which suggests they came from a different collection of the archive. Which collection these chips came from, and how their reflectance was scaled, are **UNVERIFIED**.

A third, separate set of images was used for the registration work: Copernicus Browser exports of Al Jubail Island for 10 Dec 2018 and 10 Dec 2025, bands B02, B03, B04 and B08, 1836 × 1096 pixels, EPSG:32640 (registration README). Those files are not in the repository either.

## 11. Where code and documents disagree

| Topic | What the code does | What a document says | Document |
| --- | --- | --- | --- |
| CRS | EPSG:32640 everywhere | Nothing in the repository says 32639 | none found |
| Reflectance offset | Scale and offset applied once, never clipped | Scale factors "still need to be recorded" | data README |
| Reflectance in the Seeing Orbit chips | Not made by this code | Optical values clipped at 0; cause not shown | test-chip zip, pipeline README |
| GMW version | Stores v4.0.19 and v4.1.12; trains and scores on v4.0.19 | v4.1.12 only | test-chip zip, U-Net report |
| GMW version | as above | "GMW v4.1" | Meeting 6 minutes |
| GMW version | as above | "Global Mangrove Watch reference maps", no version | data README |
| Water index | MNDWI = (green − SWIR1) / (green + SWIR1) | NDWI = (green − NIR) / (green + NIR) | U-Net report, test-chip zip; `docs/PROGRESS.md` and `docs/PROJECT_PLAN.md` also say NDWI |
| Channels per chip | 8 | 6 | test-chip zip, U-Net report |
| Catalog column name | `bounds_epsg32640` | `bounds` | pipeline README |
| Number of sites | 13 | Eastern Mangroves, with Jubail and Reem Island as possible additions | data README |
| Train/test division | 5 site folds plus locked test sites | "Choose thresholds on `--split val` only" | pipeline README (older rule, still valid for the `split` column) |
| Coarse label pixel size | Not checked by the code | about 30 m in one place, 25 m in another | pipeline README; Meeting 5 minutes |
| Checksums of the exports | No checksum file exists in the repository | Checksums "recorded in the data documentation" | `docs/PROGRESS.md` |
| Folder layout | `2_Code`, `3_Data`, `4_Results`, `6_Work` | `src/`, `data/`, `results/` | the repository itself |

## 12. Everything marked UNVERIFIED

1. The contents of `catalog.csv`, `raw_manifest.csv`, `norm.json`, `norm_trainval_2020.json`, `folds_site_cv.csv`, `QA_download.csv`, `scenes_selected.csv` and `sites.geojson`. None is in the repository; their layout is known only from the code.
2. The exact dates and image IDs of the 104 images.
3. The data names and boundaries of the seven locked sites, and their chip counts.
4. Which sites have which `proposed_split`.
5. The true pixel size, CRS and download source of the two GMW label rasters, and whether the cited Zenodo link is correct.
6. Whether any project file outside this pipeline is in EPSG:32639.
7. Why the Seeing Orbit chips have optical values clipped at 0, and which archive collection and scaling they used.
8. What `07` means in `KAB_2020_07`, and which image the hand label was drawn on.
9. The mean and standard deviation values actually stored in the two normalisation files.
10. Whether the code in this repository is the exact version that produced the files in `results/`. The scripts that ran the baselines' cross-validation (`run_cv.py`, `score.py`) are not here.
