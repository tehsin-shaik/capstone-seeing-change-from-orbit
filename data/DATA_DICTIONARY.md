# Data Dictionary

A reference to the datasets, files, columns, spectral bands and labels used in this project, including their meanings and sources.

## How to read this document

- **Source** names the file each fact comes from, using the names in the table below.
- Limitations that affect interpretation or reproducibility are listed in [section 9](#9-data-limitations-and-unavailable-metadata).
- Dataset-wide values describe what the pipeline code does. Values checked directly on the Train_Val dataset or the hand-label files are marked as such.

### Sources

| Cited as | File |
| --- | --- |
| data README | [data/README.md](README.md) |
| hand labels | [data/hand_labels/README.md](hand_labels/README.md) and the label files, read with `ogrinfo` |
| landmark write-up | [docs/LANDMARK_DETECTION.md](../docs/LANDMARK_DETECTION.md) |
| baselines RESULTS | [results/mangrove_segmentation/baselines/RESULTS.md](../results/mangrove_segmentation/baselines/RESULTS.md) |
| baselines cv_results | [results/mangrove_segmentation/baselines/cv_results.csv](../results/mangrove_segmentation/baselines/cv_results.csv) |
| hand-label results | [results/mangrove_segmentation/hand_labels/](../results/mangrove_segmentation/hand_labels/RESULTS.md) |
| pipeline README | [src/mangrove_segmentation/README.md](../src/mangrove_segmentation/README.md) |
| `baseline_ndvi.py` | [src/mangrove_segmentation/baseline_ndvi.py](../src/mangrove_segmentation/baseline_ndvi.py) |
| `baseline_rf.py` | [src/mangrove_segmentation/baseline_rf.py](../src/mangrove_segmentation/baseline_rf.py) |
| `baseline_svm.py` | [src/mangrove_segmentation/baseline_svm.py](../src/mangrove_segmentation/baseline_svm.py) |
| `chipio.py` | [src/mangrove_segmentation/chipio.py](../src/mangrove_segmentation/chipio.py) |
| `collect.py` | [src/mangrove_segmentation/collect.py](../src/mangrove_segmentation/collect.py) |
| `evaluate.py` | [src/mangrove_segmentation/evaluate.py](../src/mangrove_segmentation/evaluate.py) |
| `label_agreement.py` | [src/mangrove_segmentation/label_agreement.py](../src/mangrove_segmentation/label_agreement.py) |
| `landcover.py` | [src/mangrove_segmentation/landcover.py](../src/mangrove_segmentation/landcover.py) |
| `make_datasets.py` | [src/mangrove_segmentation/make_datasets.py](../src/mangrove_segmentation/make_datasets.py) |
| `paths.py` | [src/mangrove_segmentation/paths.py](../src/mangrove_segmentation/paths.py) |
| `prepare.py` | [src/mangrove_segmentation/prepare.py](../src/mangrove_segmentation/prepare.py) |
| `run_all.py` | [src/mangrove_segmentation/run_all.py](../src/mangrove_segmentation/run_all.py) |
| `run_cv.py` | [src/mangrove_segmentation/supporting_scripts/baselines/run_cv.py](../src/mangrove_segmentation/supporting_scripts/baselines/run_cv.py) |
| `dinov3/train_dinov3.py` | [src/mangrove_segmentation/dinov3/train_dinov3.py](../src/mangrove_segmentation/dinov3/train_dinov3.py) |
| `dinov3/train_dinov3_lora.py` | [src/mangrove_segmentation/dinov3/train_dinov3_lora.py](../src/mangrove_segmentation/dinov3/train_dinov3_lora.py) |
| `train_unet.py` | [src/mangrove_segmentation/train_unet.py](../src/mangrove_segmentation/train_unet.py) |
| notebook 00 / 01 / 02 | [src/mangrove_segmentation/notebooks/](../src/mangrove_segmentation/notebooks/) |

## 1. Sites

The pipeline uses **13 UAE sites** (notebook 00: "104 Sentinel-2 images (13 UAE sites × 8 dates)").

| Site name in the data | Role | Fold | 2020 chips | Source |
| --- | --- | --- | --- | --- |
| `eastern_mangroves` | cross-validation | 1 | 8 | baselines cv_results; `make_datasets.py` |
| `jubail_island` | cross-validation | 2 | 80 | same |
| `ras_ghurab` | cross-validation | 3 | 50 | same |
| `ghantoot` | cross-validation | 4 | 41 | same |
| `al_aryam_islands` | cross-validation | 5 | 36 together | same |
| `al_nouf` | cross-validation | 5 | (with Al Aryam) | same |
| Abu Al Abyad | fixed test, **locked** | none | not described (locked) | `make_datasets.py` (text of `README_LOCKED`) |
| Marawah | fixed test, **locked** | none | not described (locked) | same |
| Ras Al Khor | external, **locked** | none | not described (locked) | same |
| Al Zorah | external, **locked** | none | not described (locked) | same |
| Khor Al Beida | external, **locked** | none | not described (locked) | same |
| Ras Al Khaimah creek | external, **locked** | none | not described (locked) | same |
| Khor Kalba | external, **locked** | none | not described (locked) | same |

- The six cross-validation sites are in Abu Dhabi and hold 215 chips from 2020 (baselines RESULTS). Al Aryam and Al Nouf are 1.5 km apart and share fold 5 (notebook 02).
- "Locked" means the test data must not be visually inspected, explored, used for training, or used for model selection or hyperparameter tuning. Predictions are evaluated against the reference labels once, after the model and evaluation procedure have been finalised (`make_datasets.py`).
- The locked sites' chips are in the locked `Test_Dataset`, so their dates, chip counts and reference data are not described here (section 9).
- Site boundaries come from `3_Data/sites/sites.geojson`, a longitude/latitude (EPSG:4326) file with one feature per site, a `name` property and a `proposed_split` property (`collect.py`, `prepare.py`). Each chip's position is recorded in the `bounds_epsg32640` column of the catalog.
- The fold and role of each site come from `folds_site_cv.csv` (columns `site`, `fold`, `role`). The copy in the Train_Val dataset lists the six cross-validation sites with the folds above, all with role `cv`.

## 2. Satellite imagery

| Fact | Value | Source |
| --- | --- | --- |
| Satellite and product | Sentinel-2 Level-2A surface reflectance | `collect.py` |
| Where it is downloaded from | Earth Search v1 STAC, `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-c1-l2a` (public files on AWS, no login) | `collect.py` |
| Images per site | 8 dates selected per site: four in 2020, March and November of 2018 and 2025, each under 10% scene-level cloud. A selected acquisition contributes chips to Train_Val only if at least one chip passes the 5% chip-level cloud threshold. Al Nouf has chips from only seven dates in Train_Val, because no chips from its 16 November 2025 acquisition passed that threshold | notebook 00; Train_Val `catalog.csv` |
| Images used | Each chip's source image is recorded in the `item_id` column of the catalog; the dates are listed below | Train_Val `catalog.csv` |
| Example image ID | `S2A_T40RBN_20201117T070153_L2A` | pipeline README |
| Pixel size | 10 m | `collect.py` (`RES = 10.0`) |
| Total size | about 2.4 GB of raw images | notebook 00, `run_all.py` |

### Acquisition dates in Train_Val

| Site | 2018 | 2020 | 2025 |
| --- | --- | --- | --- |
| `eastern_mangroves` | 2018-03-18, 2018-11-23 | 2020-02-16, 2020-05-18, 2020-08-24, 2020-11-17 | 2025-03-18, 2025-11-16 |
| `jubail_island` | 2018-03-18, 2018-11-23 | 2020-02-16, 2020-05-18, 2020-08-24, 2020-11-17 | 2025-03-18, 2025-11-16 |
| `ras_ghurab` | 2018-03-18, 2018-11-23 | 2020-02-16, 2020-05-18, 2020-08-24, 2020-11-17 | 2025-03-18, 2025-11-16 |
| `ghantoot` | 2018-03-20, 2018-11-05 | 2020-02-23, 2020-05-18, 2020-08-31, 2020-11-09 | 2025-03-18, 2025-11-13 |
| `al_aryam_islands` | 2018-03-08, 2018-11-23 | 2020-01-17, 2020-04-01, 2020-09-23, 2020-11-12 | 2025-03-31, 2025-11-01 |
| `al_nouf` | 2018-03-18, 2018-11-23 | 2020-02-16, 2020-05-16, 2020-08-19, 2020-11-17 | 2025-03-18 |

Source: Train_Val `catalog.csv`.

### Chip channels

Each chip has 8 channels: the 6 Sentinel-2 bands downloaded by `collect.py` (positions 0 to 5) and 2 indices calculated from them by `prepare.py` (positions 6 and 7). The random forest, the SVM, the U-Net and the spectral branch of the DINOv3 models use all 8. The DINOv3 backbone itself sees only red, green and blue, and the NDVI baseline uses only NDVI and MNDWI.

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

All six spectral bands are warped onto the 10 m grid with bilinear resampling, including the 10 m bands. For images from UTM zone 40 tiles the 10 m pixels line up with the output grid (both sit on whole 10 m coordinates), so their values are effectively unchanged. This was checked on one zone 40 chip (`al_nouf_2018-11-23_r00000_c00768`, tile 40QBM): all 65,536 blue values equal `DN × 0.0001 − 0.1` of the source tile to within 0.00000002. Images from zone 39 tiles are reprojected, so their values are interpolated.

The **SCL** band (Sentinel-2's own scene classification) is also downloaded, with nearest-neighbour resampling. It is used for the cloud mask and is not a model input (`collect.py`).

### Raw image files

Written by `collect.py`, one per site and date:

- Path: `raw/<site>/<YYYY-MM-DD>.tif`
- 7 bands, 32-bit float, NaN = no data. Bands 1 to 6 are B02, B03, B04, B08, B11, B12 as reflectance. Band 7 is SCL.
- A pixel is no data if any of the six bands, or SCL, has a raw value (DN) of 0 in the source file. This is tested before the scale and offset are applied, so a DN of 0 means no data, while a reflectance of exactly 0 (DN 1000) is a valid value.
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
| `offset_applied` | Which scale and offset were applied, and where they came from (see section 3) |
| `path` | File path relative to the output folder |
| `bounds_epsg32640` | Site box as `minx,miny,maxx,maxy` in metres, EPSG:32640 |
| `valid_fraction` | Share of the site's pixels that have data (0 to 1) |
| `masked_fraction` | Share of the pixels with data that SCL marks as cloud or shadow (0 to 1) |

### Registration images

A separate set of images was used for the registration work: Copernicus Browser exports of Al Jubail Island for 10 Dec 2018 and 10 Dec 2025, bands B02, B03, B04 and B08, 1836 × 1096 pixels, EPSG:32640 (landmark write-up). Those files are not in the repository either.

## 3. Reflectance scaling

Source: `collect.py`.

- Sentinel-2 files store whole numbers (DN). Each band of each image carries its own `scale` and `offset` in its metadata, and `collect.py` converts each value once: `reflectance = DN × scale + offset`.
- If an image has no such metadata, the code falls back on the processing baseline: 04.00 or later gets offset −0.1, older gets 0.
- The rule used is recorded per image in the `offset_applied` column and in the GeoTIFF's tags. In the Train_Val catalog, all 454 chip entries (from 47 site-and-date images, each worked out from its own metadata) record `stac_raster_bands:scale=0.0001,offset=-0.1`, so reflectance is `DN × 0.0001 − 0.1` throughout and the fallback was not used.
- No later step changes reflectance; `prepare.py` only reads it. Values are not clipped, so dark water can be slightly negative (about 14% of red-band pixels in the 2020 test scene, pipeline README). Only NDVI and MNDWI are clipped, to −1…1.

## 4. Coordinate system (CRS)

**What the code does: EPSG:32640 (WGS 84 / UTM zone 40N), 10 m pixels, for every site.**

| Where | What it does | Source |
| --- | --- | --- |
| Download | Warps every image to EPSG:32640 at 10 m, on a grid snapped outward to whole 10 m steps | `collect.py` (`DST_CRS`, `site_grid`) |
| Chips | Each chip stores `crs = "EPSG:32640"` and its own 6-number position (`transform`). The CRS text is written as a fixed string, not read from the image | `prepare.py` |
| Catalog | Chip box stored in the column `bounds_epsg32640` | `prepare.py` |
| Land cover | Warped onto each chip's EPSG:32640 grid | `landcover.py` |
| Hand labels | Every label file is in EPSG:32640 | hand labels (`ogrinfo`) |
| Registration images | EPSG:32640 | landmark write-up |
| Inputs in other systems | `sites.geojson` and ESA WorldCover are longitude/latitude (EPSG:4326). The GMW label rasters are read in whatever CRS they carry (a code comment says EPSG:4326) and warped to the chip grid | `collect.py`, `landcover.py`, `prepare.py` |

## 5. Chips

A chip is a 256 × 256 pixel square (2.56 km) cut from one raw image. Written by `prepare.py`.

### How chips are cut and chosen

- Squares do not overlap and start at the top-left corner of the image. Leftover strips at the right and bottom edges are not used.
- A chip is dropped if more than 5% of its pixels with data are cloud or shadow, or more than 10% of all its pixels are no data. The cloud share leaves no-data pixels out of the count (`prepare.py`, `chip_stats`).
- A chip counts as a mangrove chip if at least 1% of it is mangrove under **either** label.
- Background chips (less than 1% mangrove under both labels) are sampled at random (seed 42), pooled over all dates of a site, so that they make up about 20% of that site's kept chips: the number taken is the mangrove-chip count × 0.20 / 0.80, rounded. Rounding can push a small site slightly above 20% (Al Nouf 22%), and a site with fewer background candidates than that keeps all of them (Jubail Island 12%).
- The same place appears once per date.

### Chip file contents

Path: `chips/<chip_id>.npz` (compressed NumPy). Source: `prepare.py`.

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `image` | float32, 8 × 256 × 256 | The 8 channels of section 2, in that order. Reflectance, NaN = no data |
| `label_v4019` | uint8, 256 × 256 | GMW v4.0.19 label: 1 mangrove, 0 not mangrove, 255 no label |
| `label_v4112` | uint8, 256 × 256 | GMW v4.1.12 label, same coding |
| `cloud` | bool, 256 × 256 | True where SCL is 3, 8, 9 or 10 (cloud shadow, medium cloud, high cloud, thin cirrus) |
| `nodata` | bool, 256 × 256 | True where any band is NaN or SCL is 0 |
| `transform` | float64, 6 numbers | Position of the chip on the ground (affine a, b, c, d, e, f) in EPSG:32640 |
| `crs` | text | `"EPSG:32640"` |
| `channels` | text, 8 names | Channel names, in order |

**Usable pixel.** Every model and the scoring code use only pixels that are finite in all 8 channels, not cloud, not no data, and have a label other than 255 (`chipio.py`, `evaluate.py`). Training uses the same mask. The random forest (`baseline_rf.py`), the SVM (`baseline_svm.py`), the U-Net (`train_unet.py`), DINOv3 (`dinov3/train_dinov3.py`, whose data rules `dinov3/train_dinov3_lora.py` reuses) and the scoring code (`evaluate.py`, `supporting_scripts/`) all combine `valid` from `chipio.py` with the label's 255 mask, so cloud and no-data pixels are excluded in the same way in training and evaluation.

### `catalog.csv` columns

One row per kept chip. The file itself is not in the repository; the 17 columns below are exactly the ones `prepare.py` writes, in this order.

| # | Column | Meaning | How it is filled |
| --- | --- | --- | --- |
| 1 | `chip_id` | Unique chip name | `<site>_<date>_r<row>_c<col>`, row and col padded to 5 digits |
| 2 | `site` | Site name | From the manifest |
| 3 | `date` | Image date | `YYYY-MM-DD` |
| 4 | `year` | Image year | First 4 characters of `date` |
| 5 | `split` | Site-level split | `proposed_split` of the site in `sites.geojson`, or `unassigned`. Can be overridden on the command line. See the note below |
| 6 | `row` | Top edge of the chip, in pixels from the top of the raw image | Multiple of 256 |
| 7 | `col` | Left edge of the chip, in pixels from the left of the raw image | Multiple of 256 |
| 8 | `bounds_epsg32640` | Chip box, `minx,miny,maxx,maxy`, metres, EPSG:32640 | Rounded to whole metres |
| 9 | `cloud_pct` | Percent of the chip's pixels with data that are cloud or shadow | 2 decimals |
| 10 | `nodata_pct` | Percent of all the chip's pixels that are no data | 2 decimals |
| 11 | `mangrove_pct_v4019` | Percent mangrove under GMW v4.0.19, among pixels that have data and a label | 2 decimals. Empty when there is no v4.0.19 label for that year (every year except 2020) |
| 12 | `mangrove_pct_v4112` | Same for GMW v4.1.12 | Empty when there is no label for that year |
| 13 | `kind` | `mangrove` or `background` | `mangrove` if at least 1% mangrove under either label |
| 14 | `offset_applied` | Reflectance rule used for the source image | Copied from the manifest (section 3) |
| 15 | `item_id` | Earth Search image ID(s) | Copied from the manifest |
| 16 | `source_file` | Raw image the chip was cut from | Path relative to the manifest's folder |
| 17 | `path` | Chip file | `chips/<chip_id>.npz`, relative to the output folder |

Notes:

- `chipio.py` needs only `chip_id`, `site` and `split` to load a catalog.
- **`split` and the reported results.** The baseline `fit` functions train on chips whose `split` is `train` and tune on chips whose `split` is `val`. For the reported cross-validation results, the scripts write a temporary catalog for each fold and set `split` from the folds (`run_cv.py`, `src/change_detection/common.py`), so those results follow `folds_site_cv.csv`, not the stored `split` values. In the Train_Val catalog, `split` is `train` for Eastern Mangroves, Jubail Island, Ras Ghurab and Ghantoot, and `val` for Al Aryam Islands and Al Nouf.
- `make_datasets.py` writes a second catalog inside each dataset folder (`Train_Val_Dataset`, `Test_Dataset/fixed_test`, `Test_Dataset/external`). It has the same 17 columns plus two more, and `path` is rewritten to `chips/<file name>`:

| Extra column | Meaning |
| --- | --- |
| `role` | `cv`, `test_fixed` or `external_fixed`, from `folds_site_cv.csv` |
| `fold` | Fold number 1 to 5 for cross-validation sites, from the same file |

- At run time the training code adds `water_b04_p5`, `hazy`, `x0`, `y0`, `x1`, `y1`. These are computed in memory and not saved in the catalog (`train_unet.py`).

## 6. Normalisation files

Normalising means subtracting a mean and dividing by a standard deviation per channel so that all channels are on a similar scale. There are two files, and the deep-learning code uses neither.

| File | Written by | Computed from | Keys |
| --- | --- | --- | --- |
| `norm.json` | `prepare.py` | Chips whose `split` is `train`; pixels that have data, are not cloud, and are finite | `channels` (8 names), `mean` (8 numbers), `std` (8 numbers), `n_pixels` (8 counts, one per channel), `computed_from` (text) |
| `norm_trainval_2020.json` | `make_datasets.py` | All 2020 chips of the Train/Val dataset, all 5 folds; pixels finite in all 8 channels. The cloud mask is **not** applied here, so cloudy pixels are included | `channels`, `mean`, `std`, `n_pixels` (one count), `source` (text) |

- Values are rounded to 6 decimals. If there are no `train` chips, `norm.json` has `mean` and `std` set to null and a `note`.
- How to apply: `x = (image − mean) / std`, channel by channel (pipeline README).
- **What the models do:** the U-Net computes its own mean and standard deviation inside each fold, from that fold's training chips only (`train_unet.py`, `fold_stats`). It does not read either file. The random forest and the NDVI rule use raw values with no normalisation (`baseline_rf.py`, `baseline_ndvi.py`).
- `norm_trainval_2020.json` is part of the Train_Val dataset (computed from 14,088,151 pixels). `norm.json` is an intermediate `prepare.py` output that no reported model reads.

## 7. Labels and other reference data

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

- **Native resolution vs chip grid.** Both labels are stored on the 10 m chip grid. Nearest-neighbour warping assigns each output pixel on the 10 m grid the value of its nearest source-label pixel. This preserves the categorical values (0, 1 and 255), but does not add spatial detail. The v4.1.12 labels therefore retain their coarser native boundaries; they should not be interpreted as independently verified 10 m reference boundaries.
- **Evidence from the chips.** In every row and every column of a chip, the label was split into runs of consecutive pixels of the same class (mangrove or non-mangrove). Only runs with a class change on both sides were counted; runs touching the chip edge were left out. Over all 215 Train_Val chips from 2020, 0.5% of the v4.1.12 runs are a single pixel (10 m) long, against 18.2% for v4.0.19 (241,110 and 393,022 runs). A label with roughly 30 m pixels rarely produces a feature only 10 m wide.
- **Training and evaluation reference.** GMW v4.0.19 is the primary reference label used by the project. It is the default label for both baseline models, is explicitly selected in `train_unet.py`, and is used by `evaluate.py`. Consequently, the reported model results are based on 2020 chips, the only year covered by v4.0.19. GMW v4.1.12 is retained alongside it for the 2018 and 2025 datasets, primarily for temporal comparison and mangrove change mapping.
- **Differences between versions.** The two GMW versions do not produce identical mangrove boundaries. They should therefore be treated as separate reference datasets rather than interchangeable labels. Results obtained using one version should not be directly compared with results using the other without accounting for differences in their coverage, resolution and classification.
- **Model-generated labels.** GMW labels are generated through a remote-sensing classification process rather than manually delineated by project team members. They represent model-derived mangrove classifications and may contain classification errors or boundary inaccuracies.
- **Source data.** The original GMW raster files are stored outside the repository. GMW v4.0.19 is available from [Zenodo](https://doi.org/10.5281/zenodo.12756047) and provides a 10 m mangrove map for 2020 derived from Sentinel-2 imagery. GMW v4.1.12 is available from [Zenodo](https://zenodo.org/records/21346457) and provides an annual mangrove time series derived from Landsat and Sentinel-2 imagery. The v4.1.12 label resolution is approximately 30 m, as documented in the project pipeline.

### Hand labels

`data/hand_labels/` holds mangrove outlines drawn independently by team members, with one folder per labeller. Each labeller drew the same 6 Train_Val chips from 2020: Eastern Mangroves, Ghantoot and Ras Ghurab (18 May), and Jubail Island (two chips on 16 Feb, one on 18 May). Each file has a `mangrove` and an `unsure` polygon layer, in EPSG:32640. Polygon counts, areas and the rules for using the labels are in [data/hand_labels/README.md](hand_labels/README.md).

Because these chips belong to Train_Val, model predictions are taken from out-of-fold (OOF) evaluation. For each chip, the random forest prediction comes from a model trained without that chip's entire fold. The U-Net prediction comes from the model corresponding to the same held-out fold, which was also trained without that entire fold (`train_unet.py`). This ensures that neither model was trained on the chip being evaluated and that both predictions correspond to the same held-out spatial split. All six hand-labelled chips have OOF predictions from both models (`label_agreement.py`, `results/mangrove_segmentation/hand_labels/per_chip.csv`).

### ESA WorldCover (land cover)

WorldCover is used as contextual reference data for error analysis (errors broken down by land-cover type). It is not a model input or a training label. Source: `landcover.py`.

- Chips from before 2021 use the 2020 map (v100); chips from 2021 or later use the 2021 map (v200).
- Stored as `landcover/<chip_id>.npz`, key `lc`, one uint8 WorldCover class code per pixel. The two codes used most in this project are:

| Code | Class |
| --- | --- |
| 95 | Mangroves |
| 80 | Permanent water bodies |

### Haze flag

A scene is "hazy" if the 5th percentile of red (B04) over water is above 0.05. The value is in the column `water_b04_p5` of `3_Data/QA_download.csv`, by site and date (`make_datasets.py`, `train_unet.py`). The file is not in the repository.

The fifth percentile represents the lower end of the red-band reflectance distribution over water. A value above 0.05 indicates elevated red reflectance in pixels expected to be dark, and is used as an indicator of possible haze. This is a screening flag rather than a direct measurement of atmospheric haze.

## 8. File and naming conventions

| Thing | Pattern | Example | Source |
| --- | --- | --- | --- |
| Site name | lower case, underscores | `eastern_mangroves` | baselines cv_results |
| Raw image | `raw/<site>/<YYYY-MM-DD>.tif` | `raw/eastern_mangroves/2020-11-17.tif` | `collect.py` |
| Chip ID | `<site>_<YYYY-MM-DD>_r<row, 5 digits>_c<col, 5 digits>` | `eastern_mangroves_2020-11-17_r00000_c00256` | `prepare.py` |
| Chip file | `chips/<chip_id>.npz` | `chips/eastern_mangroves_2020-11-17_r00000_c00256.npz` | `prepare.py` |
| Predicted mask | `<chip_id>.npz` with key `mask` (also accepts `.npy` and `.tif`); 1 = mangrove, 0 = non-mangrove | `eastern_mangroves_2020-11-17_r00000_c00256.npz` | `chipio.py` |
| GMW label raster | `gmw_v4019_2020_uae.tif`, `gmw_v4112_<year>_uae.tif` | `gmw_v4112_2018_uae.tif` | `prepare.py` |
| Land-cover window | `landcover/raw/wc<year>_<site>.tif` | `landcover/raw/wc2020_eastern_mangroves.tif` | `landcover.py` |
| Land cover per chip | `landcover/<chip_id>.npz` | `landcover/eastern_mangroves_2020-11-17_r00000_c00256.npz` | `landcover.py` |
| U-Net weights | `unet_fold<k>.pt` | `unet_fold1.pt` | `train_unet.py` |
| Earth Search image ID | as given by the archive | `S2A_T40RBN_20201117T070153_L2A` | pipeline README |

Examples show the format only; they are not claims that those exact files exist.

Chip row and column values identify the chip's position in the source image grid. They are zero-based pixel offsets of the chip's top-left corner, formatted as five digits, rather than geographic coordinates. Each chip covers a 256 × 256 pixel window.

Predicted masks are read as true/false: any non-zero value counts as mangrove and 0 as non-mangrove. In an `.npz` file without a `mask` key, the first array is used (`chipio.py`).

**Folder names.** The code expects to live in a project folder laid out like this (`paths.py`). The expected folder names are inherited from the original project layout; the repository uses the corresponding paths shown in the final column.

| Folder the code expects | Holds | Where it is in this repository |
| --- | --- | --- |
| `2_Code/` | Scripts | `src/mangrove_segmentation/` |
| `3_Data/` | Small inputs: sites, labels, image list, folds, haze table | not present |
| `4_Results/` | Tables, figures, reports | `results/mangrove_segmentation/` (part of it) |
| `6_Work/` | Large or reproducible data and model artefacts: raw images, chips, datasets, land cover, model weights | not present, by design |

The project root is configured through the `MANGROVE_ROOT` environment variable in `paths.py`. Scripts that import `paths.py` (`run_all.py`, `make_datasets.py`, `landcover.py`, `train_unet.py`, `baseline_svm.py`, the DINOv3 scripts and the baseline and U-Net scoring scripts) follow it. `collect.py`, `prepare.py`, `evaluate.py` and `label_agreement.py` take their folders as command-line arguments instead.

### Dataset directory contents

`make_datasets.py` packages the chips into two dataset folders. The Train/Val dataset:

```text
Train_Val_Dataset/
├── catalog.csv              one row per chip (section 5)
├── chips/                   one .npz file per chip (section 5)
├── folds_site_cv.csv        site, fold and role (section 1)
├── hazy_scenes.csv          haze flag per site and date (below)
├── landcover/               WorldCover class per chip (section 7)
├── norm_trainval_2020.json  per-channel mean and std (section 6)
└── README.md                short description of the dataset
```

The shared Train_Val copy includes an empty `landcover/` directory. Land-cover files were generated separately in the full working dataset and are not included in this packaged copy.

`hazy_scenes.csv` columns:

| Column | Meaning |
| --- | --- |
| `site` | Site name |
| `date` | Acquisition date, `YYYY-MM-DD` |
| `water_b04_p5` | Fifth percentile of B04 reflectance over water |
| `hazy` | Haze flag, `True` when `water_b04_p5` is above 0.05 (section 7) |

`hazy_scenes.csv` holds scene-level metadata, with one record per site and date (48 records). `catalog.csv` holds chip-level metadata, with one record per chip (454 entries). The haze table also includes acquisitions that contributed no chips: Al Nouf's 16 November 2025 acquisition has a haze record but no chips, because none of its chips passed the chip-level cloud threshold.

The locked test dataset has the same structure, split into the fixed test and external sites. Its contents are not described here (section 9).

```text
Test_Dataset/
├── fixed_test/
│   ├── catalog.csv
│   ├── chips/
│   └── landcover/
├── external/
│   ├── catalog.csv
│   ├── chips/
│   └── landcover/
└── README_LOCKED.md
```

## 9. Data limitations and unavailable metadata

1. **Locked test sites.** The locked test dataset contains additional site-level imagery and reference labels. The site names are listed in section 1, as they are in `make_datasets.py`. Their boundaries, chip counts and reference data are withheld to preserve the independence of the final evaluation. Test-set statistics are therefore not reported here.
2. **Site boundaries and scene selection.** The `sites.geojson` and `scenes_selected.csv` files are not included in the repository. The Train_Val catalog retains chip positions and source-image identifiers, but reconstructing the complete image-selection and chip-generation process requires these original inputs.
3. **GMW label rasters.** The original GMW raster files are not included in the repository. They must be obtained from the source records listed in section 7 before the label-preparation stage can be reproduced.
4. **Haze metadata.** The `water_b04_p5` values used to assign the haze flag are stored in `hazy_scenes.csv` within the Train_Val dataset. The original `QA_download.csv` file and the script used to calculate these values are not included in the repository, so the haze calculations cannot be independently regenerated from the supplied files.
