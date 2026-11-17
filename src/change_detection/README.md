# Change detection: first model-based change map

These scripts make a mangrove change map for one training site, 2018 to 2025, with the random forest baseline, and check how much of the "change" is model noise. The results are in [results/change_detection/RESULTS.md](../../results/change_detection/RESULTS.md).

## What the scripts do

| Script | What it does |
|---|---|
| `inventory.py` | Lists which sites have chips in both 2018 and 2025 on the same chip grid, which scenes are hazy and which labels the chips carry. |
| `train_rf.py` | Trains the random forest with one site held out, so the model has never seen the site. |
| `change_map.py` | Predicts that site's 2018 and 2025 chips, makes one mask per year, builds the change map, runs the sanity checks and draws the figure. |
| `noise_floor_all_sites.py` | Repeats the same-year check for every site, each with the forest that had it held out. |
| `common.py` | Shared helpers. Not run on its own. |

## What they depend on

- `baseline_rf.py` and `chipio.py` in `src/mangrove_segmentation`. They are imported unchanged. Nothing is copied.
- The team's script `run_cv.py`, which ran the baseline cross-validation. It is not in this repository. The scripts here do not need that file to run. Its steps for one fold are written out again in `common.py` (`train_fold` and `drop_near`): the 2020 chips, the GMW v4.0.19 label, hazy scenes dropped from training, the one-chip buffer, tuning on an inner fold with the threshold grid widened to 0.95, and the refit on all four training folds with seed 0.
- To check that this copy of the steps is faithful, each forest is scored on its held-out 2020 chips. For all five folds the IoU matches the B2 row in `results/mangrove_segmentation/baselines/cv_results.csv` to four decimals. If `run_cv.py` changes, this code will not follow it.
- The Train/Val dataset, which is not in the repository. The scripts read `catalog.csv`, `hazy_scenes.csv` and `chips/` from it. They never read the test datasets.
- The Python packages of `src/mangrove_segmentation/environment.yml`. `rasterio` is only needed for the GeoTIFF copy of the map. The published forests were reproduced with scikit-learn 1.9.0.

## How to run

Run from `src/change_detection`. Everything runs on a CPU in a few minutes.

```
python inventory.py             --train-val <Train_Val_Dataset> --out <output folder>
python train_rf.py              --train-val <Train_Val_Dataset> --out <output folder> --site jubail_island
python change_map.py            --train-val <Train_Val_Dataset> --out <output folder> --site jubail_island
python noise_floor_all_sites.py --train-val <Train_Val_Dataset> --out <output folder>
```

- `--site` is the site name as written in the catalog, for example `ghantoot`. The default is `jubail_island`. `train_rf.py` must be run for a site before `change_map.py`.
- `--variant a` keeps hazy scenes in training. The default is `b`, which drops them, as in the reported baseline.
- `--qa` gives another haze table. The default is `hazy_scenes.csv` inside the Train/Val folder.

## Outputs

Keep the output folder outside the repository. It receives files that must not be committed: the fitted forest (`rf.joblib`), the change map as `change_map.npz` and `change_map.tif`, and working catalogs.

For each site, in `<output folder>/<site>/`:

- `summary.csv`: class areas in hectares, how much of the change sits on mangrove edges, model area against the labels, a rough image alignment check.
- `noise_floor.csv`: every pair of dates compared as if it were a change map. The rows marked `same year` are the noise floor.
- `sensitivity.csv`: the 2018 to 2025 change made in other reasonable ways.
- `label_gain_pixels.csv`: NDVI and model probability where the GMW v4.1.12 label says gain.
- `change_map_figure.png`: the 2018 image, the 2025 image and the change map.

`noise_floor_all_sites.py` writes four tables to `<output folder>/all_sites/`.

## How a year's mask is made

Every non-hazy scene of the year is predicted. A pixel is mangrove if the forest's probability is above the threshold chosen on the inner fold. Per pixel, the scenes with a usable value vote and more than half wins. With two scenes a 1-1 tie is possible. A tie is broken by the mean of the two probabilities against the same threshold. Where a year has one usable scene, that scene decides.

One pixel is 10 m by 10 m, so 100 m2 or 0.01 ha.
