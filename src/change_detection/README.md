# Change detection: first model-based change map

These scripts make a 2018 to 2025 mangrove change map for one training site with the random forest baseline, and check how much of the "change" is model noise. The results are in [results/change_detection/RESULTS.md](../../results/change_detection/RESULTS.md).

## Scripts

- `inventory.py` lists which sites have chips in both years, which scenes are hazy and which labels exist.
- `train_rf.py` trains the random forest with one site held out.
- `change_map.py` predicts that site, builds the change map, runs the checks and draws the figure.
- `noise_floor_all_sites.py` runs the same-year check for every site.
- `common.py` holds the shared helpers.

## What they depend on

- `baseline_rf.py` and `chipio.py` in `src/mangrove_segmentation`, imported unchanged.
- The team's script `run_cv.py`, which is not in this repository. The scripts do not need the file to run. Its steps for one fold are written out again in `common.py`. As a check, each forest's held-out IoU matches `results/mangrove_segmentation/baselines/cv_results.csv` to four decimals.
- The Train/Val dataset (`catalog.csv`, `hazy_scenes.csv`, `chips/`), which is not in the repository.

## How to run

From `src/change_detection`, on a CPU, a few minutes each:

```
python inventory.py             --train-val <Train_Val_Dataset> --out <output folder>
python train_rf.py              --train-val <Train_Val_Dataset> --out <output folder> --site jubail_island
python change_map.py            --train-val <Train_Val_Dataset> --out <output folder> --site jubail_island
python noise_floor_all_sites.py --train-val <Train_Val_Dataset> --out <output folder>
```

Keep the output folder outside the repository. It also receives the fitted forest and the map as rasters, which must not be committed.
