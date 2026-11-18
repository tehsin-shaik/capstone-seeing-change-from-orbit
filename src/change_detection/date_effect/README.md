# Date effect: what causes it?

These scripts test three possible reasons why the random forest's mangrove area changes with the image date. The results are in [results/change_detection/date_effect/RESULTS.md](../../../results/change_detection/date_effect/RESULTS.md).

- `tide/` compares the swings with the water level, from the image and from a tide model.
- `image_differences/` measures how much the images differ on pixels that shouldn't change, corrects the images and runs the forests again.
- `robustness/` compares four versions of the forest (current, trained with brightness changes, indices only, area from probabilities).

Run the folders in that order. The later ones read tables written by the earlier ones.

## What they need

- The Train/Val dataset (`catalog.csv`, `hazy_scenes.csv`, `chips/`), which is not in the repository.
- `common.py` in `src/change_detection` (and through it `baseline_rf.py` and `chipio.py`). It is found automatically. Use `--code` if it is somewhere else.
- For `tide/01` and `tide/03` only: the `pyTMD` package (3.1.0 was used) and `pyproj`.
- For `tide/03` only: two open tide models, which are not in the repository. EOT20 ocean tides from https://doi.org/10.17882/79489 unpacked as `<models>/EOT20/ocean_tides/*.nc`, and GOT4.10c from NASA GSFC unpacked as `<models>/GOT4.10c/grids_oceantide/*.d`. Neither needs a login.
- `tide/01` and `image_differences/01` look up each scene in the public Sentinel-2 archive (Earth Search) to get the sensing time and the processing version. They read the record only, no imagery.

## How to run

`<work>` folders get predictions, image stacks and fitted forests. Keep them outside the repository. `<out>` folders get the small tables and figures.

```
cd tide
python 01_scene_times.py        --train-val <Train_Val_Dataset> --out <tide out>
python 02_predict_all_scenes.py --train-val <Train_Val_Dataset> --work <tide work>
python 03_tide_model.py         --models <models> --out <tide out>
python 04_analysis.py           --work <tide work> --out <tide out> --train-val <Train_Val_Dataset> --disagreement <repo>/results/mangrove_segmentation/model_disagreement/per_site_date.csv
python 05_figures.py            --out <tide out>

cd ../image_differences
python 01_scene_meta.py         --train-val <Train_Val_Dataset> --out <images out>
python 02_stacks_and_forests.py --train-val <Train_Val_Dataset> --work <images work>
python 03_stable_targets.py     --work <images work> --out <images out>
python 04_track_swings.py       --out <images out> --tide-table <tide out>/scene_table.csv
python 05_corrected_check.py    --work <images work> --out <images out> --committed <repo>/results/change_detection/all_sites/noise_floor_all_pairs.csv
python 06_figures.py            --out <images out>
python 07_after_checks.py       --work <images work> --out <images out>

cd ../robustness
python 01_train_versions.py     --train-val <Train_Val_Dataset> --work <robustness work> --lines <images out>/correction_lines.csv --a-forests <images work>/forests
python 02_score_versions.py     --train-val <Train_Val_Dataset> --work <robustness work> --images-work <images work> --out <robustness out> --committed-pairs <repo>/results/change_detection/all_sites/noise_floor_all_pairs.csv
python 03_figures.py            --out <robustness out>
python 04_after_checks.py       --out <robustness out>
```

Everything runs on a CPU. Most steps take a few minutes. `image_differences/05` takes about 12 minutes, and `robustness/01` and `robustness/02` about 20 minutes each.

## What was fixed before running

- The number that counts in `image_differences/05` and `robustness/02` is the average size of the same-year swing per site. A same-year pair is two non-hazy dates of one site and year, compared like `noise_floor_all_sites.py` does. A pair's swing is the size of the net change as a % of the first date's predicted mangrove. A site's number is the mean over its pairs, and the main number is the mean of the six sites.
- Image correction: "fixes it" if that mean falls by at least half and no site gets worse by more than 2 points, "helps" if it falls by at least a quarter. The correction uses no labels and no model output. Its reference date is the site's most typical date.
- Four versions: a version "helps" if that mean falls by at least a quarter against the current forest and held-out IoU (all held-out 2020 chips, mean of five folds) is no more than 0.01 lower.
- For the brightness copies, lines measured at the held-out site are never used, and Al Nouf's lines are left out because its fit was fragile.

## Checks built into the scripts

- `tide/02` and `image_differences/02` print each forest's held-out IoU next to the committed one. They should be the same.
- `image_differences/05` and `robustness/02` stop if the same-year pairs on the images as they are don't match `results/change_detection/all_sites/noise_floor_all_pairs.csv`. `robustness/02` also stops if the current forest doesn't give the committed IoU per fold.
- `robustness/01` first trains with no change at all and stops if that does not give the current forest again.
- `image_differences/07` and `robustness/04` were added after the results were seen, to understand them. They are not part of the tests.
