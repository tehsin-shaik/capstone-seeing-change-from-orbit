# Final test results

24 Nov 2026 (23 Nov, 21:47 UTC). This is the one test on the locked sites. It ran once, with the model and the rules fixed beforehand in [`PLAN.md`](PLAN.md). Nothing was changed before or after the run: the script, the plan, the model, the settings and the threshold are the ones that were merged.

| Item | Value |
|---|---|
| Script | [`final_test.py`](../../../src/mangrove_segmentation/final_test.py) `--run-test` |
| Git commit the test ran on | `dfe08d77d4c424489cb7d4ba72a43b36f4761f9a` |
| Model sha256 | `0d4cb8f90629d5c30ad64c3dbfcd8ad1356a39806f0f5b0fb9bc49dfcf6dd19c` |
| Model | Random forest, variant (b), threshold 0.85 |
| Label | GMW v4.0.19, mangrove class |
| Scoring | `evaluate.py` as it is. Pixels pooled per site and per group. 95% interval from 1,000 bootstrap draws of chips, seed 0. |

The run finished without an error, so no code was fixed.

## fixed_test

IoU, F1, precision and recall with their 95% interval. "Scored" is the number of chips with a v4.0.19 label. "Skipped" is the number of chips without one (the 2018 and 2025 chips).

| Site | Scored | Skipped | Mangrove px | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| abu_al_abyad | 21 | 17 | 68,249 | 0.780 [0.732, 0.819] | 0.876 [0.845, 0.900] | 0.858 [0.820, 0.891] | 0.895 [0.865, 0.920] |
| marawah | 17 | 18 | 27,730 | 0.723 [0.646, 0.775] | 0.839 [0.785, 0.873] | 0.829 [0.739, 0.885] | 0.849 [0.801, 0.886] |
| **pooled** | 38 | 35 | 95,979 | 0.763 [0.723, 0.798] | 0.866 [0.839, 0.888] | 0.850 [0.814, 0.880] | 0.882 [0.855, 0.904] |

## external

| Site | Scored | Skipped | Mangrove px | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| ajman_al_zorah | 7 | 11 | 41,883 | 0.785 [0.756, 0.813] | 0.880 [0.861, 0.897] | 0.927 [0.909, 0.951] | 0.837 [0.792, 0.879] |
| khor_al_beida | 51 | 65 | 354,622 | 0.819 [0.776, 0.855] | 0.900 [0.874, 0.922] | 0.950 [0.920, 0.974] | 0.856 [0.818, 0.887] |
| khor_kalba | 14 | 16 | 42,807 | 0.746 [0.596, 0.887] | 0.855 [0.747, 0.940] | 0.807 [0.626, 0.975] | 0.909 [0.881, 0.935] |
| ras_al_khaimah_creek | 15 | 20 | 69,348 | 0.905 [0.860, 0.926] | 0.950 [0.925, 0.961] | 0.953 [0.911, 0.974] | 0.947 [0.927, 0.956] |
| ras_al_khor | 11 | 9 | 26,446 | 0.640 [0.571, 0.695] | 0.780 [0.727, 0.820] | 0.831 [0.735, 0.894] | 0.735 [0.646, 0.826] |
| **pooled** | 98 | 121 | 535,106 | 0.811 [0.772, 0.843] | 0.896 [0.871, 0.915] | 0.929 [0.889, 0.956] | 0.865 [0.838, 0.889] |

Every 2020 chip of the seven sites was scored: 136 chips. The 156 chips that were skipped are all from 2018 or 2025. Hazy scenes were not removed from the test set.

## Next to cross-validation

| Score | IoU | Chips | Sites |
|---|---|---|---|
| Test, fixed_test, pixel-pooled | 0.763 [0.723, 0.798] | 38 | 2 |
| Test, external, pixel-pooled | 0.811 [0.772, 0.843] | 98 | 5 |
| Cross-validation, pixel-pooled (the matching one) | 0.686 [0.653, 0.713] | 215 | 6 |
| Cross-validation, mean of the 5 folds | 0.704 (sd 0.101) | 215 | 6 |

Both pooled test scores are above the cross-validation scores. Both test intervals lie above the interval of the pixel-pooled cross-validation score.

The 0.704 is a mean over folds, so the number that matches it is a mean over test sites. The mean IoU of the 7 test sites is 0.771 (sd 0.082). It is 0.751 for the two fixed_test sites and 0.779 for the five external sites. These three means are computed from the tables above.

## Against the expected range

`PLAN.md` said we expected roughly 0.6 to 0.8 IoU per site.

| Site | Group | IoU | Inside 0.6 to 0.8 |
|---|---|---|---|
| ras_al_khor | external | 0.640 | yes |
| marawah | fixed_test | 0.723 | yes |
| khor_kalba | external | 0.746 | yes |
| abu_al_abyad | fixed_test | 0.780 | yes |
| ajman_al_zorah | external | 0.785 | yes |
| khor_al_beida | external | 0.819 | no, above |
| ras_al_khaimah_creek | external | 0.905 | no, above |

Five of the seven sites are inside the range. Two are above it. None is below it. The lowest site is ras_al_khor at 0.640 and the highest is ras_al_khaimah_creek at 0.905.

## How the test set was built

The test chips come from the same build as Train/Val. All 454 Train/Val chip files are in the full chips folder, and for all 454 the sha256 equals the Train/Val copy (454 of 454 hashes equal).

A second build of the chips, received later, was rejected because 0 of 454 hashes matched. It was not used for anything.

The steps were:

1. The full chips folder holds 746 chip files from all 13 sites. Only the 454 Train/Val files were hashed.
2. The other 292 files were sorted by file name only (`<site>_<date>_r<row>_c<col>.npz`). No locked chip was opened to do this. They belong to exactly 7 sites, the ones in `PLAN.md`. Every file name fitted the pattern and no file was left over.
3. The 292 files were hard linked into `Test_Dataset/fixed_test/chips` and `Test_Dataset/external/chips`, the two folders `make_datasets.py` writes. Each folder got a `catalog.csv` with chip_id, site, split and path, taken from the file names.

The site names in the files are the written names of the [data dictionary](../../../data/DATA_DICTIONARY.md) in lower case with underscores. The one that differs is Al Zorah, which is `ajman_al_zorah` in the files.

Chips per site and year:

| Group | Site | 2018 | 2020 | 2025 | All |
|---|---|---|---|---|---|
| fixed_test | abu_al_abyad | 7 | 21 | 10 | 38 |
| fixed_test | marawah | 5 | 17 | 13 | 35 |
| fixed_test | all | 12 | 38 | 23 | 73 |
| external | ras_al_khor | 5 | 11 | 4 | 20 |
| external | ajman_al_zorah | 5 | 7 | 6 | 18 |
| external | khor_al_beida | 32 | 51 | 33 | 116 |
| external | ras_al_khaimah_creek | 11 | 15 | 9 | 35 |
| external | khor_kalba | 8 | 14 | 8 | 30 |
| external | all | 61 | 98 | 60 | 219 |

The script predicted all 292 chips. Only the 2020 chips have a v4.0.19 label, so only those were scored.

## What this does not show

- It is one run: one model, one seed, run once. We have no second run to compare it with.
- The scores are against GMW v4.0.19, which is a machine-made map and not ground truth. A score here says how close the forest is to GMW at these sites.
- It covers 2020 only. The 2018 and 2025 chips of the test sites have no v4.0.19 label and were not scored.
- The intervals come from redrawing chips of the same sites. They do not cover other sites. Some sites have few chips (7 at ajman_al_zorah, 11 at ras_al_khor).
- Only the random forest was run on the test set, so this does not compare models.
- We have not tested why any site is higher or lower than another, so this page gives no reasons.

## Files

- [`results.json`](results.json): every number above, with the commit and the model sha256.
- [`tables.csv`](tables.csv): the two tables, one row per site.
- [`per_chip_fixed_test.csv`](per_chip_fixed_test.csv) and [`per_chip_external.csv`](per_chip_external.csv): TP, FP, FN and IoU per chip.
- [`TEST_RUN_DONE.json`](TEST_RUN_DONE.json): the record the script wrote when the scores existed. The script will not run the test again while this file is in its output folder.

The masks, the model file and the test chips are kept outside the repository.
