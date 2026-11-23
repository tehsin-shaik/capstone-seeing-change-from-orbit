# Plan for the final test

24 Nov 2026. This page was written and committed before the test was run. It fixes the model, the test set, the scoring and the rules, so that none of them can move once we have seen a score. The script is [`final_test.py`](../../../src/mangrove_segmentation/final_test.py). Test_Dataset was not opened to write this page or the script.

## The frozen model

| Item | Value |
|---|---|
| Model | Random forest, variant (b): hazy scenes dropped from training |
| Settings | max_depth 12, min_samples_leaf 10. All 5 folds picked these. |
| Trees | 200 |
| Training pixels | 200,000, class-balanced (100,000 mangrove, 100,000 not) |
| Seed | 0 |
| Threshold | 0.85. A pixel is mangrove if its probability is above 0.85. |
| Features | The 8 chip channels of each pixel (B02, B03, B04, B08, B11, B12, NDVI, MNDWI), exactly as `baseline_rf.pixels()` builds them |
| Training data | Every 2020 Train/Val chip from all six sites, minus hazy scenes: 167 of the 215 chips |
| Label | GMW v4.0.19 |

The threshold is the median of the five fold thresholds from cross-validation: 0.85, 0.90, 0.85, 0.85, 0.80. They are in [`fold_fits.csv`](../../change_detection/all_sites/fold_fits.csv), with the settings each fold picked.

A scene is hazy if its dark-water red value is above 0.05, the same rule as in cross-validation. That drops 48 chips from 7 scenes: the August image of Eastern Mangroves, Jubail, Ras Ghurab, Ghantoot and Al Nouf, Al Aryam on 1 April and Al Nouf on 16 May.

The forest is fitted with the same lines as the final refit in `run_cv.py`. The only change is that no site is held out, so there is no buffer to drop. Nothing is tuned in `final_test.py`. The numbers above are written into the script as constants.

The fitted model is kept outside the repository. The test script refuses to use any other file.

Model file sha256: `0d4cb8f90629d5c30ad64c3dbfcd8ad1356a39806f0f5b0fb9bc49dfcf6dd19c`

## The test set

The test set is every chip of the 7 locked sites listed in section 1 of the [data dictionary](../../../data/DATA_DICTIONARY.md).

| Group | Sites |
|---|---|
| fixed_test | Abu Al Abyad, Marawah |
| external | Ras Al Khor, Al Zorah, Khor Al Beida, Ras Al Khaimah creek, Khor Kalba |

The two groups are the two folders `make_datasets.py` writes (`fixed_test` and `external`). They are scored separately.

## Scoring

Scoring is `evaluate.py` as it is. The label is GMW v4.0.19 and the class is mangrove. Pixels are pooled per site and over all chips of a group. The 95% interval comes from 1,000 bootstrap draws of chips with seed 0. Cloud and no-data pixels are ignored.

Every test chip with a label is scored, hazy ones included. The v4.0.19 label exists for 2020 only, so `evaluate.py` skips the 2018 and 2025 chips. We will report how many it skipped.

## Why the random forest

The top models are tied on IoU. Mean leave-one-site-out IoU is 0.704 for the random forest, 0.702 for the RBF SVM and 0.704 for the U-Net. The paired differences are 0.001 (forest minus SVM, -0.004 to 0.006) and 0.000 (U-Net minus forest, -0.021 to 0.025). Both intervals include zero. See [model comparison](../model_comparison/RESULTS.md).

With a tie, we take the model that is steadier and simpler.

The forest is the steadiest across the 24 site and date combinations of 2020. Its sd is 0.101 and its worst one is 0.502. For the U-Net these are 0.138 and 0.365. The U-Net row is our retrain. See [confidence](../confidence/RESULTS.md), table A.

It gains the most from combining dates. With a vote over the non-hazy dates of a chip cell (up to three), it goes from 0.714 to 0.752. The RBF SVM gains 0.033 and DINOv3 with LoRA 0.015. The U-Net was not in that check. See [multi-date vote](../multi_date_vote/RESULTS.md).

It runs on a CPU. Fitting the final model took 7 seconds on a desktop PC, with no GPU.

It scores 0.864 against the majority hand label. This is supporting evidence only. It is 6 chips, all from Train/Val sites, and the majority label leans towards tight edges, which suits the forest. See [hand labels](../hand_labels/RESULTS.md).

## The rules

1. The test runs once.
2. The score is reported whatever it is.
3. Nothing is changed and re-run after scores exist. That covers the model, the threshold, the test set and the scoring.
4. Failures are analysed, not fixed. If a site scores badly we look at why and write it up. We do not retrain.

The script enforces the first rule. `--run-test` refuses to start if any of these is true:

- `TEST_RUN_DONE.json` already exists in the output folder.
- The git working tree is dirty.
- This page is not committed.
- The model file does not have the sha256 written above.
- A test folder holds a Train/Val site, or does not hold the number of sites in the table above (2 and 5).

All of these are checked before any test chip is predicted. As soon as the first score exists, the script writes `TEST_RUN_DONE.json` with the time, the git commit and the model sha256, even if a later step fails.

## What we expect

We expect roughly 0.6 to 0.8 IoU per site. This comes from the 24 site and date combinations in cross-validation: mean 0.69, sd 0.10, worst 0.50.

The test sites are new places, so a site outside this range would not be a surprise. Our hardest Train/Val site, Jubail, scored 0.56. We write the range down now so that we cannot adjust it later.

## What will be reported

For fixed_test and for external: IoU, F1, precision and recall with their 95% intervals, per site and pooled. They will be shown next to the cross-validation IoU of 0.704.

The 0.704 is a mean over 5 folds. The pooled test rows are pixel-pooled, and the matching cross-validation number for those is the pooled out-of-fold IoU, 0.686 (0.653 to 0.713). We will show both.

## What this plan doesn't cover

- Scores are against GMW, which is itself a machine-made map. A low score at a site can be the label as much as the model.
- Only the random forest is run on the test set. The test cannot say whether the SVM or the U-Net would have done better there.
- The test has one model and one seed. The interval covers a different draw of chips from the same sites, not new sites.

## How to run

```
cd src/mangrove_segmentation
python final_test.py fit                                    # done on 24 Nov 2026
python final_test.py --dry-run                              # Train/Val chips only, checks the code
python final_test.py --run-test --test-root <Test_Dataset>  # the real test, once
```

The dry run scores Train/Val chips the forest was trained on. Those numbers only show that the code runs. They are not reported.

Models, masks and tables go to a folder outside the repository (`--out`).
