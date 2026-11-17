# SVM baseline (B3): leave-one-site-out results (Train/Val only)

Label: GMW v4.0.19 (10 m). 2020 chips only (215 chips, 6 Abu Dhabi sites, 5 folds). Test_Dataset was not opened. Scores come from `evaluate.py` (pooled over pixels per fold, 95% bootstrap CI over chips). The code is `src/mangrove_segmentation/baseline_svm.py cv`. It was run once with the default linear SVM and once with `--kernel rbf_approx`.

## Headline (mean ± sd across the 5 held-out folds)

| Model | Variant | IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| B1 NDVI + MNDWI threshold | (a) all 2020 dates | 0.598 ± 0.098 | 0.745 ± 0.081 | 0.716 ± 0.144 | 0.797 ± 0.053 |
| B1 NDVI + MNDWI threshold | (b) hazy scenes dropped from training | 0.601 ± 0.092 | 0.747 ± 0.075 | 0.725 ± 0.133 | 0.786 ± 0.047 |
| B2 random forest | (a) all 2020 dates | 0.700 ± 0.121 | 0.819 ± 0.086 | 0.781 ± 0.133 | 0.870 ± 0.030 |
| B2 random forest | (b) hazy scenes dropped from training | 0.704 ± 0.101 | 0.823 ± 0.071 | 0.808 ± 0.109 | 0.842 ± 0.032 |
| B3 SVM, linear | (a) all 2020 dates | 0.608 ± 0.092 | 0.752 ± 0.075 | 0.739 ± 0.110 | 0.776 ± 0.090 |
| B3 SVM, linear | (b) hazy scenes dropped from training | 0.597 ± 0.093 | 0.744 ± 0.077 | 0.760 ± 0.115 | 0.740 ± 0.103 |
| B3 SVM, approximate RBF | (a) all 2020 dates | 0.704 ± 0.099 | 0.823 ± 0.069 | 0.780 ± 0.092 | 0.873 ± 0.046 |
| B3 SVM, approximate RBF | (b) hazy scenes dropped from training | 0.702 ± 0.105 | 0.822 ± 0.074 | 0.784 ± 0.099 | 0.865 ± 0.052 |

The B1 and B2 rows are copied from `../baselines/RESULTS.md`. The sd is the sample sd (divide by n minus 1), as in that table. The script prints an sd that divides by n, so its values are a little smaller.

## Per fold (IoU on the held-out site, variant b)

| Fold | Held-out site | Chips | B1 NDVI | B2 random forest | B3 SVM linear | B3 SVM approx. RBF |
|---|---|---|---|---|---|---|
| 1 | eastern mangroves | 8 | 0.663 | 0.821 | 0.691 | 0.830 |
| 2 | jubail island | 80 | 0.456 | 0.562 | 0.442 | 0.558 |
| 3 | ras ghurab | 50 | 0.608 | 0.697 | 0.619 | 0.689 |
| 4 | ghantoot | 41 | 0.695 | 0.777 | 0.629 | 0.774 |
| 5 | al aryam islands,al nouf | 36 | 0.583 | 0.663 | 0.602 | 0.660 |
| **Mean** | | | **0.601** | **0.704** | **0.597** | **0.702** |

Variant (a), same order of folds. SVM linear: 0.698, 0.459, 0.606, 0.667, 0.608 (mean 0.608). SVM approximate RBF: 0.821, 0.567, 0.694, 0.776, 0.665 (mean 0.704).

The 95% CI of each fold is in `cv_results.csv`. It is about ±0.03 to ±0.08 wide. A difference of 0.01 between two models on one fold does not mean anything.

## What this shows

The linear SVM is no better than the NDVI rule (mean IoU 0.60). The SVM with a curved boundary matches the random forest (0.70). So the gain over NDVI comes from combining the 8 values in a nonlinear way. It does not come from the choice of classifier. The bar for the U-Net stays at 0.70.

## Comparability with the B1 and B2 runs

The team's script `run_cv.py` produced B1 and B2. It is not in this repository. I compared the `cv` command with it line by line and aligned the two before these runs.

- The chips scored are the same. Chip counts and mangrove pixel counts per fold are identical to `../baselines/cv_results.csv`. This was checked for all 10 rows of both runs.
- The training sets are the same. Hazy chips dropped from training in variant (b) are 46, 33, 38, 41, 34 per fold. Buffer chips dropped in variant (a) are 8, 16, 8, 0, 0. Both match `../baselines/RESULTS.md`.
- The pixel samples follow the same procedure. Tuning uses 100,000 balanced pixels and validates on 300,000 pixels of the inner fold. The final fit uses 200,000 balanced pixels. They are drawn with the random forest's own sampler after restarting the random generator from seed 0, as `run_cv.py` does. So the pixels should be the ones the forest was trained on. This was not checked against the forest's saved run.
- The inner folds and the scoring are the same. The inner fold is fold 3 (fold 4 when fold 3 is held out). Scoring uses `evaluate.py` with 1000 bootstrap resamples and seed 0.
- No fit chose a threshold on the edge of the grid. The chosen values are 0.25 to 1.0 and the grid runs from -2 to 4. The automatic widening was never needed.
- Some things are not the same. The SVM tunes C (0.01, 0.1, 1) where the forest tunes depth and leaf size. The SVM cuts a signed score and the forest cuts a probability. scikit-learn 1.9.0 was used here. The version used for B1 and B2 is not recorded.

## Notes

- C hardly matters for the linear SVM. The inner fold IoU changes by at most 0.003 across the three C values.
- C matters more for the approximate RBF and its grid may be too short. 6 of its 10 fits chose C = 1, which is the largest value tried. At Jubail the inner fold IoU was still rising (0.597, 0.629, 0.649 for C = 0.01, 0.1, 1). A larger C was not tried. So 0.70 may be a little lower than what this model can reach.
- The RBF settings were not tuned. It uses 300 landmark pixels and gamma = 1/8. These are the values written in the code.
- Dropping hazy scenes does not help the SVM. It lowers the mean IoU by 0.011 for the linear one and by 0.002 for the approximate RBF. Both are well inside the spread between folds.
- Every number here is from one run with seed 0.

## Files

- `linear/cv_results.csv` and `rbf_approx/cv_results.csv`. One row per variant and fold with IoU, F1, precision, recall, the IoU CI and the IoU per site. The columns are the same as in the baselines' table.
- `linear/fold_fits.csv` and `rbf_approx/fold_fits.csv`. The chosen C and threshold per fit, the grid searched, the inner fold IoU for each C, the chips and pixels used, and the chips dropped.
- The scores per chip and the predicted masks are not in the repository.

## Method

For each held-out fold, C and the score threshold are tuned on an inner split. The SVM trains on 3 folds and validates on the first of folds 3, 4, 5, 2, 1 that is not held out. It is then refitted with that C on 200k balanced pixels from all 4 training folds. The held-out chips are predicted with that threshold. Training chips within one chip (2,560 m) of any held-out chip are dropped. Features are standardised from the training pixels.
