# SVM baseline

`baseline_svm.py` adds a support vector machine (SVM) as a third baseline, next to the NDVI rule (B1) and the random forest (B2). Like the random forest, it classifies one pixel at a time from its 8 values (B02, B03, B04, B08, B11, B12, NDVI and MNDWI) as mangrove or not.

It is built to match `baseline_rf.py`. It uses the same chips, catalog and labels (GMW v4.0.19 by default), and it samples its 200,000 class-balanced training pixels with the random forest's own function. The predicted masks have the same format, scoring goes through `evaluate.py`, and the results table has the same columns as the baselines' `cv_results.csv`. On validation data it tunes the SVM's `C` setting and the cut-off on its score.

## Why it's linear

A standard SVM with an RBF kernel compares training pixels with each other, so its cost grows roughly with the square of the number of pixels. With 200,000 training pixels and about 14 million pixels to predict, that would take hours to days. A linear SVM (`LinearSVC`) scales in proportion to the data and trains in seconds.

The trade-off is that a linear SVM can only draw one flat boundary through the 8 values, so it will probably score closer to the NDVI rule than to the random forest. If a curved boundary is wanted, `--kernel rbf_approx` adds a fast approximation of the RBF kernel. Features are standardised first, because the bands and the indices are on different scales.

## How to run it

From `src/mangrove_segmentation`, leave-one-site-out cross-validation on the Train/Val dataset:

```
python baseline_svm.py cv --catalog <Train_Val_Dataset>/catalog.csv --chips <Train_Val_Dataset> --qa <QA_download.csv> --out results_svm
```

This follows the protocol in the baselines' `RESULTS.md`: five folds, one site held out each time, the SVM tuned on an inner fold and refitted on the other four, and a one-chip buffer around the held-out site. It runs with all 2020 dates and again with hazy scenes dropped from training. It writes `cv_results.csv`, `fold_fits.csv`, `per_chip.csv`, `config.json` and the predicted masks.

The same `fit` and `predict` commands as the random forest also work:

```
python baseline_svm.py fit --catalog catalog.csv --chips chips/ --out models/svm.joblib
python baseline_svm.py predict --catalog catalog.csv --chips chips/ --model models/svm.joblib --split test --pred-out preds/svm
```

The tests are in `tests/test_baseline_svm.py`.

## Status

The SVM was run on the real Train/Val data on 18 Nov 2026 (2020 chips, 5 folds, test data not opened). Mean leave-one-site-out IoU, with hazy scenes dropped from training:

| Model | Mean IoU |
|---|---|
| NDVI rule (B1) | 0.601 |
| Random forest (B2) | 0.704 |
| SVM, linear | 0.597 |
| SVM, approximate RBF (`--kernel rbf_approx`) | 0.702 |

The linear SVM is level with the NDVI rule and the approximate RBF SVM matches the random forest. Per-fold scores, the checks that the runs are comparable with B1 and B2, and the caveats are in [results/mangrove_segmentation/svm/RESULTS.md](../../results/mangrove_segmentation/svm/RESULTS.md).

The `cv` command has been compared line by line with `run_cv.py`, the team's script that ran the B1 and B2 cross-validation (not in this repository), and aligned with it: same pixel sample sizes, same seed handling, and a threshold grid that widens if the best value lands on its edge.
