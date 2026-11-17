# SVM baseline

`baseline_svm.py` adds a support vector machine (SVM) as a third baseline. The other two are the NDVI rule (B1) and the random forest (B2). Like the random forest, the SVM looks at one pixel at a time. It uses the pixel's 8 values (B02, B03, B04, B08, B11, B12, NDVI and MNDWI) to decide if it is mangrove or not.

The script is built to match `baseline_rf.py`. It uses the same chips, catalog and labels (GMW v4.0.19 by default). It picks its 200,000 balanced training pixels with the random forest's own function. The predicted masks have the same format. Scoring goes through `evaluate.py`. The results table has the same columns as the baselines' `cv_results.csv`. On validation data it tunes the SVM's `C` setting and the cutoff on its score.

## Why it's linear

A standard SVM with an RBF kernel compares training pixels with each other. Its cost grows roughly with the square of the number of pixels. With 200,000 training pixels and about 14 million pixels to predict, that would take hours to days. A linear SVM (`LinearSVC`) scales in proportion to the data and trains in seconds.

The downside is that a linear SVM can only draw one flat boundary through the 8 values. If a curved boundary is wanted, `--kernel rbf_approx` adds a fast approximation of the RBF kernel. Features are standardised first because the bands and the indices are on different scales.

## How to run it

From `src/mangrove_segmentation`, leave-one-site-out cross-validation on the Train/Val dataset:

```
python baseline_svm.py cv --catalog <Train_Val_Dataset>/catalog.csv --chips <Train_Val_Dataset> --qa <QA_download.csv> --out results_svm
```

This follows the protocol in the baselines' `RESULTS.md`. There are five folds and one site is held out each time. The SVM is tuned on an inner fold and refitted on the other four. A buffer of one chip is kept around the held-out site. It runs with all 2020 dates and again with hazy scenes dropped from training. It writes `cv_results.csv`, `fold_fits.csv`, `per_chip.csv`, `config.json` and the predicted masks.

The same `fit` and `predict` commands as the random forest also work:

```
python baseline_svm.py fit --catalog catalog.csv --chips chips/ --out models/svm.joblib
python baseline_svm.py predict --catalog catalog.csv --chips chips/ --model models/svm.joblib --split test --pred-out preds/svm
```

The tests are in `tests/test_baseline_svm.py`.

## Results

The SVM has been run on the real Train/Val data (2020 chips, 5 folds). The test data was not opened. The table gives the mean leave-one-site-out IoU with hazy scenes dropped from training.

| Model | Mean IoU |
|---|---|
| NDVI rule (B1) | 0.601 |
| Random forest (B2) | 0.704 |
| SVM, linear | 0.597 |
| SVM, approximate RBF (`--kernel rbf_approx`) | 0.702 |

The linear SVM is level with the NDVI rule. The approximate RBF SVM matches the random forest.

Two caveats about the RBF run. Most of its fits chose the largest C in the grid, so a larger C might score a little higher. Its landmark count and gamma were not tuned.

The scores per fold, the checks that the runs are comparable with B1 and B2, and the other notes are in [results/mangrove_segmentation/svm/RESULTS.md](../../results/mangrove_segmentation/svm/RESULTS.md).

The team's script `run_cv.py` ran the B1 and B2 cross-validation. It is not in this repository. The `cv` command has been compared with it line by line and aligned with it. It uses the same pixel sample sizes and the same seed handling. Its threshold grid widens if the best value lands on the edge.
