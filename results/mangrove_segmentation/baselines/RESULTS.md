# Stage 4 baselines: leave-one-site-out results (Train/Val only)

23 Oct 2026. Label: GMW v4.0.19 (10 m), 2020 chips only (215 chips, 6 Abu Dhabi sites, 5 folds). Test_Dataset was not opened. Scores from `code/evaluate.py` (pixel-pooled per fold, 95% bootstrap CI over chips).

**Bar for the U-Net: mean leave-one-site-out IoU 0.70 (B2 random forest, variant b: hazy scenes dropped from training; ±0.10 across folds; pooled out-of-fold IoU 0.686, 95% CI 0.653–0.713).**

## Headline (mean ± sd across the 5 held-out folds)

| Model | Variant | IoU | F1 | Precision | Recall | Pooled OOF IoU [95% CI] |
|---|---|---|---|---|---|---|
| B1 NDVI + MNDWI threshold | (a) all 2020 dates | 0.598 ± 0.098 | 0.745 ± 0.081 | 0.716 ± 0.144 | 0.797 ± 0.053 | 0.572 [0.533, 0.604] |
| B1 NDVI + MNDWI threshold | (b) hazy scenes dropped from training | 0.601 ± 0.092 | 0.747 ± 0.075 | 0.725 ± 0.133 | 0.786 ± 0.047 | 0.579 [0.542, 0.611] |
| B2 random forest | (a) all 2020 dates | 0.700 ± 0.121 | 0.819 ± 0.086 | 0.781 ± 0.133 | 0.870 ± 0.030 | 0.673 [0.636, 0.704] |
| B2 random forest | (b) hazy scenes dropped from training | 0.704 ± 0.101 | 0.823 ± 0.071 | 0.808 ± 0.109 | 0.842 ± 0.032 | 0.686 [0.653, 0.713] |

## Per fold (IoU on the held-out site)

| Fold | Held-out site | Chips | B1 (a) | B1 (b) | B2 (a) | B2 (b) | Hazy chips dropped from training (b) | Buffer chips dropped (a) |
|---|---|---|---|---|---|---|---|---|
| 1 | eastern mangroves | 8 | 0.663 | 0.663 | 0.836 | 0.821 | 46 | 8 |
| 2 | jubail island | 80 | 0.441 | 0.456 | 0.529 | 0.562 | 33 | 16 |
| 3 | ras ghurab | 50 | 0.609 | 0.608 | 0.690 | 0.697 | 38 | 8 |
| 4 | ghantoot | 41 | 0.695 | 0.695 | 0.792 | 0.777 | 41 | 0 |
| 5 | al aryam islands,al nouf | 36 | 0.583 | 0.583 | 0.653 | 0.663 | 34 | 0 |

Fold 5 per site (B2 b): Al Aryam 0.638, Al Nouf 0.684. Fold 1 (Eastern Mangroves) has only 8 chips, so its score is the least stable. Per-site IoU for every model: `per_site_iou.csv`. Fitted thresholds and RF settings per fold: `fold_fits.csv`.

## What this shows

- **B2 adds about +0.10 IoU over B1** (0.70 vs 0.60 mean), and wins on every fold. The gain is mostly precision (0.78–0.81 vs 0.72) with higher recall too. The RF almost removes look-alike vegetation errors (table below).
- **Dropping hazy scenes helps only slightly:** +0.004 mean IoU for B2 and +0.003 for B1, with pooled OOF IoU 0.686 vs 0.673 for B2, well inside the CIs. It trades recall for precision and helps most at Jubail (+0.03 for B2). Rule used: dark-water red (5th percentile B04 over water) above 0.05. That drops 33–46 training chips per fold, mostly the August/September dates plus Al Aryam 2020-04-01 and Al Nouf May/August. Keep variant (b) as the default, but it is not a big lever.
- **Jubail is the hardest site** (B1 0.44–0.46, B2 0.53–0.56). Many thin, scattered patches show up as false positives along creeks (see `figs/examples.png`, row 1), and EDA found its reference maps agree only at 0.60.
- **Not comparable to the EDA's 0.33:** that number was NDVI-only on GMW v4.1.12 (~30 m) across all years for the old fixed val sites. These are 10 m labels, 2020 only, with a water-index rule and thresholds fit across 4 sites.

## Where the false positives are (out-of-fold, variant a; ESA WorldCover class of each pixel)

| WorldCover class | Non-mangrove px | B1 FP rate in class | B2 FP rate in class | B1 share of all FP | B2 share of all FP |
|---|---|---|---|---|---|
| tree | 19,116 | 85.99% | 23.55% | 3.3% | 1.2% |
| shrub | 47,389 | 28.61% | 1.53% | 2.7% | 0.2% |
| grass | 11,038 | 84.55% | 0.45% | 1.9% | 0.0% |
| crop | 4,052 | 38.55% | 14.88% | 0.3% | 0.2% |
| built-up | 394,185 | 3.42% | 0.21% | 2.7% | 0.2% |
| bare | 4,356,530 | 0.8% | 0.12% | 7.0% | 1.5% |
| water | 6,502,148 | 2.43% | 1.18% | 31.8% | 21.1% |
| herbaceous wetland | 100,558 | 18.47% | 8.12% | 3.7% | 2.2% |
| mangrove (WorldCover) | 1,307,882 | 17.75% | 20.39% | 46.6% | 73.3% |

- **Most remaining B2 false positives (73%) fall on pixels WorldCover maps as mangrove but GMW v4.0.19 does not.** That is label disagreement more than model error, which the hand-labelled 10 m chips will settle. Another 21% are water, mainly channel and patch edges.
- **B1's look-alike errors are large per class but small in total:** it flags 86% of WorldCover tree pixels, 85% of grass and 29% of shrub as mangrove. B2 cuts these to 24%, 0.5% and 1.5%. SWIR and the visible bands carry that separation.

## Random forest notes

- **Feature importance** (mean over folds, variant a): NDVI 0.33, B08 0.14, B02 0.12, B11 0.10, B04 0.10, B03 0.09, B12 0.08, MNDWI 0.05 (`figs/rf_feature_importance.png`).
- **Decision threshold:** baseline_rf.py's grid stopped at 0.8 and fold 1 first picked 0.8. The grid was extended to 0.95 in the wrapper (`scripts/run_cv.py`; code/ unchanged) and everything re-run. Chosen thresholds are now 0.80–0.90 and none sits at the 0.95 edge.
- **Calibration** (`rf_calibration_inner.csv`, inner-validation fold, natural class mix, about 10% mangrove): balanced pixel sampling makes the RF over-confident. Pixels scored 0.5–0.6 are mangrove 17% of the time, 0.8–0.9 → 47%, 0.9–1.0 → 85%. So the high optimal thresholds are expected (prior shift from 50% to about 10% mangrove), not a grid artefact. Any model trained with balanced or weighted loss (e.g. the U-Net) must tune its threshold on validation, not use 0.5.

## Files

- `cv_results.csv`: per model × variant × fold, IoU/F1/precision/recall, IoU CI, per-site IoU.
- `per_chip.csv`: out-of-fold confusion counts and IoU per chip.
- `oof_pooled.csv`: pooled out-of-fold scores (all chips and non-hazy chips).
- `per_site_iou.csv`, `landcover_errors.csv`, `fold_fits.csv`, `rf_feature_importances.csv`, `rf_calibration_inner.csv`.
- `figs/iou_per_fold.png`, `figs/rf_feature_importance.png`, `figs/examples.png`.
- `preds/<model>_<variant>/`: out-of-fold masks.
- `scripts/run_cv.py` (fit and predict) and `scripts/score.py` (scoring and figures). Seeds fixed at 0.

Method: per held-out fold, B1 thresholds (NDVI > t, MNDWI < w) are grid-searched on the 4 training folds. B2 is tuned (3 depth/leaf settings × thresholds 0.20–0.95, 100 trees, 100k balanced pixels) on an inner split: train on 3 folds, validate on the first of folds 3, 4, 5, 2, 1 that is not held out. It is then refit with 200 trees on 200k balanced pixels from all 4 training folds. Training chips within one chip (2,560 m) of any held-out chip are dropped: 8 when Eastern Mangroves is held out, 16 for Jubail and 8 for Ras Ghurab.
