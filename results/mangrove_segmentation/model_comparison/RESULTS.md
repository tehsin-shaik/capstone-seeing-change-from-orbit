# Paired comparison of the mangrove models

Label: GMW v4.0.19 (10 m). 2020 chips only (215 chips, 6 sites, 5 folds), variant (b) hazy scenes dropped. Test_Dataset was not opened. Nothing was retrained, this only reads the `cv_results.csv` each model already has. Code: `src/mangrove_segmentation/compare_models.py`.

## Result

RF, SVM (approx. RBF), U-Net, frozen DINOv3 and DINOv3 LoRA are tied on our 5 folds. None of them beats another on all 5 folds, and the biggest gap between them is 0.018 IoU. For comparison, the gap between our easiest and hardest site is about 0.30.

All five beat the NDVI rule and the linear SVM on every fold, by roughly 0.09 to 0.11 IoU.

From the 26 Oct meeting: pick the final model from CV only, and take the simpler one if there's a tie. So the pick should be RF or SVM (approx. RBF). Which of those two is "simpler" is up for discussion, since `model_size.py` can't compare their sizes directly.

## IoU per held-out fold

| Fold | Held-out site | NDVI rule | SVM (linear) | SVM (approx. RBF) | Random forest | U-Net | DINOv3 frozen | DINOv3 LoRA |
|---|---|---|---|---|---|---|---|---|
| 1 | eastern mangroves | 0.663 | 0.691 | 0.830 | 0.821 | 0.860 | 0.797 | 0.842 |
| 2 | jubail island | 0.456 | 0.442 | 0.558 | 0.562 | 0.553 | 0.550 | 0.550 |
| 3 | ras ghurab | 0.608 | 0.619 | 0.689 | 0.697 | 0.711 | 0.698 | 0.697 |
| 4 | ghantoot | 0.695 | 0.629 | 0.774 | 0.777 | 0.770 | 0.779 | 0.754 |
| 5 | al aryam islands + al nouf | 0.583 | 0.602 | 0.660 | 0.663 | 0.626 | 0.606 | 0.630 |
| Mean | | 0.601 | 0.597 | 0.702 | 0.704 | 0.704 | 0.686 | 0.694 |

## Pairs among the top five

A is the model with the higher mean. Difference = A minus B. The interval is a 95% paired bootstrap over the folds (we used all 3,125 possible resamples, since there are only 5 folds).

| A | B | Mean difference | 95% interval | Folds A wins |
|---|---|---|---|---|
| Random forest | SVM (approx. RBF) | 0.001 | -0.004 to 0.006 | 4 of 5 |
| U-Net | Random forest | 0.000 | -0.021 to 0.025 | 2 of 5 |
| U-Net | SVM (approx. RBF) | 0.002 | -0.017 to 0.022 | 2 of 5 |
| Random forest | DINOv3 LoRA | 0.009 | -0.008 to 0.025 | 3 of 5 |
| U-Net | DINOv3 LoRA | 0.010 | 0.002 to 0.017 | 4 of 5 |
| DINOv3 LoRA | DINOv3 frozen | 0.008 | -0.011 to 0.031 | 2 of 5 |
| Random forest | DINOv3 frozen | 0.018 | 0.001 to 0.039 | 3 of 5 |
| U-Net | DINOv3 frozen | 0.018 | 0.000 to 0.043 | 4 of 5 |

All 21 pairs (with Wilcoxon p-values) are in `pairwise.csv`, and the per-fold IoU is in `fold_iou.csv`.

We only count a difference as real if A wins every fold and the interval doesn't include 0. The interval on its own isn't enough here: U-Net vs LoRA has an interval above 0, but LoRA still wins one of the folds.

## Limits

- With 5 folds the Wilcoxon test can't go below p = 0.0625, even if one model wins every fold. After Holm correction for 21 pairs every p is 1.0. So the p-values are only there for completeness, the win counts and intervals are what matter.
- "Tied" means 5 sites can't separate them, not that they're proven equal. Per-chip scores (215 chips) would give a stronger test, but only per-fold scores are in the repo right now.
- All scores are against GMW, which is a machine-learning map itself. A model that copies GMW's mistakes still scores well.
- These are CV numbers only. The locked test sites are still unopened.

## How to run

```
cd src/mangrove_segmentation
python compare_models.py --out ../../results/mangrove_segmentation/model_comparison
python -m pytest -q tests/test_compare_models.py
```
