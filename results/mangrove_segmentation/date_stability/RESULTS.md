# Model steadiness across dates, and how sure we are of the ~0.70

Uses `per_chip_scores.csv` (TP/FP/FN per chip, date and model; U-Net and frozen DINOv3 are retrains). Code: `src/mangrove_segmentation/date_stability.py`.

## Across image dates

IoU pooled per site and date (24 site-dates in 2020), then per model:

| Model | Mean | SD | Min | Max | Worst site-date |
|---|---|---|---|---|---|
| Random forest | 0.692 | 0.101 | 0.502 | 0.889 | Jubail 2020-08-24 |
| SVM (approx. RBF) | 0.689 | 0.104 | 0.472 | 0.866 | Jubail 2020-08-24 |
| U-Net | 0.691 | 0.138 | 0.365 | 0.891 | Al Aryam 2020-11-12 |
| DINOv3 LoRA | 0.671 | 0.129 | 0.332 | 0.907 | Al Aryam 2020-11-12 |
| DINOv3 frozen | 0.660 | 0.128 | 0.369 | 0.864 | Al Aryam 2020-11-12 |

RF and SVM move the least between dates and never drop below about 0.47. The U-Net and both DINOv3 models have a higher spread and all hit their worst date on the same image (Al Aryam, 12 Nov 2020), so that image is the first one to open. Per-site spreads are in `per_site_spread.csv`, every site-date in `iou_by_site_date.csv`.

## Random subsets (how sure are we of the 0.70?)

Chip locations resampled within each fold 2,000 times (all dates of a spot kept together), mean-over-folds IoU each time:

| Model | Mean IoU | SD | 95% interval | Lowest | Highest | Range |
|---|---|---|---|---|---|---|
| Random forest | 0.704 | 0.016 | 0.668 to 0.728 | 0.639 | 0.747 | 0.108 |
| SVM (approx. RBF) | 0.702 | 0.016 | 0.666 to 0.728 | 0.633 | 0.745 | 0.112 |
| U-Net | 0.713 | 0.017 | 0.675 to 0.741 | 0.643 | 0.761 | 0.118 |
| DINOv3 LoRA | 0.694 | 0.021 | 0.645 to 0.729 | 0.586 | 0.750 | 0.164 |
| DINOv3 frozen | 0.685 | 0.019 | 0.641 to 0.716 | 0.606 | 0.736 | 0.131 |

The ~0.70 is good to about ±0.035 (95%). The foundation models don't have a tighter range here, it's the other way round: RF and SVM are the tightest and LoRA the widest.

## Limits

- 24 site-dates, all from 2020, and only one model run each.
- Scores are against GMW, so a steady model can be steadily copying GMW's mistakes.
