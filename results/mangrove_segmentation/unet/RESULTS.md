# Stage 5, M1: U-Net (gpu preset), leave-one-site-out CV

Train_Val_Dataset only (Test_Dataset never read). 2020 chips, label GMW v4.0.19 (10 m), hazy scenes dropped from training
(variant b, the same rule as the baselines). The same 5 folds, 1-chip buffer and scoring code (`evaluate.py`) as B1/B2.

## Headline (mean ± sd over 5 folds)

| Model                     | iou           | f1            | precision     | recall        |
|:--------------------------|:--------------|:--------------|:--------------|:--------------|
| U-Net (gpu preset)        | 0.704 ± 0.120 | 0.822 ± 0.083 | 0.791 ± 0.111 | 0.858 ± 0.059 |
| B2 random forest          | 0.704 ± 0.101 | 0.823 ± 0.071 | 0.808 ± 0.109 | 0.842 ± 0.032 |
| B1 NDVI + MNDWI threshold | 0.601 ± 0.092 | 0.747 ± 0.075 | 0.725 ± 0.133 | 0.786 ± 0.047 |

**Verdict: yes, it beats the random-forest bar (0.70)** (U-Net 0.704 vs B2 0.704, difference +0.000).
U-Net beats B2 on 2 of 5 folds. Its CI is entirely above B2's on 0 fold(s) and entirely below on 0 fold(s).
Pooled out-of-fold U-Net IoU is 0.686 (95% CI 0.652–0.716).

## Per fold

|   Fold | Held-out site            | U-Net IoU (95% CI)   | B2 IoU (95% CI)     |   U-Net − B2 | CIs overlap   |   B1 IoU |
|-------:|:-------------------------|:---------------------|:--------------------|-------------:|:--------------|---------:|
|      1 | eastern mangroves        | 0.860 (0.822–0.900)  | 0.821 (0.754–0.875) |        0.04  | yes           |    0.663 |
|      2 | jubail island            | 0.553 (0.486–0.614)  | 0.562 (0.489–0.626) |       -0.009 | yes           |    0.456 |
|      3 | ras ghurab               | 0.711 (0.680–0.740)  | 0.697 (0.664–0.724) |        0.015 | yes           |    0.608 |
|      4 | ghantoot                 | 0.770 (0.715–0.810)  | 0.777 (0.731–0.809) |       -0.007 | yes           |    0.695 |
|      5 | al aryam islands,al nouf | 0.626 (0.582–0.668)  | 0.663 (0.626–0.699) |       -0.037 | yes           |    0.583 |

## How it was trained

- **Model:** U-Net with 4 levels, base width 32, BatchNorm, 7,764,481 parameters, 8 input channels (B02 B03 B04 B08 B11 B12 NDVI MNDWI).
- **Normalisation:** per fold, from that fold's training chips only.
- **Loss:** BCE (pos_weight 4) + soft Dice, ignoring unlabelled, NaN and cloud pixels.
- **Augmentation:** random 256×256 crops (1 per chip per epoch), rot90 and flips, and a ±0.02 reflectance shift (NDVI/MNDWI recomputed).
- **Optimiser:** Adam, lr 0.001, cosine schedule, at most 100 epochs, early stopping (patience 15) on inner-validation IoU, batch 16.
- **Threshold:** chosen on the inner-validation fold (grid 0.05–0.95). Chosen per fold: F1 0.55, F2 0.75, F3 0.55, F4 0.5, F5 0.55.
- **Inner validation:** the same fold B2 used (first of 3, 4, 5, 2, 1 that is not the held-out fold).
- **Runtime:** 4 min in total on cuda. Best epochs: F1 33/48, F2 64/79, F3 44/59, F4 26/41, F5 40/55.

## What limits this run (what a bigger GPU run should change)

1. **No refit on all 4 training folds.** B2 was refitted on all 4 after tuning; the U-Net trains on 3 (one is kept for early stopping). On GPU, refit on all 4 folds for the best epoch count, or average the fold models.
2. **Small and short:** base width 16, 128-pixel crops, at most 100 epochs. GPU preset: width 32, full 256 crops, about 100 epochs.
3. **Few chips:** 56–107 training chips per fold after dropping hazy scenes and the buffer. More augmentation, or keeping the hazy scenes (B2 barely differed between a and b), may help.
4. **One seed per fold;** fold-to-fold spread is large (sd above), so differences under about 0.03 are within noise.

Files: `cv_results.csv`, `per_chip.csv`, `fold_fits.csv`, `history.csv`, `figs/` (iou_per_fold, training_curves, examples).
Model weights and predictions: `6_Work/runs/Unet_gpu/` (`unet_fold*.pt`, `preds/`).
