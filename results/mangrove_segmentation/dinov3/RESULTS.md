# DINOv3 (frozen), leave-one-site-out CV

Train_Val_Dataset only (Test_Dataset never read). 2020 chips, label GMW v4.0.19 (10 m), hazy scenes dropped from training
(variant b), the same 5 folds, 1-chip buffer and scoring code as B1, B2 and the U-Net.

**Verdict:** **No: does not beat the random forest yet** (DINOv3 0.686 vs B2 0.704)

## Headline (mean ± sd over folds)

| Model             | iou           | f1            | precision     | recall        |
|:------------------|:--------------|:--------------|:--------------|:--------------|
| DINOv3 frozen     | 0.686 ± 0.107 | 0.810 ± 0.077 | 0.833 ± 0.130 | 0.801 ± 0.079 |
| B1 NDVI threshold | 0.601 ± 0.092 | 0.747 ± 0.075 | 0.725 ± 0.133 | 0.786 ± 0.047 |
| B2 random forest  | 0.704 ± 0.101 | 0.823 ± 0.071 | 0.808 ± 0.109 | 0.842 ± 0.032 |
| U-Net             | 0.704 ± 0.120 | 0.822 ± 0.083 | 0.791 ± 0.111 | 0.858 ± 0.059 |

## Per fold

|   Fold | Held-out site            | DINOv3 IoU (95% CI)   |   B2 IoU |   DINOv3 − B2 | CIs overlap   |   U-Net IoU |   B1 IoU |
|-------:|:-------------------------|:----------------------|---------:|--------------:|:--------------|------------:|---------:|
|      1 | eastern mangroves        | 0.797 (0.753–0.838)   |    0.821 |        -0.024 | yes           |       0.86  |    0.663 |
|      2 | jubail island            | 0.550 (0.479–0.614)   |    0.562 |        -0.011 | yes           |       0.553 |    0.456 |
|      3 | ras ghurab               | 0.698 (0.672–0.721)   |    0.697 |         0.002 | yes           |       0.711 |    0.608 |
|      4 | ghantoot                 | 0.779 (0.730–0.812)   |    0.777 |         0.002 | yes           |       0.77  |    0.695 |
|      5 | al aryam islands,al nouf | 0.606 (0.528–0.674)   |    0.663 |        -0.057 | yes           |       0.626 |    0.583 |

## Setup

- **Backbone:** facebook/dinov3-vitl16-pretrain-sat493m, frozen;
  RGB = B04/B03/B02 stretched 2–98 % per chip, upsampled x2 (token = 80 m).
- **Decoder:** 1x1 projection of the DINOv3 tokens to 64 channels, upsampled to 10 m,
  + a light conv branch over the 8 spectral channels (B02–B12, NDVI, MNDWI), then 2 conv layers (32 channels).
- **Training:** BCE (pos_weight 4) + Dice, AdamW, cosine schedule, up to 40 epochs, early stopping (patience 10)
  on inner-validation IoU, batch 8, augmentation: brightness ±0.02 + flips. Device: cuda.
- **Threshold:** chosen on the inner-validation fold (grid 0.05–0.95): F1 0.55, F2 0.7, F3 0.5, F4 0.45, F5 0.6.
- **Runtime:** 2 min of training (feature extraction not included).

Files: `cv_results.csv`, `per_chip.csv`, `fold_fits.csv`, `history.csv`, `config.json`, `figs/`, `dinov3_fold*.pt`, `preds/`.
