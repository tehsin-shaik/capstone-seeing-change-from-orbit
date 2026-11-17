# DINOv3 with LoRA: leave-one-site-out results (Train/Val only)

Label: GMW v4.0.19 (10 m). 2020 chips only (215 chips, 6 Abu Dhabi sites, 5 folds). Hazy scenes are dropped from training (variant b). Test_Dataset was not opened. The folds, the one-chip buffer and the scoring code (`evaluate.py`) are the same as for the other models. The code is `src/mangrove_segmentation/dinov3/train_dinov3_lora.py`, run once on all five folds with its default settings.

LoRA does not beat the random forest. Its mean IoU is 0.694 and the random forest's is 0.704. LoRA is 0.008 above the frozen DINOv3 run (0.686), which is inside the noise.

## Headline (mean ± sd across the 5 held-out folds, variant b)

| Model | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| B1 NDVI + MNDWI threshold | 0.601 ± 0.092 | 0.747 ± 0.075 | 0.725 ± 0.133 | 0.786 ± 0.047 |
| B2 random forest | 0.704 ± 0.101 | 0.823 ± 0.071 | 0.808 ± 0.109 | 0.842 ± 0.032 |
| B3 SVM, linear | 0.597 ± 0.093 | 0.744 ± 0.077 | 0.760 ± 0.115 | 0.740 ± 0.103 |
| B3 SVM, approximate RBF | 0.702 ± 0.105 | 0.822 ± 0.074 | 0.784 ± 0.099 | 0.865 ± 0.052 |
| U-Net | 0.704 ± 0.120 | 0.822 ± 0.083 | 0.791 ± 0.111 | 0.858 ± 0.059 |
| DINOv3 frozen | 0.686 ± 0.107 | 0.810 ± 0.077 | 0.833 ± 0.130 | 0.801 ± 0.079 |
| DINOv3 LoRA | 0.694 ± 0.112 | 0.815 ± 0.079 | 0.828 ± 0.123 | 0.811 ± 0.065 |

The other rows are computed from the `cv_results.csv` files in `../baselines`, `../svm`, `../unet` and `../dinov3`. The sd is the sample sd (divide by n minus 1). Every model was scored on the same chips: the chip counts and mangrove pixel counts per fold are identical in all the tables.

## Per fold (IoU on the held-out site, variant b)

| Fold | Held-out site | Chips | B1 NDVI | B2 random forest | B3 SVM linear | B3 SVM approx. RBF | U-Net | DINOv3 frozen | DINOv3 LoRA |
|---|---|---|---|---|---|---|---|---|---|
| 1 | eastern mangroves | 8 | 0.663 | 0.821 | 0.691 | 0.830 | 0.860 | 0.797 | 0.842 |
| 2 | jubail island | 80 | 0.456 | 0.562 | 0.442 | 0.558 | 0.553 | 0.550 | 0.550 |
| 3 | ras ghurab | 50 | 0.608 | 0.697 | 0.619 | 0.689 | 0.711 | 0.698 | 0.697 |
| 4 | ghantoot | 41 | 0.695 | 0.777 | 0.629 | 0.774 | 0.770 | 0.779 | 0.754 |
| 5 | al aryam islands,al nouf | 36 | 0.583 | 0.663 | 0.602 | 0.660 | 0.626 | 0.606 | 0.630 |
| Mean | | | 0.601 | 0.704 | 0.597 | 0.702 | 0.704 | 0.686 | 0.694 |

LoRA is above the random forest on fold 1 (+0.021), level on fold 3 and below it on folds 2, 4 and 5 (-0.012, -0.024, -0.033). Its 95% interval overlaps the random forest's and the U-Net's on every fold.

## LoRA minus frozen (same folds, same scoring)

| Fold | Held-out site | Frozen IoU (95% CI) | LoRA IoU (95% CI) | LoRA − frozen | CIs overlap |
|---|---|---|---|---|---|
| 1 | eastern mangroves | 0.797 (0.753–0.838) | 0.842 (0.774–0.899) | +0.045 | yes |
| 2 | jubail island | 0.550 (0.479–0.614) | 0.550 (0.486–0.606) | -0.001 | yes |
| 3 | ras ghurab | 0.698 (0.672–0.721) | 0.697 (0.673–0.719) | -0.001 | yes |
| 4 | ghantoot | 0.779 (0.730–0.812) | 0.754 (0.687–0.796) | -0.025 | yes |
| 5 | al aryam islands,al nouf | 0.606 (0.528–0.674) | 0.630 (0.561–0.685) | +0.024 | yes |
| Mean | | 0.686 | 0.694 | +0.008 | |

LoRA is higher on two folds, lower on one and level on two. The largest gain is on fold 1, which has only 8 chips and is the least stable fold.

## Caveats

- There is one seed per fold. The spread between folds is large (sd 0.11), so a difference under about 0.03 between two models is within noise. The mean difference from frozen is 0.008.
- The 95% intervals of LoRA and frozen overlap on every fold. No fold shows a clear win for either.
- In this run folds 2 and 3 ran all 40 epochs. Training stopped there because of the cap, not because the inner-validation IoU had stopped improving for 10 epochs. Their best epochs were 36 and 32. This was checked afterwards with an 80-epoch run (see the 80-epoch section below). The cap cost little: the mean IoU went from 0.694 to 0.700.
- The default settings were not tuned on our data. Rank 8, alpha 16, dropout 0.05, the query and value projections, all 24 blocks and the two learning rates are the values written in the code.
- As in the frozen run and the U-Net, each model trains on three folds and the fourth is kept for early stopping and for choosing the threshold. The random forest is refitted on all four.

## 80-epoch run

The run was repeated with `--epochs 80` and nothing else changed, to see whether the 40-epoch cap held the model back. All the other tables in this file are the 40-epoch run. The 80-epoch tables are in `epochs80/`.

| Fold | Held-out site | 40-epoch IoU | 80-epoch IoU | Difference | Best epoch of epochs run, 40-epoch run | Best epoch of epochs run, 80-epoch run |
|---|---|---|---|---|---|---|
| 1 | eastern mangroves | 0.842 | 0.844 | +0.003 | 20 of 30 | 20 of 30 |
| 2 | jubail island | 0.550 | 0.574 | +0.024 | 36 of 40 | 42 of 52 |
| 3 | ras ghurab | 0.697 | 0.700 | +0.002 | 32 of 40 | 28 of 38 |
| 4 | ghantoot | 0.754 | 0.754 | 0.000 | 11 of 21 | 11 of 21 |
| 5 | al aryam islands,al nouf | 0.630 | 0.627 | -0.003 | 26 of 36 | 44 of 54 |
| Mean | | 0.694 | 0.700 | +0.005 | | |

No fold reached 80 epochs. All five stopped early because the inner-validation IoU had not improved for 10 epochs. The mean IoU went from 0.694 to 0.700, and only fold 2 moved by more than 0.003. So the epoch cap caveat is answered: the cap was not what held LoRA back. There is still one seed per fold, the 95% intervals of the two runs overlap on every fold, and 0.700 is still just under the random forest's 0.704.

## Setup

- Backbone: `facebook/dinov3-vitl16-pretrain-sat493m` (ViT-L). LoRA with rank 8 and alpha 16 on `q_proj` and `v_proj` in all 24 blocks. That is 48 adapted layers and 786,432 trainable numbers, 0.26% of the backbone.
- Input to DINOv3: B04, B03, B02 stretched 2 to 98% per chip and upsampled by 2, so one token covers 80 m.
- Decoder: the same as the frozen run. It also takes the 8 spectral channels through a small convolution branch.
- Training: BCE (positive weight 4) plus Dice, AdamW, cosine schedule, up to 40 epochs, patience 10 on inner-validation IoU, 8 chips per update, flips and a ±0.02 brightness shift. Learning rate 0.001 for the decoder and 0.0001 for LoRA. Mixed precision and gradient checkpointing were on.
- Threshold chosen on the inner-validation fold (grid 0.05 to 0.95): 0.55, 0.85, 0.55, 0.60, 0.45 for folds 1 to 5.
- Best epoch out of epochs run: 20 of 30, 36 of 40, 32 of 40, 11 of 21, 26 of 36.
- Cost on an 8 GB RTX 5060 Ti: 36 minutes of training for the five folds, 2,916 MB peak GPU memory, about 13 seconds per epoch. The frozen run took 2 minutes of training after its features were cached.

## Model size next to score

This is the table printed by `src/mangrove_segmentation/model_size.py`. It counts the numbers each model learns. The neural networks are built from the training code with random weights and counted directly. IoU is the mean over the five folds, variant b.

| Model | Total parameters | Trainable parameters | Mean IoU |
|---|---|---|---|
| NDVI rule | 2 | 2 | 0.601 |
| Random forest (depth None, leaf 1) | 119,999,600 | 119,999,600 | 0.704 |
| Random forest (depth 20, leaf 5) | 23,999,600 | 23,999,600 | 0.704 |
| Random forest (depth 12, leaf 10) | 2,457,200 | 2,457,200 | 0.704 |
| SVM (linear) | 9 | 9 | 0.597 |
| U-Net | 7,764,481 | 7,764,481 | 0.704 |
| DINOv3 frozen | 303,236,993 | 107,393 | 0.686 |
| DINOv3 + LoRA | 304,023,425 | 893,825 | 0.694 |

- The three random forest rows are upper limits, not counts. The size of a forest is only known after it is fitted, and the fitted forests are not in the repository. The three rows are the three settings the tuning chooses between. One forest fitted outside the repository with Jubail held out (depth 12, leaf 10) was counted with `--rf-model` and has 376,088 numbers.
- The DINOv3 rows are the ViT-L backbone (303,129,600, frozen) plus the decoder (107,393). LoRA adds 786,432 trainable numbers to that.
- LoRA trains about 8 times as many numbers as the frozen run and scores 0.008 higher. The U-Net trains about 9 times as many as LoRA and scores 0.010 higher. Both differences are within noise.
- The approximate RBF SVM is not in the script's table.

## Files

- `cv_results.csv`. One row per fold with IoU, F1, precision, recall, the IoU CI and the IoU per site. The columns are the same as in the other models' tables. The `model` column says `DINOv3_LoRA`.
- `fold_fits.csv`. Per fold: training and inner-validation chips, best epoch, epochs run, the chosen threshold and the minutes taken.
- `history.csv`. Training loss and inner-validation IoU for every epoch.
- `lora_vs_frozen.csv`. The LoRA minus frozen table above.
- `config.json`. The settings of the run, with peak GPU memory and seconds per epoch for each fold.
- `iou_per_fold.png`, `training_curves.png`, `examples.png`.
- `epochs80/cv_results.csv` and `epochs80/fold_fits.csv`. The same two tables for the 80-epoch run.
- The scores per chip, the predicted masks and the model weights are not in the repository.
