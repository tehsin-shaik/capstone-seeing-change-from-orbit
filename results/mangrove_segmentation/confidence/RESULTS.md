# How sure are we of the IoU numbers (Train/Val only)

Our supervisor asked how confident we are in the IoU numbers. Here we take the held-out predictions of every model (5 leave-one-site-out folds, hazy scenes dropped from training, 215 chips from 2020, scored against GMW v4.0.19) and look at how much the IoU moves across image dates and sites, how much it moves when we resample the chips, and how much it moves when we train again with another seed. No model was trained for tables A to D. For table E we trained the main models again. The script is `src/mangrove_segmentation/confidence.py`. Test_Dataset was not opened.

The first seven rows are the runs with all 8 channels (for DINOv3 that means RGB into DINOv3 and all 8 channels into the decoder branch). The last five are the RGB only runs. The U-Net and frozen DINOv3 rows in tables A to D use our retrain on our own PC, because we do not have the masks of the committed runs. For the U-Net this matters: the retrain has a mean IoU of 0.713 and the committed run 0.704.

## Table A: spread over the 24 site and date combinations

IoU pooled over the chips of one site on one date, then the mean, sd, min and max over the 24 combinations. The sd is the sample sd in all tables.

| Model                              | Mean   | SD    | Min   | Max   | Lowest site and date   | Highest site and date    |
|:-----------------------------------|:-------|:------|:------|:------|:-----------------------|:-------------------------|
| NDVI rule                          | 0.591  | 0.135 | 0.138 | 0.763 | Al Aryam 12 Nov        | Eastern Mangroves 16 Feb |
| SVM linear                         | 0.578  | 0.153 | 0.140 | 0.780 | Al Aryam 12 Nov        | Eastern Mangroves 16 Feb |
| SVM RBF                            | 0.689  | 0.104 | 0.472 | 0.866 | Jubail 24 Aug          | Eastern Mangroves 17 Nov |
| Random forest                      | 0.692  | 0.101 | 0.502 | 0.889 | Jubail 24 Aug          | Eastern Mangroves 17 Nov |
| U-Net                              | 0.691  | 0.138 | 0.365 | 0.891 | Al Aryam 12 Nov        | Eastern Mangroves 18 May |
| DINOv3 + LoRA                      | 0.670  | 0.129 | 0.332 | 0.907 | Al Aryam 12 Nov        | Eastern Mangroves 17 Nov |
| DINOv3 frozen                      | 0.660  | 0.128 | 0.369 | 0.864 | Al Aryam 12 Nov        | Eastern Mangroves 17 Nov |
| Random forest, RGB                 | 0.532  | 0.129 | 0.122 | 0.701 | Al Aryam 12 Nov        | Ghantoot 9 Nov           |
| SVM RBF, RGB                       | 0.506  | 0.130 | 0.099 | 0.676 | Al Aryam 12 Nov        | Ghantoot 18 May          |
| DINOv3 + LoRA, RGB, branch removed | 0.553  | 0.136 | 0.268 | 0.793 | Al Aryam 12 Nov        | Eastern Mangroves 16 Feb |
| DINOv3 + LoRA, RGB, branch fed RGB | 0.598  | 0.132 | 0.221 | 0.813 | Al Aryam 12 Nov        | Eastern Mangroves 16 Feb |
| DINOv3 frozen, RGB, branch removed | 0.512  | 0.154 | 0.144 | 0.709 | Al Aryam 12 Nov        | Ghantoot 31 Aug          |

## Table B: by time of year and by site

Each site has four images from 2020, not three, and the days are not the same for every site, so we grouped them by time of year. IoU pooled over all chips of a group.

| Model                              | Jan-Feb   | Apr-May   | Aug-Sep   | Nov   | Mean   | SD    | Min   | Max   |
|:-----------------------------------|:----------|:----------|:----------|:------|:-------|:------|:------|:------|
| NDVI rule                          | 0.653     | 0.601     | 0.570     | 0.525 | 0.587  | 0.054 | 0.525 | 0.653 |
| SVM linear                         | 0.636     | 0.605     | 0.509     | 0.561 | 0.578  | 0.055 | 0.509 | 0.636 |
| SVM RBF                            | 0.716     | 0.689     | 0.663     | 0.672 | 0.685  | 0.023 | 0.663 | 0.716 |
| Random forest                      | 0.731     | 0.681     | 0.653     | 0.676 | 0.685  | 0.033 | 0.653 | 0.731 |
| U-Net                              | 0.743     | 0.707     | 0.662     | 0.665 | 0.694  | 0.038 | 0.662 | 0.743 |
| DINOv3 + LoRA                      | 0.712     | 0.674     | 0.662     | 0.656 | 0.676  | 0.025 | 0.656 | 0.712 |
| DINOv3 frozen                      | 0.708     | 0.679     | 0.670     | 0.646 | 0.676  | 0.026 | 0.646 | 0.708 |
| Random forest, RGB                 | 0.623     | 0.506     | 0.537     | 0.562 | 0.557  | 0.049 | 0.506 | 0.623 |
| SVM RBF, RGB                       | 0.549     | 0.552     | 0.515     | 0.509 | 0.531  | 0.022 | 0.509 | 0.552 |
| DINOv3 + LoRA, RGB, branch removed | 0.598     | 0.545     | 0.581     | 0.585 | 0.577  | 0.023 | 0.545 | 0.598 |
| DINOv3 + LoRA, RGB, branch fed RGB | 0.652     | 0.575     | 0.625     | 0.610 | 0.616  | 0.032 | 0.575 | 0.652 |
| DINOv3 frozen, RGB, branch removed | 0.567     | 0.528     | 0.561     | 0.568 | 0.556  | 0.019 | 0.528 | 0.568 |

| Model                              | Al Aryam   | Al Nouf   | Eastern Mangroves   | Ghantoot   | Jubail   | Ras Ghurab   | Mean   | SD    | Min   | Max   |
|:-----------------------------------|:-----------|:----------|:--------------------|:-----------|:---------|:-------------|:-------|:------|:------|:------|
| NDVI rule                          | 0.501      | 0.665     | 0.663               | 0.695      | 0.456    | 0.608        | 0.598  | 0.098 | 0.456 | 0.695 |
| SVM linear                         | 0.543      | 0.663     | 0.691               | 0.629      | 0.443    | 0.619        | 0.598  | 0.091 | 0.443 | 0.691 |
| SVM RBF                            | 0.648      | 0.671     | 0.830               | 0.774      | 0.558    | 0.689        | 0.695  | 0.096 | 0.558 | 0.830 |
| Random forest                      | 0.638      | 0.684     | 0.821               | 0.777      | 0.562    | 0.697        | 0.696  | 0.093 | 0.562 | 0.821 |
| U-Net                              | 0.611      | 0.701     | 0.867               | 0.785      | 0.552    | 0.706        | 0.703  | 0.114 | 0.552 | 0.867 |
| DINOv3 + LoRA                      | 0.584      | 0.671     | 0.842               | 0.754      | 0.550    | 0.697        | 0.683  | 0.108 | 0.550 | 0.842 |
| DINOv3 frozen                      | 0.548      | 0.655     | 0.795               | 0.779      | 0.551    | 0.699        | 0.671  | 0.107 | 0.548 | 0.795 |
| Random forest, RGB                 | 0.442      | 0.501     | 0.610               | 0.663      | 0.467    | 0.581        | 0.544  | 0.087 | 0.442 | 0.663 |
| SVM RBF, RGB                       | 0.408      | 0.462     | 0.556               | 0.645      | 0.438    | 0.560        | 0.511  | 0.090 | 0.408 | 0.645 |
| DINOv3 + LoRA, RGB, branch removed | 0.457      | 0.477     | 0.715               | 0.673      | 0.461    | 0.598        | 0.564  | 0.115 | 0.457 | 0.715 |
| DINOv3 + LoRA, RGB, branch fed RGB | 0.550      | 0.578     | 0.752               | 0.697      | 0.526    | 0.592        | 0.616  | 0.089 | 0.526 | 0.752 |
| DINOv3 frozen, RGB, branch removed | 0.384      | 0.416     | 0.677               | 0.641      | 0.465    | 0.582        | 0.527  | 0.123 | 0.384 | 0.677 |

## Table C: bootstrap over the held-out chips

Chips are drawn again with replacement inside each held-out site, 2,000 times. Each time TP, FP and FN are pooled per fold and the IoU is averaged over the 5 folds. The last two columns do the same but draw chip positions and keep the four dates of a position together, because the same ground on four dates is not four independent chips.

| Model                              | Mean IoU, 5 folds   | Bootstrap mean   | SD    | 95% interval   | Width   | 95% interval, dates of a position kept together   | Width    |
|:-----------------------------------|:--------------------|:-----------------|:------|:---------------|:--------|:--------------------------------------------------|:---------|
| NDVI rule                          | 0.601               | 0.600            | 0.013 | 0.574 to 0.625 | 0.051   | 0.553 to 0.634                                    | 0.081    |
| SVM linear                         | 0.597               | 0.596            | 0.015 | 0.566 to 0.624 | 0.057   | 0.556 to 0.627                                    | 0.071    |
| SVM RBF                            | 0.702               | 0.701            | 0.011 | 0.680 to 0.721 | 0.041   | 0.665 to 0.729                                    | 0.065    |
| Random forest                      | 0.704               | 0.703            | 0.011 | 0.679 to 0.725 | 0.045   | 0.667 to 0.731                                    | 0.064    |
| U-Net                              | 0.713               | 0.713            | 0.011 | 0.690 to 0.734 | 0.044   | 0.674 to 0.741                                    | 0.068    |
| DINOv3 + LoRA                      | 0.694               | 0.693            | 0.013 | 0.667 to 0.718 | 0.051   | 0.647 to 0.729                                    | 0.082    |
| DINOv3 frozen                      | 0.685               | 0.684            | 0.012 | 0.659 to 0.708 | 0.049   | 0.642 to 0.719                                    | 0.076    |
| Random forest, RGB                 | 0.559               | 0.558            | 0.013 | 0.531 to 0.582 | 0.051   | 0.522 to 0.584                                    | 0.061    |
| SVM RBF, RGB                       | 0.527               | 0.526            | 0.015 | 0.496 to 0.555 | 0.058   | 0.482 to 0.560                                    | 0.078    |
| DINOv3 + LoRA, RGB, branch removed | 0.583               | 0.582            | 0.014 | 0.555 to 0.609 | 0.054   | 0.534 to 0.622                                    | 0.088    |
| DINOv3 + LoRA, RGB, branch fed RGB | 0.626               | 0.625            | 0.015 | 0.596 to 0.655 | 0.059   | 0.573 to 0.667                                    | 0.095    |
| DINOv3 frozen, RGB, branch removed | 0.553               | 0.551            | 0.013 | 0.525 to 0.578 | 0.053   | 0.501 to 0.590                                    | 0.089    |

## Table D: paired bootstrap of the difference between two models

Both models are scored on the same resampled chips each time.

| First model   | Second model                       | First minus second   | 95% interval     | Includes zero   | 95% interval, positions   | Includes zero    |
|:--------------|:-----------------------------------|:---------------------|:-----------------|:----------------|:--------------------------|:-----------------|
| Random forest | DINOv3 + LoRA                      | +0.009               | -0.004 to +0.024 | yes             | -0.009 to +0.033          | yes              |
| Random forest | U-Net                              | -0.009               | -0.024 to +0.004 | yes             | -0.018 to -0.001          | no               |
| Random forest | SVM RBF                            | +0.002               | -0.005 to +0.009 | yes             | -0.005 to +0.009          | yes              |
| SVM RBF       | SVM linear                         | +0.105               | +0.084 to +0.127 | no              | +0.085 to +0.127          | no               |
| Random forest | Random forest, RGB                 | +0.145               | +0.124 to +0.170 | no              | +0.120 to +0.172          | no               |
| SVM RBF       | SVM RBF, RGB                       | +0.175               | +0.148 to +0.204 | no              | +0.145 to +0.207          | no               |
| DINOv3 + LoRA | DINOv3 + LoRA, RGB, branch removed | +0.112               | +0.096 to +0.128 | no              | +0.092 to +0.129          | no               |
| DINOv3 + LoRA | DINOv3 + LoRA, RGB, branch fed RGB | +0.068               | +0.054 to +0.082 | no              | +0.049 to +0.086          | no               |
| DINOv3 frozen | DINOv3 frozen, RGB, branch removed | +0.133               | +0.117 to +0.152 | no              | +0.111 to +0.158          | no               |

## Table E: training again with other seeds

Same 5 folds and same settings, only the seed changes. Seed 0 is the committed run. For the random forest and the SVM the seed is the number itself. The U-Net and DINOv3 scripts seed fold k with the number k, so for them seed N means fold k is seeded with k + 100 x N. Each seed column is the mean IoU over the 5 folds. SD over folds is the sd of the 5 fold IoUs, averaged over the seeds. SD over seeds, one fold is the sd over the seeds of the IoU of a single fold, averaged over the 5 folds. The last two columns are the check we did first: the committed setting trained again on our PC, and the largest change this gave on a single fold.

| Model                              | Seed 0   | Seed 1   | Seed 2   | Seed 3   | Mean over seeds   | SD over seeds   | SD over folds   | SD over seeds, one fold   | Seed 0 run again   | Largest fold change, seed 0 again   |
|:-----------------------------------|:---------|:---------|:---------|:---------|:------------------|:----------------|:----------------|:--------------------------|:-------------------|:------------------------------------|
| Random forest                      | 0.7039   | 0.7006   | 0.6998   | 0.7031   | 0.7018            | 0.0019          | 0.1022          | 0.0039                    | 0.7039             | 0.0000                              |
| Random forest, RGB                 | 0.5590   | 0.5583   | 0.5597   | 0.5589   | 0.5590            | 0.0006          | 0.0858          | 0.0022                    | 0.5590             | 0.0000                              |
| SVM RBF                            | 0.7024   | 0.7024   | 0.7019   |          | 0.7022            | 0.0003          | 0.1050          | 0.0010                    | 0.7024             | 0.0000                              |
| SVM RBF, RGB                       | 0.5274   | 0.5275   | 0.5278   |          | 0.5276            | 0.0002          | 0.0891          | 0.0011                    | 0.5274             | 0.0000                              |
| U-Net                              | 0.7042   | 0.7118   | 0.7055   |          | 0.7071            | 0.0041          | 0.1208          | 0.0125                    | 0.7048             | 0.0170                              |
| DINOv3 + LoRA                      | 0.6944   | 0.6873   | 0.6912   |          | 0.6910            | 0.0036          | 0.1130          | 0.0145                    | 0.6943             | 0.0029                              |
| DINOv3 + LoRA, RGB, branch removed | 0.5829   | 0.5815   | 0.5449   |          | 0.5698            | 0.0216          | 0.0961          | 0.0277                    | 0.5826             | 0.0017                              |
| DINOv3 + LoRA, RGB, branch fed RGB | 0.6264   | 0.6259   | 0.6167   |          | 0.6230            | 0.0055          | 0.0945          | 0.0176                    | 0.6270             | 0.0316                              |

The random forest and the SVM give exactly the committed IoU on every fold when we run them again. The U-Net and DINOv3 + LoRA do not, because training on the GPU is not exactly repeatable: the mean comes back within 0.001, but a single fold moves by up to 0.017 for the U-Net and 0.032 for DINOv3 + LoRA with the branch fed RGB. Our two earlier U-Net retrains with the committed seed gave 0.713 and 0.706, so for the U-Net a new seed (0.712 and 0.706) changes no more than running the same seed again. The 0.022 for DINOv3 + LoRA with the branch removed comes mostly from fold 1, which holds out Eastern Mangroves with only 8 chips and went from 0.715 to 0.664 to 0.544 over the three seeds.

## What the tables show

The bootstrap puts each mean IoU inside about ±0.02 to ±0.03 (95% width 0.04 to 0.06), or ±0.03 to ±0.05 when the dates of a position are kept together, and both are much smaller than the spread between places (sd 0.10 to 0.15 over site and date, 0.09 to 0.12 over the 5 folds). Training again with another seed moves the mean IoU less than that for every model but one: the sd over seeds is 0.002 or less for the random forest and the SVM, about 0.004 for the U-Net and DINOv3 + LoRA, 0.006 for DINOv3 + LoRA with the branch fed RGB, and 0.022 with the branch removed, which is as much as the bootstrap. The differences that are real are RBF SVM over linear SVM (+0.106) and every 8 channel run over its RGB run (+0.068 to +0.175): their intervals stay well away from zero, and the 8 channel run is above the RGB run for every seed we trained. The differences between the random forest and the RBF SVM, the U-Net and DINOv3 + LoRA are all 0.01 or less and inside the noise (their means over the seeds are 0.691 to 0.707), with only random forest against U-Net borderline, and that row uses our U-Net retrain, which is 0.009 above the committed U-Net. No model has a clearly tighter bootstrap range than another, but the random forest and the RBF SVM are the steadiest in the other two ways: across site and date (sd 0.10, never under 0.47, while the U-Net and both DINOv3 models drop to 0.33 to 0.37 on their worst image) and across seeds. Jubail is the hardest site for almost every 8 channel model (0.55 to 0.56 for the stronger ones) and Eastern Mangroves the easiest for all of them except the NDVI rule, the hardest single image is Al Aryam on 12 Nov for most models (only 2 chips) and Jubail on 24 Aug, a hazy scene, for the random forest and the RBF SVM, and by time of year Jan-Feb is the best for every 8 channel model and Aug-Sep or Nov the worst.

## What this does not show

The bootstrap only answers what would happen with a different draw of chips from the same held-out sites. It says nothing about new sites (we have 5 folds and their sd is about 0.10) or about errors in the GMW labels. Some site and date groups are very small (Al Nouf on 16 May is 1 chip, Al Aryam on 12 Nov is 2), so their IoU is not reliable on its own. The Aug-Sep images are hazy at five of the six sites. Table E has only three or four seeds per model, so its sd is rough, and for the GPU models it also contains the change we get from running the same seed twice. The mean over 5 folds gives the 8 chips of Eastern Mangroves the same weight as the 80 chips of Jubail, which is why one small fold can move a mean.

## How to get the numbers again

`confidence.py scores` adds the NDVI rule and the linear SVM to the per chip file (both gave the committed IoU on every fold), and `confidence.py tables` writes the CSVs of tables A to D from it. `confidence.py seed` trains one model with one seed without changing the training scripts, and `confidence.py seed-table` writes table E from the run folders. `table_A_site_date_iou.csv` has the IoU of every site and date, and `table_E_per_fold.csv` the IoU of every fold and seed. The per chip file, weights, masks and logs are kept outside the repo.
