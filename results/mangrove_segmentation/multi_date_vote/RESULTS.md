# Multi-date vote check

Each chip cell of a held-out site is seen on up to three non-hazy dates in 2020. For each model, the masks of those dates were combined pixel by pixel: a pixel is mangrove if more than half of the dates say so. The combined mask was then scored against GMW v4.0.19 once per cell. The combining step never looks at the labels. The models are the random forest (B2), the approximate RBF SVM (B3) and DINOv3 with LoRA (80-epoch run), all with hazy scenes dropped from training. Combining dates raises the IoU a little, by 0.02 to 0.04, but it does not remove the ceiling near 0.70. The code is `src/mangrove_segmentation/multi_date_vote.py`. Train_Val_Dataset only.

## One date against all dates

All 75 cells, mean over the 5 folds. "One date" is the average over the cell's single dates, on the same pixels.

| Model | IoU, one date → all dates | Precision | Recall |
|---|---|---|---|
| Random forest | 0.714 → 0.752 | 0.805 → 0.839 | 0.860 → 0.874 |
| RBF SVM | 0.708 → 0.741 | 0.781 → 0.809 | 0.880 → 0.894 |
| DINOv3 + LoRA | 0.707 → 0.722 | 0.829 → 0.867 | 0.829 → 0.810 |

Most of the gain is precision: a false alarm that shows up on one date gets voted out. The per-site numbers are in `one_vs_all_per_site.csv`. Jubail Island stays near 0.62 for all three models.

## One, two and three dates

The 31 cells that have three non-hazy dates, mean over folds, same cells and same pixels at each step.

| Model | 1 date | 2 dates | 3 dates |
|---|---|---|---|
| Random forest | 0.693 | 0.709 | 0.728 |
| RBF SVM | 0.686 | 0.712 | 0.726 |
| DINOv3 + LoRA | 0.638 | 0.623 or 0.652 | 0.673 |

Three dates are better than one at every site for every model. With two dates the two masks can disagree. For the random forest and the SVM the model's average score then decides. No scores were saved for LoRA, so it has two numbers: a disagreement counted as not mangrove (0.623) and as mangrove (0.652). In the first table LoRA uses not mangrove.

## What it doesn't show

- These numbers cannot be compared with the reported 0.70. Here hazy scenes are left out and each cell counts once, so one date already gives 0.71. Compare inside these tables only.
- It doesn't show that the ceiling is gone. The errors that are left are the ones that repeat on every date, and this check cannot say whether they are label errors or model errors.
- It doesn't carry over to the 2018 and 2025 change maps as it is. Those years have two dates each, where the gain is smaller and depends on how a disagreement is settled.
- Some of it rests on few cells. Eastern Mangroves has 2 cells, and fold 5 in the second table is a single Al Aryam cell. There is one seed per fold, and only voting was tried, not a model trained on several dates.

The other tables are the CSVs in this folder (`cells_per_site.csv` gives how many cells have one, two or three dates). The masks and the per-cell counts are not in the repository.
