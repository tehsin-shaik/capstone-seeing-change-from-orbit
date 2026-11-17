# Model disagreement check against GMW

For the 215 held-out 2020 chips, the masks of three models were compared with the GMW v4.0.19 label: the random forest (B2), the approximate RBF SVM (B3) and DINOv3 with LoRA (80-epoch run), all with hazy scenes dropped from training. The check looks at the pixels where all three models give the same answer and GMW gives the other one. All three reject 9.0% of the GMW mangrove pixels, and all three call 1.22% of the GMW non-mangrove pixels mangrove, which is an area equal to 12.9% of the GMW mangrove. More than half of each model's errors are errors all three make. The code is `src/mangrove_segmentation/model_disagreement.py`. Train_Val_Dataset only. In the tables, a pixel is on an edge if one of its 8 neighbours is the other class in GMW, and pixels are counted once per date.

## Per site

| Site | Chips | GMW mangrove rejected by all three | Of those, on a GMW edge (all GMW mangrove) | GMW non-mangrove called mangrove by all three | Of those, touching GMW mangrove (all non-mangrove) |
|---|---|---|---|---|---|
| Eastern Mangroves | 8 | 7.6% | 48% (14%) | 0.44% | 92% (5.1%) |
| Jubail Island | 80 | 11.9% | 65% (37%) | 1.84% | 41% (2.6%) |
| Ras Ghurab | 50 | 10.8% | 77% (35%) | 1.27% | 60% (3.8%) |
| Ghantoot | 41 | 4.1% | 92% (32%) | 0.91% | 75% (4.1%) |
| Al Aryam Islands | 25 | 17.5% | 79% (43%) | 0.14% | 64% (1.3%) |
| Al Nouf | 11 | 4.9% | 94% (44%) | 0.60% | 66% (3.5%) |
| All sites pooled | 215 | 9.0% | 72% (33%) | 1.22% | 52% (3.1%) |

## IoU with those pixels left out

| Site | Random forest | RBF SVM | DINOv3 + LoRA |
|---|---|---|---|
| Eastern Mangroves | 0.821 → 0.895 | 0.830 → 0.903 | 0.844 → 0.923 |
| Jubail Island | 0.562 → 0.782 | 0.558 → 0.781 | 0.574 → 0.792 |
| Ras Ghurab | 0.697 → 0.855 | 0.689 → 0.839 | 0.700 → 0.862 |
| Ghantoot | 0.777 → 0.859 | 0.774 → 0.851 | 0.754 → 0.832 |
| Al Aryam Islands | 0.638 → 0.788 | 0.648 → 0.791 | 0.598 → 0.750 |
| Al Nouf | 0.684 → 0.759 | 0.671 → 0.742 | 0.654 → 0.740 |
| Mean over the 5 folds | 0.704 → 0.832 | 0.702 → 0.827 | 0.700 → 0.831 |

## What it shows and what it doesn't

- It shows that the errors are shared. Of each model's misses, 54 to 65% are missed by all three, and 50 to 59% of each model's false alarms are made by all three. So the ceiling near 0.70 is common to three very different models.
- It shows that the disagreement sits at GMW's borders. GMW v4.1.12 (about 30 m) also calls 65% of the all-three false alarms mangrove, against 2.6% of all non-mangrove pixels.
- It doesn't show that GMW is wrong at those pixels. The label is the same on every date, but the disagreement changes with the scene: all three reject 15.6% of GMW mangrove in the May scenes and 4.3% in the November scenes, and the all-three false alarms are 4.4% of the GMW mangrove area in May and 20.3% in November (`per_site_date.csv`). That part cannot be label error. The three models also learned from GMW and see the same pixels, so they can share a blind spot.
- The left-out IoU is a ceiling, not a score. Leaving out pixels a model gets wrong always raises its IoU. About 0.83 is what the models would reach if every one of those pixels were a label error.

## Figure and chips to hand label

- [worst_chip_jubail_island.png](worst_chip_jubail_island.png) is the Jubail chip with the most such pixels (row 1024, col 1024, 18 May 2020): the image, the GMW label, and where all three models go against GMW. All three reject 93.9 ha of its GMW mangrove. On 17 November 2020 the same cell has 15.6 ha rejected.
- `top_chips_to_hand_label.csv` lists the 15 held-out chips with the most such pixels, with site, chip id, date and ground box. They cover 11 different chip cells. The last column gives the lowest count for the same cell over its dates, to tell a steady disagreement from a one-scene one.
- `per_site.csv` and `per_fold.csv` hold the two tables above with more columns. The masks and the scores per chip are not in the repository.
