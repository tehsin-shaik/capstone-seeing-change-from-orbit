# Hand labels: four people, GMW and the models

The other results score the models against Global Mangrove Watch (GMW), which is itself a machine-made map. Here four people labelled the same 6 chips on their own, without looking at GMW, the models or each other's work. That answers two questions:

1. How much do people agree with each other? This is roughly the best any map can be expected to score against a person.
2. How close are GMW and the models to each person, and to the labels combined?

Code: [`src/mangrove_segmentation/label_agreement.py`](../../../src/mangrove_segmentation/label_agreement.py). Labels: [data/hand_labels](../../../data/hand_labels/README.md) (one folder per labeller: `Tehsin/`, `Khaled/`, `Zain/` and `Ayesha/`).

## Setup

- **Chips:** 6 Train_Val chips from 2020: Jubail Island (two on 16 Feb, one on 18 May), Ras Ghurab, Eastern Mangroves and Ghantoot (18 May).
- **Labellers:** Tehsin, Khaled, Zain and Ayesha. All four drew `mangrove` and `unsure` layers.
- **Pixels scored:** on each chip, every comparison uses the same pixels. Cloud and no-data pixels, and pixels any of the four put in `unsure`, are left out. That leaves 78–93% of each chip.
- **Combined labels:** `agreed` is what all four marked as mangrove, `majority` is what at least three of the four marked, and `union` is what any of them marked.
- **Random forest:** variant b (hazy scenes dropped), retrained for each chip's held-out fold with the same steps as the [baseline cross-validation](../baselines/RESULTS.md). It never saw the chip's site, and it was trained on GMW v4.0.19.
- **U-Net:** the out-of-fold predictions of the base-8 U-Net from the [efficiency study](../efficiency/RESULTS.md) (0.683 IoU against GMW). Predictions from the full-size U-Net and DINOv3 were not available for this run.
- **Pooled IoU** adds up the pixel counts over all 6 chips before dividing.

Each new labeller's `unsure` areas remove a few more pixels from every comparison, so the Tehsin–Khaled number moves slightly between runs (0.687 with two labellers, 0.710 with three, 0.720 here).

## Person vs person

| Chip | Tehsin vs Khaled | Tehsin vs Zain | Tehsin vs Ayesha | Khaled vs Zain | Khaled vs Ayesha | Zain vs Ayesha |
| --- | --- | --- | --- | --- | --- | --- |
| Jubail Island, 16 Feb, `r00256_c00768` | 0.777 | 0.772 | 0.916 | 0.847 | 0.808 | 0.814 |
| Jubail Island, 16 Feb, `r01024_c00000` | 0.856 | 0.886 | 0.970 | 0.875 | 0.871 | 0.896 |
| Jubail Island, 18 May, `r01024_c01024` | 0.608 | 0.666 | 0.760 | 0.637 | 0.724 | 0.669 |
| Ras Ghurab, 18 May | 0.734 | 0.790 | 0.671 | 0.715 | 0.677 | 0.656 |
| Eastern Mangroves, 18 May | 0.642 | 0.689 | 0.791 | 0.775 | 0.612 | 0.635 |
| Ghantoot, 18 May | 0.736 | 0.797 | 0.905 | 0.796 | 0.794 | 0.829 |
| **All 6 (pooled)** | **0.720** | **0.769** | **0.828** | **0.768** | **0.743** | **0.747** |

Mangrove area on the scored pixels, all 6 chips: Tehsin 700 ha, Ayesha 761 ha, Zain 806 ha, Khaled 889 ha. All four agree on 604 ha, at least three mark 719 ha, and at least one marks 991 ha.

All four mark the same dense canopy. The differences are at the edges: Tehsin draws tightest and Khaled widest, with Ayesha and Zain in between. Ayesha's edges are closest to Tehsin's (0.828, the highest pair). Jubail Island on 18 May is the hardest chip for most pairs (0.61–0.76), and Ras Ghurab is the hardest for Ayesha (0.66–0.68).

## GMW and the models against each label

Pooled IoU over the 6 chips (precision / recall in brackets):

| Compared with | Tehsin | Khaled | Zain | Ayesha | Agreed | Majority | Union |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GMW v4.0.19 (10 m) | 0.726 (0.75 / 0.96) | 0.717 (0.83 / 0.84) | 0.695 (0.78 / 0.86) | 0.749 (0.79 / 0.93) | 0.635 | 0.736 (0.77 / 0.95) | 0.746 |
| GMW v4.1.12 (~30 m) | 0.692 (0.72 / 0.94) | 0.715 (0.82 / 0.85) | 0.678 (0.76 / 0.86) | 0.725 (0.77 / 0.93) | 0.611 | 0.704 (0.74 / 0.94) | 0.746 |
| Random forest | **0.862** (0.90 / 0.96) | 0.736 (0.93 / 0.78) | **0.779** (0.91 / 0.84) | **0.820** (0.91 / 0.89) | **0.771** | **0.864** (0.91 / 0.94) | 0.737 |
| U-Net base 8 | 0.810 (0.84 / 0.96) | **0.760** (0.91 / 0.82) | 0.768 (0.87 / 0.86) | 0.801 (0.87 / 0.91) | 0.721 | 0.820 (0.86 / 0.95) | **0.765** |

Per chip IoU against the majority label:

| Chip | GMW v4.0.19 (10 m) | GMW v4.1.12 (~30 m) | Random forest | U-Net base 8 |
| --- | --- | --- | --- | --- |
| Jubail Island, 16 Feb, `r00256_c00768` | 0.795 | 0.809 | 0.928 | 0.882 |
| Jubail Island, 16 Feb, `r01024_c00000` | 0.867 | 0.871 | 0.908 | 0.927 |
| Jubail Island, 18 May, `r01024_c01024` | 0.552 | 0.550 | 0.840 | 0.709 |
| Ras Ghurab, 18 May | 0.639 | 0.589 | 0.819 | 0.781 |
| Eastern Mangroves, 18 May | 0.814 | 0.740 | 0.846 | 0.780 |
| Ghantoot, 18 May | 0.843 | 0.784 | 0.874 | 0.856 |

Per chip scores against each labeller are in [`per_chip.csv`](per_chip.csv). Full numbers: [`agreement.csv`](agreement.csv), [`per_chip.csv`](per_chip.csv) and [`summary.csv`](summary.csv).

## What it shows

- **GMW is a little below the level at which people agree with each other.** The six pairs of labellers agree at 0.72–0.83 IoU. GMW v4.0.19 scores 0.70–0.75 against each person and 0.736 against the majority. So GMW is close to, but slightly worse than, another human labeller.
- **The models match the people better than GMW does, even though they were trained on GMW.** Against the majority label, the random forest scores 0.864 and the U-Net 0.820, against 0.736 for GMW. The largest gaps are on Jubail Island (18 May) and Ras Ghurab, where GMW marks a lot of area that the people did not (precision 0.55 and 0.65 against the majority), while the random forest's precision is 0.94 and 0.85. The models' ~0.70 against GMW therefore understates how well they match human interpretation.
- **Which map "wins" depends on who drew the label.** The random forest draws tight edges, so it is closest to Tehsin (0.862), Ayesha (0.820) and Zain (0.779). Against Khaled's wider labels it misses about a fifth of his mangrove (recall 0.78), and the U-Net is closest instead (0.760).
- **Everything finds the dense canopy.** Against the area all four agree on, recall is 0.96–0.97 for all four maps. The differences are about how far out from the canopy each one draws.
- **The majority label is a reasonable single ground truth, but it leans tight.** With four labellers, "at least three of four" (719 ha) sits between the `agreed` area (604 ha) and the `union` (991 ha), close in size to Tehsin's and Ayesha's labels. Because it needs three votes, it favours tighter edges, which suits the random forest. A "half or more" rule (at least two of four) would be wider.

## Limits

- 6 chips and 4 labellers, all from 2020 and from the Train_Val sites. More chips would narrow the numbers.
- The models are out-of-fold, but the chips are not from the locked test sites, so this is not a final test score.
- The full-size U-Net, the SVM and DINOv3 are not in this run. The base-8 U-Net scores 0.683 against GMW, a little under the full U-Net's 0.709.
- The majority label is used as the ground truth. It is saved per chip in [data/hand_labels/majority](../../../data/hand_labels/README.md#majority-ground-truth). The agreed and union labels are reported above for comparison.

## Re-run

```
cd src/mangrove_segmentation
python label_agreement.py --train-val <Train_Val_Dataset> \
  --labels ../../data/hand_labels/Tehsin/*_Tehsin.gpkg ../../data/hand_labels/Khaled/*.gpkg \
           ../../data/hand_labels/Zain/*.gpkg ../../data/hand_labels/Ayesha/*.gpkg \
  --pred "U-Net base 8=<efficiency gpu_b8 run>/preds/UNET_b_no_haze" \
  --out <work folder> --save-majority ../../data/hand_labels/majority
```

`*_Tehsin.gpkg` skips `KAB_2020_07.gpkg`, which is not a Train_Val chip. Training the 4 random forests takes about 15 minutes on a laptop CPU. They are cached in `<work folder>/_work`, so re-runs take seconds.
