# Same-year noise check for the U-Net

The random forest change map in [../RESULTS.md](../RESULTS.md) was noisy. At five of six sites the 2018 to 2025 change was no bigger than the "change" between two dates of the same year. This check asks whether the U-Net is steadier across dates. It is not.

The U-Net was retrained with `src/mangrove_segmentation/train_unet.py` as it is (gpu preset), one model per fold, so every site is predicted by a model that never saw it. The rest is the same as for the forest: the same non-hazy scenes of 2018, 2020 and 2025, the same pixels, the same per-pixel vote for each year, and the same comparison code from `src/change_detection/change_map.py`. Train_Val_Dataset only.

## Result per site

Net change is gain minus loss, as a percent of the first date's mangrove area. The noise range is the lowest and highest net "change" between two dates of the same year. A site is above noise if its 2018 to 2025 change is bigger than its largest same-year swing.

| Site | RF noise range | U-Net noise range | RF change | U-Net change | RF above noise? | U-Net above noise? |
|---|---|---|---|---|---|---|
| Eastern Mangroves | -30.9% to +37.9% | -16.2% to +7.9% | +2.0% | +3.5% | No | No |
| Jubail Island | -17.0% to +92.2% | -22.8% to +106.6% | -11.1% | -14.5% | No | No |
| Ras Ghurab | -15.3% to +36.9% | -30.6% to +53.8% | +3.0% | +10.7% | No | No |
| Ghantoot | -3.7% to +17.8% | -7.0% to +22.0% | +25.1% | +36.6% | Yes | Yes |
| Al Aryam Islands | -28.0% to +10.9% | -5.7% to +104.6% | -25.1% | -13.7% | No | No |
| Al Nouf | +19.6% (one pair) | -12.2% (one pair) | -10.5% | +23.2% | No | By the rule, but not reliable |

The U-Net's largest same-year swing is bigger than the forest's at four of the six sites. It is smaller only at Eastern Mangroves and Al Nouf, the two sites with the least data (2 chips per pair, and a single pair). Over all 26 same-year pairs the U-Net has the smaller swing in 11, and the median swing is 16.4% for the U-Net against 15.2% for the forest.

Ghantoot is above its noise with both models. The U-Net gives +36.6% with the vote, +37.0% on pixels with two dates in both years, +29.9% March to March and +36.6% November to November, against a largest same-year swing of 22.0%. The two models agree on the direction of change at five of the six sites. At Al Nouf they disagree (-10.5% and +23.2%) on one chip and one same-year pair, so we do not count it as a result.

## Retraining

The retrained folds score IoU 0.867, 0.552, 0.706, 0.785 and 0.658 on the held-out sites. The committed `results/mangrove_segmentation/unet/cv_results.csv` has 0.860, 0.553, 0.711, 0.770 and 0.626. The mean is 0.713 against 0.704, every fold is inside the committed 95% interval and the largest gap is 0.031 (fold 5). Training does not repeat exactly on a GPU, so everything was run a second time. The second run has a mean IoU of 0.706 and gives the same answer: the U-Net's largest swing is smaller at the same two sites only, and Ghantoot is above noise (+40.4% against 23.1%). The change itself moves between the two runs, for example +10.7% and +15.3% at Ras Ghurab and +23.2% and +47.5% at Al Nouf.

## What this doesn't show

- It doesn't show that the U-Net maps mangroves worse. Its held-out IoU is about the same as the forest's. It shows that its masks move at least as much from date to date.
- It doesn't show that Ghantoot really gained mangrove. Both models learned from the same GMW labels, so they can share an error. No ground truth was used.
- It doesn't explain the swings. On the 17 pairs that compare five or more chips, the two models swing together (correlation 0.95), and the largest swing for both is May to November 2020 at Jubail. That points at the images more than at the model, but the cause was not checked.
- The noise range is rough. Each site has at most five same-year pairs, and the +104.6% at Al Aryam is one chip with 6 ha of mangrove.

`comparison.csv` is the table above with more columns, and `comparison_second_run.csv` is the same for the second run. `pairs_rf_vs_unet.csv` has every same-year pair for both models. `noise_floor_all_pairs.csv`, `change_2018_2025_all_sites.csv`, `site_summary.csv` and `fold_fits.csv` have the same columns as the forest's files in `../all_sites/`. `retrain_vs_committed.csv` compares both training runs with the committed scores. The script that predicts the scenes with the U-Net is not in the repository. Weights and predicted masks are kept outside it.
