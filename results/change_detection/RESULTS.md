# First model-based change map and same-year noise check

Mangrove change from 2018 to 2025 at training sites, mapped with the random forest baseline (B2). For each site the forest is trained on the 2020 chips of the other folds, so it has never seen the site. Train_Val_Dataset only. Test_Dataset was not opened. The code is in `src/change_detection`.

## Short answer

- A change map can be made end to end, but the random forest's mangrove area moves a lot from one scene to the next. At most sites the 2018 to 2025 change is no bigger than the "change" between two dates of the same year.
- At Jubail Island the map says -135 ha (-11.1%). The same model shows -17% to +92% between two dates of one year there, so this is inside the noise.
- Ghantoot has the lowest noise. Its map says +212 ha (+25.1%), and four versions of it agree (+21% to +28%). That is above its largest same-year swing (+17.8%), by a modest margin.
- None of this is checked against ground truth.

## How the maps are made

- Model: random forest B2, trained per fold as in the baseline cross-validation, with hazy scenes dropped from training (variant b). Each forest was scored on its held-out 2020 chips and matches the B2 row in `../mangrove_segmentation/baselines/cv_results.csv` to four decimals on all five folds (`all_sites/fold_fits.csv`).
- Scenes: every non-hazy scene of the site. No 2018 or 2025 scene is hazy at any site. All hazy scenes are in 2020 and are skipped.
- One mask per year: per pixel, the year's scenes vote. A 1-1 tie between two scenes is broken by the mean of the two probabilities against the same threshold.
- Classes: stable mangrove, gain, loss, stable non-mangrove. One pixel is 100 m2, so 0.01 ha.
- Net change in percent is gain minus loss, divided by the first date's mangrove area.

Which sites have chips in both years, on the same chip grid (`inventory/`):

| Site | Cells in both years | 2018 chips (Mar / Nov) | 2025 chips (Mar / Nov) |
|---|---|---|---|
| Jubail Island | 25 | 14 / 25 | 25 / 25 |
| Ras Ghurab | 17 | 7 / 17 | 22 / 19 |
| Ghantoot | 11 | 9 / 10 | 11 / 11 |
| Al Aryam Islands | 7 | 8 / 2 | 9 / 7 |
| Eastern Mangroves | 3 | 3 / 3 | 3 / 2 |
| Al Nouf | 1 | 1 / 2 | 4 / none |

Every 2018 and 2025 chip carries the GMW v4.1.12 label (about 30 m) of its own year. The 10 m GMW v4.0.19 label exists for 2020 only.

## Jubail Island, 2018 to 2025

Figure: `jubail_island/change_map_figure.png`. The forest's threshold from the inner fold is 0.90. Area compared: 16,320 ha (25 chips).

| Class | Hectares |
|---|---|
| Stable mangrove | 905.8 |
| Gain | 176.1 |
| Loss | 310.7 |
| Stable non-mangrove | 14,927.2 |

Mangrove area is 1,216.4 ha in 2018 and 1,081.9 ha in 2025. Net change -134.6 ha, which is -11.1%.

Why this is inside the noise:

- Two dates of the same year differ by more than that. March to November 2018 gives +39.2% (14 chips). March to November 2025 gives -15.0%. February to November 2020 gives +53.7%. The full list is in the all-sites table below.
- The sign depends on the dates used.

| Version | Chips | Net ha | Net % |
|---|---|---|---|
| Vote over all dates (the map) | 25 | -134.6 | -11.1% |
| November 2018 vs November 2025 | 25 | -276.2 | -20.8% |
| March 2018 vs March 2025 | 14 | +134.0 | +33.2% |
| Vote, only pixels with two dates in both years | 14 | -3.5 | -0.8% |
| Vote, ties counted as not mangrove | 25 | -178.6 | -15.6% |
| Vote, ties counted as mangrove | 25 | -26.6 | -2.0% |

- March 2018 has only 14 of the 25 chips. Where both years have two dates the net change is about zero. About 131 ha of the 135 ha loss comes from the 11 chips where 2018 has only the November scene.
- Most of the change is at patch borders. 72.6% of the loss is on the edge of the 2018 mangrove (43.6% of all 2018 mangrove pixels are edge pixels). 57.6% of the gain touches 2018 mangrove (4.2% of all non-mangrove pixels do). "Edge" means one of the 8 neighbours is the other class.
- On the pixels the model calls stable mangrove, median NDVI is 0.56 in the November 2018 scene and 0.43 in the November 2025 scene. NDVI is the forest's most important input, so a scene that reads less green loses mangrove pixels at a threshold of 0.90.

Against the labels at Jubail:

| Year | Label | Model ha | Label ha | Model / label | IoU |
|---|---|---|---|---|---|
| 2018 | GMW v4.1.12 (about 30 m) | 1,216.4 | 1,095.4 | 1.11 | 0.64 |
| 2020 | GMW v4.0.19 (10 m) | 1,134.5 | 1,017.7 | 1.11 | 0.63 |
| 2025 | GMW v4.1.12 (about 30 m) | 1,082.2 | 1,822.5 | 0.59 | 0.53 |

GMW v4.1.12 says Jubail gained 725 ha (+66%) from 2018 to 2025. The model says it lost 135 ha. 64% of the label's gain is never mangrove in the model. The pixels the label calls gain did not get greener: their median NDVI is 0.31 in November 2018 and 0.24 in November 2025. Which of the two is right has not been checked.

## Same-year check at every site

Each row compares two non-hazy dates of one year with the forest that had the site held out. Nothing real can change this much within a year, so the net "change" is the noise floor. Held-out IoU is on the site's 2020 chips against GMW v4.0.19. All pairs are in `all_sites/noise_floor_all_pairs.csv`.

| Site | Year | Dates | Chips compared | Net "change" | Held-out IoU |
|---|---|---|---|---|---|
| Ghantoot | 2018 | 20 Mar, 5 Nov | 8 | -0.6% | 0.777 |
| Ghantoot | 2020 | 23 Feb, 18 May | 8 | -3.7% | 0.777 |
| Ghantoot | 2020 | 23 Feb, 9 Nov | 10 | +15.9% | 0.777 |
| Ghantoot | 2020 | 18 May, 9 Nov | 9 | +17.8% | 0.777 |
| Ghantoot | 2025 | 18 Mar, 13 Nov | 9 | +4.6% | 0.777 |
| Eastern Mangroves | 2018 | 18 Mar, 23 Nov | 2 | +12.8% | 0.821 |
| Eastern Mangroves | 2020 | 16 Feb, 18 May | 2 | -30.9% | 0.821 |
| Eastern Mangroves | 2020 | 16 Feb, 17 Nov | 2 | -4.7% | 0.821 |
| Eastern Mangroves | 2020 | 18 May, 17 Nov | 2 | +37.9% | 0.821 |
| Eastern Mangroves | 2025 | 18 Mar, 16 Nov | 2 | -0.2% | 0.821 |
| Ras Ghurab | 2018 | 18 Mar, 23 Nov | 6 | +24.4% | 0.697 |
| Ras Ghurab | 2020 | 16 Feb, 18 May | 5 | -15.3% | 0.697 |
| Ras Ghurab | 2020 | 16 Feb, 17 Nov | 12 | +14.6% | 0.697 |
| Ras Ghurab | 2020 | 18 May, 17 Nov | 8 | +36.9% | 0.697 |
| Ras Ghurab | 2025 | 18 Mar, 16 Nov | 18 | -0.4% | 0.697 |
| Al Aryam Islands | 2018 | 8 Mar, 23 Nov | 2 | -10.6% | 0.638 |
| Al Aryam Islands | 2020 | 17 Jan, 23 Sep | 5 | -5.7% | 0.638 |
| Al Aryam Islands | 2020 | 17 Jan, 12 Nov | 1 | -28.0% | 0.638 |
| Al Aryam Islands | 2020 | 23 Sep, 12 Nov | 1 | -2.0% | 0.638 |
| Al Aryam Islands | 2025 | 31 Mar, 1 Nov | 7 | +10.9% | 0.638 |
| Al Nouf | 2020 | 16 Feb, 17 Nov | 3 | +19.6% | 0.684 |
| Jubail Island | 2018 | 18 Mar, 23 Nov | 14 | +39.2% | 0.562 |
| Jubail Island | 2020 | 16 Feb, 18 May | 15 | -17.0% | 0.562 |
| Jubail Island | 2020 | 16 Feb, 17 Nov | 25 | +53.7% | 0.562 |
| Jubail Island | 2020 | 18 May, 17 Nov | 15 | +92.2% | 0.562 |
| Jubail Island | 2025 | 18 Mar, 16 Nov | 25 | -15.0% | 0.562 |

Notes on the table:

- 2020 has three non-hazy dates at most sites, so it gives three pairs.
- Al Nouf has one usable pair. Its two 2018 scenes share no chip and it has one 2025 scene.
- The Al Aryam rows with 1 chip cover under 10 ha of mangrove and mean little.
- Held-out IoU does not predict the noise. Eastern Mangroves has the best IoU and swings by up to 38%, on only 2 chips.

Largest same-year swing per site, and the 2018 to 2025 change next to it (`all_sites/site_summary.csv`):

| Site | Largest same-year net "change" | 2018 to 2025, vote over all dates | Chips |
|---|---|---|---|
| Ghantoot | 17.8% | +25.1% | 11 |
| Al Nouf | 19.6% (one pair) | -10.5% | 1 |
| Al Aryam Islands | 28.0% | -25.1% | 7 |
| Ras Ghurab | 36.9% | +3.0% | 17 |
| Eastern Mangroves | 37.9% | +2.0% | 3 |
| Jubail Island | 92.2% | -11.1% | 25 |

Only Ghantoot's change is larger than its largest same-year swing.

November 2020 scenes read higher than February or May at most sites. From February to November 2020 the model's mangrove area goes up at Jubail (+53.7%), Al Nouf (+19.6%), Ghantoot (+15.9%) and Ras Ghurab (+14.6%). It does not at Eastern Mangroves (-4.7%), or at Al Aryam (-28.0%, January to November, on 1 chip). From May to November 2020 it goes up at all four sites that have a May scene: Jubail (+92.2%), Eastern Mangroves (+37.9%), Ras Ghurab (+36.9%) and Ghantoot (+17.8%). The cause has not been checked. Tide height, sun angle, haze and the atmospheric correction are all possible and none was tested. One thing that can be read from `hazy_scenes.csv` is that the 17 November 2020 scene has negative dark-water red at Eastern Mangroves, Jubail and Ras Ghurab. Whether that is related was not tested either.

## Ghantoot, 2018 to 2025

Figure: `ghantoot/change_map_figure.png`. The forest's threshold from the inner fold is 0.85. Area compared: 7,204 ha (11 of the site's 15 chip cells have chips in both years).

| Class | Hectares |
|---|---|
| Stable mangrove | 816.9 |
| Gain | 241.9 |
| Loss | 29.5 |
| Stable non-mangrove | 6,115.3 |

Mangrove area is 846.4 ha in 2018 and 1,058.8 ha in 2025.

The same change made four ways:

| Version | Chips | Net ha | Net % |
|---|---|---|---|
| Vote over all four dates (the map) | 11 | +212.4 | +25.1% |
| Vote, only pixels with two dates in both years | 8 | +210.2 | +25.3% |
| March 2018 vs March 2025 | 9 | +184.3 | +21.3% |
| November 2018 vs November 2025 | 9 | +235.3 | +27.8% |

How this compares with the noise at Ghantoot:

- The four versions agree in sign and size. At Jubail they ranged from -21% to +33%.
- The 2018 scenes agree with each other (-0.6%) and so do the 2025 scenes (+4.6%). The change is well above both.
- The largest same-year swing at Ghantoot is +17.8%, from the 9 November 2020 scene. The change is above that too, but not by much: 25% against 18%.
- Change that all four dates agree on is 135.7 ha of gain and 11.0 ha of loss.

Other checks at Ghantoot:

- Greenness. On the pixels the model calls gain, median NDVI is 0.16 in the November 2018 scene and 0.41 in the November 2025 scene. On the pixels it calls stable mangrove it is 0.53 and 0.58. So the gain pixels did get greener, and the two scenes do not differ much on stable mangrove.
- Edges. 59.4% of the gain touches 2018 mangrove. 14.4 ha of gain is more than 10 pixels from any 2018 mangrove.
- Labels. On the same pixels GMW v4.1.12 goes from 771.2 ha to 1,102.5 ha, which is +43.0%. The model and the label agree on the direction. They agree less on where: 49.4% of the model's gain is gain in the label, and 30.0% of it is never mangrove in the label.
- Model area against the label: 1.10 times the label in 2018 (IoU 0.71) and 0.96 times in 2025 (IoU 0.76).
- A rough check of image alignment (best match of the near-infrared band) puts the 2018 and 2025 scenes within about 0.3 pixel of each other.

## What this does not show

- It does not show that Ghantoot really gained mangrove. Being above the same-year noise rules out one kind of error: a scene reading high or low. An error shared by both scenes of a year would pass this check.
- It does not show that the new green pixels are mangrove. The forest can call other vegetation mangrove, and no ground truth or hand label was used here.
- The margin at Ghantoot is small against the worst case. If both 2025 scenes read as high as the November 2020 scene does, much of the +25% would go.
- The noise floor itself is rough. Each site has at most five same-year pairs, and some compare only 1 or 2 chips.
- GMW v4.1.12 is a model product at about 30 m, not ground truth. At Jubail it disagrees with the forest on the direction of change, and which is right has not been checked.
- It does not explain the scene-to-scene swings. The cause of the high November 2020 scenes has not been checked.
- Image alignment was only checked roughly on these chips. The registration work in `results/registration` used different image files.
- Only the random forest, with one seed, variant b and one threshold per fold, was tried. With a 0.5 cut instead of the tuned threshold the net change is -2.8% at Jubail and +16.6% at Ghantoot. No other model was tried.
- Only training sites were mapped. The locked test sites were not used.

## Files

- `jubail_island/` and `ghantoot/`, one folder per mapped site.
  - `change_map_figure.png`. The 2018 image, the 2025 image and the change map.
  - `summary.csv`. Class areas, the edge check, the label checks and the alignment check, one number per row.
  - `noise_floor.csv`. Every pair of dates at the site. The rows marked `same year` are the noise floor.
  - `sensitivity.csv`. The versions of the 2018 to 2025 change.
  - `label_gain_pixels.csv`. NDVI and model probability by group of pixels.
- `all_sites/`
  - `noise_floor_all_pairs.csv`. The same-year table above, with hectares.
  - `change_2018_2025_all_sites.csv`. The 2018 to 2025 change at every site, in up to four versions.
  - `site_summary.csv`. One row per site.
  - `fold_fits.csv`. Threshold, settings and held-out IoU of each forest, next to the committed B2 result.
- `inventory/`. Scenes, chip grid and labels per site.
- The change maps as rasters, the predicted masks and the fitted forests are not in the repository.
