# Registration check at Al Aryam Islands

Phase correlation found Al Aryam's March pair to be the most shifted of all six sites (4.5 m, mostly north), which is past the point where the SIFT rule would start correcting. So both Al Aryam pairs were run through `src/registration/real_raster_check.py`, unchanged, and the change map was redone with the correction applied.

## What was run

- The same Sentinel-2 L2A scenes the Train/Val chips come from (1332 x 1332 px), downloaded with `src/mangrove_segmentation/collect.py` and turned into 6-band copies with `src/registration/make_6band_copy.py`.
- **March:** 8 March 2018 (reference) against 31 March 2025.
- **November:** 23 November 2018 (reference) against 1 November 2025.

## Results

| | March | November |
| --- | --- | --- |
| Matches / RANSAC inliers | 535 / 393 | 396 / 282 |
| RMSE with no correction | 7.96 m | 7.43 m |
| Best held-out RMSE | 5.89 m (translation) | 6.09 m (affine) |
| Gain over no correction | 2.07 m | 1.34 m |
| What the script chose | translation | affine |
| Shift found (translation row) | 5.4 m: 1.7 m east, 5.1 m north | 4.3 m: 1.9 m east, 3.8 m north |
| Phase correlation, same pair | 4.5 m: 2.0 m east, 4.1 m north | 2.9 m: 1.3 m east, 2.6 m north |
| Known-warp round trip error | 0.06 to 0.18 m (warps of 135 to 207 m) | 0.07 to 0.16 m (same warps) |

This is the first real pair where the script applies a correction. Both pairs clear the 1 m rule. In November the affine is only 0.02 m better than a plain shift, so it is a shift in practice.

The two methods agree on the direction: the 2025 image sits a few metres north and a little east of 2018. SIFT puts it about 1 m further than phase correlation does. The landmark scatter here is larger than at Jubail or Eastern Mangroves (about 6 m against 5 m), so the landmarks are less precise at this site.

## Change map with the correction

`src/registration/shift_chips.py` makes a copy of the Train/Val dataset in which the Al Aryam 2025 chips are moved by the translation above, for both dates. The translation was used in November as well, since the affine adds nothing. On the chips, phase correlation against the 2018 chip drops from 5.1 to 0.7 m (March, 7 chips) and from 3.8 to 0.9 m (November, 2 chips), as `change_rerun/shift_report.json` shows.

`src/change_detection/noise_floor_all_sites.py` was then run on the original and on the shifted dataset:

| Al Aryam | Original | Shifted 2025 |
| --- | --- | --- |
| Net change, vote over all dates | -27.0% (-39.0 ha) | -28.8% (-41.6 ha) |
| Gain / loss | 13.0 / 52.0 ha | 10.1 / 51.7 ha |
| Changed area, % of 2018 mangrove | 45.1% | 42.8% |
| March to March / November to November | -30.2% / -1.4% | -33.0% / -2.1% |
| Worst same-year swing | 29.0% | 29.0% |
| Above noise? | No | No |

The shift mostly removes fake gain: about 3 ha of it, a quarter of the total, was mangrove edges that only looked new because 2025 was off by half a pixel. Loss hardly moves. The net loss gets a little bigger but still does not beat the worst same-year swing, so the conclusion for Al Aryam stays the same. The other five sites come out exactly the same in both runs.

## Limits

- The numbers in the "Original" column don't match `results/change_detection/RESULTS.md` (-25.1% and 28.0% there). In this rerun the fold 5 forest chose a 0.85 threshold instead of 0.8 (held-out IoU 0.617 against 0.638). The other folds match the committed run to four decimals. Both columns here come from the same rerun, so the comparison between them is fair.
- Shifting each 256 px chip on its own repeats the edge row and column of each chip. Half a pixel is small, but a few edge pixels are not exact.
- Only the random forest was rerun.

## Rerun

```
python src/registration/make_6band_copy.py <download.tif> <out_6band.tif>
python src/registration/real_raster_check.py --reference <2018_6band.tif> --moving <2025_6band.tif> --out <results folder> --rasters <folder outside the repo>
python src/registration/shift_chips.py --train-val <Train_Val_Dataset> --out <shifted copy> --site al_aryam_islands \
    --shift 2025-03-31=2018-03-08=results/registration/al_aryam/march/gee_pair_baseline.csv \
    --shift 2025-11-01=2018-11-23=results/registration/al_aryam/november/gee_pair_baseline.csv
cd src/change_detection && python noise_floor_all_sites.py --train-val <shifted copy> --out <output folder>
```

`march/` and `november/` hold the outputs of `real_raster_check.py`. The checkerboard figures are left out because the script's figure crop is set for Jubail. Rasters and downloads are not committed.
