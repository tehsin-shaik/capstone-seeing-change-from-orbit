# Registration check at Eastern Mangroves

This repeats the Jubail check from `docs/REGISTRATION_METHOD.md` at a second training site. `src/registration/real_raster_check.py` was run as it is, with no changes.

## What was run

- Sentinel-2 L2A images of `eastern_mangroves` (677 x 900 px) downloaded with `src/mangrove_segmentation/collect.py`. The reference is 23 November 2018 (`S2B_T40RBN_20181123T070514_L2A`) and the moving image is 3 December 2025 (`S2B_T40RBN_20251203T065606_L2A`).
- December 2018 was too cloudy. The archive collect.py searches has only one December 2018 scene for this site (23 December), and about 42% of the site is cloud or cloud shadow in it. The nearest clear scene, 23 November 2018, was used instead.
- The script only reads Earth Engine style files, so `src/registration/make_6band_copy.py` makes 6-band copies of the downloads first (uint16, bands named B2 to B12, SCL dropped, same grid).

## Results next to Jubail

| | Jubail, Khaled's download | Jubail, Earth Engine export | Eastern Mangroves |
| --- | --- | --- | --- |
| Matches | 315 | 314 | 383 |
| RANSAC inliers | 252 | 231 | 343 |
| RMSE with no correction | 5.25 m (0.53 px) | 4.83 m (0.48 px) | 5.72 m (0.57 px) |
| Best model on held-out landmarks | translation, 5.25 m | similarity, 4.71 m | translation, 4.94 m |
| Gain over no correction | 0.003 m | 0.12 m | 0.78 m |
| What the script chose | none | none | none |
| Whole-image shift | 0.56 m | 1.01 m | 2.91 m |
| Known-warp round trip error | not run | 0.11 to 0.16 m | 0.06 to 0.12 m |
| Water share of the image, 2018 / 2025 | 46.6% / 45.7% | not reported | 21.0% / 21.9% |

The Jubail numbers come from the method doc and the CSVs in `results/registration/`. The whole-image shift is the translation row (`tx_m`, `ty_m`) of each baseline CSV. The Eastern water share is NDWI > 0 over the whole image, worked out separately and not by the script. UNVERIFIED: whether Khaled's 46.6% / 45.7% used the same rule, because that code is not in the repo.

## The 2.9 m shift

The 2025 image sits about 2.9 m east of the 2018 image (a bootstrap over the 343 inliers gives 2.5 to 3.3 m). It is about the same near the water and in the built-up area, and in each half of the image. The cloudy 23 December 2018 scene against the same 2025 image gave 2.7 m. UNVERIFIED: which of the two images is off, and whether a method that does not use SIFT would find the same shift.

The script still chose "none" because its rule needs the held-out RMSE to drop by at least 1 m. Single landmarks scatter by about 4.9 m, and a shift adds to that as a square root: sqrt(4.94² + 2.9²) = 5.73 m. So removing the shift only moves the RMSE from 5.72 to 4.94 m, a gain of 0.78 m. With this much scatter a shift has to reach about 3.3 to 3.5 m (a third of a pixel) before the rule applies a correction.

## Rerun

`real_raster_check.json` is the summary and the other files are the script's own outputs. The figure title says "Jubail" and the crop is a narrow strip because both are fixed in the script. Rasters and downloads are not committed.

```
python src/registration/make_6band_copy.py <download.tif> <out_6band.tif>
python src/registration/real_raster_check.py --reference <2018_6band.tif> --moving <2025_6band.tif> --out <results folder> --rasters <folder outside the repo>
```
