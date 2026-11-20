# Track A: Landmark detection for image registration

**Owner:** Khaled Alzaabi. **Pairs with:** Tehsin Shaik (alignment algorithm and registration error).

## Goal

To measure mangrove change, the 2018 and 2025 images must line up so that the same ground point sits on the same pixel in both. This module finds the **fixed points (landmarks)** used to line them up, and decides which of them can be trusted.

## Data used

Sentinel-2 Level-2A over Al Jubail Island, Abu Dhabi, taken on **2018-12-10** and **2025-12-10**. Both were downloaded from the Copernicus Browser with these settings: Analytical, TIFF 32-bit float, 10 m, UTM 40N (EPSG:32640), bands B02, B03, B04 and B08. Each image is 1836 × 1096 px, about 18 × 11 km. The raw imagery is not stored in this repo.

## Method

1. **Grayscale.** Each date is converted to grayscale from red, green and blue. Each date is contrast-stretched separately so that brightness differences between the dates matter less.
2. **Search area (filter).** This decides where corners are allowed to be found. Three rules are compared:
   - *No filter*: the whole image.
   - *Ignore open water*: skip the inside of water bodies (waves, sun glint, submerged sandbanks) but keep shorelines and creeks.
   - *Built-up/bare ground only*: the textbook rule. It skips water, vegetation and a 20 m buffer around both.
3. **Corner detectors compared.** Harris, Shi-Tomasi, FAST, ORB, SIFT and AKAZE. Each gets the same budget of 3,000 points. Harris, Shi-Tomasi and FAST don't produce a description of each point, so SIFT descriptions are used for them. That way only the choice of points differs between detectors.
4. **Matching.** A match is kept only if the point's best match is clearly better than its second-best (ratio test, 0.8) and the two points pick each other in both directions.
5. **Rejecting bad corners with RANSAC.** RANSAC finds the single affine transformation that most matches agree with (within 1.5 px, which is 15 m) and rejects the rest. Corners on buildings that were built or demolished between 2018 and 2025 have no true partner, so they are rejected here.
6. **Evaluation with a known answer.** The downloaded pair turns out to be already aligned to within about 0.1–0.3 px (see results). So the 2025 image is deliberately misaligned with a random but known affine transformation: up to ±15 px shift, ±2° rotation, ±2 % scale and ±1 % shear. This is done 20 times, and the error is measured as the average distance between where the recovered and the true transformation place each pixel.

## Results

**1. The real 2018/2025 pair is already aligned.** Every detector estimates less than 0.4 px (4 m) of movement between the two dates. Sentinel-2 L2A products are already corrected onto a common geometric reference. On the Sentinel-2 archive, the registration step is therefore a safeguard and a check rather than a large correction. It will matter more for other sources: Sentinel-1 radar, Landsat, or high-resolution imagery.

**2. Most reliable combination: SIFT + "ignore open water".** It has the smallest worst-case error. Over the 20 known-warp trials:

| Detector | Filter | Inlier ratio | Median error | Worst error |
|---|---|---|---|---|
| SIFT | ignore open water | 0.74 | 0.12 px (1.2 m) | **0.18 px** |
| SIFT | none | 0.70 | 0.11 px | 0.30 px |
| Harris | ignore open water | 0.65 | 0.14 px | 0.26 px |
| FAST | none | 0.69 | 0.17 px | 0.47 px |
| ORB | built-up only | 0.22 | 0.94 px | 2.51 px (fails 45 % of trials) |

The full table is in `results/registration/summary.csv`, and the per-trial rows are in `results_synthetic.csv`. The median errors are close to the ~0.1 px natural offset between the two dates, which is the accuracy limit of this test. SIFT and ORB results can differ slightly between computers, in the second decimal place, because OpenCV uses processor-specific optimisations for them. The other detectors reproduce exactly.

**3. The textbook filter makes things worse here.** Keeping only built-up and bare ground was the worst filter for every detector. It lowered the share of good matches for all six detectors and produced the only failures. The reason is that between 2018 and 2025 the "stable" built-up land is exactly what changed: Jubail Island, Saadiyat and Ramhan Island were all developed. Meanwhile the mangrove creek network barely moved. On the unfiltered matches, shoreline corners survived RANSAC 78–85 % of the time, versus 63–70 % for built-up and bare ground.

**4. Caveat: tide.** The creek and shoreline landmarks only work because both images happen to have almost the same tide: water covers 46.6 % of the 2018 image and 45.7 % of the 2025 one. At a different tide the waterline moves, and those corners would stop being fixed points. **The next test is a pair taken at clearly different tides.**

## Outputs (in `results/registration/`)

- `landmarks_matched.csv`: the 315 matched landmark pairs from SIFT + ignore-open-water on the real pair, with pixel coordinates in both dates, UTM easting/northing, and a flag for whether RANSAC kept the point. **This is the input for the alignment step.** A full affine transformation needs at least **3** landmarks (6 unknowns). With 2 landmarks only shift, rotation and uniform scale can be solved.
- `fig_search_areas.png`: where each filter lets the detector look.
- `fig_landmarks.png`: kept and rejected landmarks on both dates.
- `fig_detector_comparison.png`: error by detector and filter.
- `fig_checkerboard.png`: a known misalignment before and after correction.

## How to run

```
python -m pip install -r requirements.txt
python landmarks.py --data "D:/MyStuff/University/SDP/data/jubail" --out ../../results/registration
python figures.py   --data "D:/MyStuff/University/SDP/data/jubail" --out ../../results/registration
```

`--data` is a folder containing the two Copernicus Browser zips. The earlier date is treated as the reference. The full run takes about 5 minutes.

## Other notes

- In the 2018 image, the near-infrared band (B08) is exactly 0 over about 16 % of the scene, all of it water. The 2025 image has no such zeros. Older Sentinel-2 processing clipped very dark water to 0, and the current processing doesn't. This does not affect registration, but it shifts NDWI and NDVI over water between the two dates, so Track B should know about it.

## Next steps

1. Repeat on a pair at a clearly different tide, to test whether shoreline landmarks still hold.
2. Test more pairs and areas (other mangrove sites, other years).
3. Spread the landmarks evenly across the image (a few per grid cell), so the affine fit isn't dominated by one dense area.
