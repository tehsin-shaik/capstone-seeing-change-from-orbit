# Track A: Aligning the 2018 and 2025 images

This is the second half of Track A, done by Tehsin Shaik. Khaled finds the matching
landmarks (`src/registration/landmarks.py`, see its README). Tehsin's part uses them to
line up the 2025 image with the 2018 image and measure how far off it still is.

- Code: [`src/registration/affine_alignment.py`](../src/registration/affine_alignment.py)
  and [`src/registration/real_raster_check.py`](../src/registration/real_raster_check.py)
- Tests: [`src/registration/tests/test_affine_alignment.py`](../src/registration/tests/test_affine_alignment.py)
- Results: `results/registration/`

## Why we need this

Change detection compares the same spot on the ground in two years. If 2025
is shifted by even one pixel (10 m), every mangrove edge and shoreline shows
up as fake "change". So before comparing years we have to know how well the
images line up, and fix it if needed.

## 2 landmarks vs 3 landmarks

Each landmark pair gives two equations (x and y).

| Landmarks | Equations | What you can fit | Corrects |
| --- | --- | --- | --- |
| 2 | 4 | similarity (4 unknowns) | shift, rotation, one scale |
| 3, not on a line | 6 | full affine (6 unknowns) | also shear and different x/y scale |

With 2 landmarks a full affine isn't possible (4 equations, 6 unknowns), so
the 2-landmark case uses a similarity. If the 3 points are on a line, or
nearly, the affine fit blows up, so triplets with a side under 1 km or an
angle under 15° are rejected. These limits were chosen by judgement and can be
changed in the code.

## How the error is measured

All positions are EPSG:32640 (UTM 40N) map coordinates in metres, not pixels,
because pixel positions depend on how each image was cropped.

The error for a landmark is the distance between the moved 2025 point and its
2018 partner. It's always measured on landmarks **not** used for the fit.
Error on the fit points is close to zero by definition, so it tells you nothing.

- 2 vs 3 trials: pick 3 random landmarks, fit, and measure on the other ~249.
  The 2-landmark fit uses the first two of the same three, so both methods
  are compared on exactly the same draw. 2,000 trials were run.
- All-landmark fit: 10-fold cross-validation.

The real pair turned out to be aligned already, so a **known warp** was also
tested: move the 2025 landmarks by one of Khaled's random affines (up to
150 m shift, ±2° rotation, ±2% scale, ±1% shear) and check whether it gets
undone.

## Results

**The real 10 Dec 2018 / 10 Dec 2025 pair is already aligned.** 252 RANSAC
inliers out of 315 matches. With no correction the error is RMSE 5.25 m,
about half a pixel. Fitting any model on all landmarks doesn't do better on
held-out landmarks (5.25–5.27 m), so no correction is applied
(`selected_transform.json`).

**2 vs 3 landmarks** (median held-out RMSE, accepted triplets):

| Method | Real pair | Known warp |
| --- | --- | --- |
| No correction | 5.3 m | 187.6 m |
| 2 landmarks, similarity | 7.9 m | 28.0 m |
| 3 landmarks, similarity | 6.7 m | 20.7 m |
| 3 landmarks, full affine | 8.8 m | **8.9 m** |

- When there's real misalignment with shear, 3 landmarks with a full affine
  are much better than 2. On the same trials the affine is 16.1 m better
  (95% CI 14.7 to 17.8 m) and wins 89% of the time.
- On the already-aligned pair, fitting with only 2 or 3 points makes things
  slightly worse, because the fit follows the noise in those few points.
- Rejected (nearly straight) triplets give much bigger errors, up to
  kilometres, so the shape of the triangle matters as much as the count.

![2 vs 3 landmarks](../results/registration/fig_2_vs_3_landmarks.png)

![Example trial](../results/registration/fig_alignment_example.png)

## Check on the real 6-band images

`real_raster_check.py` uses Tehsin's Earth Engine exports of Jubail (B2 B3 B4
B8 B11 B12, 10 m, same dates), not Khaled's download.

1. Running Khaled's SIFT + "ignore open water" matching on this export gives
   314 matches and 231 inliers, RMSE 4.83 m, and again no correction is
   needed. Same conclusion as Khaled, from a separate export.
2. The real 2025 GeoTIFF was warped 5 times by known random affines (151–236 m
   off), and each warp was found again from landmarks and undone. The image came
   back to within 0.11–0.16 m, and band B4 correlation went from 0.61–0.74
   to 0.998. This shows the warp code works on real rasters. Both images
   are the same date though, so it doesn't test matching across years.
3. Warping with "no correction" gives back the exact same file, with CRS,
   band names and data type unchanged.

![Real-raster checkerboards](../results/registration/real_raster/fig_real_raster_checkerboard.png)

In the left panel roads and creeks continue across the 2018/2025 tiles. The
tiles that look different are real changes (new buildings) and brightness,
not misalignment.

## Limits

- The SIFT checks cover Jubail and Eastern Mangroves (see `results/registration/`).
  A second check without SIFT, at all 6 training sites, is in
  [`results/registration/phase_correlation/`](../results/registration/phase_correlation/RESULTS.md).
  Sentinel-1 still needs to be checked.
- Errors compare 2025 with 2018, not with true ground positions.
- An affine is one transform for the whole image, so local distortion isn't fixed.
- The known-warp numbers depend on how big Khaled's random warps are.

In practice, alignment should use all RANSAC inliers, not just 2 or 3. The
experiment shows why the minimum number of points isn't enough.

## Running it

```
pip install -r src/registration/requirements.txt pytest
python src/registration/affine_alignment.py experiment     # 2 vs 3 results and figures
python -m pytest -q src/registration/tests
python src/registration/real_raster_check.py               # needs the two GeoTIFFs in data/raw/
python src/registration/affine_alignment.py warp --moving 2025.tif --reference 2018.tif --out 2025_aligned.tif
```

`alignment_trials.csv` (one row per trial, about 3 MB) is recreated by the
experiment and not committed. The aligned rasters go to
`data/processed/registration/` and aren't committed either.
