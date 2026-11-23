# Registration check at different tides

The earlier checks used image pairs taken at about the same tide. This runs `src/registration/real_raster_check.py`, with no changes, on pairs where the water level is clearly different, at Jubail Island and Eastern Mangroves (both training sites).

## How the pairs were picked

`src/registration/tide_scene_scan.py` lists the 2018 and 2025 scenes under 10% tile cloud in the archive collect.py uses (27 dates in 2018, 93 in 2025) with the share of each site that is water. It reads the reduced 40 m copies inside each file, so nothing is downloaded. The list is `scene_water_fractions.csv`. Per site, the pair is the 2018 and 2025 scene whose water shares differ the most, with both under 5% cloud (tile and site). UNVERIFIED: water share stands in for the tide and was not checked against a tide table.

- Jubail: 20 March 2018 (32.5% water) and 10 January 2025 (63.6%), 31.1 points apart.
- Eastern: 2 August 2018 and 10 January 2025. NDWI > 0 gives 7.9% and 21.4%, 13.5 points apart, but about half of that is not tide. The 2018 scene has negative reflectance over deep water, and NDWI > 0 then counts that water as land. Counting water as green brighter than near-infrared gives 14.7% and 21.4%, about 7 points. The largest real difference at Eastern is 7.7 points (22 August 2018), so this site is a weak test.

## Results next to the same-tide pairs

| | Jubail same tide, from the method doc | Jubail same tide, new | Jubail different tides | Eastern same tide | Eastern different tides |
| --- | --- | --- | --- | --- | --- |
| Dates, 2018 / 2025 | 10 Dec / 10 Dec | 23 Nov / 26 Nov | 20 Mar / 10 Jan | 23 Nov / 3 Dec | 2 Aug / 10 Jan |
| Water share, 2018 / 2025 | 46.6% / 45.7% | 49.3% / 50.0% | 32.7% / 62.9% | 21.0% / 21.9% | 15.7% / 22.9% |
| Matches | 314 | 350 | 297 | 383 | 220 |
| RANSAC inliers | 231 (74%) | 265 (76%) | 202 (68%) | 343 (90%) | 179 (81%) |
| RMSE with no correction | 4.83 m | 5.60 m | 5.79 m | 5.72 m | 5.07 m |
| Whole-image shift | 1.01 m | 1.20 m | 1.01 m | 2.91 m | 1.23 m |
| Kept landmarks, dry land / shoreline | not worked out | 151 / 114 | 77 / 125 | 211 / 132 | 98 / 81 |
| Shoreline matches that RANSAC kept | not worked out | 84% | 65% | 86% | 74% |

The first column is the Earth Engine export in `docs/REGISTRATION_METHOD.md`, with the water share from `docs/LANDMARK_DETECTION.md`. The new Jubail pair is a collect.py download on the same grid as the different-tide pair. Water shares in the table are at 10 m, so they differ a little from the 40 m scan. Shoreline means within 30 m of water in either image (`tide_landmark_zones.py`). At Jubail the flats that flood count as shoreline, so that zone grows from 27% to 65% of the non-water area and the counts move with it.

Two more different-tide pairs agree. Jubail 20 March 2018 / 27 January 2025: 288 matches, 218 inliers, 5.75 m, shift 2.34 m. Eastern 22 August 2018 / 10 January 2025: 253 matches, 196 inliers, 5.08 m, shift 1.48 m.

## What this shows

Registration holds at different tides in the way the script measures it. The script chose "none" in every run, the RMSE is 5.1 to 5.8 m against 4.8 to 5.7 m at the same tide, and the whole-image shift is 1.0 to 2.3 m, inside the 0.6 to 2.9 m seen at the same tide. The cost is fewer landmarks: kept landmarks drop from 265 to 202 at Jubail and from 343 to 179 at Eastern. There are fewer matches to start with, and RANSAC keeps a smaller share of the shoreline ones. The share of dry-land matches it keeps hardly moves (71% to 73% at Jubail, 92% to 88% at Eastern).

At Jubail the kept dry-land and shoreline landmarks disagree by 2.4 m in easting (dry land -2.50 m, shoreline -0.07 m). In the same-tide pair they disagree by 0.5 m. UNVERIFIED: the cause. It could be the tide or the different season.

## What this doesn't show

- Tide is not the only thing that differs. The different-tide pairs are also in different seasons, and the two picked pairs are seen from opposite sides of the satellite track. Part of the drop in matches could be that.
- The RMSE only counts landmarks RANSAC kept, and those agree within 15 m by rule. A landmark that moved with the tide is thrown out, not counted, so the RMSE cannot get much worse.
- It says the images line up. It does not say a change map is safe: 30.7% of Jubail is water in only one of the two images, and those pixels still look like change. Some of that is new construction, the rest is tide.
- Two sites, and the pairs share images (10 January 2025 at both sites, 20 March 2018 in both Jubail pairs). No check that does not use SIFT.

## Files and rerun

Each pair folder has the script's `real_raster_check.json`, `gee_pair_baseline.csv` and `landmarks_gee_pair.csv`, plus `landmark_zones.csv` and `zone_summary.csv`. The Eastern same-tide folder only has the zone files, the rest is in `results/registration/eastern_mangroves/`, where RESULTS.md also says how the downloads and 6-band copies are made. The figure is the picked Jubail pair. No rasters are committed.

```
python src/registration/tide_scene_scan.py --sites <sites.geojson> --out <scenes.csv> --only jubail_island eastern_mangroves
python src/registration/real_raster_check.py --reference <2018_6band.tif> --moving <2025_6band.tif> --out <pair folder> --rasters <folder outside the repo>
python src/registration/tide_landmark_zones.py --reference <2018 download.tif> --moving <2025 download.tif> --landmarks <pair folder>/landmarks_gee_pair.csv --out <pair folder>
```

Figure: [landmarks at Jubail Island at the two tides](fig_tide_landmarks_jubail.png).
