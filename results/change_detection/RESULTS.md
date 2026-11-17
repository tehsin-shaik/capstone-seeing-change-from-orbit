# First model-based change map and same-year noise check

For each training site, the random forest baseline (B2) was trained on the 2020 chips of the other folds, so it had never seen the site. It then predicted the site's 2018 and 2025 scenes, and each year's mask is a per-pixel vote over that year's scenes. To see how much of the "change" is model noise, two dates of the same year were compared in the same way, where nothing real can have changed much. At five of the six sites the 2018 to 2025 change is no bigger than the largest same-year swing. Only Ghantoot is above its noise, by a small margin.

## Result per site

Net change is gain minus loss, as a percent of the first date's mangrove area.

| Site | 2018 to 2025 net change | Worst same-year swing | Above noise? |
|---|---|---|---|
| Ghantoot | +25.1% (+212 ha) | 17.8% | Yes, by a small margin |
| Al Nouf | -10.5% (1 chip) | 19.6% (one pair) | No |
| Al Aryam Islands | -25.1% | 28.0% | No |
| Ras Ghurab | +3.0% | 36.9% | No |
| Eastern Mangroves | +2.0% | 37.9% | No |
| Jubail Island | -11.1% (-135 ha) | 92.2% | No |

At Ghantoot the change comes out the same four ways: +25.1% with the vote, +25.3% on pixels with two dates in both years, +21.3% March to March and +27.8% November to November. At Jubail the same four versions run from -20.8% to +33.2%.

## Figures

- [Jubail Island](jubail_island/change_map_figure.png)
- [Ghantoot](ghantoot/change_map_figure.png)

Each shows the 2018 image, the 2025 image and the change map.

## What this doesn't show

- It doesn't show that Ghantoot really gained mangrove. No ground truth or hand label was used. An error shared by both scenes of a year would pass the same-year check.
- It doesn't show that the new green pixels at Ghantoot are mangrove. The forest can call other vegetation mangrove.
- It doesn't explain the swings. November 2020 scenes read higher than February or May at most sites, and the cause has not been checked.
- The noise floor is rough. Each site has at most five same-year pairs, and some compare only 1 or 2 chips. Only one model was tried.

## Everything else

The other numbers are in the CSVs in this folder: `jubail_island/` and `ghantoot/` (class areas in hectares, edge check, label comparison, all date pairs), `all_sites/` (every same-year pair, the 2018 to 2025 change per site, the held-out IoU of each forest) and `inventory/` (scenes, chip grid and labels per site). The code and how to run it are in `src/change_detection`.
