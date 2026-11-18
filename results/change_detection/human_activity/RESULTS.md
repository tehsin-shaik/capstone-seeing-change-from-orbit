# New bare or built ground near the mangroves

The question was whether there is human activity near the mangroves, like construction, that could explain change. This check maps land that turned into bare or built ground between 2018 and 2025 at Jubail Island and Ghantoot. It uses a fixed rule on band values and no trained model. Sand and concrete look alike to the rule, so the result is called "new bare or built ground", not construction. The code is `src/change_detection/new_bare_ground.py`. Train_Val_Dataset only.

## The rule

- A pixel is bare or built ground if NDVI is below 0.15, MNDWI is below 0 and SWIR1 (B11) reflectance is above 0.20. That is: not green, not water, and bright. It is new if it was not bare in November 2018 and is bare in November 2025. The numbers are the same for both sites and every date.
- Noise check: the same rule on two dates of the same year (2018, 2020 and 2025), forwards and backwards, so six runs per site. Nothing much can turn bare within a year, so these should be near zero.
- The mangrove outline is the GMW v4.1.12 label for 2018.

## New bare or built ground near the 2018 mangrove

| Site | Zone | Hectares | Share of zone | Same-year noise, share of zone |
|---|---|---|---|---|
| Jubail Island | Inside the 2018 mangrove | 6.4 | 0.58% | 0.01% to 0.35% |
| Jubail Island | Within 100 m | 48.9 | 2.31% | 0.34% to 1.61% |
| Jubail Island | Within 300 m | 209.3 | 3.69% | 0.41% to 2.28% |
| Jubail Island | Within 500 m | 328.3 | 4.02% | 0.52% to 2.42% |
| Ghantoot | Inside the 2018 mangrove | 0.5 | 0.07% | 0.00% to 0.07% |
| Ghantoot | Within 100 m | 15.3 | 1.40% | 0.29% to 1.81% |
| Ghantoot | Within 300 m | 27.0 | 1.21% | 0.28% to 1.85% |
| Ghantoot | Within 500 m | 34.1 | 1.18% | 0.37% to 1.96% |

At Jubail every zone is above its noise range, and the share is about twice as high within 500 m of the mangrove (4.02%) as further away (1.85%). The same-year runs show no such difference. The largest noise run would account for about 200 of the 328 ha, so roughly 130 ha is above the noise. At Ghantoot every zone is inside its noise range, so the rule finds nothing there.

## What it doesn't show

- It doesn't show construction. The rule cannot tell concrete from sand. It also misses building on ground that was already bare: the housing at Jubail and the industrial area at Ghantoot are in the 2025 images but are not flagged.
- It doesn't show that this ground explains mangrove change. Little mangrove turned bare directly: 6.4 of 1,095 ha at Jubail and 0.5 of 755 ha at Ghantoot. The model's loss pixels are about as close to new bare ground as they are to the "new bare ground" of a same-year pair (`model_loss_distance.csv`).
- The noise check compares March with November, not the same month, and the tide is not controlled. Near the mangrove at Ghantoot most of the new bare ground was water in 2018, which is what a lower tide looks like.
- The outline is a 30 m label, so a 100 m ring is about three label pixels wide.

Figures: [Jubail Island](new_bare_ground_jubail_island.png) and [Ghantoot](new_bare_ground_ghantoot.png), each with the 2018 image, the 2025 image and the new bare or built ground over the 2018 mangrove. `pairs.csv` has every pair for the whole area, including March to March and a version where both months agree. `rings.csv` has every pair by zone.
