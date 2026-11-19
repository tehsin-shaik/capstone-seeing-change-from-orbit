# Why do the predictions change with the image date?

The change map work found that the random forest's mangrove area for the same site moves a lot between two dates of the same year (see [../RESULTS.md](../RESULTS.md)). Here we tested three possible reasons. In short: tide explains about 1% of it, differences between the images themselves go with about half of it, and nothing we tested makes the model steady.

The numbers are in the CSVs of the three folders and the scripts are in `src/change_detection/date_effect`. All three tests use the random forest baseline (B2), each site predicted by the forest that never saw it, on the non-hazy scenes of the six training sites (2018, 2020, 2025). Train_Val_Dataset only.

Everywhere below, the swing is a same-year swing. For one scene it is its predicted mangrove area as a % above or below the average of the dates of the same site and the same year. A scene is never compared with another year, so real change between 2018, 2020 and 2025 is not mixed in. Water level and the image measures are compared the same way. This leaves 37 scenes in 16 site-years (18 different dates, because three sites share the same satellite passes). Al Nouf only counts for 2020, since its two 2018 dates share no chip and it has one date in 2025.

## 1. Tide

For every scene we got the water level two ways: the share of the site that is water in the image (green brighter than near-infrared), and the tide height from an open tide model (EOT20) at the time the image was taken. The two agree with each other (r = 0.72), so both do measure water level.

Neither goes with the same-year swings. The tide model explains about 1% of the same-year swing and the water share also about 1%. If we shuffle the dates inside each site and year, a link at least that strong comes up 72% and 60% of the time, so this is no different from chance. At pixel level, about a quarter of the pixels that flip between two dates of the same year are in or next to the strip that is sometimes water and sometimes land (15% expected by chance). The other three quarters are nowhere near moving water.

The time of year goes with the same-year swing more than tide does: November against the rest gives 21%, and the height of the sun gives 39%. But most of that is one pass in May 2020 (without it the sun height gives 10%), and shuffled dates do as well 6% to 20% of the time. The satellite always passes at about 11 am, so in this data November images are also high tide images, and tide and season can't be fully separated.

Figures: [all sites](tide/test1_all_sites.png), [site by site](tide/test1_per_site.png).

### Tide gauge check

The tide heights come from a model, so we looked for real tide gauge records to check them against. There is no public tide gauge for the UAE in the open archives (UHSLC, GESLA, IOC). The nearest ones with data on our dates are in Oman (2018 and 2020 dates) and Kuwait (2025 dates), 180 to 830 km from our sites.

At those gauges the model is typically within about 10 cm of the measured level at the overpass times (4 cm at Muscat and Majis, 11 cm at Khawr Wudam and in Kuwait), while the level itself changes by more than a metre from date to date. But these gauges are in different waters. The Oman ones are on the Gulf of Oman, where the tide is about 4 hours ahead of the tide at our sites, and Kuwait is at the other end of the Gulf. So this checks the model and our timing, not the water level at our sites.

The same-year test still shows tide explains about 1%. Putting the gauge readings in place of the model, or adding the gauge's difference from the model to our heights, gives nothing different from chance. UAE gauge data does exist: a network of 34 water level stations was [installed for Bayanat in 2017](https://www.hydro-international.com/content/article/tide-gauge-and-water-temperature-monitoring-network-for-the-uae). It is not public and would have to be requested from them.

Tables: [stations found](tide/gauge/stations_found.csv), [gauge against model](tide/gauge/gauge_vs_model.csv), [same-year test](tide/gauge/gauge_same_year_test.csv). Figure: [gauges at the overpass times](tide/gauge/gauge_vs_model.png).

## 2. Image differences

Next we checked whether the images themselves are comparable between dates. We picked pixels that shouldn't change (deep water, and bright bare ground that is stable on every date) using band rules only, no labels. They do change. Within the same site and year, bright ground is up to 12% darker or brighter than the year's average, and on 17 November 2020 a quarter to a third of the pixels at three sites have a band at or below zero.

These differences go with the same-year swings. Deep water brightness and bright ground brightness together account for about half of the same-year swing (46%). When a scene reads darker than usual, the forest predicts more mangrove. This is less certain than the size suggests, though. Inside one year a site has only two or three dates to compare, and shuffled dates do as well 14% of the time. Deep water in the red band on its own accounts for 30% (shuffled dates do that well 5% of the time). Adding tide to the two measures changes nothing (47%). The processing version is the same for all scenes of a site and year, so it can't explain same-year swings.

Then we tried to correct it. Each scene got one straight line per band so that its stable pixels match a reference date of the same site, and the same forests were run again. Before running we decided that the number to watch is the average same-year swing per site: for each pair of dates of the same site and year, the net change between the two as a % of the first date's predicted area, averaged over the site's pairs. It had to fall by at least a quarter to count as helping. It fell from 19.8% to 15.2%, which is 23%, and only to 17.2% without Al Nouf, which has a single pair. IoU stayed the same (0.717 and 0.718 on the non-hazy 2020 chips). So the correction does not fix it.

Figures: [measures against same-year swings](image_differences/track_swings.png), [before and after the correction](image_differences/before_after.png).

## 3. Robustness

Last we compared four versions of the forest, all trained on the same 2020 chips and folds. The rule was set before running: a version helps if the average same-year swing per site (the pair number from section 2) falls by at least a quarter and IoU is no more than 0.01 lower.

| Version | Average same-year swing | Held-out IoU |
|---|---|---|
| A. Current forest | 19.8% | 0.704 |
| B. Trained with extra copies of each chip, changed in brightness like the real between-date differences | 18.4% | 0.706 |
| C. Trained on NDVI and MNDWI only | 19.0% | 0.657 |
| D. Forest A, area from the predicted probabilities instead of the threshold | 14.0% | same as A |

B and C don't help. D passes the rule, but only on paper. The sum of probabilities is about 1.7 times the thresholded area, so the same movement gets divided by a bigger number. In hectares D swings more than A (119 ha against 96 ha). We also made every image 10% darker and 10% brighter to see how far A's area moves. It moves by +6.9% and -4.1% on average. So the forest does react to brightness, but not by enough to explain same-year swings of 20%.

Figures: [sensitivity of A](robustness/sensitivity_A.png), [the four versions](robustness/versions.png).

## Where this leaves us

The date effect is not tide. It goes with how different the images are from date to date, but making the images more alike, or training the forest on brightness changes, did not remove it. So a change map made from single dates with this model still has a same-year noise level of about 20% of the mangrove area on average (9% at Ghantoot, 43% at Jubail), and only changes clearly above that can be trusted.

An earlier version of sections 1 and 2 measured each scene against the site's average over all three years, which mixes in real change. The sizes were the same (tide 1%, the two image measures 47%), but the image link looked much more certain than it is (shuffled dates did as well 0.4% of the time, against 14% with same-year swings). Those numbers are kept in the CSVs in rows and columns marked "all years".

## What none of this shows

- It doesn't show what the cause is. Image differences go with the same-year swings, but that is a link over 18 dates that could still be chance. Both image measures also rise and fall with the height of the sun (r about 0.6), which explains nearly as much on its own, so we can't tell the two apart.
- The tide heights at our sites were not checked against a tide gauge, because no UAE gauge is public. The model was only checked at gauges in Oman and Kuwait. UNVERIFIED.
- Only the random forest was tested. The U-Net and DINOv3 may react differently.
- Only one simple correction and one augmentation setting were tried. A better one might work.
- There is no ground truth, so we can't say which date is right.
- Training sites only. Al Nouf has very few chips and its numbers are the weakest.
