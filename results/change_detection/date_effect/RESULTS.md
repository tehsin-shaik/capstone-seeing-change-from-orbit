# Why do the predictions change with the image date?

The change map work found that the random forest's mangrove area for the same site moves a lot between two dates of the same year (see [../RESULTS.md](../RESULTS.md)). Here we tested three possible reasons. In short: tide explains about 1% of it, differences between the images themselves go with about half of it, and nothing we tested makes the model steady.

The numbers are in the CSVs of the three folders and the scripts are in `src/change_detection/date_effect`. All three tests use the random forest baseline (B2), each site predicted by the forest that never saw it, on the 40 non-hazy scenes of the six training sites (2018, 2020, 2025). That is only 18 different dates, because three sites share the same satellite passes. Train_Val_Dataset only. The "swing" of a scene is its predicted mangrove area as a % above or below the site's average.

## 1. Tide

For every scene we got the water level two ways: the share of the site that is water in the image (green brighter than near-infrared), and the tide height from an open tide model (EOT20) at the time the image was taken. The two agree with each other (r = 0.73), so both do measure water level.

Neither goes with the swings. The tide model explains about 1% of the swing and the water share about 4%. If we shuffle the dates, a link at least that strong comes up 72% and 43% of the time, so this is no different from chance. At pixel level, about a quarter of the pixels that flip between two dates of the same year are in or next to the strip that is sometimes water and sometimes land (15% expected by chance). The other three quarters are nowhere near moving water.

Month doesn't explain it either (November against the rest gives 6%). One thing to know is that the satellite always passes at about 11 am, so in this data November images are high tide images, and tide and season can't be fully separated.

Figures: [all sites](tide/test1_all_sites.png), [site by site](tide/test1_per_site.png).

## 2. Image differences

Next we checked whether the images themselves are comparable between dates. We picked pixels that shouldn't change (deep water, and bright bare ground that is stable on every date) using band rules only, no labels. They do change. Bright ground is 12% darker to 18% brighter than the site's average depending on the date, and on 17 November 2020 a quarter to a third of the pixels at three sites have a band at or below zero.

These differences go with the swings. Deep water brightness and bright ground brightness together account for about half of the swing (47%, and shuffled dates do that well only 0.4% of the time). When a scene reads darker than usual, the forest predicts more mangrove. The processing version is the same for all scenes of a year, so it can't explain same-year swings.

Then we tried to correct it. Each scene got one straight line per band so that its stable pixels match a reference date of the same site, and the same forests were run again. Before running we decided that the number to watch is the average same-year swing per site, and that it had to fall by at least a quarter to count as helping. It fell from 19.8% to 15.2%, which is 23%, and only to 17.2% without Al Nouf, which has a single pair. IoU stayed the same (0.717 and 0.718 on the non-hazy 2020 chips). So the correction does not fix it.

Figures: [measures against swings](image_differences/track_swings.png), [before and after the correction](image_differences/before_after.png).

## 3. Robustness

Last we compared four versions of the forest, all trained on the same 2020 chips and folds. The rule was set before running: a version helps if the average same-year swing falls by at least a quarter and IoU is no more than 0.01 lower.

| Version | Average same-year swing | Held-out IoU |
|---|---|---|
| A. Current forest | 19.8% | 0.704 |
| B. Trained with extra copies of each chip, changed in brightness like the real between-date differences | 18.4% | 0.706 |
| C. Trained on NDVI and MNDWI only | 19.0% | 0.657 |
| D. Forest A, area from the predicted probabilities instead of the threshold | 14.0% | same as A |

B and C don't help. D passes the rule, but only on paper. The sum of probabilities is about 1.7 times the thresholded area, so the same movement gets divided by a bigger number. In hectares D swings more than A (119 ha against 96 ha). We also made every image 10% darker and 10% brighter to see how far A's area moves. It moves by +6.9% and -4.1% on average. So the forest does react to brightness, but not by enough to explain swings of 20%.

Figures: [sensitivity of A](robustness/sensitivity_A.png), [the four versions](robustness/versions.png).

## Where this leaves us

The date effect is not tide. It goes with how different the images are from date to date, but making the images more alike, or training the forest on brightness changes, did not remove it. So a change map made from single dates with this model still has a noise level of about 20% of the mangrove area on average (9% at Ghantoot, 43% at Jubail), and only changes clearly above that can be trusted.

## What none of this shows

- It doesn't show what the cause is. Image differences go with the swings, but that is a link over 18 dates, and it could also be the height of the sun or the side the satellite looks from.
- The tide heights were not checked against a tide gauge. UNVERIFIED.
- Only the random forest was tested. The U-Net and DINOv3 may react differently.
- Only one simple correction and one augmentation setting were tried. A better one might work.
- There is no ground truth, so we can't say which date is right.
- Training sites only. Al Nouf has very few chips and its numbers are the weakest.
