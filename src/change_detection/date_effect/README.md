# Date effect: what causes it?

These scripts test why the random forest's mangrove area changes with the image date. The results are in [results/change_detection/date_effect/RESULTS.md](../../../results/change_detection/date_effect/RESULTS.md).

- `tide_test.py` compares the swings with the water level, from the image and from a tide model.
- `image_differences_test.py` measures how much the images differ on pixels that shouldn't change, corrects the images and runs the forests again.
- `robustness_test.py` compares four versions of the forest (current, trained with brightness changes, indices only, area from probabilities).
- `gauge_check.py` checks the tide model against tide gauges in Oman and Kuwait.
- `shared.py` holds what they share: the held-out forests, the same-year swing and the shuffled-dates test.

Each script writes one `summary.csv` with the numbers RESULTS.md quotes. The working tables, the figure scripts, the gauge download scripts and the extra checks were left out to keep the repo small.

## What they need

- The Train/Val dataset (`catalog.csv`, `hazy_scenes.csv`, `chips/`), which is not in the repository.
- `common.py` in `src/change_detection` (and through it `baseline_rf.py` and `chipio.py`). It is found automatically.
- For `tide_test.py` and `gauge_check.py`: the `pyTMD` package (3.1.0 was used) and `pyproj`, and the open EOT20 tide model from https://doi.org/10.17882/79489 unpacked as `<models>/EOT20/ocean_tides/*.nc`. No login is needed.
- `tide_test.py` looks up each scene in the public Sentinel-2 archive (Earth Search) to get the sensing time, the sun height and the processing version. It reads the record only, no imagery.
- For `gauge_check.py`: the gauge records, downloaded beforehand into one folder. From the IOC Sea Level Station Monitoring Facility (`https://www.ioc-sealevelmonitoring.org/service.php?query=data&code=<code>&timestart=...&timestop=...&format=json`), the 15 days around each scene date for the stations `maji`, `diba`, `wuda` and `kisr`, saved as `<records>/ioc/<code>/*.json`. From the UHSLC server, the hourly fast delivery record of Muscat (station 110) for 2018 to 2021 as `<records>/uhslc/muscat_hourly.csv`.

## How to run

Run them in this order. They share one `<work>` folder, which gets the forests, the image stacks and the working tables. Keep it outside the repository.

```
python tide_test.py              --train-val <Train_Val_Dataset> --models <models> --work <work> --out <results>/tide
python image_differences_test.py --work <work> --out <results>/image_differences
python robustness_test.py        --train-val <Train_Val_Dataset> --work <work> --out <results>/robustness
python gauge_check.py            --models <models> --gauge-records <records> --work <work> --out <results>/tide/gauge
```

Everything runs on a CPU. All four together took about 20 minutes. A step whose output is already in `<work>` is skipped.

## How the swing is measured

The swing is always a same-year swing. A scene's swing (`dev_pct`) is its predicted mangrove area as a % above or below the average of the dates of the same site and the same year, and water level and the image measures are taken the same way. A scene is never compared with another year, so real change between 2018, 2020 and 2025 is not mixed in. A site and year with a single usable date drops out. Rows marked "all years" hold an older version that compared with the site's average over all its years. They are kept for comparison only.

## What was fixed before running

- The number that counts in the correction and in the four versions is the average size of the same-year swing per site. A same-year pair is two non-hazy dates of one site and year, compared like `noise_floor_all_sites.py` does. A pair's swing is the size of the net change as a % of the first date's predicted mangrove. A site's number is the mean over its pairs, and the main number is the mean of the six sites.
- Image correction: "helps" if that mean falls by at least a quarter. The correction uses no labels and no model output. Its reference date is the site's most typical date.
- Four versions: a version "helps" if that mean falls by at least a quarter against the current forest and held-out IoU (all held-out 2020 chips, mean of five folds) is no more than 0.01 lower.
- For the brightness copies, lines measured at the held-out site are never used, and Al Nouf's lines are left out because its fit was fragile.

## Checks built into the scripts

- The forests are trained again from `common.py`, and the run stops if a forest's held-out IoU is not the committed one.
- `image_differences_test.py` and `robustness_test.py` stop if the same-year pairs on the images as they are don't match `results/change_detection/all_sites/noise_floor_all_pairs.csv`.
- `robustness_test.py` first trains with no change at all and stops if that does not give the current forest again.
