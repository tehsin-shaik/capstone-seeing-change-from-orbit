# Changes on branch `zain-data-dictionary`

15 Nov 2026. Five new files. No existing file was edited or deleted, and nothing has been committed or pushed.

| File | What it is |
| --- | --- |
| [docs/DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Reference for every data file, column, band and label |
| [src/mangrove_segmentation/baseline_svm.py](../src/mangrove_segmentation/baseline_svm.py) | SVM baseline, built to match the random forest baseline |
| [src/mangrove_segmentation/tests/test_baseline_svm.py](../src/mangrove_segmentation/tests/test_baseline_svm.py) | Automatic tests for the SVM baseline, on made-up chips |
| [src/mangrove_segmentation/make_checksums.py](../src/mangrove_segmentation/make_checksums.py) | Writes a checksum table for a folder of exported files |
| [docs/CHANGES_ZAIN.md](CHANGES_ZAIN.md) | This summary |

The short version for the supervisor:

- The data dictionary is complete for the layout of the data, but it describes the data through the code, because the data files are not in the repository.
- The SVM baseline is written and passes its tests on made-up chips. **It has not been run on the real mangrove chips, so there is no SVM score yet.**
- The checksum script works and was checked against known checksums. **It has not been run on the Sentinel-2 exports**, so there is still no checksum table for them.

## 1. Data dictionary

**What it is.** One document that answers "what does this file, column or number mean?" for the whole project: the 13 sites, the Sentinel-2 bands, how reflectance is computed, the coordinate system, what is inside a chip, all 17 columns of `catalog.csv`, the normalisation files, the mangrove label versions, and the file naming rules. Every fact names the file it came from.

**How to use it.** Read it; nothing to run. Section 11 lists every place where the code and a document disagree. Section 12 lists everything that could not be confirmed.

**The main findings.**

1. **There are two different datasets in the repository** and they are easy to confuse. The main pipeline (the code in `src/mangrove_segmentation`) uses 8 channels, 13 sites and Global Mangrove Watch (GMW) v4.0.19 as its training label. The "Seeing Orbit" set (the zip of test chips and the U-Net report PDF) uses 6 channels, Eastern Mangroves only, and GMW v4.1.12. They also use different water indices: MNDWI in the pipeline, NDWI in Seeing Orbit.
2. **Coordinate system: the code uses EPSG:32640 everywhere.** The number 32639 does not appear anywhere in the repository. The likely origin of the confusion is that one site, Al Aryam, straddles Sentinel-2 tiles from UTM zone 39, which are delivered in EPSG:32639. The download script converts everything to 32640.
3. **Reflectance: the code applies the scale and offset once** (`DN × 0.0001 − 0.1`) and never clips the result. The Seeing Orbit test chips are different: 30% to 54% of their optical values are exactly 0 and none are negative, which is what clipping at 0 produces. The script that made them is not in the repository, so the cause is not confirmed.
4. **GMW version: the code stores two versions in every chip and trains and scores on v4.0.19** (10 m, 2020 only). v4.1.12 (about 30 m; 2018, 2020, 2025) is stored alongside. The Seeing Orbit material uses v4.1.12 only.
5. **`docs/PROGRESS.md` says checksums of the exports were recorded in the data documentation. They were not.** `data/README.md` only lists checksums as something that should be kept. The checksum script below is the fix.

**What is UNVERIFIED.** The full list is section 12 of the dictionary. The important ones:

- `catalog.csv`, `norm.json`, `sites.geojson`, the fold table and the haze table are not in the repository. Their layout comes from the code that writes or reads them, not from the files.
- The exact image dates, the boundaries of the sites, and the data names of the seven locked test sites.
- What `07` means in `KAB_2020_07`. The commit message explains `KAB` (Khor Al Beida) and `2020`, and nothing explains `07`.
- Whether the code in the repository is the exact version that produced the numbers in `results/`.

**Two things to raise with the team**, found while writing it:

- The hand-drawn label `KAB_2020_07` is for Khor Al Beida, which the code lists as a locked external test site ("do not open, plot, train on").
- Three files the code needs for cross-validation are missing from the repository: `folds_site_cv.csv`, `QA_download.csv` and `sites.geojson`. So are the scripts that ran the baselines (`run_cv.py`, `score.py`).

## 2. SVM baseline

**What it is.** The project plan gives Zain a "random forest and SVM baseline". The random forest exists (`baseline_rf.py`, called B2). This adds the SVM. An SVM (support vector machine) is a classic classifier that looks at one pixel at a time, here its 8 values (6 bands, NDVI, MNDWI), and decides mangrove or not.

**How it matches the random forest.**

| | Random forest (`baseline_rf.py`) | SVM (`baseline_svm.py`) |
| --- | --- | --- |
| Inputs | Chips and catalog, label v4.0.19 by default | Same |
| Training pixels | 200,000, half mangrove and half not | Same. It calls the random forest's own sampling function |
| What is tuned | Tree settings and a probability cut-off, on validation data | The SVM's `C` setting and a score cut-off, on validation data |
| Commands | `fit`, `predict` | `fit`, `predict` with the same options, plus `cv` |
| Output masks | One `<chip_id>.npz` per chip | Same |
| Scoring | `evaluate.py` | Same |
| Results table | `cv_results.csv` | Same columns, in the same order |

**Why a linear SVM.** The usual curved-boundary SVM (RBF kernel) compares every training pixel with every other one, so it becomes very slow as data grows. With 200,000 training pixels and about 14 million pixels to predict, it would take hours to days. The linear version scales in proportion to the data. The cost is that it can only draw one flat boundary through the 8 values, so it is expected to score closer to the NDVI rule than to the random forest. If a curved boundary is wanted, `--kernel rbf_approx` adds a standard fast approximation of the RBF kernel. This reasoning is also written in the comments at the top of the script.

**Leave-one-site-out cross-validation.** `baseline_rf.py` does not contain the cross-validation itself. A separate script, `run_cv.py`, ran it, and that script is not in the repository. The `cv` command in `baseline_svm.py` therefore rebuilds the procedure from the two places that describe it: the baselines' `RESULTS.md` and `train_unet.py`. In each of 5 rounds one site is held out; the SVM is tuned on an inner fold, refitted on the other 4 folds, and scored on the held-out site. Training chips within one chip (2,560 m) of a held-out chip are dropped. It runs twice: with all 2020 images (a), and with hazy images dropped from training (b).

**How to run it.**

```
cd src/mangrove_segmentation

# cross-validation, to compare with B1, B2, U-Net and DINOv3
python baseline_svm.py cv --catalog <Train_Val_Dataset>/catalog.csv --chips <Train_Val_Dataset> --qa <QA_download.csv> --out results_svm

# or the same train / validate / test route as baseline_rf.py
python baseline_svm.py fit     --catalog catalog.csv --chips chips/ --label v4019 --out models/svm.joblib
python baseline_svm.py predict --catalog catalog.csv --chips chips/ --model models/svm.joblib --split test --pred-out preds/svm
python evaluate.py --pred preds/svm --catalog catalog.csv --chips chips/ --split test --label v4019 --name "B3 SVM" --out results/svm

# the tests
python -m pytest -q tests/
```

`<Train_Val_Dataset>` is the folder `6_Work/datasets/Train_Val_Dataset` in the team's shared project folder. The `cv` command writes `cv_results.csv`, `fold_fits.csv`, `per_chip.csv`, `cv_catalog_2020.csv`, `config.json` and the predicted masks.

**What was tested.**

- **The existing 6 tests still pass.**
- **4 new tests pass**, in `tests/test_baseline_svm.py`. They build 10 tiny made-up chips across 5 pretend sites and check that: the cross-validation runs for both variants; the results table has the same columns as the baselines' real `cv_results.csv`; each round is scored only on its held-out site; chips next to a held-out chip are dropped; hazy scenes are dropped from training but still scored; chips from other years are ignored; and the `fit` / `predict` / `evaluate.py` route works for both the linear and the `rbf_approx` option.

**What could not be run, and is UNVERIFIED.**

- **The SVM has not been run on real data.** The chips, the catalog, the fold table and the haze table are not in the repository. There is no SVM IoU to report and no row to add to the model comparison yet. The made-up chips are perfectly separable, so the 1.000 scores in the tests mean only that the code works.
- **Running time on the real data** is not measured.
- **Whether the rebuilt cross-validation matches the original `run_cv.py` in every detail.** The written description was followed, including the pixel budgets (100,000 pixels for tuning, 200,000 for the final fit). One detail is not written down for the baselines: whether training chips next to the inner validation fold were also dropped. The SVM script drops them, as `train_unet.py` does. If `run_cv.py` is added to the repository, the two should be compared.
- **The cut-off grid** (−2 to 4 on the SVM score) was chosen without real data. The script prints a note if the best cut-off lands on the edge of the grid, which would mean the grid needs widening.
- **The project's own Python environment.** This computer has no conda, so the tests ran with the Python 3.12 that ships with QGIS, plus scikit-learn 1.9.1 installed in a temporary folder. They were not run inside the project's `mangrove` environment.

## 3. Checksum script

**What it is.** The large Sentinel-2 exports are kept outside Git. A checksum is a 64-character fingerprint of a file: if one byte changes, the fingerprint changes. This script walks a folder, including sub-folders, and writes a CSV with one row per file: `filename`, `size_bytes`, `sha256`. Committing that CSV lets anyone confirm that the file they hold is exactly the file an experiment used.

**How to run it.**

```
cd src/mangrove_segmentation
python make_checksums.py "D:/path/to/exports" --out ../../data/checksums_exports.csv --pattern "*.tif"
```

Leave out `--pattern` to include every file. It only reads files; it never changes, moves or deletes anything. It needs no extra packages.

**What was tested.** As asked, it was not run on made-up data. It was checked on two real things already in the repository, with the output written to a temporary folder:

- The 13 files inside the test-chip zip. The script's checksums match the zip's own `SHA256SUMS.txt` for all 13.
- `data/hand_labels/KAB_2020_07.gpkg`. The script's checksum matches the one from the standard `sha256sum` tool.

**What is not done.** It has not been run on the Sentinel-2 exports, because they are not in the repository. Until someone runs it on the real export folder and commits the CSV, the sentence in `docs/PROGRESS.md` about recorded checksums is still not true.

## 4. What is left to do

1. Get the Train/Val dataset from the team's shared storage and run `baseline_svm.py cv`. Add the result to the model comparison.
2. Run `make_checksums.py` on the export folder and commit the CSV.
3. Ask the team for the missing input files (`folds_site_cv.csv`, `QA_download.csv`, `sites.geojson`) and the baseline scripts (`run_cv.py`, `score.py`), so that the results in the repository can be reproduced from the repository.
4. Ask whoever made `KAB_2020_07` what `07` means, and record the hand-label naming rule.
5. Review these five files, then commit them.
