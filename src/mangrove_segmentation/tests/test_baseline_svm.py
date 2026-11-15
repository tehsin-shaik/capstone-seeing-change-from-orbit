"""Tests for baseline_svm.py on tiny synthetic chips (no real data needed).

Run from the code folder:  python -m pytest -q tests/
These check that the code runs and writes the right files and columns. They say nothing
about how well an SVM maps real mangroves.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baseline_svm as b3  # noqa: E402
import evaluate as ev  # noqa: E402
from chipio import DEFAULT_CHANNELS  # noqa: E402

FIXED_COLUMNS = ["model", "variant", "fold", "held_site", "chips", "mangrove_px", "iou", "f1", "precision",
                 "recall", "iou_ci95_lo", "iou_ci95_hi"]
BASELINE_CSV = Path(__file__).resolve().parents[3] / "results" / "mangrove_segmentation" / "baselines" / "cv_results.csv"


def make_chip(path, rng, n_mangrove_cols, size=16):
    """Chip in prepare.py format. Mangrove = left columns, with high NDVI/NIR and low MNDWI."""
    lab = np.zeros((size, size), np.uint8); lab[:, :n_mangrove_cols] = 1
    m = lab == 1
    img = rng.normal(0.05, 0.01, (8, size, size)).astype(np.float32)
    img[DEFAULT_CHANNELS.index("B08")] = np.where(m, 0.30, 0.04) + rng.normal(0, 0.01, m.shape)
    img[DEFAULT_CHANNELS.index("NDVI")] = np.where(m, 0.70, 0.05) + rng.normal(0, 0.03, m.shape)
    img[DEFAULT_CHANNELS.index("MNDWI")] = np.where(m, -0.40, 0.30) + rng.normal(0, 0.03, m.shape)
    np.savez_compressed(path, image=img, label_v4019=lab, label_v4112=lab,
                        cloud=np.zeros(m.shape, bool), nodata=np.zeros(m.shape, bool),
                        channels=np.array(DEFAULT_CHANNELS))


def bounds(x0, y0=0, size_m=2560):
    return f"{x0:.0f},{y0:.0f},{x0 + size_m:.0f},{y0 + size_m:.0f}"


@pytest.fixture
def trainval(tmp_path):
    """5 sites = 5 folds, 2 dates each. s2's chips touch s1's (buffer test); s3's second date is hazy."""
    root = tmp_path / "Train_Val_Dataset"; (root / "chips").mkdir(parents=True)
    rng = np.random.default_rng(0)
    x0 = {"s1": 0, "s2": 2560, "s3": 100_000, "s4": 200_000, "s5": 300_000}   # s2 is right next to s1
    rows, qa = [], []
    for k, site in enumerate(x0, start=1):
        for j, date in enumerate(["2020-02-01", "2020-11-01"]):
            cid = f"{site}_{date}_r00000_c00000"
            make_chip(root / "chips" / f"{cid}.npz", rng, n_mangrove_cols=3 + k + j)
            rows.append({"chip_id": cid, "site": site, "date": date, "year": 2020, "split": "train",
                         "bounds_epsg32640": bounds(x0[site]), "fold": k, "path": f"chips/{cid}.npz"})
            qa.append({"site": site, "date": date, "water_b04_p5": 0.09 if (site, j) == ("s3", 1) else 0.01})
    # a 2018 chip must be ignored by the 2020-only protocol
    rows.append({**rows[0], "chip_id": "s1_2018-03-01_r00000_c00000", "date": "2018-03-01", "year": 2018,
                 "path": "chips/missing.npz"})
    pd.DataFrame(rows).to_csv(root / "catalog.csv", index=False)
    pd.DataFrame(qa).to_csv(tmp_path / "QA_download.csv", index=False)
    return root, tmp_path / "QA_download.csv"


def test_cv_runs_and_writes_baseline_format(trainval, tmp_path):
    root, qa = trainval
    out = tmp_path / "svm_cv"
    b3.run_cv(root / "catalog.csv", root, out, qa=qa, cap=2000, reps=50)
    cv = pd.read_csv(out / "cv_results.csv")
    # same fixed columns, in the same order, as the baselines' table, then one iou_<site> column per site
    assert list(cv.columns[:12]) == FIXED_COLUMNS
    assert list(cv.columns[12:]) == [f"iou_{s}" for s in ("s1", "s2", "s3", "s4", "s5")]
    if BASELINE_CSV.exists():
        assert list(pd.read_csv(BASELINE_CSV, nrows=0).columns[:12]) == FIXED_COLUMNS
    assert len(cv) == 10 and set(cv.variant) == {"a_all_dates", "b_no_haze"}      # 2 variants x 5 folds
    assert sorted(cv[cv.variant == "a_all_dates"].fold) == [1, 2, 3, 4, 5]
    assert (cv.chips == 2).all()                                                   # the 2018 chip is not used
    assert (cv.iou > 0.9).all()                                                    # classes are separable here
    assert ((cv.iou_ci95_lo <= cv.iou) & (cv.iou <= cv.iou_ci95_hi)).all()
    # each fold is scored only on its own held-out site
    r = cv[(cv.variant == "a_all_dates") & (cv.fold == 3)].iloc[0]
    assert r.held_site == "s3" and r.iou_s3 == pytest.approx(r.iou) and np.isnan(r.iou_s1)

    fits = pd.read_csv(out / "fold_fits.csv").set_index(["variant", "fold"])
    assert fits.loc[("a_all_dates", 1), "buffer_chips_dropped"] == 2              # s2 touches held-out s1
    assert fits.loc[("a_all_dates", 4), "buffer_chips_dropped"] == 0
    assert fits.loc[("a_all_dates", 4), "hazy_chips_dropped"] == 0
    assert fits.loc[("b_no_haze", 4), "hazy_chips_dropped"] == 1                  # the hazy s3 scene
    assert fits.loc[("b_no_haze", 3), "hazy_chips_dropped"] == 0                  # s3 held out: nothing to drop...
    assert cv[(cv.variant == "b_no_haze") & (cv.fold == 3)].chips.iloc[0] == 2    # ...and its hazy chip is still scored
    assert (fits.inner_fold != fits.index.get_level_values("fold")).all()

    assert len(list((out / "preds" / "SVM_a_all_dates").glob("*.npz"))) == 10
    assert len(pd.read_csv(out / "per_chip.csv")) == 20
    json.loads((out / "config.json").read_text())


def test_cv_variant_b_needs_qa(trainval, tmp_path):
    root, _ = trainval
    with pytest.raises(SystemExit):
        b3.run_cv(root / "catalog.csv", root, tmp_path / "x", qa=None)


@pytest.mark.parametrize("kernel", b3.KERNELS)
def test_fit_predict_evaluate_like_baseline_rf(tmp_path, kernel):
    chips = tmp_path / "chips"; chips.mkdir()
    rng = np.random.default_rng(1)
    rows = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        make_chip(chips / f"c{i}.npz", rng, n_mangrove_cols=4 + 2 * i)
        rows.append({"chip_id": f"c{i}", "site": f"s{i}", "split": split})
    cat = tmp_path / "catalog.csv"; pd.DataFrame(rows).to_csv(cat, index=False)

    svm, meta = b3.fit(cat, chips, cap=2000, kernel=kernel)
    assert meta["model"] == "svm" and meta["kernel"] == kernel and meta["val_iou"] > 0.95
    b3.predict(cat, chips, {"svm": svm, "meta": meta}, "test", tmp_path / "p_svm")
    res, _ = ev.evaluate(tmp_path / "p_svm", cat, chips, split="test", reps=50)
    assert res["pooled"]["iou"] > 0.95
    json.dumps(meta, default=str)
