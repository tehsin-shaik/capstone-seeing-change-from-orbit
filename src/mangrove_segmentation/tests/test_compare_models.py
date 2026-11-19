"""Tests for compare_models.py using small made-up cv_results files, plus one check on the real ones.

Run from the code folder:  python -m pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import compare_models as cm  # noqa: E402

SITES = ["eastern_mangroves", "jubail_island", "ras_ghurab", "ghantoot", "al_aryam_islands,al_nouf"]
TOY_MODELS = [("Weak", "weak", "W"), ("Strong", "strong", "S"), ("Twin", "twin", "T")]


def write_cv(folder: Path, code: str, ious, sites=SITES, variant=cm.VARIANT):
    """Tiny cv_results.csv with just the columns compare_models.py uses."""
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(model=code, variant=variant, fold=range(1, len(ious) + 1), held_site=sites,
                      iou=ious)).to_csv(folder / "cv_results.csv", index=False)


@pytest.fixture
def toy(tmp_path):
    # Strong is Weak + 0.1 on every fold, Twin is Strong with small ups and downs
    weak = np.array([0.60, 0.40, 0.55, 0.65, 0.50])
    write_cv(tmp_path / "weak", "W", weak)
    write_cv(tmp_path / "strong", "S", weak + 0.10)
    write_cv(tmp_path / "twin", "T", weak + 0.10 + np.array([0.01, -0.01, 0.005, -0.005, 0.0]))
    return tmp_path


def test_clear_difference_and_top_group(toy):
    table = cm.load_fold_iou(toy, TOY_MODELS)
    pairs = cm.pairwise(table)
    strong_vs_weak = pairs[(pairs.model_a == "Strong") & (pairs.model_b == "Weak")].iloc[0]
    assert strong_vs_weak.mean_diff == pytest.approx(0.10)
    assert strong_vs_weak.a_wins == 5 and strong_vs_weak.clear_difference
    # 5/5 wins is the lowest p a 5-fold Wilcoxon can give (2/32)
    assert strong_vs_weak.wilcoxon_p == pytest.approx(0.0625)
    twin_pair = pairs[{"Strong", "Twin"} == pairs[["model_a", "model_b"]].apply(set, axis=1)].iloc[0]
    assert not twin_pair.clear_difference
    # Weak is clearly worse, the other two tie
    assert cm.top_group(table, pairs) == ["Strong", "Twin"]


def test_model_a_has_the_higher_mean(toy):
    pairs = cm.pairwise(cm.load_fold_iou(toy, TOY_MODELS))
    assert (pairs.mean_diff >= 0).all()


def test_exact_bootstrap_matches_simple_cases():
    assert cm.exact_bootstrap_ci(np.full(5, 0.2)) == pytest.approx((0.2, 0.2))
    lo, hi = cm.exact_bootstrap_ci(np.array([-0.02, -0.01, 0.0, 0.01, 0.02]))
    assert lo == pytest.approx(-hi) and lo < 0 < hi           # symmetric input, symmetric interval
    with pytest.raises(ValueError):
        cm.exact_bootstrap_ci(np.zeros(8))


def test_identical_models_give_p_of_one():
    assert cm.wilcoxon_p(np.zeros(5)) == 1.0


def test_holm_never_lowers_p_and_caps_at_one():
    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    adjusted = cm.holm(p)
    assert (adjusted >= p).all() and (adjusted <= 1).all()
    assert adjusted.iloc[0] == pytest.approx(0.04)             # smallest p x 4 tests


def test_missing_model_is_skipped(toy):
    table = cm.load_fold_iou(toy, TOY_MODELS + [("Ghost", "not_there", "G")])
    assert "Ghost" not in table.columns


def test_mismatched_folds_raise(toy):
    write_cv(toy / "twin", "T", [0.7] * 5, sites=list(reversed(SITES)))
    with pytest.raises(ValueError, match="different folds"):
        cm.load_fold_iou(toy, TOY_MODELS)


def test_duplicate_fold_rows_raise(tmp_path):
    write_cv(tmp_path / "weak", "W", [0.5, 0.6], sites=["a", "b"])
    df = pd.read_csv(tmp_path / "weak" / "cv_results.csv")
    pd.concat([df, df]).to_csv(tmp_path / "weak" / "cv_results.csv", index=False)
    with pytest.raises(ValueError, match="more than one row per fold"):
        cm.load_fold_iou(tmp_path, TOY_MODELS[:1])


def test_other_variants_are_ignored(toy):
    # a_all_dates rows shouldn't end up in the b_no_haze comparison
    df = pd.read_csv(toy / "weak" / "cv_results.csv")
    extra = df.assign(variant="a_all_dates", iou=0.99)
    pd.concat([df, extra]).to_csv(toy / "weak" / "cv_results.csv", index=False)
    assert cm.load_fold_iou(toy, TOY_MODELS)["Weak"].max() < 0.99


@pytest.mark.skipif(not (cm.RESULTS / "unet" / "cv_results.csv").exists(), reason="results folder not present")
def test_real_results_load_with_matching_folds():
    table = cm.load_fold_iou()
    assert len(table) == 5
    assert table["Random forest"].mean() == pytest.approx(0.704, abs=5e-4)
