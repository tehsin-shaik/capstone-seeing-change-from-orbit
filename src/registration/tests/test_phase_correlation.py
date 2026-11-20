"""Tests for phase_correlation_check.py on made-up images with a known shift (no real data needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import fourier_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import phase_correlation_check as pc  # noqa: E402


def scene(seed=0, n=256):
    """Smooth random 'landscape' so there's structure at many scales, like a real chip."""
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=(n, n)))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))
    return np.real(np.fft.ifft2(f / (k + 0.02) ** 1.5))


def shifted(img, d_row, d_col):
    return np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(img), (d_row, d_col))))


@pytest.mark.parametrize("d_row,d_col", [(0, 3), (-2, 0), (0.3, -0.25), (0, 0)])
def test_recovers_known_shift_in_metres(d_row, d_col):
    ref = scene()
    east, north, _ = pc.offset_m(ref, shifted(ref, d_row, d_col))
    # content moved d_col columns right = east, d_row rows down = south
    assert east == pytest.approx(d_col * 10, abs=0.3)
    assert north == pytest.approx(-d_row * 10, abs=0.3)


def test_brightness_change_doesnt_move_the_answer():
    ref = scene(1)
    east, north, _ = pc.offset_m(ref, 1.3 * shifted(ref, 0, 0.5) + 0.2)
    assert east == pytest.approx(5, abs=0.3) and north == pytest.approx(0, abs=0.3)


def test_season_pairs_match_months():
    assert pc.season_pairs(["2018-03-18", "2018-11-23"], ["2025-03-18", "2025-11-16"]) == [
        ("2018-03-18", "2025-03-18"), ("2018-11-23", "2025-11-16")]
    assert pc.season_pairs(["2018-03-18"], ["2025-11-16"]) == []


def test_summary_takes_the_median():
    d = pd.DataFrame(dict(site="s", pair="p", east_m=[1, 2, 3, 100], north_m=[0, 0, 0, 0], bands_agree=True))
    s = pc.summarise(d, reps=50).iloc[0]
    assert s.east_m == 2.5 and s.chips == 4                  # one bad chip doesn't drag it


def test_summary_drops_chips_where_the_bands_disagree():
    d = pd.DataFrame(dict(site="s", pair="p", east_m=[1, 2, 3, 600], north_m=0.0,
                          bands_agree=[True, True, True, False]))
    s = pc.summarise(d, reps=50).iloc[0]
    assert s.chips == 3 and s.east_m == 2 and s.enough_chips
