"""Tests for shift_chips.py on a made-up chip with a known shift."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import fourier_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shift_chips as sc  # noqa: E402


def scene(seed=0, n=256):
    rng = np.random.default_rng(seed)
    f = np.fft.fft2(rng.normal(size=(n, n)))
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n)))
    return np.real(np.fft.ifft2(f / (k + 0.02) ** 1.5)).astype(np.float32)


def test_pixel_shift_direction():
    # 2025 is 5 m north and 2 m east of 2018 -> translation moving->reference is (-2, -5) m
    assert np.allclose(sc.pixel_shift(-2.0, -5.0), (0.5, -0.2))


def test_shift_removes_known_offset(tmp_path):
    ref = scene()
    mov = np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(ref), (-0.5, 0.2)))).astype(np.float32)
    src = tmp_path / "tv"
    (src / "chips").mkdir(parents=True)
    pd.DataFrame({"chip_id": []}).to_csv(src / "catalog.csv", index=False)
    blank = np.zeros((256, 256), bool)
    for d, img in (("2018-03-08", ref), ("2025-03-31", mov)):
        np.savez(src / "chips" / f"site_{d}_r00000_c00000.npz", image=np.stack([img] * 8), cloud=blank, nodata=blank)
    pd.DataFrame({"model": ["none", "translation"], "tx_m": [0.0, -2.0], "ty_m": [0.0, -5.0]}).to_csv(tmp_path / "b.csv", index=False)
    out = tmp_path / "shifted"
    sc.main(["--train-val", str(src), "--out", str(out), "--site", "site",
             "--shift", f"2025-03-31=2018-03-08={tmp_path / 'b.csv'}"])
    rep = pd.read_json(out / "shift_report.json").T.iloc[0]
    assert rep.median_offset_before_m > 4.5
    assert rep.median_offset_after_m < 1.5
    assert (out / "catalog.csv").is_symlink()
    assert (out / "chips" / "site_2018-03-08_r00000_c00000.npz").is_symlink()
    assert not (out / "chips" / "site_2025-03-31_r00000_c00000.npz").is_symlink()
