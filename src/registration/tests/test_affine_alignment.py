"""Tests for src/registration/affine_alignment.py.

Most tests make up a transform, apply it to random points and check that the
fitting code gets the same transform back. The last two write small fake
GeoTIFFs to check the raster warp.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import affine_alignment as aa  # noqa: E402

# 50 random points roughly inside the Jubail scene, in UTM metres
RNG = np.random.default_rng(0)
PTS = np.column_stack([RNG.uniform(238000, 256000, 50), RNG.uniform(2709000, 2720000, 50)])


def make(A, t):
    """Build a 2x3 transform from a 2x2 matrix and a shift."""
    return np.hstack([np.asarray(A, float), np.asarray(t, float)[:, None]])


def similarity(rot_deg, s, t):
    r = np.radians(rot_deg)
    return make(s * np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]), t)


def test_two_points_recover_similarity_exactly():
    # 2 landmarks = 4 equations = exactly enough for a similarity
    M = similarity(1.5, 1.01, [12.0, -7.0])
    est = aa.fit_similarity(PTS[:2], aa.apply(M, PTS[:2]))
    np.testing.assert_allclose(est, M, atol=1e-6, rtol=1e-9)
    assert aa.errors(est, PTS, aa.apply(M, PTS)).max() < 1e-6


def test_two_points_cannot_recover_shear():
    # a tiny shear (1%) that a similarity can't represent -> error elsewhere in the scene
    M = make([[1.0, 0.01], [0.0, 1.0]], [0, 0])
    est = aa.fit_similarity(PTS[:2], aa.apply(M, PTS[:2]))
    assert aa.errors(est, PTS, aa.apply(M, PTS)).max() > 1.0


def test_three_points_recover_affine_exactly():
    M = make([[1.012, 0.008], [-0.004, 0.991]], [55.0, -140.0])
    idx = [0, 1, 2]
    assert aa.triplet_ok(PTS[idx], min_sep=0, min_angle=1)
    est = aa.fit_affine(PTS[idx], aa.apply(M, PTS[idx]))
    np.testing.assert_allclose(est, M, atol=1e-6, rtol=1e-9)


def test_affine_needs_three_points_and_rejects_collinear():
    with pytest.raises(ValueError):
        aa.fit_affine(PTS[:2], PTS[:2])
    # 3 points on a straight line can't define an affine
    line = np.array([[240000.0, 2710000.0], [241000.0, 2711000.0], [242000.0, 2712000.0]])
    with pytest.raises(ValueError):
        aa.fit_affine(line, line + 5)
    assert aa.min_angle_deg(line) == pytest.approx(0.0, abs=1e-6)
    assert not aa.triplet_ok(line)
    # nudging the middle point by 30 m still leaves it almost a line, so still rejected
    nearly = line + np.array([[0, 0], [0, 30.0], [0, 0]])
    assert not aa.triplet_ok(nearly)


def test_decompose_round_trip():
    d = aa.decompose(similarity(2.0, 0.98, [3.0, 4.0]))
    assert d["rot_deg"] == pytest.approx(2.0)
    assert d["scale_x"] == pytest.approx(0.98) and d["scale_y"] == pytest.approx(0.98)
    assert d["shear"] == pytest.approx(0.0, abs=1e-12)
    assert (d["tx_m"], d["ty_m"]) == (3.0, 4.0)


def test_known_warp_is_inverted(tmp_path):
    # Khaled's random warp, converted to map units, should be undone exactly by an affine fit
    G = make([[10.0, 0.0], [0.0, -10.0]], [237787.7, 2720328.3])
    W, _ = aa.known_warp_map(np.random.default_rng(1), G)
    truth = np.linalg.inv(aa.to3(W))[:2]
    est = aa.fit_affine(aa.apply(W, PTS), PTS)
    assert aa.grid_rmse(est, truth, aa.scene_grid(G)) < 1e-6


def test_real_landmarks_load_inliers_only():
    # Khaled's file has 315 matches, 252 of them RANSAC inliers
    ids, mov, ref = aa.load_landmarks(aa.DEFAULT_RESULTS / "landmarks_for_tehsin.csv")
    assert len(ids) == 252 and mov.shape == ref.shape == (252, 2)
    assert 200000 < mov[:, 0].min() and mov[:, 1].max() < 3000000


def test_warp_raster_shifts_all_bands_and_keeps_metadata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # fake 6-band image where every pixel value is different, so a shift is easy to spot
    h, w, bands = 40, 50, 6
    data = np.stack([np.add.outer(np.arange(h), np.arange(w) * (b + 1)).astype("float32") for b in range(bands)])
    profile = dict(driver="GTiff", height=h, width=w, count=bands, dtype="float32", crs="EPSG:32640",
                   transform=from_origin(240000, 2715000, 10, 10), nodata=-9999.0)
    moving = tmp_path / "moving.tif"
    with rasterio.open(moving, "w", **profile) as dst:
        dst.write(data)
        for b in range(bands):
            dst.set_band_description(b + 1, f"B{b}")
    # pretend the moving image is 20 m (2 px) too far east, so move it 20 m west
    M = make(np.eye(2), [-20.0, 0.0])
    out = tmp_path / "aligned.tif"
    aa.warp_raster(moving, out, M, reference_path=moving, resampling="nearest")
    with rasterio.open(out) as r:
        assert r.crs.to_epsg() == 32640 and r.count == bands and r.transform == profile["transform"]
        assert r.descriptions == tuple(f"B{b}" for b in range(bands))
        assert json.loads(r.tags()["REGISTRATION_TRANSFORM"]) == M.tolist()
        got = r.read()
    np.testing.assert_array_equal(got[:, :, :-2], data[:, :, 2:])
    # the 2 columns that came from outside the image should be nodata
    assert (got[:, :, -2:] == -9999.0).all()


def test_warp_raster_without_nodata_keeps_zeros_and_masks_outside(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    # like our Earth Engine exports: uint16, no nodata value, and some real 0 pixels
    h, w = 30, 40
    data = np.full((2, h, w), 500, "uint16")
    data[:, :, 10] = 0
    profile = dict(driver="GTiff", height=h, width=w, count=2, dtype="uint16", crs="EPSG:32640",
                   transform=from_origin(240000, 2715000, 10, 10))
    moving = tmp_path / "moving.tif"
    with rasterio.open(moving, "w", **profile) as dst:
        dst.write(data)
    out = tmp_path / "aligned.tif"
    aa.warp_raster(moving, out, make(np.eye(2), [-20.0, 0.0]), reference_path=moving, resampling="nearest")
    assert not list(tmp_path.glob("*.msk"))   # mask should be inside the tif, not a side file
    with rasterio.open(out) as r:
        assert r.nodata is None
        got, mask = r.read(), r.read_masks(1)
    # the real zeros are kept, and "outside the image" is in the mask instead
    assert (got[:, :, 8] == 0).all() and (got[:, :, :8] == 500).all()
    assert (mask[:, :-2] == 255).all() and (mask[:, -2:] == 0).all()
