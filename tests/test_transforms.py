"""Unit tests for pluriseg.transforms."""

import numpy as np
import pytest

from pluriseg.transforms import normalize_percentile


def test_normalize_percentile_maps_1_99_to_0_1():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (64, 64)).astype(np.float32)
    out = normalize_percentile(img)
    p1, p99 = np.percentile(out, (1, 99))
    assert p1 == pytest.approx(0.0, abs=1e-5)
    assert p99 == pytest.approx(1.0, abs=1e-5)


def test_normalize_percentile_clips_outliers():
    rng = np.random.default_rng(2)
    img = rng.uniform(0, 100, (32, 32)).astype(np.float32)
    img[5, 5] = 1e6  # outlier beyond the 99th percentile
    out = normalize_percentile(img)
    assert out.max() <= 1.0
    assert out[5, 5] == 1.0  # outlier clips to the 99th-percentile ceiling


def test_normalize_percentile_constant_image():
    img = np.full((16, 16), 7.0, dtype=np.float32)
    out = normalize_percentile(img)
    assert np.all(out == 0.0)


def test_normalize_percentile_channels_share_gain():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 100, (16, 16, 3)).astype(np.float32)
    out = normalize_percentile(img)
    lo, hi = np.percentile(img, (1, 99))
    expected = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    np.testing.assert_allclose(out, expected, atol=1e-6)
