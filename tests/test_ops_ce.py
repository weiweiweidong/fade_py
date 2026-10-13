"""Tests for ops/ce.py — LoG filter, padding helpers, and compute_ce."""

from __future__ import annotations

import numpy as np

from fade.ops.ce import (
    _build_log_filter,
    _pad_border,
    _unpad_border,
    compute_ce,
    compute_ce_from_channels,
)


# ---------------------------------------------------------------------------
# _build_log_filter
# ---------------------------------------------------------------------------

class TestBuildLogFilter:

    def test_returns_2d_row_vector(self):
        gx = _build_log_filter()
        assert gx.ndim == 2
        assert gx.shape[0] == 1

    def test_has_multiple_coefficients(self):
        gx = _build_log_filter()
        assert gx.shape[1] > 1

    def test_dc_response_near_zero(self):
        """LoG is a high-pass filter → sum of coefficients ≈ 0."""
        gx = _build_log_filter()
        assert abs(gx.sum()) < 1e-6

    def test_roughly_antisymmetric_center(self):
        """LoG kernel should have a large negative peak near the center."""
        gx = _build_log_filter()
        # The center region should contain the negative lobe
        mid = gx.shape[1] // 2
        assert gx[0, mid] < 0

    def test_custom_sigma(self):
        gx1 = _build_log_filter(sigma=2.0)
        gx2 = _build_log_filter(sigma=4.0)
        # Different sigmas → different kernels
        assert gx1.shape != gx2.shape or not np.allclose(gx1, gx2)


# ---------------------------------------------------------------------------
# _pad_border / _unpad_border round-trip
# ---------------------------------------------------------------------------

class TestPadBorder:

    def test_shape_increases_by_border(self):
        img = np.ones((10, 10))
        padded = _pad_border(img, 20)
        assert padded.shape == (30, 30)

    def test_round_trip_even_border(self):
        img = np.arange(100.0).reshape(10, 10)
        padded = _pad_border(img, 20)
        recovered = _unpad_border(padded, 20)
        np.testing.assert_array_equal(img, recovered)

    def test_round_trip_2d_random(self):
        rng = np.random.default_rng(7)
        img = rng.random((20, 30))
        for border in [10, 20]:
            padded = _pad_border(img, border)
            recovered = _unpad_border(padded, border)
            np.testing.assert_array_equal(img, recovered)

    def test_padded_image_contains_original(self):
        """Original image should appear verbatim inside the padded result."""
        img = np.arange(50.0).reshape(5, 10)
        border = 20
        uc = border // 2  # = 10, dc = 9
        padded = _pad_border(img, border)
        # The original image sits at rows [uc:uc+H], cols [uc:uc+W]
        H, W = img.shape
        np.testing.assert_array_equal(padded[uc:uc+H, uc:uc+W], img)


# ---------------------------------------------------------------------------
# compute_ce
# ---------------------------------------------------------------------------

class TestComputeCe:

    def test_returns_three_arrays(self, standard_rgb):
        result = compute_ce(standard_rgb)
        assert len(result) == 3

    def test_output_shapes_match_input(self, standard_rgb):
        H, W = standard_rgb.shape[:2]
        ce_gray, ce_by, ce_rg = compute_ce(standard_rgb)
        assert ce_gray.shape == (H, W)
        assert ce_by.shape == (H, W)
        assert ce_rg.shape == (H, W)

    def test_all_outputs_nonnegative(self, standard_rgb):
        ce_gray, ce_by, ce_rg = compute_ce(standard_rgb)
        assert np.all(ce_gray >= 0)
        assert np.all(ce_by >= 0)
        assert np.all(ce_rg >= 0)

    def test_no_nan_no_inf(self, standard_rgb):
        ce_gray, ce_by, ce_rg = compute_ce(standard_rgb)
        for arr in (ce_gray, ce_by, ce_rg):
            assert not np.any(np.isnan(arr))
            assert not np.any(np.isinf(arr))

    def test_uniform_image_ce_zero(self, uniform_rgb):
        """Uniform image has no edges → CE should be zero everywhere."""
        ce_gray, ce_by, ce_rg = compute_ce(uniform_rgb)
        np.testing.assert_allclose(ce_gray, 0.0)
        np.testing.assert_allclose(ce_by, 0.0)
        np.testing.assert_allclose(ce_rg, 0.0)

    def test_reproducible(self, standard_rgb):
        ce1 = compute_ce(standard_rgb)
        ce2 = compute_ce(standard_rgb)
        for a, b in zip(ce1, ce2):
            np.testing.assert_array_equal(a, b)

    def test_accepts_float_input(self, standard_rgb):
        img_float = standard_rgb.astype(np.float64)
        result = compute_ce(img_float)
        assert len(result) == 3
        for arr in result:
            assert not np.any(np.isnan(arr))

    def test_prepared_channels_match_rgb_and_remain_unchanged(self, standard_rgb):
        r, g, b = standard_rgb.astype(np.float64).transpose(2, 0, 1)
        channels = (0.299 * r + 0.587 * g + 0.114 * b, 0.5 * (r + g) - b, r - g)
        original = [channel.copy() for channel in channels]
        expected = compute_ce(standard_rgb)
        actual = compute_ce_from_channels(*channels)
        for result, reference, channel, before in zip(actual, expected, channels, original):
            np.testing.assert_array_equal(result, reference)
            np.testing.assert_array_equal(channel, before)
