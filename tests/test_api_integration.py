"""Integration tests for public API — fade() and fade_with_map().

Real-image tests are marked @pytest.mark.slow and use the three sample images
in test_image/. Run only unit tests with: pytest -m "not slow"
"""

from __future__ import annotations

import numpy as np
import pytest

from fade import fade, fade_with_map


# ---------------------------------------------------------------------------
# Smoke tests (synthetic images — no disk I/O)
# ---------------------------------------------------------------------------


class TestFadeSmoke:

    def test_returns_float(self, standard_rgb):
        result = fade(standard_rgb)
        assert isinstance(result, float)

    def test_nonnegative(self, standard_rgb):
        assert fade(standard_rgb) >= 0.0

    def test_finite(self, standard_rgb):
        assert np.isfinite(fade(standard_rgb))

    def test_tiny_image(self, tiny_rgb):
        result = fade(tiny_rgb)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_uniform_image(self, uniform_rgb):
        """Uniform image is a degenerate edge case; pipeline should not raise."""
        result = fade(uniform_rgb)
        assert isinstance(result, float)  # may be NaN — that's acceptable

    def test_nonaligned_image(self, nonaligned_rgb):
        result = fade(nonaligned_rgb)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestFadeWithMapSmoke:

    def test_returns_tuple(self, standard_rgb):
        result = fade_with_map(standard_rgb)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_is_float(self, standard_rgb):
        score, _ = fade_with_map(standard_rgb)
        assert isinstance(score, float)

    def test_map_shape(self, standard_rgb):
        _, D_map = fade_with_map(standard_rgb)
        assert D_map.shape == (8, 10)  # 64//8=8, 80//8=10

    def test_map_nonnegative(self, standard_rgb):
        _, D_map = fade_with_map(standard_rgb)
        assert np.all(D_map >= 0)

    def test_score_matches_fade(self, standard_rgb):
        """fade() and fade_with_map() should return the same score."""
        s1 = fade(standard_rgb)
        s2, _ = fade_with_map(standard_rgb)
        assert s1 == s2

    def test_tiny_image(self, tiny_rgb):
        score, D_map = fade_with_map(tiny_rgb)
        assert isinstance(score, float)
        assert D_map.shape == (2, 2)

    def test_nonaligned_map_shape(self, nonaligned_rgb):
        _, D_map = fade_with_map(nonaligned_rgb)
        assert D_map.shape == (8, 9)


class TestFadeReproducibility:

    def test_same_input_same_score(self, standard_rgb):
        s1 = fade(standard_rgb)
        s2 = fade(standard_rgb)
        assert s1 == s2

    def test_same_input_same_map(self, standard_rgb):
        _, m1 = fade_with_map(standard_rgb)
        _, m2 = fade_with_map(standard_rgb)
        np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# Real-image integration tests (marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFadeRealImages:
    """End-to-end tests against known-good FADE scores from the notebook."""

    TOLERANCE = 0.10  # 10% relative tolerance

    def test_clear_image_score(self, real_clear_image):
        expected = 0.9113393845
        score = fade(real_clear_image)
        assert (
            abs(score - expected) / expected < self.TOLERANCE
        ), f"Clear image: got {score:.4f}, expected ~{expected}"

    def test_moderate_fog_score(self, real_moderate_fog_image):
        expected = 2.7121977531
        score = fade(real_moderate_fog_image)
        assert (
            abs(score - expected) / expected < self.TOLERANCE
        ), f"Moderate fog: got {score:.4f}, expected ~{expected}"

    def test_dense_fog_score(self, real_dense_fog_image):
        expected = 5.7674386227
        score = fade(real_dense_fog_image)
        assert (
            abs(score - expected) / expected < self.TOLERANCE
        ), f"Dense fog: got {score:.4f}, expected ~{expected}"

    def test_fog_ordering(
        self, real_clear_image, real_moderate_fog_image, real_dense_fog_image
    ):
        """Clear < Moderate < Dense fog in FADE score."""
        s_clear = fade(real_clear_image)
        s_moderate = fade(real_moderate_fog_image)
        s_dense = fade(real_dense_fog_image)
        assert s_clear < s_moderate < s_dense, (
            f"Expected ordering violated: clear={s_clear:.2f}, "
            f"moderate={s_moderate:.2f}, dense={s_dense:.2f}"
        )

    def test_fade_with_map_real_clear(self, real_clear_image):
        score, D_map = fade_with_map(real_clear_image)
        assert isinstance(score, float)
        assert D_map.ndim == 2
        # Finite map values must be non-negative (NaN patches are acceptable)
        assert np.all(D_map[np.isfinite(D_map)] >= 0)

    def test_real_map_shape_clear(self, real_clear_image):
        H, W = real_clear_image.shape[:2]
        _, D_map = fade_with_map(real_clear_image)
        assert D_map.shape == (H // 8, W // 8)

    def test_real_map_shape_dense_fog(self, real_dense_fog_image):
        H, W = real_dense_fog_image.shape[:2]
        _, D_map = fade_with_map(real_dense_fog_image)
        assert D_map.shape == (H // 8, W // 8)
