"""Tests for models/context.py — FeatureContext properties and helpers."""

from __future__ import annotations

import numpy as np

from fade.models.context import FeatureContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(img: np.ndarray) -> FeatureContext:
    return FeatureContext(image=img)


# ---------------------------------------------------------------------------
# Basic construction and patch grid
# ---------------------------------------------------------------------------

class TestFeatureContextConstruction:

    def test_patch_row_num_standard(self, standard_rgb):
        ctx = make_context(standard_rgb)
        assert ctx.patch_row_num == 64 // 8  # = 8

    def test_patch_col_num_standard(self, standard_rgb):
        ctx = make_context(standard_rgb)
        assert ctx.patch_col_num == 80 // 8  # = 10

    def test_non_aligned_image_crops(self, nonaligned_rgb):
        """70×75×3 should crop to 64×72."""
        ctx = make_context(nonaligned_rgb)
        assert ctx.image_rgb.shape == (64, 72, 3)
        assert ctx.patch_row_num == 8
        assert ctx.patch_col_num == 9

    def test_image_rgb_shape(self, standard_rgb):
        ctx = make_context(standard_rgb)
        assert ctx.image_rgb.shape == (64, 80, 3)

    def test_tiny_image(self, tiny_rgb):
        ctx = make_context(tiny_rgb)
        assert ctx.patch_row_num == 2
        assert ctx.patch_col_num == 2


# ---------------------------------------------------------------------------
# Color channel properties
# ---------------------------------------------------------------------------

class TestColorChannels:

    def test_R_shape_and_range(self, standard_context):
        R = standard_context.R
        assert R.shape == (64, 80)
        assert R.dtype == np.float64
        assert R.min() >= 0
        assert R.max() <= 255

    def test_G_shape_and_range(self, standard_context):
        G = standard_context.G
        assert G.shape == (64, 80)
        assert G.min() >= 0
        assert G.max() <= 255

    def test_B_shape_and_range(self, standard_context):
        B = standard_context.B
        assert B.shape == (64, 80)
        assert B.min() >= 0
        assert B.max() <= 255

    def test_Ig_int_dtype_and_range(self, standard_context):
        Ig = standard_context.Ig_int
        assert Ig.dtype == np.int32
        assert Ig.min() >= 0
        assert Ig.max() <= 255

    def test_Ig_int_formula(self, standard_context):
        """Ig_int = round(0.2989*R + 0.5870*G + 0.1140*B)."""
        ctx = standard_context
        expected = np.round(0.2989 * ctx.R + 0.5870 * ctx.G + 0.1140 * ctx.B).astype(np.int32)
        np.testing.assert_array_equal(ctx.Ig_int, expected)

    def test_Id_range(self, standard_context):
        Id = standard_context.Id
        assert Id.min() >= 0.0
        assert Id.max() <= 1.0

    def test_Id_is_min_channel(self, standard_context):
        ctx = standard_context
        expected = np.minimum(np.minimum(ctx.R, ctx.G), ctx.B) / 255.0
        np.testing.assert_allclose(ctx.Id, expected)

    def test_Is_range(self, standard_context):
        Is = standard_context.Is
        assert Is.min() >= 0.0
        assert Is.max() <= 1.0 + 1e-6

    def test_rg_formula(self, standard_context):
        ctx = standard_context
        np.testing.assert_array_equal(ctx.rg, ctx.R - ctx.G)

    def test_by_formula(self, standard_context):
        ctx = standard_context
        np.testing.assert_allclose(ctx.by, 0.5 * (ctx.R + ctx.G) - ctx.B)

    def test_uniform_Id_is_constant(self, uniform_context):
        """All-128 image → Id = 128/255 everywhere."""
        expected = 128.0 / 255.0
        np.testing.assert_allclose(uniform_context.Id, expected, rtol=1e-6)

    def test_uniform_Is_is_zero(self, uniform_context):
        """Gray image has no saturation."""
        np.testing.assert_allclose(uniform_context.Is, 0.0, atol=1e-6)

    def test_saturation_black_gray_and_primary_colors(self):
        image = np.tile(np.array([
            [0, 0, 0], [128, 128, 128], [255, 255, 255],
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [50, 100, 150],
        ], dtype=np.uint8), (8, 1, 1))
        actual = FeatureContext(image).Is
        expected = np.tile([0, 0, 0, 1, 1, 1, 1, 2 / 3], (8, 1))
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        assert actual.dtype == np.float32


# ---------------------------------------------------------------------------
# MSCN properties
# ---------------------------------------------------------------------------

class TestMscnProperties:

    def test_mscn_shape(self, standard_context):
        assert standard_context.mscn.shape == (64, 80)

    def test_sigma_map_shape(self, standard_context):
        assert standard_context.sigma_map.shape == (64, 80)

    def test_sigma_map_nonnegative(self, standard_context):
        assert np.all(standard_context.sigma_map >= 0)

    def test_cv_shape(self, standard_context):
        assert standard_context.cv.shape == (64, 80)

    def test_mscn_v_shape(self, standard_context):
        assert standard_context.mscn_v.shape == (64, 80)

    def test_mscn_v_patches_shape(self, standard_context):
        ctx = standard_context
        assert standard_context.mscn_v_patches.shape == (ctx.patch_row_num, ctx.patch_col_num, 64)

    def test_no_nan_in_mscn_standard(self, standard_context):
        assert not np.any(np.isnan(standard_context.mscn))

    def test_no_nan_in_mscn_uniform(self, uniform_context):
        assert not np.any(np.isnan(uniform_context.mscn))


# ---------------------------------------------------------------------------
# CE properties
# ---------------------------------------------------------------------------

class TestCeProperties:

    def test_CE_gray_shape(self, standard_context):
        assert standard_context.CE_gray.shape == (64, 80)

    def test_CE_by_shape(self, standard_context):
        assert standard_context.CE_by.shape == (64, 80)

    def test_CE_rg_shape(self, standard_context):
        assert standard_context.CE_rg.shape == (64, 80)

    def test_CE_gray_nonnegative(self, standard_context):
        assert np.all(standard_context.CE_gray >= 0)

    def test_CE_uniform_zero(self, uniform_context):
        np.testing.assert_allclose(uniform_context.CE_gray, 0.0)
        np.testing.assert_allclose(uniform_context.CE_by, 0.0)
        np.testing.assert_allclose(uniform_context.CE_rg, 0.0)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:

    def test_R_cached(self, standard_context):
        r1 = standard_context.R
        r2 = standard_context.R
        assert r1 is r2

    def test_mscn_cached(self, standard_context):
        m1 = standard_context.mscn
        m2 = standard_context.mscn
        assert m1 is m2

    def test_sigma_map_shares_cache_with_mscn(self, tiny_rgb):
        ctx = FeatureContext(image=tiny_rgb)
        _ = ctx.mscn
        # sigma_map is populated as a side-effect; accessing it should not re-compute
        s1 = ctx.sigma_map
        s2 = ctx.sigma_map
        assert s1 is s2

    def test_CE_gray_cached(self, standard_context):
        c1 = standard_context.CE_gray
        c2 = standard_context.CE_gray
        assert c1 is c2


# ---------------------------------------------------------------------------
# to_patches helper
# ---------------------------------------------------------------------------

class TestToPatches:

    def test_output_shape(self, standard_context):
        arr = np.ones((64, 80))
        patches = standard_context.to_patches(arr)
        assert patches.shape == (8, 10, 64)

    def test_data_sum_preserved(self, standard_context):
        rng = np.random.default_rng(5)
        arr = rng.random((64, 80))
        patches = standard_context.to_patches(arr)
        assert abs(arr.sum() - patches.sum()) < 1e-10
