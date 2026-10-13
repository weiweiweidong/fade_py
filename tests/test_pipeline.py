"""Tests for pipeline.py — FadePipeline class methods."""

from __future__ import annotations

import numpy as np
import pytest

from fade.pipeline import FadePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline() -> FadePipeline:
    return FadePipeline.default()


# ---------------------------------------------------------------------------
# FadePipeline.default
# ---------------------------------------------------------------------------

class TestFadePipelineDefault:

    def test_has_12_features(self, pipeline):
        assert len(pipeline.features) == 12

    def test_feature_names_in_order(self, pipeline):
        expected = [f"f{i:02d}" for i in range(1, 13)]
        actual = [f.name for f in pipeline.features]
        assert actual == expected


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------

class TestBuildContext:

    def test_returns_feature_context(self, pipeline, standard_rgb):
        from fade.models.context import FeatureContext
        ctx = pipeline.build_context(standard_rgb)
        assert isinstance(ctx, FeatureContext)

    def test_context_patch_dimensions(self, pipeline, standard_rgb):
        ctx = pipeline.build_context(standard_rgb)
        assert ctx.patch_row_num == 8
        assert ctx.patch_col_num == 10


# ---------------------------------------------------------------------------
# compute_feature_maps
# ---------------------------------------------------------------------------

class TestComputeFeatureMaps:

    def test_returns_12_maps(self, pipeline, standard_context):
        maps = pipeline.compute_feature_maps(standard_context)
        assert len(maps) == 12

    def test_each_map_is_2d(self, pipeline, standard_context):
        maps = pipeline.compute_feature_maps(standard_context)
        for m in maps:
            assert m.ndim == 2

    def test_each_map_shape_matches_patch_grid(self, pipeline, standard_context):
        maps = pipeline.compute_feature_maps(standard_context)
        for m in maps:
            assert m.shape == (standard_context.patch_row_num, standard_context.patch_col_num)

    def test_reproducible(self, pipeline, standard_context):
        maps1 = pipeline.compute_feature_maps(standard_context)
        maps2 = pipeline.compute_feature_maps(standard_context)
        for m1, m2 in zip(maps1, maps2):
            np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_shape_columns_equals_12(self, pipeline, standard_context):
        maps = pipeline.compute_feature_maps(standard_context)
        feat = pipeline.build_feature_matrix(maps)
        assert feat.shape[1] == 12

    def test_shape_rows_equals_n_patches(self, pipeline, standard_context):
        ctx = standard_context
        maps = pipeline.compute_feature_maps(ctx)
        feat = pipeline.build_feature_matrix(maps)
        assert feat.shape[0] == ctx.patch_row_num * ctx.patch_col_num

    def test_log1p_applied(self, pipeline, standard_context):
        """Values should be >= 0 because log1p(x >= 0) >= 0."""
        maps = pipeline.compute_feature_maps(standard_context)
        feat = pipeline.build_feature_matrix(maps)
        valid = feat[np.isfinite(feat)]
        assert np.all(valid >= 0)

    def test_empty_feature_maps_returns_empty(self, pipeline):
        result = pipeline.build_feature_matrix([])
        assert result.shape == (0, 0)

    def test_dtype_float64(self, pipeline, standard_context):
        maps = pipeline.compute_feature_maps(standard_context)
        feat = pipeline.build_feature_matrix(maps)
        assert feat.dtype == np.float64

    def test_feature_matrix_transform_does_not_mutate_maps(self, pipeline):
        maps = [np.arange(6, dtype=np.float64).reshape(2, 3), np.ones((2, 3))]
        before = [feature_map.copy() for feature_map in maps]
        actual = pipeline.build_feature_matrix(maps)
        expected = np.column_stack([np.log1p(feature_map.ravel()) for feature_map in before])
        np.testing.assert_array_equal(actual, expected)
        for feature_map, original in zip(maps, before):
            np.testing.assert_array_equal(feature_map, original)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:

    def test_returns_tuple(self, pipeline, standard_rgb):
        result = pipeline.run(standard_rgb)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_score_is_float(self, pipeline, standard_rgb):
        score, _ = pipeline.run(standard_rgb)
        assert isinstance(score, float)

    def test_score_nonnegative(self, pipeline, standard_rgb):
        score, _ = pipeline.run(standard_rgb)
        assert score >= 0.0

    def test_map_shape(self, pipeline, standard_rgb):
        _, D_map = pipeline.run(standard_rgb)
        assert D_map.shape == (8, 10)  # 64//8=8, 80//8=10

    def test_map_nonnegative(self, pipeline, standard_rgb):
        _, D_map = pipeline.run(standard_rgb)
        assert np.all(D_map >= 0)

    def test_uniform_image_runs_without_error(self, pipeline, uniform_rgb):
        """Uniform image is degenerate; pipeline should not raise (score may be NaN)."""
        score, D_map = pipeline.run(uniform_rgb)
        assert isinstance(score, float)

    def test_tiny_image_runs_without_error(self, pipeline, tiny_rgb):
        score, D_map = pipeline.run(tiny_rgb)
        assert isinstance(score, float)
        assert D_map.shape == (2, 2)

    def test_nonaligned_image_map_shape(self, pipeline, nonaligned_rgb):
        """70×75 image → cropped to 64×72 → map (8, 9)."""
        _, D_map = pipeline.run(nonaligned_rgb)
        assert D_map.shape == (8, 9)

    def test_reproducible(self, pipeline, standard_rgb):
        s1, m1 = pipeline.run(standard_rgb)
        s2, m2 = pipeline.run(standard_rgb)
        assert s1 == s2
        np.testing.assert_array_equal(m1, m2)

    def test_score_formula_nonnegative(self, pipeline, standard_rgb):
        """Score = Df / (Dff + 1.0) ≥ 0 since both distances ≥ 0."""
        score, _ = pipeline.run(standard_rgb)
        assert score >= 0
