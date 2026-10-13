"""Tests for ops/entropy.py — patch_entropy."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy

from fade.ops.entropy import patch_entropy


class TestPatchEntropy:

    def test_output_shape(self):
        img = np.zeros((16, 16), dtype=np.int32)
        result = patch_entropy(img, patch_row_num=2, patch_col_num=2, patch_size=8)
        assert result.shape == (2, 2)

    def test_uniform_patch_entropy_zero(self):
        """All pixels the same value → entropy = 0."""
        img = np.full((8, 8), 100, dtype=np.int32)
        result = patch_entropy(img, 1, 1, 8)
        assert abs(result[0, 0]) < 1e-10

    def test_two_equal_groups_entropy_one_bit(self):
        """50/50 split between exactly 2 values → entropy = 1.0 bit."""
        img = np.zeros((8, 8), dtype=np.int32)
        img[:4, :] = 0
        img[4:, :] = 1
        result = patch_entropy(img, 1, 1, 8)
        assert abs(result[0, 0] - 1.0) < 1e-10

    def test_all_unique_values_entropy(self):
        """64 unique values in one patch → entropy = log2(64) = 6."""
        img = np.arange(64, dtype=np.int32).reshape(8, 8)
        result = patch_entropy(img, 1, 1, 8)
        assert abs(result[0, 0] - np.log2(64)) < 1e-10

    def test_upper_bound_is_8(self):
        """No patch can exceed 8 bits (256 possible intensity levels)."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64), dtype=np.int32)
        result = patch_entropy(img, 8, 8, 8)
        assert np.all(result <= 8.0 + 1e-10)

    def test_lower_bound_is_zero(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64), dtype=np.int32)
        result = patch_entropy(img, 8, 8, 8)
        assert np.all(result >= 0.0)

    def test_no_nan_no_inf(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 256, (64, 64), dtype=np.int32)
        result = patch_entropy(img, 8, 8, 8)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_multi_patch_independence(self):
        """Patches compute entropy independently."""
        img = np.zeros((16, 16), dtype=np.int32)
        # patch (0,0): 64 unique values → high entropy
        img[:8, :8] = np.arange(64, dtype=np.int32).reshape(8, 8)
        # patch (1,1): constant 0 → zero entropy
        img[8:, 8:] = 0
        result = patch_entropy(img, 2, 2, 8)
        assert result[0, 0] > 1.0
        assert abs(result[1, 1]) < 1e-10

    def test_non_aligned_input_crops_silently(self):
        img = np.zeros((17, 17), dtype=np.int32)
        result = patch_entropy(img, 2, 2, 8)
        assert result.shape == (2, 2)

    def test_all_zeros_image_entropy_zero(self):
        img = np.zeros((16, 16), dtype=np.int32)
        result = patch_entropy(img, 2, 2, 8)
        np.testing.assert_allclose(result, 0.0)

    def test_reproducible(self):
        rng = np.random.default_rng(99)
        img = rng.integers(0, 256, (32, 32), dtype=np.int32)
        r1 = patch_entropy(img, 4, 4, 8)
        r2 = patch_entropy(img, 4, 4, 8)
        np.testing.assert_array_equal(r1, r2)

    def test_large_grid_matches_independent_entropy_at_batch_boundaries(self):
        rng = np.random.default_rng(7)
        rows, cols = 70, 65
        for ps in (4, 8):
            image = rng.integers(0, 256, (rows * ps + 1, cols * ps + 3), dtype=np.int32)
            actual = patch_entropy(image, rows, cols, ps)
            for index in (0, 4095, 4096, 4097, rows * cols - 1):
                row, col = divmod(index, cols)
                patch = image[row * ps : (row + 1) * ps, col * ps : (col + 1) * ps]
                _, counts = np.unique(patch, return_counts=True)
                np.testing.assert_allclose(actual[row, col], entropy(counts, base=2), atol=1e-12)

    def test_empty_patch_grid(self):
        actual = patch_entropy(np.empty((0, 16), dtype=np.int32), 0, 2, 8)
        assert actual.shape == (0, 2)
