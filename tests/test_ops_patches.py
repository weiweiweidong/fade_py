"""Tests for ops/patches.py — to_patches."""

from __future__ import annotations

import numpy as np

from fade.ops.patches import to_patches


class TestToPatches:

    def test_shape_single_patch(self):
        arr = np.arange(64.0).reshape(8, 8)
        result = to_patches(arr, patch_row_num=1, patch_col_num=1, patch_size=8)
        assert result.shape == (1, 1, 64)

    def test_shape_multi_patch(self):
        arr = np.zeros((32, 40))
        result = to_patches(arr, patch_row_num=4, patch_col_num=5, patch_size=8)
        assert result.shape == (4, 5, 64)

    def test_crops_non_aligned_input(self):
        arr = np.zeros((35, 37))
        result = to_patches(arr, patch_row_num=4, patch_col_num=4, patch_size=8)
        assert result.shape == (4, 4, 64)

    def test_data_preserved_all_values_present(self):
        arr = np.arange(64.0).reshape(8, 8)
        result = to_patches(arr, 1, 1, 8)
        np.testing.assert_array_equal(np.sort(result.ravel()), np.arange(64.0))

    def test_top_left_patch_correct_values(self):
        """Patch (0,0) contains the top-left 8×8 block, in row-major order."""
        arr = np.zeros((16, 16))
        arr[:8, :8] = 1.0
        result = to_patches(arr, 2, 2, 8)
        np.testing.assert_array_equal(result[0, 0], np.ones(64))
        np.testing.assert_array_equal(result[0, 1], np.zeros(64))
        np.testing.assert_array_equal(result[1, 0], np.zeros(64))
        np.testing.assert_array_equal(result[1, 1], np.zeros(64))

    def test_identity_single_patch_row_major(self):
        arr = np.arange(64.0).reshape(8, 8)
        result = to_patches(arr, 1, 1, 8)
        np.testing.assert_array_equal(result[0, 0], arr.ravel())

    def test_sum_preserved(self):
        rng = np.random.default_rng(0)
        arr = rng.random((32, 40))
        result = to_patches(arr, 4, 5, 8)
        assert abs(arr[:32, :40].sum() - result.sum()) < 1e-10

    def test_preserves_float32_dtype(self):
        arr = np.ones((16, 16), dtype=np.float32)
        result = to_patches(arr, 2, 2, 8)
        assert result.dtype == np.float32

    def test_preserves_int32_dtype(self):
        arr = np.ones((16, 16), dtype=np.int32)
        result = to_patches(arr, 2, 2, 8)
        assert result.dtype == np.int32

    def test_patch_size_4(self):
        arr = np.arange(16.0).reshape(4, 4)
        result = to_patches(arr, patch_row_num=1, patch_col_num=1, patch_size=4)
        assert result.shape == (1, 1, 16)
        np.testing.assert_array_equal(np.sort(result.ravel()), np.arange(16.0))

    def test_adjacent_patches_non_overlapping(self):
        """Values written to adjacent regions appear in distinct patches."""
        arr = np.zeros((16, 16))
        arr[:8, :8] = 1.0
        arr[:8, 8:] = 2.0
        arr[8:, :8] = 3.0
        arr[8:, 8:] = 4.0
        result = to_patches(arr, 2, 2, 8)
        assert np.all(result[0, 0] == 1.0)
        assert np.all(result[0, 1] == 2.0)
        assert np.all(result[1, 0] == 3.0)
        assert np.all(result[1, 1] == 4.0)
