"""Tests for ops/distance.py — batch_mahalanobis."""

from __future__ import annotations

import numpy as np

from fade.ops.distance import batch_mahalanobis


class TestBatchMahalanobis:

    def _identity_cov(self, d: int) -> np.ndarray:
        return np.eye(d, dtype=np.float64)

    def test_output_shape(self):
        N, D = 10, 12
        mu = np.ones((N, D))
        var = np.ones(N)
        cov = self._identity_cov(D)
        result = batch_mahalanobis(mu, var, cov)
        assert result.shape == (N,)

    def test_all_distances_nonnegative(self):
        rng = np.random.default_rng(0)
        N, D = 20, 12
        mu = rng.standard_normal((N, D))
        var = rng.random(N) * 2 + 0.1
        cov = np.eye(D)
        result = batch_mahalanobis(mu, var, cov)
        assert np.all(result >= 0)

    def test_zero_vector_gives_zero_distance(self):
        N, D = 5, 12
        mu = np.zeros((N, D))
        var = np.ones(N)
        cov = self._identity_cov(D)
        result = batch_mahalanobis(mu, var, cov)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_identity_cov_euclidean_approx(self):
        """With identity covariance and unit patch variance, distance ≈ sqrt(mu^T mu / 2)."""
        N, D = 3, 4
        mu = np.array([[1.0, 0, 0, 0],
                        [0, 2.0, 0, 0],
                        [1.0, 1.0, 0, 0]])
        var = np.zeros(N)  # patch_var=0 → batch_cov = I/2
        cov = self._identity_cov(D)
        result = batch_mahalanobis(mu, var, cov)
        # batch_cov = (0 * I + I) / 2 = I/2
        # x = solve(I/2, mu.T) = 2*mu
        # dist_sq = mu · 2*mu = 2 * ||mu||^2
        expected = np.sqrt(2.0) * np.array([1.0, 2.0, np.sqrt(2.0)])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_reproducible(self):
        rng = np.random.default_rng(42)
        N, D = 15, 12
        mu = rng.standard_normal((N, D))
        var = rng.random(N)
        cov = np.eye(D)
        r1 = batch_mahalanobis(mu, var, cov)
        r2 = batch_mahalanobis(mu, var, cov)
        np.testing.assert_array_equal(r1, r2)

    def test_no_nan_no_inf_for_typical_input(self):
        rng = np.random.default_rng(3)
        N, D = 50, 12
        mu = rng.standard_normal((N, D))
        var = rng.random(N) * 5
        cov = np.eye(D)
        result = batch_mahalanobis(mu, var, cov)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_larger_patch_var_increases_distance(self):
        """Larger patch variance inflates batch_cov, but effect on distance depends on mu."""
        D = 4
        mu = np.ones((1, D)) * 2.0
        cov = self._identity_cov(D)
        d_small = batch_mahalanobis(mu, np.array([0.0]), cov)
        d_large = batch_mahalanobis(mu, np.array([100.0]), cov)
        # With larger patch_var, batch_cov is larger, inverse is smaller → smaller distance
        assert d_large[0] < d_small[0]

    def test_large_batch_matches_direct_solve(self):
        rng = np.random.default_rng(8)
        n, d = 5003, 12
        matrix = rng.standard_normal((d, d))
        cov = matrix @ matrix.T + np.eye(d)
        mu = rng.standard_normal((n, d))
        var = rng.random(n)
        var[4096] = np.nan
        mu[4097, 0] = np.nan
        actual = batch_mahalanobis(mu, var, cov)
        for index in (0, 4095, 4096, 4097, n - 1):
            solution = np.linalg.solve((var[index] + cov) / 2.0, mu[index])
            expected = np.sqrt(max(mu[index] @ solution, 0.0))
            np.testing.assert_allclose(actual[index], expected, rtol=1e-12, equal_nan=True)

    def test_float32_input_preserves_dtype(self):
        mu = np.ones((5003, 4), dtype=np.float32)
        var = np.ones(5003, dtype=np.float32)
        cov = np.eye(4, dtype=np.float32)
        actual = batch_mahalanobis(mu, var, cov)
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, np.sqrt(1.6), rtol=1e-6)
