"""Tests for models/reference_stats.py — load_reference_stats."""

from __future__ import annotations

import numpy as np

from fade.models.reference_stats import load_reference_stats


class TestLoadReferenceStats:

    def test_returns_two_dicts(self):
        fogfree, foggy = load_reference_stats()
        assert isinstance(fogfree, dict)
        assert isinstance(foggy, dict)

    def test_fogfree_has_mu_key(self):
        fogfree, _ = load_reference_stats()
        assert "mu_fogfreeparam" in fogfree

    def test_fogfree_has_cov_key(self):
        fogfree, _ = load_reference_stats()
        assert "cov_fogfreeparam" in fogfree

    def test_foggy_has_mu_key(self):
        _, foggy = load_reference_stats()
        assert "mu_foggyparam" in foggy

    def test_foggy_has_cov_key(self):
        _, foggy = load_reference_stats()
        assert "cov_foggyparam" in foggy

    def test_mu_fogfree_shape(self):
        fogfree, _ = load_reference_stats()
        mu = np.ravel(fogfree["mu_fogfreeparam"])
        assert mu.shape == (12,)

    def test_cov_fogfree_shape(self):
        fogfree, _ = load_reference_stats()
        cov = fogfree["cov_fogfreeparam"]
        assert cov.shape == (12, 12)

    def test_mu_foggy_shape(self):
        _, foggy = load_reference_stats()
        mu = np.ravel(foggy["mu_foggyparam"])
        assert mu.shape == (12,)

    def test_cov_foggy_shape(self):
        _, foggy = load_reference_stats()
        cov = foggy["cov_foggyparam"]
        assert cov.shape == (12, 12)

    def test_cov_fogfree_symmetric(self):
        fogfree, _ = load_reference_stats()
        cov = fogfree["cov_fogfreeparam"].astype(np.float64)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_cov_foggy_symmetric(self):
        _, foggy = load_reference_stats()
        cov = foggy["cov_foggyparam"].astype(np.float64)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_cov_fogfree_positive_semidefinite(self):
        fogfree, _ = load_reference_stats()
        cov = fogfree["cov_fogfreeparam"].astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-8)

    def test_cov_foggy_positive_semidefinite(self):
        _, foggy = load_reference_stats()
        cov = foggy["cov_foggyparam"].astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-8)

    def test_lru_cache_returns_same_object(self):
        result1 = load_reference_stats()
        result2 = load_reference_stats()
        assert result1 is result2

    def test_mu_values_finite(self):
        fogfree, foggy = load_reference_stats()
        mu_ff = np.ravel(fogfree["mu_fogfreeparam"]).astype(np.float64)
        mu_fg = np.ravel(foggy["mu_foggyparam"]).astype(np.float64)
        assert np.all(np.isfinite(mu_ff))
        assert np.all(np.isfinite(mu_fg))
