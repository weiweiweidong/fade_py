"""Distance operators."""

from __future__ import annotations

import numpy as np

_DISTANCE_BATCH_SIZE = 4096


def batch_mahalanobis(mu_matrix: np.ndarray, patch_var: np.ndarray, ref_cov: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distances without allocating all patch covariances."""
    dtype = np.result_type(mu_matrix, patch_var, ref_cov)
    if not np.issubdtype(dtype, np.inexact):
        dtype = np.dtype(np.float64)
    distances = np.empty(mu_matrix.shape[0], dtype=dtype)
    for start in range(0, len(distances), _DISTANCE_BATCH_SIZE):
        stop = start + _DISTANCE_BATCH_SIZE
        mu = mu_matrix[start:stop]
        batch_cov = (patch_var[start:stop, np.newaxis, np.newaxis] + ref_cov) / 2.0
        x = np.linalg.solve(batch_cov, mu[:, :, np.newaxis])
        dist_sq = np.einsum("ni,ni->n", mu, x[:, :, 0])
        distances[start:stop] = np.sqrt(np.maximum(dist_sq, 0.0))
    return distances

