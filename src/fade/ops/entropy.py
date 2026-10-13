"""Entropy operators."""

from __future__ import annotations

import numpy as np

from .patches import to_patches

_HISTOGRAM_BATCH_SIZE = 4096


def patch_entropy(Ig_int: np.ndarray, patch_row_num: int, patch_col_num: int, patch_size: int) -> np.ndarray:
    """Compute patch entropy with bounded histogram working memory."""
    N = patch_row_num * patch_col_num
    area = patch_size * patch_size

    flat = to_patches(Ig_int, patch_row_num, patch_col_num, patch_size).reshape(N, area)

    ent = np.empty(N, dtype=np.float64)
    for start in range(0, N, _HISTOGRAM_BATCH_SIZE):
        batch = flat[start : start + _HISTOGRAM_BATCH_SIZE]
        offsets = np.arange(len(batch), dtype=np.int32) * 256
        flat_idx = batch + offsets[:, np.newaxis]
        probs = np.bincount(flat_idx.ravel(), minlength=len(batch) * 256).reshape(
            len(batch), 256
        ).astype(np.float64)
        probs /= area
        log_probs = np.zeros_like(probs)
        np.log2(probs, out=log_probs, where=probs > 0.0)
        probs *= log_probs
        ent[start : start + len(batch)] = -probs.sum(axis=1)
    return ent.reshape(patch_row_num, patch_col_num)

