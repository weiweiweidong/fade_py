"""Reusable low-level operators for FADE."""

from .ce import compute_ce, compute_ce_from_channels
from .distance import batch_mahalanobis
from .entropy import patch_entropy
from .mscn import compute_mscn_components, gaussian_window
from .patches import to_patches

__all__ = [
    "batch_mahalanobis",
    "compute_ce",
    "compute_ce_from_channels",
    "compute_mscn_components",
    "gaussian_window",
    "patch_entropy",
    "to_patches",
]

