"""Pipeline orchestration for FADE."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .features.base import Feature
from .features.registry import default_feature_registry
from .models.context import FeatureContext
from .models.reference_stats import load_reference_stats
from .ops.distance import batch_mahalanobis


@dataclass(slots=True)
class FadePipeline:
    """
    FADE pipeline orchestrator.

    Notes
    -----
    Runs full 12-feature FADE using modularized feature extraction.
    """

    features: Sequence[Feature]

    @classmethod
    def default(cls) -> "FadePipeline":
        return cls(features=default_feature_registry())

    def build_context(self, image: np.ndarray) -> FeatureContext:
        return FeatureContext(image=image)

    def compute_feature_maps(self, context: FeatureContext) -> list[np.ndarray]:
        """Run every registered feature module and return 2-D patch maps."""
        return [feature.compute(context) for feature in self.features]

    def build_feature_matrix(self, feature_maps: Sequence[np.ndarray]) -> np.ndarray:
        """
        Build the per-patch feature matrix from feature maps.

        Returns
        -------
        np.ndarray
            Shape (N_patches, N_features), transformed by log1p.
        """
        if not feature_maps:
            return np.empty((0, 0), dtype=np.float64)
        flattened = [feature_map.ravel() for feature_map in feature_maps]
        feat = np.column_stack(flattened).astype(np.float64, copy=False)
        return np.log1p(feat, out=feat)

    def run(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Execute FADE end-to-end.

        Compute full FADE from modularized feature extraction and distance
        inference against full 12-D reference statistics.
        """
        context = self.build_context(image)
        feature_maps = self.compute_feature_maps(context)
        feat = self.build_feature_matrix(feature_maps)
        if feat.size == 0:
            raise ValueError("No features are registered in the pipeline.")
        if feat.shape[1] != 12:
            raise ValueError(f"FADE expects 12 features, got {feat.shape[1]}.")

        patch_var = np.nanvar(feat, axis=1, ddof=1)

        fogfree, foggy = load_reference_stats()
        mu_free = np.ravel(fogfree["mu_fogfreeparam"]).astype(np.float64, copy=False)
        cov_free = fogfree["cov_fogfreeparam"].astype(np.float64, copy=False)
        mu_foggy = np.ravel(foggy["mu_foggyparam"]).astype(np.float64, copy=False)
        cov_foggy = foggy["cov_foggyparam"].astype(np.float64, copy=False)

        mu_diff_free = mu_free[np.newaxis, :] - feat
        dist_free = batch_mahalanobis(mu_diff_free, patch_var, cov_free)
        Df_map = dist_free.reshape(context.patch_row_num, context.patch_col_num)

        mu_diff_foggy = mu_foggy[np.newaxis, :] - feat
        dist_foggy = batch_mahalanobis(mu_diff_foggy, patch_var, cov_foggy)
        Dff_map = dist_foggy.reshape(context.patch_row_num, context.patch_col_num)

        # nanmean may receive an all-NaN slice for degenerate inputs (e.g. uniform
        # images where every patch produces NaN features). Suppress the resulting
        # "Mean of empty slice" RuntimeWarning — NaN is the correct result in that case.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Df = np.nanmean(dist_free)
            Dff = np.nanmean(dist_foggy)

        score = Df / (Dff + 1.0)
        D_map = Df_map / (Dff_map + 1.0)
        return float(score), D_map
