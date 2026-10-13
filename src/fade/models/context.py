"""Execution context shared by all feature modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ..ops.ce import compute_ce_from_channels
from ..ops.mscn import compute_mscn_components
from ..ops.patches import to_patches


@dataclass(slots=True)
class FeatureContext:
    """Shared precomputed tensors and helpers for feature computation."""

    image: np.ndarray
    patch_size: int = 8
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    @property
    def image_rgb(self) -> np.ndarray:
        if "image_rgb" not in self._cache:
            row, col, _ = self.image.shape
            pr = row // self.patch_size
            pc = col // self.patch_size
            self._cache["image_rgb"] = self.image[: pr * self.patch_size, : pc * self.patch_size, :3]
        return self._cache["image_rgb"]

    @property
    def patch_row_num(self) -> int:
        return self.image_rgb.shape[0] // self.patch_size

    @property
    def patch_col_num(self) -> int:
        return self.image_rgb.shape[1] // self.patch_size

    @property
    def R(self) -> np.ndarray:
        if "R" not in self._cache:
            self._cache["R"] = self.image_rgb[:, :, 0].astype(np.float64)
        return self._cache["R"]

    @property
    def G(self) -> np.ndarray:
        if "G" not in self._cache:
            self._cache["G"] = self.image_rgb[:, :, 1].astype(np.float64)
        return self._cache["G"]

    @property
    def B(self) -> np.ndarray:
        if "B" not in self._cache:
            self._cache["B"] = self.image_rgb[:, :, 2].astype(np.float64)
        return self._cache["B"]

    @property
    def Ig_int(self) -> np.ndarray:
        if "Ig_int" not in self._cache:
            self._cache["Ig_int"] = np.round(0.2989 * self.R + 0.5870 * self.G + 0.1140 * self.B).astype(np.int32)
        return self._cache["Ig_int"]

    @property
    def Id(self) -> np.ndarray:
        if "Id" not in self._cache:
            self._cache["Id"] = np.minimum(np.minimum(self.R, self.G), self.B) / 255.0
        return self._cache["Id"]

    @property
    def Is(self) -> np.ndarray:
        if "Is" not in self._cache:
            # Keep the original float32 normalization and rounding, but compute
            # only saturation: hue and value are not used by any FADE feature.
            rgb = self.image_rgb.astype(np.float32) / 255.0
            maximum = rgb.max(axis=2)
            minimum = rgb.min(axis=2)
            if np.any(maximum > 1) or np.any(minimum < 0):
                raise ValueError("Input array must be in the range [0, 255].")
            self._cache["Is"] = np.divide(
                maximum - minimum,
                maximum,
                out=np.zeros_like(maximum),
                where=maximum > 0,
            )
        return self._cache["Is"]

    @property
    def rg(self) -> np.ndarray:
        if "rg" not in self._cache:
            self._cache["rg"] = self.R - self.G
        return self._cache["rg"]

    @property
    def by(self) -> np.ndarray:
        if "by" not in self._cache:
            self._cache["by"] = 0.5 * (self.R + self.G) - self.B
        return self._cache["by"]

    @property
    def mscn(self) -> np.ndarray:
        if "mscn" not in self._cache:
            mscn, sigma_map, cv = compute_mscn_components(self.Ig_int.astype(np.float64))
            self._cache["mscn"] = mscn
            self._cache["sigma_map"] = sigma_map
            self._cache["cv"] = cv
        return self._cache["mscn"]

    @property
    def sigma_map(self) -> np.ndarray:
        _ = self.mscn
        return self._cache["sigma_map"]

    @property
    def cv(self) -> np.ndarray:
        _ = self.mscn
        return self._cache["cv"]

    @property
    def mscn_v(self) -> np.ndarray:
        if "mscn_v" not in self._cache:
            self._cache["mscn_v"] = self.mscn * np.roll(self.mscn, 1, axis=0)
        return self._cache["mscn_v"]

    @property
    def mscn_v_patches(self) -> np.ndarray:
        if "mscn_v_patches" not in self._cache:
            self._cache["mscn_v_patches"] = self.to_patches(self.mscn_v)
        return self._cache["mscn_v_patches"]

    @property
    def CE_gray(self) -> np.ndarray:
        if "CE_gray" not in self._cache:
            # CE uses unrounded gray with 0.299, unlike the MSCN/entropy gray.
            gray = 0.299 * self.R + 0.587 * self.G + 0.114 * self.B
            ce_gray, ce_by, ce_rg = compute_ce_from_channels(gray, self.by, self.rg)
            self._cache["CE_gray"] = ce_gray
            self._cache["CE_by"] = ce_by
            self._cache["CE_rg"] = ce_rg
        return self._cache["CE_gray"]

    @property
    def CE_by(self) -> np.ndarray:
        _ = self.CE_gray
        return self._cache["CE_by"]

    @property
    def CE_rg(self) -> np.ndarray:
        _ = self.CE_gray
        return self._cache["CE_rg"]

    def to_patches(self, arr: np.ndarray) -> np.ndarray:
        """Split a 2-D map into flattened non-overlapping patches."""
        return to_patches(arr, self.patch_row_num, self.patch_col_num, self.patch_size)
