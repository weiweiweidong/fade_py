"""Feature registries."""

from __future__ import annotations

from .base import Feature
from . import f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11, f12


def canonical_feature_registry() -> list[Feature]:
    """Return all features in canonical FADE order."""
    return [
        f01.feature,
        f02.feature,
        f03.feature,
        f04.feature,
        f05.feature,
        f06.feature,
        f07.feature,
        f08.feature,
        f09.feature,
        f10.feature,
        f11.feature,
        f12.feature,
    ]


def default_feature_registry() -> list[Feature]:
    """Return default production feature set."""
    return canonical_feature_registry()
