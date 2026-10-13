"""Tests for features/registry.py — feature registration and ordering."""

from __future__ import annotations


from fade.features.registry import (
    canonical_feature_registry,
    default_feature_registry,
)


class TestCanonicalFeatureRegistry:

    def test_returns_exactly_12_features(self):
        features = canonical_feature_registry()
        assert len(features) == 12

    def test_first_feature_is_f01(self):
        features = canonical_feature_registry()
        assert features[0].name == "f01"

    def test_last_feature_is_f12(self):
        features = canonical_feature_registry()
        assert features[-1].name == "f12"

    def test_feature_names_in_order(self):
        features = canonical_feature_registry()
        expected = [f"f{i:02d}" for i in range(1, 13)]
        actual = [f.name for f in features]
        assert actual == expected

    def test_each_feature_has_name_attribute(self):
        for f in canonical_feature_registry():
            assert hasattr(f, "name")
            assert isinstance(f.name, str)

    def test_each_feature_has_compute_method(self):
        for f in canonical_feature_registry():
            assert callable(getattr(f, "compute", None))

    def test_names_unique(self):
        features = canonical_feature_registry()
        names = [f.name for f in features]
        assert len(names) == len(set(names))


class TestDefaultRegistry:

    def test_default_returns_12(self):
        assert len(default_feature_registry()) == 12

    def test_default_same_names_as_canonical(self):
        canonical_names = [f.name for f in canonical_feature_registry()]
        default_names = [f.name for f in default_feature_registry()]
        assert canonical_names == default_names
