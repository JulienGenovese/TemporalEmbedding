"""Transaction encoder dispatcher and feature-schema fitting helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.hier_transformer.features import (
    CategoricalFeature,
    DEFAULT_FEATURES,
    DatetimeFeature,
    FeatureSpec,
    FeatureSpecBase,
    HighCardCategoricalFeature,
    NumericEncoder,
    NumericFeature,
    NumericNormalizer,
    Vocabulary,
    categorical_vocab_sizes,
    numeric_field_names,
)


class TransactionEncoder(nn.Module):
    """Schema-driven encoder: ``batch dict -> (B, T, n_fields, d_field)``."""

    def __init__(
        self,
        features: list[FeatureSpec],
        d_field: int = 64,
        n_frequencies: int = 16,
    ):
        super().__init__()
        self.features = features
        self.encoders = nn.ModuleList(
            [feature.build(d_field, n_frequencies) for feature in features],
        )

    @property
    def n_fields(self) -> int:
        return sum(feature.n_slots for feature in self.features)

    @staticmethod
    def fit_features(features: list[FeatureSpec], data) -> list[FeatureSpec]:
        """Fit schema-owned preprocessing from the training dataframe."""
        columns = getattr(data, "columns", ())
        for feature in features:
            if feature.name not in columns:
                continue
            values = data[feature.name]
            if hasattr(values, "to_numpy"):
                values = values.to_numpy(copy=True)
            if isinstance(feature, NumericFeature):
                feature.fit(values)
            elif isinstance(feature, CategoricalFeature) and feature.vocab_size is None:
                feature.fit(values)
        return features

    @staticmethod
    def prepare_feature_column(feature: FeatureSpec, values) -> np.ndarray:
        """Convert one raw dataframe column to the array expected by ``encode``."""
        if hasattr(values, "to_numpy"):
            values = values.to_numpy(copy=True)

        if isinstance(feature, NumericFeature):
            return np.asarray(values, dtype=np.float32)
        if isinstance(feature, HighCardCategoricalFeature):
            return np.fromiter(
                (HighCardCategoricalFeature.to_stable_id(value) for value in values),
                dtype=np.int64,
                count=len(values),
            )
        return np.asarray(values, dtype=np.int64)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a batch of transactions into per-field embeddings.

        Args:
            batch: dict mapping each feature name to its (B, T) tensor
                (float for numerics, long for categoricals / timestamps).
                May also carry ``<name>__mtm_mask`` boolean keys, consumed
                by the feature's ``encode`` to inject [MASK] embeddings.
        Returns:
            (B, T, n_fields, d_field) — one embedding per field slot, in
            schema order; a feature contributes ``n_slots`` consecutive
            slots (e.g. signed numeric → value + sign, datetime → hour /
            dow / dom / month).
        """
        fields: list[torch.Tensor] = []
        for feature, encoder in zip(self.features, self.encoders):
            # each encode() returns n_slots tensors of shape (B, T, d_field)
            fields.extend(feature.encode(encoder, batch))
        return torch.stack(fields, dim=2)


__all__ = [
    "CategoricalFeature",
    "DEFAULT_FEATURES",
    "DatetimeFeature",
    "FeatureSpec",
    "FeatureSpecBase",
    "HighCardCategoricalFeature",
    "NumericEncoder",
    "NumericFeature",
    "NumericNormalizer",
    "TransactionEncoder",
    "Vocabulary",
    "categorical_vocab_sizes",
    "numeric_field_names",
]
