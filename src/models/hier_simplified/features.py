"""Simplified feature specs: linear numeric encoding, single-table hashing.

Ablation counterparts of the rich encoders in
``hier_transformer.features``. The rest of the schema (``cocau``,
``timestamp``) is shared unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.hier_transformer.features import (
    CategoricalFeature,
    DatetimeFeature,
    FeatureSpec,
    HighCardCategoricalFeature,
    NumericFeature,
    mtm_mask_key,
)


class LinearNumericEncoder(nn.Module):
    """``Linear(2, d_field)`` over (normalized magnitude, sign).

    Replaces the frequency bank + separate sign-embedding slot: all
    embeddings live on a plane in R^d_field, any non-linearity is left to
    the layers downstream. Masked positions get the same learned [MASK]
    treatment as :class:`~src.models.hier_transformer.features.NumericEncoder`.
    """

    def __init__(self, d_field: int = 64):
        super().__init__()
        self.projection = nn.Linear(2, d_field)
        self.mask_token = nn.Parameter(torch.randn(d_field) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        sign: torch.Tensor,
        mtm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.projection(torch.stack([x, sign], dim=-1))
        if mtm_mask is not None:
            out = torch.where(mtm_mask.unsqueeze(-1), self.mask_token, out)
        return out


@dataclass
class LinearNumericFeature(NumericFeature):
    """Single-slot numeric field: sign folded into the value encoder.

    ``signed`` is ignored — the sign always enters as the second input
    channel of :class:`LinearNumericEncoder` instead of taking up a
    field slot of its own.
    """

    @property
    def n_slots(self) -> int:
        return 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return LinearNumericEncoder(d_field)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = batch[self.name]
        mtm_mask = batch.get(mtm_mask_key(self.name))
        norm = self.normalizer(x) if self.normalizer else x.abs()
        return [module(norm, torch.sign(x), mtm_mask)]


@dataclass
class SingleHashCategoricalFeature(HighCardCategoricalFeature):
    """High-cardinality field with one hash table instead of two.

    Collisions are no longer disambiguated by the second table: two
    merchants landing in the same bucket share their embedding outright.
    """

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return nn.Embedding(self.hash_buckets, d_field, padding_idx=0)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        hash_a, _ = self.to_hash_buckets(batch[self.name], self.hash_buckets)
        return [module(hash_a)]


SIMPLIFIED_FEATURES: list[FeatureSpec] = [
    LinearNumericFeature("importo"),
    SingleHashCategoricalFeature("merchant"),
    CategoricalFeature("cocau", 501),
    DatetimeFeature("timestamp"),
]  # total = 7 field slots (importo folds its sign slot away)
