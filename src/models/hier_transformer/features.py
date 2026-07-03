"""Feature specifications and preprocessing helpers for transaction encoding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib

import torch
import torch.nn as nn


class Vocabulary:
    """Bijection from raw categorical values to contiguous indices (0 = padding)."""

    def __init__(self) -> None:
        self._val2idx: dict[int, int] = {}
        self._lookup: torch.Tensor | None = None

    def fit(self, values) -> Vocabulary:
        unique = sorted(set(int(v) for v in values if int(v) != 0))
        self._val2idx = {value: i + 1 for i, value in enumerate(unique)}
        if unique:
            lookup = torch.zeros(max(unique) + 1, dtype=torch.long)
            for value, idx in self._val2idx.items():
                lookup[value] = idx
        else:
            lookup = torch.zeros(1, dtype=torch.long)
        self._lookup = lookup
        return self

    @property
    def size(self) -> int:
        return len(self._val2idx) + 1

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if self._lookup is None:
            raise RuntimeError("Vocabulary not fitted; call fit() first")
        return self._lookup.to(ids.device)[ids.long().clamp(max=len(self._lookup) - 1)]


class NumericNormalizer:
    """Clip + log1p + z-score normalization fitted on non-zero magnitudes."""

    def __init__(self, clip_pct: float = 99.0) -> None:
        self.clip_pct = clip_pct
        self._clip_hi: float = 0.0
        self._mean: float = 0.0
        self._std: float = 1.0
        self._fitted: bool = False

    def fit(self, values) -> NumericNormalizer:
        values_t = torch.as_tensor(values, dtype=torch.float32).abs()
        nonzero = values_t[values_t != 0]
        if len(nonzero) == 0:
            self._fitted = True
            return self

        self._clip_hi = float(torch.quantile(nonzero, self.clip_pct / 100.0))
        transformed = torch.log1p(nonzero.clamp(max=self._clip_hi))
        self._mean = float(transformed.mean())
        self._std = float(transformed.std(unbiased=False).clamp(min=1e-8))
        self._fitted = True
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("NumericNormalizer not fitted; call fit() first")
        x = x.abs()
        mask = x == 0
        out = (torch.log1p(x.clamp(max=self._clip_hi)) - self._mean) / self._std
        return out.masked_fill(mask, 0.0)

    @property
    def fitted(self) -> bool:
        return self._fitted

    def state_dict(self) -> dict:
        return {
            "clip_pct": self.clip_pct,
            "clip_hi": self._clip_hi,
            "mean": self._mean,
            "std": self._std,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> "NumericNormalizer":
        self.clip_pct = state["clip_pct"]
        self._clip_hi = state["clip_hi"]
        self._mean = state["mean"]
        self._std = state["std"]
        self._fitted = state["fitted"]
        return self


class FeatureSpecBase(ABC):
    """Abstract base class for feature specifications."""

    name: str

    @property
    @abstractmethod
    def n_slots(self) -> int:
        """Number of field embeddings produced by this feature."""

    @abstractmethod
    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        """Build the PyTorch module used to encode this feature."""

    @abstractmethod
    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Encode this feature from a batch dictionary."""


class NumericEncoder(nn.Module):
    """Learnable sin/cos frequency bank + linear projection to d_field."""

    def __init__(self, d_field: int = 64, n_frequencies: int = 16):
        super().__init__()
        self.frequencies = nn.Parameter(torch.logspace(0, 3, n_frequencies))
        self.projection = nn.Linear(2 * n_frequencies, d_field)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.frequencies
        return self.projection(torch.cat([angles.sin(), angles.cos()], dim=-1))


def _decompose_unix_timestamp(
    ts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unix seconds -> (hour[1..24], dow[1..7], dom[1..31]); 0 for padding."""
    mask = ts == 0
    hour = (ts // 3600) % 24 + 1
    dow = (ts // 86400 + 3) % 7 + 1

    julian_day = ts // 86400 + 2440588
    a = julian_day + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153
    dom = e - (153 * m + 2) // 5 + 1

    return (
        hour.masked_fill(mask, 0),
        dow.masked_fill(mask, 0),
        dom.masked_fill(mask, 0),
    )


@dataclass
class NumericFeature(FeatureSpecBase):
    """Continuous numeric field, optionally split into value + sign slots."""

    name: str
    signed: bool = False
    normalizer: NumericNormalizer | None = field(default=None, repr=False)

    def fit(self, values, *, clip_pct: float = 99.0) -> NumericFeature:
        self.normalizer = NumericNormalizer(clip_pct).fit(values)
        return self

    @property
    def n_slots(self) -> int:
        return 2 if self.signed else 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        value_encoder = NumericEncoder(d_field, n_frequencies)
        if not self.signed:
            return value_encoder
        return nn.ModuleDict({
            "value": value_encoder,
            "sign": nn.Embedding(3, d_field, padding_idx=0),
        })

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = batch[self.name]
        if not self.signed:
            return [module(self.normalizer(x) if self.normalizer else x)]
        sign_ids = (x > 0).long() + (x < 0).long() * 2
        norm_abs = self.normalizer(x) if self.normalizer else x.abs()
        return [module["value"](norm_abs), module["sign"](sign_ids)]


@dataclass
class CategoricalFeature(FeatureSpecBase):
    """Integer-ID categorical field backed by an embedding table."""

    name: str
    vocab_size: int | None = None
    vocab: Vocabulary | None = field(default=None, repr=False)

    def fit(self, values) -> CategoricalFeature:
        self.vocab = Vocabulary().fit(values)
        self.vocab_size = self.vocab.size
        return self

    @property
    def n_slots(self) -> int:
        return 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        if self.vocab_size is None:
            raise RuntimeError(
                f"CategoricalFeature('{self.name}'): set vocab_size or call fit() first",
            )
        return nn.Embedding(self.vocab_size, d_field, padding_idx=0)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        ids = batch[self.name]
        if self.vocab is not None:
            ids = self.vocab(ids)
        return [module(ids)]


@dataclass
class DatetimeFeature(FeatureSpecBase):
    """Unix timestamp decomposed into hour, day-of-week and day-of-month slots."""

    name: str

    @property
    def n_slots(self) -> int:
        return 3

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return nn.ModuleList([
            nn.Embedding(25, d_field, padding_idx=0),
            nn.Embedding(8, d_field, padding_idx=0),
            nn.Embedding(32, d_field, padding_idx=0),
        ])

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        components = _decompose_unix_timestamp(batch[self.name].long())
        return [emb(component) for emb, component in zip(module, components)]


@dataclass
class HighCardCategoricalFeature(FeatureSpecBase):
    """High-cardinality field via stable ids and two hash embedding tables."""

    name: str
    hash_buckets: int = 5003

    @property
    def n_slots(self) -> int:
        return 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return nn.ModuleList([
            nn.Embedding(self.hash_buckets, d_field, padding_idx=0),
            nn.Embedding(self.hash_buckets, d_field, padding_idx=0),
        ])

    @staticmethod
    def _stable_string_id(s: str) -> int:
        digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF or 1

    @staticmethod
    def to_stable_id(value) -> int:
        if value is None or value == "" or value == 0:
            return 0
        if isinstance(value, str):
            return HighCardCategoricalFeature._stable_string_id(value)
        return int(value)

    @classmethod
    def prepare(cls, values) -> torch.Tensor:
        return torch.tensor(
            [[cls.to_stable_id(value) for value in row] for row in values],
            dtype=torch.long,
        )

    @staticmethod
    def to_hash_buckets(ids: torch.Tensor, n_buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = ids == 0
        hash_a = ids.long() * 2654435761 % (n_buckets - 1) + 1
        hash_b = ids.long() * 2246822519 % (n_buckets - 1) + 1
        return hash_a.masked_fill(mask, 0), hash_b.masked_fill(mask, 0)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        emb_a, emb_b = module
        hash_a, hash_b = self.to_hash_buckets(batch[self.name], self.hash_buckets)
        return [emb_a(hash_a) + emb_b(hash_b)]


FeatureSpec = NumericFeature | CategoricalFeature | DatetimeFeature | HighCardCategoricalFeature

DEFAULT_FEATURES: list[FeatureSpec] = [
    NumericFeature("importo", signed=True),
    HighCardCategoricalFeature("merchant"),
    CategoricalFeature("cocau", 501),
]


def categorical_vocab_sizes(features: list[FeatureSpec]) -> dict[str, int]:
    """Return ``{name: vocab_size}`` for every CategoricalFeature in the schema."""
    return {
        feature.name: feature.vocab_size
        for feature in features
        if isinstance(feature, CategoricalFeature)
    }


def numeric_field_names(features: list[FeatureSpec]) -> list[str]:
    """Return the names of every NumericFeature in the schema."""
    return [feature.name for feature in features if isinstance(feature, NumericFeature)]
