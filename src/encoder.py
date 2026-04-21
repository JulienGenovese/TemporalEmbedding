"""Encoding modules for raw transaction fields — see architecture/encoder.md."""
# to have the same typing in all python 
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Vocabulary — maps raw categorical values → contiguous indices
# ---------------------------------------------------------------------------

class Vocabulary:
    """Bijection from raw categorical values to contiguous indices (0 = padding).

    Call ``fit(values)`` on raw training data, then pass raw-valued tensors
    directly into the encoder — the mapping is applied automatically.
    """

    def __init__(self) -> None:
        self._val2idx: dict[int, int] = {}
        self._lookup: torch.Tensor | None = None

    def fit(self, values) -> Vocabulary:
        unique = sorted(set(int(v) for v in values if int(v) != 0))
        self._val2idx = {v: i + 1 for i, v in enumerate(unique)}
        if unique:
            lookup = torch.zeros(max(unique) + 1, dtype=torch.long)
            for v, idx in self._val2idx.items():
                lookup[v] = idx
        else:
            lookup = torch.zeros(1, dtype=torch.long)
        self._lookup = lookup
        return self

    @property
    def size(self) -> int:
        return len(self._val2idx) + 1

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        if self._lookup is None:
            raise RuntimeError("Vocabulary not fitted — call fit() first")
        return self._lookup.to(ids.device)[ids.long().clamp(max=len(self._lookup) - 1)]


# ---------------------------------------------------------------------------
# NumericNormalizer — clip + log1p + standardise from fitted statistics
# ---------------------------------------------------------------------------

class NumericNormalizer:
    """Learns clip bound (percentile), then applies clip → log1p → z-score.

    Fitted on non-zero values only (zeros are treated as padding and preserved).
    For signed features pass the raw (possibly negative) values — the normalizer
    takes ``abs()`` internally during both ``fit()`` and ``__call__()``.
    """

    def __init__(self, clip_pct: float = 99.0) -> None:
        self.clip_pct = clip_pct
        self._clip_hi: float = 0.0
        self._mean: float = 0.0
        self._std: float = 1.0
        self._signed: bool = False
        self._fitted: bool = False

    def fit(self, values, *, signed: bool = False) -> NumericNormalizer:
        self._signed = signed
        t = torch.as_tensor(values, dtype=torch.float32)
        if signed:
            t = t.abs()
        nonzero = t[t != 0]
        if len(nonzero) == 0:
            self._fitted = True
            return self
        self._clip_hi = float(torch.quantile(nonzero, self.clip_pct / 100.0))
        transformed = torch.log1p(nonzero.clamp(max=self._clip_hi))
        self._mean = float(transformed.mean())
        self._std = float(transformed.std().clamp(min=1e-8))
        self._fitted = True
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("NumericNormalizer not fitted — call fit() first")
        if self._signed:
            x = x.abs()
        mask = x == 0
        out = (torch.log1p(x.clamp(min=0, max=self._clip_hi)) - self._mean) / self._std
        return out.masked_fill(mask, 0.0)


# ---------------------------------------------------------------------------
# Primitive: frequency-bank encoder for continuous values
# ---------------------------------------------------------------------------

class FeatureSpecProtocol(Protocol):
    
    #feature name
    name: str 
    # number of tensors returned 
    n_slots: int
    # constructor of the class
    def build(self, d_field: int, n_frequencies: int) -> nn.Module: ... 
    # main
    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]: ...


class NumericEncoder(nn.Module):
    """Learnable sin/cos frequency bank + linear projection to d_field.

    Frequencies are initialised log-spaced in [1, 1000].  Input should be
    pre-processed (e.g. log1p + clip) or the absolute value of a signed field.
    
    - d_field: final dimension of the embedding
    - n_frequencies: number of frequencies to learn
    
    """

    def __init__(self, d_field: int = 64, n_frequencies: int = 16):
        super().__init__()
        self.frequencies = nn.Parameter(torch.logspace(0, 3, n_frequencies))
        self.projection = nn.Linear(2 * n_frequencies, d_field)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.frequencies
        return self.projection(torch.cat([angles.sin(), angles.cos()], dim=-1))


# ---------------------------------------------------------------------------
# Datetime decomposition (pure integer tensor arithmetic, GPU-safe)
# ---------------------------------------------------------------------------

def _decompose_unix_timestamp(
    ts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unix seconds → (hour[1..24], dow[1..7], dom[1..31]); 0 for padding.
    Work with pure arithmetic for GPU computation. 
    Avoid datetime su use it at runtime everywhere
    """
    mask = ts == 0
    hour = (ts // 3600) % 24 + 1
    dow  = (ts // 86400 + 3) % 7 + 1          # Jan 1 1970 was Thursday so we add 3 days to have mon=1 and sun=7

    # Day of month via Fliegel-Van Flandern Julian Day Number algorithm
    JD = ts // 86400 + 2440588
    a = JD + 32044
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


# ---------------------------------------------------------------------------
# Feature specs — each owns its own build() / encode() / n_slots
# ---------------------------------------------------------------------------

@dataclass
class NumericFeature(FeatureSpecProtocol):
    """Continuous numeric field.

    If ``signed=True`` the raw value may be negative → encoder splits it into
    abs(x) (NumericEncoder) + a sign embedding, yielding 2 field slots.

    Call ``fit(raw_values)`` to learn a ``NumericNormalizer`` (clip → log1p →
    z-score) that is applied automatically at encode time.  Without ``fit()``,
    the raw values are passed directly to the frequency bank (legacy mode).
    """
    name: str
    signed: bool = False
    normalizer: NumericNormalizer | None = field(default=None, repr=False)

    def fit(self, values, *, clip_pct: float = 99.0) -> NumericFeature:
        self.normalizer = NumericNormalizer(clip_pct).fit(values, signed=self.signed)
        return self

    @property
    def n_slots(self) -> int:
        return 2 if self.signed else 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        value_enc = NumericEncoder(d_field, n_frequencies)
        if not self.signed:
            return value_enc
        # module dict instead of dict to register the dict to trace the parameters
        return nn.ModuleDict({
            "value": value_enc,
            "sign":  nn.Embedding(3, d_field, padding_idx=0), # 0 for padding, 1 for positive and 2 for negative
        })

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = batch[self.name]
        if not self.signed:
            return [module(self.normalizer(x) if self.normalizer else x)]
        sign_ids = (x > 0).long() + (x < 0).long() * 2   # 0=zero/pad, 1=pos, 2=neg
        norm_abs = self.normalizer(x) if self.normalizer else x.abs()
        return [module["value"](norm_abs), module["sign"](sign_ids)]


@dataclass
class CategoricalFeature(FeatureSpecProtocol):
    """Integer-ID categorical field → ``nn.Embedding(vocab_size, d_field, padding_idx=0)``.

    Two usage modes:
      1. Explicit ``vocab_size`` — batch values are already contiguous indices.
      2. Call ``fit(raw_values)`` — builds a ``Vocabulary`` that maps raw IDs to
         contiguous indices automatically at encode time.
    """
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
            raise RuntimeError(f"CategoricalFeature('{self.name}'): set vocab_size or call fit() first")
        return nn.Embedding(self.vocab_size, d_field, padding_idx=0)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        ids = batch[self.name]
        if self.vocab is not None:
            ids = self.vocab(ids)
        return [module(ids)]


@dataclass
class DatetimeFeature(FeatureSpecProtocol):
    """Unix-timestamp field (int64 seconds; 0 = padding), decomposed into
    hour_of_day [1..24], day_of_week [1..7], day_of_month [1..31] — 3 slots."""
    name: str

    @property
    def n_slots(self) -> int:
        return 3

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return nn.ModuleList([
            nn.Embedding(25, d_field, padding_idx=0),   # hour_of_day
            nn.Embedding( 8, d_field, padding_idx=0),   # day_of_week
            nn.Embedding(32, d_field, padding_idx=0),   # day_of_month
        ])

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        components = _decompose_unix_timestamp(batch[self.name].long())
        return [emb(c) for emb, c in zip(module, components)]


@dataclass
class HighCardCategoricalFeature(FeatureSpecProtocol):
    """High-cardinality field via double hashing.

    Accepts raw integer IDs in ``batch[name]`` and computes two independent
    hash buckets internally (Knuth multiplicative hashing with different primes).
    Padding (ID == 0) is preserved as bucket 0.
    """
    name: str
    hash_buckets: int = 8192

    @property
    def n_slots(self) -> int:
        return 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return nn.ModuleList([
            nn.Embedding(self.hash_buckets, d_field, padding_idx=0),
            nn.Embedding(self.hash_buckets, d_field, padding_idx=0),
        ])

    @staticmethod
    def _double_hash(ids: torch.Tensor, n_buckets: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = ids == 0
        h_a = ids.long() * 2654435761 % (n_buckets - 1) + 1
        h_b = ids.long() * 2246822519 % (n_buckets - 1) + 1
        return h_a.masked_fill(mask, 0), h_b.masked_fill(mask, 0)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        emb_a, emb_b = module
        h_a, h_b = self._double_hash(batch[self.name], self.hash_buckets)
        return [emb_a(h_a) + emb_b(h_b)]


FeatureSpec = NumericFeature | CategoricalFeature | DatetimeFeature | HighCardCategoricalFeature


def categorical_vocab_sizes(features: list[FeatureSpec]) -> dict[str, int]:
    """Return ``{name: vocab_size}`` for every CategoricalFeature in the schema."""
    return {f.name: f.vocab_size for f in features if isinstance(f, CategoricalFeature)}


# ---------------------------------------------------------------------------
# Thin dispatcher
# ---------------------------------------------------------------------------

class TransactionEncoder(nn.Module):
    """Schema-driven encoder: ``batch dict → (B, T, n_fields, d_field)``.

    Iterates ``features`` in order, delegating to each spec's ``encode()``.
    """

    def __init__(
        self,
        features: list[FeatureSpec],
        d_field: int = 64,
        n_frequencies: int = 16,
    ):
        super().__init__()
        self.features = features
        self.encoders = nn.ModuleList(
            [f.build(d_field, n_frequencies) for f in features]
        )

    @property
    def n_fields(self) -> int:
        return sum(f.n_slots for f in self.features)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fields: list[torch.Tensor] = []
        for feat, enc in zip(self.features, self.encoders):
            fields.extend(feat.encode(enc, batch))
        return torch.stack(fields, dim=2)


# ---------------------------------------------------------------------------
# Usage examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 4, 16    # small sizes for readable output

    # ------------------------------------------------------------------
    # Example 1 — minimal schema, one feature of each type
    # ------------------------------------------------------------------
    # Simulate raw training data for fitting
    raw_amounts  = [120.5, -47.0, -9.9, 300.0, 9.0, -500.0, 15.3, 88.0]
    raw_balances = [1000.0, 950.0, 500.0, 800.0, 815.0, 1200.0, 340.0]
    raw_mccs     = [5411, 5812, 742, 5999, 100, 200]

    mcc_feature = CategoricalFeature("mcc").fit(raw_mccs)
    features: list[FeatureSpec] = [
        NumericFeature("amount", signed=True).fit(raw_amounts),    # 2 slots (abs + sign)
        NumericFeature("balance").fit(raw_balances),                # 1 slot
        mcc_feature,
        DatetimeFeature("timestamp"),            # 3 slots (hour, dow, dom)
        HighCardCategoricalFeature("merchant", hash_buckets=1024),
    ]
    encoder = TransactionEncoder(features, d_field=D, n_frequencies=8)
    print(f"n_fields = {encoder.n_fields}   (expected 8 = 2+1+1+3+1)")
    print(f"mcc vocab_size = {mcc_feature.vocab_size}  (auto-fitted from 6 unique values + padding)")

    # Build a batch.  Padding is position T-1 in every row:
    # numerics = 0.0, long IDs = 0, timestamps = 0.
    # Categoricals and high-card fields use raw IDs — indexing is automatic.
    batch = {
        "amount":     torch.tensor([[ 120.5, -47.0,  0.0,  0.0],
                                    [ -9.9,  300.0, 9.0, 0.0]]),
        "balance":    torch.tensor([[1000.0, 950.0, 950.0, 0.0],
                                    [ 500.0, 800.0, 815.0, 0.0]]),
        "mcc":        torch.tensor([[ 5411, 5812,  742,   0],
                                    [  742, 5411, 5999,   0]]),
        "timestamp":  torch.tensor([[1_577_836_800,   # 2020-01-01 00:00 UTC (Wed)
                                     1_609_459_200,   # 2021-01-01 00:00 UTC (Fri)
                                     1_640_995_200,   # 2022-01-01 00:00 UTC (Sat)
                                     0],
                                    [1_580_515_200,   # 2020-02-01
                                     1_583_020_800,   # 2020-03-01
                                     1_614_556_800,   # 2021-03-01
                                     0]]),
        "merchant":   torch.tensor([[ 42, 113,  7,  0],
                                    [  7,  42, 999, 0]]),
    }
    for key, value in batch.items():
        print(key, ": ", value)
    out = encoder(batch)
    print(f"output shape  = {tuple(out.shape)}   (expected ({B}, {T}, 8, {D}))")

    # ------------------------------------------------------------------
    # Example 2 — inspect the datetime decomposition in isolation
    # ------------------------------------------------------------------
    print("\nDatetime decomposition of 'timestamp':")
    hour, dow, dom = _decompose_unix_timestamp(batch["timestamp"])
    for name, t in [("hour", hour), ("dow", dow), ("dom", dom)]:
        print(f"  {name:4s} = {t.tolist()}")
    # Padding positions (ts==0) are masked back to 0 in every component.

    # ------------------------------------------------------------------
    # Example 3 — signed numeric: sign is extracted internally
    # ------------------------------------------------------------------
    x = batch["amount"]
    sign_ids = (x > 0).long() + (x < 0).long() * 2       # 0=pad/zero, 1=pos, 2=neg
    print(f"\nSign IDs for 'amount' (internal):\n  {sign_ids.tolist()}")

    # ------------------------------------------------------------------
    # Example 4 — extending the schema: just add specs, no encoder changes
    # ------------------------------------------------------------------
    extended = features + [
        CategoricalFeature("channel").fit(range(1, 10)),
        NumericFeature("fee"),
    ]
    big_encoder = TransactionEncoder(extended, d_field=D)
    print(f"\nExtended schema: n_fields = {big_encoder.n_fields}   (expected 10)")

    # ------------------------------------------------------------------
    # Example 5 — recover vocab sizes for downstream MTM heads
    # ------------------------------------------------------------------
    print(f"\nCategorical vocab sizes: {categorical_vocab_sizes(extended)}")
