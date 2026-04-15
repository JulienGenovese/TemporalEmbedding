"""
Encoding modules for raw transaction fields.

Each FeatureSpec dataclass owns its own ``build()`` / ``encode()`` / ``n_slots``;
``TransactionEncoder`` is a thin dispatcher that iterates the schema.

Output shape: (B, T, n_fields, d_field)

We usually don't have the same number of transaction for each customer. For this reason we pad it.


Customer A: [tx1, tx2, tx3, tx4, tx5]       ← 5 transazioni reali
Customer B: [tx1, tx2, tx3,  0,   0]        ← 3 reali + 2 padding
Customer C: [tx1, tx2, tx3, tx4, tx5]       ← 5 real transaction

"""
# to have the same typing in all python 
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Primitive: frequency-bank encoder for continuous values
# ---------------------------------------------------------------------------

class FeatureSpecProtocol(Protocol):
    # Una specifica per spiegare come operare
    name: str
    n_slots: int
    def build(self, d_field: int, n_frequencies: int) -> nn.Module: ...
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
        self.frequencies = nn.Parameter(torch.logspace(0, 3, n_frequencies)) # frequenze learnable ma parto da guessing, e do dimensione
        self.projection = nn.Linear(2 * n_frequencies, d_field) # matrice learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x.unsqueeze(-1) * self.frequencies # aggiunge una dimensione alla fine e moltiplica per le frequenze, dim(X) = (B, Tx, 1) * (dim(f), ) = (B, Tx, dim(f))
        # se angles ha shape (batch, seq, n_freq) -> torch.cat([..., ...], dim=-1) → (batch, seq, 2*n_freq)
        # linear applica y = xA^T + b
        # (batch, seq, 2*n_freq)*(2*n_freq, d_field) = (batch, seq, d_field)
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
    """
    name: str
    signed: bool = False

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
            return [module(x)]
        sign_ids = (x > 0).long() + (x < 0).long() * 2   # 0=zero/pad, 1=pos, 2=neg
        return [module["value"](x.abs()), module["sign"](sign_ids)]


@dataclass
class CategoricalFeature:
    """Integer-ID categorical field → ``nn.Embedding(vocab_size, d_field, padding_idx=0)``."""
    name: str
    vocab_size: int

    @property
    def n_slots(self) -> int:
        return 1

    def build(self, d_field: int, n_frequencies: int) -> nn.Module:
        return nn.Embedding(self.vocab_size, d_field, padding_idx=0)

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [module(batch[self.name])]


@dataclass
class DatetimeFeature:
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
class DoubleHashFeature:
    """High-cardinality field via double hashing.  Expects batch keys
    ``f"{name}_a"`` and ``f"{name}_b"`` (pre-computed bucket IDs)."""
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

    def encode(self, module: nn.Module, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        emb_a, emb_b = module
        return [emb_a(batch[f"{self.name}_a"]) + emb_b(batch[f"{self.name}_b"])]


FeatureSpec = NumericFeature | CategoricalFeature | DatetimeFeature | DoubleHashFeature


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
            # feat.encode lavora sulla colonna di interesse. 
            # fields e' una lista di n_fields tensori di dimensione (B, seq, d_fields)
            fields.extend(feat.encode(enc, batch))
        # impiliamo la lista sulla dimensione 2, quindi abbiamo (B, seq, n_fields,d_fields)
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
    features: list[FeatureSpec] = [
        NumericFeature("amount", signed=True),   # 2 slots (abs + sign)
        NumericFeature("balance"),               # 1 slot
        CategoricalFeature("mcc", vocab_size=10_000),
        DatetimeFeature("timestamp"),            # 3 slots (hour, dow, dom)
        DoubleHashFeature("merchant", hash_buckets=1024),
    ]
    encoder = TransactionEncoder(features, d_field=D, n_frequencies=8)
    print(f"n_fields = {encoder.n_fields}   (expected 8 = 2+1+1+3+1)")

    # Build a batch.  Padding is position T-1 in every row:
    # numerics = 0.0, long IDs = 0, timestamps = 0.
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
        "merchant_a": torch.tensor([[ 42, 113,  7,  0],
                                    [  7,  42, 999, 0]]),
        "merchant_b": torch.tensor([[901, 234, 55,  0],
                                    [ 55, 901, 300, 0]]),
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
        CategoricalFeature("channel", vocab_size=10),
        NumericFeature("fee"),
    ]
    big_encoder = TransactionEncoder(extended, d_field=D)
    print(f"\nExtended schema: n_fields = {big_encoder.n_fields}   (expected 10)")

    # ------------------------------------------------------------------
    # Example 5 — recover vocab sizes for downstream MTM heads
    # ------------------------------------------------------------------
    print(f"\nCategorical vocab sizes: {categorical_vocab_sizes(extended)}")
