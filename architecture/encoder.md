# TransactionEncoder (`src/encoder.py`)

## Overview

Schema-driven encoder that converts raw transaction fields into embeddings.

**Output shape:** `(B, T, n_fields, d_field)`

## Padding convention

We usually don't have the same number of transactions for each customer. For this reason we pad:

```
Customer A: [tx1, tx2, tx3, tx4, tx5]       ← 5 real transactions
Customer B: [tx1, tx2, tx3,  0,   0]        ← 3 real + 2 padding
Customer C: [tx1, tx2, tx3, tx4, tx5]       ← 5 real transactions
```

Padding is represented by **index 0** for categoricals/hashes and **value 0** for numerics/timestamps. All `nn.Embedding` instances use `padding_idx=0` so their gradients are zero on pad positions.

## Design

Each `FeatureSpec` dataclass owns its own `build()` / `encode()` / `n_slots`. `TransactionEncoder` is a thin dispatcher that iterates the schema:

```python
for feat, enc in zip(self.features, self.encoders):
    fields.extend(feat.encode(enc, batch))
```

## Feature specs

| Spec | Batch key(s) | Field slots | Sub-module(s) |
|------|--------------|-------------|---------------|
| `NumericFeature(name, signed=False)` | `name` (float) | 1 | `NumericEncoder` (learnable sin/cos frequency bank) |
| `NumericFeature(name, signed=True)` | `name` (float) | 2 | `NumericEncoder(abs)` + `nn.Embedding(3, d_field)` sign (0=pad/zero, 1=pos, 2=neg) |
| `CategoricalFeature(name, vocab_size)` | `name` (long) | 1 | `nn.Embedding(vocab_size, d_field, padding_idx=0)` |
| `DatetimeFeature(name)` | `name` (int64 Unix ts, 0 = pad) | 3 | Three `nn.Embedding`s — decomposes via `_decompose_unix_timestamp` into hour[1..24] / dow[1..7] / dom[1..31] using the Fliegel-Van Flandern Julian Day algorithm (all integer tensor arithmetic, GPU-safe) |
| `HighCardCategoricalFeature(name, hash_buckets=8192)` | `name_a`, `name_b` (long) | 1 | Two independent `nn.Embedding`s, summed |

## NumericEncoder

Learnable sin/cos frequency bank + linear projection to `d_field`. Frequencies are initialised log-spaced in [1, 1000].

```
x (B, T) → unsqueeze → (B, T, 1) * frequencies (n_freq,) → (B, T, n_freq)
         → [sin, cos] → (B, T, 2*n_freq)
         → Linear(2*n_freq, d_field) → (B, T, d_field)
```

## Extensibility

Adding a new feature type = one new dataclass with `build` / `encode` / `n_slots`, zero changes elsewhere. `TransactionEncoder.n_fields` is computed from the schema and passed into `FieldTransformer`.
