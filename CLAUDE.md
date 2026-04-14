# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses **uv** (requires Python 3.14) and has no test framework configured.

```bash
# Install / sync dependencies
uv sync

# Run the model's built-in smoke test (forward pass + shape checks)
uv run python -m src.model

# Run arbitrary scripts
uv run python -m src.<module>
```

Scripts inside `src/` use relative imports (`from .encoder import ...`), so always invoke them via `python -m src.<name>` rather than `python src/<name>.py`.

## Architecture

A hierarchical Transformer for banking-transaction client embeddings. The pipeline has three stacked stages plus optional pre-training heads:

```
batch (dict of (B,T) tensors)
        │
        ▼
TransactionEncoder      src/encoder.py          → (B, T, n_fields, d_field)
        │   schema-driven: per-field sub-encoders
        ▼
FieldTransformer        src/field_encoder.py    → (B, T, d_model)
        │   intra-transaction attention + AttentionPooling across fields
        ▼
SequenceTransformer     src/sequence_encoder.py → (B, d_model)
        │   prepends [CLS], TimeAwarePositionalEncoding driven by delta_t,
        │   gradient-checkpointed Transformer layers, returns h_CLS
        ▼
       h_cls   ── optional ──▶ MTMHead + ContrastiveHead (src/loss.py)
```

Defaults: `d_field=64`, `d_model=128`, field layers=2×4heads, sequence layers=4×8heads, ~2.5M params.

### The `TransactionEncoder` is schema-driven (key design)

Never hard-code field names or vocab sizes inside the encoder. Each feature spec in `src/encoder.py` is a dataclass that **owns its own** `build()` (constructs sub-modules), `encode()` (produces field tensors), and `n_slots` (count). `TransactionEncoder` itself is a thin dispatcher that iterates the schema:

```python
for feat, enc in zip(self.features, self.encoders):
    fields.extend(feat.encode(enc, batch))
```

| Spec | Batch key(s) | Field slots | Sub-module(s) |
|------|--------------|-------------|---------------|
| `NumericFeature(name, signed=False)` | `name` (float) | 1 | `NumericEncoder` (learnable sin/cos frequency bank) |
| `NumericFeature(name, signed=True)`  | `name` (float) | 2 | `NumericEncoder(abs)` + `nn.Embedding(3, d_field)` sign (0=pad/zero, 1=pos, 2=neg) |
| `CategoricalFeature(name, vocab_size)` | `name` (long) | 1 | `nn.Embedding(vocab_size, d_field, padding_idx=0)` |
| `DatetimeFeature(name)` | `name` (int64 Unix ts, 0 = pad) | 3 | three `nn.Embedding`s — encoder decomposes via `_decompose_unix_timestamp` into hour[1..24] / dow[1..7] / dom[1..31] using the Fliegel-Van Flandern Julian Day algorithm (all integer tensor arithmetic, GPU-safe) |
| `DoubleHashFeature(name, hash_buckets=8192)` | `name_a`, `name_b` (long) | 1 | two independent `nn.Embedding`s, summed |

**Adding a new feature type** = one new dataclass with `build` / `encode` / `n_slots`, zero changes elsewhere.

`TransactionEncoder.n_fields` is computed from the schema and is passed into `FieldTransformer` (which sizes its learnable field-type positional encoding accordingly). The default schema lives in `src/model.py` as `DEFAULT_FEATURES` and produces 13 slots to match the original design.

### Consequence: loss heads track the schema

`MTMHead` (`src/loss.py`) takes a `vocab_sizes` dict. `TransactionTransformer` derives it via `categorical_vocab_sizes(features)` (in `src/encoder.py`) so MTM heads automatically cover whatever categorical features are in the schema. `mtm_loss` iterates the `cat_*` keys actually present in `preds` rather than a hard-coded list — adding/removing a `CategoricalFeature` requires no loss-side changes.

### Padding convention

Padding is represented by **index 0** for categoricals/hashes and **value 0** for numerics/timestamps. All `nn.Embedding` instances use `padding_idx=0` so their gradients are zero on pad positions. A separate boolean `padding_mask` in the batch dict (True = padded) is consumed by `FieldTransformer`/`SequenceTransformer` for attention masking.

### Time flows end-to-end

`delta_t` (seconds between consecutive transactions) is both a regular numeric input field AND used directly by `SequenceTransformer.TimeAwarePositionalEncoding` — a fixed sinusoidal PE driven by real time gaps instead of integer positions. The [CLS] token is given `delta_t=0`.

## Notes on modifying the schema

Adding a new feature: append a spec to `DEFAULT_FEATURES` in `src/model.py`, provide the expected batch key(s) at forward time. No other edits needed — `TransactionEncoder`, `FieldTransformer`, and `MTMHead` all adapt automatically.
