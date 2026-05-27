# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses **uv** (requires Python 3.14) and has no test framework configured.

```bash
# Install / sync dependencies
uv sync

# Forward-pass smoke test (shape checks only, synthetic in-memory batch)
uv run python -m src.model

# Generate the synthetic CSV dataset (~400k rows / 4000 clients) under data/
uv run python -m src.make_dataset

# Full end-to-end pre-training (MTM + InfoNCE) on the CSV — CPU-friendly defaults
uv run python -m src.train

# Plot the training curves saved by src.train (reads checkpoints/history.json)
uv run python -m src.plots

# Run arbitrary scripts
uv run python -m src.<module>
```

Scripts inside `src/` use relative imports (`from .encoder import ...`), so always invoke them via `python -m src.<name>` rather than `python src/<name>.py`.

`src.train` runs with the default settings (30 epochs × ~400 batches/epoch × 16 samples, with early stopping) and writes `checkpoints/model_final.pt`.

## Architecture

A hierarchical Transformer for banking-transaction client embeddings. The pipeline has three stacked stages plus optional pre-training heads:

```
batch (dict of (B,T) tensors)
        │
        ▼
TransactionEncoder      src/encoder.py          → (B, T, n_fields, d_field)
        │   schema-driven: per-field sub-encoders
        ▼
FieldTransformer        src/field_transformer.py → (B, T, d_model)
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
| `HighCardCategoricalFeature(name, hash_buckets=5003)` | `name` (long, raw int IDs — strings can be pre-converted via `HighCardCategoricalFeature.prepare`/`._to_int` which uses FNV-1a) | 1 | two independent `nn.Embedding`s, summed (double-hash computed internally via Knuth multiplicative constants) |

**Adding a new feature type** = one new dataclass with `build` / `encode` / `n_slots`, zero changes elsewhere.

`TransactionEncoder.n_fields` is computed from the schema and is passed into `FieldTransformer` (which sizes its learnable field-type positional encoding accordingly). The default schema lives in `src/model.py` as `DEFAULT_FEATURES` and produces 5 slots (`importo` signed → 2, `merchant` hash → 1, `mcc`/`macro_tipo` → 2). It does **not** include a `DatetimeFeature`: the `timestamp` column is loaded into the batch but consumed only to derive `delta_t`, not embedded.

### Consequence: loss heads track the schema

`MTMHead` (`src/loss.py`) takes a `vocab_sizes` dict. `TransactionTransformer` derives it via `categorical_vocab_sizes(features)` (in `src/encoder.py`) so MTM heads automatically cover whatever categorical features are in the schema. `mtm_loss` iterates the `cat_*` keys actually present in `preds` rather than a hard-coded list — adding/removing a `CategoricalFeature` requires no loss-side changes.

### Padding convention

Padding is represented by **index 0** for categoricals/hashes and **value 0** for numerics/timestamps. All `nn.Embedding` instances use `padding_idx=0` so their gradients are zero on pad positions. A separate boolean `padding_mask` in the batch dict (True = padded) is consumed by `FieldTransformer`/`SequenceTransformer` for attention masking.

### Time flows end-to-end

`delta_t` (seconds between consecutive transactions) is derived at load time and used directly by `SequenceTransformer.TimeAwarePositionalEncoding` — a fixed sinusoidal PE driven by real time gaps instead of integer positions. The [CLS] token is given `delta_t=0`. `delta_t` is **not** part of the encoder schema (`DEFAULT_FEATURES`), so it is never an MTM target.

## Notes on modifying the schema

Adding a new feature: append a spec to `DEFAULT_FEATURES` in `src/model.py`, provide the expected batch key(s) at forward time. No other edits needed — `TransactionEncoder`, `FieldTransformer`, and `MTMHead` all adapt automatically.

## Synthetic data pipeline (CPU smoke test)

```
src/make_dataset.py   ──▶  data/transactions.csv  (~400k rows, 4000 clients)
                                  │
                                  ▼
src/data.py           load_dataframe → fit_features (numeric normalizers via
                      NumericFeature.fit) → TransactionDataset (per-client
                      windows of length seq_len, delta_t computed from
                      consecutive timestamps, right-padded) → 
                      PairedClientBatchSampler (≥2 windows per client per
                      batch, so InfoNCE always sees positive pairs) → collate
                                  │
                                  ▼
src/train.py          builds the model with the fitted features, runs the
                      joint MTM + InfoNCE objective, saves
                      checkpoints/model_final.pt
```

CSV columns: `client_id, timestamp, importo, merchant, mcc, macro_tipo`. `merchant` is a string; `data.py` hashes it to an int64 ID via `HighCardCategoricalFeature._to_int` (FNV-1a) before tensorising.

`delta_t` is **derived** at load time from per-client `np.diff(timestamp)` — the CSV does not store it.

`fit_features` only fits `NumericNormalizer`s; categorical vocab sizes are kept at the explicit values in `DEFAULT_FEATURES` (the synthetic generator stays inside those ranges).

## Training metrics & plotting

`src.train` records one entry per step into ``checkpoints/history.json``:

| Field             | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `loss`            | total loss = MTM + λ·InfoNCE                                  |
| `loss_mtm`        | masked-token-modeling loss (averaged over fields with masked positions) |
| `loss_contrastive`| InfoNCE                                                       |
| `infonce_acc`     | top-1 retrieval accuracy in-batch (`info_nce_metrics` in `src/loss.py`) |
| `infonce_acc_random` | random baseline `(k-1)/(B-1)` averaged over anchors (same helper) |
| `infonce_lift`    | normalized lift `(acc - acc_random) / (1 - acc_random)` (same helper) |
| `temperature`     | learnable temperature of the contrastive head                 |
| `grad_norm`       | total grad norm before clipping                               |
| `mtm_breakdown`   | dict `{cat_<name>: CE, num_<name>: smoothL1}` per masked field |

`src.plots` reads that JSON and writes:
* ``checkpoints/plots/training_curves.png`` — 6-panel summary (totale, MTM in log, InfoNCE, accuracy, temperature, grad norm) with a moving-average overlay
* ``checkpoints/plots/mtm_breakdown.png``  — per-field MTM curves (categorical CE on linear axis, numeric smooth-L1 on log axis)

`uv run python -m src.plots --tensorboard` additionally replays the same JSON history into TensorBoard event files under ``runs/<timestamp>/`` via `TensorBoardExporter` (a post-hoc class in `src/plots.py`, no training-loop changes). View with `tensorboard --logdir runs`. Layout: **one chart per metric** (bare tag — `loss`, `loss_mtm`, `loss_contrastive`, `accuracy`, `lift`, `temperature`, `grad_norm`, `lr`), with `train` / `val` written as separate TB *runs* (sub-directories) so they overlay on the same chart distinguished by the run legend. The random accuracy baseline is a `random` run sharing the `accuracy` chart. Per-field MTM curves go under `mtm/*` on the `train` run. The x-axis is the global training step throughout — per-epoch val points are placed at the step that closed their epoch.

Note: numeric MTM targets are **normalised** — `Trainer._build_mtm_targets` applies each `NumericFeature.normalizer` (clip → log1p → z-score) so the smooth-L1 term lives on the same scale as the encoder input and the categorical cross-entropy, rather than being dominated by the raw euro magnitudes of `importo`.
