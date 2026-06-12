# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses **uv** (requires Python ≥3.10, <3.12) and has no test framework configured.
All runtime hyper-parameters and I/O paths live in `config.toml` at the repo root.

The canonical entrypoint is the **`py` Typer CLI** (`cli.py`, exposed as the `py` script via
`pyproject.toml`):

```bash
# Install / sync dependencies
uv sync

# Generate a synthetic CSV dataset under data/ (~400k rows / 4000 clients)
uv run py syntetic --type vanilla     # i.i.d. amounts
uv run py syntetic --type coherent    # merchant-correlated amounts

# Full end-to-end pre-training (MTM + InfoNCE) — CPU-friendly defaults
uv run py train --type hier

# Window-level client embeddings from a trained checkpoint → data/pred/
uv run py pred --type hier

# Replay training history into TensorBoard event files under runs/<type>/
uv run py plot --type hier
tensorboard --logdir runs
```

`--type` accepts `hier` or `base`; `base` is currently routed to the same `hier_transformer`
pipeline (logged as such). The CLI also accepts `-type` as an alias for `--type` (normalized in
`cli.main`). Dataset names/folders, the training CSV path (`train_path`), checkpoint dir, and all
hyper-parameters are read from `config.toml`.

Forward-pass smoke test (shape checks only, synthetic in-memory batch):

```bash
uv run python -m src.models.hier_transformer.model
```

Modules inside `src/` use relative imports, so invoke them via `python -m src.<dotted.path>`
rather than `python src/.../file.py`.

`py train` runs with the `config.toml` defaults (30 epochs, batch = `clients_per_batch ×
windows_per_pair`, with validation + early stopping) and writes `checkpoints/model_final.pt` plus
`history.json` / `train_eval_history.json` / `val_history.json`.

## Repository layout

```
cli.py                              Typer CLI (syntetic / train / pred / plot)
config.toml                         all hyper-parameters & paths
src/
  config.py                         TOML loader (`config`, `get_config`)
  constant.py                       DataConfig — canonical column names
  datasets/                         synthetic data generation
    main.py                           generate(dataset_type) dispatcher
    experiments/{vanilla,coherent_sintetic,common}.py
    generators/{cluster,merchant,timestamp,transactions,abc}.py
    utils/{config,entities}.py
  models/
    hier_transformer/              the model + training/prediction pipeline
      encoder.py field_transformer.py sequence_encoder.py
      model.py loss.py data.py train.py pred.py hier_config.py
    baseline/                      (placeholder)
  plots/tensorboard.py             history.json → TensorBoard exporter
  eval/                            (placeholder)
```

## Architecture

A hierarchical Transformer for banking-transaction client embeddings. The pipeline has three
stacked stages plus optional pre-training heads (all under `src/models/hier_transformer/`):

```
batch (dict of (B,T) tensors)
        │
        ▼
TransactionEncoder      encoder.py            → (B, T, n_fields, d_field)
        │   schema-driven: per-field sub-encoders
        ▼
FieldTransformer        field_transformer.py  → (B, T, d_model)
        │   intra-transaction attention + AttentionPooling across fields
        ▼
SequenceTransformer     sequence_encoder.py   → (B, d_model)
        │   prepends [CLS], TimeAwarePositionalEncoding driven by delta_t,
        │   gradient-checkpointed Transformer layers, returns h_CLS
        ▼
       h_cls   ── optional ──▶ MTMHead + ContrastiveHead (loss.py)
```

`TransactionTransformer` (`model.py`) is the backbone; `EmbeddingModel` wraps it with production
helpers (`embed` for inference, `save`/`load` for checkpoint round-trip). Architecture defaults
(from `config.toml [model.hierTransformer.architecture]`, surfaced via `ModelConfig` in
`hier_config.py`): `d_field=64`, `d_model=128`, `n_frequencies=16`, field layers=2×4heads,
sequence layers=4×8heads, `dim_feedforward=512`, `dropout=0.1`, ~2.5M params.

### The `TransactionEncoder` is schema-driven (key design)

Never hard-code field names or vocab sizes inside the encoder. Each feature spec in `encoder.py`
is a dataclass that **owns its own** `build()` (constructs sub-modules), `encode()` (produces field
tensors), and `n_slots` (count). `TransactionEncoder` itself is a thin dispatcher that iterates the
schema:

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

**Adding a new feature type** = one new dataclass with `build` / `encode` / `n_slots`, zero changes
elsewhere.

`TransactionEncoder.n_fields` is computed from the schema and is passed into `FieldTransformer`
(which sizes its learnable field-type positional encoding accordingly). The default schema lives in
`model.py` as `DEFAULT_FEATURES`:

```python
DEFAULT_FEATURES = [
    NumericFeature("importo", signed=True),   # → 2 slots
    HighCardCategoricalFeature("merchant"),    # → 1 slot
    CategoricalFeature("cocau", 501),          # → 1 slot
]                                              # total = 4 field slots
```

It does **not** include a `DatetimeFeature`: the `timestamp` column is loaded into the batch but
consumed only to derive `delta_t`, not embedded.

### Consequence: loss heads track the schema

`MTMHead` (`loss.py`) takes `vocab_sizes` + `numeric_names` dicts. `TransactionTransformer` derives
them via `categorical_vocab_sizes(features)` and `numeric_field_names(features)` (in `encoder.py`)
so MTM heads automatically cover whatever categorical/numeric features are in the schema. `mtm_loss`
iterates the keys actually present in `preds` rather than a hard-coded list — adding/removing a
feature requires no loss-side changes. `HighCardCategoricalFeature` and `DatetimeFeature` are never
MTM targets.

### Padding convention

Padding is represented by **index 0** for categoricals/hashes and **value 0** for
numerics/timestamps. All `nn.Embedding` instances use `padding_idx=0` so their gradients are zero on
pad positions. A separate boolean `padding_mask` in the batch dict (True = padded) is consumed by
`FieldTransformer`/`SequenceTransformer` for attention masking.

### Time flows end-to-end

`delta_t` (seconds between consecutive transactions) is derived at load time and used directly by
`SequenceTransformer.TimeAwarePositionalEncoding` — a fixed sinusoidal PE driven by real time gaps
instead of integer positions. The [CLS] token is given `delta_t=0`. `delta_t` is **not** part of the
encoder schema, so it is never an MTM target.

## Notes on modifying the schema

Adding a new feature: append a spec to `DEFAULT_FEATURES` in `model.py`, provide the expected batch
key(s) at forward time (and as a column in the CSV / `DataConfig.feature_cols`). No other edits
needed — `TransactionEncoder`, `FieldTransformer`, and `MTMHead` all adapt automatically.

## Synthetic data pipeline

```
src/datasets/  ──▶  data/transactions_<vanilla|coherent>*.csv  (~400k rows, 4000 clients)
   (py syntetic --type ...)        │
                                   ▼
src/models/hier_transformer/data.py
        DataModule:  _load_dataframe (CSV/parquet file OR folder of either)
                   → _split_clients (deterministic val_frac hold-out by client)
                   → _fit_features (NumericFeature.fit only)
                   → TransactionDataset (per-client windows of length seq_len,
                     delta_t from consecutive timestamps, right-padded; integer
                     columns stored at smallest lossless width to save RAM)
                   → PairedClientBatchSampler (windows_per_pair distinct windows
                     per client per batch, so InfoNCE always sees positive pairs)
                   → collate
                                   │
                                   ▼
src/models/hier_transformer/train.py
        Trainer builds EmbeddingModel from the fitted features, runs the joint
        MTM + InfoNCE objective, restores best-val weights, saves
        checkpoints/model_final.pt
```

Column names are centralized in `DataConfig` (`src/constant.py`):
`client_id, cluster, timestamp, importo, merchant, cocau`. Embedded feature columns are
`feature_cols = [importo, merchant, cocau]`; `cluster` is the synthetic ground-truth label (used for
downstream evaluation), `timestamp` is read only to compute `delta_t`. `merchant` is a string and is
hashed to an int64 ID via `HighCardCategoricalFeature._to_int` (FNV-1a) inside the Dataset before
tensorising.

`delta_t` is **derived** at load time from per-client `np.diff(timestamp)` — the CSV does not store
it. `_fit_features` only fits `NumericNormalizer`s; categorical vocab sizes stay at the explicit
values in `DEFAULT_FEATURES` (the synthetic generator stays inside those ranges).

`generate(dataset_type)` (`src/datasets/main.py`) dispatches to `experiments/vanilla.py` or
`experiments/coherent_sintetic.py`; the coherent variant correlates amounts with merchant via
`merchant_amount_weight`. Sampling parameters live under `[dataset.sampling]` in `config.toml`.

## Prediction

`py pred` (`pred.py`) loads `checkpoints/model_final.pt`, re-fits features identically to train time,
and emits **one embedding per distinct window** via `PredictionTransactionDataset` (deterministic,
non-overlapping windows + temporal bounds). Output goes to `data/pred/<pred_file_name>` (CSV or
parquet by extension) with columns `client_id, client_code, window_slot, window_start_ts,
window_end_ts, window_start, window_end, emb_0..emb_{d_model-1}`.

## Training metrics & plotting

`Trainer` records one entry per step into `checkpoints/history.json`, plus per-epoch
`train_eval_history.json` and `val_history.json` (mean over the train/val loaders):

| Field             | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `loss`            | total loss = MTM + λ·InfoNCE                                  |
| `loss_mtm`        | masked-token-modeling loss (averaged over fields with masked positions) |
| `loss_contrastive`| InfoNCE                                                       |
| `infonce_acc`     | top-1 retrieval accuracy in-batch (`info_nce_metrics` in `loss.py`) |
| `infonce_acc_random` | random baseline `(k-1)/(B-1)` averaged over anchors (same helper) |
| `infonce_lift`    | normalized lift `(acc - acc_random) / (1 - acc_random)` (same helper) |
| `temperature`     | learnable temperature of the contrastive head                 |
| `grad_norm`       | total grad norm before clipping                               |
| `lr`              | current learning rate (ExponentialLR)                         |
| `mtm_breakdown`   | dict `{cat_<name>: CE, num_<name>: smoothL1}` per masked field |

`py plot` (`src/plots/tensorboard.py`, `TensorBoardExporter`) replays that JSON history into
TensorBoard event files under `runs/<type>/` (override with `--history` / `--runs-dir`). Layout:
**one chart per metric** (bare tag — `loss`, `loss_mtm`, `loss_contrastive`, `accuracy`, `lift`,
`temperature`, `grad_norm`, `lr`), with `train` / `val` written as separate TB *runs* (sub-dirs) so
they overlay on the same chart distinguished by the run legend. The random accuracy baseline is a
`random` run sharing the `accuracy` chart. Per-field MTM curves go under `mtm/*` on the `train` run.
The x-axis is the global training step throughout — per-epoch val points are placed at the step that
closed their epoch.

Note: numeric MTM targets are **normalised** — `Trainer._build_mtm_targets` applies each
`NumericFeature.normalizer` (clip → log1p → z-score) so the smooth-L1 term lives on the same scale as
the encoder input and the categorical cross-entropy, rather than being dominated by the raw euro
magnitudes of `importo`. **It also zeroes out masked positions in the batch before `model(batch)`**
so the encoder cannot see the values it is asked to predict.
</content>
