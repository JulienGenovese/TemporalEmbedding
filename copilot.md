# Copilot instructions for this repository

## Project at a glance

- **Name**: `embeddingclient`
- **Goal**: train a hierarchical Transformer that produces customer embeddings from transaction sequences.
- **Core pipeline**:
  1. generate synthetic transactions (`src.make_dataset`)
  2. build windowed dataset/dataloaders (`src.data`)
  3. pretrain model with MTM + InfoNCE (`src.train`)
  4. plot metrics (`src.plots`)

## Tech stack

- Python package managed with **uv**
- Main ML framework: **PyTorch**
- Also present in dependencies: TensorFlow/Transformers (do not remove unless explicitly requested)

## Environment and execution

- Python version from `pyproject.toml`: **>=3.10,<3.12**
- Install deps:

```bash
uv sync
```

- Run modules from repo root with `-m`:

```bash
uv run python -m src.datasets.sintetic
uv run python -m src.models.hier_transformer.train
uv run python -m src.plots.plots
```

## Data/model contract (important)

- Training feature columns are defined in `src/config.py`:
  - embedded fields: `importo`, `merchant`, `mcc`, `macro_tipo`
  - non-embedded pipeline fields: `client_id`, `timestamp`, derived `delta_t`
- Keep generator output (`src/make_dataset.py`) aligned with this schema.
- `0` is reserved as padding id for categorical tensors.

## Key files

- `src/make_dataset.py`: synthetic transaction generator
- `src/data.py`: dataset windowing, padding masks, paired sampler for InfoNCE
- `src/encoder.py`: field encoders and feature specs
- `src/field_transformer.py`: intra-transaction attention
- `src/sequence_encoder.py`: temporal sequence transformer
- `src/loss.py`: MTM + contrastive losses/metrics
- `src/model.py`: backbone + wrapper + default features
- `src/train.py`: end-to-end pretraining loop and checkpointing
- `src/config.py`: all training/data/model hyperparameters

## Coding conventions for contributions

- Prefer small, surgical edits; preserve current architecture and naming.
- Keep training defaults CPU-friendly (smoke-test scale) unless asked otherwise.
- Do not silently change feature schema/vocab ranges; update all dependent modules/docs together.
- Keep scripts runnable as modules (`python -m src.<name>`).
- If changing data columns, also update:
  - `src/config.py` (`FEATURE_COLS`, `DataConfig`)
  - `src/model.py` (`DEFAULT_FEATURES`)
  - docs (`README.md`, `dataset.md`)

## Validation checklist after code changes

1. `uv run python -m src.make_dataset`
2. `uv run python -m src.train`
3. `uv run python -m src.plots`

Ensure artifacts are produced under `data/` and `checkpoints/` without breaking module execution.

