# Copilot instructions for this repository

## Project at a glance

- **Name**: `embeddingclient`
- **Goal**: train a hierarchical Transformer that produces customer embeddings from transaction sequences.
- **Primary entrypoint**: Typer CLI exposed as `py` (`src/cli.py`).
- **Core pipeline**:
  1. generate synthetic transactions (`py generate --type vanilla|coherent`)
  2. build windowed dataset/dataloaders (`src.models.hier_transformer.data`)
  3. pretrain model with MTM + InfoNCE (`py train --type hier|base`)
  4. export metrics to TensorBoard (`py plot --type hier|base`)

## Tech stack

- Python package managed with **uv**
- Main ML framework: **PyTorch**
- CLI framework: **Typer**

## Environment and execution

- Python version from `pyproject.toml`: **>=3.10,<3.12**
- Install deps:

```bash
uv sync
```

- Run from repo root via CLI:

```bash
uv run py --help
uv run py generate --type coherent
uv run py train --type hier
uv run py plot --type hier --history checkpoints/history.json
```

Notes:
- `--type` also accepts the legacy alias `-type`.
- `train` defaults to `hier`.
- `plot` is a dedicated command (not an option of `train`).
- `base` is currently routed to the same training/plotting pipeline used by `hier`.

## Synthetic data knobs (noise + generation params)

- File: `src/datasets/utils/config.py`
- Main difficulty knob: `NoiseConfig.noise_level` (range `[0,1]`, letto da `[dataset.sampling]`)
- Derived automatically from `noise_level`:
  - `p_offpattern`
  - `p_global_merchant`
  - `p_refund`
  - `sigma_spending`

Other generator parameters:

- `SamplingConfig`: `n_transactions`, `n_clients`, `alpha_dirichlet`, `min_tx_per_client`, `seed`
- `AmountConfig`: `spending_probability`, `lognormal_sigma`
- `MerchantConfig`: merchant pools, `common_merchants`, `p_common_merchant`
- `CategoricalConfig`: `cocau_vocab`, `p_noise`
- `OutputConfig`: `vanilla_out_path`, `coherent_out_path`

## Model/training knobs

- File: `src/models/hier_transformer/hier_config.py`
- `TrainingConfig`: data/training/runtime parameters (`epochs`, `seq_len`, `clients_per_batch`, `mask_prob`, `contrastive_weight`, `lr`, `weight_decay`, `lr_gamma`, `val_frac`, `device`, `train_path`, `pred_path`, `pred_file_name`, ...)
- `ModelConfig`: architecture parameters (`d_field`, `d_model`, `field_n_layers`, `field_n_heads`, `seq_n_layers`, `seq_n_heads`, `dim_feedforward`, `dropout`, `n_frequencies`)

## Data/model contract (important)

- Training feature columns are defined in `src/config.py`.
- Keep generator output aligned with this schema across both generators:
  - `src/datasets/vanilla_sintetic.py`
  - `src/datasets/coherent_sintetic.py`
  - shared build path in `src/datasets/common.py`
- `0` is reserved as padding id for categorical tensors.

## Key files

- `src/cli.py`: Typer CLI (`generate`, `train`, `plot`)
- `src/datasets/vanilla_sintetic.py`: vanilla synthetic generator entrypoint
- `src/datasets/coherent_sintetic.py`: coherent synthetic generator entrypoint
- `src/datasets/common.py`: shared synthetic dataset generation core
- `src/datasets/utils/config.py`: synthetic dataset hyperparameters (including noise/output paths)
- `src/models/hier_transformer/data.py`: dataset windowing, padding masks, paired sampler for InfoNCE
- `src/models/hier_transformer/encoder.py`: field encoders and feature specs
- `src/models/hier_transformer/field_transformer.py`: intra-transaction attention
- `src/models/hier_transformer/sequence_encoder.py`: temporal sequence transformer
- `src/models/hier_transformer/loss.py`: MTM + contrastive losses/metrics
- `src/models/hier_transformer/model.py`: backbone + wrapper + default features
- `src/models/hier_transformer/train.py`: end-to-end pretraining loop and checkpointing
- `src/models/hier_transformer/hier_config.py`: training/model hyperparameters
- `src/plots/tensorboard.py`: TensorBoard export from training histories

## Coding conventions for contributions

- Prefer small, surgical edits; preserve current architecture and naming.
- Keep training defaults CPU-friendly (smoke-test scale) unless asked otherwise.
- Do not silently change feature schema/vocab ranges; update all dependent modules/docs together.
- Keep CLI behavior/documentation in sync when adding/changing commands or options.
- If changing data columns, also update:
  - `src/config.py` (`FEATURE_COLS`, `DataConfig`)
  - `src/models/hier_transformer/model.py` (`DEFAULT_FEATURES`)
  - docs (`README.md`, `copilot.md`)

## Validation checklist after code changes

1. `uv run py generate --type coherent`
2. `uv run py train --type hier`
3. `uv run py plot --type hier --history checkpoints/history.json`

Ensure artifacts are produced under `data/`, `checkpoints/`, and `runs/` without breaking CLI execution.
