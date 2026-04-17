# Pre-training Heads & Loss (`src/loss.py`)

## Overview

Two self-supervised pre-training objectives:

```
Pre-training loss = λ_mtm · L_mtm + λ_con · L_contrastive
```

After pre-training, both heads are removed and `h_CLS` is used directly.

## MTMHead — Masked Token Modeling

Reconstructs masked fields from `(B, T, d_model)` transaction embeddings:

| Field type | Projection | Loss |
|---|---|---|
| Categorical (mcc, canale, ...) | `Linear(d_model → vocab_size)` | Cross-entropy |
| Numeric (importo, saldo_post, delta_t) | `Linear(d_model → 1)` | Smooth-L1 |
| Full transaction | `Linear(d_model → d_model)` | MSE |

The categorical heads are schema-driven: `MTMHead` takes a `vocab_sizes` dict derived via `categorical_vocab_sizes(features)`, so adding/removing a `CategoricalFeature` requires no loss-side changes.

## ContrastiveHead — InfoNCE

Projects `h_CLS` into a lower-dimensional normalized space for contrastive learning (CoLES-style).

```
h_CLS (B, d_model) → Linear(128, 128) → ReLU → Linear(128, 64) → L2 normalize → z (B, 64)
```

Uses a **learnable temperature** (log scale for stability, init ~ 0.07, clamped to [0.01, 1.0]).

### InfoNCE loss

- **Positive pairs:** subsequences from the same client (matched via `client_ids`)
- **Negatives:** all other samples in the batch
- Similarity matrix: `z @ z.T / temperature`
- Self-similarity masked out, log-softmax over columns

## Combined loss

```python
L = L_mtm + λ · L_contrastive   # default λ = 0.5
```
