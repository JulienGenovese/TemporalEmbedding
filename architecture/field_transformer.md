# FieldTransformer (`src/field_encoder.py`)

## Overview

Intra-transaction attention over fields. Takes the per-field embeddings from `TransactionEncoder` and produces a single vector per transaction via attention pooling.

- **Input:** `(B, T, 13, 64)` — batch × sequence × fields × embedding per field
- **Output:** `(B, T, 128)`

## Architecture diagram

```
                ┌─────────────────────────┐
                │  Input (B, T, 13, 64)   │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐    ┌─────────────────────┐
                │  input_proj             │◄───│  field_type_emb     │
                │  Linear(64 → 128)       │ +  │  (13, 128) learned  │
                └────────────┬────────────┘    └─────────────────────┘
                             │
                    reshape (B·T, 13, 128)
                             │
                ┌────────────▼────────────┐
                │  TransformerEncoder      │
                │  ┌────────────────────┐  │
                │  │ Layer 1  (pre-LN)  │  │
                │  │ 4 heads, d_ff=512  │  │
                │  └────────┬───────────┘  │
                │  ┌────────▼───────────┐  │
                │  │ Layer 2  (pre-LN)  │  │
                │  │ 4 heads, d_ff=512  │  │
                │  └────────┬───────────┘  │
                └───────────┼──────────────┘
                            │
                ┌───────────▼──────────────┐
                │  AttentionPooling         │
                │  query (1,1,128) learned  │
                │  13 fields → 1 vector     │
                └───────────┬──────────────┘
                            │
                    reshape (B, T, 128)
                            │
                ┌───────────▼──────────────┐
                │  LayerNorm(128)           │
                └───────────┬──────────────┘
                            │
                ┌───────────▼──────────────┐
                │  Output (B, T, 128)       │
                │  → Sequence Transformer   │
                └──────────────────────────┘
```

## Components

### input_proj — `Linear(d_field → d_model)`
Decouples the input dimension from the model dimension.

### field_type_emb — `(n_fields, d_model)` learnable
Positional encoding per field type. Prevents the model from confusing which field is which (e.g. amount vs merchant).

### TransformerEncoder — 2 layers, 4 heads, pre-LN
Processes all transactions in parallel by reshaping to `(B·T, 13, 128)`. The sequence length is tiny (13 fields), so Flash Attention is not needed.

### AttentionPooling — 13 fields → 1 vector
A learnable query attends to all field representations. Chosen over mean pooling to let the model learn which fields matter most.
