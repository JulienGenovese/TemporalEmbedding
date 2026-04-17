# SequenceTransformer (`src/sequence_encoder.py`)

## Overview

Inter-transaction attention over the client's transaction sequence. Prepends a [CLS] token, adds time-aware positional encoding, and returns `h_CLS` as the client embedding.

- **Input:** `(B, T, 128)` from FieldTransformer + `(B, T)` delta_t in seconds
- **Output:** `(B, 128)` — client embedding (h_CLS)

## Architecture diagram

```
(B, T, 128) from FieldTransformer          (B, T) delta_t in seconds
            │                                        │
            │    ┌───────────────────┐               │
            │    │  cls_token        │               │
            │    │  (1, 1, 128)      │               │
            │    │  learned          │               │
            │    └────────┬──────────┘               │
            │             │                          │
            ▼             ▼                          │
    ┌─────────────────────────────┐                  │
    │  cat([CLS], x)              │                  │
    │  (B, T+1, 128)              │                  │
    └──────────────┬──────────────┘                  │
                   │                                 │
                   │         ┌─────────────────────────────────┐
                   │         │  TimeAwarePositionalEncoding    │
                   │         │  delta_t → sin/cos frequencies  │
                   │         │  (B, T+1, 128)                 │
                   │         └───────────────┬─────────────────┘
                   │                         │
                   │          + sum           │
                   ◄──────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────────┐
    │  Transformer Encoder (×4 layers)             │
    │  ┌────────────────────────────────────────┐  │
    │  │ Layer 1  pre-LN, 8 heads, d_ff=512    │  │
    │  │ + gradient checkpointing (training)    │  │
    │  └───────────────────┬────────────────────┘  │
    │  ┌───────────────────▼────────────────────┐  │
    │  │ Layer 2  pre-LN, 8 heads, d_ff=512    │  │
    │  └───────────────────┬────────────────────┘  │
    │  ┌───────────────────▼────────────────────┐  │
    │  │ Layer 3  pre-LN, 8 heads, d_ff=512    │  │
    │  └───────────────────┬────────────────────┘  │
    │  ┌───────────────────▼────────────────────┐  │
    │  │ Layer 4  pre-LN, 8 heads, d_ff=512    │  │
    │  └───────────────────┬────────────────────┘  │
    └──────────────────────┼───────────────────────┘
                           │
                           ▼
            ┌──────────────────────────┐
            │  LayerNorm(128)          │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  extract x[:, 0, :]      │
            │  [CLS] token only        │
            └─────────────┬────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │  h_CLS  (B, 128)         │
            │  → downstream heads      │
            └──────────────────────────┘
```

## TimeAwarePositionalEncoding

Uses the same sinusoidal idea as standard PE but driven by actual time gaps (in seconds) rather than integer positions. The [CLS] token at position 0 receives a zero time encoding. Frequencies are fixed (not learnable) following Vaswani et al.

```
PE(Δt) = [sin(ω₀·Δt), sin(ω₁·Δt), …, sin(ω₆₃·Δt), cos(ω₀·Δt), cos(ω₁·Δt), …, cos(ω₆₃·Δt)]
```

## Gradient checkpointing

Each Transformer layer is individually wrapped with `torch.utils.checkpoint` during training to reduce memory usage. Disabled at inference time.
