# Hierarchical Transformer for Banking Transaction Embeddings

## Pipeline

```
Raw fields → TransactionEncoder → FieldTransformer → SequenceTransformer → [CLS] embedding
```

## Design choices

- `d_field = 64` (encoding output per field)
- `d_model = 128` (transformer hidden dim)
- Field count derived from feature schema (default: 13 fields)
- Field Transformer: 2 layers, 4 heads, attention pooling
- Sequence Transformer: 4 layers, 8 heads, [CLS] token
- Memory optimizations: gradient checkpointing, SDPA (Flash Attention), mixed precision ready, in-place operations where safe
- Total parameters: ~2.5M
- Target hardware: single T4 / L4 GPU

## Full architecture diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Raw batch (B, T)                                                  │
│                                                                    │
│  importo, saldo_post, delta_t          (float)                     │
│  merchant_a, merchant_b                (int — double hash)         │
│  mcc, canale, macro_tipo, sotto_tipo   (int — categorical)         │
│  divisa                                (int — categorical)         │
│  timestamp                             (int64 — unix epoch)        │
│  padding_mask                          (bool)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  TransactionEncoder                                           │
│                                                               │
│  NumericFeature   → sinusoidal projection → (B, T, n_f, 64)  │
│  CategoricalFeature → nn.Embedding        → (B, T, 1,   64)  │
│  HighCardCategoricalFeature  → embed_a + embed_b   → (B, T, 1,   64)  │
│  DatetimeFeature    → hour/dow/month embs → (B, T, 3,   64)  │
│                                                               │
│  Stack along field dim → (B, T, 13, 64)                      │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  FieldTransformer                                             │
│                                                               │
│  input_proj: Linear(64 → 128)                                │
│  + field_type_emb (13, 128) learned                          │
│  reshape → (B·T, 13, 128)                                    │
│  TransformerEncoder × 2 layers (pre-LN, 4 heads, d_ff=512)  │
│  AttentionPooling: 13 fields → 1 vector                     │
│  reshape → (B, T, 128)                                       │
│  LayerNorm(128)                                              │
│                                                               │
│  Output: (B, T, 128) — one embedding per transaction         │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  SequenceTransformer                                          │
│                                                               │
│  Prepend [CLS] token (1, 1, 128) learned                    │
│  + TimeAwarePositionalEncoding(delta_t) → sin/cos             │
│  (B, T+1, 128)                                               │
│                                                               │
│  TransformerEncoder × 4 layers (pre-LN, 8 heads, d_ff=512)  │
│  + gradient checkpointing (training only)                    │
│                                                               │
│  LayerNorm(128)                                              │
│  Extract x[:, 0, :] → h_CLS                                 │
│                                                               │
│  Output: (B, 128) — one embedding per client                 │
└──────────┬─────────────────────────────────┬──────────────────┘
           │                                 │
           ▼                                 ▼
┌──────────────────────┐          ┌─────────────────────────┐
│  MTMHead             │          │  ContrastiveHead        │
│                      │          │                         │
│  input: (B, T, 128)  │          │  input: (B, 128)        │
│  transaction embeds  │          │  h_CLS                  │
│                      │          │                         │
│  per-field linear    │          │  Linear(128, 64)        │
│  projections →       │          │  L2 normalize           │
│  reconstruct masked  │          │  → contrastive_z        │
│  fields              │          │  (B, 64)                │
│                      │          │                         │
│  Loss: cross-entropy │          │  Loss: InfoNCE          │
│  per masked position │          │  across batch           │
└──────────────────────┘          └─────────────────────────┘

Pre-training loss = λ_mtm · L_mtm + λ_con · L_contrastive

After pre-training, heads are removed.
h_CLS (B, 128) is used as feature for downstream tasks:
    → fine-tuning heads (churn, fraud, default)
    → or as input to XGBoost / LightGBM
```
