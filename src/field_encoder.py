"""
Field Transformer — intra-transaction attention over fields.

AttentionPooling  — learned attention pooling over the field dimension
FieldTransformer  — projects d_field → d_model, applies Transformer layers,
                    attention-pools to a single vector per transaction
                    
L'input arriva come (B, T, 13, 64) — batch × sequenza temporale × campi × embedding per campo.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learned attention pooling over the field dimension.

    A learnable query attends to all field representations and produces
    a single vector per transaction step.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, n_fields, d_model)
        Returns:
            (B*T, d_model)
        """
        BT = x.size(0)
        q = self.query.expand(BT, -1, -1)  # (B*T, 1, d_model)
        out, _ = self.attn(q, x, x)        # (B*T, 1, d_model)
        return out.squeeze(1)               # (B*T, d_model)


class FieldTransformer(nn.Module):
    """Intra-transaction attention over fields.

    Projects d_field → d_model, adds field-type positional encoding,
    applies 2 Transformer encoder layers, then attention-pools to a
    single vector per transaction.

    Input:  (B, T, n_fields, d_field)
    Output: (B, T, d_model)
    """

    def __init__(self, 
                 n_fields: int = 13,
                 d_field: int = 64, 
                 d_model: int = 128,
                 n_layers: int = 2,
                 n_heads: int = 4, 
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Shared projection from d_field to d_model
        # serve per svincolare la dimensione degli input dalla dimensione che serve al modello
        self.input_proj = nn.Linear(d_field, d_model)

        # Learnable field-type positional encoding -> serve per evitare importo = amazon merchant=500euro
        self.field_type_emb = nn.Parameter(torch.randn(n_fields, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True, #per dire che la prima dimensione e' il batch
            norm_first=True,  # Pre-LN for training stability
        )
        # prendo il layer e lo impilo n_layers piu' volte.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling: n_fields → 1
        # questo serve perche' vogliamo arrivare a un punto nello spazio, non n_fields. Potremmo mare un mean pooling ma abbiamo preferito un attention mechanism
        self.pool = AttentionPooling(d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_fields, d_field)
            padding_mask: (B, T) — True where transactions are padded
        Returns:
            (B, T, d_model)
        """
        B, T, F, _ = x.shape

        # Project and add field-type encoding
        x = self.input_proj(x) + self.field_type_emb  # (B, T, F, d_model)

        # Reshape to process all transactions in parallel: (B*T, F, d_model)
        x = x.reshape(B * T, F, self.d_model)

        # Transformer encoder (F=13 is tiny, no need for Flash Attention here)
        x = self.encoder(x)  # (B*T, F, d_model)

        # Attention pooling over fields
        x = self.pool(x)  # (B*T, d_model)

        # Reshape back
        x = x.reshape(B, T, self.d_model)  # (B, T, d_model)

        return self.layer_norm(x)
