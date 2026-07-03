"""Field Transformer: intra-transaction attention over fields."""


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
        batch_transactions = x.size(0)
        query = self.query.expand(batch_transactions, -1, -1)
        out, _ = self.attn(query, x, x)
        return out.squeeze(1)


class FieldTransformer(nn.Module):
    """Intra-transaction attention over fields."""

    def __init__(
        self,
        n_fields: int = 13,
        d_field: int = 64,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(d_field, d_model)

        self.field_type_emb = nn.Parameter(torch.randn(n_fields, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pool = AttentionPooling(d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_fields, d_field)
        Returns:
            (B, T, d_model)
        """
        batch_size, seq_len, n_fields, _ = x.shape

        x = self.input_proj(x) + self.field_type_emb
        x = x.reshape(batch_size * seq_len, n_fields, self.d_model)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.reshape(batch_size, seq_len, self.d_model)
        return self.layer_norm(x)