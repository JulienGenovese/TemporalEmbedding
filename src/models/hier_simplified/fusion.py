"""Concat + MLP field fusion: the non-attention alternative to FieldTransformer."""

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Fuse per-field embeddings by concatenation + 2-layer MLP.

    Drop-in replacement for
    :class:`~src.models.hier_transformer.field_transformer.FieldTransformer`:
    same ``(B, T, n_fields, d_field) -> (B, T, d_model)`` contract, but no
    intra-transaction attention. Field identity is implicit in the
    concatenation order (each field always occupies the same input
    positions), so no field-type positional encoding is needed. The hidden
    non-linearity is what captures field interactions — a single linear
    layer would only produce a weighted sum of slots.
    """

    def __init__(
        self,
        n_fields: int = 8,
        d_field: int = 64,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_fields * d_field, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_fields, d_field)
        Returns:
            (B, T, d_model)
        """
        batch_size, seq_len, n_fields, d_field = x.shape
        return self.mlp(x.reshape(batch_size, seq_len, n_fields * d_field))
