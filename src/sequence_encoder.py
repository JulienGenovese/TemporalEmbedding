"""Sequence Transformer — inter-transaction attention. See architecture/sequence_transformer.md."""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TimeAwarePositionalEncoding(nn.Module):
    """Positional encoding based on inter-transaction time deltas.

    Uses the same sinusoidal idea as standard PE but driven by actual
    time gaps (in seconds) rather than integer positions. The [CLS] token
    at position 0 receives a zero time encoding.

    Frequencies are fixed (not learnable) following Vaswani et al.
    """

    def __init__(self, d_model: int = 128, max_timescale: float = 1e6):
        super().__init__()
        self.d_model = d_model
        # Pre-compute frequency bands (fixed)
        half = d_model // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32) * (math.log(max_timescale) / half)
        )
        self.register_buffer("freqs", freqs)  # (d_model/2,)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: (B, T+1) — time deltas in seconds. Position 0 ([CLS]) should be 0.
        Returns:
            (B, T+1, d_model) — positional encoding to ADD to the sequence
        # PE(Δt)=[sin(ω0​Δt),sin(ω1​Δt),…,sin(ω63​Δt),cos(ω0​Δt),cos(ω1​Δt),…,cos(ω63​Δt)]

        """
        angles = delta_t.unsqueeze(-1) * self.freqs  # (B, T+1, d_model/2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, T+1, d_model)


class SequenceTransformer(nn.Module):
    """Inter-transaction attention over the client's transaction sequence.

    Prepends a learnable [CLS] token, adds time-aware positional encoding,
    applies 4 Transformer encoder layers with gradient checkpointing and
    SDPA (Flash Attention), returns h_CLS as the client embedding.

    Input:  (B, T, d_model) — from FieldTransformer
    Output: (B, d_model) — client embedding (h_CLS)
    """

    def __init__(self, 
                 d_model: int = 128, 
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Time-aware positional encoding
        self.time_pe = TimeAwarePositionalEncoding(d_model)

        # Individual layers (for gradient checkpointing control)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, 
                x: torch.Tensor,
                delta_t: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — transaction embeddings from FieldTransformer
            delta_t: (B, T) — time deltas in seconds between consecutive transactions
            padding_mask: (B, T) — True where transactions are padded
        Returns:
            (B, d_model) — h_CLS client embedding
        """
        B, T, _ = x.shape

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, T+1, d_model)

        # Build time encoding: [CLS] gets delta_t=0
        cls_time = torch.zeros(B, 1, device=delta_t.device, dtype=delta_t.dtype)
        full_delta_t = torch.cat([cls_time, delta_t], dim=1)  # (B, T+1)
        x = x + self.time_pe(full_delta_t)

        # Extend padding mask for [CLS] (never masked)
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, device=padding_mask.device, dtype=torch.bool)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, T+1)

        # Apply Transformer layers with optional gradient checkpointing
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(
                    layer, x, None, padding_mask,
                    use_reentrant=False
                )
            else:
                x = layer(x, src_key_padding_mask=padding_mask)

        x = self.final_norm(x)

        # Extract [CLS] output
        h_cls = x[:, 0, :]  # (B, d_model)
        return h_cls
