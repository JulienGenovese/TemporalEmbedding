"""Sequence Transformer: inter-transaction attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_TIME_ALPHA_BETA = 10.0


class TimeDeltaEncoding(nn.Module):
    """Positional encoding based on inter-transaction time deltas.
    Uses the same sinusoidal idea as standard PE but driven by actual
    time gaps (in seconds) rather than integer positions. The [CLS] token
    at position 0 receives a zero time encoding.
    Frequencies are fixed (not learnable) following Vaswani et al. The
    bank spans periods geometrically from ``min_timescale`` (fastest
    channel) to ``max_timescale`` (slowest channel), both in seconds, so
    it can be aligned with the actual range of inter-transaction gaps.
    Setting ``min_timescale`` above the smallest real delta_t avoids
    aliasing the high-frequency channels.
    """

    def __init__(
        self,
        d_model: int = 128,
        min_timescale: float = 60.0,
        max_timescale: float = 1e6,
    ):
        super().__init__()
        if d_model < 4 or d_model % 2 != 0:
            raise ValueError("d_model must be an even integer >= 4")
        if min_timescale <= 0 or max_timescale <= min_timescale:
            raise ValueError("Expected 0 < min_timescale < max_timescale")

        self.d_model = d_model
        half = d_model // 2
        log_min, log_max = math.log(min_timescale), math.log(max_timescale)
        steps = torch.arange(half, dtype=torch.float32)
        timescales = torch.exp(log_min + steps * (log_max - log_min) / (half - 1))
        freqs = 1.0 / timescales
        self.register_buffer("freqs", freqs)  # (d_model/2,)

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: (B, T+1), time deltas in seconds. Position 0 ([CLS]) should be 0.
        Returns:
            (B, T+1, d_model), positional encoding to add to the sequence.
        """
        angles = delta_t.unsqueeze(-1) * self.freqs  # (B, T+1, d_model/2)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, T+1, d_model)


class SequenceTransformer(nn.Module):
    """Inter-transaction attention over the client's transaction sequence."""

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        time_alpha_init: float = 0.1,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if time_alpha_init < 0:
            raise ValueError("time_alpha_init must be >= 0")

        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.time_delta_encoding = TimeDeltaEncoding(d_model)
        self.time_log_alpha = nn.Parameter(torch.tensor(
            _softplus_inverse(float(time_alpha_init)),
            dtype=torch.float32,
        ))

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

    @property
    def time_alpha(self) -> torch.Tensor:
        """Positive learned scale applied to the time-delta encoding."""
        return F.softplus(self.time_log_alpha, beta=_TIME_ALPHA_BETA)

    def _prepend_cls(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls, x], dim=1)

    @staticmethod
    def _prepend_zero_time(delta_t: torch.Tensor) -> torch.Tensor:
        batch_size = delta_t.size(0)
        cls_time = torch.zeros(
            batch_size,
            1,
            device=delta_t.device,
            dtype=delta_t.dtype,
        )
        return torch.cat([cls_time, delta_t], dim=1)

    @staticmethod
    def _prepend_cls_mask(
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if padding_mask is None:
            return None
        padding_mask = padding_mask.to(dtype=torch.bool)
        batch_size = padding_mask.size(0)
        cls_mask = torch.zeros(
            batch_size,
            1,
            device=padding_mask.device,
            dtype=torch.bool,
        )
        return torch.cat([cls_mask, padding_mask], dim=1)

    def _run_layer(
        self,
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(layer, x, None, padding_mask, use_reentrant=False)
        return layer(x, src_key_padding_mask=padding_mask)

    def forward(
        self,
        x: torch.Tensor,
        delta_t: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model), transaction embeddings from FieldTransformer
            delta_t: (B, T), time deltas in seconds between consecutive transactions
            padding_mask: (B, T), True where transactions are padded
        Returns:
            (B, d_model), h_CLS client embedding
        """
        x = self._prepend_cls(x)
        time_encoding = self.time_delta_encoding(self._prepend_zero_time(delta_t))
        x = x + self.time_alpha * time_encoding
        padding_mask = self._prepend_cls_mask(padding_mask)

        for layer in self.layers:
            x = self._run_layer(layer, x, padding_mask)

        x = self.final_norm(x)
        return x[:, 0, :]


def _softplus_inverse(value: float) -> float:
    if value == 0:
        return -20.0
    return math.log(math.expm1(_TIME_ALPHA_BETA * value)) / _TIME_ALPHA_BETA