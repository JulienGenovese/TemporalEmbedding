"""
Training script for the hierarchical TransactionTransformer.

Runs a self-supervised pre-training loop (MTM + InfoNCE) on synthetic batches
and saves the final model + periodic checkpoints to ``checkpoints/``.

Usage:
    uv run python -m src.train
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .model import TransactionTransformer, DEFAULT_FEATURES, count_parameters
from .encoder import (
    NumericFeature, CategoricalFeature, DatetimeFeature, HighCardCategoricalFeature,
    FeatureSpec,
)
from .loss import combined_pretrain_loss


# ---------------------------------------------------------------------------
# Synthetic batch generator
# ---------------------------------------------------------------------------

TS_BASE = 1577836800  # 2020-01-01 00:00 UTC
TS_RANGE = 126_230_400  # ~4 years in seconds


def make_synthetic_batch(
    features: list[FeatureSpec],
    batch_size: int,
    seq_len: int,
    n_clients: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Generate a random batch mimicking the real schema.

    Returns (batch, client_ids). Each sample is assigned to one of ``n_clients``
    so the InfoNCE head has positive pairs in-batch.
    """
    B, T = batch_size, seq_len
    batch: dict[str, torch.Tensor] = {}

    for feat in features:
        if isinstance(feat, NumericFeature):
            # Float tensor; signed features can go negative
            scale = 500.0 if feat.signed else 1000.0
            vals = torch.randn(B, T, device=device) * scale
            if feat.name == "delta_t":
                vals = vals.abs() * 86.4  # keep in reasonable range of seconds
            batch[feat.name] = vals
        elif isinstance(feat, CategoricalFeature):
            batch[feat.name] = torch.randint(1, feat.vocab_size, (B, T), device=device)
        elif isinstance(feat, HighCardCategoricalFeature):
            batch[f"{feat.name}_a"] = torch.randint(1, feat.hash_buckets, (B, T), device=device)
            batch[f"{feat.name}_b"] = torch.randint(1, feat.hash_buckets, (B, T), device=device)
        elif isinstance(feat, DatetimeFeature):
            batch[feat.name] = (torch.randint(0, TS_RANGE, (B, T), device=device) + TS_BASE).long()
        else:
            raise TypeError(f"Unsupported feature spec: {type(feat).__name__}")

    # Random padding: last k transactions padded for each sample (k in [0, T//4])
    pad_lengths = torch.randint(0, max(T // 4, 1), (B,), device=device)
    pad_mask = torch.arange(T, device=device).unsqueeze(0) >= (T - pad_lengths).unsqueeze(1)
    batch["padding_mask"] = pad_mask

    client_ids = torch.randint(0, n_clients, (B,), device=device)
    return batch, client_ids


def build_mtm_targets(
    batch: dict[str, torch.Tensor],
    features: list[FeatureSpec],
    mask_prob: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build MTM targets and per-field masks.

    Targets are copies of the original batch values; masks are random boolean
    tensors (True = position is masked) that skip padded positions.
    """
    pad_mask = batch.get("padding_mask")
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}

    for feat in features:
        # Keys to grab from the batch
        if isinstance(feat, HighCardCategoricalFeature):
            # MTM doesn't target hash fields; skip
            continue
        if isinstance(feat, DatetimeFeature):
            # Datetime isn't a MTM target either (decomposed internally)
            continue
        name = feat.name
        targets[name] = batch[name].clone()

        B, T = batch[name].shape
        mask = torch.rand(B, T, device=batch[name].device) < mask_prob
        if pad_mask is not None:
            mask = mask & ~pad_mask  # don't mask padded positions
        masks[name] = mask

    return targets, masks


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    *,
    steps: int = 100,
    batch_size: int = 16,
    seq_len: int = 64,
    n_clients: int = 8,
    mask_prob: float = 0.15,
    contrastive_weight: float = 0.5,
    lr: float = 1e-4,
    grad_clip: float = 1.0,
    log_every: int = 10,
    ckpt_every: int = 50,
    ckpt_dir: str | Path = "checkpoints",
    device: str | None = None,
    seed: int = 0,
) -> Path:
    """Run pre-training and save the final model. Returns the final ckpt path."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = TransactionTransformer(pretrain=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Device: {device}")
    print(f"Params: {count_parameters(model)['total']:,}")
    print(f"Training {steps} steps, batch={batch_size}, T={seq_len}")

    model.train()
    for step in range(1, steps + 1):
        batch, client_ids = make_synthetic_batch(
            DEFAULT_FEATURES, batch_size, seq_len, n_clients, device
        )
        targets, mtm_mask = build_mtm_targets(batch, DEFAULT_FEATURES, mask_prob)

        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        output["temperature"] = model.contrastive_head.temperature

        losses = combined_pretrain_loss(
            output, targets, mtm_mask, client_ids, contrastive_weight
        )
        loss = losses["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == 1:
            print(
                f"step {step:>4}/{steps} | loss={loss.item():.4f} "
                f"mtm={losses['loss_mtm'].item():.4f} "
                f"con={losses['loss_contrastive'].item():.4f}"
            )

        if step % ckpt_every == 0 and step < steps:
            path = ckpt_dir / f"ckpt_step_{step}.pt"
            torch.save({"step": step, "model_state": model.state_dict()}, path)
            print(f"  saved checkpoint → {path}")

    final_path = ckpt_dir / "model_final.pt"
    torch.save(
        {
            "step": steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        final_path,
    )
    print(f"\nFinal model saved → {final_path}")
    return final_path


if __name__ == "__main__":
    train()
