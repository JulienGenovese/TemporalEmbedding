"""
End-to-end pre-training of TransactionTransformer on a synthetic CSV dataset.

Reads ``data/transactions.csv`` (produced by :mod:`src.make_dataset`),
fits per-feature normalizers/vocabularies, and runs the joint
MTM + InfoNCE objective.  Default hyper-parameters (see
:class:`src.config.TrainingArgs`) are sized for a CPU smoke test
(~3-5 min on a laptop).

Usage:
    uv run python -m src.make_dataset    # one-off
    uv run python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from .config import TrainingArgs
from .data import (
    PairedClientBatchSampler, TransactionDataset,
    collate, fit_features, load_dataframe,
)
from .encoder import (
    CategoricalFeature, DatetimeFeature, FeatureSpec,
    HighCardCategoricalFeature, NumericFeature,
)
from .loss import combined_pretrain_loss, info_nce_accuracy
from .model import DEFAULT_FEATURES, TransactionTransformer, count_parameters


# ---------------------------------------------------------------------------
# MTM masking — copy targets, then randomly null out positions in the input
# ---------------------------------------------------------------------------

def build_mtm_targets(
    batch: dict[str, torch.Tensor],
    features: list[FeatureSpec],
    mask_prob: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Copy original values as targets, build per-field boolean masks, and
    zero out masked positions in ``batch`` so the encoder cannot see them.

    Padded positions are never masked.  Hash and datetime fields are not MTM
    targets (consistent with :class:`MTMHead`).
    """
    pad_mask = batch.get("padding_mask")
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}

    for feat in features:
        if isinstance(feat, (HighCardCategoricalFeature, DatetimeFeature)):
            continue
        name = feat.name
        targets[name] = batch[name].clone()

        m = torch.rand_like(batch[name], dtype=torch.float32) < mask_prob
        if pad_mask is not None:
            m = m & ~pad_mask
        masks[name] = m
        batch[name] = batch[name].masked_fill(m, 0)

    return targets, masks


# ---------------------------------------------------------------------------
# Dataset / loader construction
# ---------------------------------------------------------------------------

def build_dataloader(args: TrainingArgs) -> tuple[DataLoader, list[FeatureSpec]]:
    """Load the CSV, fit per-feature normalizers, and build the paired loader."""
    logger.info("Loading dataframe from {}", args.csv_path)
    df = load_dataframe(args.csv_path)
    logger.info(
        "Loaded {:,} rows / {} clients",
        len(df), df["client_id"].nunique(),
    )

    logger.info("Fitting per-feature normalizers/vocabularies")
    features = fit_features(df, DEFAULT_FEATURES)
    logger.debug(
        "Feature schema: {}",
        [(f.__class__.__name__, f.name) for f in features],
    )

    logger.info(
        "Building TransactionDataset (seq_len={}, windows_per_client={})",
        args.seq_len, args.windows_per_client,
    )
    dataset = TransactionDataset(
        df,
        client_col="client_id",
        timestamp_col="timestamp",
        feature_cols=[
            "importo", "saldo_post", "merchant", "mcc",
            "canale", "macro_tipo", "sotto_tipo", "divisa",
        ],
        seq_len=args.seq_len,
        windows_per_client=args.windows_per_client,
        seed=args.seed,
    )
    logger.info("Dataset built: {:,} windows total", len(dataset))

    sampler = PairedClientBatchSampler(
        dataset,
        clients_per_batch=args.clients_per_batch,
        windows_per_pair=args.windows_per_pair,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    logger.info(
        "Loader ready: {} batches/epoch, batch size {} "
        "({} clients × {} windows)",
        len(sampler), args.batch_size,
        args.clients_per_batch, args.windows_per_pair,
    )
    return loader, features


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    loader: DataLoader,
    features: list[FeatureSpec],
    args: TrainingArgs,
) -> Path:
    """Run pre-training on the given loader and save a final checkpoint."""
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device is : {}", device)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoint directory: {}", ckpt_dir.resolve())

    on_cpu = device == "cpu"
    use_ckpt = not on_cpu
    logger.info(
        "Building TransactionTransformer (device={}, gradient_checkpointing={})",
        device, use_ckpt,
    )
    model = TransactionTransformer(
        features=features,
        pretrain=True,
        use_gradient_checkpointing=use_ckpt,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    param_stats = count_parameters(model)
    logger.info("Model parameters: {:,} total", param_stats["total"])
    logger.debug("Parameter breakdown: {}", param_stats)
    logger.info(
        "Optimizer: AdamW(lr={}), grad_clip={}, mask_prob={}, "
        "contrastive_weight={}",
        args.lr, args.grad_clip, args.mask_prob, args.contrastive_weight,
    )

    model.train()
    step = 0
    history: list[dict] = []
    final_path = ckpt_dir / "model_final.pt"
    history_path = ckpt_dir / "history.json"

    logger.info("Starting training: {} epoch(s)", args.epochs)
    for epoch in range(1, args.epochs + 1):
        logger.info("=== Epoch {}/{} ===", epoch, args.epochs)
        epoch_losses: list[float] = []
        for batch, client_ids in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            client_ids = client_ids.to(device)

            targets, mtm_mask = build_mtm_targets(batch, features, args.mask_prob)

            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            output["temperature"] = model.contrastive_head.temperature

            losses = combined_pretrain_loss(
                output, targets, mtm_mask, client_ids, args.contrastive_weight,
            )
            loss = losses["loss"]
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                acc = info_nce_accuracy(output["contrastive_z"], client_ids)

            step += 1
            entry = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.item()),
                "loss_mtm": float(losses["loss_mtm"].item()),
                "loss_contrastive": float(losses["loss_contrastive"].item()),
                "infonce_acc": float(acc.item()),
                "temperature": float(model.contrastive_head.temperature.item()),
                "grad_norm": float(grad_norm.item()),
                "mtm_breakdown": {
                    k: float(v.item()) for k, v in losses["mtm_breakdown"].items()
                },
            }
            history.append(entry)
            epoch_losses.append(entry["loss"])

            if step % args.log_every == 0 or step == 1:
                logger.info(
                    "epoch {} step {:>4} | loss={:.4f} mtm={:.4f} con={:.4f} "
                    "acc={:.3f} |g|={:.2f}",
                    epoch, step, loss.item(),
                    losses["loss_mtm"].item(), losses["loss_contrastive"].item(),
                    acc.item(), grad_norm.item(),
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(
            "Epoch {} done — avg loss={:.4f} over {} steps",
            epoch, avg_loss, len(epoch_losses),
        )

    logger.info("Training complete after {} steps", step)
    logger.info("Saving final checkpoint → {}", final_path)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        final_path,
    )
    logger.info("Saving history → {}", history_path)
    history_path.write_text(json.dumps(history, indent=2))
    logger.success("All artifacts written under {}", ckpt_dir.resolve())
    return final_path


if __name__ == "__main__":
    args = TrainingArgs()
    logger.info("TrainingArgs: {}", args)
    loader, features = build_dataloader(args)
    train(loader, features, args)
