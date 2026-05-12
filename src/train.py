"""
End-to-end pre-training of TransactionTransformer on a synthetic CSV dataset.

Reads ``data/transactions.csv`` (produced by :mod:`src.make_dataset`),
fits per-feature normalizers/vocabularies, and runs the joint
MTM + InfoNCE objective.  Default hyper-parameters are sized for a
CPU smoke test (~3-5 min on a laptop).

Usage:
    uv run python -m src.make_dataset    # one-off
    uv run python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

def build_dataloader(
    csv_path: str | Path,
    *,
    seq_len: int,
    windows_per_client: int,
    clients_per_batch: int,
    windows_per_pair: int,
    seed: int,
) -> tuple[DataLoader, list[FeatureSpec]]:
    """Load the CSV, fit per-feature normalizers, and build the paired loader."""
    df = load_dataframe(csv_path)
    features = fit_features(df, DEFAULT_FEATURES)
    dataset = TransactionDataset(
        df,
        client_col="client_id",
        timestamp_col="timestamp",
        feature_cols=[
            "importo", "saldo_post", "merchant", "mcc",
            "canale", "macro_tipo", "sotto_tipo", "divisa",
        ],
        seq_len=seq_len,
        windows_per_client=windows_per_client,
        seed=seed,
    )
    sampler = PairedClientBatchSampler(
        dataset,
        clients_per_batch=clients_per_batch,
        windows_per_pair=windows_per_pair,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    print(f"Rows: {len(df):,}  clients: {df['client_id'].nunique()}")
    print(f"Batches/epoch: {len(sampler)}  batch size: {clients_per_batch * windows_per_pair}")
    return loader, features


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    loader: DataLoader,
    features: list[FeatureSpec],
    *,
    epochs: int = 2,
    mask_prob: float = 0.15,
    contrastive_weight: float = 0.5,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    log_every: int = 5,
    ckpt_dir: str | Path = "checkpoints",
    device: str | None = None,
    seed: int = 0,
) -> Path:
    """Run pre-training on the given loader and save a final checkpoint."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    on_cpu = device == "cpu"
    model = TransactionTransformer(
        features=features,
        pretrain=True,
        use_gradient_checkpointing=not on_cpu,  # checkpointing slows CPU down
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Device: {device}")
    print(f"Params: {count_parameters(model)['total']:,}")

    model.train()
    step = 0
    history: list[dict] = []
    final_path = ckpt_dir / "model_final.pt"
    history_path = ckpt_dir / "history.json"
    for epoch in range(1, epochs + 1):
        for batch, client_ids in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            client_ids = client_ids.to(device)

            targets, mtm_mask = build_mtm_targets(batch, features, mask_prob)

            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            output["temperature"] = model.contrastive_head.temperature

            losses = combined_pretrain_loss(
                output, targets, mtm_mask, client_ids, contrastive_weight,
            )
            loss = losses["loss"]
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

            if step % log_every == 0 or step == 1:
                print(
                    f"epoch {epoch} step {step:>4} | loss={loss.item():.4f} "
                    f"mtm={losses['loss_mtm'].item():.4f} "
                    f"con={losses['loss_contrastive'].item():.4f} "
                    f"acc={acc.item():.3f} "
                    f"|g|={grad_norm.item():.2f}"
                )

    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        final_path,
    )
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nFinal model saved → {final_path}")
    print(f"History saved      → {history_path}")
    return final_path


if __name__ == "__main__":
    SEED = 0
    loader, features = build_dataloader(
        Path("data") / "transactions.csv",
        seq_len=32,
        windows_per_client=4,
        clients_per_batch=8,
        windows_per_pair=2,
        seed=SEED,
    )
    train(loader, features, seed=SEED)
