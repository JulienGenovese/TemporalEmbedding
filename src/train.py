"""
End-to-end pre-training of :class:`EmbeddingModel` on a synthetic CSV dataset.

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
from .data import DataModule
from .encoder import (
    DatetimeFeature, FeatureSpec, HighCardCategoricalFeature,
)
from .loss import PretrainLoss, info_nce_accuracy
from .model import EmbeddingModel, count_parameters


class Trainer:
    """Pre-training loop encapsulated as a callable object.

    Owns the model, loss, dataloader and optimizer; ``__call__`` runs the
    full training schedule and returns the path of the final checkpoint.
    """

    def __init__(
        self,
        args: TrainingArgs,
        model: EmbeddingModel,
        loss: PretrainLoss,
        loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.args = args
        self.loader = loader
        self.val_loader = val_loader
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device is : {}", self.device)

        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoint directory: {}", self.ckpt_dir.resolve())

        param_stats = count_parameters(self.model)
        logger.info("Model parameters: {:,} total", param_stats["total"])
        logger.debug("Parameter breakdown: {}", param_stats)
        logger.info(
            "Optimizer: AdamW(lr={}), grad_clip={}, mask_prob={}, "
            "contrastive_weight={}",
            args.lr, args.grad_clip, args.mask_prob, args.contrastive_weight,
        )
        if self.val_loader is not None:
            logger.info(
                "Validation enabled: every {} epoch(s)", max(1, args.val_every),
            )
        else:
            logger.info("Validation disabled (no val_loader)")

        self.history: list[dict] = []
        self.train_eval_history: list[dict] = []
        self.val_history: list[dict] = []
        self.step = 0

    def __call__(self) -> Path:
        return self._train()

    def _build_mtm_targets(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Copy original values as targets, build per-field boolean masks,
        and zero out masked positions in ``batch`` so the encoder cannot
        see them.  Padded positions are never masked.  Hash and datetime
        fields are not MTM targets (consistent with :class:`MTMHead`).
        """
        pad_mask = batch.get("padding_mask")
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}

        for feat in self.model.features:
            if isinstance(feat, (HighCardCategoricalFeature, DatetimeFeature)):
                continue
            name = feat.name
            targets[name] = batch[name].clone()

            m = torch.rand_like(batch[name], dtype=torch.float32) < self.args.mask_prob
            if pad_mask is not None:
                m = m & ~pad_mask
            masks[name] = m
            batch[name] = batch[name].masked_fill(m, 0)

        return targets, masks

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        client_ids: torch.Tensor,
    ) -> dict:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        client_ids = client_ids.to(self.device)

        targets, mtm_mask = self._build_mtm_targets(batch)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch)
        output["temperature"] = self.model.backbone.contrastive_head.temperature

        losses = self.loss(output, targets, mtm_mask, client_ids)
        loss = losses["loss"]
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.grad_clip,
        )
        self.optimizer.step()

        with torch.no_grad():
            acc = info_nce_accuracy(output["contrastive_z"], client_ids)

        self.step += 1
        entry = {
            "step": self.step,
            "loss": float(loss.item()),
            "loss_mtm": float(losses["loss_mtm"].item()),
            "loss_contrastive": float(losses["loss_contrastive"].item()),
            "infonce_acc": float(acc.item()),
            "temperature": float(self.model.backbone.contrastive_head.temperature.item()),
            "grad_norm": float(grad_norm.item()),
            "mtm_breakdown": {
                k: float(v.item()) for k, v in losses["mtm_breakdown"].items()
            },
        }
        return entry

    def _epoch(self, epoch: int) -> float:
        logger.info("=== Epoch {}/{} ===", epoch, self.args.epochs)
        epoch_losses: list[float] = []
        for batch, client_ids in self.loader:
            entry = self._step(batch, client_ids)
            entry["epoch"] = epoch
            self.history.append(entry)
            epoch_losses.append(entry["loss"])

            if self.step % self.args.log_every == 0 or self.step == 1:
                logger.info(
                    "epoch {} step {:>4} | loss={:.4f} mtm={:.4f} con={:.4f} "
                    "acc={:.3f} |g|={:.2f}",
                    epoch, self.step, entry["loss"],
                    entry["loss_mtm"], entry["loss_contrastive"],
                    entry["infonce_acc"], entry["grad_norm"],
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(
            "Epoch {} done — avg loss={:.4f} over {} steps",
            epoch, avg_loss, len(epoch_losses),
        )
        return avg_loss

    def _eval_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """One no-grad pass over ``loader``.

        Uses the same MTM masking + combined loss as training so the
        numbers are directly comparable to those produced by a parallel
        eval over the train loader.  Returns an aggregate (mean) dict
        for the epoch.
        """
        self.model.eval()
        agg = {"loss": [], "loss_mtm": [], "loss_contrastive": [], "infonce_acc": []}
        with torch.no_grad():
            for batch, client_ids in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                client_ids = client_ids.to(self.device)
                targets, mtm_mask = self._build_mtm_targets(batch)

                output = self.model(batch)
                output["temperature"] = self.model.backbone.contrastive_head.temperature
                losses = self.loss(output, targets, mtm_mask, client_ids)
                acc = info_nce_accuracy(output["contrastive_z"], client_ids)

                agg["loss"].append(float(losses["loss"].item()))
                agg["loss_mtm"].append(float(losses["loss_mtm"].item()))
                agg["loss_contrastive"].append(float(losses["loss_contrastive"].item()))
                agg["infonce_acc"].append(float(acc.item()))
        self.model.train()

        n = max(len(agg["loss"]), 1)
        return {
            "epoch": epoch,
            "n_batches": len(agg["loss"]),
            "loss": sum(agg["loss"]) / n,
            "loss_mtm": sum(agg["loss_mtm"]) / n,
            "loss_contrastive": sum(agg["loss_contrastive"]) / n,
            "infonce_acc": sum(agg["infonce_acc"]) / n,
        }

    def _train(self) -> Path:
        torch.manual_seed(self.args.seed)
        self.model.train()

        logger.info("Starting training: {} epoch(s)", self.args.epochs)
        for epoch in range(1, self.args.epochs + 1):
            self._epoch(epoch)
            if self.args.val_every > 0 and epoch % self.args.val_every == 0:
                train_eval = self._eval_epoch(self.loader, epoch)
                self.train_eval_history.append(train_eval)
                if self.val_loader is not None:
                    val_eval = self._eval_epoch(self.val_loader, epoch)
                    self.val_history.append(val_eval)
                    logger.info(
                        "EVAL epoch {} | "
                        "train: loss={:.4f} mtm={:.4f} con={:.4f} acc={:.3f} | "
                        "val:   loss={:.4f} mtm={:.4f} con={:.4f} acc={:.3f}",
                        epoch,
                        train_eval["loss"], train_eval["loss_mtm"],
                        train_eval["loss_contrastive"], train_eval["infonce_acc"],
                        val_eval["loss"], val_eval["loss_mtm"],
                        val_eval["loss_contrastive"], val_eval["infonce_acc"],
                    )
                else:
                    logger.info(
                        "EVAL epoch {} | train: loss={:.4f} mtm={:.4f} "
                        "con={:.4f} acc={:.3f} (no val loader)",
                        epoch, train_eval["loss"], train_eval["loss_mtm"],
                        train_eval["loss_contrastive"], train_eval["infonce_acc"],
                    )
        logger.info("Training complete after {} steps", self.step)
        return self._save()

    def _save(self) -> Path:
        final_path = self.ckpt_dir / "model_final.pt"
        history_path = self.ckpt_dir / "history.json"
        train_eval_path = self.ckpt_dir / "train_eval_history.json"
        val_history_path = self.ckpt_dir / "val_history.json"
        logger.info("Saving final checkpoint → {}", final_path)
        self.model.save(
            final_path,
            optimizer_state=self.optimizer.state_dict(),
            step=self.step,
        )
        logger.info("Saving history → {}", history_path)
        history_path.write_text(json.dumps(self.history, indent=2))
        if self.train_eval_history:
            logger.info("Saving train_eval history → {}", train_eval_path)
            train_eval_path.write_text(json.dumps(self.train_eval_history, indent=2))
        if self.val_history:
            logger.info("Saving val history → {}", val_history_path)
            val_history_path.write_text(json.dumps(self.val_history, indent=2))
        logger.success("All artifacts written under {}", self.ckpt_dir.resolve())
        return final_path


if __name__ == "__main__":
    args = TrainingArgs()
    logger.info("TrainingArgs: {}", args)

    data = DataModule(args)
    train_loader, val_loader, features = data()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_ckpt = device != "cpu"
    logger.info(
        "Building EmbeddingModel (device={}, gradient_checkpointing={})",
        device, use_ckpt,
    )
    model = EmbeddingModel(
        features=features, pretrain=True, use_gradient_checkpointing=use_ckpt,
    )
    loss = PretrainLoss(contrastive_weight=args.contrastive_weight)

    trainer = Trainer(
        args=args, model=model, loss=loss,
        loader=train_loader, val_loader=val_loader,
    )
    trainer()
