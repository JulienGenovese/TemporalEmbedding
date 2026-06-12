"""
End-to-end pre-training of :class:`EmbeddingModel` on a synthetic CSV dataset.

Reads ``data/transactions.csv`` (produced by :mod:`src.make_dataset`),
fits per-feature normalizers/vocabularies, and runs the joint
MTM + InfoNCE objective.  Default hyper-parameters (see
:class:`src.config.TrainingConfig`) are sized for a CPU smoke test
(~3-5 min on a laptop).

Usage:
    uv run python -m src.make_dataset    # one-off
    uv run python -m src.train
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from .hier_config import TrainingConfig
from .data import DataModule
from .encoder import (
    DatetimeFeature, FeatureSpec, HighCardCategoricalFeature, NumericFeature,
)
from .loss import PretrainLoss, info_nce_metrics
from .model import EmbeddingModel, count_parameters


class Trainer:
    
    """Pre-training loop encapsulated as a callable object.

    Owns the model, loss, dataloader and optimizer; ``__call__`` runs the
    full training schedule and returns the path of the final checkpoint.
    """

    def __init__(
        self,
        args: TrainingConfig,
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

        # Mixed precision: float16 autocast on CUDA only (CPU keeps full
        # precision so the smoke-test numerics stay reproducible).
        self.device_type = self.device.split(":")[0]
        self.use_amp = self.device_type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        logger.info("AMP autocast: {}", self.use_amp)

        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=args.lr_gamma,
        )

        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Checkpoint directory: {}", self.ckpt_dir.resolve())

        param_stats = count_parameters(self.model)
        logger.info("Model parameters: {:,} total", param_stats["total"])
        logger.debug("Parameter breakdown: {}", param_stats)
        logger.info(
            "Optimizer: AdamW(lr={}, weight_decay={}), "
            "ExponentialLR(gamma={}), grad_clip={}, mask_prob={}, "
            "contrastive_weight={}",
            args.lr, args.weight_decay, args.lr_gamma, args.grad_clip,
            args.mask_prob, args.contrastive_weight,
        )
        if self.val_loader is not None:
            logger.info(
                "Validation enabled: every {} epoch(s)", max(1, args.val_every),
            )
        else:
            logger.info("Validation disabled (no val_loader)")

        # Early stopping: monitors the validation loss. Only active when a
        # val_loader exists and val_every > 0 — otherwise there is no signal.
        self.early_stopping = (
            args.early_stopping_patience > 0
            and self.val_loader is not None
            and args.val_every > 0
        )
        if self.early_stopping:
            logger.info(
                "Early stopping enabled: patience={} val check(s), min_delta={}",
                args.early_stopping_patience, args.early_stopping_min_delta,
            )
        elif args.early_stopping_patience > 0:
            logger.warning(
                "Early stopping requested but inactive (needs a val_loader "
                "and val_every > 0)",
            )
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.epochs_no_improve = 0

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

        Numeric targets are normalised (clip → log1p → z-score) so the
        smooth-L1 term lives on the same scale as the encoder input and
        the categorical cross-entropy, instead of being dominated by the
        raw euro magnitudes of ``importo``.
        """
        pad_mask = batch.get("padding_mask")
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}

        for feat in self.model.features:
            if isinstance(feat, (HighCardCategoricalFeature, DatetimeFeature)):
                continue
            name = feat.name
            if isinstance(feat, NumericFeature) and feat.normalizer is not None:
                targets[name] = feat.normalizer(batch[name])
            else:
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
        with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
            output = self.model(batch)
            output["temperature"] = self.model.backbone.contrastive_head.temperature
            losses = self.loss(output, targets, mtm_mask, client_ids)
            loss = losses["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.grad_clip,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            metrics = info_nce_metrics(output["contrastive_z"], client_ids)

        self.step += 1
        entry = {
            "step": self.step,
            "loss": float(loss.item()),
            "loss_mtm": float(losses["loss_mtm"].item()),
            "loss_contrastive": float(losses["loss_contrastive"].item()),
            "infonce_acc": float(metrics["infonce_acc"].item()),
            "infonce_acc_random": float(metrics["infonce_acc_random"].item()),
            "infonce_lift": float(metrics["infonce_lift"].item()),
            "temperature": float(self.model.backbone.contrastive_head.temperature.item()),
            "grad_norm": float(grad_norm.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
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
                    "[TRAIN] epoch {} step {:>4} | loss={:.4f} mtm={:.4f} con={:.4f} "
                    "acc={:.3f} rand={:.3f} lift={:.3f}",
                    epoch, self.step, entry["loss"],
                    entry["loss_mtm"], entry["loss_contrastive"],
                    entry["infonce_acc"], entry["infonce_acc_random"],
                    entry["infonce_lift"],
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
        agg: dict[str, list[float]] = {
            "loss": [], "loss_mtm": [], "loss_contrastive": [],
            "infonce_acc": [], "infonce_acc_random": [], "infonce_lift": [],
        }
        with torch.no_grad():
            for batch, client_ids in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                client_ids = client_ids.to(self.device)
                targets, mtm_mask = self._build_mtm_targets(batch)

                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    output = self.model(batch)
                    output["temperature"] = self.model.backbone.contrastive_head.temperature
                    losses = self.loss(output, targets, mtm_mask, client_ids)
                metrics = info_nce_metrics(output["contrastive_z"], client_ids)

                agg["loss"].append(float(losses["loss"].item()))
                agg["loss_mtm"].append(float(losses["loss_mtm"].item()))
                agg["loss_contrastive"].append(float(losses["loss_contrastive"].item()))
                agg["infonce_acc"].append(float(metrics["infonce_acc"].item()))
                agg["infonce_acc_random"].append(float(metrics["infonce_acc_random"].item()))
                agg["infonce_lift"].append(float(metrics["infonce_lift"].item()))
        self.model.train()

        n = max(len(agg["loss"]), 1)
        return {
            "epoch": epoch,
            "n_batches": len(agg["loss"]),
            "loss": sum(agg["loss"]) / n,
            "loss_mtm": sum(agg["loss_mtm"]) / n,
            "loss_contrastive": sum(agg["loss_contrastive"]) / n,
            "infonce_acc": sum(agg["infonce_acc"]) / n,
            "infonce_acc_random": sum(agg["infonce_acc_random"]) / n,
            "infonce_lift": sum(agg["infonce_lift"]) / n,
        }

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Update best-checkpoint bookkeeping from ``val_loss``.

        Snapshots the model weights whenever the validation loss improves
        by more than ``early_stopping_min_delta``; otherwise increments the
        no-improvement counter.  Returns ``True`` when patience is exhausted
        and training should stop.
        """
        if val_loss < self.best_val_loss - self.args.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            logger.info(
                "New best val loss {:.4f} at epoch {}", val_loss, epoch,
            )
            return False

        self.epochs_no_improve += 1
        logger.info(
            "No val-loss improvement for {}/{} check(s) (best {:.4f} @ epoch {})",
            self.epochs_no_improve, self.args.early_stopping_patience,
            self.best_val_loss, self.best_epoch,
        )
        return self.epochs_no_improve >= self.args.early_stopping_patience

    def _train(self) -> Path:
        torch.manual_seed(self.args.seed)
        self.model.train()

        logger.info("Starting training: {} epoch(s)", self.args.epochs)
        for epoch in range(1, self.args.epochs + 1):
            self._epoch(epoch)
            self.scheduler.step()
            logger.info(
                "LR after epoch {}: {:.2e}",
                epoch, self.optimizer.param_groups[0]["lr"],
            )
            if self.args.val_every > 0 and epoch % self.args.val_every == 0:
                train_eval = self._eval_epoch(self.loader, epoch)
                self.train_eval_history.append(train_eval)
                if self.val_loader is not None:
                    val_eval = self._eval_epoch(self.val_loader, epoch)
                    self.val_history.append(val_eval)
                    logger.info(
                        "[EVAL] epoch {} | "
                        "TRAIN: loss={:.4f} mtm={:.4f} con={:.4f} acc={:.3f} "
                        "rand={:.3f} lift={:.3f} | "
                        "VALIDATION: loss={:.4f} mtm={:.4f} con={:.4f} acc={:.3f} "
                        "rand={:.3f} lift={:.3f}",
                        epoch,
                        train_eval["loss"], train_eval["loss_mtm"],
                        train_eval["loss_contrastive"], train_eval["infonce_acc"],
                        train_eval["infonce_acc_random"], train_eval["infonce_lift"],
                        val_eval["loss"], val_eval["loss_mtm"],
                        val_eval["loss_contrastive"], val_eval["infonce_acc"],
                        val_eval["infonce_acc_random"], val_eval["infonce_lift"],
                    )
                    if self.early_stopping and self._check_early_stopping(
                        val_eval["loss"], epoch,
                    ):
                        logger.info(
                            "Early stopping at epoch {} — no improvement for "
                            "{} validation check(s)",
                            epoch, self.args.early_stopping_patience,
                        )
                        break
                else:
                    logger.info(
                        "[EVAL] epoch {} | TRAIN: loss={:.4f} mtm={:.4f} "
                        "con={:.4f} acc={:.3f} rand={:.3f} lift={:.3f} "
                        "(no validation loader)",
                        epoch, train_eval["loss"], train_eval["loss_mtm"],
                        train_eval["loss_contrastive"], train_eval["infonce_acc"],
                        train_eval["infonce_acc_random"], train_eval["infonce_lift"],
                    )
        logger.info("Training complete after {} steps", self.step)

        if self.best_state is not None:
            logger.info(
                "Restoring best weights from epoch {} (val loss {:.4f})",
                self.best_epoch, self.best_val_loss,
            )
            self.model.load_state_dict(self.best_state)
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


def main() -> Path:
    args = TrainingConfig()
    logger.info("TrainingConfig: {}", args)

    data = DataModule(args)
    train_loader, val_loader, features = data()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_ckpt = device != "cpu"
    logger.info(
        "Building EmbeddingModel (device={}, gradient_checkpointing={})",
        device, use_ckpt,
    )
    model = EmbeddingModel(
        features=features, 
        pretrain=True, 
        use_gradient_checkpointing=use_ckpt,
        **asdict(args.model),
    )
    loss = PretrainLoss(contrastive_weight=args.contrastive_weight)

    trainer = Trainer(
        args=args, model=model, loss=loss,
        loader=train_loader, val_loader=val_loader,
    )
    return trainer()


if __name__ == "__main__":
    main()
