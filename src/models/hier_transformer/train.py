"""
End-to-end pre-training of :class:`EmbeddingModel` on a synthetic CSV dataset.

Reads ``data/transactions.csv`` (produced by :mod:`src.make_dataset`),
fits per-feature normalizers/vocabularies, and runs the joint
MTM + InfoNCE objective. Default settings (see
:class:`src.models.hier_transformer.hier_config.HierTransformerConfig`) are sized
for a CPU smoke test
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

from src.models.hier_transformer.artifacts import (
    HISTORY_FILENAME,
    MODEL_CHECKPOINT_FILENAME,
    RUN_METADATA_FILENAME,
    TRAIN_EVAL_HISTORY_FILENAME,
    VAL_HISTORY_FILENAME,
    create_training_run_dir,
    replace_latest_artifact_dirs,
)
from src.models.hier_transformer.data import DataModule
from src.models.hier_transformer.features import (
    DatetimeFeature,
    FeatureSpec,
    HighCardCategoricalFeature,
    NumericFeature,
    mtm_mask_key,
)
from src.models.hier_transformer.hier_config import HierTransformerConfig
from src.models.hier_transformer.loss import PretrainLoss, info_nce_metrics
from src.models.hier_transformer.model import EmbeddingModel, count_parameters


EVAL_METRICS = (
    "loss",
    "loss_mtm",
    "loss_contrastive",
    "infonce_acc",
    "infonce_acc_random",
    "infonce_lift",
)


def build_mtm_targets(
    batch: dict[str, torch.Tensor],
    features: list[FeatureSpec],
    mask_prob: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build MTM targets/masks, zero masked inputs and publish the masks.

    Masked positions are zeroed in ``batch[name]`` (so no encoder can see
    the original value) and the boolean mask is stored under
    ``mtm_mask_key(name)`` so the feature's ``encode`` substitutes its
    learned [MASK] representation instead of collapsing onto padding.
    """
    pad_mask = batch.get("padding_mask")
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}

    for feat in features:
        if isinstance(feat, (HighCardCategoricalFeature, DatetimeFeature)):
            continue
        name = feat.name
        if isinstance(feat, NumericFeature) and feat.normalizer is not None:
            targets[name] = feat.normalizer(batch[name])
        else:
            targets[name] = batch[name].clone()

        mask = torch.rand_like(batch[name], dtype=torch.float32) < mask_prob
        if pad_mask is not None:
            mask = mask & ~pad_mask
        masks[name] = mask
        batch[name] = batch[name].masked_fill(mask, 0)
        batch[mtm_mask_key(name)] = mask

    return targets, masks


def mean_metrics(
    metrics: dict[str, list[float]],
    epoch: int,
) -> dict[str, float]:
    """Average accumulated scalar metrics for one epoch."""
    n_batches = len(metrics["loss"])
    denom = max(n_batches, 1)
    return {
        "epoch": epoch,
        "n_batches": n_batches,
        **{name: sum(values) / denom for name, values in metrics.items()},
    }


class Trainer:
    """Pre-training loop encapsulated as a callable object.

    Owns the model, loss, dataloader and optimizer; ``__call__`` runs the
    full training schedule and returns the path of the final checkpoint.
    """

    def __init__(
        self,
        args: HierTransformerConfig,
        model: EmbeddingModel,
        loss: PretrainLoss,
        loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.args = args
        self.path_config = args.paths
        self.training_config = args.training
        self.runtime_config = args.runtime
        self.loader = loader
        self.val_loader = val_loader
        self.device = self.runtime_config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
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
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.training_config.lr_gamma,
        )

        self.artifacts_root = Path(self.path_config.model_output_dir)
        self.dataset_name, self.run_id, self.model_output_dir = create_training_run_dir(
            artifacts_root=self.artifacts_root,
            train_path=self.path_config.train_path,
        )
        logger.info("Model artifacts root: {}", self.artifacts_root.resolve())
        logger.info("Training artifact directory: {}", self.model_output_dir.resolve())

        param_stats = count_parameters(self.model)
        logger.info("Model parameters: {:,} total", param_stats["total"])
        logger.debug("Parameter breakdown: {}", param_stats)
        logger.info(
            "Optimizer: AdamW(lr={}, weight_decay={}), "
            "ExponentialLR(gamma={}), grad_clip={}, mask_prob={}, "
            "contrastive_weight={}",
            self.training_config.lr,
            self.training_config.weight_decay,
            self.training_config.lr_gamma,
            self.training_config.grad_clip,
            self.training_config.mask_prob,
            self.training_config.contrastive_weight,
        )
        if self.val_loader is not None:
            logger.info(
                "Validation enabled: every {} epoch(s)",
                max(1, self.training_config.val_every),
            )
        else:
            logger.info("Validation disabled (no val_loader)")

        # Early stopping: monitors the validation loss. Only active when a
        # val_loader exists and val_every > 0 — otherwise there is no signal.
        self.early_stopping = (
            self.training_config.early_stopping_patience > 0
            and self.val_loader is not None
            and self.training_config.val_every > 0
        )
        if self.early_stopping:
            logger.info(
                "Early stopping enabled: patience={} val check(s), min_delta={}",
                self.training_config.early_stopping_patience,
                self.training_config.early_stopping_min_delta,
            )
        elif self.training_config.early_stopping_patience > 0:
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

    def _forward_and_loss(
        self,
        batch: dict[str, torch.Tensor],
        client_ids: torch.Tensor,
    ) -> tuple[dict, dict, torch.Tensor]:
        """Shared forward path used by both training and evaluation.

        Moves the batch to the device, builds the MTM targets/masks (which
        also zeroes out masked positions in ``batch``), runs the model under
        autocast and computes the combined loss.  Returns ``(output, losses,
        client_ids)`` — the caller decides whether to backprop.
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        client_ids = client_ids.to(self.device)

        targets, mtm_mask = build_mtm_targets(
            batch,
            self.model.features,
            self.training_config.mask_prob,
        )

        with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
            output = self.model(batch)
            losses = self.loss(output, targets, mtm_mask, client_ids)

        return output, losses, client_ids

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        client_ids: torch.Tensor,
    ) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        output, losses, client_ids = self._forward_and_loss(batch, client_ids)
        loss = losses["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.training_config.grad_clip,
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
            "time_alpha": float(output["time_alpha"].item()),
            "grad_norm": float(grad_norm.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "mtm_breakdown": {
                k: float(v.item()) for k, v in losses["mtm_breakdown"].items()
            },
        }
        return entry

    def _epoch(self, epoch: int) -> float:
        logger.info("=== Epoch {}/{} ===", epoch, self.training_config.epochs)
        epoch_losses: list[float] = []
        for batch, client_ids in self.loader:
            entry = self._step(batch, client_ids)
            entry["epoch"] = epoch
            self.history.append(entry)
            epoch_losses.append(entry["loss"])

            if self.step % self.training_config.log_every == 0 or self.step == 1:
                logger.info(
                    "[TRAIN] epoch {} step {:>4} | loss={:.4f} mtm={:.4f} con={:.4f} "
                    "time_alpha={:.4f} acc={:.3f} rand={:.3f} lift={:.3f}",
                    epoch, self.step, entry["loss"],
                    entry["loss_mtm"], entry["loss_contrastive"],
                    entry["time_alpha"],
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
        agg: dict[str, list[float]] = {name: [] for name in EVAL_METRICS}
        with torch.no_grad():
            for batch, client_ids in loader:
                output, losses, client_ids = self._forward_and_loss(batch, client_ids)
                metrics = info_nce_metrics(output["contrastive_z"], client_ids)

                agg["loss"].append(float(losses["loss"].item()))
                agg["loss_mtm"].append(float(losses["loss_mtm"].item()))
                agg["loss_contrastive"].append(float(losses["loss_contrastive"].item()))
                agg["infonce_acc"].append(float(metrics["infonce_acc"].item()))
                agg["infonce_acc_random"].append(float(metrics["infonce_acc_random"].item()))
                agg["infonce_lift"].append(float(metrics["infonce_lift"].item()))
        self.model.train()

        return mean_metrics(agg, epoch)

    @staticmethod
    def _metric_values(entry: dict[str, float]) -> tuple[float, ...]:
        return (
            entry["loss"],
            entry["loss_mtm"],
            entry["loss_contrastive"],
            entry["infonce_acc"],
            entry["infonce_acc_random"],
            entry["infonce_lift"],
        )

    def _log_eval(
        self,
        epoch: int,
        train_eval: dict[str, float],
        val_eval: dict[str, float] | None,
    ) -> None:
        if val_eval is None:
            logger.info(
                "[EVAL] epoch {} | TRAIN: loss={:.4f} mtm={:.4f} "
                "con={:.4f} acc={:.3f} rand={:.3f} lift={:.3f} "
                "(no validation loader)",
                epoch,
                *self._metric_values(train_eval),
            )
            return

        logger.info(
            "[EVAL] epoch {} | "
            "TRAIN: loss={:.4f} mtm={:.4f} con={:.4f} "
            "acc={:.3f} rand={:.3f} lift={:.3f} | "
            "VALIDATION: loss={:.4f} mtm={:.4f} con={:.4f} acc={:.3f} "
            "rand={:.3f} lift={:.3f}",
            epoch,
            *self._metric_values(train_eval),
            *self._metric_values(val_eval),
        )

    def _run_eval(self, epoch: int) -> bool:
        train_eval = self._eval_epoch(self.loader, epoch)
        self.train_eval_history.append(train_eval)

        val_eval = None
        if self.val_loader is not None:
            val_eval = self._eval_epoch(self.val_loader, epoch)
            self.val_history.append(val_eval)

        self._log_eval(epoch, train_eval, val_eval)
        if (
            val_eval is not None
            and self.early_stopping
            and self._check_early_stopping(val_eval["loss"], epoch)
        ):
            logger.info(
                "Early stopping at epoch {} — no improvement for "
                "{} validation check(s)",
                epoch,
                self.training_config.early_stopping_patience,
            )
            return True
        return False

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Update best-checkpoint bookkeeping from ``val_loss``.

        Snapshots the model weights whenever the validation loss improves
        by more than ``early_stopping_min_delta``; otherwise increments the
        no-improvement counter.  Returns ``True`` when patience is exhausted
        and training should stop.
        """
        if (
            val_loss
            < self.best_val_loss - self.training_config.early_stopping_min_delta
        ):
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
            self.epochs_no_improve,
            self.training_config.early_stopping_patience,
            self.best_val_loss, self.best_epoch,
        )
        return self.epochs_no_improve >= self.training_config.early_stopping_patience

    def _train(self) -> Path:
        torch.manual_seed(self.runtime_config.seed)
        self.model.train()

        logger.info("Starting training: {} epoch(s)", self.training_config.epochs)
        for epoch in range(1, self.training_config.epochs + 1):
            self._epoch(epoch)
            self.scheduler.step()
            logger.info(
                "LR after epoch {}: {:.2e}",
                epoch, self.optimizer.param_groups[0]["lr"],
            )
            if (
                self.training_config.val_every > 0
                and epoch % self.training_config.val_every == 0
            ):
                if self._run_eval(epoch):
                    break
        logger.info("Training complete after {} steps", self.step)

        if self.best_state is not None:
            logger.info(
                "Restoring best weights from epoch {} (val loss {:.4f})",
                self.best_epoch, self.best_val_loss,
            )
            self.model.load_state_dict(self.best_state)
        return self._save()

    def _save(self) -> Path:
        final_path = self.model_output_dir / MODEL_CHECKPOINT_FILENAME
        history_path = self.model_output_dir / HISTORY_FILENAME
        train_eval_path = self.model_output_dir / TRAIN_EVAL_HISTORY_FILENAME
        val_history_path = self.model_output_dir / VAL_HISTORY_FILENAME
        metadata_path = self.model_output_dir / RUN_METADATA_FILENAME
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
        metadata_path.write_text(json.dumps(
            {
                "dataset": self.dataset_name,
                "run_id": self.run_id,
                "train_path": str(self.path_config.train_path),
                "artifact_dir": str(self.model_output_dir),
            },
            indent=2,
        ))
        replace_latest_artifact_dirs(
            artifacts_root=self.artifacts_root,
            dataset=self.dataset_name,
            run_dir=self.model_output_dir,
        )
        logger.success("All artifacts written under {}", self.model_output_dir.resolve())
        return final_path


def main() -> Path:
    args = HierTransformerConfig()
    logger.info("HierTransformerConfig: {}", args)

    data = DataModule(args)
    train_loader, val_loader, features = data()

    device = args.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    loss = PretrainLoss(contrastive_weight=args.training.contrastive_weight)

    trainer = Trainer(
        args=args, model=model, loss=loss,
        loader=train_loader, val_loader=val_loader,
    )
    return trainer()