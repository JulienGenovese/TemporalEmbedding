"""Pre-training entrypoint for the simplified (concat-fusion) variant.

Same schedule as ``hier_transformer.train`` — DataModule, Trainer and
PretrainLoss are reused as-is; only the model class and the artifact
directory change.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger

from src.models.hier_simplified.config import load_config
from src.models.hier_simplified.features import SIMPLIFIED_FEATURES
from src.models.hier_simplified.model import SimplifiedEmbeddingModel
from src.models.hier_transformer.data import DataModule
from src.models.hier_transformer.loss import PretrainLoss
from src.models.hier_transformer.train import Trainer


def main() -> Path:
    args = load_config()
    logger.info("HierSimplified config: {}", args)

    data = DataModule(args, features=SIMPLIFIED_FEATURES)
    train_loader, val_loader, features = data()

    device = args.runtime.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_ckpt = device != "cpu"
    logger.info(
        "Building SimplifiedEmbeddingModel (device={}, gradient_checkpointing={})",
        device, use_ckpt,
    )
    model = SimplifiedEmbeddingModel(
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
