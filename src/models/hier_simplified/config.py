"""Config for the simplified pipeline: hier_transformer's, with its own output paths.

Hyper-parameters, data pipeline and training settings are shared with
``model.hier_transformer.*`` so the two variants stay directly
comparable; only the artifact/prediction destinations differ, read from
``[model.hier_simplified.paths]``.
"""

from dataclasses import replace
from pathlib import Path

from src.config import config
from src.models.hier_transformer.hier_config import HierTransformerConfig

_PATHS_SECTION = "model.hier_simplified.paths"


def load_config() -> HierTransformerConfig:
    args = HierTransformerConfig()
    args.paths = replace(
        args.paths,
        model_output_dir=config.get(
            _PATHS_SECTION, "model_output_dir", value_type=Path,
        ),
        pred_output_path=config.get(
            _PATHS_SECTION, "pred_output_path", value_type=Path,
        ),
    )
    return args
