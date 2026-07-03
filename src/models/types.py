from enum import Enum
class ModelType(str, Enum):
    """Supported model variants for train/pred commands."""

    HIER = "hier"
