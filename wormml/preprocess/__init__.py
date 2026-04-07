"""
Preprocessing subpackage.

Each camera has distinct preprocessing needs that are encoded in per-camera
PreprocessConfig objects.  A unified ``preprocess_dataset`` entry-point
accepts a config and applies the correct pipeline automatically.
"""

from wormml.preprocess.base import preprocess_dataset
from wormml.preprocess.configs import (
    OGPreprocessConfig,
    TauPreprocessConfig,
    LBPreprocessConfig,
    UVAPreprocessConfig,
    get_config,
    CAMERA_CONFIGS,
)

__all__ = [
    "preprocess_dataset",
    "OGPreprocessConfig",
    "TauPreprocessConfig",
    "LBPreprocessConfig",
    "UVAPreprocessConfig",
    "get_config",
    "CAMERA_CONFIGS",
]
