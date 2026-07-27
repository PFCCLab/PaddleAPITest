"""Test input generation, configuration, and materialization primitives."""

from .telemetry import LegacyGenerationEvent, capture_legacy_generation
from .tensor_config import (
    USE_CACHED_NUMPY,
    TensorConfig,
    cached_numpy,
    get_cached_numpy_array,
)

__all__ = [
    "USE_CACHED_NUMPY",
    "TensorConfig",
    "cached_numpy",
    "get_cached_numpy_array",
    "LegacyGenerationEvent",
    "capture_legacy_generation",
]
