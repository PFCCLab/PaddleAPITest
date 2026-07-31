"""输入生成、配置与物化的基础原语。"""

from .logical_values import TensorPayload
from .tensor_config import (
    USE_CACHED_NUMPY,
    TensorConfig,
    cached_numpy,
    get_cached_numpy_array,
)

__all__ = [
    "USE_CACHED_NUMPY",
    "TensorConfig",
    "TensorPayload",
    "cached_numpy",
    "get_cached_numpy_array",
]
