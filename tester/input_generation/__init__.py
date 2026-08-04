"""输入生成、配置与物化的基础原语。"""

from .backend import (
    InputBackend,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    create_input_backend,
)
from .tensor_config import (
    USE_CACHED_NUMPY,
    TensorConfig,
    cached_numpy,
    get_cached_numpy_array,
)
from .value import InputValue

__all__ = [
    "USE_CACHED_NUMPY",
    "TensorConfig",
    "InputValue",
    "InputBackend",
    "NumPyInputBackend",
    "PaddleInputBackend",
    "TorchInputBackend",
    "create_input_backend",
    "cached_numpy",
    "get_cached_numpy_array",
]
