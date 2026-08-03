"""输入生成、配置与物化的基础原语。"""

from .input_backend import (
    InputBackend,
    NumpyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    create_input_backend,
)
from .input_data import InputData
from .tensor_config import (
    USE_CACHED_NUMPY,
    TensorConfig,
    cached_numpy,
    get_cached_numpy_array,
)

__all__ = [
    "USE_CACHED_NUMPY",
    "TensorConfig",
    "InputData",
    "InputBackend",
    "NumpyInputBackend",
    "PaddleInputBackend",
    "TorchInputBackend",
    "create_input_backend",
    "cached_numpy",
    "get_cached_numpy_array",
]
