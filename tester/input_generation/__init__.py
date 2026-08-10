"""输入生成、配置与物化的基础原语。"""

from .backend import (
    InputBackend,
    InputBackendPolicy,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    create_input_backend,
    resolve_input_backend_name,
    resolve_input_backend_policy,
)
from .tensor_config import (
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
    "InputBackendPolicy",
    "NumPyInputBackend",
    "PaddleInputBackend",
    "TorchInputBackend",
    "create_input_backend",
    "resolve_input_backend_name",
    "resolve_input_backend_policy",
    "cached_numpy",
    "get_cached_numpy_array",
]


def __getattr__(name):
    if name == "USE_CACHED_NUMPY":
        from ..runtime_config import numpy_cache_enabled

        return numpy_cache_enabled()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
