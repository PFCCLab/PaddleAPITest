"""输入生成、配置与物化的基础原语。"""

from typing import TYPE_CHECKING

from .backend import (
    InputBackend,
    InputBackendCapabilityError,
    InputBackendPolicy,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    create_input_backend,
    resolve_input_backend_name,
    resolve_input_backend_policy,
)
from .value import InputValue

if TYPE_CHECKING:
    from .tensor_config import TensorConfig, cached_numpy, get_cached_numpy_array

__all__ = [
    "USE_CACHED_NUMPY",
    "TensorConfig",
    "InputValue",
    "InputBackend",
    "InputBackendCapabilityError",
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
    if name in {"TensorConfig", "cached_numpy", "get_cached_numpy_array"}:
        # engineV4 的 spawn 子进程必须先完成 GPU 隔离，再由 TensorConfig 加载 Paddle。
        from . import tensor_config

        return getattr(tensor_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
