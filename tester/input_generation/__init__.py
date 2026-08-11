"""输入生成、配置与物化的基础原语。"""

from .backend import (
    InputBackend,
    InputBackendCapabilityError,
    InputBackendPolicy,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    create_input_backend,
    resolve_input_backend_policy,
    resolve_input_backend_name,
)
from .tensor_config import TensorConfig, cached_numpy, get_cached_numpy_array
from .value import InputValue
from .value_generators import InputConfigRandomState, derive_input_seed

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
    "resolve_input_backend_policy",
    "resolve_input_backend_name",
    "cached_numpy",
    "get_cached_numpy_array",
    "InputConfigRandomState",
    "derive_input_seed",
]


def __getattr__(name):
    # 可变环境开关必须按访问时读取，不能在模块导入阶段固化旧值。
    if name == "USE_CACHED_NUMPY":
        from .tensor_config import numpy_auxiliary_cache_enabled

        return numpy_auxiliary_cache_enabled()
    raise AttributeError(name)
