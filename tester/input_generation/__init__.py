"""输入生成、配置与物化的基础原语。"""

from .backend import (
    InputBackend,
    InputBackendCapabilityError,
    InputBackendPolicy,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    resolve_input_backend_policy,
)
from .tensor_config import TensorConfig
from .value import InputValue
from .value_generators import InputConfigRandomState, derive_input_seed

__all__ = [
    "TensorConfig",
    "InputValue",
    "InputBackend",
    "InputBackendCapabilityError",
    "InputBackendPolicy",
    "NumPyInputBackend",
    "PaddleInputBackend",
    "TorchInputBackend",
    "resolve_input_backend_policy",
    "InputConfigRandomState",
    "derive_input_seed",
]
