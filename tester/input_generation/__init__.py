"""输入生成、配置与物化的基础原语。"""

from .generation_backend import (
    GenerationBackend,
    NumpyGenerationBackend,
    TorchGenerationBackend,
    create_generation_backend,
)
from .input_values import InputValue
from .tensor_config import (
    USE_CACHED_NUMPY,
    TensorConfig,
    cached_numpy,
    get_cached_numpy_array,
)

__all__ = [
    "USE_CACHED_NUMPY",
    "TensorConfig",
    "InputValue",
    "GenerationBackend",
    "NumpyGenerationBackend",
    "TorchGenerationBackend",
    "create_generation_backend",
    "cached_numpy",
    "get_cached_numpy_array",
]
