"""输入生成、配置与物化的基础原语。"""

from .backend import (
    InputBackend,
    InputBackendCapabilityError,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
)
from .backend_runtime import (
    InputBackendPolicy,
    InputBackendRuntime,
    clear_input_backend_runtime,
    create_input_backend,
    generate_output_grad,
    prepare_input_backend,
    resolve_input_backend_policy,
)
from .materialization import (
    MaterializationPlan,
    build_materialization_plan,
    generated_value_nbytes,
    iter_unique_tensor_configs,
    tensor_config_tree_nbytes,
    tensor_config_tree_numel,
)
from .tensor_config import TensorConfig
from .value_generators import InputConfigRandomState, derive_input_seed
from .values import InputTensorPath, InputTensorSpec, InputValue

__all__ = [
    "TensorConfig",
    "InputTensorPath",
    "InputTensorSpec",
    "InputValue",
    "MaterializationPlan",
    "build_materialization_plan",
    "generated_value_nbytes",
    "iter_unique_tensor_configs",
    "tensor_config_tree_nbytes",
    "tensor_config_tree_numel",
    "InputBackend",
    "InputBackendCapabilityError",
    "InputBackendPolicy",
    "InputBackendRuntime",
    "clear_input_backend_runtime",
    "NumPyInputBackend",
    "PaddleInputBackend",
    "TorchInputBackend",
    "create_input_backend",
    "generate_output_grad",
    "prepare_input_backend",
    "resolve_input_backend_policy",
    "InputConfigRandomState",
    "derive_input_seed",
]
