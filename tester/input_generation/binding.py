"""签名绑定与路径映射。"""

from __future__ import annotations

import collections
import hashlib
from dataclasses import dataclass

from tester.api_config.parameter_binding import bind_input_parameters

from .tensor_config import TensorConfig
from .values import InputTensorPath, InputTensorSpec


@dataclass(frozen=True)
class InputTensorBinding:
    path: InputTensorPath
    parameter_name: str | None
    input_spec: InputTensorSpec

    @property
    def shape(self):
        # 高频规格直接代理到只读快照，规则无需了解 InputTensorSpec 的存储层级。
        return self.input_spec.shape

    @property
    def dtype(self):
        return self.input_spec.dtype

    @property
    def place(self):
        return self.input_spec.place

    @property
    def is_contiguous(self):
        return self.input_spec.is_contiguous

    @property
    def strides(self):
        return self.input_spec.strides


@dataclass(frozen=True)
class InputApiBinding:
    """规则侧看到的一次 APIConfig 绑定结果。"""

    api_name: str
    binding_source: str
    tensor_bindings: tuple[InputTensorBinding, ...]
    arguments: tuple[tuple[str, object], ...] = ()
    parameter_names: tuple[str, ...] = ()
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class InputGenerationContext:
    """一次输入生成所需的绑定和 backend seed 元数据。"""

    input_binding: InputApiBinding
    config_fingerprint: str
    seed: int
    backend_policy: object
    # 数值策略随生成上下文传递，规则执行阶段不再读取进程环境。
    input_max_abs: float


def _contains_identity(value, target):
    # 参数列表中的 TensorConfig 仍按对象 identity 归属到顶层参数名。
    if value is target:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(item, target) for item in value)
    return False


def _iter_top_level_inputs(api_config):
    yield from (
        (InputTensorPath.positional(index), value, None)
        for index, value in enumerate(api_config.args)
    )
    yield from (
        (InputTensorPath.keyword(key), value, key) for key, value in api_config.kwargs.items()
    )


def _map_input_parameter_names_by_path(api_config, arguments):
    # path 是写回配置的稳定地址，参数名只承担规则分发职责。
    parameter_names = {}
    for path, value, fallback_name in _iter_top_level_inputs(api_config):
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        parameter_names[path] = names[0] if len(names) == 1 else fallback_name
    return parameter_names


def _collect_input_tensor_bindings(value, path, parameter_name, output, path_by_tensor_id):
    # 嵌套 TensorConfig 列表保留顶层参数名，但会扩展 InputTensorPath。
    if isinstance(value, TensorConfig):
        previous_path = path_by_tensor_id.get(id(value))
        if previous_path is not None:
            # 同一对象对应多个 path 时无法确定写回位置，因此在绑定阶段拒绝。
            raise ValueError(
                f"TensorConfig is reused across input paths: {previous_path} and {path}"
            )
        path_by_tensor_id[id(value)] = path
        output.append(
            InputTensorBinding(
                path=path,
                parameter_name=parameter_name,
                input_spec=InputTensorSpec.from_tensor_config(value),
            )
        )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _collect_input_tensor_bindings(
                child,
                path.child(index),
                parameter_name,
                output,
                path_by_tensor_id,
            )


def bind_input_tensors(api_config):
    # `InputApiBinding` 是规则层唯一应该直接读取的绑定对象。
    parameter_binding = bind_input_parameters(
        api_config.api_name,
        api_config.args,
        api_config.kwargs,
        include_name_parameter=api_config.api_name
        in {"paddle.Tensor.unflatten", "paddle.unflatten"},
    )
    arguments = (
        collections.OrderedDict()
        if parameter_binding.source == "unresolved"
        else parameter_binding.arguments
    )
    # 未解析 API 仍可通过关键字回退识别 Tensor，但不会猜测位置参数名。
    parameter_names_by_path = _map_input_parameter_names_by_path(api_config, arguments)
    tensors = []
    path_by_tensor_id = {}
    for path, value, _fallback_name in _iter_top_level_inputs(api_config):
        _collect_input_tensor_bindings(
            value,
            path,
            parameter_names_by_path.get(path),
            tensors,
            path_by_tensor_id,
        )
    return InputApiBinding(
        api_name=api_config.api_name,
        binding_source=parameter_binding.source,
        tensor_bindings=tuple(tensors),
        arguments=tuple(arguments.items()),
        parameter_names=parameter_binding.parameter_names,
        unresolved_reason=parameter_binding.unresolved_reason
        or (
            "API has no inspectable signature or public alias"
            if parameter_binding.source == "unresolved"
            else None
        ),
    )


def build_input_generation_context(api_config, seed, backend_policy, input_max_abs):
    # 配置文本指纹使不同调用在原生 backend 中获得稳定且隔离的随机流。
    return InputGenerationContext(
        input_binding=bind_input_tensors(api_config),
        config_fingerprint=hashlib.sha256(api_config.config.encode()).hexdigest(),
        seed=seed,
        backend_policy=backend_policy,
        input_max_abs=input_max_abs,
    )
