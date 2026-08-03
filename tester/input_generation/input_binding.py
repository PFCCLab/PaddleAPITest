"""签名绑定与路径映射。"""

from __future__ import annotations

import collections
import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import yaml

from .tensor_config import TensorConfig
from .tensor_path import TensorPath
from .tensor_spec import TensorSpec

BASE_CONFIG = Path(__file__).resolve().parents[1] / "base_config.yaml"

# C-ops 别名映射留在这里，确保调度层不依赖 inspect 逻辑。
COPS_PUBLIC_ALIASES = {
    "paddle._C_ops.add_": "paddle.add",
    "paddle._C_ops.bitwise_not": "paddle.bitwise_not",
    "paddle._C_ops.clip": "paddle.clip",
    "paddle._C_ops.concat": "paddle.concat",
    "paddle._C_ops.flatten_": "paddle.flatten",
    "paddle._C_ops.matmul": "paddle.matmul",
    "paddle._C_ops.multiply_": "paddle.multiply",
    "paddle._C_ops.numel": "paddle.numel",
    "paddle._C_ops.put_along_axis": "paddle.put_along_axis",
    "paddle._C_ops.put_along_axis_": "paddle.put_along_axis",
    "paddle._C_ops.reshape_": "paddle.reshape",
    "paddle._C_ops.scale_": "paddle.scale",
    "paddle._C_ops.subtract_": "paddle.subtract",
    "paddle._C_ops.transpose": "paddle.transpose",
    "paddle._C_ops.uniform": "paddle.uniform",
}


def _load_no_sig_ops() -> tuple[str, ...]:
    data = yaml.safe_load(BASE_CONFIG.read_text())
    return tuple(data.get("single_op_no_signature_apis", []))


# 这些 Tensor 方法没有稳定的 inspect 签名，因此 `base_config.yaml` 仍是它们的唯一来源。
SINGLE_OP_NO_SIGNATURE_APIS = _load_no_sig_ops()

SINGLE_OP_PARAMETER_NAMES = {
    f"paddle.Tensor.{method}": ("self", "y") for method in SINGLE_OP_NO_SIGNATURE_APIS
}

# 手工参数名只覆盖 inspect.signature 不可靠的 API；正常公共 API 仍应走运行时绑定。
MANUAL_PARAMETER_NAMES = {
    **SINGLE_OP_PARAMETER_NAMES,
    "paddle.Tensor.__getitem__": ("self", "item"),
    "paddle.Tensor.__setitem__": ("self", "item", "value"),
    "paddle._C_ops.adam_": (
        "param",
        "grad",
        "learning_rate",
        "moment1",
        "moment2",
        "moment2_max",
        "beta1_pow",
        "beta2_pow",
        "master_param",
        "skip_update",
        "beta1",
        "beta2",
        "epsilon",
        "lazy_mode",
        "min_row_size_to_use_multithread",
        "multi_precision",
        "use_global_beta_pow",
        "amsgrad",
    ),
    "paddle._C_ops.adamw_": (
        "param",
        "grad",
        "learning_rate",
        "moment1",
        "moment2",
        "moment2_max",
        "beta1_pow",
        "beta2_pow",
        "master_param",
        "skip_update",
        "beta1",
        "beta2",
        "epsilon",
        "lr_ratio",
        "coeff",
        "with_decay",
        "lazy_mode",
        "min_row_size_to_use_multithread",
        "multi_precision",
        "use_global_beta_pow",
        "amsgrad",
    ),
    "paddle._C_ops.merged_adam_": (
        "param",
        "grad",
        "learning_rate",
        "moment1",
        "moment2",
        "moment2_max",
        "beta1_pow",
        "beta2_pow",
        "master_param",
        "beta1",
        "beta2",
        "epsilon",
        "multi_precision",
        "use_global_beta_pow",
        "amsgrad",
    ),
    "paddle._C_ops.full_": ("x", "shape", "value", "dtype"),
    "paddle._C_ops.fused_linear_param_grad_add": (
        "x",
        "dout",
        "dweight",
        "dbias",
        "multi_precision",
        "has_bias",
    ),
    "paddle._C_ops.gaussian": ("shape", "mean", "std", "seed", "dtype"),
    "paddle._C_ops.matmul_grad": ("x", "y", "dout", "transpose_x", "transpose_y"),
    "paddle._C_ops.squared_l2_norm": ("x",),
    "paddle._C_ops.swiglu_grad": ("x", "y", "dout"),
    "paddle._C_ops._run_custom_op": ("op_name", "arg1", "arg2", "arg3", "arg4"),
}


def get_arg(api_config, position, name, default=None):
    if 0 <= position < len(api_config.args):
        return api_config.args[position]
    if name in api_config.kwargs:
        return api_config.kwargs[name]
    return default


@dataclass(frozen=True)
class ParameterBinding:
    path: TensorPath
    parameter_name: str | None


@dataclass(frozen=True)
class TensorBinding:
    path: TensorPath
    parameter_name: str | None
    spec: TensorSpec


@dataclass(frozen=True)
class InputBinding:
    """规则侧看到的一次 APIConfig 绑定结果。"""

    api_name: str
    binding_source: str
    parameter_bindings: tuple[ParameterBinding, ...]
    tensors: tuple[TensorBinding, ...]
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class InputContext:
    """一次生成 case 的运行时上下文。"""

    call: InputBinding
    config_fingerprint: str
    seed: int
    gpu_enabled: bool


@dataclass(frozen=True)
class SignatureResolution:
    """一次签名解析的结果。"""

    signature: inspect.Signature | None
    source: str
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class ArgumentBindingResolution:
    """一次参数绑定的结果。"""

    arguments: collections.OrderedDict
    source: str
    unresolved_reason: str | None = None


def resolve_api(api_name):
    import paddle

    value = paddle
    parts = api_name.split(".")
    if not parts or parts[0] != "paddle":
        raise ValueError(f"unsupported API root: {api_name}")
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def resolve_signature(api_name, api=None):
    api = api or resolve_api(api_name)
    try:
        signature = inspect.signature(api)
    except (TypeError, ValueError):
        signature = None
    source = "signature"
    if signature is None:
        # 某些 C-ops 会通过可读的 Paddle 公共 API 暴露。对公共别名绑定能保持参数名稳定。
        public_api_name = COPS_PUBLIC_ALIASES.get(api_name)
        if public_api_name is None:
            return SignatureResolution(
                signature=None,
                source="unresolved",
                unresolved_reason="API has no inspectable signature or public alias",
            )
        public_api = resolve_api(public_api_name)
        try:
            signature = inspect.signature(public_api)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            return SignatureResolution(
                signature=None,
                source="unresolved",
                unresolved_reason=f"public alias has no signature: {public_api_name}",
            )
        source = f"public-alias:{public_api_name}"
    return SignatureResolution(signature=signature, source=source)


def _manual_arguments(args, kwargs, parameter_names):
    return collections.OrderedDict(
        (
            name,
            args[position] if 0 <= position < len(args) else kwargs.get(name),
        )
        for position, name in enumerate(parameter_names)
    )


def bind_parameters(
    api_name,
    args,
    kwargs,
    *,
    api=None,
    keep_name=False,
):
    if api_name in MANUAL_PARAMETER_NAMES:
        return ArgumentBindingResolution(
            arguments=_manual_arguments(args, kwargs, MANUAL_PARAMETER_NAMES[api_name]),
            source="manual",
        )
    if api_name in ("paddle.Tensor.view", "paddle.view"):
        # view/reshape 接受可变长 shape，这里归一成一个参数名。
        rest = args[1:]
        if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
            return ArgumentBindingResolution(
                arguments=collections.OrderedDict([("x", args[0]), ("shape_or_dtype", list(rest))]),
                source="variadic-view",
            )
    if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
        # reshape 和 view 一样有可变长 shape 问题。
        rest = args[1:]
        if rest and all(isinstance(arg, int) for arg in rest):
            return ArgumentBindingResolution(
                arguments=collections.OrderedDict([("x", args[0]), ("shape", list(rest))]),
                source="variadic-reshape",
            )

    resolution = resolve_signature(api_name, api=api)
    signature = resolution.signature
    if signature is None:
        return ArgumentBindingResolution(
            arguments=collections.OrderedDict(),
            source=resolution.source,
            unresolved_reason=resolution.unresolved_reason,
        )

    if resolution.source.startswith("public-alias:"):
        positional_count = sum(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
        valid_names = set(signature.parameters)
        kwargs = {key: value for key, value in kwargs.items() if key in valid_names}
        bound = signature.bind(*args[:positional_count], **kwargs)
    else:
        bound = signature.bind(*args, **kwargs)

    arguments = collections.OrderedDict(bound.arguments)
    if not keep_name:
        arguments.pop("name", None)
    if api_name == "paddle.arange" and "end" not in arguments:
        # `paddle.arange(start)` 的语义等价于 `arange(0, start)`。
        arguments["end"] = arguments["start"]
        arguments["start"] = 0
    if api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"}:
        arguments["name"] = None
    return ArgumentBindingResolution(
        arguments=arguments,
        source=resolution.source,
        unresolved_reason=resolution.unresolved_reason,
    )


def _contains_identity(value, target):
    if value is target:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(item, target) for item in value)
    return False


def _path_parameters(api_config, arguments):
    bindings = []
    for index, value in enumerate(api_config.args):
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        bindings.append(
            ParameterBinding(
                path=TensorPath.positional(index),
                parameter_name=names[0] if len(names) == 1 else None,
            )
        )
    for key, value in api_config.kwargs.items():
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        bindings.append(
            ParameterBinding(
                path=TensorPath.keyword(key),
                parameter_name=names[0] if len(names) == 1 else key,
            )
        )
    return tuple(bindings)


def _walk_tensors(value, path, parameter_name, output):
    # 嵌套 TensorConfig 列表保留顶层参数名，但会扩展 TensorPath。
    if isinstance(value, TensorConfig):
        output.append(
            TensorBinding(
                path=path,
                parameter_name=parameter_name,
                spec=TensorSpec.from_tensor_config(value),
            )
        )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_tensors(child, path.child(index), parameter_name, output)


def bind_input(api_config):
    # `InputBinding` 是规则层唯一应该直接读取的绑定对象。
    resolution = bind_parameters(
        api_config.api_name,
        api_config.args,
        api_config.kwargs,
        keep_name=api_config.api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"},
    )
    arguments = (
        collections.OrderedDict() if resolution.source == "unresolved" else resolution.arguments
    )
    path_parameters = _path_parameters(api_config, arguments)
    parameter_by_path = {binding.path: binding.parameter_name for binding in path_parameters}
    tensors = []
    for index, value in enumerate(api_config.args):
        path = TensorPath.positional(index)
        _walk_tensors(value, path, parameter_by_path.get(path), tensors)
    for key, value in api_config.kwargs.items():
        path = TensorPath.keyword(key)
        _walk_tensors(value, path, parameter_by_path.get(path), tensors)
    return InputBinding(
        api_name=api_config.api_name,
        binding_source=resolution.source,
        parameter_bindings=path_parameters,
        tensors=tuple(tensors),
        unresolved_reason=resolution.unresolved_reason
        or (
            "API has no inspectable signature or public alias"
            if resolution.source == "unresolved"
            else None
        ),
    )


def build_input_context(
    api_config,
    seed,
    gpu_enabled,
):
    return InputContext(
        call=bind_input(api_config),
        config_fingerprint=hashlib.sha256(api_config.config.encode()).hexdigest(),
        seed=seed,
        gpu_enabled=gpu_enabled,
    )
