"""签名绑定与路径映射。"""

from __future__ import annotations

import collections
import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import yaml

from .input_path import InputPath
from .tensor_config import TensorConfig
from .tensor_view import TensorView

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
    path: InputPath
    parameter_name: str | None


@dataclass(frozen=True)
class TensorBinding:
    path: InputPath
    parameter_name: str | None
    spec: TensorView


@dataclass(frozen=True)
class BoundInput:
    """规则侧看到的一次 APIConfig 绑定结果。"""

    api_name: str
    binding_source: str
    parameter_bindings: tuple[ParameterBinding, ...]
    tensors: tuple[TensorBinding, ...]
    unresolved_reason: str | None = None

    def parameter_for(self, path):
        top_level = path.top_level()
        for binding in self.parameter_bindings:
            if binding.path == top_level:
                return binding.parameter_name
        return None

    def tensor_at(self, path):
        for binding in self.tensors:
            if binding.path == path:
                return binding
        raise KeyError(str(path))


@dataclass(frozen=True)
class InputGenerationContext:
    """一次生成 case 的运行时上下文。"""

    call: BoundInput
    config_fingerprint: str
    seed: int
    use_torch: bool
    gpu_enabled: bool

    @classmethod
    def create(
        cls,
        call,
        config_text,
        seed=0,
        use_torch=True,
        gpu_enabled=False,
    ):
        fingerprint = hashlib.sha256(config_text.encode()).hexdigest()
        return cls(
            call=call,
            config_fingerprint=fingerprint,
            seed=int(seed),
            use_torch=bool(use_torch),
            gpu_enabled=bool(gpu_enabled),
        )


@dataclass(frozen=True)
class SignatureResolution:
    """一次签名解析的结果。"""

    signature: inspect.Signature | None
    source: str
    resolved_api_name: str | None = None
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class ArgumentBindingResolution:
    """一次参数绑定的结果。"""

    arguments: collections.OrderedDict
    source: str
    resolved_api_name: str | None = None
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


def resolve_signature(api_name, api=None, signature_cache=None):
    api = api or resolve_api(api_name)
    # 签名结果按 resolver 缓存，避免同一个 case 中重复 import 和 inspect。
    if signature_cache is not None and api_name in signature_cache:
        signature = signature_cache[api_name]
    else:
        try:
            signature = inspect.signature(api)
        except (TypeError, ValueError):
            signature = None
        if signature_cache is not None:
            signature_cache[api_name] = signature
    source = "signature"
    resolved_api_name = api_name
    if signature is None:
        # 某些 C-ops 会通过可读的 Paddle 公共 API 暴露。对公共别名绑定能保持参数名稳定。
        public_api_name = COPS_PUBLIC_ALIASES.get(api_name)
        if public_api_name is None:
            return SignatureResolution(
                signature=None,
                source="unresolved",
                resolved_api_name=None,
                unresolved_reason="API has no inspectable signature or public alias",
            )
        public_api = resolve_api(public_api_name)
        if signature_cache is not None and public_api_name in signature_cache:
            signature = signature_cache[public_api_name]
        else:
            try:
                signature = inspect.signature(public_api)
            except (TypeError, ValueError):
                signature = None
            if signature_cache is not None:
                signature_cache[public_api_name] = signature
        if signature is None:
            return SignatureResolution(
                signature=None,
                source="unresolved",
                resolved_api_name=public_api_name,
                unresolved_reason=f"public alias has no signature: {public_api_name}",
            )
        source = f"public-alias:{public_api_name}"
        resolved_api_name = public_api_name
    return SignatureResolution(
        signature=signature, source=source, resolved_api_name=resolved_api_name
    )


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
    signature_cache=None,
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

    resolution = resolve_signature(api_name, api=api, signature_cache=signature_cache)
    signature = resolution.signature
    if signature is None:
        return ArgumentBindingResolution(
            arguments=collections.OrderedDict(),
            source=resolution.source,
            resolved_api_name=resolution.resolved_api_name,
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
        resolved_api_name=resolution.resolved_api_name,
        unresolved_reason=resolution.unresolved_reason,
    )


@dataclass(frozen=True)
class BindingResolution:
    arguments: collections.OrderedDict
    source: str
    path_parameters: tuple[ParameterBinding, ...]
    unresolved_reason: str | None = None


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
                path=InputPath.positional(index),
                parameter_name=names[0] if len(names) == 1 else None,
            )
        )
    for key, value in api_config.kwargs.items():
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        bindings.append(
            ParameterBinding(
                path=InputPath.keyword(key),
                parameter_name=names[0] if len(names) == 1 else key,
            )
        )
    return tuple(bindings)


class SignatureResolver:
    """按 case 复用签名缓存的解析器。"""

    def __init__(self):
        self._signature_cache = {}

    def _resolved(self, api_config, arguments, source, keep_name=False):
        # 大多数 API 都把 `name` 当元数据，而不是输入生成参数。
        arguments = collections.OrderedDict(arguments)
        if not keep_name:
            arguments.pop("name", None)
        return BindingResolution(
            arguments=arguments,
            source=source,
            path_parameters=_path_parameters(api_config, arguments),
        )

    def resolve(self, api_config, api=None):
        api_name = api_config.api_name
        resolution = bind_parameters(
            api_name,
            api_config.args,
            api_config.kwargs,
            api=api or None,
            signature_cache=self._signature_cache,
            keep_name=api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"},
        )
        if resolution.source == "unresolved":
            return BindingResolution(
                arguments=collections.OrderedDict(),
                source="unresolved",
                path_parameters=_path_parameters(api_config, {}),
                unresolved_reason=resolution.unresolved_reason
                or "API has no inspectable signature or public alias",
            )
        return self._resolved(
            api_config,
            resolution.arguments,
            resolution.source,
            keep_name=api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"},
        )


def _walk_tensors(value, path, parameter_name, output):
    # 嵌套 TensorConfig 列表保留顶层参数名，但会扩展 InputPath。
    if isinstance(value, TensorConfig):
        output.append(
            TensorBinding(
                path=path,
                parameter_name=parameter_name,
                spec=TensorView.from_tensor_config(value),
            )
        )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_tensors(child, path.child(index), parameter_name, output)


def bind_input(api_config, resolver=None):
    # `BoundInput` 是规则层唯一应该直接读取的绑定对象。
    resolver = resolver or SignatureResolver()
    resolution = resolver.resolve(api_config)
    parameter_by_path = {
        binding.path: binding.parameter_name for binding in resolution.path_parameters
    }
    tensors = []
    for index, value in enumerate(api_config.args):
        path = InputPath.positional(index)
        _walk_tensors(value, path, parameter_by_path.get(path), tensors)
    for key, value in api_config.kwargs.items():
        path = InputPath.keyword(key)
        _walk_tensors(value, path, parameter_by_path.get(path), tensors)
    return BoundInput(
        api_name=api_config.api_name,
        binding_source=resolution.source,
        parameter_bindings=resolution.path_parameters,
        tensors=tuple(tensors),
        unresolved_reason=resolution.unresolved_reason,
    )


def build_input_context(
    api_config,
    resolver=None,
    seed=0,
    use_torch=True,
    gpu_enabled=False,
):
    call = bind_input(api_config, resolver=resolver)
    return InputGenerationContext.create(
        call=call,
        config_text=api_config.config,
        seed=seed,
        use_torch=use_torch,
        gpu_enabled=gpu_enabled,
    )
