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
    "paddle.Tensor.clone": ("self",),
    "paddle.Tensor.detach": ("self",),
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


@dataclass(frozen=True)
class TensorBinding:
    path: TensorPath
    parameter_name: str | None
    spec: TensorSpec

    @property
    def shape(self):
        # 高频规格直接代理到只读快照，规则无需了解 TensorSpec 的存储层级。
        return self.spec.shape

    @property
    def dtype(self):
        return self.spec.dtype

    @property
    def place(self):
        return self.spec.place

    @property
    def is_contiguous(self):
        return self.spec.is_contiguous

    @property
    def strides(self):
        return self.spec.strides


@dataclass(frozen=True)
class InputBinding:
    """规则侧看到的一次 APIConfig 绑定结果。"""

    api_name: str
    binding_source: str
    tensors: tuple[TensorBinding, ...]
    arguments: tuple[tuple[str, object], ...] = ()
    parameter_names: tuple[str, ...] = ()
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class InputContext:
    """一次输入生成所需的绑定和 backend seed 元数据。"""

    call: InputBinding
    config_fingerprint: str
    seed: int


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
    parameter_names: tuple[str, ...] = ()
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
    # inspectable 公共 API 是参数名真源；仅对无签名 C-op 使用显式别名。
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
    # 只记录调用实际提供的值，不能把缺失参数伪装成显式 None。
    arguments = collections.OrderedDict()
    for position, name in enumerate(parameter_names):
        if position < len(args):
            arguments[name] = args[position]
        elif name in kwargs:
            arguments[name] = kwargs[name]
    return arguments


def bind_parameters(
    api_name,
    args,
    kwargs,
    *,
    api=None,
    keep_name=False,
):
    if api_name in MANUAL_PARAMETER_NAMES:
        parameter_names = MANUAL_PARAMETER_NAMES[api_name]
        return ArgumentBindingResolution(
            arguments=_manual_arguments(args, kwargs, parameter_names),
            source="manual",
            parameter_names=parameter_names,
        )
    if api_name in ("paddle.Tensor.view", "paddle.view"):
        # view/reshape 接受可变长 shape，这里归一成一个参数名。
        rest = args[1:]
        if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
            return ArgumentBindingResolution(
                arguments=collections.OrderedDict([("x", args[0]), ("shape_or_dtype", list(rest))]),
                source="variadic-view",
                parameter_names=("x", "shape_or_dtype"),
            )
    if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
        # reshape 和 view 一样有可变长 shape 问题。
        rest = args[1:]
        if rest and all(isinstance(arg, int) for arg in rest):
            return ArgumentBindingResolution(
                arguments=collections.OrderedDict([("x", args[0]), ("shape", list(rest))]),
                source="variadic-reshape",
                parameter_names=("x", "shape"),
            )

    resolution = resolve_signature(api_name, api=api)
    signature = resolution.signature
    if signature is None:
        return ArgumentBindingResolution(
            arguments=collections.OrderedDict(),
            source=resolution.source,
            parameter_names=(),
            unresolved_reason=resolution.unresolved_reason,
        )

    if resolution.source.startswith("public-alias:"):
        # C-op 可能携带公共签名不存在的内部属性，绑定前需过滤这些关键字。
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
        parameter_names=tuple(name for name in signature.parameters if keep_name or name != "name"),
        unresolved_reason=resolution.unresolved_reason,
    )


def _contains_identity(value, target):
    # 参数列表中的 TensorConfig 仍按对象 identity 归属到顶层参数名。
    if value is target:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(item, target) for item in value)
    return False


def _top_level_values(api_config):
    yield from (
        (TensorPath.positional(index), value, None) for index, value in enumerate(api_config.args)
    )
    yield from ((TensorPath.keyword(key), value, key) for key, value in api_config.kwargs.items())


def _parameter_names_by_path(api_config, arguments):
    # path 是写回配置的稳定地址，参数名只承担规则分发职责。
    parameter_names = {}
    for path, value, fallback_name in _top_level_values(api_config):
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        parameter_names[path] = names[0] if len(names) == 1 else fallback_name
    return parameter_names


def _walk_tensors(value, path, parameter_name, output, path_by_tensor_id):
    # 嵌套 TensorConfig 列表保留顶层参数名，但会扩展 TensorPath。
    if isinstance(value, TensorConfig):
        previous_path = path_by_tensor_id.get(id(value))
        if previous_path is not None:
            # 同一对象对应多个 path 时无法确定写回位置，因此在绑定阶段拒绝。
            raise ValueError(
                f"TensorConfig is reused across input paths: {previous_path} and {path}"
            )
        path_by_tensor_id[id(value)] = path
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
            _walk_tensors(
                child,
                path.child(index),
                parameter_name,
                output,
                path_by_tensor_id,
            )


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
    # 未解析 API 仍可通过关键字回退识别 Tensor，但不会猜测位置参数名。
    parameter_names_by_path = _parameter_names_by_path(api_config, arguments)
    tensors = []
    path_by_tensor_id = {}
    for path, value, _fallback_name in _top_level_values(api_config):
        _walk_tensors(
            value,
            path,
            parameter_names_by_path.get(path),
            tensors,
            path_by_tensor_id,
        )
    return InputBinding(
        api_name=api_config.api_name,
        binding_source=resolution.source,
        tensors=tuple(tensors),
        arguments=tuple(arguments.items()),
        parameter_names=resolution.parameter_names,
        unresolved_reason=resolution.unresolved_reason
        or (
            "API has no inspectable signature or public alias"
            if resolution.source == "unresolved"
            else None
        ),
    )


def build_input_context(api_config, seed):
    # 配置文本指纹使不同调用在原生 backend 中获得稳定且隔离的随机流。
    return InputContext(
        call=bind_input(api_config),
        config_fingerprint=hashlib.sha256(api_config.config.encode()).hexdigest(),
        seed=seed,
    )
