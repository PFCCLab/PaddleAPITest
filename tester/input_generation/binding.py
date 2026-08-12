"""签名绑定与路径映射。"""

from __future__ import annotations

import collections
import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import yaml

from .tensor_config import TensorConfig
from .values import InputTensorPath, InputTensorSpec

INPUT_BASE_CONFIG = Path(__file__).resolve().parents[1] / "base_config.yaml"

# C-ops 别名映射留在这里，确保调度层不依赖 inspect 逻辑。
INPUT_C_OP_PUBLIC_ALIASES = {
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


def _load_signatureless_input_apis() -> tuple[str, ...]:
    base_config = yaml.safe_load(INPUT_BASE_CONFIG.read_text())
    return tuple(base_config.get("single_op_no_signature_apis", []))


# 这些 Tensor 方法没有稳定的 inspect 签名，因此 `base_config.yaml` 仍是它们的唯一来源。
INPUT_APIS_WITHOUT_SIGNATURE = _load_signatureless_input_apis()

INPUT_SINGLE_OP_PARAMETER_NAMES = {
    f"paddle.Tensor.{method}": ("self", "y") for method in INPUT_APIS_WITHOUT_SIGNATURE
}

# 手工参数名只覆盖 inspect.signature 不可靠的 API；正常公共 API 仍应走运行时绑定。
INPUT_MANUAL_PARAMETER_NAMES = {
    **INPUT_SINGLE_OP_PARAMETER_NAMES,
    # copy_ 没有可反射签名，保留目标 Tensor 作为第一个位置输入。
    "paddle.Tensor.copy_": ("self", "other", "blocking"),
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
    "paddle._C_ops.full_": ("x", "shape", "value", "dtype", "place"),
    "paddle._C_ops.fused_linear_param_grad_add": (
        "x",
        "dout",
        "dweight",
        "dbias",
        "multi_precision",
        "has_bias",
    ),
    "paddle._C_ops.gaussian": ("shape", "mean", "std", "seed", "dtype", "place"),
    "paddle._C_ops.matmul_grad": ("x", "y", "dout", "transpose_x", "transpose_y"),
    "paddle._C_ops.squared_l2_norm": ("x",),
    "paddle._C_ops.swiglu_grad": ("x", "y", "dout"),
    "paddle._C_ops._run_custom_op": ("op_name", "arg1", "arg2", "arg3", "arg4"),
    "paddle._C_ops.uniform": ("shape", "dtype", "min", "max", "seed", "place"),
}

# 默认值与手工参数名共同构成无签名 API 的唯一调用契约。
# 输入生成阶段不展开这些默认值，以便缺失参数仍由后续执行阶段准确分类。
# 执行阶段显式展开默认值，使规则侧获得与真实 Paddle 调用一致的完整参数集。
INPUT_MANUAL_PARAMETER_DEFAULTS = {
    "paddle.Tensor.copy_": {"blocking": True},
    "paddle._C_ops._run_custom_op": {
        "arg1": None,
        "arg2": None,
        "arg3": None,
        "arg4": None,
    },
    "paddle._C_ops.full_": {"place": None},
    "paddle._C_ops.gaussian": {"place": None},
    "paddle._C_ops.uniform": {
        "dtype": None,
        "min": 0,
        "max": 1.0,
        "seed": 0,
        "place": None,
    },
}


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


@dataclass(frozen=True)
class InputSignatureResult:
    """一次签名解析的结果。"""

    signature: inspect.Signature | None
    source: str
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class InputParameterBindingResult:
    """一次参数绑定的结果。"""

    # source 与 unresolved_reason 必须随参数结果传递，规则层不能重新猜测失败原因。
    arguments: collections.OrderedDict
    source: str
    parameter_names: tuple[str, ...] = ()
    unresolved_reason: str | None = None


def resolve_input_api(api_name):
    import paddle

    resolved_api = paddle
    api_path_parts = api_name.split(".")
    if not api_path_parts or api_path_parts[0] != "paddle":
        raise ValueError(f"unsupported API root: {api_name}")
    for part in api_path_parts[1:]:
        resolved_api = getattr(resolved_api, part)
    return resolved_api


def resolve_input_signature(api_name, api=None):
    # inspectable 公共 API 是参数名真源；仅对无签名 C-op 使用显式别名。
    api = api or resolve_input_api(api_name)
    try:
        signature = inspect.signature(api)
    except (TypeError, ValueError):
        signature = None
    source = "signature"
    if signature is None:
        # 某些 C-ops 会通过可读的 Paddle 公共 API 暴露。对公共别名绑定能保持参数名稳定。
        public_api_name = INPUT_C_OP_PUBLIC_ALIASES.get(api_name)
        if public_api_name is None:
            return InputSignatureResult(
                signature=None,
                source="unresolved",
                unresolved_reason="API has no inspectable signature or public alias",
            )
        public_api = resolve_input_api(public_api_name)
        try:
            signature = inspect.signature(public_api)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            return InputSignatureResult(
                signature=None,
                source="unresolved",
                unresolved_reason=f"public alias has no signature: {public_api_name}",
            )
        source = f"public-alias:{public_api_name}"
    return InputSignatureResult(signature=signature, source=source)


def _bind_manual_input_arguments(api_name, args, kwargs, parameter_names, *, apply_defaults):
    defaults = INPUT_MANUAL_PARAMETER_DEFAULTS.get(api_name, {})
    signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults.get(name, inspect.Parameter.empty),
            )
            for name in parameter_names
        ]
    )
    # 输入生成允许不完整配置留给执行层分类；Torch 执行必须验证完整调用。
    bind = signature.bind if apply_defaults else signature.bind_partial
    bound = bind(*args, **kwargs)
    if apply_defaults:
        bound.apply_defaults()
    return collections.OrderedDict(bound.arguments)


def _canonicalize_tensor_receiver(api_name, arguments, parameter_names):
    # Paddle 的方法签名会把接收者命名为 self、x 或其他首参名。
    # 规则协议固定使用 x，统一后下游无需再次猜测接收者名称。
    parameter_names = tuple(parameter_names)
    if not api_name.startswith("paddle.Tensor.") or not parameter_names:
        return arguments, parameter_names
    receiver_name = parameter_names[0]
    if receiver_name in arguments and receiver_name != "x":
        items = [
            ("x" if name == receiver_name else name, value) for name, value in arguments.items()
        ]
        arguments = collections.OrderedDict(items)
    if receiver_name != "x":
        parameter_names = ("x", *parameter_names[1:])
    return arguments, parameter_names


def _validate_variadic_shape_kwargs(kwargs):
    # 可变 shape 分支绕过 inspect.bind，因此必须在此保留完整绑定的失败语义。
    unexpected = set(kwargs) - {"name"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"got unexpected keyword arguments: {names}")


def split_tensor_method_arguments(api_name, arguments):
    """Restore a bound Tensor method invocation without depending on receiver spelling."""
    # 接收者必须恢复为首个位置参数，供 GenericRule 按 Tensor 方法协议调用。
    # inspect 绑定生成的 args/kwargs 容器需要展开，不能作为普通关键字传入算子。
    call_args = []
    call_kwargs = collections.OrderedDict(arguments)
    if api_name.startswith("paddle.Tensor.") and "x" in call_kwargs:
        call_args.append(call_kwargs.pop("x"))
    call_args.extend(call_kwargs.pop("args", ()))
    variadic_kwargs = call_kwargs.pop("kwargs", {})
    call_kwargs.update(variadic_kwargs)
    return call_args, call_kwargs


def bind_input_parameters(
    api_name,
    args,
    kwargs,
    *,
    api=None,
    include_name_parameter=False,
    apply_defaults=False,
):
    if api_name in INPUT_MANUAL_PARAMETER_NAMES:
        parameter_names = INPUT_MANUAL_PARAMETER_NAMES[api_name]
        arguments = _bind_manual_input_arguments(
            api_name,
            args,
            kwargs,
            parameter_names,
            apply_defaults=apply_defaults,
        )
        arguments, parameter_names = _canonicalize_tensor_receiver(
            api_name, arguments, parameter_names
        )
        return InputParameterBindingResult(
            arguments=arguments,
            source="manual",
            parameter_names=parameter_names,
        )
    if api_name in ("paddle.Tensor.view", "paddle.view"):
        # view/reshape 接受可变长 shape，这里归一成一个参数名。
        rest = args[1:]
        if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
            if apply_defaults:
                _validate_variadic_shape_kwargs(kwargs)
            return InputParameterBindingResult(
                arguments=collections.OrderedDict([("x", args[0]), ("shape_or_dtype", list(rest))]),
                source="variadic-view",
                parameter_names=("x", "shape_or_dtype"),
            )
    if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
        # reshape 和 view 一样有可变长 shape 问题。
        rest = args[1:]
        if rest and all(isinstance(arg, int) for arg in rest):
            if apply_defaults:
                _validate_variadic_shape_kwargs(kwargs)
            return InputParameterBindingResult(
                arguments=collections.OrderedDict([("x", args[0]), ("shape", list(rest))]),
                source="variadic-reshape",
                parameter_names=("x", "shape"),
            )

    signature_result = resolve_input_signature(api_name, api=api)
    signature = signature_result.signature
    if signature is None:
        return InputParameterBindingResult(
            arguments=collections.OrderedDict(),
            source=signature_result.source,
            parameter_names=(),
            unresolved_reason=signature_result.unresolved_reason,
        )

    if signature_result.source.startswith("public-alias:"):
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

    # 只有执行侧需要完整默认参数；输入生成仍保留用户实际提供的参数集合。
    if apply_defaults:
        bound.apply_defaults()
    arguments = collections.OrderedDict(bound.arguments)
    if not include_name_parameter:
        arguments.pop("name", None)
    if api_name == "paddle.arange" and arguments.get("end") is None:
        # `paddle.arange(start)` 的语义等价于 `arange(0, start)`。
        arguments["end"] = arguments["start"]
        arguments["start"] = 0
    if api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"}:
        arguments["name"] = None
    parameter_names = tuple(
        name for name in signature.parameters if include_name_parameter or name != "name"
    )
    arguments, parameter_names = _canonicalize_tensor_receiver(api_name, arguments, parameter_names)
    return InputParameterBindingResult(
        arguments=arguments,
        source=signature_result.source,
        parameter_names=parameter_names,
        unresolved_reason=signature_result.unresolved_reason,
    )


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


def bind_api_inputs(api_config):
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


def build_input_generation_context(api_config, seed, backend_policy):
    # 配置文本指纹使不同调用在原生 backend 中获得稳定且隔离的随机流。
    return InputGenerationContext(
        input_binding=bind_api_inputs(api_config),
        config_fingerprint=hashlib.sha256(api_config.config.encode()).hexdigest(),
        seed=seed,
        backend_policy=backend_policy,
    )
