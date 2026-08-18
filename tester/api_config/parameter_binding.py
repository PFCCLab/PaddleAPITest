"""Paddle parameter binding shared by execution and input generation."""

from __future__ import annotations

import collections
import functools
import inspect
from dataclasses import dataclass
from pathlib import Path

import yaml

INPUT_BASE_CONFIG = Path(__file__).resolve().parents[1] / "base_config.yaml"
# 配置文件相对 tester 定位，避免执行入口改变当前工作目录后失效。

# C-op 公共别名是绑定协议的一部分，避免调用方各自解释无签名算子。
INPUT_C_OP_PUBLIC_ALIASES = {
    # 公开别名只用于恢复稳定参数名，不改变实际执行的 C-op。
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
    # 无签名方法只能依赖仓库契约，不能用调用时的值反推参数名。
    base_config = yaml.safe_load(INPUT_BASE_CONFIG.read_text())
    return tuple(base_config.get("single_op_no_signature_apis", []))


# 这些 Tensor 方法没有稳定签名，配置文件仍是它们的唯一参数来源。
INPUT_APIS_WITHOUT_SIGNATURE = _load_signatureless_input_apis()
INPUT_SINGLE_OP_PARAMETER_NAMES = {
    f"paddle.Tensor.{method}": ("self", "y") for method in INPUT_APIS_WITHOUT_SIGNATURE
}

# 手工契约只覆盖反射不可靠的 API；正常公共 API 仍由运行时签名绑定。
INPUT_MANUAL_PARAMETER_NAMES = {
    # 手工表覆盖 C++ 暴露的不可反射函数，条目顺序就是位置参数协议。
    **INPUT_SINGLE_OP_PARAMETER_NAMES,
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

# 执行侧展开默认值，输入生成侧保留缺省参数以便后续准确分类。
INPUT_MANUAL_PARAMETER_DEFAULTS = {
    # 默认值仅在 apply_defaults=True 的执行调用中展开。
    "paddle.Tensor.copy_": {"blocking": True},
    "paddle._C_ops._run_custom_op": {"arg1": None, "arg2": None, "arg3": None, "arg4": None},
    "paddle._C_ops.full_": {"place": None},
    "paddle._C_ops.gaussian": {"place": None},
    "paddle._C_ops.uniform": {"dtype": None, "min": 0, "max": 1.0, "seed": 0, "place": None},
}


@dataclass(frozen=True)
class InputSignatureResult:
    """一次签名解析的结果。"""

    # source 让上层区分反射、别名和无法解析三种绑定来源。
    signature: inspect.Signature | None
    source: str
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class InputParameterBindingResult:
    """一次 Paddle 调用的参数绑定结果。"""

    # 失败原因随结果传递，调用方不能重新猜测无签名 API 的契约。
    arguments: collections.OrderedDict
    source: str
    parameter_names: tuple[str, ...] = ()
    unresolved_reason: str | None = None


def resolve_input_api(api_name):
    # 只允许 Paddle 根路径，防止绑定层意外解析任意模块对象。
    import paddle

    resolved_api = paddle
    api_path_parts = api_name.split(".")
    if not api_path_parts or api_path_parts[0] != "paddle":
        raise ValueError(f"unsupported API root: {api_name}")
    for part in api_path_parts[1:]:
        resolved_api = getattr(resolved_api, part)
    return resolved_api


def _inspect_input_signature(api):
    try:
        return inspect.signature(api)
    except (TypeError, ValueError):
        return None


@functools.lru_cache(maxsize=512)
def _inspect_input_signature_cached(api):
    """缓存 worker 内稳定 callable 的反射元数据。"""
    return _inspect_input_signature(api)


def _inspect_input_signature_with_cache(api):
    try:
        # callable 进入键后，运行时重绑定会自然触发新一轮反射。
        return _inspect_input_signature_cached(api)
    except TypeError:
        # 不可哈希 callable 保留原语义，不因缓存支持范围而失败。
        return _inspect_input_signature(api)


def resolve_input_signature(api_name, api=None):
    # 可反射 API 使用真实签名；无签名 C-op 只接受明确的公共别名。
    api = api or resolve_input_api(api_name)
    signature = _inspect_input_signature_with_cache(api)
    source = "signature"
    # api 参数由执行侧传入时可避免重复解析，但解析规则保持一致。
    if signature is None:
        public_api_name = INPUT_C_OP_PUBLIC_ALIASES.get(api_name)
        if public_api_name is None:
            return InputSignatureResult(
                None, "unresolved", "API has no inspectable signature or public alias"
            )
        public_api = resolve_input_api(public_api_name)
        # 别名仍逐次解析当前 callable，避免 monkeypatch 或运行时重绑定命中旧签名。
        signature = _inspect_input_signature_with_cache(public_api)
        if signature is None:
            return InputSignatureResult(
                None, "unresolved", f"public alias has no signature: {public_api_name}"
            )
        source = f"public-alias:{public_api_name}"
    return InputSignatureResult(signature=signature, source=source)


def _bind_manual_input_arguments(api_name, args, kwargs, parameter_names, *, apply_defaults):
    defaults = INPUT_MANUAL_PARAMETER_DEFAULTS.get(api_name, {})
    signature = inspect.Signature(
        # 临时签名复用 inspect 的重复参数、缺失参数错误语义。
        parameters=[
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults.get(name, inspect.Parameter.empty),
            )
            for name in parameter_names
        ]
    )
    bind = signature.bind if apply_defaults else signature.bind_partial
    bound = bind(*args, **kwargs)
    if apply_defaults:
        bound.apply_defaults()
    return collections.OrderedDict(bound.arguments)


def _canonicalize_tensor_receiver(api_name, arguments, parameter_names):
    # Tensor 方法的接收者统一命名为 x，隐藏 Paddle 的 self/首参拼写差异。
    parameter_names = tuple(parameter_names)
    if not api_name.startswith("paddle.Tensor.") or not parameter_names:
        return arguments, parameter_names
    receiver_name = parameter_names[0]
    # 只有 Tensor method 需要 receiver 归一化，函数首参不能改名。
    if receiver_name in arguments and receiver_name != "x":
        arguments = collections.OrderedDict(
            ("x" if name == receiver_name else name, value) for name, value in arguments.items()
        )
    if receiver_name != "x":
        parameter_names = ("x", *parameter_names[1:])
    return arguments, parameter_names


def _validate_variadic_shape_kwargs(kwargs):
    # 变长 shape 绕过 inspect.bind，仍需保留未知关键字的失败语义。
    unexpected = set(kwargs) - {"name"}
    # name 是 Paddle 的非数据参数，其他未知关键字必须继续报错。
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"got unexpected keyword arguments: {names}")


def split_tensor_method_arguments(api_name, arguments):
    """Restore a bound Tensor method invocation for generated Torch code."""
    # 绑定结果以参数名保存，执行时再恢复 method receiver 的位置参数。
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
    # 所有调用路径都返回同一结果对象，调用方只消费 source 与 arguments。
    if api_name in INPUT_MANUAL_PARAMETER_NAMES:
        parameter_names = INPUT_MANUAL_PARAMETER_NAMES[api_name]
        arguments = _bind_manual_input_arguments(
            api_name, args, kwargs, parameter_names, apply_defaults=apply_defaults
        )
        arguments, parameter_names = _canonicalize_tensor_receiver(
            api_name, arguments, parameter_names
        )
        return InputParameterBindingResult(arguments, "manual", parameter_names)
    if api_name in ("paddle.Tensor.view", "paddle.view"):
        rest = args[1:]
        if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
            # 变长整数 shape 统一成单一列表，避免规则处理 args 容器。
            if apply_defaults:
                _validate_variadic_shape_kwargs(kwargs)
            return InputParameterBindingResult(
                collections.OrderedDict([("x", args[0]), ("shape_or_dtype", list(rest))]),
                "variadic-view",
                ("x", "shape_or_dtype"),
            )
    if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
        rest = args[1:]
        if rest and all(isinstance(arg, int) for arg in rest):
            # reshape 的变长协议与 view 相同，但保留独立 source 标签。
            if apply_defaults:
                _validate_variadic_shape_kwargs(kwargs)
            return InputParameterBindingResult(
                collections.OrderedDict([("x", args[0]), ("shape", list(rest))]),
                "variadic-reshape",
                ("x", "shape"),
            )
    if api_name in ("paddle.Tensor.expand", "paddle.expand"):
        rest = args[1:]
        if rest and all(isinstance(arg, int) for arg in rest):
            # expand 同样接受逐维位置参数，统一收敛为规则使用的 shape 列表。
            if apply_defaults:
                _validate_variadic_shape_kwargs(kwargs)
            # source 标签让输入生成和执行日志能区分变长协议。
            return InputParameterBindingResult(
                collections.OrderedDict([("x", args[0]), ("shape", list(rest))]),
                "variadic-expand",
                ("x", "shape"),
            )
    if (
        api_name in ("paddle.empty", "paddle.zeros")
        and len(args) > 1
        and all(isinstance(arg, int) for arg in args)
    ):
        # 仅整数序列采用变长 shape 协议，列表 shape 仍交由原生签名校验。
        # empty/zeros 的多整数位置协议在直接绑定调用中也要收敛为 shape。
        signature_result = resolve_input_signature(api_name, api=api)
        signature = signature_result.signature
        if signature is not None:
            bound = signature.bind(list(args), **kwargs)
            if apply_defaults:
                bound.apply_defaults()
            source = "variadic-empty" if api_name == "paddle.empty" else "variadic-zeros"
            return InputParameterBindingResult(
                collections.OrderedDict(bound.arguments),
                source,
                tuple(bound.arguments),
            )

    signature_result = resolve_input_signature(api_name, api=api)
    signature = signature_result.signature
    if signature is None:
        # unresolved 不伪造参数名；输入层可回退到关键字路径识别 Tensor。
        return InputParameterBindingResult(
            collections.OrderedDict(),
            signature_result.source,
            (),
            signature_result.unresolved_reason,
        )
    if signature_result.source.startswith("public-alias:"):
        # C-op 内部关键字可能不在公共签名中，过滤后再执行标准绑定。
        positional_count = sum(
            parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in signature.parameters.values()
        )
        valid_names = set(signature.parameters)
        kwargs = {key: value for key, value in kwargs.items() if key in valid_names}
        bound = signature.bind(*args[:positional_count], **kwargs)
    else:
        # 公共 API 的异常直接保留 inspect.bind 语义，供上层分类。
        bound = signature.bind(*args, **kwargs)
    if apply_defaults:
        # 仅 Torch 执行需要完整默认集，生成阶段保持用户实际输入集合。
        bound.apply_defaults()
    arguments = collections.OrderedDict(bound.arguments)
    if not include_name_parameter:
        # name 只用于 Paddle 调用，不应成为生成规则的输入 Tensor 参数。
        arguments.pop("name", None)
    if api_name == "paddle.arange" and arguments.get("end") is None:
        # 单参数 arange 需要先归一化为显式 start/end，保持规则参数稳定。
        arguments["end"] = arguments["start"]
        arguments["start"] = 0
    if api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"}:
        arguments["name"] = None
    parameter_names = tuple(
        name for name in signature.parameters if include_name_parameter or name != "name"
    )
    # receiver 归一化放在最终参数名生成后，保证 arguments 与 names 同步。
    arguments, parameter_names = _canonicalize_tensor_receiver(api_name, arguments, parameter_names)
    return InputParameterBindingResult(
        arguments, signature_result.source, parameter_names, signature_result.unresolved_reason
    )
