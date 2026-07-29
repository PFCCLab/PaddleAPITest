"""Shared signature and alias mappings for input-generation binding."""

from __future__ import annotations

import collections
import inspect
from dataclasses import dataclass
from pathlib import Path

import yaml

BASE_CONFIG = Path(__file__).resolve().parents[2] / "base_config.yaml"

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


def _load_single_op_no_signature_apis() -> tuple[str, ...]:
    data = yaml.safe_load(BASE_CONFIG.read_text())
    return tuple(data.get("single_op_no_signature_apis", []))


SINGLE_OP_NO_SIGNATURE_APIS = _load_single_op_no_signature_apis()

SINGLE_OP_PARAMETER_NAMES = {
    f"paddle.Tensor.{method}": ("self", "y") for method in SINGLE_OP_NO_SIGNATURE_APIS
}

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

OPTIMIZER_APIS = {
    # moment1, moment2, moment2_max (must be non-negative for amsgrad)
    "paddle._C_ops.adamw_": {3: "zeros", 4: "zeros", 5: "zeros"},
    "paddle._C_ops.adam_": {3: "zeros", 4: "zeros", 5: "zeros"},
    "paddle._C_ops.merged_adam_": {3: "zeros", 4: "zeros", 5: "zeros"},
}


def build_no_signature_api_mappings():
    return {
        f"paddle.Tensor.{method}": {
            "self": lambda cfg: get_arg(cfg, 0, "self"),
            "y": lambda cfg: get_arg(cfg, 1, "y"),
        }
        for method in SINGLE_OP_NO_SIGNATURE_APIS
    }


def get_arg(api_config, position, name, default=None):
    if 0 <= position < len(api_config.args):
        return api_config.args[position]
    if name in api_config.kwargs:
        return api_config.kwargs[name]
    return default


def _parameter_mapping(parameter_names):
    def getter(position, name):
        return lambda cfg: get_arg(cfg, position, name)

    return {name: getter(position, name) for position, name in enumerate(parameter_names)}


@dataclass(frozen=True)
class SignatureResolution:
    signature: inspect.Signature | None
    source: str
    resolved_api_name: str | None = None
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class ArgumentBindingResolution:
    arguments: collections.OrderedDict
    source: str
    resolved_api_name: str | None = None
    unresolved_reason: str | None = None


def resolve_dotted_api(api_name):
    import paddle

    value = paddle
    parts = api_name.split(".")
    if not parts or parts[0] != "paddle":
        raise ValueError(f"unsupported API root: {api_name}")
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def resolve_api_signature(api_name, api=None, signature_cache=None):
    api = api or resolve_dotted_api(api_name)
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
        public_api_name = COPS_PUBLIC_ALIASES.get(api_name)
        if public_api_name is None:
            return SignatureResolution(
                signature=None,
                source="unresolved",
                resolved_api_name=None,
                unresolved_reason="API has no inspectable signature or public alias",
            )
        public_api = resolve_dotted_api(public_api_name)
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


def bind_api_arguments(
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
        rest = args[1:]
        if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
            return ArgumentBindingResolution(
                arguments=collections.OrderedDict([("x", args[0]), ("shape_or_dtype", list(rest))]),
                source="variadic-view",
            )
    if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
        rest = args[1:]
        if rest and all(isinstance(arg, int) for arg in rest):
            return ArgumentBindingResolution(
                arguments=collections.OrderedDict([("x", args[0]), ("shape", list(rest))]),
                source="variadic-reshape",
            )

    resolution = resolve_api_signature(api_name, api=api, signature_cache=signature_cache)
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


NO_SIGNATURE_API_MAPPINGS = {
    **build_no_signature_api_mappings(),
    **{
        api_name: _parameter_mapping(parameter_names)
        for api_name, parameter_names in MANUAL_PARAMETER_NAMES.items()
        if api_name.startswith("paddle._C_ops.")
    },
}
