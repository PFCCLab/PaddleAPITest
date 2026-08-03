from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import paddle
import yaml
from tester.dtype_utils import to_torch_dtype

with (Path(__file__).parent.parent / "base_config.yaml").open(encoding="utf-8") as stream:
    _BASE_CONFIG = yaml.safe_load(stream)

_TENSOR_BINARY_METHODS = tuple(_BASE_CONFIG.get("single_op_no_signature_apis", ()))
del _BASE_CONFIG

_COPS_PUBLIC_ALIASES = {
    "paddle._C_ops.add_": "paddle.add",
    "paddle._C_ops.bitwise_not": "paddle.bitwise_not",
    "paddle._C_ops.clip": "paddle.clip",
    "paddle._C_ops.concat": "paddle.concat",
    "paddle._C_ops.flatten_": "paddle.flatten",
    "paddle._C_ops.matmul": "paddle.matmul",
    "paddle._C_ops.multiply_": "paddle.multiply",
    "paddle._C_ops.numel": "paddle.numel",
    "paddle._C_ops.put_along_axis_": "paddle.put_along_axis",
    "paddle._C_ops.reshape_": "paddle.reshape",
    "paddle._C_ops.scale_": "paddle.scale",
    "paddle._C_ops.subtract_": "paddle.subtract",
    "paddle._C_ops.transpose": "paddle.transpose",
}

_MANUAL_ARGUMENT_NAMES = {
    f"paddle.Tensor.{method}": ("self", "y") for method in _TENSOR_BINARY_METHODS
}
_MANUAL_ARGUMENT_NAMES.update(
    {
        "paddle.Tensor.clone": ("self",),
        "paddle.Tensor.detach": ("self",),
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
        "paddle._C_ops.matmul_grad": (
            "x",
            "y",
            "dout",
            "transpose_x",
            "transpose_y",
        ),
        "paddle._C_ops.squared_l2_norm": ("x",),
        "paddle._C_ops.swiglu_grad": ("x", "y", "dout"),
        "paddle._C_ops._run_custom_op": (
            "op_name",
            "arg1",
            "arg2",
            "arg3",
            "arg4",
        ),
        "paddle._C_ops.uniform": ("shape", "dtype", "min", "max", "seed", "place"),
    }
)
MANUAL_ARGUMENT_NAMES = MappingProxyType(_MANUAL_ARGUMENT_NAMES)
del _MANUAL_ARGUMENT_NAMES

_MANUAL_ARGUMENT_DEFAULTS = {
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

_SIGNATURE_CACHE: dict[str, inspect.Signature | None] = {}


def resolve_paddle_api(api_name: str) -> Callable[..., Any]:
    if not api_name.startswith("paddle."):
        raise ValueError(f"Invalid Paddle API name {api_name!r}")
    api: Any = paddle
    for component in api_name.split(".")[1:]:
        api = getattr(api, component)
    return api


def _signature_for(api_name: str, api: Callable[..., Any] | Any | None) -> inspect.Signature | None:
    if api is not None:
        try:
            return inspect.signature(api)
        except (TypeError, ValueError):
            return None
    if api_name not in _SIGNATURE_CACHE:
        try:
            _SIGNATURE_CACHE[api_name] = inspect.signature(resolve_paddle_api(api_name))
        except (TypeError, ValueError):
            _SIGNATURE_CACHE[api_name] = None
    return _SIGNATURE_CACHE[api_name]


def _bind_manual_arguments(
    api_name: str,
    names: Sequence[str],
    positional: Sequence[Any],
    keyword: Mapping[str, Any],
) -> OrderedDict[str, Any]:
    defaults = _MANUAL_ARGUMENT_DEFAULTS.get(api_name, {})
    signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults.get(name, inspect.Parameter.empty),
            )
            for name in names
        ]
    )
    bound = signature.bind(*positional, **keyword)
    bound.apply_defaults()
    return OrderedDict(bound.arguments)


def _normalize_tensor_self_argument(api_name: str, bound: OrderedDict[str, Any]) -> None:
    if not api_name.startswith("paddle.Tensor."):
        return
    if "self" in bound:
        receiver_name = "self"
    elif "x" not in bound and bound:
        receiver_name = next(iter(bound))
    else:
        return
    if receiver_name != "x":
        bound["x"] = bound.pop(receiver_name)
        bound.move_to_end("x", last=False)


def _normalize_dtype_arguments(bound: OrderedDict[str, Any]) -> None:
    for name, value in tuple(bound.items()):
        if "dtype" in name or isinstance(value, paddle.dtype):
            bound[name] = to_torch_dtype(value, strict=False)


def _finalize_bound_arguments(api_name: str, bound: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
    _normalize_tensor_self_argument(api_name, bound)
    _normalize_dtype_arguments(bound)
    return bound


def bind_paddle_arguments(
    api_name: str,
    positional: Sequence[Any],
    keyword: Mapping[str, Any],
    *,
    api: Callable[..., Any] | Any | None = None,
) -> OrderedDict[str, Any]:
    """Bind one Paddle invocation to the named inputs used by generated Torch code."""
    if api_name in ("paddle.Tensor.view", "paddle.view"):
        rest = positional[1:]
        if len(rest) > 1 and all(isinstance(value, int) for value in rest):
            unexpected = set(keyword) - {"name"}
            if unexpected:
                names = ", ".join(sorted(unexpected))
                raise TypeError(f"got unexpected keyword arguments: {names}")
            return _finalize_bound_arguments(
                api_name,
                OrderedDict(x=positional[0], shape_or_dtype=list(rest)),
            )

    if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
        rest = positional[1:]
        if rest and all(isinstance(value, int) for value in rest):
            unexpected = set(keyword) - {"name"}
            if unexpected:
                names = ", ".join(sorted(unexpected))
                raise TypeError(f"got unexpected keyword arguments: {names}")
            return _finalize_bound_arguments(
                api_name,
                OrderedDict(x=positional[0], shape=list(rest)),
            )

    manual_names = MANUAL_ARGUMENT_NAMES.get(api_name)
    if manual_names is not None:
        return _finalize_bound_arguments(
            api_name,
            _bind_manual_arguments(api_name, manual_names, positional, keyword),
        )

    signature = _signature_for(api_name, api)
    if signature is None:
        alias_name = _COPS_PUBLIC_ALIASES.get(api_name)
        if alias_name is None:
            raise ValueError(f"API {api_name} has no argument binding contract")
        signature = _signature_for(alias_name, None)
        if signature is None:
            raise ValueError(f"API {alias_name} has no inspectable signature")
        signature = signature.replace(
            parameters=[
                parameter for parameter in signature.parameters.values() if parameter.name != "name"
            ]
        )
    bound_arguments = signature.bind(*positional, **keyword)
    bound_arguments.apply_defaults()
    bound = OrderedDict(bound_arguments.arguments)
    bound.pop("name", None)
    if api_name == "paddle.Tensor.item":
        bound["indices"] = bound.pop("args")
    if api_name == "paddle.arange" and bound["end"] is None:
        bound["end"] = bound["start"]
        bound["start"] = 0
    if api_name in ("paddle.topk", "paddle.Tensor.topk") and bound.get("axis") is None:
        bound["axis"] = -1
    if api_name in ("paddle.gather", "paddle.Tensor.gather") and bound.get("axis") is None:
        bound["axis"] = 0
    return _finalize_bound_arguments(api_name, bound)


__all__ = ["MANUAL_ARGUMENT_NAMES", "bind_paddle_arguments", "resolve_paddle_api"]
