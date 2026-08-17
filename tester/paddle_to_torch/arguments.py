from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import paddle
from tester.api_config.dtype_utils import to_torch_dtype
from tester.api_config.parameter_binding import bind_input_parameters, resolve_input_api


def resolve_paddle_api(api_name: str) -> Callable[..., Any]:
    return resolve_input_api(api_name)


def _normalize_dtype_value(name: str, value: Any) -> Any:
    # Tensor.to 等可变参数 API 会把 dtype 放进嵌套的 args/kwargs 容器。
    # 递归转换保证 dtype 适配仍只属于本层，而接收者绑定继续由 binding.py 负责。
    if isinstance(value, Mapping):
        return type(value)(
            (key, _normalize_dtype_value(str(key), item)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_normalize_dtype_value(name, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_dtype_value(name, item) for item in value)
    if "dtype" in name or isinstance(value, paddle.dtype):
        return to_torch_dtype(value, strict=False)
    return value


def _normalize_dtype_arguments(bound: OrderedDict[str, Any]) -> None:
    for name, value in tuple(bound.items()):
        bound[name] = _normalize_dtype_value(name, value)


def bind_paddle_arguments(
    api_name: str,
    positional: Sequence[Any],
    keyword: Mapping[str, Any],
    *,
    api: Callable[..., Any] | Any | None = None,
) -> OrderedDict[str, Any]:
    """Bind one Paddle invocation to the named inputs used by generated Torch code."""
    # 共享绑定层统一负责签名、默认值和接收者协议，本层只做 Torch dtype 适配。
    binding = bind_input_parameters(
        api_name,
        positional,
        keyword,
        api=api,
        apply_defaults=True,
    )
    if binding.source == "unresolved":
        raise ValueError(f"API {api_name} has no argument binding contract")
    bound = OrderedDict(binding.arguments)
    if api_name == "paddle.Tensor.item":
        bound["indices"] = bound.pop("args")
    if api_name in ("paddle.topk", "paddle.Tensor.topk") and bound.get("axis") is None:
        bound["axis"] = -1
    if api_name in ("paddle.gather", "paddle.Tensor.gather") and bound.get("axis") is None:
        bound["axis"] = 0
    _normalize_dtype_arguments(bound)
    return bound


__all__ = ["bind_paddle_arguments", "resolve_paddle_api"]
