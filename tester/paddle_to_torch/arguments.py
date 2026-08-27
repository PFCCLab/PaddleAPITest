from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import paddle
from tester.api_config.dtype_utils import to_torch_dtype
from tester.api_config.parameter_binding import bind_input_parameters, resolve_input_api

# bool 是 int 的子类，必须与 int 一起排除，否则 False/True 会被当成 BOOL/INT16 码。
_NON_DTYPE_SCALAR_TYPES = (bool, int, float, complex)


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
    if isinstance(value, _NON_DTYPE_SCALAR_TYPES):
        # 参数名只能说明“可能是 dtype”：paddle.Tensor.view 的 shape_or_dtype 是
        # Sequence[int] | DTypeLike 联合参数，形状分量同样会走到这里。而 VarType
        # 枚举与 int 相等且同 hash，to_torch_dtype 的 by_value 查表会把 0~6 当成
        # dtype 码（1 -> torch.int16），把 view([1, 4096, -1]) 破坏成
        # view([torch.int16, 4096, -1])。因此纯 Python 数值一律按原值透传，
        # 真正的 dtype 值仍由下面的名称/类型分支处理。
        return value
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
