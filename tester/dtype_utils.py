from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy
import paddle


@lru_cache(maxsize=1)
def _torch_dtype_maps():
    """仅在 Torch 路径首次需要 dtype 转换时构造映射。"""
    # 本模块也被 Paddle-only 的 TensorConfig 导入，因此 Torch 必须停留在调用边界内。
    import torch

    # 名称映射同时接受配置规范名和历史简写，不能只保留 torch.dtype 的标准拼写。
    by_name = MappingProxyType(
        {
            "bool": torch.bool,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "uint16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp64": torch.float64,
            "bf16": torch.bfloat16,
            "float": torch.float64,
            "double": torch.float64,
            "int": torch.int64,
        }
    )
    # 值映射覆盖 NumPy、Paddle VarType 和 Python scalar type，保持原转换兼容面。
    by_value = MappingProxyType(
        {
            numpy.float32: torch.float32,
            paddle.float32: torch.float32,
            paddle.base.libpaddle.VarDesc.VarType.FP32: torch.float32,
            numpy.float16: torch.float16,
            paddle.float16: torch.float16,
            paddle.base.libpaddle.VarDesc.VarType.FP16: torch.float16,
            numpy.float64: torch.float64,
            paddle.float64: torch.float64,
            paddle.base.libpaddle.VarDesc.VarType.FP64: torch.float64,
            float: torch.float64,
            numpy.int16: torch.int16,
            paddle.int16: torch.int16,
            paddle.base.libpaddle.VarDesc.VarType.INT16: torch.int16,
            numpy.int8: torch.int8,
            paddle.int8: torch.int8,
            paddle.base.libpaddle.VarDesc.VarType.INT8: torch.int8,
            numpy.bool_: torch.bool,
            paddle.bool: torch.bool,
            paddle.base.libpaddle.VarDesc.VarType.BOOL: torch.bool,
            bool: torch.bool,
            numpy.uint16: torch.bfloat16,
            paddle.bfloat16: torch.bfloat16,
            paddle.base.libpaddle.VarDesc.VarType.BF16: torch.bfloat16,
            numpy.uint8: torch.uint8,
            paddle.uint8: torch.uint8,
            paddle.base.libpaddle.VarDesc.VarType.UINT8: torch.uint8,
            numpy.int32: torch.int32,
            paddle.int32: torch.int32,
            paddle.base.libpaddle.VarDesc.VarType.INT32: torch.int32,
            numpy.int64: torch.int64,
            paddle.int64: torch.int64,
            paddle.base.libpaddle.VarDesc.VarType.INT64: torch.int64,
            int: torch.int64,
            numpy.complex64: torch.complex64,
            paddle.complex64: torch.complex64,
            paddle.base.libpaddle.VarDesc.VarType.COMPLEX64: torch.complex64,
            numpy.complex128: torch.complex128,
            paddle.complex128: torch.complex128,
            paddle.base.libpaddle.VarDesc.VarType.COMPLEX128: torch.complex128,
            complex: torch.complex128,
        }
    )
    # MappingProxy 防止后续调用方修改全局 dtype 协议，lru_cache 避免重复构造。
    return torch, by_name, by_value


def _dtype_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (paddle.dtype, paddle.base.libpaddle.VarDesc.VarType)):
        return str(value)
    return None


def to_torch_dtype(value: Any, *, strict: bool = True) -> Any:
    # 调用该 API 即进入 Torch 语义；Paddle-only 路径不得为了类型预判调用它。
    torch, by_name, by_value = _torch_dtype_maps()
    if value is None or isinstance(value, torch.dtype):
        return value
    try:
        dtype = by_value.get(value)
    except TypeError:
        dtype = None
    if dtype is not None:
        return dtype
    name = _dtype_name(value)
    if name is not None:
        dtype_key = name.split(".")[-1].lower()
        dtype = by_name.get(dtype_key)
        if dtype is not None:
            return dtype
    if strict:
        raise ValueError(f"Unsupported dtype: {value}")
    return value
