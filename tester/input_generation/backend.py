"""Backend abstractions for generated input values."""

from __future__ import annotations

import numbers
import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy
from tester.dtype_utils import to_torch_dtype

from .value_generators import (
    INPUT_NUMPY_RANDOM_STATE,
    derive_input_stream_seed,
    resolve_input_dtype,
)


class InputBackend(Protocol):
    """Value construction interface used by input-generation rules."""

    name: str

    def commit(self) -> None: ...

    def resolve_input_dtype(self, dtype: str) -> str: ...

    def random(self, shape=None, dtype=None): ...

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None): ...

    def randint(self, low, high=None, shape=None, dtype=None): ...

    def randn(self, *shape, dtype=None): ...

    def choice(self, values, shape=None, replace=True, p=None): ...

    def asarray(self, value, dtype=None, copy=True): ...

    def cast(self, value, dtype): ...

    def reshape(self, value, shape): ...

    def flatten(self, value): ...

    def view_dtype(self, value, dtype): ...

    def arange(self, *args, dtype=None): ...

    def zeros(self, shape, dtype=None): ...

    def ones(self, shape, dtype=None): ...

    def full(self, shape, fill_value, dtype=None): ...

    def where(self, condition, x, y): ...

    def minimum(self, x, y): ...

    def maximum(self, x, y): ...

    def abs(self, value): ...

    def sort(self, value): ...

    def cumsum(self, value, axis=None): ...

    def sum(self, value, axis=None, keepdims=False): ...

    def power(self, x, y): ...

    def count_nonzero(self, value): ...

    def nonzero(self, value): ...

    def prod(self, value): ...

    def ndindex(self, shape): ...

    def einsum(self, expression, *operands): ...

    def dot(self, left, right): ...

    def matmul(self, left, right): ...

    def swapaxes(self, value, axis1, axis2): ...

    def triu(self, value, k=0): ...

    def tril(self, value, k=0): ...

    def conj(self, value): ...

    def eye(self, size, dtype=None): ...

    def ascontiguousarray(self, value): ...

    def finfo(self, dtype): ...


@dataclass
class NumPyInputBackend:
    """NumPy implementation of the input-generation backend."""

    input_random_state: object = INPUT_NUMPY_RANDOM_STATE

    name = "numpy"

    def commit(self) -> None:
        commit = getattr(self.input_random_state, "commit", None)
        if commit is not None:
            commit()

    def resolve_input_dtype(self, dtype: str) -> str:
        return resolve_input_dtype(dtype)

    def _storage_dtype(self, dtype):
        if dtype is None:
            return None
        if isinstance(dtype, str):
            dtype_name = dtype.replace("paddle.", "")
        else:
            try:
                dtype_name = numpy.dtype(dtype).name
            except TypeError:
                dtype_name = str(dtype).split(".")[-1]
        return resolve_input_dtype(dtype_name)

    def random(self, shape=None, dtype=None):
        value = self.input_random_state.random(shape)
        storage_dtype = self._storage_dtype(dtype)
        return numpy.asarray(value).astype(storage_dtype) if storage_dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        value = self.input_random_state.uniform(low=low, high=high, shape=shape)
        storage_dtype = self._storage_dtype(dtype)
        return numpy.asarray(value).astype(storage_dtype) if storage_dtype is not None else value

    def randint(self, low, high=None, shape=None, dtype=None):
        value = self.input_random_state.randint(low, high, shape=shape)
        storage_dtype = self._storage_dtype(dtype)
        return numpy.asarray(value).astype(storage_dtype) if storage_dtype is not None else value

    def randn(self, *shape, dtype=None):
        value = self.input_random_state.randn(*shape)
        storage_dtype = self._storage_dtype(dtype)
        return numpy.asarray(value).astype(storage_dtype) if storage_dtype is not None else value

    def choice(self, values, shape=None, replace=True, p=None):
        return self.input_random_state.choice(values, shape=shape, replace=replace, p=p)

    def asarray(self, value, dtype=None, copy=True):
        return numpy.array(value, dtype=self._storage_dtype(dtype), copy=copy)

    def cast(self, value, dtype):
        return numpy.asarray(value).astype(self._storage_dtype(dtype))

    def reshape(self, value, shape):
        return numpy.reshape(value, shape)

    def flatten(self, value):
        return numpy.reshape(value, -1)

    def view_dtype(self, value, dtype):
        return numpy.asarray(value).view(dtype)

    def arange(self, *args, dtype=None):
        return numpy.arange(*args, dtype=self._storage_dtype(dtype))

    def zeros(self, shape, dtype=None):
        return numpy.zeros(shape, dtype=self._storage_dtype(dtype))

    def ones(self, shape, dtype=None):
        return numpy.ones(shape, dtype=self._storage_dtype(dtype))

    def full(self, shape, fill_value, dtype=None):
        return numpy.full(shape, fill_value, dtype=self._storage_dtype(dtype))

    def where(self, condition, x, y):
        return numpy.where(condition, x, y)

    def minimum(self, x, y):
        return numpy.minimum(x, y)

    def maximum(self, x, y):
        return numpy.maximum(x, y)

    def abs(self, value):
        return numpy.abs(value)

    def sort(self, value):
        return numpy.sort(value)

    def cumsum(self, value, axis=None):
        return numpy.cumsum(value, axis=axis)

    def sum(self, value, axis=None, keepdims=False):
        return numpy.sum(value, axis=axis, keepdims=keepdims)

    def power(self, x, y):
        return numpy.power(x, y)

    def count_nonzero(self, value):
        return numpy.count_nonzero(value)

    def nonzero(self, value):
        return numpy.nonzero(value)

    def prod(self, value):
        return numpy.prod(value)

    def ndindex(self, shape):
        return numpy.ndindex(shape)

    def einsum(self, expression, *operands):
        return numpy.einsum(expression, *operands)

    def dot(self, left, right):
        return numpy.dot(left, right)

    def matmul(self, left, right):
        return numpy.matmul(left, right)

    def swapaxes(self, value, axis1, axis2):
        return numpy.swapaxes(value, axis1, axis2)

    def triu(self, value, k=0):
        return numpy.triu(value, k=k)

    def tril(self, value, k=0):
        return numpy.tril(value, k=k)

    def conj(self, value):
        return numpy.conj(value)

    def eye(self, size, dtype=None):
        return numpy.eye(size, dtype=self._storage_dtype(dtype))

    def ascontiguousarray(self, value):
        return numpy.ascontiguousarray(value)

    def finfo(self, dtype):
        return numpy.finfo(self._storage_dtype(dtype))


@dataclass
class TorchInputBackend(NumPyInputBackend):
    """Torch implementation of the input-generation backend."""

    device: str = "cpu"
    name = "torch"
    _generator: object = field(init=False, repr=False)

    def __post_init__(self):
        seed = derive_input_stream_seed(
            getattr(self.input_random_state, "seed", 0),
            getattr(self.input_random_state, "config_fingerprint", ""),
            modulus=2**63,
        )
        torch = self._torch()
        self._device = torch.device(self.device)
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(seed)

    def _torch(self):
        import torch

        return torch

    def _torch_shape(self, shape):
        return _normalize_shape(shape, scalar_empty=False)

    def _torch_dtype(self, dtype):
        if isinstance(dtype, str):
            torch = self._torch()
            unsigned_dtype = {
                "uint16": torch.uint16,
                "uint32": torch.uint32,
                "uint64": torch.uint64,
            }.get(dtype.replace("paddle.", ""))
            if unsigned_dtype is not None:
                return unsigned_dtype
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype is None:
            return None
        return to_torch_dtype(storage_dtype)

    def _resolve_torch_float_dtype(self, dtype):
        torch = self._torch()
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype == "float64":
            return torch.float64
        return torch.float32

    def random(self, shape=None, dtype=None):
        torch = self._torch()
        torch_shape = self._torch_shape(shape)
        value = torch.rand(
            torch_shape,
            dtype=torch.float32,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        torch = self._torch()
        torch_shape = self._torch_shape(shape)
        value = torch.empty(
            torch_shape,
            dtype=self._resolve_torch_float_dtype(dtype),
            device=self._device,
        ).uniform_(
            float(low),
            float(high),
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randint(self, low, high=None, shape=None, dtype=None):
        torch = self._torch()
        torch_shape = self._torch_shape(shape)
        if high is None:
            low, high = 0, low
        value = torch.randint(
            int(low),
            int(high),
            torch_shape,
            dtype=torch.int64,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randn(self, *shape, dtype=None):
        torch = self._torch()
        value = torch.randn(
            tuple(shape),
            dtype=torch.float32,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def choice(self, values, shape=None, replace=True, p=None):
        torch = self._torch()
        scalar, torch_shape, num_samples = _choice_shape(shape, scalar_empty=False)

        if isinstance(values, numbers.Integral):
            population = torch.arange(int(values), device=self._device)
        else:
            population = self.asarray(values)

        if p is not None:
            weights = self.asarray(p, dtype="float64")
            indices = torch.multinomial(
                weights, num_samples, replacement=replace, generator=self._generator
            )
        elif replace:
            indices = torch.randint(
                0,
                len(population),
                (num_samples,),
                dtype=torch.int64,
                device=self._device,
                generator=self._generator,
            )
        else:
            if num_samples > len(population):
                raise ValueError("Cannot take a larger sample than population when replace=False")
            indices = torch.randperm(
                len(population),
                dtype=torch.int64,
                device=self._device,
                generator=self._generator,
            )[:num_samples]

        result = population[indices]
        if scalar:
            return result.item()
        return self.reshape(result, torch_shape)

    def asarray(self, value, dtype=None, copy=True):
        torch = self._torch()
        torch_dtype = self._torch_dtype(dtype)
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self._device, dtype=torch_dtype)
            return tensor.clone() if copy else tensor
        tensor = torch.as_tensor(value, dtype=torch_dtype, device=self._device)
        return tensor.clone() if copy else tensor

    def cast(self, value, dtype):
        torch_dtype = self._torch_dtype(dtype)
        if torch_dtype is None:
            return value
        return self.asarray(value, copy=False).to(dtype=torch_dtype)

    def reshape(self, value, shape):
        return self.asarray(value, copy=False).reshape(self._torch_shape(shape))

    def flatten(self, value):
        return self.asarray(value, copy=False).flatten()

    def view_dtype(self, value, dtype):
        return self.asarray(value, copy=False).view(self._torch_dtype(dtype))

    def arange(self, *args, dtype=None):
        torch = self._torch()
        return torch.arange(*args, dtype=self._torch_dtype(dtype), device=self._device)

    def zeros(self, shape, dtype=None):
        torch = self._torch()
        return torch.zeros(
            self._torch_shape(shape),
            dtype=self._torch_dtype(dtype),
            device=self._device,
        )

    def ones(self, shape, dtype=None):
        torch = self._torch()
        return torch.ones(
            self._torch_shape(shape),
            dtype=self._torch_dtype(dtype),
            device=self._device,
        )

    def full(self, shape, fill_value, dtype=None):
        torch = self._torch()
        return torch.full(
            self._torch_shape(shape),
            fill_value,
            dtype=self._torch_dtype(dtype),
            device=self._device,
        )

    def where(self, condition, x, y):
        torch = self._torch()
        return torch.where(
            self.asarray(condition, copy=False).bool(),
            self.asarray(x, copy=False),
            self.asarray(y, copy=False),
        )

    def minimum(self, x, y):
        torch = self._torch()
        return torch.minimum(self.asarray(x, copy=False), self.asarray(y, copy=False))

    def maximum(self, x, y):
        torch = self._torch()
        return torch.maximum(self.asarray(x, copy=False), self.asarray(y, copy=False))

    def abs(self, value):
        torch = self._torch()
        return torch.abs(self.asarray(value, copy=False))

    def sort(self, value):
        torch = self._torch()
        return torch.sort(self.asarray(value, copy=False)).values

    def cumsum(self, value, axis=None):
        if axis is None:
            value = self.asarray(value, copy=False).reshape(-1)
            axis = 0
        return self.asarray(value, copy=False).cumsum(dim=axis)

    def sum(self, value, axis=None, keepdims=False):
        return self.asarray(value, copy=False).sum(dim=axis, keepdim=keepdims)

    def power(self, x, y):
        torch = self._torch()
        return torch.pow(self.asarray(x, copy=False), self.asarray(y, copy=False))

    def count_nonzero(self, value):
        torch = self._torch()
        return int(torch.count_nonzero(self.asarray(value, copy=False)).item())

    def nonzero(self, value):
        torch = self._torch()
        return torch.nonzero(self.asarray(value, copy=False), as_tuple=True)

    def prod(self, value):
        torch = self._torch()
        if isinstance(value, torch.Tensor):
            return value.prod()
        result = 1
        for item in value:
            result *= int(item)
        return result

    def ndindex(self, shape):
        return numpy.ndindex(shape)

    def einsum(self, expression, *operands):
        torch = self._torch()
        return torch.einsum(expression, *(self.asarray(item, copy=False) for item in operands))

    def dot(self, left, right):
        torch = self._torch()
        return torch.dot(self.asarray(left, copy=False), self.asarray(right, copy=False))

    def matmul(self, left, right):
        torch = self._torch()
        return torch.matmul(self.asarray(left, copy=False), self.asarray(right, copy=False))

    def swapaxes(self, value, axis1, axis2):
        return self.asarray(value, copy=False).swapaxes(axis1, axis2)

    def triu(self, value, k=0):
        torch = self._torch()
        return torch.triu(self.asarray(value, copy=False), diagonal=k)

    def tril(self, value, k=0):
        torch = self._torch()
        return torch.tril(self.asarray(value, copy=False), diagonal=k)

    def conj(self, value):
        return self.asarray(value, copy=False).conj()

    def eye(self, size, dtype=None):
        torch = self._torch()
        return torch.eye(size, dtype=self._torch_dtype(dtype), device=self._device)

    def ascontiguousarray(self, value):
        return self.asarray(value, copy=False).contiguous()

    def finfo(self, dtype):
        torch = self._torch()
        return torch.finfo(self._torch_dtype(dtype))


@dataclass
class PaddleInputBackend(NumPyInputBackend):
    """Paddle implementation of the input-generation backend."""

    device: str = "cpu"
    name = "paddle"

    def __post_init__(self):
        # 随机值由 config-local NumPy 流生成，再物化到 Paddle，避免 paddle.seed 污染全局流。
        paddle = self._paddle()
        self._place = paddle.CPUPlace()
        if self.device.startswith(("gpu", "cuda")):
            device_id = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            self._place = paddle.CUDAPlace(device_id)

    def _paddle(self):
        import paddle

        return paddle

    def _paddle_shape(self, shape):
        return _normalize_shape(shape, scalar_empty=True)

    def _paddle_dtype(self, dtype):
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype is None:
            return None
        return getattr(self._paddle(), storage_dtype)

    def random(self, shape=None, dtype=None):
        return self.asarray(super().random(shape=shape, dtype=dtype), dtype=dtype, copy=False)

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        return self.asarray(
            super().uniform(low=low, high=high, shape=shape, dtype=dtype),
            dtype=dtype,
            copy=False,
        )

    def randint(self, low, high=None, shape=None, dtype=None):
        return self.asarray(
            super().randint(low, high, shape=shape, dtype=dtype),
            dtype=dtype,
            copy=False,
        )

    def randn(self, *shape, dtype=None):
        return self.asarray(super().randn(*shape, dtype=dtype), dtype=dtype, copy=False)

    def choice(self, values, shape=None, replace=True, p=None):
        value = super().choice(values, shape=shape, replace=replace, p=p)
        return self.asarray(value, copy=False)

    def asarray(self, value, dtype=None, copy=True):
        paddle = self._paddle()
        paddle_dtype = self._paddle_dtype(dtype)
        if isinstance(value, paddle.Tensor):
            tensor = value._copy_to(self._place, False)
            if paddle_dtype is not None and tensor.dtype != paddle_dtype:
                tensor = paddle.cast(tensor, dtype=paddle_dtype)
            return tensor.clone() if copy else tensor
        tensor = paddle.to_tensor(value, dtype=paddle_dtype, place=self._place)
        return tensor.clone() if copy else tensor

    def cast(self, value, dtype):
        paddle_dtype = self._paddle_dtype(dtype)
        if paddle_dtype is None:
            return value
        return self._paddle().cast(self.asarray(value, copy=False), dtype=paddle_dtype)

    def reshape(self, value, shape):
        return self._paddle().reshape(self.asarray(value, copy=False), self._paddle_shape(shape))

    def flatten(self, value):
        return self._paddle().flatten(self.asarray(value, copy=False))

    def view_dtype(self, value, dtype):
        return self.asarray(value, copy=False).view(self._paddle_dtype(dtype))

    def arange(self, *args, dtype=None):
        return self._paddle().arange(*args, dtype=self._paddle_dtype(dtype), device=self.device)

    def zeros(self, shape, dtype=None):
        return self._paddle().zeros(
            self._paddle_shape(shape),
            dtype=self._paddle_dtype(dtype),
            device=self.device,
        )

    def ones(self, shape, dtype=None):
        return self._paddle().ones(
            self._paddle_shape(shape),
            dtype=self._paddle_dtype(dtype),
            device=self.device,
        )

    def full(self, shape, fill_value, dtype=None):
        return self._paddle().full(
            self._paddle_shape(shape),
            fill_value,
            dtype=self._paddle_dtype(dtype),
            device=self.device,
        )

    def where(self, condition, x, y):
        paddle = self._paddle()
        x_tensor = self.asarray(x, copy=False)
        y_tensor = self.asarray(y, copy=False)
        result_dtype = x_tensor.dtype
        if result_dtype in {paddle.int8, paddle.int16, paddle.uint8}:
            result = paddle.where(
                self.asarray(condition, copy=False).astype("bool"),
                paddle.cast(x_tensor, "int32"),
                paddle.cast(y_tensor, "int32"),
            )
            return paddle.cast(result, result_dtype)
        return paddle.where(self.asarray(condition, copy=False).astype("bool"), x_tensor, y_tensor)

    def minimum(self, x, y):
        paddle = self._paddle()
        x_tensor = self.asarray(x, copy=False)
        y_tensor = self.asarray(y, copy=False)
        result_dtype = x_tensor.dtype
        if result_dtype in {paddle.bool, paddle.int8, paddle.int16, paddle.uint8}:
            result = paddle.minimum(
                paddle.cast(x_tensor, "int32"),
                paddle.cast(y_tensor, "int32"),
            )
            return paddle.cast(result, result_dtype)
        return paddle.minimum(x_tensor, y_tensor)

    def maximum(self, x, y):
        paddle = self._paddle()
        x_tensor = self.asarray(x, copy=False)
        y_tensor = self.asarray(y, copy=False)
        result_dtype = x_tensor.dtype
        if result_dtype in {paddle.bool, paddle.int8, paddle.int16, paddle.uint8}:
            result = paddle.maximum(
                paddle.cast(x_tensor, "int32"),
                paddle.cast(y_tensor, "int32"),
            )
            return paddle.cast(result, result_dtype)
        return paddle.maximum(x_tensor, y_tensor)

    def abs(self, value):
        return self._paddle().abs(self.asarray(value, copy=False))

    def sort(self, value):
        return self._paddle().sort(self.asarray(value, copy=False))

    def cumsum(self, value, axis=None):
        return self._paddle().cumsum(self.asarray(value, copy=False), axis=axis)

    def sum(self, value, axis=None, keepdims=False):
        return self._paddle().sum(self.asarray(value, copy=False), axis=axis, keepdim=keepdims)

    def power(self, x, y):
        paddle = self._paddle()
        return paddle.pow(self.asarray(x, copy=False), self.asarray(y, copy=False))

    def count_nonzero(self, value):
        return int(self._paddle().count_nonzero(self.asarray(value, copy=False)).item())

    def nonzero(self, value):
        return self._paddle().nonzero(self.asarray(value, copy=False), as_tuple=True)

    def prod(self, value):
        paddle = self._paddle()
        if isinstance(value, paddle.Tensor):
            return value.prod()
        result = 1
        for item in value:
            result *= int(item)
        return result

    def ndindex(self, shape):
        return numpy.ndindex(shape)

    def einsum(self, expression, *operands):
        paddle = self._paddle()
        return paddle.einsum(expression, *(self.asarray(item, copy=False) for item in operands))

    def dot(self, left, right):
        paddle = self._paddle()
        return paddle.dot(self.asarray(left, copy=False), self.asarray(right, copy=False))

    def matmul(self, left, right):
        paddle = self._paddle()
        return paddle.matmul(self.asarray(left, copy=False), self.asarray(right, copy=False))

    def swapaxes(self, value, axis1, axis2):
        return self._paddle().swapaxes(self.asarray(value, copy=False), axis1, axis2)

    def triu(self, value, k=0):
        return self._paddle().triu(self.asarray(value, copy=False), diagonal=k)

    def tril(self, value, k=0):
        return self._paddle().tril(self.asarray(value, copy=False), diagonal=k)

    def conj(self, value):
        return self._paddle().conj(self.asarray(value, copy=False))

    def eye(self, size, dtype=None):
        return self._paddle().eye(size, dtype=self._paddle_dtype(dtype), device=self.device)

    def ascontiguousarray(self, value):
        return self.asarray(value, copy=False).contiguous()

    def finfo(self, dtype):
        return numpy.finfo(self._storage_dtype(dtype))


INPUT_BACKEND_ENV_VAR = "PADDLEAPITEST_INPUT_BACKEND"
# Only used when no config-local RNG is available, such as legacy TensorConfig helpers.
_DEFAULT_INPUT_BACKENDS = {}
_TRUE_VALUES = {"true", "1", "yes", "y"}
_SUPPORTED_INPUT_BACKENDS = frozenset({"numpy", "torch", "paddle"})
_PADDLE_NATIVE_GPU_MODES = frozenset({"paddle_only", "paddle_cinn", "paddle_gpu_performance"})
_CUSTOM_DEVICE_MODES = frozenset({"paddle_custom_device", "custom_device_vs_gpu"})


@dataclass(frozen=True)
class InputBackendPolicy:
    """一次运行已解析且可直接执行的输入 backend 策略。"""

    requested: str | None
    resolved: str
    device: str

    @property
    def logical_device(self):
        return "cpu" if self.resolved == "numpy" else self.device


def _env_flag(name, default="False") -> bool:
    return os.getenv(name, default).lower() in _TRUE_VALUES


def _normalize_shape(shape, *, scalar_empty):
    if shape is None:
        return [] if scalar_empty else ()
    if isinstance(shape, numbers.Integral):
        return [int(shape)] if scalar_empty else (int(shape),)
    return list(shape) if scalar_empty else tuple(shape)


def _choice_shape(shape, *, scalar_empty):
    scalar = shape is None
    normalized = _normalize_shape(shape, scalar_empty=scalar_empty)
    return scalar, normalized, 1 if scalar else int(numpy.prod(normalized))


def resolve_input_backend_policy(*, use_gpu_mode, operation_mode=None, requested=None):
    """按显式 override、测试模式和 GPU mode 解析唯一 backend 策略。"""
    normalized_requested = requested.strip().lower() if requested is not None else None
    if normalized_requested not in _SUPPORTED_INPUT_BACKENDS | {None}:
        # 显式配置错误必须在 worker 或 GPU runtime 启动前失败。
        raise ValueError(f"unsupported input generation backend: {requested!r}")

    if normalized_requested is not None:
        resolved = normalized_requested
    elif not use_gpu_mode or operation_mode in _CUSTOM_DEVICE_MODES:
        resolved = "numpy"
    elif operation_mode in _PADDLE_NATIVE_GPU_MODES:
        resolved = "paddle"
    else:
        # accuracy/stable、Torch 性能和旧直接调用继续使用 Torch 私有 generator。
        resolved = "torch"

    # device 是逻辑值生成位置，不代表最终 Paddle/Torch 算子设备。
    device = {
        "numpy": "cpu",
        "torch": "cuda:0" if use_gpu_mode else "cpu",
        "paddle": "gpu:0" if use_gpu_mode else "cpu",
    }[resolved]
    return InputBackendPolicy(normalized_requested, resolved, device)


def resolve_input_backend_name() -> str:
    """兼容直接模块调用，返回只基于环境解析的 backend 名称。"""
    return resolve_input_backend_policy(
        use_gpu_mode=_env_flag("USE_GPU_MODE"),
        requested=os.environ.get(INPUT_BACKEND_ENV_VAR),
    ).resolved


def create_input_backend(
    input_random_state=INPUT_NUMPY_RANDOM_STATE,
    policy: InputBackendPolicy | None = None,
) -> InputBackend:
    # 无 policy 只服务历史直接调用；engine 路径必须传入 runtime config 的最终策略。
    policy = policy or resolve_input_backend_policy(
        use_gpu_mode=_env_flag("USE_GPU_MODE"),
        requested=os.environ.get(INPUT_BACKEND_ENV_VAR),
    )
    normalized = policy.resolved
    device = policy.device

    cache_key = (normalized, device)
    if input_random_state is INPUT_NUMPY_RANDOM_STATE and cache_key in _DEFAULT_INPUT_BACKENDS:
        return _DEFAULT_INPUT_BACKENDS[cache_key]

    if normalized == "numpy":
        input_backend = NumPyInputBackend(input_random_state)
    elif normalized == "torch":
        input_backend = TorchInputBackend(input_random_state, device=device)
    else:
        input_backend = PaddleInputBackend(input_random_state, device=device)

    if input_random_state is INPUT_NUMPY_RANDOM_STATE:
        _DEFAULT_INPUT_BACKENDS[cache_key] = input_backend
    return input_backend
