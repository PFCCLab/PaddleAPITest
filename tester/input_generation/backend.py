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
    InputConfigRandomState,
    derive_input_seed,
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


class InputBackendCapabilityError(ValueError):
    """输入 backend 无法按协议物化某个逻辑 dtype 或原语。"""


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
        # 逻辑 dtype 属于配置协议，生成原语只接收 backend 能稳定构造的 storage dtype。
        # BF16、FP8 和宽无符号整数因此必须在这一层统一降级，不能由各条规则自行判断。
        # 映射后的 dtype 仍需显式校验，避免把底层库偶然抛出的错误当作协议定义。
        if dtype is None:
            return None
        if isinstance(dtype, str):
            dtype_name = dtype.replace("paddle.", "")
        else:
            try:
                dtype_name = numpy.dtype(dtype).name
            except TypeError:
                dtype_name = str(dtype).split(".")[-1]
        storage_dtype = resolve_input_dtype(dtype_name)
        try:
            numpy.dtype(storage_dtype)
        except TypeError as err:
            raise InputBackendCapabilityError(
                f"{self.name} backend does not support input dtype {dtype_name!r} "
                f"(storage dtype {storage_dtype!r})"
            ) from err
        return storage_dtype

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
        return numpy.asarray(value).view(self._storage_dtype(dtype))

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
        # Torch 私有 Generator 只消费本 backend 的 stream，不触碰默认 generator。
        stream_kind = getattr(self.input_random_state, "stream_kind", "")
        if not stream_kind.startswith("output_grad:"):
            stream_kind = "torch"
        seed = derive_input_seed(
            getattr(self.input_random_state, "seed", 0),
            getattr(self.input_random_state, "config_fingerprint", ""),
            stream_kind,
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
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype is None:
            return None
        try:
            return to_torch_dtype(storage_dtype)
        except (AttributeError, TypeError, ValueError) as err:
            raise InputBackendCapabilityError(
                f"{self.name} backend does not support storage dtype {storage_dtype!r}"
            ) from err

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
        # Paddle 无稳定的逐实例 Generator；随机值改由独立 NumPy stream 生成。
        # 这里绝不能播种 Paddle 全局 generator，避免规则失败污染后续配置。
        stream_kind = getattr(self.input_random_state, "stream_kind", "")
        if not stream_kind.startswith("output_grad:"):
            stream_kind = "paddle"
        self._random_source = InputConfigRandomState(
            getattr(self.input_random_state, "seed", 0),
            getattr(self.input_random_state, "config_fingerprint", ""),
            stream_kind=stream_kind,
        )
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
        try:
            return getattr(self._paddle(), storage_dtype)
        except AttributeError as err:
            raise InputBackendCapabilityError(
                f"{self.name} backend does not support storage dtype {storage_dtype!r}"
            ) from err

    def _resolve_paddle_float_dtype(self, dtype):
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype == "float64":
            return "float64"
        return "float32"

    def random(self, shape=None, dtype=None):
        value = self.asarray(self._random_source.random(shape), dtype="float32", copy=False)
        return self.cast(value, dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        value = self.asarray(
            self._random_source.uniform(low=float(low), high=float(high), shape=shape),
            dtype=self._resolve_paddle_float_dtype(dtype),
            copy=False,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randint(self, low, high=None, shape=None, dtype=None):
        if high is None:
            low, high = 0, low
        value = self.asarray(
            self._random_source.randint(int(low), int(high), shape=shape),
            dtype="int64",
            copy=False,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randn(self, *shape, dtype=None):
        value = self.asarray(self._random_source.randn(*shape), dtype="float32", copy=False)
        return self.cast(value, dtype) if dtype is not None else value

    def choice(self, values, shape=None, replace=True, p=None):
        # choice 也只能消费局部 NumPy stream，禁止 multinomial/randperm 隐式播种。
        paddle = self._paddle()
        if isinstance(values, numbers.Integral):
            population = int(values)
        else:
            if isinstance(values, paddle.Tensor):
                # GPU Tensor 先复制到 CPU；该复制是数据搬运，不会触发随机算子。
                values = values.cpu() if "gpu" in str(values.place).lower() else values
                population = values.numpy()
            else:
                population = numpy.asarray(values)
        if p is not None and hasattr(p, "cpu"):
            p = p.cpu() if "gpu" in str(p.place).lower() else p
            if isinstance(p, paddle.Tensor):
                p = p.numpy()
        result = self._random_source.choice(
            population,
            shape=shape,
            replace=replace,
            p=None if p is None else numpy.asarray(p),
        )
        if shape is None:
            return result.item() if hasattr(result, "item") else result
        return self.asarray(result, copy=False)

    def asarray(self, value, dtype=None, copy=True):
        paddle = self._paddle()
        paddle_dtype = self._paddle_dtype(dtype)
        if isinstance(value, paddle.Tensor):
            same_place = str(value.place).lower() == str(self._place).lower()
            tensor = value if same_place else value._copy_to(self._place, False)
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
NUMPY_AUXILIARY_CACHE_ENV_VAR = "PADDLEAPITEST_CACHE_NUMPY_AUXILIARY"
# Only used when no config-local RNG is available, such as legacy TensorConfig helpers.
_DEFAULT_INPUT_BACKENDS = {}
_TRUE_VALUES = {"true", "1", "yes", "y"}
_VALID_INPUT_BACKENDS = frozenset({"numpy", "torch", "paddle"})
# 模式默认值由 policy 唯一拥有，worker 和物化层只能消费解析结果。
# cache 只控制辅助 NumPy 数据，不得再参与 backend 选择。
# 未声明模式的直接模块调用保留 NumPy/CPU 的历史默认行为。
_MODE_DEFAULT_BACKENDS = {
    "paddle_only": "paddle",
    "paddle_cinn": "paddle",
    "paddle_gpu_performance": "paddle",
    "paddle_custom_device": "paddle",
    "custom_device_vs_gpu": "paddle",
    "torch_gpu_performance": "torch",
    "paddle_torch_gpu_performance": "torch",
    "accuracy": "torch",
    "accuracy_dual_gpu": "torch",
    "accuracy_stable": "torch",
    "accuracy_stable_dual_gpu": "torch",
}
_GPU_NATIVE_MODES = frozenset(
    {
        "paddle_gpu_performance",
        "torch_gpu_performance",
        "paddle_torch_gpu_performance",
    }
)


def _env_flag(name, default="False") -> bool:
    return os.getenv(name, default).lower() in _TRUE_VALUES


def _cache_numpy_auxiliary_enabled() -> bool:
    """读取辅助 NumPy cache 开关；旧变量只作为兼容输入。"""
    return _env_flag(
        NUMPY_AUXILIARY_CACHE_ENV_VAR,
        os.getenv("USE_CACHED_NUMPY", "False"),
    )


@dataclass(frozen=True)
class InputBackendPolicy:
    """一次运行内共享的输入 backend 请求、解析结果和逻辑值设备。"""

    requested: str | None
    resolved: str
    logical_device: str
    use_gpu_mode: bool
    cache_numpy_auxiliary: bool
    mode: str | None = None


def resolve_input_backend_policy(
    *,
    requested=None,
    use_gpu_mode=None,
    use_cached_numpy=None,
    mode=None,
) -> InputBackendPolicy:
    """一次性解析 backend、模式默认值和逻辑值设备。"""
    # requested 表示用户覆盖，resolved 表示本次运行不可再变的最终选择。
    if requested is None:
        requested = os.environ.get(INPUT_BACKEND_ENV_VAR)
    normalized_requested = (requested or "").strip().lower() or None
    # 显式请求必须先校验，不能被 cache 或默认分支静默覆盖。
    if normalized_requested is not None and normalized_requested not in _VALID_INPUT_BACKENDS:
        raise ValueError(f"unsupported input generation backend: {requested!r}")
    if use_gpu_mode is None:
        use_gpu_mode = _env_flag("USE_GPU_MODE")
    else:
        use_gpu_mode = bool(use_gpu_mode)
    if use_cached_numpy is None:
        use_cached_numpy = _cache_numpy_auxiliary_enabled()
    else:
        use_cached_numpy = bool(use_cached_numpy)
    mode = mode or os.environ.get("PADDLEAPITEST_TEST_MODE") or None
    # 模式默认值只在没有显式请求时生效，保证命令行报告能解释覆盖关系。
    resolved = normalized_requested or _MODE_DEFAULT_BACKENDS.get(mode)
    if resolved is None:
        resolved = "torch" if use_gpu_mode else "numpy"
    # 性能模式本身要求 GPU-native 输入；显式 NumPy 仍是可见的 CPU 降级。
    effective_gpu_mode = use_gpu_mode or (
        mode in _GPU_NATIVE_MODES and normalized_requested != "numpy"
    )
    # logical_device 描述生成值的位置，不等同于 Paddle/Torch 算子的执行设备。
    logical_device = {
        "numpy": "cpu",
        "torch": "cuda:0" if effective_gpu_mode else "cpu",
        "paddle": "gpu:0" if effective_gpu_mode else "cpu",
    }[resolved]
    return InputBackendPolicy(
        requested=normalized_requested,
        resolved=resolved,
        logical_device=logical_device,
        use_gpu_mode=effective_gpu_mode,
        cache_numpy_auxiliary=use_cached_numpy,
        mode=mode,
    )


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


def resolve_input_backend_name(**kwargs) -> str:
    """兼容旧调用方，只返回已解析的 backend 名称。"""
    return resolve_input_backend_policy(**kwargs).resolved


def create_input_backend(
    input_random_state=INPUT_NUMPY_RANDOM_STATE,
    *,
    policy: InputBackendPolicy | None = None,
) -> InputBackend:
    policy = policy or resolve_input_backend_policy()
    normalized = policy.resolved
    device = policy.logical_device

    # 只有无 config-local RNG 的兼容调用可以复用实例，避免随机流在配置之间共享。
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


def prepare_input_backend(policy=None):
    """准备现有 backend 的进程级模块、device context 和物化通道。"""
    # policy 已在 engine 主进程解析；独立调用时才允许沿用环境变量解析结果。
    policy = policy or resolve_input_backend_policy()
    backend = create_input_backend(policy=policy)
    if getattr(backend, "_runtime_prepared", False):
        return backend

    # 常量探针不读取 RNG 或 cache，只把首次真实物化移到 worker ready 之前。
    probe = backend.asarray(numpy.zeros((1,), dtype="float32"), copy=False)
    if backend.name == "torch" and policy.logical_device.startswith("cuda"):
        backend._torch().cuda.synchronize(backend._device)
    elif backend.name == "paddle" and policy.logical_device.startswith(("gpu", "cuda")):
        backend._paddle().device.cuda.synchronize()
    del probe
    # 标记只写入兼容 backend 实例，不能扩展为跨 config 的随机状态缓存。
    backend._runtime_prepared = True
    return backend


def generate_output_grad(
    *,
    dtype,
    shape,
    backend_name,
    device,
    seed,
    config_fingerprint,
    stream_index=0,
    cache_enabled=False,
):
    """用现有 backend 和独立随机流生成 output grad。"""
    dtype = str(dtype)
    # stream 序号属于当前 config，保证多个输出梯度不会复用同一首样本。
    stream_kind = f"output_grad:{backend_name}:{int(stream_index)}"
    if cache_enabled:
        if backend_name != "numpy":
            raise ValueError("output-grad cache requires the NumPy backend")
        # 延迟导入避免 tensor_config -> backend 的既有依赖形成模块环。
        from .tensor_config import get_cached_numpy_array

        return get_cached_numpy_array(
            dtype,
            shape,
            generation_kind=stream_kind,
            scale=1.0,
            random_seed=seed,
            config_fingerprint=config_fingerprint,
        )

    # 非缓存路径继续使用 backend 私有随机源，避免污染框架默认 generator。
    random_state = InputConfigRandomState(seed, config_fingerprint, stream_kind=stream_kind)
    if backend_name == "numpy":
        backend = NumPyInputBackend(random_state)
    elif backend_name == "torch":
        backend = TorchInputBackend(random_state, device=device)
    elif backend_name == "paddle":
        backend = PaddleInputBackend(random_state, device=device)
    else:
        raise ValueError(f"unsupported output-grad backend: {backend_name!r}")

    if "int" in dtype:
        return backend.randint(-65535, 65535, shape=shape, dtype=dtype)
    if dtype in {"float8_e5m2", "float8_e4m3fn"}:
        base_dtype = "float16"
    elif dtype == "bfloat16":
        base_dtype = "float32"
    else:
        base_dtype = dtype
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = backend.random(shape, dtype=real_dtype) - 0.5
        imag = backend.random(shape, dtype=real_dtype) - 0.5
        return backend.cast(real + 1j * imag, dtype)
    value = backend.uniform(-0.5, 0.5, shape=shape, dtype=base_dtype)
    return backend.cast(value, dtype) if base_dtype != dtype else value
