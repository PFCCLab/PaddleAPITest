"""Backend abstractions for generated input values."""

from __future__ import annotations

import hashlib
import numbers
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy
from tester.dtype_utils import to_torch_dtype

from .value_gen import NUMPY_RNG, generation_dtype


class InputBackend(Protocol):
    """Value construction interface used by input-generation rules."""

    name: str

    def commit(self) -> None: ...

    def generation_dtype(self, dtype: str) -> str: ...

    def random(self, shape=None, size=None, dtype=None): ...

    def uniform(self, low=0.0, high=1.0, shape=None, size=None, dtype=None): ...

    def randint(self, low, high=None, shape=None, size=None, dtype=None): ...

    def randn(self, *shape, size=None, dtype=None): ...

    def choice(self, values, shape=None, size=None, replace=True, p=None): ...

    def asarray(self, value, dtype=None, copy=True, order="K"): ...

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
class NumpyInputBackend:
    """NumPy implementation of the input-generation backend."""

    rng: object = NUMPY_RNG

    name = "numpy"

    def commit(self) -> None:
        commit = getattr(self.rng, "commit", None)
        if commit is not None:
            commit()

    def generation_dtype(self, dtype: str) -> str:
        return generation_dtype(dtype)

    def random(self, shape=None, size=None, dtype=None):
        shape = shape if shape is not None else size
        value = self.rng.random(shape)
        return numpy.asarray(value).astype(dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, size=None, dtype=None):
        shape = shape if shape is not None else size
        value = self.rng.uniform(low=low, high=high, size=shape)
        return numpy.asarray(value).astype(dtype) if dtype is not None else value

    def randint(self, low, high=None, shape=None, size=None, dtype=None):
        shape = shape if shape is not None else size
        value = self.rng.randint(low, high, size=shape)
        return numpy.asarray(value).astype(dtype) if dtype is not None else value

    def randn(self, *shape, size=None, dtype=None):
        if size is not None and not shape:
            shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
        value = self.rng.randn(*shape)
        return numpy.asarray(value).astype(dtype) if dtype is not None else value

    def choice(self, values, shape=None, size=None, replace=True, p=None):
        shape = shape if shape is not None else size
        return self.rng.choice(values, size=shape, replace=replace, p=p)

    def asarray(self, value, dtype=None, copy=True, order="K"):
        return numpy.array(value, dtype=dtype, copy=copy, order=order)

    def cast(self, value, dtype):
        return numpy.asarray(value).astype(dtype)

    def reshape(self, value, shape):
        return numpy.reshape(value, shape)

    def flatten(self, value):
        return numpy.reshape(value, -1)

    def view_dtype(self, value, dtype):
        return numpy.asarray(value).view(dtype)

    def arange(self, *args, dtype=None):
        return numpy.arange(*args, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return numpy.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        return numpy.full(shape, fill_value, dtype=dtype)

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
        return numpy.eye(size, dtype=dtype)

    def ascontiguousarray(self, value):
        return numpy.ascontiguousarray(value)

    def finfo(self, dtype):
        return numpy.finfo(dtype)


@dataclass
class TorchInputBackend(NumpyInputBackend):
    """Torch implementation of the input-generation backend."""

    device: str = "cpu"
    name = "torch"
    _generator: object = field(init=False, repr=False)

    def __post_init__(self):
        seed_material = (
            f"{getattr(self.rng, 'seed', 0)}:{getattr(self.rng, 'config_fingerprint', '')}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16) % (2**63)
        torch = self._torch()
        self._device = torch.device(self.device)
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(seed)

    def _torch(self):
        import torch

        return torch

    def _shape(self, shape=None, size=None):
        return shape if shape is not None else size

    def _torch_shape(self, shape):
        if shape is None:
            return ()
        if isinstance(shape, numbers.Integral):
            return (int(shape),)
        return tuple(shape)

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
        return generation_dtype(dtype_name)

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

    def _torch_float_generation_dtype(self, dtype):
        torch = self._torch()
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype == "float64":
            return torch.float64
        return torch.float32

    def random(self, shape=None, size=None, dtype=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        torch_shape = self._torch_shape(shape)
        value = torch.rand(
            torch_shape,
            dtype=torch.float32,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, size=None, dtype=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        torch_shape = self._torch_shape(shape)
        value = torch.empty(
            torch_shape,
            dtype=self._torch_float_generation_dtype(dtype),
            device=self._device,
        ).uniform_(
            float(low),
            float(high),
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randint(self, low, high=None, shape=None, size=None, dtype=None):
        torch = self._torch()
        shape = self._shape(shape, size)
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

    def randn(self, *shape, size=None, dtype=None):
        torch = self._torch()
        if size is not None and not shape:
            shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
        value = torch.randn(
            tuple(shape),
            dtype=torch.float32,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def choice(self, values, shape=None, size=None, replace=True, p=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        scalar = shape is None
        torch_shape = (
            () if scalar else tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        )
        num_samples = 1 if scalar else int(numpy.prod(torch_shape))

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

    def asarray(self, value, dtype=None, copy=True, order="K"):
        torch = self._torch()
        if order not in {"K", "C", None}:
            raise ValueError(f"unsupported torch input array order: {order!r}")
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
class PaddleInputBackend(NumpyInputBackend):
    """Paddle implementation of the input-generation backend."""

    device: str = "cpu"
    name = "paddle"

    def __post_init__(self):
        seed_material = (
            f"{getattr(self.rng, 'seed', 0)}:{getattr(self.rng, 'config_fingerprint', '')}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:8], 16) % (2**31)
        paddle = self._paddle()
        paddle.seed(seed)
        self._place = paddle.CPUPlace()
        if self.device.startswith(("gpu", "cuda")):
            device_id = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            self._place = paddle.CUDAPlace(device_id)

    def _paddle(self):
        import paddle

        return paddle

    def _shape(self, shape=None, size=None):
        return shape if shape is not None else size

    def _paddle_shape(self, shape):
        if shape is None:
            return []
        if isinstance(shape, numbers.Integral):
            return [int(shape)]
        return list(shape)

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
        return generation_dtype(dtype_name)

    def _paddle_dtype(self, dtype):
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype is None:
            return None
        return getattr(self._paddle(), storage_dtype)

    def _paddle_float_generation_dtype(self, dtype):
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype == "float64":
            return "float64"
        return "float32"

    def random(self, shape=None, size=None, dtype=None):
        paddle = self._paddle()
        shape = self._shape(shape, size)
        value = paddle.rand(
            self._paddle_shape(shape),
            dtype="float32",
            device=self.device,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, size=None, dtype=None):
        paddle = self._paddle()
        shape = self._shape(shape, size)
        value = paddle.uniform(
            self._paddle_shape(shape),
            dtype=self._paddle_float_generation_dtype(dtype),
            min=float(low),
            max=float(high),
            device=self.device,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randint(self, low, high=None, shape=None, size=None, dtype=None):
        paddle = self._paddle()
        shape = self._shape(shape, size)
        if high is None:
            low, high = 0, low
        value = paddle.randint(
            int(low),
            int(high),
            self._paddle_shape(shape),
            dtype="int64",
            device=self.device,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def randn(self, *shape, size=None, dtype=None):
        paddle = self._paddle()
        if size is not None and not shape:
            shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
        value = paddle.randn(
            self._paddle_shape(shape),
            dtype="float32",
            device=self.device,
        )
        return self.cast(value, dtype) if dtype is not None else value

    def choice(self, values, shape=None, size=None, replace=True, p=None):
        paddle = self._paddle()
        shape = self._shape(shape, size)
        scalar = shape is None
        paddle_shape = (
            [] if scalar else list(shape) if isinstance(shape, (list, tuple)) else [shape]
        )
        num_samples = 1 if scalar else int(numpy.prod(paddle_shape))

        if isinstance(values, numbers.Integral):
            population = paddle.arange(int(values), dtype="int64", device=self.device)
        else:
            population = self.asarray(values, copy=False).flatten()

        if p is not None:
            weights = self.asarray(p, dtype="float64", copy=False).flatten()
            indices = paddle.multinomial(weights, num_samples, replacement=replace)
        elif replace:
            indices = paddle.randint(
                0,
                int(population.shape[0]),
                [num_samples],
                dtype="int64",
                device=self.device,
            )
        else:
            if num_samples > int(population.shape[0]):
                raise ValueError("Cannot take a larger sample than population when replace=False")
            indices = paddle.randperm(
                int(population.shape[0]),
                dtype="int64",
                device=self.device,
            )[:num_samples]

        result = paddle.gather(population, indices)
        if scalar:
            return result.item()
        return self.reshape(result, paddle_shape)

    def asarray(self, value, dtype=None, copy=True, order="K"):
        paddle = self._paddle()
        if order not in {"K", "C", None}:
            raise ValueError(f"unsupported paddle input array order: {order!r}")
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
# Only used when no case-local RNG is available, such as legacy TensorConfig helpers.
_DEFAULT_BACKENDS = {}
_WARNED_CACHED_TORCH_BACKEND = False


def _use_cached_numpy() -> bool:
    return os.getenv("USE_CACHED_NUMPY", "False").lower() in {"true", "1", "yes", "y"}


def _use_gpu_mode() -> bool:
    return os.getenv("USE_GPU_MODE", "False").lower() in {"true", "1", "yes", "y"}


def create_input_backend(rng=NUMPY_RNG) -> InputBackend:
    global _WARNED_CACHED_TORCH_BACKEND

    requested = os.environ.get(INPUT_BACKEND_ENV_VAR)
    normalized_requested = (requested or "numpy").strip().lower()

    if _use_cached_numpy():
        if normalized_requested in {"torch", "paddle"} and not _WARNED_CACHED_TORCH_BACKEND:
            message = (
                "USE_CACHED_NUMPY=True requires the numpy input backend; "
                f"ignoring {INPUT_BACKEND_ENV_VAR}={normalized_requested}."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            print(f"[WARNING] {message}", file=sys.stderr, flush=True)
            _WARNED_CACHED_TORCH_BACKEND = True
        normalized = "numpy"
    elif _use_gpu_mode() and requested is None:
        normalized = "torch"
    else:
        normalized = normalized_requested

    if normalized not in {"numpy", "torch", "paddle"}:
        raise ValueError(f"unsupported input generation backend: {requested!r}")

    if normalized == "torch":
        device = "cuda:0" if _use_gpu_mode() else "cpu"
    elif normalized == "paddle":
        device = "gpu:0" if _use_gpu_mode() else "cpu"
    else:
        device = "cpu"
    cache_key = (normalized, device)
    if rng is NUMPY_RNG and cache_key in _DEFAULT_BACKENDS:
        return _DEFAULT_BACKENDS[cache_key]
    if normalized == "numpy":
        backend = NumpyInputBackend(rng)
    elif normalized == "torch":
        backend = TorchInputBackend(rng, device=device)
    else:
        backend = PaddleInputBackend(rng, device=device)
    if rng is NUMPY_RNG:
        _DEFAULT_BACKENDS[cache_key] = backend
    return backend
