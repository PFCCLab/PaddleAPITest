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

from .value_sample import NUMPY_RNG, generation_dtype


class GenerationBackend(Protocol):
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
class NumpyGenerationBackend:
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
        return value.astype(dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, shape=None, size=None, dtype=None):
        shape = shape if shape is not None else size
        value = self.rng.uniform(low=low, high=high, size=shape)
        return value.astype(dtype) if dtype is not None else value

    def randint(self, low, high=None, shape=None, size=None, dtype=None):
        shape = shape if shape is not None else size
        kwargs = {"size": shape}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self.rng.randint(low, high, **kwargs)

    def randn(self, *shape, size=None, dtype=None):
        if size is not None and not shape:
            shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
        value = self.rng.randn(*shape)
        return value.astype(dtype) if dtype is not None else value

    def choice(self, values, shape=None, size=None, replace=True, p=None):
        shape = shape if shape is not None else size
        return self.rng.choice(values, size=shape, replace=replace, p=p)

    def asarray(self, value, dtype=None, copy=True, order="K"):
        return numpy.array(value, dtype=dtype, copy=copy, order=order)

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
class TorchGenerationBackend(NumpyGenerationBackend):
    """Torch-backed generator that preserves NumPy-compatible rule values."""

    name = "torch"
    _generator: object = field(init=False, repr=False)

    def __post_init__(self):
        seed_material = (
            f"{getattr(self.rng, 'seed', 0)}:{getattr(self.rng, 'config_fingerprint', '')}"
        )
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16) % (2**63)
        torch = self._torch()
        self._generator = torch.Generator(device="cpu")
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
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype is None:
            return None
        return to_torch_dtype(storage_dtype)

    def _numpy_dtype(self, dtype):
        storage_dtype = self._storage_dtype(dtype)
        return None if storage_dtype is None else numpy.dtype(storage_dtype)

    def _torch_float_generation_dtype(self, dtype):
        torch = self._torch()
        storage_dtype = self._storage_dtype(dtype)
        if storage_dtype == "float64":
            return torch.float64
        return torch.float32

    def _to_numpy(self, value, dtype=None, scalar=False):
        torch = self._torch()
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                value = value.cpu()
            value = value.numpy()
        if dtype is not None:
            value = numpy.asarray(value).astype(self._numpy_dtype(dtype))
        if scalar and hasattr(value, "item"):
            return value.item()
        return value

    def random(self, shape=None, size=None, dtype=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        scalar = shape is None
        torch_shape = self._torch_shape(shape)
        value = torch.rand(torch_shape, dtype=torch.float32, generator=self._generator)
        return self._to_numpy(value, dtype=dtype, scalar=scalar)

    def uniform(self, low=0.0, high=1.0, shape=None, size=None, dtype=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        scalar = shape is None
        torch_shape = self._torch_shape(shape)
        value = torch.empty(torch_shape, dtype=self._torch_float_generation_dtype(dtype)).uniform_(
            float(low), float(high), generator=self._generator
        )
        return self._to_numpy(value, dtype=dtype, scalar=scalar)

    def randint(self, low, high=None, shape=None, size=None, dtype=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        scalar = shape is None
        torch_shape = self._torch_shape(shape)
        if high is None:
            low, high = 0, low
        value = torch.randint(
            int(low), int(high), torch_shape, dtype=torch.int64, generator=self._generator
        )
        return self._to_numpy(value, dtype=dtype, scalar=scalar)

    def randn(self, *shape, size=None, dtype=None):
        torch = self._torch()
        if size is not None and not shape:
            shape = tuple(size) if isinstance(size, (list, tuple)) else (size,)
        value = torch.randn(tuple(shape), dtype=torch.float32, generator=self._generator)
        return self._to_numpy(value, dtype=dtype, scalar=not shape)

    def choice(self, values, shape=None, size=None, replace=True, p=None):
        torch = self._torch()
        shape = self._shape(shape, size)
        scalar = shape is None
        torch_shape = (
            () if scalar else tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        )
        num_samples = 1 if scalar else int(numpy.prod(torch_shape))

        if isinstance(values, numbers.Integral):
            population = numpy.arange(int(values))
        else:
            population = numpy.asarray(values)

        if p is not None:
            weights = torch.as_tensor(numpy.asarray(p, dtype="float64"), dtype=torch.float64)
            indices = torch.multinomial(
                weights, num_samples, replacement=replace, generator=self._generator
            )
        elif replace:
            indices = torch.randint(
                0, len(population), (num_samples,), dtype=torch.int64, generator=self._generator
            )
        else:
            if num_samples > len(population):
                raise ValueError("Cannot take a larger sample than population when replace=False")
            indices = torch.randperm(len(population), dtype=torch.int64, generator=self._generator)[
                :num_samples
            ]

        result = population[indices.numpy()]
        if scalar:
            return result.item()
        return result.reshape(torch_shape)


INPUT_BACKEND_ENV_VAR = "PADDLEAPITEST_INPUT_BACKEND"
_DEFAULT_BACKENDS = {}
_WARNED_CACHED_TORCH_BACKEND = False


def _use_cached_numpy() -> bool:
    return os.getenv("USE_CACHED_NUMPY", "False").lower() in {"true", "1", "yes", "y"}


def create_generation_backend(rng=NUMPY_RNG) -> GenerationBackend:
    global _WARNED_CACHED_TORCH_BACKEND

    name = os.environ.get(INPUT_BACKEND_ENV_VAR, "numpy")
    normalized = (name or "numpy").strip().lower()
    if normalized not in {"numpy", "torch"}:
        raise ValueError(f"unsupported input generation backend: {name!r}")
    if _use_cached_numpy() and normalized == "torch":
        if not _WARNED_CACHED_TORCH_BACKEND:
            message = (
                "USE_CACHED_NUMPY=True requires the numpy input backend; "
                f"ignoring {INPUT_BACKEND_ENV_VAR}=torch."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            print(f"[WARNING] {message}", file=sys.stderr, flush=True)
            _WARNED_CACHED_TORCH_BACKEND = True
        normalized = "numpy"
    if rng is NUMPY_RNG and normalized in _DEFAULT_BACKENDS:
        return _DEFAULT_BACKENDS[normalized]
    if normalized == "numpy":
        backend = NumpyGenerationBackend(rng)
    else:
        backend = TorchGenerationBackend(rng)
    if rng is NUMPY_RNG:
        _DEFAULT_BACKENDS[normalized] = backend
    return backend
