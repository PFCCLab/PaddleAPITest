"""Backend abstractions for generated input values."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy
from tester.api_config.dtype_utils import to_torch_dtype

from .value_generators import (
    INPUT_NUMPY_RANDOM_STATE,
    derive_input_seed,
    resolve_input_dtype,
)

# 大 Tensor 才切换目标 storage dtype，避免改变小配置既有随机序列。
DIRECT_DTYPE_NUMEL_THRESHOLD = 1 << 20


def _normalize_shape(shape, *, scalar_empty):
    """统一 NumPy 与原生 backend 对标量 shape 的边界表示。"""
    if shape is None:
        return [] if scalar_empty else ()
    if isinstance(shape, numbers.Integral):
        return [int(shape)] if scalar_empty else (int(shape),)
    return list(shape) if scalar_empty else tuple(shape)


def _choice_shape(shape, *, scalar_empty):
    """返回采样目标的标量标志、规范 shape 和元素数量。"""
    scalar = shape is None
    normalized = _normalize_shape(shape, scalar_empty=scalar_empty)
    return scalar, normalized, 1 if scalar else int(numpy.prod(normalized))


def _large_shape(shape):
    """判断是否值得避免高精度临时 Tensor。"""
    _, _, numel = _choice_shape(shape, scalar_empty=False)
    return numel >= DIRECT_DTYPE_NUMEL_THRESHOLD


def _direct_float_dtype(dtype, shape):
    """返回大 Tensor 可直接生成的浮点 storage dtype。"""
    # 低精度直生只对超过阈值的随机输入生效，避免改变小配置的随机序列。
    if not _large_shape(shape):
        return None
    storage_dtype = resolve_input_dtype(dtype) if dtype is not None else None
    return storage_dtype if storage_dtype in {"float16", "bfloat16"} else None


def _direct_int32(dtype, shape, low, high):
    """仅在边界可表达时使用 int32 原生随机算子。"""
    # randint 边界必须能由 int32 表达，避免生成后 cast 造成索引回绕。
    if not _large_shape(shape) or resolve_input_dtype(dtype) != "int32":
        return False
    return -(1 << 31) <= int(low) < int(high) <= (1 << 31)


@runtime_checkable
class InputBackend(Protocol):
    """Value construction interface used by input-generation rules."""

    name: str

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

    def remainder(self, x, y): ...

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


def _resolve_storage_dtype(dtype, backend_name):
    """将逻辑 dtype 归一化为 backend 可稳定构造的 storage dtype。"""
    # 三个 backend 共享逻辑 dtype 协议，但各自负责最终的原生物化。
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
            f"{backend_name} backend does not support input dtype {dtype_name!r} "
            f"(storage dtype {storage_dtype!r})"
        ) from err
    return storage_dtype


@dataclass
class NumPyInputBackend:
    """NumPy implementation of the input-generation backend."""

    input_random_state: object = INPUT_NUMPY_RANDOM_STATE

    name = "numpy"

    def resolve_input_dtype(self, dtype: str) -> str:
        return resolve_input_dtype(dtype)

    def _storage_dtype(self, dtype):
        return _resolve_storage_dtype(dtype, self.name)

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
        _, numpy_shape, num_samples = _choice_shape(shape, scalar_empty=False)
        # 零采样不依赖概率或 replacement，必须返回空结果且不推进随机流。
        if num_samples == 0:
            if isinstance(values, numbers.Integral):
                population = self.arange(int(values))
            else:
                population = self.asarray(values)
            return self.zeros(numpy_shape, dtype=population.dtype)
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

    def remainder(self, x, y):
        # 统一取模语义，保证 GPU/CPU 生成器的循环 index 行为一致。
        return numpy.remainder(x, y)

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
class TorchInputBackend:
    """Torch implementation of the input-generation backend."""

    input_random_state: object = INPUT_NUMPY_RANDOM_STATE
    device: str = "cpu"
    prepared: object = field(default=None, repr=False)
    name = "torch"
    _generator: object = field(init=False, repr=False)

    def resolve_input_dtype(self, dtype: str) -> str:
        # native backend 不再从 NumPy 继承协议实现，避免隐式回落。
        return resolve_input_dtype(dtype)

    def _storage_dtype(self, dtype):
        return _resolve_storage_dtype(dtype, self.name)

    def __post_init__(self):
        prepared = self.prepared
        if prepared is None:
            self._torch_module = self._torch()
            self._device = self._torch_module.device(self.device)
        else:
            # preparation 只提供进程级不可变句柄；config generator 仍在下方独立创建。
            self._torch_module = prepared._torch_module
            self._device = prepared._device
        # Torch 私有 Generator 只消费本 backend 的 stream，不触碰默认 generator。
        stream_kind = getattr(self.input_random_state, "stream_kind", "")
        if not stream_kind.startswith("output_grad:"):
            stream_kind = "torch"
        seed = derive_input_seed(
            getattr(self.input_random_state, "seed", 0),
            getattr(self.input_random_state, "config_fingerprint", ""),
            stream_kind,
        )
        self._generator = self._torch_module.Generator(device=self._device)
        self._generator.manual_seed(seed)

    def reset_input_random_state(self, input_random_state):
        """重置当前 case 的私有 generator，保持跨 case 随机流独立。"""
        stream_kind = getattr(input_random_state, "stream_kind", "")
        if not stream_kind.startswith("output_grad:"):
            stream_kind = "torch"
        seed = derive_input_seed(
            getattr(input_random_state, "seed", 0),
            getattr(input_random_state, "config_fingerprint", ""),
            stream_kind,
        )
        self._generator.manual_seed(seed)

    def _torch(self):
        module = getattr(self, "_torch_module", None)
        if module is not None:
            return module
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
        direct_dtype = _direct_float_dtype(dtype, shape)
        value = torch.rand(
            torch_shape,
            dtype=self._torch_dtype(direct_dtype) if direct_dtype else torch.float32,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None and not direct_dtype else value

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        torch = self._torch()
        torch_shape = self._torch_shape(shape)
        direct_dtype = _direct_float_dtype(dtype, shape)
        value = torch.empty(
            torch_shape,
            dtype=(
                self._torch_dtype(direct_dtype)
                if direct_dtype
                else self._resolve_torch_float_dtype(dtype)
            ),
            device=self._device,
        ).uniform_(
            float(low),
            float(high),
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None and not direct_dtype else value

    def randint(self, low, high=None, shape=None, dtype=None):
        torch = self._torch()
        torch_shape = self._torch_shape(shape)
        if high is None:
            low, high = 0, low
        direct_int32 = _direct_int32(dtype, shape, low, high)
        value = torch.randint(
            int(low),
            int(high),
            torch_shape,
            dtype=torch.int32 if direct_int32 else torch.int64,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None and not direct_int32 else value

    def randn(self, *shape, dtype=None):
        torch = self._torch()
        direct_dtype = _direct_float_dtype(dtype, shape)
        value = torch.randn(
            tuple(shape),
            dtype=self._torch_dtype(direct_dtype) if direct_dtype else torch.float32,
            device=self._device,
            generator=self._generator,
        )
        return self.cast(value, dtype) if dtype is not None and not direct_dtype else value

    def choice(self, values, shape=None, replace=True, p=None):
        torch = self._torch()
        scalar, torch_shape, num_samples = _choice_shape(shape, scalar_empty=False)

        if isinstance(values, numbers.Integral):
            population = self.arange(int(values))
        else:
            population = self.asarray(values)

        # 零采样绕过 randint/randperm/multinomial，避免空 population 的原生报错。
        if num_samples == 0:
            return self.zeros(torch_shape, dtype=population.dtype)

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

    def remainder(self, x, y):
        # 使用设备端 remainder，避免将大批量 index 拉回 CPU 计算。
        torch = self._torch()
        return torch.remainder(self.asarray(x, copy=False), y)

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
class PaddleInputBackend:
    """Paddle implementation of the input-generation backend."""

    input_random_state: object = INPUT_NUMPY_RANDOM_STATE
    device: str = "cpu"
    prepared: object = field(default=None, repr=False)
    name = "paddle"

    def resolve_input_dtype(self, dtype: str) -> str:
        # native backend 不再从 NumPy 继承协议实现，避免隐式回落。
        return resolve_input_dtype(dtype)

    def _storage_dtype(self, dtype):
        return _resolve_storage_dtype(dtype, self.name)

    def __post_init__(self):
        prepared = self.prepared
        if prepared is None:
            self._paddle_module = self._paddle()
            self._place = self._paddle_module.CPUPlace()
            self._generator = self._paddle_module.framework.core.default_cpu_generator()
            if self.device.startswith(("gpu", "cuda")):
                device_id = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
                self._place = self._paddle_module.CUDAPlace(device_id)
                self._generator = self._paddle_module.framework.core.default_cuda_generator(
                    device_id
                )
        else:
            # place 与默认 generator 属于进程/设备资源，config 只拥有私有状态快照。
            self._paddle_module = prepared._paddle_module
            self._place = prepared._place
            self._generator = prepared._generator
        # forward 与 output-grad 用不同 stream identity，不能共享首个随机样本。
        stream_kind = getattr(self.input_random_state, "stream_kind", "")
        if not stream_kind.startswith("output_grad:"):
            stream_kind = "paddle"
        seed = derive_input_seed(
            getattr(self.input_random_state, "seed", 0),
            getattr(self.input_random_state, "config_fingerprint", ""),
            stream_kind,
        )
        # Paddle 只有设备级默认 generator；初始化私有状态后立即恢复进程原状态。
        process_state = self._generator.get_state()
        try:
            self._generator.manual_seed(seed)
            self._random_state = self._generator.get_state()
        finally:
            self._generator.set_state(process_state)

    def reset_input_random_state(self, input_random_state):
        """重置当前 case 的设备 generator 快照，不污染进程级随机状态。"""
        stream_kind = getattr(input_random_state, "stream_kind", "")
        if not stream_kind.startswith("output_grad:"):
            stream_kind = "paddle"
        seed = derive_input_seed(
            getattr(input_random_state, "seed", 0),
            getattr(input_random_state, "config_fingerprint", ""),
            stream_kind,
        )
        process_state = self._generator.get_state()
        try:
            self._generator.manual_seed(seed)
            self._random_state = self._generator.get_state()
        finally:
            self._generator.set_state(process_state)

    def _paddle(self):
        module = getattr(self, "_paddle_module", None)
        if module is not None:
            return module
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

    def _run_random(self, function):
        """临时挂载当前 config 的 Paddle RNG 状态，并隔离被测算子随机流。"""
        process_state = self._generator.get_state()
        try:
            self._generator.set_state(self._random_state)
            value = function()
            # 只有原生随机调用成功才推进当前 backend 的私有 stream。
            self._random_state = self._generator.get_state()
            return value
        finally:
            self._generator.set_state(process_state)

    def random(self, shape=None, dtype=None):
        paddle = self._paddle()
        direct_dtype = _direct_float_dtype(dtype, shape)
        # 默认随机原语先生成稳定 float32 storage，再按逻辑 dtype 转换。
        value = self._run_random(
            lambda: paddle.rand(
                self._paddle_shape(shape),
                dtype=direct_dtype or "float32",
                device=self.device,
            )
        )
        return self.cast(value, dtype) if dtype is not None and not direct_dtype else value

    def uniform(self, low=0.0, high=1.0, shape=None, dtype=None):
        paddle = self._paddle()
        direct_dtype = _direct_float_dtype(dtype, shape)
        # seed=0 表示消费刚挂载的设备 generator，而不是创建算子私有固定 seed。
        value = self._run_random(
            lambda: paddle.uniform(
                self._paddle_shape(shape),
                dtype=direct_dtype or self._resolve_paddle_float_dtype(dtype),
                min=float(low),
                max=float(high),
                seed=0,
                device=self.device,
            )
        )
        return self.cast(value, dtype) if dtype is not None and not direct_dtype else value

    def randint(self, low, high=None, shape=None, dtype=None):
        if high is None:
            low, high = 0, low
        paddle = self._paddle()
        direct_int32 = _direct_int32(dtype, shape, low, high)
        value = self._run_random(
            lambda: paddle.randint(
                int(low),
                int(high),
                self._paddle_shape(shape),
                dtype="int32" if direct_int32 else "int64",
                device=self.device,
            )
        )
        return self.cast(value, dtype) if dtype is not None and not direct_int32 else value

    def randn(self, *shape, dtype=None):
        paddle = self._paddle()
        value = self._run_random(
            lambda: paddle.randn(
                self._paddle_shape(shape),
                dtype="float32",
                device=self.device,
            )
        )
        return self.cast(value, dtype) if dtype is not None else value

    def choice(self, values, shape=None, replace=True, p=None):
        paddle = self._paddle()
        scalar, paddle_shape, num_samples = _choice_shape(shape, scalar_empty=True)
        if isinstance(values, numbers.Integral):
            population = self.arange(int(values))
        else:
            population = self.asarray(values, copy=False)

        # Paddle 的 randperm(0) 会在 GPU 抛 CUDA invalid argument，空结果直接物化。
        if num_samples == 0:
            return self.zeros(paddle_shape, dtype=population.dtype)

        if p is not None:
            # 带权采样由 Paddle multinomial 消费同一私有 stream，不回落到主存数组。
            weights = self.asarray(p, dtype="float64", copy=False)
            indices = self._run_random(
                lambda: paddle.multinomial(weights, num_samples, replacement=replace)
            )
        elif replace:
            # 有放回采样只生成 Paddle 索引，population 始终保持 backend-native。
            indices = self._run_random(
                lambda: paddle.randint(
                    0,
                    len(population),
                    [num_samples],
                    dtype="int64",
                    device=self.device,
                )
            )
        else:
            if num_samples > len(population):
                raise ValueError("Cannot take a larger sample than population when replace=False")
            # 无放回采样通过原生 randperm 实现，保持与其他随机原语相同的状态推进。
            indices = self._run_random(
                lambda: paddle.randperm(
                    len(population),
                    dtype="int64",
                    device=self.device,
                )[:num_samples]
            )

        result = paddle.gather(population, indices)
        return result.item() if scalar else self.reshape(result, paddle_shape)

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

    def remainder(self, x, y):
        # Paddle backend 保持与 NumPy/Torch 相同的非负 index 取模结果。
        return self._paddle().remainder(self.asarray(x, copy=False), y)

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
