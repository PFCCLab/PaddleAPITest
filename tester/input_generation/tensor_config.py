from __future__ import annotations

import collections
import copy
import math
import os
import random
import re

import numpy
import paddle
import yaml
from tester.dtype_utils import to_torch_dtype

from .input_values import clear_input_value, input_value, write_input_value

# 优化器的零填充位置属于物化层专用数据。
OPTIMIZER_APIS = {
    "paddle._C_ops.adamw_": {3: "zeros", 4: "zeros", 5: "zeros"},
    "paddle._C_ops.adam_": {3: "zeros", 4: "zeros", 5: "zeros"},
    "paddle._C_ops.merged_adam_": {3: "zeros", 4: "zeros", 5: "zeros"},
}


class _LazyTorch:
    def __getattr__(self, name):
        import torch

        globals()["torch"] = torch
        return getattr(torch, name)


torch = _LazyTorch()

USE_CACHED_NUMPY = os.getenv("USE_CACHED_NUMPY", "False").lower() == "true"
TEST_NON_CONTIGUOUS = os.getenv("TEST_NON_CONTIGUOUS", "0").lower() in ("true", "1")
USE_GPU_MODE = os.getenv("USE_GPU_MODE", "False").lower() == "true"
cached_numpy = {}
AUTOGRAD_DTYPES = frozenset(
    ["float32", "float64", "float16", "complex64", "complex128", "bfloat16"]
)
FLOAT8_DTYPES = frozenset(["float8_e5m2", "float8_e4m3fn"])
CAST_THROUGH_INTERMEDIATE_DTYPES = frozenset(["bfloat16"]) | FLOAT8_DTYPES


def _load_forward_only_apis():
    config_path = os.path.join(os.path.dirname(__file__), "..", "base_config.yaml")
    with open(config_path, encoding="utf-8") as f:
        return frozenset(yaml.safe_load(f).get("forward_only_apis", []))


FORWARD_ONLY_APIS = _load_forward_only_apis()
optimizer_apis = OPTIMIZER_APIS


def is_gpu_mode():
    return os.getenv("USE_GPU_MODE", str(USE_GPU_MODE)).lower() in ("true", "1", "yes", "y")


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _numel(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _normalize_cache_dtype(dtype):
    if dtype in ["float8_e5m2", "float8_e4m3fn"]:
        return "float16"
    if dtype == "bfloat16":
        return "float32"
    return str(dtype)


def get_cached_numpy_array(
    dtype,
    shape,
    generation_kind="input",
    scale=1.2,
    int_low=-65535,
    int_high=65535,
):
    dtype = _normalize_cache_dtype(dtype)
    shape = _shape_tuple(shape)
    key = (dtype, shape, generation_kind, float(scale), int(int_low), int(int_high))
    if key in cached_numpy:
        return cached_numpy[key]

    if "int" in dtype:
        tensor = numpy.random.randint(int_low, int_high, size=shape, dtype="int64").astype(dtype)
    elif dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real_part = ((numpy.random.random(shape) - 0.5) * scale).astype(real_dtype)
        imag_part = ((numpy.random.random(shape) - 0.5) * scale).astype(real_dtype)
        tensor = (real_part + 1j * imag_part).astype(dtype)
    else:
        tensor = ((numpy.random.random(shape) - 0.5) * scale).astype(dtype)
    cached_numpy[key] = tensor
    return tensor


not_zero_apis = frozenset(
    [
        "paddle.Tensor.__div__",
        "paddle.Tensor.__floordiv__",
        "paddle.Tensor.__mod__",
        "paddle.Tensor.__rdiv__",
        "paddle.Tensor.__rfloordiv__",
        "paddle.Tensor.__rmod__",
        "paddle.Tensor.__rtruediv__",
        "paddle.Tensor.__truediv__",
        "paddle.Tensor.divide",
        "paddle.Tensor.floor_divide",
        "paddle.Tensor.floor_mod",
        "paddle.Tensor.mod",
        "paddle.divide",
        "paddle.floor_divide",
        "paddle.floor_mod",
        "paddle.mod",
        "paddle.nn.functional.kl_div",
        "paddle.sparse.divide",
    ]
)


def generate_unique_array(num_items, float_dtype):
    def get_integer_dtype(float_dtype):
        float_dtype = numpy.dtype(float_dtype)
        if float_dtype == numpy.float16:
            return numpy.uint16, 16
        elif float_dtype == numpy.float32:
            return numpy.uint32, 32
        elif float_dtype == numpy.float64:
            return numpy.uint64, 64
        else:
            raise ValueError(f"Unsupported float dtype: {float_dtype}")

    integer_dtype, bits = get_integer_dtype(float_dtype)
    max_int = (1 << bits) - 1
    current_start_value = 1
    return_list = []
    attempt_count = 0
    while len(return_list) < num_items and attempt_count < 3:
        nums_to_generate = int(num_items * 1.5)
        if current_start_value >= max_int:
            raise ValueError(
                f"Cannot generate {num_items} unique items of type {float_dtype} within the range."
            )
        end_value = min(current_start_value + nums_to_generate, max_int)
        random_arr = numpy.arange(current_start_value, end_value, dtype=integer_dtype)
        float_arr = random_arr.view(float_dtype)
        if return_list is None:
            return_list = float_arr[numpy.isfinite(float_arr)]
        else:
            return_list = numpy.unique(
                numpy.concatenate([return_list, float_arr[numpy.isfinite(float_arr)]])
            )
        current_start_value = end_value
        attempt_count += 1
    if len(return_list) < num_items:
        raise ValueError(f"Could not generate {num_items} unique items of type {float_dtype}")
    return return_list[:num_items]


class TensorConfig:
    """一次参数位置上的可变 Tensor 配置。

    这个类同时承担三件事：保存参数元信息、缓存不同框架的物化结果，以及维护
    逻辑值与框架张量的一致性。
    """

    def __init__(self, shape, dtype, place=None, is_contiguous=True, strides=None):
        self.shape = shape
        self.dtype = dtype
        self.place = place
        self.is_contiguous = is_contiguous
        self.strides = strides
        self.numpy_tensor = None
        self.paddle_tensor = None
        self.torch_tensor = None
        self.cpu_tensor = None
        self.shuffle_dims = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.shape = copy.deepcopy(self.shape)
        result.dtype = copy.deepcopy(self.dtype)
        result.place = copy.deepcopy(self.place)
        result.is_contiguous = self.is_contiguous
        result.strides = copy.deepcopy(self.strides)
        result.numpy_tensor = None
        result.paddle_tensor = None
        result.torch_tensor = None
        result.cpu_tensor = None
        result.shuffle_dims = None
        return result

    def __str__(self):
        if self.place is not None:
            return f'Tensor({self.shape},"{self.dtype}",place={self.place})'
        return f'Tensor({self.shape},"{self.dtype}")'

    def __repr__(self):
        return self.__str__()

    def to_torch_dtype(self, dtype):
        return to_torch_dtype(dtype)

    def numel(self):
        return _numel(self.shape)

    def get_cached_numpy(self, dtype, shape, generation_kind="input", scale=1.2):
        return get_cached_numpy_array(dtype, shape, generation_kind=generation_kind, scale=scale)

    def _use_gpu(self, api_config=None, dtype=None):
        if not is_gpu_mode():
            return False
        if self.place is not None and "cpu" in str(self.place).lower():
            return False
        return "gpu" in paddle.device.get_device()

    def _supports_autograd(self, dtype=None):
        dtype = dtype or self.dtype
        return dtype in AUTOGRAD_DTYPES

    def _requires_autograd(self, api_config, dtype=None):
        if not self._supports_autograd(dtype):
            return False
        api_name = getattr(api_config, "api_name", "")
        api = api_name[api_name.rindex(".") + 1 :] if "." in api_name else api_name
        if api in FORWARD_ONLY_APIS:
            return False
        return getattr(api_config, "test_backward", True)

    def _set_paddle_autograd(self, tensor, api_config, dtype=None):
        tensor.stop_gradient = not self._requires_autograd(api_config, dtype)
        return tensor

    def _set_torch_autograd(self, tensor, api_config, dtype=None):
        if self._requires_autograd(api_config, dtype):
            tensor = tensor.detach().requires_grad_(True)
        return tensor

    def _gpu_paddle_dense(self, dtype=None):
        dtype = dtype or self.dtype
        shape = tuple(self.shape)
        if dtype == "bool":
            return paddle.randint(0, 2, shape, dtype="int32").cast("bool")
        if "int" in dtype or dtype == "uint8":
            return paddle.randint(-65535, 65535, shape, dtype="int64").cast(dtype)
        if dtype.startswith("complex"):
            real_dtype = "float32" if dtype == "complex64" else "float64"
            real_part = (paddle.rand(shape, dtype=real_dtype) - 0.5) * 1.2
            imag_part = (paddle.rand(shape, dtype=real_dtype) - 0.5) * 1.2
            return (real_part + 1j * imag_part).cast(dtype)
        base_dtype = "float16" if dtype in FLOAT8_DTYPES else "float32"
        return ((paddle.rand(shape, dtype=base_dtype) - 0.5) * 1.2).cast(dtype)

    def _gpu_torch_pat_dense(self, dtype, pattern_dtype=None):
        pattern_dtype = pattern_dtype or torch.float32
        torch_dtype = self.to_torch_dtype(dtype)
        shape = tuple(self.shape)
        device = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.empty(shape, device=device, dtype=torch_dtype)
        flat_tensor = tensor.reshape(-1)
        numel = self.numel()
        if numel == 0:
            return tensor

        chunk_elems = min(numel, 8 * 1024 * 1024)
        pattern_elems = min(chunk_elems, 1024 * 1024)
        pattern = torch.linspace(
            -0.6,
            0.6,
            steps=pattern_elems,
            device=device,
            dtype=pattern_dtype,
        ).to(dtype=torch_dtype)
        for start in range(0, numel, chunk_elems):
            end = min(start + chunk_elems, numel)
            offset = start
            while offset < end:
                next_offset = min(offset + pattern_elems, end)
                flat_tensor[offset:next_offset] = pattern[: next_offset - offset]
                offset = next_offset
        return tensor

    def _gpu_torch_dense(self, dtype=None):
        dtype = dtype or self.dtype
        torch_dtype = self.to_torch_dtype(dtype)
        shape = tuple(self.shape)
        device = torch.device("cuda", torch.cuda.current_device())
        if dtype == "bool":
            return torch.randint(0, 2, shape, device=device, dtype=torch.int8).to(torch.bool)
        if "int" in dtype or dtype == "uint8":
            return torch.randint(-65535, 65535, shape, device=device, dtype=torch.int64).to(
                dtype=torch_dtype
            )
        if dtype.startswith("complex"):
            real_dtype = torch.float32 if dtype == "complex64" else torch.float64
            real_part = (torch.rand(shape, device=device, dtype=real_dtype) - 0.5) * 1.2
            imag_part = (torch.rand(shape, device=device, dtype=real_dtype) - 0.5) * 1.2
            return (real_part + 1j * imag_part).to(dtype=torch_dtype)
        if dtype in FLOAT8_DTYPES:
            return self._gpu_torch_pat_dense(dtype, pattern_dtype=torch.float16)
        if dtype in ("float16", "bfloat16"):
            return self._gpu_torch_pat_dense(dtype, pattern_dtype=torch.float32)
        return ((torch.rand(shape, device=device, dtype=torch.float32) - 0.5) * 1.2).to(
            dtype=torch_dtype
        )

    def _gpu_strided_paddle(self, values, api_config, dtype=None):
        dtype = dtype or self.dtype
        intermediate_dtype = "float16" if dtype in FLOAT8_DTYPES else dtype
        flat_tensor = paddle.zeros(
            [self._strided_storage_size()],
            dtype=intermediate_dtype,
            device=self.place,
        )
        tensor = paddle.as_strided(flat_tensor, self.shape, self.strides)
        if self.numel() > 0:
            tensor[...] = values.cast(intermediate_dtype)
        if dtype in FLOAT8_DTYPES:
            flat_tensor = paddle.cast(flat_tensor, dtype=dtype)
            tensor = paddle.as_strided(flat_tensor, self.shape, self.strides)
        return self._set_paddle_autograd(tensor, api_config, dtype)

    def _gpu_strided_torch(self, values, api_config, dtype=None):
        dtype = dtype or self.dtype
        device = torch.device("cuda", torch.cuda.current_device())
        intermediate_dtype = torch.float16 if dtype in FLOAT8_DTYPES else self.to_torch_dtype(dtype)
        flat_tensor = torch.zeros(
            self._strided_storage_size(),
            dtype=intermediate_dtype,
            device=device,
        )
        tensor = torch.as_strided(flat_tensor, tuple(self.shape), tuple(self.strides))
        if self.numel() > 0:
            tensor.copy_(values.to(dtype=intermediate_dtype))
        if dtype in FLOAT8_DTYPES:
            flat_tensor = flat_tensor.to(dtype=self.to_torch_dtype(dtype))
            tensor = torch.as_strided(flat_tensor, tuple(self.shape), tuple(self.strides))
        return self._set_torch_autograd(tensor, api_config, dtype)

    def _make_gpu_paddle_tensor(self, api_config, dtype=None):
        dtype = dtype or self.dtype
        if not self.is_contiguous and self.strides is not None:
            source_dtype = "float16" if dtype in FLOAT8_DTYPES else dtype
            values = self._gpu_paddle_dense(source_dtype)
            return self._gpu_strided_paddle(values, api_config, dtype)
        return self._set_paddle_autograd(self._gpu_paddle_dense(dtype), api_config, dtype)

    def _make_gpu_torch_tensor(self, api_config, dtype=None):
        dtype = dtype or self.dtype
        if not self.is_contiguous and self.strides is not None:
            source_dtype = "float16" if dtype in FLOAT8_DTYPES else dtype
            values = self._gpu_torch_dense(source_dtype)
            return self._gpu_strided_torch(values, api_config, dtype)
        return self._set_torch_autograd(self._gpu_torch_dense(dtype), api_config, dtype)

    def _make_gpu_tensor_pair(self, api_config, dtype=None):
        dtype = dtype or self.dtype
        source_dtype = "float16" if dtype in FLOAT8_DTYPES else dtype
        torch_source = self._gpu_torch_dense(source_dtype)
        paddle_source = paddle.utils.dlpack.from_dlpack(
            torch.utils.dlpack.to_dlpack(torch_source.detach())
        )
        if not self.is_contiguous and self.strides is not None:
            try:
                paddle_tensor = self._gpu_strided_paddle(paddle_source, api_config, dtype)
                torch_tensor = self._gpu_strided_torch(torch_source, api_config, dtype)
                return paddle_tensor, torch_tensor
            finally:
                del torch_source, paddle_source
        if dtype in FLOAT8_DTYPES:
            paddle_tensor = paddle_source.cast(dtype)
            torch_tensor = torch_source.to(dtype=self.to_torch_dtype(dtype))
            del torch_source, paddle_source
        else:
            paddle_tensor = paddle_source
            torch_tensor = torch_source
            del torch_source
        paddle_tensor = self._set_paddle_autograd(paddle_tensor, api_config, dtype)
        torch_tensor = self._set_torch_autograd(torch_tensor, api_config, dtype)
        return paddle_tensor, torch_tensor

    def get_gpu_paddle_tensor(self, api_config, dtype=None):
        dtype = dtype or self.dtype
        if self.paddle_tensor is None:
            if getattr(api_config, "use_torch", True):
                self.paddle_tensor, self.torch_tensor = self._make_gpu_tensor_pair(
                    api_config, dtype
                )
            else:
                self.paddle_tensor = self._make_gpu_paddle_tensor(api_config, dtype)
        return self.paddle_tensor

    def get_gpu_torch_tensor(self, api_config, dtype=None):
        dtype = dtype or self.dtype
        if self.torch_tensor is None:
            if self.paddle_tensor is None:
                self.paddle_tensor, self.torch_tensor = self._make_gpu_tensor_pair(
                    api_config, dtype
                )
            else:
                torch_tensor = torch.utils.dlpack.from_dlpack(
                    paddle.utils.dlpack.to_dlpack(self.paddle_tensor.detach())
                )
                self.torch_tensor = self._set_torch_autograd(
                    torch_tensor.clone().detach(), api_config, dtype
                )
        return self.torch_tensor

    def generate_random_axes(self, api_config):
        x_shape = self.get_arg(api_config, 0, "x").shape
        max_dim = max(len(x_shape), 1)  # 标量时至少按 1 维处理。

        if len(self.shape) == 0:
            dim = numpy.random.randint(0, max_dim)
            if numpy.random.rand() > 0.5:
                dim -= max_dim
            return numpy.array(dim, dtype=self.dtype)

        if len(self.shape) == 1:
            dims = numpy.random.choice(max_dim, size=self.shape[0], replace=False)
            mask = numpy.random.rand(self.shape[0]) > 0.5
            dims = numpy.where(mask, dims - max_dim, dims)
            return numpy.array(dims, dtype=self.dtype)

        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {api_config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
        )

    def generate_random_index(self, api_config, allow_none=False):
        axis = self.get_arg(api_config, 2, "axis")
        if axis is None and not allow_none:
            raise ValueError("Axis is None")

        x_shape = self.get_arg(api_config, 0, "x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
        if len(self.shape) >= 1:
            return numpy.random.randint(0, x_shape[axis], size=self.shape, dtype=self.dtype)

        raise ValueError(
            f"Invalid shape for 'index' Tensor in {api_config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
        )

    def random_axis(self, api_config, arg_pos, kwargs_name):
        cfg = self.get_arg(api_config, arg_pos, kwargs_name)
        if isinstance(cfg, TensorConfig):
            max_idx = len(cfg.shape)
            return self.random_numpy([], data_type=self.dtype, min=0, max=max_idx)
        else:
            raise ValueError(f"Invalid axis config={cfg} in {api_config.api_name}")

    def get_paddle_tensor(self, api_config):
        if self.paddle_tensor is None:
            if self.cpu_tensor is not None:
                torch_tensor = self.cpu_tensor.to(
                    device=torch.device("cuda:0") if self._use_gpu(api_config) else "cpu",
                    copy=True,
                )
                self.paddle_tensor = paddle.utils.dlpack.from_dlpack(
                    torch.utils.dlpack.to_dlpack(torch_tensor)
                )
                self.paddle_tensor.stop_gradient = not self._requires_autograd(api_config)
                return self.paddle_tensor
            if self._logical_numpy_tensor(api_config) is None and self._use_gpu(api_config):
                return self.get_gpu_paddle_tensor(api_config)
            if not self.is_contiguous and self.strides is not None:
                self.paddle_tensor = self._create_paddle_strided(api_config)
                print(
                    f"[non-contiguous] target strides: {self.strides}, "
                    f"actual strides: {self.paddle_tensor.strides}, "
                    f"shape: {list(self.paddle_tensor.shape)}, "
                    f"dtype: {self.paddle_tensor.dtype}, "
                    f"is_contiguous: {self.paddle_tensor.is_contiguous()}"
                )
            else:
                requires_autograd = self._requires_autograd(api_config)
                intermediate_dtype = (
                    "float32"
                    if self.dtype == "bfloat16"
                    else ("float16" if self.dtype in FLOAT8_DTYPES else self.dtype)
                )
                self.paddle_tensor = paddle.to_tensor(
                    self._logical_numpy_tensor(api_config),
                    dtype=intermediate_dtype,
                    place=self.place,
                )

                if self.dtype == "bfloat16":
                    self.paddle_tensor = paddle.cast(self.paddle_tensor, dtype="bfloat16")
                elif self.dtype in FLOAT8_DTYPES:
                    self.paddle_tensor = paddle.cast(self.paddle_tensor, dtype=self.dtype)
                self.paddle_tensor.stop_gradient = not requires_autograd
        if TEST_NON_CONTIGUOUS:
            if not self.shuffle_dims:
                ndim = self.paddle_tensor.dim()
                self.shuffle_dims = list(range(ndim))
                random.shuffle(self.shuffle_dims)
            print("paddle shuffle:", self.shuffle_dims)
            return paddle.transpose(self.paddle_tensor, self.shuffle_dims)
        return self.paddle_tensor

    def _strided_storage_size(self):
        storage_size = 1
        for i in range(len(self.shape)):
            if self.shape[i] > 0:
                storage_size += (self.shape[i] - 1) * self.strides[i]
        return storage_size

    def _create_paddle_strided(self, api_config):
        """基于共享逻辑 NumPy 输入创建非连续 Paddle Tensor。"""
        flag_name = "FLAGS_check_nan_inf"
        original_flag = paddle.get_flags([flag_name])
        paddle.set_flags({flag_name: False})
        try:
            intermediate_dtype = "float16" if self.dtype in FLOAT8_DTYPES else self.dtype
            storage_size = self._strided_storage_size()
            flat_tensor = paddle.zeros(
                [storage_size],
                dtype=intermediate_dtype,
                device=self.place,
            )
            tensor = paddle.as_strided(flat_tensor, self.shape, self.strides)
            logical_tensor = self._logical_numpy_tensor(api_config)
            if logical_tensor.size > 0:
                tensor[...] = paddle.to_tensor(
                    logical_tensor,
                    dtype=intermediate_dtype,
                    place=self.place,
                )
            if self.dtype in FLOAT8_DTYPES:
                flat_tensor = paddle.cast(flat_tensor, dtype=self.dtype)
                tensor = paddle.as_strided(flat_tensor, self.shape, self.strides)

            tensor.stop_gradient = not self._requires_autograd(api_config)
            return tensor
        finally:
            paddle.set_flags(original_flag)

    def get_torch_tensor(self, api_config):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        torch.set_default_device(device)
        if self.torch_tensor is None:
            if self.cpu_tensor is not None:
                self.torch_tensor = self.cpu_tensor.to(device=device, copy=True)
                if self._requires_autograd(api_config):
                    self.torch_tensor = self.torch_tensor.detach().requires_grad_(True)
                return self.torch_tensor
            if self._logical_numpy_tensor(api_config) is None and self._use_gpu(api_config):
                return self.get_gpu_torch_tensor(api_config)
            if not self.is_contiguous and self.strides is not None:
                self.torch_tensor = self._create_torch_strided(api_config)
            else:
                needs_cast = self.dtype in CAST_THROUGH_INTERMEDIATE_DTYPES
                if needs_cast:
                    intermediate_torch_dtype = (
                        torch.float32 if self.dtype == "bfloat16" else torch.float16
                    )
                else:
                    intermediate_torch_dtype = self.to_torch_dtype(self.dtype)
                requires_grad = self._requires_autograd(api_config)
                self.torch_tensor = torch.tensor(
                    self._logical_numpy_tensor(api_config),
                    dtype=intermediate_torch_dtype,
                    requires_grad=requires_grad and not needs_cast,
                )
                if needs_cast:
                    self.torch_tensor = self.torch_tensor.to(dtype=self.to_torch_dtype(self.dtype))
                    if requires_grad:
                        self.torch_tensor = self.torch_tensor.detach().requires_grad_(True)
        if TEST_NON_CONTIGUOUS:
            if not self.shuffle_dims:
                ndim = self.torch_tensor.dim()
                self.shuffle_dims = list(range(ndim))
                random.shuffle(self.shuffle_dims)
            print("torch shuffle:", self.shuffle_dims)
            return torch.permute(self.torch_tensor, self.shuffle_dims)
        return self.torch_tensor

    def _create_torch_strided(self, api_config):
        """基于共享逻辑 NumPy 输入创建非连续 Torch Tensor。"""
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        needs_intermediate = self.dtype in FLOAT8_DTYPES
        if needs_intermediate:
            intermediate_torch_dtype = torch.float16
        else:
            intermediate_torch_dtype = self.to_torch_dtype(self.dtype)

        flat_tensor = torch.empty(
            self._strided_storage_size(),
            dtype=intermediate_torch_dtype,
            device=device,
        )
        tensor = torch.as_strided(flat_tensor, self.shape, self.strides)
        logical_tensor = self._logical_numpy_tensor(api_config)
        if logical_tensor.size > 0:
            tensor.copy_(
                torch.tensor(
                    logical_tensor,
                    dtype=intermediate_torch_dtype,
                    device=device,
                )
            )
        if self.dtype in FLOAT8_DTYPES:
            flat_tensor = flat_tensor.to(dtype=self.to_torch_dtype(self.dtype))
            tensor = torch.as_strided(flat_tensor, self.shape, self.strides)

        if self._requires_autograd(api_config):
            tensor = tensor.detach().requires_grad_(True)
        return tensor

    def _logical_numpy_tensor(self, api_config):
        return input_value(api_config, self)

    def clear_tensor(self, api_config=None):
        if api_config is not None:
            # 清理 TensorConfig 时，也要清掉其关联的逻辑 payload。
            clear_input_value(api_config, self)
        self.torch_tensor = None
        self.paddle_tensor = None
        self.numpy_tensor = None
        self.cpu_tensor = None
        if not is_gpu_mode():
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        del self.paddle_tensor
        self.paddle_tensor = None
        if not is_gpu_mode():
            paddle.device.cuda.empty_cache()

    def clear_numpy_tensor(self, api_config=None):
        if api_config is not None:
            clear_input_value(api_config, self)
        del self.numpy_tensor
        self.numpy_tensor = None

    def clear_torch_tensor(self):
        del self.torch_tensor
        self.torch_tensor = None
        if not is_gpu_mode():
            torch.cuda.empty_cache()

    def save_cpu_copy(self, api_config):
        """保留一份不可变 CPU 副本，用于重建隔离后的测试输入。"""
        if self.cpu_tensor is not None:
            return
        tensor = self.get_torch_tensor(api_config)
        self.cpu_tensor = tensor.detach().to(device="cpu", copy=True)
        self.paddle_tensor = None
        self.torch_tensor = None

    def clear_cpu_copy(self):
        self.cpu_tensor = None

    def fill_numpy_tensor(self, full_value):
        self.numpy_tensor = numpy.full(shape=self.shape, fill_value=full_value, dtype=self.dtype)

    def check_arg(self, api_config, arg_pos, arg_name):
        """检查api_config中的参数是否与当前实例匹配。
        必须同时提供参数位置与参数名称, 具体请查看API文档。

        Args:
            api_config (ApiConfig): API配置对象, 包含args和kwargs。
            arg_pos (int): 参数的位置索引。
            arg_name (str): 参数的名称。

        Returns:
            bool: 如果参数匹配当前实例，则返回 True; 否则返回 False。

        """
        return (hasattr(self, "index") and self.index == arg_pos) or (
            hasattr(self, "key") and self.key == arg_name
        )

    def get_arg(self, api_config, arg_pos, arg_name, default=None):
        """从api_config中获取参数值。
        必须同时提供参数位置与参数名称, 具体请查看API文档。

        Args:
            api_config (ApiConfig): API配置对象, 包含args和kwargs。
            arg_pos (int): 参数的位置索引。
            arg_name (str): 参数的名称。
            default (Any, optional): 参数的默认值。默认为None。

        Returns:
            Any: 参数的值。如果参数位置索引有效, 则返回args列表中对应位置的值;
                    如果参数名称在kwargs字典中存在, 则返回对应名称的值;
                    否则返回默认值。

        """
        if 0 <= arg_pos < len(api_config.args):
            return api_config.args[arg_pos]
        if arg_name in api_config.kwargs:
            return api_config.kwargs[arg_name]
        return default

    def get_initialized_value(self, api_config, arg_pos=None, arg_name=None):
        """从 api_config 中取已初始化的 numpy_tensor，而不是直接读当前 TensorConfig。"""
        # 未初始化时返回 None，因为 numpy_tensor 本身就是 None。
        if arg_pos is not None and 0 <= arg_pos < len(api_config.args):
            if isinstance(api_config.args[arg_pos], TensorConfig):
                return input_value(api_config, api_config.args[arg_pos])
            else:
                return api_config.args[arg_pos]
        if arg_name and arg_name in api_config.kwargs:
            if isinstance(api_config.kwargs[arg_name], TensorConfig):
                return input_value(api_config, api_config.kwargs[arg_name])
            else:
                return api_config.kwargs[arg_name]
        # 参数不在 api_config 中时返回 None。
        if arg_pos >= len(api_config.args) or arg_name not in api_config.kwargs:
            return None
        # 下面处理参数不合法的错误分支。
        if arg_pos is None and arg_name is None:
            raise ValueError("either arg_pos or arg_name must be provided.")
        elif arg_pos:
            if arg_pos < 0:
                raise IndexError(
                    f"argument position {arg_pos} is out of range for api_config with {len(api_config.args)} arguments."
                )
            else:
                # args[arg_pos] 不是 TensorConfig。
                raise TypeError(f"argument at position {arg_pos} is not of type TensorConfig.")
        else:
            # kwargs[arg_name] 不是 TensorConfig。
            raise TypeError(f"argument '{arg_name}' is not of type TensorConfig.")

    def set_tensor_arg_value(self, api_config, arg_pos=None, arg_name=None, value=None):
        if (
            arg_pos is not None
            and 0 <= arg_pos < len(api_config.args)
            and isinstance(api_config.args[arg_pos], TensorConfig)
        ):
            write_input_value(api_config, api_config.args[arg_pos], value)
        elif (
            arg_name
            and arg_name in api_config.kwargs
            and isinstance(api_config.kwargs[arg_name], TensorConfig)
        ):
            write_input_value(api_config, api_config.kwargs[arg_name], value)
        else:
            raise ValueError(
                f"argument at position {arg_pos} or name '{arg_name}' is not of type TensorConfig."
            )

    def random_numpy(self, shape=None, data_type=None, min=None, max=None):
        """按 shape 和 dtype 生成位于 [min, max) 的随机 NumPy 数组。"""
        if "int" in data_type:
            min = min if min is not None else -65535
            max = max if max is not None else 65535
            numpy_tensor = (numpy.random.randint(min, max, size=shape)).astype(data_type)
        elif data_type.startswith("complex"):
            real_dtype = "float32" if data_type == "complex64" else "float64"
            real_min = min if min is not None else numpy.finfo(real_dtype).min / 2
            real_max = max if max is not None else numpy.finfo(real_dtype).max / 2
            real_part = numpy.random.uniform(real_min, real_max, size=shape).astype(real_dtype)
            imag_part = numpy.random.uniform(real_min, real_max, size=shape).astype(real_dtype)
            numpy_tensor = (real_part + 1j * imag_part).astype(data_type)
        else:
            dtype = "float32" if data_type == "bfloat16" else data_type
            min = min if min is not None else numpy.finfo(dtype).min / 2
            max = max if max is not None else numpy.finfo(dtype).max / 2
            numpy_tensor = (numpy.random.uniform(min, max, size=shape)).astype(dtype)
        return numpy_tensor
