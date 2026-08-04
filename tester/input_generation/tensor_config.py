from __future__ import annotations

import copy
import os
import random

import numpy
import paddle
import yaml
from tester.dtype_utils import to_torch_dtype

from .backend import create_input_backend
from .value import clear_input_value, read_input_value, read_input_value_backend

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
_TRUE_VALUES = {"true", "1", "yes", "y"}


def _load_forward_only_apis():
    config_path = os.path.join(os.path.dirname(__file__), "..", "base_config.yaml")
    with open(config_path, encoding="utf-8") as f:
        return frozenset(yaml.safe_load(f).get("forward_only_apis", []))


FORWARD_ONLY_APIS = _load_forward_only_apis()
optimizer_apis = OPTIMIZER_APIS


def is_gpu_mode():
    return os.getenv("USE_GPU_MODE", str(USE_GPU_MODE)).lower() in _TRUE_VALUES


def _torch_compute_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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
        self.input_value = None
        self.input_value_backend = None
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
        result.input_value = None
        result.input_value_backend = None
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

    def numel(self):
        return _numel(self.shape)

    def _use_gpu(self):
        if not is_gpu_mode():
            return False
        if self.place is not None and "cpu" in str(self.place).lower():
            return False
        return "gpu" in paddle.device.get_device()

    def _requires_autograd(self, api_config):
        if self.dtype not in AUTOGRAD_DTYPES:
            return False
        api_name = getattr(api_config, "api_name", "")
        api = api_name[api_name.rindex(".") + 1 :] if "." in api_name else api_name
        if api in FORWARD_ONLY_APIS:
            return False
        return getattr(api_config, "test_backward", True)

    def _torch_device_for_paddle(self, api_config):
        if self.place is not None and "cpu" in str(self.place).lower():
            return torch.device("cpu")
        paddle_device = paddle.device.get_device()
        if "gpu" in paddle_device or "cuda" in paddle_device or self._use_gpu():
            return _torch_compute_device()
        return torch.device("cpu")

    def _missing_input_error(self, api_config, framework):
        return ValueError(
            f"TensorConfig has no generated input value before {framework} materialization: "
            f"api={getattr(api_config, 'api_name', '<unknown>')}, "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"backend={create_input_backend().name}"
        )

    def _cast_intermediate_dtype(self):
        if self.dtype == "bfloat16":
            return "float32"
        if self.dtype in FLOAT8_DTYPES:
            return "float16"
        return self.dtype

    def _torch_cast_dtype(self):
        if self.dtype == "bfloat16":
            return torch.float32
        if self.dtype in FLOAT8_DTYPES:
            return torch.float16
        return to_torch_dtype(self.dtype)

    def _float8_intermediate_dtype(self):
        return "float16" if self.dtype in FLOAT8_DTYPES else self.dtype

    def _torch_float8_intermediate_dtype(self):
        return torch.float16 if self.dtype in FLOAT8_DTYPES else to_torch_dtype(self.dtype)

    def _logical_numel(self, value):
        if hasattr(value, "numel"):
            return value.numel()
        return getattr(value, "size", 0)

    def _logical_paddle_tensor(self, api_config, dtype, place=None):
        value = read_input_value(api_config, self)
        backend = read_input_value_backend(api_config, self)
        if backend == "paddle" and isinstance(value, paddle.Tensor):
            if place is not None:
                if "cpu" in str(place).lower():
                    value = value._copy_to(paddle.CPUPlace(), False)
                elif "gpu" in str(place).lower():
                    device_id = int(str(place).rsplit(":", 1)[-1]) if ":" in str(place) else 0
                    value = value._copy_to(paddle.CUDAPlace(device_id), False)
            if dtype is not None and str(value.dtype).split(".")[-1] != str(dtype):
                value = paddle.cast(value, dtype=dtype)
            return value
        if backend == "torch" and isinstance(value, torch.Tensor):
            torch_tensor = value.detach().to(device=self._torch_device_for_paddle(api_config))
            paddle_tensor = paddle.utils.dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(torch_tensor)
            )
            if dtype is not None and str(paddle_tensor.dtype).split(".")[-1] != str(dtype):
                paddle_tensor = paddle.cast(paddle_tensor, dtype=dtype)
            return paddle_tensor
        return paddle.to_tensor(value, dtype=dtype, place=place)

    def get_paddle_tensor(self, api_config):
        if self.paddle_tensor is None:
            if self.cpu_tensor is not None:
                torch_tensor = self.cpu_tensor.to(
                    device=torch.device("cuda:0") if self._use_gpu() else "cpu",
                    copy=True,
                )
                self.paddle_tensor = paddle.utils.dlpack.from_dlpack(
                    torch.utils.dlpack.to_dlpack(torch_tensor)
                )
                self.paddle_tensor.stop_gradient = not self._requires_autograd(api_config)
                return self.paddle_tensor
            if read_input_value(api_config, self) is None:
                raise self._missing_input_error(api_config, "Paddle")
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
                intermediate_dtype = self._cast_intermediate_dtype()
                self.paddle_tensor = self._logical_paddle_tensor(
                    api_config,
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
        """基于共享逻辑输入创建非连续 Paddle Tensor。"""
        flag_name = "FLAGS_check_nan_inf"
        original_flag = paddle.get_flags([flag_name])
        paddle.set_flags({flag_name: False})
        try:
            intermediate_dtype = self._float8_intermediate_dtype()
            storage_size = self._strided_storage_size()
            flat_tensor = paddle.zeros(
                [storage_size],
                dtype=intermediate_dtype,
                device=self.place,
            )
            tensor = paddle.as_strided(flat_tensor, self.shape, self.strides)
            logical_value = read_input_value(api_config, self)
            if self._logical_numel(logical_value) > 0:
                tensor[...] = self._logical_paddle_tensor(
                    api_config,
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
        device = _torch_compute_device() if self._use_gpu() else torch.device("cpu")
        torch.set_default_device(device)
        if self.torch_tensor is None:
            if self.cpu_tensor is not None:
                self.torch_tensor = self.cpu_tensor.to(device=device, copy=True)
                if self._requires_autograd(api_config):
                    self.torch_tensor = self.torch_tensor.detach().requires_grad_(True)
                return self.torch_tensor
            if read_input_value(api_config, self) is None:
                raise self._missing_input_error(api_config, "Torch")
            if not self.is_contiguous and self.strides is not None:
                self.torch_tensor = self._create_torch_strided(api_config)
            else:
                needs_cast = self.dtype in CAST_THROUGH_INTERMEDIATE_DTYPES
                intermediate_torch_dtype = self._torch_cast_dtype()
                requires_grad = self._requires_autograd(api_config)
                self.torch_tensor = self._logical_torch_tensor(
                    api_config,
                    dtype=intermediate_torch_dtype,
                    device=device,
                    requires_grad=requires_grad and not needs_cast,
                )
                if needs_cast:
                    self.torch_tensor = self.torch_tensor.to(dtype=to_torch_dtype(self.dtype))
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
        """基于共享逻辑输入创建非连续 Torch Tensor。"""
        device = _torch_compute_device() if self._use_gpu() else torch.device("cpu")
        intermediate_torch_dtype = self._torch_float8_intermediate_dtype()

        flat_tensor = torch.empty(
            self._strided_storage_size(),
            dtype=intermediate_torch_dtype,
            device=device,
        )
        tensor = torch.as_strided(flat_tensor, self.shape, self.strides)
        logical_value = read_input_value(api_config, self)
        if self._logical_numel(logical_value) > 0:
            tensor.copy_(
                self._logical_torch_tensor(
                    api_config,
                    dtype=intermediate_torch_dtype,
                    device=device,
                )
            )
        if self.dtype in FLOAT8_DTYPES:
            flat_tensor = flat_tensor.to(dtype=to_torch_dtype(self.dtype))
            tensor = torch.as_strided(flat_tensor, self.shape, self.strides)

        if self._requires_autograd(api_config):
            tensor = tensor.detach().requires_grad_(True)
        return tensor

    def _logical_torch_tensor(self, api_config, dtype, device, requires_grad=False):
        value = read_input_value(api_config, self)
        backend = read_input_value_backend(api_config, self)
        if backend == "torch" and isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
            if requires_grad:
                tensor = tensor.detach().requires_grad_(True)
            return tensor
        if backend == "paddle" and isinstance(value, paddle.Tensor):
            paddle_tensor = value.detach()
            if self._use_gpu() and device.type == "cuda":
                paddle_tensor = paddle_tensor._copy_to(paddle.CUDAPlace(device.index or 0), False)
            elif device.type == "cpu":
                paddle_tensor = paddle_tensor._copy_to(paddle.CPUPlace(), False)
            tensor = torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(paddle_tensor)
            ).to(device=device, dtype=dtype)
            # DLPack avoids NumPy, then clone so Torch accuracy owns its input storage.
            tensor = tensor.clone()
            if requires_grad:
                tensor = tensor.detach().requires_grad_(True)
            return tensor
        return torch.tensor(value, dtype=dtype, device=device, requires_grad=requires_grad)

    def clear_tensor(self, api_config=None):
        if api_config is not None:
            # 清理 TensorConfig 时，也要清掉其关联的输入数据。
            clear_input_value(api_config, self)
        self.torch_tensor = None
        self.paddle_tensor = None
        self.input_value = None
        self.input_value_backend = None
        self.cpu_tensor = None
        if not is_gpu_mode():
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        self.paddle_tensor = None
        if not is_gpu_mode():
            paddle.device.cuda.empty_cache()

    def clear_torch_tensor(self):
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
