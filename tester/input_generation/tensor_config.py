from __future__ import annotations

import copy
import os
import random

import paddle
import yaml
from tester.dtype_utils import to_torch_dtype

from .values import clear_input_value, read_input_value, read_input_value_backend


class _LazyTorch:
    def __getattr__(self, name):
        import torch

        globals()["torch"] = torch
        return getattr(torch, name)


torch = _LazyTorch()

TEST_NON_CONTIGUOUS = os.getenv("TEST_NON_CONTIGUOUS", "0").lower() in ("true", "1")
USE_GPU_MODE = os.getenv("USE_GPU_MODE", "False").lower() == "true"
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}
# dtype 集合既服务实际物化，也服务预检；新增 dtype 时必须同步元素字节数。
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


def is_gpu_mode():
    return os.getenv("USE_GPU_MODE", str(USE_GPU_MODE)).lower() in _TRUE_VALUES


def _torch_compute_device():
    # GPU 算子缺少 CUDA 时应显式失败，不能因环境差异静默改测 CPU kernel。
    return torch.device("cuda:0")


def dtype_name(dtype):
    """将框架 dtype 和配置字符串统一为稳定名称。"""
    return str(dtype).replace("paddle.", "").replace("torch.", "").split(".")[-1]


def dtype_element_size(dtype, *, default=None):
    """返回配置 dtype 的元素字节数；未知类型必须由调用方显式处理。"""
    size = _DTYPE_BYTES.get(dtype_name(dtype))
    if size is not None:
        return size
    if default is not None:
        return int(default)
    raise ValueError(f"unknown TensorConfig dtype size: {dtype!r}")


def shape_numel(shape):
    """计算配置 shape 的逻辑元素数，标量 shape 的结果为 1。"""
    # Python int 无溢出风险，不能把超过 int32/int64 的目标 case 截断。
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def shape_storage_numel(shape, *, is_contiguous=True, strides=None):
    """计算承载一个配置 Tensor 所需的底层 storage 元素数。"""
    # 非连续 storage 取最大可达 offset + 1；零尺寸不访问任何 storage。
    logical_numel = shape_numel(shape)
    if logical_numel == 0:
        return 0
    if logical_numel < 0:
        raise ValueError(f"negative TensorConfig shape is not a valid storage: {shape!r}")
    if is_contiguous or strides is None:
        return logical_numel
    if len(shape) != len(strides):
        raise ValueError(f"shape and strides rank mismatch: shape={shape!r}, strides={strides!r}")

    storage_numel = 1
    for dimension, stride in zip(shape, strides, strict=True):
        dimension = int(dimension)
        stride = int(stride)
        if dimension < 0 or stride < 0:
            raise ValueError(
                f"negative shape/stride is not supported: shape={shape!r}, strides={strides!r}"
            )
        if dimension > 0:
            storage_numel += (dimension - 1) * stride
    return storage_numel


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
        return shape_numel(self.shape)

    def storage_numel(self):
        """返回物化该配置所需的 storage 元素数。"""
        return shape_storage_numel(
            self.shape,
            is_contiguous=self.is_contiguous,
            strides=self.strides,
        )

    def nbytes(self, *, storage=True):
        """返回逻辑 Tensor 或实际 storage 的配置字节数。"""
        numel = self.storage_numel() if storage else self.numel()
        if numel < 0:
            raise ValueError(f"negative TensorConfig numel is invalid: {self.shape!r}")
        return numel * dtype_element_size(self.dtype)

    def _paddle_kernel_uses_gpu(self, api_config):
        """判断 Paddle 被测 kernel 的输入是否应物化到 GPU。"""
        if self.place is not None and "cpu" in str(self.place).lower():
            return False
        if getattr(api_config, "test_cpu", False):
            return False
        return "gpu" in paddle.device.get_device()

    def _torch_operator_device(self, api_config):
        # 配置显式 CPU place 仍是输入协议；除此之外 Torch reference 始终使用计算卡。
        if self.place is not None and "cpu" in str(self.place).lower():
            return torch.device("cpu")
        return _torch_compute_device()

    def _requires_autograd(self, api_config):
        if self.dtype not in AUTOGRAD_DTYPES:
            return False
        api_name = getattr(api_config, "api_name", "")
        api = api_name[api_name.rindex(".") + 1 :] if "." in api_name else api_name
        if api in FORWARD_ONLY_APIS:
            return False
        return getattr(api_config, "test_backward", True)

    def _torch_source_device_for_paddle(self, api_config):
        """返回 DLPack 转入 Paddle 前，Torch 源值需要到达的设备。"""
        if self.place is not None and "cpu" in str(self.place).lower():
            return torch.device("cpu")
        if getattr(api_config, "test_cpu", False):
            return torch.device("cpu")
        if self._paddle_kernel_uses_gpu(api_config):
            return _torch_compute_device()
        return torch.device("cpu")

    def _missing_input_error(self, api_config, framework):
        return ValueError(
            f"TensorConfig has no generated input value before {framework} materialization: "
            f"api={getattr(api_config, 'api_name', '<unknown>')}, "
            f"shape={self.shape}, dtype={self.dtype}"
        )

    def _cast_intermediate_dtype(self):
        if self.dtype == "bfloat16":
            return "float32"
        if self.dtype in FLOAT8_DTYPES:
            return "float16"
        return self.dtype

    def _torch_cast_dtype(self):
        # 先统一中间 dtype，再映射到 Torch，避免两套分支漂移。
        return to_torch_dtype(self._cast_intermediate_dtype())

    def _float8_intermediate_dtype(self):
        return "float16" if self.dtype in FLOAT8_DTYPES else self.dtype

    def _torch_float8_intermediate_dtype(self):
        return to_torch_dtype(self._float8_intermediate_dtype())

    def _logical_numel(self, value):
        if hasattr(value, "numel"):
            return value.numel()
        return getattr(value, "size", 0)

    def _logical_paddle_tensor(self, api_config, dtype, place=None):
        value = read_input_value(api_config, self)
        backend = read_input_value_backend(api_config, self)
        # 有效 place 在 backend 分支前统一解析，避免 NumPy 路径绕过 test_cpu 优先级。
        operator_place = paddle.CPUPlace() if getattr(api_config, "test_cpu", False) else place
        if backend == "paddle" and isinstance(value, paddle.Tensor):
            # backend-native 逻辑值的设备属于生成策略；框架输入仍必须服从算子设备。
            target_place = operator_place
            if target_place is None:
                target_place = (
                    paddle.CUDAPlace(0)
                    if self._paddle_kernel_uses_gpu(api_config)
                    else paddle.CPUPlace()
                )
            # place 字符串在 CPU/CUDAPlace 上稳定包含设备编号，可用于避免同设备复制。
            if str(value.place).lower() != str(target_place).lower():
                # Paddle 的同 GPU `_copy_to()` 会引入额外同步，长驻 worker 中可能阻塞后续 case。
                # 只有算子 place 与生成 place 不同时才允许发生真实设备搬运。
                if "cpu" in str(target_place).lower():
                    # 生成源会在全部输入物化后立即释放，跨设备复制必须先完成所有权交接。
                    value = value._copy_to(paddle.CPUPlace(), True)
                elif "gpu" in str(target_place).lower() or isinstance(
                    target_place, paddle.CUDAPlace
                ):
                    device_id = (
                        target_place.get_device_id()
                        if isinstance(target_place, paddle.CUDAPlace)
                        else int(str(target_place).rsplit(":", 1)[-1])
                        if ":" in str(target_place)
                        else 0
                    )
                    value = value._copy_to(paddle.CUDAPlace(device_id), True)
            if dtype is not None and str(value.dtype).split(".")[-1] != str(dtype):
                value = paddle.cast(value, dtype=dtype)
            return value
        if backend == "torch" and isinstance(value, torch.Tensor):
            torch_tensor = value.detach().to(
                device=self._torch_source_device_for_paddle(api_config)
            )
            paddle_tensor = paddle.utils.dlpack.from_dlpack(
                torch.utils.dlpack.to_dlpack(torch_tensor)
            )
            if dtype is not None and str(paddle_tensor.dtype).split(".")[-1] != str(dtype):
                paddle_tensor = paddle.cast(paddle_tensor, dtype=dtype)
            return paddle_tensor
        return paddle.to_tensor(value, dtype=dtype, place=operator_place)

    def get_paddle_tensor(self, api_config):
        if self.paddle_tensor is None:
            if self.cpu_tensor is not None:
                torch_tensor = self.cpu_tensor.to(
                    device=self._torch_source_device_for_paddle(api_config),
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
        return self.storage_numel()

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
                device=(
                    paddle.CPUPlace() if getattr(api_config, "test_cpu", False) else self.place
                ),
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
        device = self._torch_operator_device(api_config)
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
        device = self._torch_operator_device(api_config)
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
            if device.type == "cuda":
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
        torch_tensor = self.torch_tensor
        paddle_tensor = self.paddle_tensor
        self.torch_tensor = None
        self.paddle_tensor = None
        self.cpu_tensor = None
        if not is_gpu_mode():
            # allocator 清理只跟随实际被释放的设备，不能由全局运行模式代替 Tensor place。
            if torch_tensor is not None and torch_tensor.device.type == "cuda":
                torch.cuda.empty_cache()
            if paddle_tensor is not None and paddle_tensor.place.is_gpu_place():
                paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        tensor = self.paddle_tensor
        self.paddle_tensor = None
        # CPU tensor 不属于 CUDA allocator；清理它时访问 CUDA 会把 CPU case 变成设备同步点。
        if not is_gpu_mode() and tensor is not None and tensor.place.is_gpu_place():
            paddle.device.cuda.empty_cache()

    def clear_torch_tensor(self):
        tensor = self.torch_tensor
        self.torch_tensor = None
        if not is_gpu_mode() and tensor is not None and tensor.device.type == "cuda":
            torch.cuda.empty_cache()

    def clear_generated_input_value(self, api_config):
        """释放规则生成值，但保留已物化 Tensor 和 stable CPU 副本。"""
        clear_input_value(api_config, self)

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
