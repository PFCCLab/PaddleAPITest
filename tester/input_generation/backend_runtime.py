"""输入 backend 的策略、factory、预热与输出梯度生命周期。"""

from __future__ import annotations

import numbers
import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy
from tester.dtype_utils import to_torch_dtype

from . import backend as _backend_impl
from .backend import (
    INPUT_NUMPY_RANDOM_STATE,
    InputBackend,
    InputConfigRandomState,
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    derive_input_seed,
)

INPUT_BACKEND_ENV_VAR = "PADDLEAPITEST_INPUT_BACKEND"
# backend 原语在初始化时读取同一缓存，避免 runtime 预热和 case backend 分叉。
_PREPARED_INPUT_BACKENDS = _backend_impl._PREPARED_INPUT_BACKENDS
_CACHED_NUMPY_OUTPUT_GRADS = {}
_TRUE_VALUES = {"true", "1", "yes", "y"}
_VALID_INPUT_BACKENDS = frozenset({"numpy", "torch", "paddle"})
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
    {"paddle_gpu_performance", "torch_gpu_performance", "paddle_torch_gpu_performance"}
)
# runtime 层只保存策略，不保存任何 API 规则状态。


def _env_flag(name, default="False") -> bool:
    # 环境变量只在策略解析时读取一次，worker 内不再重复解释字符串。
    return os.getenv(name, default).lower() in _TRUE_VALUES


def _normalize_shape(shape, *, scalar_empty):
    # NumPy 和原生 backend 对标量 shape 的表示不同，这里统一边界格式。
    if shape is None:
        return [] if scalar_empty else ()
    if isinstance(shape, numbers.Integral):
        return [int(shape)] if scalar_empty else (int(shape),)
    return list(shape) if scalar_empty else tuple(shape)


def create_input_backend(input_random_state, *, policy):
    """按冻结策略创建一次性 backend 实例。"""
    input_random_state = input_random_state or INPUT_NUMPY_RANDOM_STATE
    # factory 只依赖冻结 policy，避免生成阶段再次读取环境状态。
    device = policy.logical_device
    if policy.resolved == "numpy":
        # NumPy backend 永远使用 CPU 逻辑设备。
        return NumPyInputBackend(input_random_state)
    if policy.resolved == "torch":
        # Torch backend 的 device 由 policy 统一决定。
        return TorchInputBackend(input_random_state, device=device)
    if policy.resolved == "paddle":
        return PaddleInputBackend(input_random_state, device=device)
    raise ValueError(f"unsupported input generation backend: {policy.resolved!r}")


def resolve_input_backend_policy(
    *, requested=None, use_gpu_mode=None, use_cached_numpy=None, mode=None
):
    """一次性解析 backend、模式默认值和逻辑值设备。"""
    # 显式参数优先于环境变量，便于 engineV4 为单次运行固定行为。
    requested = os.environ.get(INPUT_BACKEND_ENV_VAR) if requested is None else requested
    normalized_requested = (requested or "").strip().lower() or None
    # 空字符串表示调用方没有覆盖默认策略。
    if normalized_requested is not None and normalized_requested not in _VALID_INPUT_BACKENDS:
        raise ValueError(f"unsupported input generation backend: {requested!r}")
    use_gpu_mode = _env_flag("USE_GPU_MODE") if use_gpu_mode is None else bool(use_gpu_mode)
    use_cached_numpy = (
        _env_flag("USE_CACHED_NUMPY") if use_cached_numpy is None else bool(use_cached_numpy)
    )
    # GPU 模式不能复用主存缓存，否则资源预估与真实物化会不一致。
    use_cached_numpy = use_cached_numpy and not use_gpu_mode
    resolved = (
        "numpy" if use_cached_numpy else normalized_requested or _MODE_DEFAULT_BACKENDS.get(mode)
    )
    resolved = resolved or ("torch" if use_gpu_mode else "numpy")
    # 性能模式默认要求原生设备，显式 NumPy 仍然保留 CPU 语义。
    effective_gpu_mode = use_gpu_mode or (mode in _GPU_NATIVE_MODES and resolved != "numpy")
    logical_device = {
        "numpy": "cpu",
        "torch": "cuda:0" if effective_gpu_mode else "cpu",
        "paddle": "gpu:0" if effective_gpu_mode else "cpu",
    }[resolved]
    return InputBackendPolicy(
        normalized_requested, resolved, logical_device, effective_gpu_mode, use_cached_numpy, mode
    )


@dataclass(frozen=True)
class InputBackendPolicy:
    """一次运行内共享的输入 backend 请求、解析结果和逻辑值设备。"""

    requested: str | None
    resolved: str
    logical_device: str
    use_gpu_mode: bool
    use_cached_numpy: bool
    mode: str | None = None


def prepare_input_backend(policy):
    """准备进程级 backend 模块和设备 context。"""
    if policy is None:
        raise ValueError("input backend policy is required for runtime preparation")
    # 预热仅创建常量探针，不推进配置输入的随机流。
    # cache key 同时包含 backend 和设备，防止 CPU/GPU context 交叉复用。
    cache_key = (policy.resolved, policy.logical_device)
    if cache_key in _PREPARED_INPUT_BACKENDS:
        return _PREPARED_INPUT_BACKENDS[cache_key]
    input_random_state = (
        INPUT_NUMPY_RANDOM_STATE
        if policy.resolved == "numpy"
        else SimpleNamespace(seed=0, config_fingerprint="", stream_kind="runtime_probe")
    )
    backend = create_input_backend(input_random_state, policy=policy)
    probe = backend.zeros((1,), dtype="float32")
    if backend.name == "torch" and policy.logical_device.startswith("cuda"):
        backend._torch().cuda.synchronize(backend._device)
    elif backend.name == "paddle" and policy.logical_device.startswith(("gpu", "cuda")):
        backend._paddle().device.cuda.synchronize()
    del probe
    _PREPARED_INPUT_BACKENDS[cache_key] = backend
    return backend


def _cached_numpy_output_grad(dtype, shape, stream_kind, seed, config_fingerprint):
    # 缓存只服务 NumPy output grad，原生 backend 仍使用各自 generator。
    if dtype in {"float8_e5m2", "float8_e4m3fn"}:
        dtype = "float16"
    elif dtype == "bfloat16":
        dtype = "float32"
    shape = _normalize_shape(shape, scalar_empty=False)
    key = (dtype, shape, stream_kind, int(seed), str(config_fingerprint))
    if key not in _CACHED_NUMPY_OUTPUT_GRADS:
        rng = numpy.random.RandomState(
            derive_input_seed(seed, config_fingerprint, f"cached_numpy:{stream_kind}")
        )
        if "int" in dtype:
            value = rng.randint(-65535, 65535, size=shape, dtype="int64").astype(dtype)
        elif dtype.startswith("complex"):
            real_dtype = "float32" if dtype == "complex64" else "float64"
            value = (rng.random(shape) - 0.5).astype(real_dtype) + 1j * (
                rng.random(shape) - 0.5
            ).astype(real_dtype)
            value = value.astype(dtype)
        else:
            value = (rng.random(shape) - 0.5).astype(dtype)
        _CACHED_NUMPY_OUTPUT_GRADS[key] = value
    return _CACHED_NUMPY_OUTPUT_GRADS[key]


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
    """用独立随机流生成 output grad。"""
    # output grad 使用独立 stream，不能推进前向输入的随机状态。
    dtype = str(dtype)
    stream_kind = f"output_grad:{backend_name}:{int(stream_index)}"
    if cache_enabled:
        # 只有 NumPy cache 可跨调用复用，原生 Tensor 绑定设备上下文。
        if backend_name != "numpy":
            raise ValueError("output-grad cache requires the NumPy backend")
        return _cached_numpy_output_grad(dtype, shape, stream_kind, seed, config_fingerprint)
    # 非缓存路径按 backend 创建私有随机源，避免污染框架全局 RNG。
    if backend_name == "numpy":
        backend = NumPyInputBackend(
            InputConfigRandomState(seed, config_fingerprint, stream_kind=stream_kind)
        )
    else:
        stream_identity = SimpleNamespace(
            seed=seed, config_fingerprint=config_fingerprint, stream_kind=stream_kind
        )
        if backend_name == "torch":
            backend = TorchInputBackend(stream_identity, device=device)
        elif backend_name == "paddle":
            backend = PaddleInputBackend(stream_identity, device=device)
        else:
            raise ValueError(f"unsupported output-grad backend: {backend_name!r}")
    if "int" in dtype:
        return backend.randint(-65535, 65535, shape=shape, dtype=dtype)
    base_dtype = (
        "float16"
        if dtype in {"float8_e5m2", "float8_e4m3fn"}
        else "float32"
        if dtype == "bfloat16"
        else dtype
    )
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        return backend.cast(
            backend.random(shape, dtype=real_dtype)
            - 0.5
            + 1j * (backend.random(shape, dtype=real_dtype) - 0.5),
            dtype,
        )
    value = backend.uniform(-0.5, 0.5, shape=shape, dtype=base_dtype)
    if base_dtype == dtype or backend_name == "numpy":
        return value
    if backend_name == "torch":
        return value.to(dtype=to_torch_dtype(dtype))
    return backend._paddle().cast(value, dtype=dtype)


__all__ = [
    "InputBackendPolicy",
    "create_input_backend",
    "generate_output_grad",
    "prepare_input_backend",
    "resolve_input_backend_policy",
]
