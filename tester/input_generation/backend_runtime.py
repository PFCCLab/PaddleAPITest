"""输入 backend 的策略、factory、预热与输出梯度生命周期。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

from tester.api_config.dtype_utils import to_torch_dtype

from .backend import (
    NumPyInputBackend,
    PaddleInputBackend,
    TorchInputBackend,
    _normalize_shape,
)
from .value_generators import (
    INPUT_NUMPY_RANDOM_STATE,
    InputConfigRandomState,
    generate_symmetric_input_value,
)
from .values import InputTensorSpec

INPUT_BACKEND_ENV_VAR = "PADDLEAPITEST_INPUT_BACKEND"
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


def create_input_backend(input_random_state, *, policy):
    """按冻结策略创建一次性 backend 实例。"""
    return _INPUT_BACKEND_RUNTIME.create(input_random_state, policy=policy)


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
    default_backend = _MODE_DEFAULT_BACKENDS.get(mode)
    if (
        not use_gpu_mode
        and normalized_requested is None
        and mode
        in {
            "accuracy",
            "accuracy_stable",
        }
    ):
        # 普通 accuracy 默认只在 CPU 生成输入，显式 backend 仍由调用方保留。
        default_backend = "numpy"
    resolved = "numpy" if use_cached_numpy else normalized_requested or default_backend
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


@dataclass(frozen=True)
class OutputGradContext:
    """冻结一次 output-grad 生成所需的运行策略。"""

    backend_name: str
    seed: int
    config_fingerprint: str
    max_abs: float
    range_configured: bool
    cache_enabled: bool = False

    @classmethod
    def from_runtime_config(cls, runtime_config, *, config_fingerprint=""):
        """从 worker 已冻结配置读取 output-grad 事实，避免调用方重复拆字段。"""
        backend_name = getattr(runtime_config, "input_backend_resolved", None)
        if backend_name is None:
            raise ValueError("runtime config has no resolved input backend")
        return cls(
            backend_name=backend_name,
            seed=int(getattr(runtime_config, "random_seed", 0)),
            config_fingerprint=str(config_fingerprint),
            max_abs=float(getattr(runtime_config, "output_grad_max_abs", 0.5)),
            range_configured=bool(
                getattr(runtime_config, "output_grad_max_abs_is_configured", False)
            ),
            cache_enabled=bool(getattr(runtime_config, "use_cached_numpy", False)),
        )


class InputBackendRuntime:
    """Own prepared backend handles for one worker process."""

    def __init__(self):
        # backend implementation 不持有 cache；clear 后新一轮 prepare 必须重新探测设备。
        self._prepared = {}
        self._cached_numpy_output_grads = {}
        self._output_grad_stream_counters = {}

    def create(self, input_random_state, *, policy):
        """Create a backend using the prepared handle selected by policy."""
        input_random_state = input_random_state or INPUT_NUMPY_RANDOM_STATE
        # factory 只依赖冻结 policy，避免生成阶段再次读取环境状态。
        device = policy.logical_device
        prepared = (
            # cache hit 只复用不可变 runtime handle，随机 generator 仍由新 backend 独立创建。
            self._prepared.get((policy.resolved, device)) if policy.resolved != "numpy" else None
        )
        if policy.resolved == "numpy":
            # NumPy backend 永远使用 CPU 逻辑设备。
            return NumPyInputBackend(input_random_state)
        if policy.resolved == "torch":
            # Torch backend 的 device 由 policy 统一决定。
            return TorchInputBackend(input_random_state, device=device, prepared=prepared)
        if policy.resolved == "paddle":
            return PaddleInputBackend(input_random_state, device=device, prepared=prepared)
        raise ValueError(f"unsupported input generation backend: {policy.resolved!r}")

    def prepare(self, policy):
        """Prepare and cache one backend module/device context."""
        if policy is None:
            raise ValueError("input backend policy is required for runtime preparation")
        # 预热仅创建常量探针，不推进配置输入的随机流。
        # cache key 同时包含 backend 和设备，防止 CPU/GPU context 交叉复用。
        cache_key = (policy.resolved, policy.logical_device)
        if cache_key in self._prepared:
            return self._prepared[cache_key]
        input_random_state = (
            INPUT_NUMPY_RANDOM_STATE
            if policy.resolved == "numpy"
            else SimpleNamespace(seed=0, config_fingerprint="", stream_kind="runtime_probe")
        )
        backend = self.create(input_random_state, policy=policy)
        probe = backend.zeros((1,), dtype="float32")
        if backend.name == "torch" and policy.logical_device.startswith("cuda"):
            backend._torch().cuda.synchronize(backend._device)
        elif backend.name == "paddle" and policy.logical_device.startswith(("gpu", "cuda")):
            backend._paddle().device.cuda.synchronize()
        del probe
        self._prepared[cache_key] = backend
        return backend

    def clear(self):
        """Drop prepared handles and cached output gradients."""
        self._prepared.clear()
        self._cached_numpy_output_grads.clear()
        self._output_grad_stream_counters.clear()

    def output_grad_context(self, runtime_config, *, config_fingerprint=""):
        # fingerprint 来自 API 配置，不属于可变的 worker runtime 选项。
        return OutputGradContext.from_runtime_config(
            runtime_config, config_fingerprint=config_fingerprint
        )

    def reset_output_grad_streams(self):
        """Reset stream identity at the output-grad slot lifecycle boundary."""
        # 每个 tester case 都从 stream 0 开始，确保相同配置的结果可复现。
        self._output_grad_stream_counters.clear()

    def generate_output_grad(
        self,
        *,
        dtype,
        shape,
        device,
        context,
        stream_index=None,
    ):
        """按冻结上下文生成并物化一个 output grad。"""
        if not isinstance(context, OutputGradContext):
            raise TypeError("output-grad context is required")
        dtype = str(dtype)
        if stream_index is None:
            # stream 序号由 owner 分配，caller 只描述一个输出规格。
            stream_index = self._output_grad_stream_counters.get(context.backend_name, 0)
            self._output_grad_stream_counters[context.backend_name] = stream_index + 1
        # 未显式设置范围时保持 legacy 的 [-0.5, 0.5) 协议。
        max_abs = context.max_abs if context.range_configured else 0.5
        stream_kind = f"output_grad:{context.backend_name}:{int(stream_index)}"
        if context.cache_enabled:
            # 原生 Tensor 绑定设备，只有 NumPy seed 允许跨调用缓存。
            if context.backend_name != "numpy":
                raise ValueError("output-grad cache requires the NumPy backend")
            return self.cached_numpy_output_grad(
                dtype,
                shape,
                stream_kind,
                context.seed,
                context.config_fingerprint,
                max_abs,
            )
        if context.backend_name == "numpy":
            # NumPy 通过独立 InputConfigRandomState 保持框架 RNG 隔离。
            backend = NumPyInputBackend(
                InputConfigRandomState(
                    context.seed, context.config_fingerprint, stream_kind=stream_kind
                )
            )
        else:
            stream_identity = SimpleNamespace(
                seed=context.seed,
                config_fingerprint=context.config_fingerprint,
                stream_kind=stream_kind,
            )
            if context.backend_name == "torch":
                # Torch backend 以 device 作为最终物化位置，而非逻辑策略输入。
                backend = TorchInputBackend(stream_identity, device=device)
            elif context.backend_name == "paddle":
                # Paddle backend 的随机状态由 backend 自己保存并恢复。
                backend = PaddleInputBackend(stream_identity, device=device)
            else:
                raise ValueError(f"unsupported output-grad backend: {context.backend_name!r}")
        if "int" in dtype:
            # 整数梯度沿用固定宽度范围，不受浮点 max_abs 旋钮影响。
            return backend.randint(-65535, 65535, shape=shape, dtype=dtype)
        base_dtype = (
            "float16"
            if dtype in {"float8_e5m2", "float8_e4m3fn"}
            else "float32"
            if dtype == "bfloat16"
            else dtype
        )
        if context.range_configured:
            # 显式范围同时约束复数两个分量，避免 caller 各自解释策略。
            spec = InputTensorSpec(tuple(shape), base_dtype, None, False, None)
            value = generate_symmetric_input_value(spec, max_abs, backend)
        elif dtype.startswith("complex"):
            # 默认复数保留历史两次 random 调用顺序，兼容已有基线。
            real_dtype = "float32" if dtype == "complex64" else "float64"
            value = backend.cast(
                backend.random(shape, dtype=real_dtype)
                - 0.5
                + 1j * (backend.random(shape, dtype=real_dtype) - 0.5),
                dtype,
            )
        else:
            # 默认实数路径保留 uniform 调用，避免改变历史采样顺序。
            value = backend.uniform(-0.5, 0.5, shape=shape, dtype=base_dtype)
        if base_dtype == dtype or context.backend_name == "numpy":
            # NumPy seed 暂不做框架 dtype 转换，转换由 slot 消费方负责。
            return value
        if context.backend_name == "torch":
            return value.to(dtype=to_torch_dtype(dtype))
        return backend._paddle().cast(value, dtype=dtype)

    def cached_numpy_output_grad(
        self,
        dtype,
        shape,
        stream_kind,
        seed,
        config_fingerprint,
        max_abs,
    ):
        """Return the process-local cached NumPy output gradient."""
        # output-grad cache 与 prepared handle 一起清理，避免跨 runtime 配置复用数组。
        if dtype in {"float8_e5m2", "float8_e4m3fn"}:
            dtype = "float16"
        elif dtype == "bfloat16":
            dtype = "float32"
        shape = _normalize_shape(shape, scalar_empty=False)
        # 范围参与 cache identity，避免窄范围梯度污染压力测试。
        key = (dtype, shape, stream_kind, int(seed), str(config_fingerprint), max_abs)
        if key not in self._cached_numpy_output_grads:
            # cache stream 名称保持历史派生方式，相同 seed 的默认梯度不漂移。
            rng = InputConfigRandomState(
                seed,
                config_fingerprint,
                f"cached_numpy:{stream_kind}",
            )
            if "int" in dtype:
                value = rng.cast(rng.randint(-65535, 65535, shape=shape), dtype)
            else:
                # cached complex 复用公共路径，保证实部和虚部独立采样。
                spec = InputTensorSpec(shape, dtype, None, False, None)
                value = generate_symmetric_input_value(spec, max_abs, rng)
            self._cached_numpy_output_grads[key] = value
        return self._cached_numpy_output_grads[key]


_INPUT_BACKEND_RUNTIME = InputBackendRuntime()


def prepare_input_backend(policy):
    """准备进程级 backend 模块和设备 context。"""
    return _INPUT_BACKEND_RUNTIME.prepare(policy)


def clear_input_backend_runtime():
    """清理当前进程的 backend handles 和 output-grad cache。"""
    _INPUT_BACKEND_RUNTIME.clear()


def reset_output_grad_streams():
    """Reset output-grad stream identity for a new tester case."""
    _INPUT_BACKEND_RUNTIME.reset_output_grad_streams()


def generate_output_grad_for_runtime(
    *, dtype, shape, device, runtime_config, config_fingerprint=""
):
    """Generate output grad from the frozen runtime config owned by the worker."""
    context = OutputGradContext.from_runtime_config(
        runtime_config, config_fingerprint=config_fingerprint
    )
    return _INPUT_BACKEND_RUNTIME.generate_output_grad(
        dtype=dtype,
        shape=shape,
        device=device,
        context=context,
    )


def generate_output_grad(
    *,
    dtype,
    shape,
    backend_name,
    device,
    seed,
    config_fingerprint,
    max_abs=0.5,
    range_configured=False,
    stream_index=0,
    cache_enabled=False,
):
    """Compatibility wrapper for callers that still pass expanded facts."""
    context = OutputGradContext(
        backend_name=backend_name,
        seed=int(seed),
        config_fingerprint=str(config_fingerprint),
        max_abs=float(max_abs),
        range_configured=bool(range_configured),
        cache_enabled=bool(cache_enabled),
    )
    return _INPUT_BACKEND_RUNTIME.generate_output_grad(
        dtype=dtype,
        shape=shape,
        device=device,
        context=context,
        stream_index=stream_index,
    )


__all__ = [
    "InputBackendPolicy",
    "InputBackendRuntime",
    "OutputGradContext",
    "clear_input_backend_runtime",
    "create_input_backend",
    "generate_output_grad",
    "generate_output_grad_for_runtime",
    "prepare_input_backend",
    "reset_output_grad_streams",
    "resolve_input_backend_policy",
]
