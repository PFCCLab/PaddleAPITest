from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

from .input_generation.backend import (
    INPUT_BACKEND_ENV_VAR,
    InputBackendPolicy,
    resolve_input_backend_policy,
)

NUMPY_CACHE_ENV_VAR = "USE_CACHED_NUMPY"
_TRUE_ENV_VALUES = {"true", "1", "yes", "y"}

_RUNTIME_OPERATION_MODES = (
    "accuracy_dual_gpu",
    "accuracy_stable_dual_gpu",
    "paddle_only",
    "paddle_cinn",
    "accuracy",
    "paddle_gpu_performance",
    "torch_gpu_performance",
    "paddle_torch_gpu_performance",
    "accuracy_stable",
    "paddle_custom_device",
    "custom_device_vs_gpu",
)


class GpuMemoryDeferred(RuntimeError):
    """动态物理显存不足；本次 case 可在稍后重试。"""


def numpy_cache_enabled(environ=None):
    """解析旧开关的输入生成缓存语义。"""
    env = os.environ if environ is None else environ
    value = env.get(NUMPY_CACHE_ENV_VAR, "False")
    return str(value).lower() in _TRUE_ENV_VALUES


def resolve_operation_mode(options):
    """返回参数规范化后的唯一主模式。"""
    modes = tuple(name for name in _RUNTIME_OPERATION_MODES if getattr(options, name, False))
    # dual flag 会同时展开基础模式，报告和策略优先保留更具体的 dual 身份。
    dual_modes = tuple(name for name in modes if name.endswith("_dual_gpu"))
    if dual_modes:
        return dual_modes[0]
    return modes[0] if modes else None


def _default_input_backend_policy():
    return InputBackendPolicy(requested=None, resolved="numpy", device="cpu")


def _resolve_runtime_input_backend(use_gpu_mode, operation_mode, use_cached_numpy):
    requested = os.environ.get(INPUT_BACKEND_ENV_VAR)
    validated = resolve_input_backend_policy(
        use_gpu_mode=use_gpu_mode,
        operation_mode=operation_mode,
        requested=requested,
    )
    if use_cached_numpy and not use_gpu_mode:
        # cache 固定 NumPy 真源，但不能借此掩盖非法的显式 backend 配置。
        return InputBackendPolicy(requested=validated.requested, resolved="numpy", device="cpu")
    return validated


@dataclass(frozen=True)
class GpuModeConfig:
    enabled: bool = False
    dual_gpu: bool = False
    comparison_device_id: int | None = None
    workers_on_gpu: int = 1
    total_memory: float = 0.0
    memory_budget: float = 0.0
    comparison_total_memory: float = 0.0
    comparison_memory_budget: float = 0.0
    memory_fraction: float = 1.0
    cleanup_pressure_ratio: float = 0.25
    cleanup_used_ratio: float = 0.90


@dataclass(frozen=True)
class TestRuntimeConfig:
    random_seed: int = 0
    bitwise_alignment: bool = False
    exit_on_error: bool = False
    test_cpu: bool = False
    gpu_mode: GpuModeConfig = field(default_factory=GpuModeConfig)
    operation_mode: str | None = None
    input_backend: InputBackendPolicy = field(default_factory=_default_input_backend_policy)
    # 追加在末尾，保持历史 positional 构造参数的顺序。
    use_cached_numpy: bool = False

    @property
    def operator_device_type(self):
        """返回算子执行设备；GPU mode 不参与该决策。"""
        # 设备协议刻意保持两个正交维度：
        # test_cpu 只选择 Paddle/Torch kernel，
        # gpu_mode 只选择逻辑输入、比较和显存治理，
        # 调用方不能再由其中一个字段推导另一个字段。
        return "cpu" if self.test_cpu else "cuda"

    @classmethod
    def from_options(cls, options):
        # 双卡是 worker 设备拓扑；具体结果生命周期仍由各 accuracy tester 自己管理。
        dual_gpu = bool(
            getattr(options, "accuracy_dual_gpu", False)
            or getattr(options, "accuracy_stable_dual_gpu", False)
        )
        gpu_mode = GpuModeConfig(
            enabled=bool(options.use_gpu_mode) or dual_gpu,
            dual_gpu=dual_gpu,
            comparison_device_id=1 if dual_gpu else None,
        )
        operation_mode = resolve_operation_mode(options)
        use_cached_numpy = bool(getattr(options, "use_cached_numpy", False))
        input_backend = _resolve_runtime_input_backend(
            gpu_mode.enabled,
            operation_mode,
            use_cached_numpy,
        )
        return cls(
            random_seed=int(options.random_seed),
            use_cached_numpy=use_cached_numpy,
            bitwise_alignment=bool(options.bitwise_alignment),
            exit_on_error=bool(options.exit_on_error),
            test_cpu=bool(options.test_cpu),
            gpu_mode=gpu_mode,
            operation_mode=operation_mode,
            input_backend=input_backend,
        )

    @classmethod
    def from_environment(cls, operation_mode=None, *, test_cpu=False):
        """为未经过 engine 的直接 tester 调用构造兼容运行策略。"""
        # 直接调用没有 options 快照，只读取一次环境并固化 policy。
        use_gpu_mode = os.getenv("USE_GPU_MODE", "False").lower() in {"true", "1", "yes", "y"}
        gpu_mode = GpuModeConfig(enabled=use_gpu_mode)
        use_cached_numpy = numpy_cache_enabled()
        input_backend = _resolve_runtime_input_backend(
            use_gpu_mode,
            operation_mode,
            use_cached_numpy,
        )
        return cls(
            use_cached_numpy=use_cached_numpy,
            test_cpu=bool(test_cpu),
            gpu_mode=gpu_mode,
            operation_mode=operation_mode,
            input_backend=input_backend,
        )

    def for_gpu(
        self,
        gpu_id,
        workers_per_gpu,
        total_memory_per_gpu,
        comparison_gpu_id=None,
    ):
        workers_on_gpu = max(1, int(workers_per_gpu.get(gpu_id, self.gpu_mode.workers_on_gpu) or 1))
        total_memory = float(total_memory_per_gpu.get(gpu_id, self.gpu_mode.total_memory) or 0.0)
        memory_budget = (
            total_memory * self.gpu_mode.memory_fraction / workers_on_gpu
            if total_memory > 0
            else 0.0
        )
        comparison_total_memory = (
            float(total_memory_per_gpu.get(comparison_gpu_id, 0.0) or 0.0)
            if comparison_gpu_id is not None
            else 0.0
        )
        comparison_memory_budget = (
            comparison_total_memory * self.gpu_mode.memory_fraction
            if comparison_total_memory > 0
            else 0.0
        )
        gpu_mode = replace(
            self.gpu_mode,
            workers_on_gpu=workers_on_gpu,
            total_memory=total_memory,
            memory_budget=memory_budget,
            comparison_total_memory=comparison_total_memory,
            comparison_memory_budget=comparison_memory_budget,
        )
        return replace(self, gpu_mode=gpu_mode)


def runtime_config_for_gpu(options, gpu_id, comparison_gpu_id=None):
    runtime_config = getattr(options, "runtime_config", None)
    if runtime_config is None:
        runtime_config = TestRuntimeConfig.from_options(options)
    return runtime_config.for_gpu(
        gpu_id,
        getattr(options, "gpu_workers_per_gpu_map", {}) or {},
        getattr(options, "gpu_total_memory_map", {}) or {},
        comparison_gpu_id=comparison_gpu_id,
    )


def limit_worker_layout(
    available_gpus,
    max_workers_per_gpu,
    pending_cases,
):
    """按待运行 case 数 breadth-first 裁剪每张 GPU 的 worker 数。"""
    if pending_cases <= 0:
        return [], {}
    limited = dict.fromkeys(available_gpus, 0)
    remaining = pending_cases
    while remaining > 0:
        allocated = False
        for gpu_id in available_gpus:
            if limited[gpu_id] >= max_workers_per_gpu[gpu_id]:
                continue
            limited[gpu_id] += 1
            remaining -= 1
            allocated = True
            if remaining == 0:
                break
        if not allocated:
            break
    limited = {gpu_id: workers for gpu_id, workers in limited.items() if workers}
    return list(limited), limited
