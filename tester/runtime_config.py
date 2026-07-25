from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

GPU_MEMORY_POLICY_ENV = "PADDLEAPITEST_GPU_MEMORY_POLICY"
GPU_MEMORY_POLICIES = frozenset({"conservative", "aggressive"})
DEFAULT_GPU_MEMORY_POLICY = "conservative"


def resolve_gpu_memory_policy(value=None):
    """Resolve and validate the process-wide GPU memory policy."""
    raw_value = os.environ.get(GPU_MEMORY_POLICY_ENV, "") if value is None else value
    policy = str(raw_value or DEFAULT_GPU_MEMORY_POLICY).strip().lower()
    if policy not in GPU_MEMORY_POLICIES:
        choices = ", ".join(sorted(GPU_MEMORY_POLICIES))
        raise ValueError(
            f"invalid {GPU_MEMORY_POLICY_ENV}={raw_value!r}: expected one of {choices}"
        )
    return policy


@dataclass(frozen=True)
class GpuModeConfig:
    enabled: bool = False
    dual_gpu: bool = False
    comparison_device_id: int | None = None
    memory_policy: str = DEFAULT_GPU_MEMORY_POLICY
    workers_on_gpu: int = 1
    total_memory: float = 0.0
    memory_budget: float = 0.0
    comparison_total_memory: float = 0.0
    comparison_memory_budget: float = 0.0
    memory_fraction: float = 0.85
    cleanup_pressure_ratio: float = 0.25
    cleanup_used_ratio: float = 0.90


@dataclass(frozen=True)
class TestRuntimeConfig:
    random_seed: int = 0
    bitwise_alignment: bool = False
    exit_on_error: bool = False
    gpu_mode: GpuModeConfig = field(default_factory=GpuModeConfig)

    @classmethod
    def from_options(cls, options):
        dual_gpu = bool(getattr(options, "accuracy_stable_dual_gpu", False))
        gpu_mode = GpuModeConfig(
            enabled=bool(options.use_gpu_mode) or dual_gpu,
            dual_gpu=dual_gpu,
            comparison_device_id=1 if dual_gpu else None,
            memory_policy=resolve_gpu_memory_policy(
                getattr(options, "gpu_memory_policy", DEFAULT_GPU_MEMORY_POLICY)
            ),
        )
        return cls(
            random_seed=int(options.random_seed),
            bitwise_alignment=bool(options.bitwise_alignment),
            exit_on_error=bool(options.exit_on_error),
            gpu_mode=gpu_mode,
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


def limit_worker_layout(available_gpus, max_workers_per_gpu, pending_cases):
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
