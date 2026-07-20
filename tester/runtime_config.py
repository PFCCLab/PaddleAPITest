from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class GpuModeConfig:
    enabled: bool = False
    workers_on_gpu: int = 1
    total_memory: float = 0.0
    memory_budget: float = 0.0
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
        gpu_mode = GpuModeConfig(
            enabled=bool(options.use_gpu_mode),
        )
        return cls(
            random_seed=int(options.random_seed),
            bitwise_alignment=bool(options.bitwise_alignment),
            exit_on_error=bool(options.exit_on_error),
            gpu_mode=gpu_mode,
        )

    def for_gpu(self, gpu_id, workers_per_gpu, total_memory_per_gpu):
        workers_on_gpu = max(1, int(workers_per_gpu.get(gpu_id, self.gpu_mode.workers_on_gpu) or 1))
        total_memory = float(total_memory_per_gpu.get(gpu_id, self.gpu_mode.total_memory) or 0.0)
        memory_budget = (
            total_memory * self.gpu_mode.memory_fraction / workers_on_gpu
            if total_memory > 0
            else 0.0
        )
        gpu_mode = replace(
            self.gpu_mode,
            workers_on_gpu=workers_on_gpu,
            total_memory=total_memory,
            memory_budget=memory_budget,
        )
        return replace(self, gpu_mode=gpu_mode)


def runtime_config_for_gpu(options, gpu_id):
    runtime_config = getattr(options, "runtime_config", None)
    if runtime_config is None:
        runtime_config = TestRuntimeConfig.from_options(options)
    return runtime_config.for_gpu(
        gpu_id,
        getattr(options, "gpu_workers_per_gpu_map", {}) or {},
        getattr(options, "gpu_total_memory_map", {}) or {},
    )
