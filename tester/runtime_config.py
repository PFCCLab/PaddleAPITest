from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

# 默认上界复现历史 `(random - 0.5) * 1.2` 的数值范围和随机流。
INPUT_MAX_ABS_ENV_VAR = "PADDLEAPITEST_INPUT_MAX_ABS"
DEFAULT_INPUT_MAX_ABS = 0.6
OUTPUT_GRAD_MAX_ABS_ENV_VAR = "PADDLEAPITEST_OUTPUT_GRAD_MAX_ABS"
DEFAULT_OUTPUT_GRAD_MAX_ABS = 0.5


def _resolve_positive_max_abs(env_var, default, environ=None):
    # 两个 max_abs 环境变量共享校验，但各自拥有独立的默认值和生效范围。
    source = os.environ if environ is None else environ
    raw_value = source.get(env_var, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as err:
        # 配置错误必须在 worker 启动前失败，不能静默退回默认范围。
        raise ValueError(f"{env_var} must be a finite positive number, got {raw_value!r}") from err
    if not math.isfinite(value) or value <= 0:
        # 非有限值会直接污染所有浮点输入，零和负数不具备范围语义。
        raise ValueError(f"{env_var} must be a finite positive number, got {raw_value!r}")
    return value


def resolve_input_max_abs(environ=None):
    """解析普通浮点输入的对称绝对上界。"""
    return _resolve_positive_max_abs(INPUT_MAX_ABS_ENV_VAR, DEFAULT_INPUT_MAX_ABS, environ)


def resolve_output_grad_max_abs(environ=None):
    # output-grad 不再隐式继承 forward 范围，避免两个测试旋钮相互覆盖。
    """解析 backward output-grad 的独立对称绝对上界。"""
    return _resolve_positive_max_abs(
        OUTPUT_GRAD_MAX_ABS_ENV_VAR,
        DEFAULT_OUTPUT_GRAD_MAX_ABS,
        environ,
    )


def input_max_abs_is_configured(environ=None):
    source = os.environ if environ is None else environ
    return INPUT_MAX_ABS_ENV_VAR in source


def output_grad_max_abs_is_configured(environ=None):
    source = os.environ if environ is None else environ
    return OUTPUT_GRAD_MAX_ABS_ENV_VAR in source


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
    input_max_abs: float = DEFAULT_INPUT_MAX_ABS
    # forward/default generator 的显式配置状态。
    input_max_abs_is_configured: bool = False
    output_grad_max_abs: float = DEFAULT_OUTPUT_GRAD_MAX_ABS
    # backward output-grad 的显式配置状态。
    output_grad_max_abs_is_configured: bool = False
    bitwise_alignment: bool = False
    test_cpu: bool = False
    input_backend_requested: str | None = None
    input_backend_resolved: str | None = None
    input_logical_device: str | None = None
    use_cached_numpy: bool = False
    input_use_gpu_mode: bool = False
    input_backend_policy: object | None = None
    gpu_mode: GpuModeConfig = field(default_factory=GpuModeConfig)
    # batch worker 传入的紧凑估算；单 case 入口为 None，仍执行完整预检。
    gpu_memory_estimate: object | None = None

    @property
    def paddle_kernel_device_type(self):
        """返回 Paddle 被测 kernel 的执行设备。"""
        # 该属性刻意不读取 gpu_mode，防止生成/比较策略反向改写 kernel place。
        return "cpu" if self.test_cpu else "cuda"

    @property
    def torch_operator_device_type(self):
        """Torch reference 固定使用 GPU，不受 Paddle CPU 测试开关影响。"""
        return "cuda"

    @classmethod
    def from_options(cls, options):
        from .input_generation.backend_runtime import resolve_input_backend_policy

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
        # options 通常已由 engine 规范化；这里保留解析能力供直接构造 runtime config 的入口使用。
        policy = resolve_input_backend_policy(
            requested=getattr(options, "input_backend_requested", None),
            # 双卡 accuracy 的输入也必须跟随有效 GPU 拓扑，否则默认 backend 会错误落到 CPU。
            use_gpu_mode=gpu_mode.enabled,
            use_cached_numpy=bool(getattr(options, "use_cached_numpy", False)),
            mode=next(
                (
                    name
                    for name in (
                        "accuracy",
                        "accuracy_dual_gpu",
                        "paddle_only",
                        "paddle_cinn",
                        "paddle_gpu_performance",
                        "torch_gpu_performance",
                        "paddle_torch_gpu_performance",
                        "accuracy_stable",
                        "accuracy_stable_dual_gpu",
                        "paddle_custom_device",
                        "custom_device_vs_gpu",
                    )
                    if getattr(options, name, False)
                ),
                None,
            ),
        )
        # 请求值和终态同时保存，日志可以区分默认选择与用户显式覆盖。
        return cls(
            random_seed=int(options.random_seed),
            input_max_abs=resolve_input_max_abs(),
            input_max_abs_is_configured=input_max_abs_is_configured(),
            output_grad_max_abs=resolve_output_grad_max_abs(),
            output_grad_max_abs_is_configured=output_grad_max_abs_is_configured(),
            bitwise_alignment=bool(options.bitwise_alignment),
            test_cpu=bool(options.test_cpu),
            input_backend_requested=policy.requested,
            input_backend_resolved=policy.resolved,
            input_logical_device=policy.logical_device,
            use_cached_numpy=policy.use_cached_numpy,
            input_use_gpu_mode=policy.use_gpu_mode,
            input_backend_policy=policy,
            gpu_mode=gpu_mode,
        )

    def for_gpu(
        self,
        gpu_id,
        workers_per_gpu,
        total_memory_per_gpu,
        comparison_gpu_id=None,
    ):
        # 每个 worker 只更新容量拓扑；输入 backend policy 在整次运行内保持不变。
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
    # GPU 拓扑只能派生容量字段，不负责补建或重新解析输入策略。
    if runtime_config is None:
        raise ValueError("runtime_config must be frozen before assigning a worker GPU")
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
