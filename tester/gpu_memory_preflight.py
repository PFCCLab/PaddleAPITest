"""GPU mode 执行前的配置级显存下界估算。"""

from __future__ import annotations

from dataclasses import dataclass

from .input_generation.backend import resolve_input_backend_name
from .input_generation.tensor_config import (
    AUTOGRAD_DTYPES,
    dtype_element_size,
    dtype_name,
    iter_unique_tensor_configs,
)

_GIB = 1024**3
_SUPPORTED_MODES = frozenset(
    {
        "paddle_only",
        "accuracy",
        "accuracy_stable",
        "accuracy_stable_dual_gpu",
    }
)

# 预检只统计能由 TensorConfig 和执行模式可靠确定的 GPU 存活集合。
# 输出、output grad、kernel workspace 等 API 相关项保持未知，交给运行时治理。
# 只有这个下界已经超过设备容量时才跳过，避免误删接近容量的有效配置。


@dataclass(frozen=True)
class MemoryStageEstimate:
    # plan 仅用于双卡的可选驻留路径；同一 plan 内的阶段必须全部可行。
    name: str
    device: str
    components: tuple[tuple[str, int], ...]
    plan: str | None = None

    @property
    def total_bytes(self):
        return sum(max(0, int(value)) for _, value in self.components)


@dataclass(frozen=True)
class GpuMemoryEstimate:
    mode: str
    input_backend: str
    stages: tuple[MemoryStageEstimate, ...]

    @property
    def peak_stage(self):
        return max(self.stages, key=lambda stage: stage.total_bytes)


@dataclass(frozen=True)
class GpuMemoryPreflightDecision:
    should_skip: bool
    estimate: GpuMemoryEstimate | None = None
    capacity_bytes: int = 0
    comparison_capacity_bytes: int = 0
    reason: str = ""
    rejected_stage: MemoryStageEstimate | None = None

    def message(self):
        if self.estimate is None:
            return self.reason or "GPU memory preflight unavailable"
        peak = self.rejected_stage or self.estimate.peak_stage
        capacity = (
            self.comparison_capacity_bytes if peak.device == "comparison" else self.capacity_bytes
        )
        components = ", ".join(
            f"{name}={_format_bytes(value)}" for name, value in peak.components if value
        )
        return (
            f"mode={self.estimate.mode}, stage={peak.name}, device={peak.device}, "
            f"estimated_peak={_format_bytes(peak.total_bytes)}, capacity={_format_bytes(capacity)}, "
            f"basis=tensor_config_lower_bound, input_backend={self.estimate.input_backend}, "
            f"components=[{components}]"
        )


def _format_bytes(value):
    return f"{int(value) / _GIB:.2f} GiB"


def _tensor_configs(api_config):
    return tuple(iter_unique_tensor_configs(api_config.args, api_config.kwargs))


def _logical_nbytes(config):
    return config.nbytes(storage=False)


def _generated_value_nbytes(config):
    # GPU 生成 backend 对 BF16/FP8 使用可生成的中间 dtype。
    name = dtype_name(config.dtype)
    generated_dtype = (
        "float32" if name == "bfloat16" else "float16" if name.startswith("float8") else name
    )
    return max(0, config.numel()) * dtype_element_size(generated_dtype)


def _input_generation_peak(configs):
    # writer 按配置顺序提交值；此前已提交值与当前局部临时量同时存活。
    resident_bytes = 0
    peak_bytes = 0
    for config in configs:
        logical_bytes = _logical_nbytes(config)
        numel = max(0, config.numel())
        name = dtype_name(config.dtype)
        generated_bytes = _generated_value_nbytes(config)
        if name.startswith("complex"):
            # 实部、虚部与复数结果的峰值，和 writer 的 source/clone 峰值均为两份逻辑值。
            temporary_peak = 2 * logical_bytes
        elif "int" in name:
            source_bytes = numel * 8
            temporary_peak = max(source_bytes + generated_bytes, 2 * generated_bytes)
        else:
            source_bytes = numel * 4
            temporary_peak = max(
                2 * source_bytes,
                source_bytes + generated_bytes,
                2 * generated_bytes,
            )
        peak_bytes = max(peak_bytes, resident_bytes + temporary_peak)
        resident_bytes += generated_bytes
    return max(peak_bytes, resident_bytes)


def _is_gpu_input(config):
    # place=None 跟随 worker 的计算设备；显式 CPU place 不产生 GPU 输入或梯度。
    return config.place is None or "cpu" not in str(config.place).lower()


def _requires_dtype_or_layout_materialization(config):
    # 中间 dtype 和非连续 storage 都会打破生成值与框架输入的直接所有权复用。
    name = dtype_name(config.dtype)
    return not config.is_contiguous or name in {
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
    }


def _framework_live_input_bytes(config):
    """返回框架输入最终持有的 GPU storage。"""
    if not _is_gpu_input(config):
        return 0
    return config.nbytes(storage=True)


def _reuses_generated_storage(config, input_backend, framework):
    # NumPy source 位于主存，任何 GPU 框架输入都必须拥有一份设备 storage。
    if not _is_gpu_input(config) or input_backend == "numpy":
        return False
    if input_backend == "paddle" and framework == "torch":
        # Paddle -> Torch 的 DLPack 路径显式 clone，Torch 不借用 Paddle 输入所有权。
        return False
    return not _requires_dtype_or_layout_materialization(config)


def _materialized_input_bytes(config, input_backend, framework):
    """返回 GPU 生成源之外由目标框架长期持有的 storage。"""
    if _reuses_generated_storage(config, input_backend, framework):
        return 0
    return _framework_live_input_bytes(config)


def _strided_intermediate_bytes(config):
    name = dtype_name(config.dtype)
    intermediate_size = (
        dtype_element_size("float16")
        if name.startswith("float8")
        else dtype_element_size(config.dtype)
    )
    return config.storage_numel() * intermediate_size


def _materialization_extra_peak(config, input_backend, framework):
    """返回一次框架输入物化相对既有 GPU 生成源的新增峰值。"""
    # 返回值不包含仍存活的 GPU 生成源，调用方在阶段组件中只统一加一次。
    target_bytes = _materialized_input_bytes(config, input_backend, framework)
    if not target_bytes or _reuses_generated_storage(config, input_backend, framework):
        return 0

    name = dtype_name(config.dtype)
    generated_bytes = _generated_value_nbytes(config)
    if config.is_contiguous:
        # cast-through dtype 在 NumPy/Paddle->Torch 路径先形成中间 Tensor，再形成最终 dtype。
        if name in {"bfloat16", "float8_e4m3fn", "float8_e5m2"}:
            needs_intermediate_copy = input_backend == "numpy" or (
                input_backend == "paddle" and framework == "torch"
            )
            return (generated_bytes if needs_intermediate_copy else 0) + target_bytes
        return target_bytes

    flat_intermediate_bytes = _strided_intermediate_bytes(config)
    # NumPy 或 Paddle -> Torch 需要创建逻辑 Tensor；原生/DLPack 路径仅 BF16 发生 cast。
    needs_logical_copy = input_backend == "numpy" or (
        input_backend == "paddle" and framework == "torch"
    )
    logical_copy_bytes = (
        max(0, config.numel())
        * (
            dtype_element_size("float16")
            if name.startswith("float8")
            else dtype_element_size(config.dtype)
        )
        if needs_logical_copy or name == "bfloat16"
        else 0
    )
    final_cast_bytes = target_bytes if name.startswith("float8") else 0
    return max(
        flat_intermediate_bytes + logical_copy_bytes,
        flat_intermediate_bytes + final_cast_bytes,
    )


def _framework_materialization(configs, input_backend, framework):
    # 输入按参数顺序物化；只延续此前输入的最终 storage，不延续其局部 cast 临时量。
    # resident_bytes 只统计新增所有者；复用生成 storage 的输入已由阶段公共组件持有。
    resident_bytes = 0
    peak_bytes = 0
    for config in configs:
        peak_bytes = max(
            peak_bytes,
            resident_bytes + _materialization_extra_peak(config, input_backend, framework),
        )
        resident_bytes += _materialized_input_bytes(config, input_backend, framework)
    return max(peak_bytes, resident_bytes), resident_bytes


def _input_grad_bytes(configs, check_grad):
    # 只计可微输入的同 shape 梯度；实际为 None 的梯度只会降低运行时占用。
    if not check_grad:
        return 0
    return sum(
        _logical_nbytes(config)
        for config in configs
        if _is_gpu_input(config) and dtype_name(config.dtype) in AUTOGRAD_DTYPES
    )


def estimate_gpu_memory(api_config, mode, *, check_grad, input_backend="torch"):
    """按测试模式构造不依赖 API 名称的阶段存活集合。"""
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unsupported GPU memory preflight mode: {mode}")
    if input_backend not in {"numpy", "torch", "paddle"}:
        raise ValueError(f"unsupported input backend: {input_backend}")

    configs = _tensor_configs(api_config)
    native_gpu_generation = input_backend != "numpy"
    # NumPy backend 的生成峰值属于主存，不能为了 GPU mode 而误计入设备容量。
    generation_peak = _input_generation_peak(configs) if native_gpu_generation else 0
    generated_input_bytes = (
        sum(_generated_value_nbytes(config) for config in configs) if native_gpu_generation else 0
    )
    torch_materialization_peak, torch_materialized_input_bytes = _framework_materialization(
        configs, input_backend, "torch"
    )
    paddle_materialization_peak, _ = _framework_materialization(configs, input_backend, "paddle")
    framework_input_bytes = sum(_framework_live_input_bytes(config) for config in configs)
    input_grad_bytes = _input_grad_bytes(configs, check_grad)

    stages = [
        MemoryStageEstimate(
            "input_generation",
            "compute",
            (("generation_live_set", generation_peak),),
        )
    ]

    if mode == "paddle_only":
        stages.append(
            MemoryStageEstimate(
                "framework_input_materialization",
                "compute",
                (
                    ("generated_inputs", generated_input_bytes),
                    ("materialized_inputs", paddle_materialization_peak),
                ),
            )
        )
    elif mode == "accuracy":
        # Accuracy 先执行 Torch，生成源在 Paddle 输入取得所有权后才释放。
        stages.extend(
            (
                MemoryStageEstimate(
                    "torch_input_materialization",
                    "compute",
                    (
                        ("generated_inputs", generated_input_bytes),
                        ("materialized_inputs", torch_materialization_peak),
                    ),
                ),
                MemoryStageEstimate(
                    "paddle_input_materialization",
                    "compute",
                    (
                        ("generated_inputs", generated_input_bytes),
                        ("materialized_inputs", paddle_materialization_peak),
                    ),
                ),
            )
        )
    else:
        # stable 保存 CPU snapshot 时可能先经 Torch 物化；每个输入完成拷贝后即释放其 GPU target。
        snapshot_extra_peak = max(
            (_materialization_extra_peak(config, input_backend, "torch") for config in configs),
            default=0,
        )
        stages.append(
            MemoryStageEstimate(
                "stable_input_snapshot",
                "compute",
                (
                    ("generated_inputs", generated_input_bytes),
                    ("snapshot_temporary", snapshot_extra_peak),
                ),
            )
        )
        # stable 从已落盘的 CPU 精确 dtype 副本重建，只分配最终框架输入。
        stages.append(
            MemoryStageEstimate(
                "framework_input_materialization",
                "compute",
                (("framework_inputs", framework_input_bytes),),
            )
        )

    framework_execution = (
        ("framework_inputs", framework_input_bytes),
        ("input_grads", input_grad_bytes),
    )
    if mode == "paddle_only":
        stages.append(
            MemoryStageEstimate("paddle_forward_backward", "compute", framework_execution)
        )
    elif mode == "accuracy":
        # Torch 结束前生成源仍供后续 Paddle 使用；Paddle 取得所有权后释放生成源。
        stages.extend(
            (
                MemoryStageEstimate(
                    "torch_forward_backward",
                    "compute",
                    (
                        ("generated_inputs", generated_input_bytes),
                        ("materialized_inputs", torch_materialized_input_bytes),
                        ("input_grads", input_grad_bytes),
                    ),
                ),
                MemoryStageEstimate("paddle_forward_backward", "compute", framework_execution),
                MemoryStageEstimate(
                    "accuracy_compare",
                    "compute",
                    (("input_grad_operand", input_grad_bytes),),
                ),
            )
        )
    else:
        stages.append(
            MemoryStageEstimate("stable_forward_backward", "compute", framework_execution)
        )
        if mode == "accuracy_stable":
            stages.append(
                MemoryStageEstimate(
                    "stable_compare",
                    "compute",
                    (("input_grad_operand", input_grad_bytes),),
                )
            )
        else:
            # 双卡可选择全驻留或分阶段流式搬运；任一完整路径可行即可运行。
            execution_bytes = sum(value for _, value in framework_execution)
            stages.extend(
                (
                    MemoryStageEstimate(
                        "dual_full_comparison_residency",
                        "comparison",
                        (("input_grad_results", 4 * input_grad_bytes),),
                        plan="full_residency",
                    ),
                    MemoryStageEstimate(
                        "dual_phased_compute_execution",
                        "compute",
                        (
                            ("framework_execution", execution_bytes),
                            ("retained_input_grads", input_grad_bytes),
                        ),
                        plan="phased_residency",
                    ),
                    MemoryStageEstimate(
                        "dual_phased_comparison_stream",
                        "comparison",
                        (("input_grad_results", 3 * input_grad_bytes),),
                        plan="phased_residency",
                    ),
                )
            )
    return GpuMemoryEstimate(mode, input_backend, tuple(stages))


def decide_gpu_memory_preflight(api_config, mode, gpu_config, *, check_grad):
    """仅当配置峰值下界明显超过设备容量时返回 skip。"""
    if not gpu_config.enabled:
        return GpuMemoryPreflightDecision(False, reason="GPU mode disabled")
    capacity_bytes = max(0, int(float(gpu_config.memory_budget or 0.0) * _GIB))
    comparison_capacity_bytes = max(
        0,
        int(float(gpu_config.comparison_memory_budget or 0.0) * _GIB),
    )
    if capacity_bytes == 0:
        return GpuMemoryPreflightDecision(False, reason="GPU capacity unavailable")
    try:
        estimate = estimate_gpu_memory(
            api_config,
            mode,
            check_grad=check_grad,
            input_backend=resolve_input_backend_name(),
        )
    except (TypeError, ValueError, OverflowError) as err:
        # 配置合法性仍由原测试流程判断；预检失败不能改变原分类。
        return GpuMemoryPreflightDecision(False, reason=f"GPU memory preflight unavailable: {err}")

    def capacity_for(stage):
        return comparison_capacity_bytes if stage.device == "comparison" else capacity_bytes

    for stage in (stage for stage in estimate.stages if stage.plan is None):
        stage_capacity = capacity_for(stage)
        if stage_capacity > 0 and stage.total_bytes > stage_capacity:
            return GpuMemoryPreflightDecision(
                True,
                estimate,
                capacity_bytes,
                comparison_capacity_bytes,
                reason="estimated live set exceeds device capacity",
                rejected_stage=stage,
            )

    plans = tuple(dict.fromkeys(stage.plan for stage in estimate.stages if stage.plan is not None))
    failed_plan_stages = []
    for plan in plans:
        over_capacity = tuple(
            stage
            for stage in estimate.stages
            if stage.plan == plan
            and capacity_for(stage) > 0
            and stage.total_bytes > capacity_for(stage)
        )
        if not over_capacity:
            return GpuMemoryPreflightDecision(
                False,
                estimate,
                capacity_bytes,
                comparison_capacity_bytes,
            )
        failed_plan_stages.extend(over_capacity)
    if failed_plan_stages:
        rejected_stage = max(
            failed_plan_stages,
            key=lambda stage: stage.total_bytes / max(1, capacity_for(stage)),
        )
        return GpuMemoryPreflightDecision(
            True,
            estimate,
            capacity_bytes,
            comparison_capacity_bytes,
            reason="all GPU residency plans exceed device capacity",
            rejected_stage=rejected_stage,
        )
    return GpuMemoryPreflightDecision(
        False,
        estimate,
        capacity_bytes,
        comparison_capacity_bytes,
    )
