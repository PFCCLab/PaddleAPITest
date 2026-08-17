"""GPU mode 执行前的配置级显存下界估算。"""

from __future__ import annotations

from dataclasses import dataclass

from ..input_generation.input_copy_policy import requires_inplace_input_copy
from ..input_generation.materialization import (
    build_materialization_plan,
    generated_value_nbytes,
    iter_unique_tensor_configs,
)
from ..input_generation.tensor_config import (
    AUTOGRAD_DTYPES,
    FLOAT8_DTYPES,
    FORWARD_ONLY_APIS,
    dtype_element_size,
    dtype_name,
)

_GIB = 1024**3
_SUPPORTED_MODES = frozenset(
    {
        "paddle_only",
        "accuracy",
        "accuracy_dual_gpu",
        "accuracy_stable",
        "accuracy_stable_dual_gpu",
    }
)


class GpuMemoryDeferred(RuntimeError):
    """动态物理显存不足；本次 case 可在稍后重试。"""


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


def should_check_grad(api_config):
    """按配置静态信息返回 worker 是否会执行输入梯度检查。"""
    api_name = getattr(api_config, "api_name", "")
    short_name = api_name.rsplit(".", 1)[-1]
    if short_name in FORWARD_ONLY_APIS:
        return False
    # float8 autograd / NumPy grad 路径在当前 Torch 工具链中不受支持。
    if any(dtype_name(config.dtype) in FLOAT8_DTYPES for config in _tensor_configs(api_config)):
        return False
    if api_name == "paddle.assign":
        args = getattr(api_config, "args", ())
        kwargs = getattr(api_config, "kwargs", {})
        has_list_arg = bool(args) and isinstance(args[0], list)
        has_second_arg = len(args) > 1 and args[1] is not None
        has_output_kwarg = kwargs.get("output") is not None
        if has_list_arg or has_second_arg or has_output_kwarg:
            return False
    return True


def _logical_nbytes(config):
    return config.nbytes(storage=False)


def _input_generation_peak(configs):
    # writer 按配置顺序提交值；此前已提交值与当前局部临时量同时存活。
    resident_bytes = 0
    peak_bytes = 0
    for config in configs:
        logical_bytes = _logical_nbytes(config)
        numel = max(0, config.numel())
        name = dtype_name(config.dtype)
        generated_bytes = generated_value_nbytes(config)
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


def _framework_live_input_bytes(config):
    """返回框架输入最终持有的 GPU storage。"""
    if not _is_gpu_input(config):
        return 0
    return config.nbytes(storage=True)


def _framework_materialization(
    configs,
    input_backend,
    framework,
    *,
    input_source_on_gpu,
    force_clone=False,
):
    # 物化规则由 TensorConfig 统一给出；此处只累积顺序阶段的驻留量。
    resident_bytes = 0
    peak_bytes = 0
    # clone_target 独立累计，避免把复用的 native source 再算成一份框架驻留。
    clone_target_bytes = 0
    for config in configs:
        plan = build_materialization_plan(
            config,
            input_backend,
            framework,
            input_source_on_gpu=input_source_on_gpu,
        )
        peak_bytes = max(
            peak_bytes,
            resident_bytes + plan.peak_bytes,
        )
        # persistent 只表示生成 source 之外、跨越当前物化步骤继续存活的 storage。
        resident_bytes += plan.persistent_bytes
        clone_target_bytes += _framework_live_input_bytes(config)
    if force_clone:
        # 复用的生成 source 已由阶段公共组件持有；这里只增加额外驻留与新 clone。
        peak_bytes = max(peak_bytes, resident_bytes + clone_target_bytes)
        # clone 完成后旧框架 storage 释放，执行阶段只持有新 target。
        resident_bytes = clone_target_bytes
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


def _comparison_input_grad_bytes(configs, check_grad):
    """统计最终会进入比较设备的逻辑梯度，不受算子执行设备或显式 place 影响。"""
    if not check_grad:
        return 0
    # CPU kernel 梯度虽不占 compute GPU，H2D 后仍是 comparison 卡驻留项。
    return sum(
        _logical_nbytes(config) for config in configs if dtype_name(config.dtype) in AUTOGRAD_DTYPES
    )


def estimate_gpu_memory(
    api_config,
    mode,
    *,
    check_grad,
    input_backend,
    input_source_on_gpu,
    paddle_kernel_on_gpu=True,
    torch_operator_on_gpu=True,
):
    """按测试模式构造不依赖 API 名称的阶段存活集合。"""
    if mode not in _SUPPORTED_MODES:
        raise ValueError(f"unsupported GPU memory preflight mode: {mode}")
    if input_backend not in {"numpy", "torch", "paddle"}:
        raise ValueError(f"unsupported input backend: {input_backend}")
    # source 位置是显式物化事实；非法组合必须失败，不能退回 backend 名称猜测。
    if input_backend == "numpy" and input_source_on_gpu:
        raise ValueError("NumPy input source cannot reside on GPU")

    configs = _tensor_configs(api_config)
    native_gpu_generation = bool(input_source_on_gpu)
    # NumPy backend 的生成峰值属于主存，不能为了 GPU mode 而误计入设备容量。
    generation_peak = _input_generation_peak(configs) if native_gpu_generation else 0
    generated_input_bytes = (
        sum(generated_value_nbytes(config) for config in configs) if native_gpu_generation else 0
    )
    # 原地 API 的 clone 是框架输入生命周期的一部分，不能留给运行时未知项。
    force_input_copy = requires_inplace_input_copy(api_config)
    if torch_operator_on_gpu:
        (
            torch_materialization_peak,
            torch_materialized_input_bytes,
        ) = _framework_materialization(
            configs,
            input_backend,
            "torch",
            input_source_on_gpu=input_source_on_gpu,
            force_clone=force_input_copy,
        )
        torch_framework_input_bytes = sum(_framework_live_input_bytes(config) for config in configs)
        torch_input_grad_bytes = _input_grad_bytes(configs, check_grad)
    else:
        torch_materialization_peak = 0
        torch_materialized_input_bytes = 0
        torch_framework_input_bytes = 0
        torch_input_grad_bytes = 0

    if paddle_kernel_on_gpu:
        paddle_materialization_peak, _ = _framework_materialization(
            configs,
            input_backend,
            "paddle",
            input_source_on_gpu=input_source_on_gpu,
            force_clone=force_input_copy,
        )
        paddle_framework_input_bytes = sum(
            _framework_live_input_bytes(config) for config in configs
        )
        paddle_input_grad_bytes = _input_grad_bytes(configs, check_grad)
    else:
        paddle_materialization_peak = 0
        paddle_framework_input_bytes = 0
        paddle_input_grad_bytes = 0

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
    elif mode in {"accuracy", "accuracy_dual_gpu"}:
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
        snapshot_extra_peak = (
            max(
                (
                    build_materialization_plan(
                        config,
                        input_backend,
                        "torch",
                        input_source_on_gpu=input_source_on_gpu,
                    ).peak_bytes
                    for config in configs
                ),
                default=0,
            )
            if torch_operator_on_gpu
            else 0
        )
        # 两侧从同一 CPU snapshot 顺序重建，分别统计才能表达 Paddle CPU + Torch GPU。
        stable_torch_framework_peak = 0
        if torch_operator_on_gpu:
            stable_torch_framework_peak, _ = _framework_materialization(
                configs,
                "numpy",
                "torch",
                input_source_on_gpu=False,
                force_clone=force_input_copy,
            )
        stable_paddle_framework_peak = 0
        if paddle_kernel_on_gpu:
            stable_paddle_framework_peak, _ = _framework_materialization(
                configs,
                "numpy",
                "paddle",
                input_source_on_gpu=False,
                force_clone=force_input_copy,
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
        stages.extend(
            (
                MemoryStageEstimate(
                    "torch_framework_input_materialization",
                    "compute",
                    (("framework_inputs", stable_torch_framework_peak),),
                ),
                MemoryStageEstimate(
                    "paddle_framework_input_materialization",
                    "compute",
                    (("framework_inputs", stable_paddle_framework_peak),),
                ),
            )
        )

    torch_execution = (
        ("framework_inputs", torch_framework_input_bytes),
        ("input_grads", torch_input_grad_bytes),
    )
    paddle_execution = (
        ("framework_inputs", paddle_framework_input_bytes),
        ("input_grads", paddle_input_grad_bytes),
    )
    comparison_input_grad_bytes = _comparison_input_grad_bytes(configs, check_grad)
    if mode == "paddle_only":
        stages.append(MemoryStageEstimate("paddle_forward_backward", "compute", paddle_execution))
    elif mode in {"accuracy", "accuracy_dual_gpu"}:
        # Torch 结束前生成源仍供后续 Paddle 使用；Paddle 取得所有权后释放生成源。
        stages.extend(
            (
                MemoryStageEstimate(
                    "torch_forward_backward",
                    "compute",
                    (
                        ("generated_inputs", generated_input_bytes),
                        ("materialized_inputs", torch_materialized_input_bytes),
                        ("input_grads", torch_input_grad_bytes),
                    ),
                ),
                MemoryStageEstimate("paddle_forward_backward", "compute", paddle_execution),
            )
        )
        if mode == "accuracy":
            stages.append(
                MemoryStageEstimate(
                    "accuracy_compare",
                    "compute",
                    (("input_grad_operands", 2 * comparison_input_grad_bytes),),
                )
            )
        else:
            # 输出大小依赖 API，运行时 copy guard 负责实测；这里只统计可确定的双侧输入梯度。
            stages.append(
                MemoryStageEstimate(
                    "accuracy_dual_backward_compare",
                    "comparison",
                    (("input_grad_results", 2 * comparison_input_grad_bytes),),
                )
            )
    else:
        stages.extend(
            (
                MemoryStageEstimate("stable_torch_forward_backward", "compute", torch_execution),
                MemoryStageEstimate("stable_paddle_forward_backward", "compute", paddle_execution),
            )
        )
        if mode == "accuracy_stable":
            # GPU 侧保留两轮梯度；CPU 侧逐轮搬运，比较峰值只增加一轮梯度。
            stable_comparison_grad_sets = 2 + int(torch_operator_on_gpu) + int(paddle_kernel_on_gpu)
            stages.append(
                MemoryStageEstimate(
                    "stable_compare",
                    "compute",
                    (
                        (
                            "input_grad_results",
                            stable_comparison_grad_sets * comparison_input_grad_bytes,
                        ),
                    ),
                )
            )
        else:
            # 双卡可选择全驻留或分阶段流式搬运；任一完整路径可行即可运行。
            torch_execution_bytes = sum(value for _, value in torch_execution)
            paddle_execution_bytes = sum(value for _, value in paddle_execution)
            execution_bytes = max(torch_execution_bytes, paddle_execution_bytes)
            retained_input_grad_bytes = max(
                torch_input_grad_bytes,
                paddle_input_grad_bytes,
            )
            stages.extend(
                (
                    MemoryStageEstimate(
                        "dual_full_comparison_residency",
                        "comparison",
                        (("input_grad_results", 4 * comparison_input_grad_bytes),),
                        plan="full_residency",
                    ),
                    MemoryStageEstimate(
                        "dual_phased_compute_execution",
                        "compute",
                        (
                            ("framework_execution", execution_bytes),
                            ("retained_input_grads", retained_input_grad_bytes),
                        ),
                        plan="phased_residency",
                    ),
                    MemoryStageEstimate(
                        "dual_phased_comparison_stream",
                        "comparison",
                        (("input_grad_results", 3 * comparison_input_grad_bytes),),
                        plan="phased_residency",
                    ),
                )
            )
    return GpuMemoryEstimate(mode, input_backend, tuple(stages))


def decide_gpu_memory_preflight(
    api_config,
    mode,
    gpu_config,
    *,
    check_grad,
    paddle_kernel_on_gpu=True,
    torch_operator_on_gpu=True,
    input_backend,
    input_source_on_gpu,
):
    # 预检只做 admission decision，不分配测试 Tensor，容量不足由调用方记为 OOM。
    """仅当配置峰值下界明显超过设备容量时返回拒绝决策。"""
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
            input_backend=input_backend,
            input_source_on_gpu=input_source_on_gpu,
            paddle_kernel_on_gpu=paddle_kernel_on_gpu,
            torch_operator_on_gpu=torch_operator_on_gpu,
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
