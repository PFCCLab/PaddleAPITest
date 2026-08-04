"""GPU mode 执行前的配置级显存下界估算。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .input_generation.tensor_config import (
    AUTOGRAD_DTYPES,
    TensorConfig,
    dtype_element_size,
    dtype_name,
    iter_unique_tensor_configs,
    shape_numel,
)

_GIB = 1024**3
_COMPARISON_WORKSPACE_BYTES = 256 << 20

# 准入模型只回答“该配置是否明显不可能在目标设备上运行”。
# 它不尝试复刻框架 allocator、kernel workspace 或算子内部优化路径。
# 已知项按阶段存活集合计数，不能把先后释放的对象简单累加。
# 未知项保持为零下界，不能用拍脑袋倍率把临界 1M case 拒之门外。
# 输出 shape 规则是下界增强；规则失败不能取消已经可靠得到的输入下界。
# 单卡容量来自 worker 的设备预算，单 worker 时等于 NVML 报告的整卡容量。
# 多 worker 共卡时预算在 runtime_config 中先均分，本模块不重复调整。
# 双卡容量分别判断，任何地方都不能把计算卡和比较卡字节相加。
# stable 的 spill 和 dual 的 phased residency 是真实可执行路径，必须纳入准入。
# 当前空闲显存不参与永久分类，瞬时压力继续交给动态 cache/spill 治理。
# 预检异常采用 fail-open，原测试流程继续负责配置合法性和真实 OOM 分类。

# alias 规则表示结果复用已有 storage；resident_bytes 仅描述结果搬运规模。
# 原地 API 也归入 alias，避免把返回的输入所有者误算成一份新输出。
# 不能确定高级索引是否复制时，宁可保留较小下界，也不做错误上界假设。
_ALIAS_APIS = frozenset(
    {
        "paddle.Tensor.__getitem__",
        "paddle.Tensor.detach",
        "paddle.Tensor.expand",
        "paddle.Tensor.flatten",
        "paddle.Tensor.reshape",
        "paddle.Tensor.transpose",
        "paddle.Tensor.view",
        "paddle.Tensor.zero_",
        "paddle._C_ops.multiply_",
        "paddle._C_ops.subtract_",
        "paddle.reshape",
        "paddle.view",
        "paddle.flatten",
        "paddle.detach",
    }
)
# same-shape 规则只收录输出 shape 和 dtype 能从配置直接确定的 API。
# dtype 参数位置并不统一，规则分支必须显式区分 cast 类与普通同形算子。
_SAME_SHAPE_APIS = frozenset(
    {
        "paddle.Tensor.__mul__",
        "paddle.Tensor.astype",
        "paddle.Tensor.cast",
        "paddle.Tensor.square",
        "paddle.cast",
        "paddle.empty_like",
        "paddle.lerp",
        "paddle.square",
    }
)
# elementwise 规则要求两个 Tensor shape 可广播；不合法 shape 会回退 unknown。
_ELEMENTWISE_APIS = frozenset(
    {
        "paddle.multiply",
        "paddle.subtract",
    }
)


@dataclass(frozen=True)
class OutputMemoryEstimate:
    # allocated_bytes 用于执行阶段，resident_bytes 用于跨框架结果保留或搬运。
    # 两者分开才能表达 view/原地结果“有逻辑大小但没有新增 storage”的语义。
    allocated_bytes: int = 0
    resident_bytes: int = 0
    confidence: str = "lower_bound"
    rule: str = "unknown"


@dataclass(frozen=True)
class MemoryStageEstimate:
    # device 是容量域，不是物理编号；目前只有 compute 和 comparison。
    # plan 为空表示所有路径共有，非空表示某个可选 residency 方案的阶段。
    name: str
    device: str
    components: tuple[tuple[str, int], ...]
    plan: str | None = None

    @property
    def total_bytes(self):
        return sum(max(0, int(value)) for _, value in self.components)


@dataclass(frozen=True)
class GpuMemoryEstimate:
    # peak_stage 只用于展示全局最大值；跨设备拒绝必须使用 rejected_stage。
    mode: str
    output: OutputMemoryEstimate
    stages: tuple[MemoryStageEstimate, ...]

    @property
    def peak_stage(self):
        return max(self.stages, key=lambda stage: stage.total_bytes)


@dataclass(frozen=True)
class GpuMemoryPreflightDecision:
    # reason 描述判定类别，message() 负责生成可写入日志和 dump 的稳定细节。
    should_skip: bool
    estimate: GpuMemoryEstimate | None = None
    capacity_bytes: int = 0
    comparison_capacity_bytes: int = 0
    reason: str = ""
    rejected_stage: MemoryStageEstimate | None = None

    def message(self):
        if self.estimate is None:
            return self.reason or "GPU memory preflight unavailable"
        # 双卡的全驻留与分阶段方案可能在不同设备失败，诊断必须指向
        # 实际导致所有方案不可行的阶段，而不是全局字节数最大的阶段。
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
            f"confidence={self.estimate.output.confidence}, rule={self.estimate.output.rule}, "
            f"components=[{components}]"
        )


def _format_bytes(value):
    return f"{int(value) / _GIB:.2f} GiB"


def _tensor_configs(api_config):
    return tuple(iter_unique_tensor_configs(api_config.args, api_config.kwargs))


def _logical_nbytes(config):
    return config.nbytes(storage=False)


def _generated_value_nbytes(config):
    # GPU mode 默认生成 backend 对 BF16/FP8 使用可生成的中间 dtype。
    # 这里统计 writer 提交后仍驻留的值，不是框架最终参数 dtype。
    name = dtype_name(config.dtype)
    generated_dtype = (
        "float32" if name == "bfloat16" else "float16" if name.startswith("float8") else name
    )
    return max(0, config.numel()) * dtype_element_size(generated_dtype)


def _shape_nbytes(shape, dtype):
    return max(0, shape_numel(shape)) * dtype_element_size(dtype)


def _kwarg_or_arg(api_config, name, index, default=None):
    return api_config.kwargs.get(
        name, api_config.args[index] if len(api_config.args) > index else default
    )


def _normalize_axis(axis, rank, *, insertion=False):
    upper = rank + 1 if insertion else rank
    axis = int(axis)
    if axis < 0:
        axis += upper
    if axis < 0 or axis >= upper:
        raise ValueError(f"axis {axis} is outside rank {rank}")
    return axis


def _broadcast_shape(left, right):
    # 广播只做元数据运算，不创建 Python 大列表或任何框架 Tensor。
    result = []
    for left_dim, right_dim in zip(reversed(left), reversed(right), strict=False):
        if left_dim == 1:
            result.append(right_dim)
        elif right_dim == 1 or left_dim == right_dim:
            result.append(left_dim)
        else:
            raise ValueError(f"cannot broadcast shapes {left!r} and {right!r}")
    longer = left if len(left) > len(right) else right
    result.extend(reversed(longer[: abs(len(left) - len(right))]))
    return tuple(reversed(result))


def _matmul_shape(left, right, *, transpose_left=False, transpose_right=False):
    # 先处理向量特例，再对 batch 维执行标准广播。
    # K 维一致性由真实 API 校验；预检只需要输出分配下界。
    left = list(left)
    right = list(right)
    if transpose_left and len(left) >= 2:
        left[-2], left[-1] = left[-1], left[-2]
    if transpose_right and len(right) >= 2:
        right[-2], right[-1] = right[-1], right[-2]
    if len(left) == 1 and len(right) == 1:
        return ()
    if len(left) == 1:
        return (*right[:-2], right[-1])
    if len(right) == 1:
        return (*left[:-2], left[-2])
    return (*_broadcast_shape(tuple(left[:-2]), tuple(right[:-2])), left[-2], right[-1])


def _output_from_shape(shape, dtype, rule):
    output_bytes = _shape_nbytes(shape, dtype)
    return OutputMemoryEstimate(output_bytes, output_bytes, "estimated", rule)


def _estimate_creation_output(api_config):
    # 创建类 API 没有 TensorConfig 输入，若漏掉会完全失去超大 shape 的保护。
    # C-ops 的 dtype 位置与公开 API 不同，因此保持少量显式 schema 映射。
    api_name = api_config.api_name
    if api_name in {"paddle.zeros", "paddle.ones", "paddle.empty", "paddle.rand", "paddle.randn"}:
        shape = _kwarg_or_arg(api_config, "shape", 0)
        if shape is not None:
            dtype = _kwarg_or_arg(api_config, "dtype", 1, "float32")
            return _output_from_shape(shape, dtype, "creation")
    if api_name == "paddle.full":
        shape = _kwarg_or_arg(api_config, "shape", 0)
        if shape is not None:
            dtype = _kwarg_or_arg(api_config, "dtype", 2, "float32")
            return _output_from_shape(shape, dtype, "creation")
    if api_name == "paddle._C_ops.gaussian" and api_config.args:
        return _output_from_shape(api_config.args[0], api_config.args[4], "creation")
    if api_name == "paddle._C_ops.uniform" and api_config.args:
        return _output_from_shape(api_config.args[0], api_config.args[1], "creation")
    if api_name == "paddle.arange":
        # arange 的一参数形式把该参数解释为 end；整数参数默认生成 int64。
        # step=0 等非法配置抛出后由上层回退，不在预检中改变错误分类。
        start = _kwarg_or_arg(api_config, "start", 0, 0)
        end = api_config.kwargs.get("end")
        if end is None:
            if len(api_config.args) >= 2:
                end = api_config.args[1]
            else:
                end, start = start, 0
        step = _kwarg_or_arg(api_config, "step", 2, 1)
        length = max(0, math.ceil((end - start) / step))
        default_dtype = (
            "float32" if any(isinstance(value, float) for value in (start, end, step)) else "int64"
        )
        dtype = _kwarg_or_arg(api_config, "dtype", 3, default_dtype)
        return _output_from_shape((length,), dtype, "arange")
    return None


def estimate_output_memory(api_config):
    """估算可由配置静态确定的输出分配和结果驻留字节。"""
    # 规则顺序体现所有权：creation 无输入，alias 不分配，随后才是新输出规则。
    # `_tensor_configs` 按对象身份去重，与公共配置树字节统计保持同一口径。
    tensors = _tensor_configs(api_config)
    first = tensors[0] if tensors else None
    api_name = api_config.api_name

    creation = _estimate_creation_output(api_config)
    if creation is not None:
        return creation
    if api_name in _ALIAS_APIS and first is not None:
        return OutputMemoryEstimate(0, _logical_nbytes(first), "estimated", "alias")
    if api_name == "paddle._C_ops.full_" and first is not None:
        return OutputMemoryEstimate(0, _logical_nbytes(first), "estimated", "inplace")
    if api_name == "paddle.assign" and first is not None:
        output = _kwarg_or_arg(api_config, "output", 1)
        if output is not None:
            # 显式 output 已作为输入 storage 统计，assign 只返回该预分配所有者。
            output_config = next(iter_unique_tensor_configs(output), None)
            resident_bytes = _logical_nbytes(output_config or first)
            return OutputMemoryEstimate(0, resident_bytes, "estimated", "preallocated")
        return _output_from_shape(first.shape, first.dtype, "same_shape")
    if api_name in _SAME_SHAPE_APIS and first is not None:
        output_dtype = (
            _kwarg_or_arg(api_config, "dtype", 1, first.dtype)
            if api_name
            in {"paddle.Tensor.astype", "paddle.Tensor.cast", "paddle.cast", "paddle.empty_like"}
            else first.dtype
        )
        return _output_from_shape(first.shape, output_dtype, "same_shape")
    if api_name in _ELEMENTWISE_APIS and len(tensors) >= 2:
        shape = _broadcast_shape(tuple(tensors[0].shape), tuple(tensors[1].shape))
        return _output_from_shape(shape, tensors[0].dtype, "broadcast")
    if api_name in {"paddle.matmul", "paddle.Tensor.matmul"} and len(tensors) >= 2:
        shape = _matmul_shape(
            tensors[0].shape,
            tensors[1].shape,
            transpose_left=bool(api_config.kwargs.get("transpose_x", False)),
            transpose_right=bool(api_config.kwargs.get("transpose_y", False)),
        )
        return _output_from_shape(shape, tensors[0].dtype, "matmul")
    if api_name == "paddle.baddbmm" and len(tensors) >= 3:
        shape = _matmul_shape(tensors[1].shape, tensors[2].shape)
        return _output_from_shape(shape, tensors[0].dtype, "baddbmm")
    if (
        api_name in {"paddle.nn.functional.linear", "paddle.compat.nn.functional.linear"}
        and len(tensors) >= 2
    ):
        shape = (*tensors[0].shape[:-1], tensors[1].shape[0])
        return _output_from_shape(shape, tensors[0].dtype, "linear")
    if api_name in {"paddle.concat", "paddle.stack", "paddle._C_ops.concat"} and api_config.args:
        sequence = api_config.args[0]
        sequence_tensors = tuple(iter_unique_tensor_configs(sequence))
        if sequence_tensors:
            shape = list(sequence_tensors[0].shape)
            axis = _kwarg_or_arg(api_config, "axis", 1, 0)
            if api_name in {"paddle.concat", "paddle._C_ops.concat"}:
                axis = _normalize_axis(axis, len(shape))
                shape[axis] = sum(int(tensor.shape[axis]) for tensor in sequence_tensors)
                rule = "concat"
            else:
                axis = _normalize_axis(axis, len(shape), insertion=True)
                shape.insert(axis, len(sequence_tensors))
                rule = "stack"
            return _output_from_shape(shape, sequence_tensors[0].dtype, rule)
    if api_name in {"paddle.split", "paddle.Tensor.split", "paddle.unbind"} and first is not None:
        # 多个输出的逻辑元素总和等于输入，运行时会分别分配但无需重复扩大。
        return OutputMemoryEstimate(
            _logical_nbytes(first),
            _logical_nbytes(first),
            "estimated",
            "partition",
        )
    return OutputMemoryEstimate()


def _input_generation_peak(configs):
    # writer 按配置顺序提交生成值；尚未生成的后续输入不属于当前存活集合。
    # resident_bytes 是此前已经提交且仍由 APIConfig 持有的生成值。
    # temporary_peak 是当前规则表达式、dtype cast 和 writer clone 的局部峰值。
    # 浮点随机表达式最多同时持有两份 source dtype 数组。
    # 整数 cast 只有 int64 source 与目标同时存在，不额外虚构第二份 int64。
    # 复数生成会同时持有实部、虚部和复数结果，按三份逻辑复数大小估算。
    # 当前值提交后局部临时量释放，下一轮只延续 generated_bytes。
    resident_bytes = 0
    peak_bytes = 0
    for config in configs:
        logical_bytes = _logical_nbytes(config)
        numel = max(0, config.numel())
        name = dtype_name(config.dtype)
        generated_bytes = _generated_value_nbytes(config)
        if name.startswith("complex"):
            temporary_peak = 3 * logical_bytes
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
        # 已提交值继续存活；当前规则的 source/cast/writer copy 在本轮结束后释放。
        peak_bytes = max(peak_bytes, resident_bytes + temporary_peak)
        resident_bytes += generated_bytes
    return max(peak_bytes, resident_bytes)


def _framework_target_extra_bytes(config, *, materialization_peak):
    """返回生成值之外框架物化需要新增的 GPU storage。"""
    # 显式 CPU 参数不会在框架物化阶段新增 GPU target，但生成阶段仍可能用 GPU。
    # 连续且无需 cast 的参数可由 DLPack/原生 backend 复用生成 storage。
    # 非连续参数必须分配跨度 storage，不能只按逻辑 numel 估算。
    # FP8 构造时先写 float16 flat storage，再 cast 到最终一字节 storage。
    # materialization_peak=True 只影响这个短暂阶段，不能延续到前反向。
    if config.place is not None and "cpu" in str(config.place).lower():
        return 0
    name = dtype_name(config.dtype)
    needs_separate_target = not config.is_contiguous or name in {
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
    }
    if not needs_separate_target:
        return 0
    target_size = dtype_element_size(config.dtype)
    if materialization_peak and name.startswith("float8"):
        target_size = dtype_element_size("float16")
    return config.storage_numel() * target_size


def _framework_live_input_bytes(config):
    """返回生成源释放后由框架参数实际持有的 GPU 字节。"""
    # separate target 已取得数据所有权后，原始生成值可以安全释放。
    # 普通连续参数本身就是生成 storage，释放 APIConfig 引用不会释放框架参数。
    if config.place is not None and "cpu" in str(config.place).lower():
        return 0
    name = dtype_name(config.dtype)
    if not config.is_contiguous or name in {
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
    }:
        return config.nbytes(storage=True)
    return _generated_value_nbytes(config)


def _is_mutating_api(api_name):
    # 双下划线魔术方法不是 Paddle 的尾下划线原地命名约定。
    return (api_name.endswith("_") and not api_name.endswith("__")) or api_name == (
        "paddle.Tensor.__setitem__"
    )


def _input_grad_bytes(configs, check_grad):
    # 只统计框架支持 autograd 的配置 dtype；None grad 属于更小的运行时结果。
    if not check_grad:
        return 0
    return sum(
        _logical_nbytes(config) for config in configs if dtype_name(config.dtype) in AUTOGRAD_DTYPES
    )


def estimate_gpu_memory(api_config, mode, *, check_grad):
    """按模式构造阶段存活集合并返回峰值下界。"""
    configs = _tensor_configs(api_config)
    generation_peak = _input_generation_peak(configs)
    generated_input_bytes = sum(_generated_value_nbytes(config) for config in configs)
    materialized_input_bytes = sum(
        _framework_target_extra_bytes(config, materialization_peak=False) for config in configs
    )
    materialization_peak_bytes = sum(
        _framework_target_extra_bytes(config, materialization_peak=True) for config in configs
    )
    inplace_copy_bytes = (
        sum(
            _logical_nbytes(config)
            for config in configs
            if config.place is None or "cpu" not in str(config.place).lower()
        )
        if _is_mutating_api(api_config.api_name)
        else 0
    )
    framework_input_bytes = (
        inplace_copy_bytes
        if inplace_copy_bytes
        else sum(_framework_live_input_bytes(config) for config in configs)
    )
    try:
        output = estimate_output_memory(api_config)
    except (IndexError, TypeError, ValueError, ZeroDivisionError):
        # 输出规则只是一项增强；规则无法处理的合法或非法配置仍保留输入下界。
        output = OutputMemoryEstimate(rule="unknown_rule_fallback")
    input_grad_bytes = _input_grad_bytes(configs, check_grad)
    # output grad seed 在反向前与输出同时存在；input grad 是反向产物下界。
    output_grad_bytes = output.resident_bytes if check_grad else 0
    if mode == "accuracy":
        # Torch 执行时生成源还要供后续 Paddle 物化，因此不能提前释放。
        # 原地输入执行时只保留复制后的参数，物化用的原 target 已经释放。
        execution_input_components = (
            ("generated_inputs", generated_input_bytes),
            (
                "inplace_input_copies" if inplace_copy_bytes else "materialized_inputs",
                inplace_copy_bytes or materialized_input_bytes,
            ),
        )
    else:
        # paddle_only 和 stable 都在执行前建立了后续所有权，可使用释放后集合。
        execution_input_components = (("framework_inputs", framework_input_bytes),)
    execution_components = (
        *execution_input_components,
        ("outputs", output.allocated_bytes),
        ("output_grads", output_grad_bytes),
        ("input_grads", input_grad_bytes),
    )
    execution_bytes = sum(value for _, value in execution_components)
    retained_bytes = output.resident_bytes + input_grad_bytes
    stages = [
        MemoryStageEstimate(
            "input_generation",
            "compute",
            (("generation_live_set", generation_peak),),
        ),
        MemoryStageEstimate(
            "framework_input_materialization",
            "compute",
            (
                ("generated_inputs", generated_input_bytes),
                ("materialized_inputs", materialization_peak_bytes),
                ("inplace_input_copies", inplace_copy_bytes),
            ),
        ),
        MemoryStageEstimate("framework_forward_backward", "compute", execution_components),
    ]

    if mode == "accuracy":
        # reference 可按动态治理 spill 到 CPU，GPU 比较只要求当前 operand 和 workspace。
        compare_bytes = max(output.resident_bytes, input_grad_bytes)
        stages.append(
            MemoryStageEstimate(
                "accuracy_compare",
                "compute",
                (("gpu_operand", compare_bytes), ("workspace", _COMPARISON_WORKSPACE_BYTES)),
            )
        )
    elif mode == "accuracy_stable":
        # 第一轮结果可按现有治理策略 spill；第二轮执行不与它强制重叠。
        # forward 与 backward 结果族顺序比较，因此取两者较大值而不是相加。
        stages.append(
            MemoryStageEstimate(
                "stable_compare",
                "compute",
                (
                    ("gpu_operand", max(output.resident_bytes, input_grad_bytes)),
                    ("workspace", _COMPARISON_WORKSPACE_BYTES),
                ),
            )
        )
    elif mode == "accuracy_stable_dual_gpu":
        # full_residency 在比较卡保留两轮、两个框架的完整结果族。
        # phased_residency 只在比较卡保留第一轮，并逐个搬运第二轮结果族。
        # phased 的代价是第二轮 Torch 结果与 Paddle 执行在计算卡发生重叠。
        # 两个方案是“或”关系，不能把它们的阶段当成同时必须满足。
        stages.extend(
            (
                MemoryStageEstimate(
                    "dual_full_comparison_residency",
                    "comparison",
                    (
                        ("retained_results", 4 * retained_bytes),
                        ("workspace", _COMPARISON_WORKSPACE_BYTES),
                    ),
                    plan="full_residency",
                ),
                MemoryStageEstimate(
                    "dual_phased_compute_execution",
                    "compute",
                    (("execution", execution_bytes), ("retained_reference", retained_bytes)),
                    plan="phased_residency",
                ),
                MemoryStageEstimate(
                    "dual_phased_comparison_stream",
                    "comparison",
                    (
                        ("retained_first_pair", 2 * retained_bytes),
                        ("streamed_result", max(output.resident_bytes, input_grad_bytes)),
                        ("workspace", _COMPARISON_WORKSPACE_BYTES),
                    ),
                    plan="phased_residency",
                ),
            )
        )
    return GpuMemoryEstimate(mode, output, tuple(stages))


def decide_gpu_memory_preflight(api_config, mode, gpu_config, *, check_grad):
    """仅当配置峰值下界明显超过设备容量时返回 skip。"""
    # 非 GPU mode 和容量探测失败都 fail-open，保持原 case 分类不变。
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
        estimate = estimate_gpu_memory(api_config, mode, check_grad=check_grad)
    except (TypeError, ValueError, OverflowError) as err:
        # 配置合法性仍由原测试流程判断；预检失败不能改变原分类。
        return GpuMemoryPreflightDecision(False, reason=f"GPU memory preflight unavailable: {err}")

    def capacity_for(stage):
        return comparison_capacity_bytes if stage.device == "comparison" else capacity_bytes

    common_stages = tuple(stage for stage in estimate.stages if stage.plan is None)
    # 公共阶段属于所有执行方案，任一明显超限即可直接拒绝。
    for stage in common_stages:
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

    plans = {stage.plan for stage in estimate.stages if stage.plan is not None}
    failed_plan_stages = []
    # 可选方案只要有一个完整可行就放行；所有方案失败才产生 memory_skip。
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
        # 日志选择相对容量超限最明显的阶段，避免大卡字节数掩盖小卡失败。
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
