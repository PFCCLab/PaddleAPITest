"""TensorConfig 的框架物化计划与配置树资源统计。"""

from __future__ import annotations

from dataclasses import dataclass

from .tensor_config import (
    CAST_THROUGH_INTERMEDIATE_DTYPES,
    FLOAT8_DTYPES,
    TensorConfig,
    dtype_element_size,
    dtype_name,
)


@dataclass(frozen=True)
class MaterializationPlan:
    """描述单个 TensorConfig 在一个框架物化阶段的 GPU 存活量。"""

    persistent_bytes: int = 0
    peak_bytes: int = 0
    source_bytes: int = 0
    temporary_bytes: int = 0

    # peak_bytes 包含 temporary_bytes 存活期间的目标和源 storage。


def generated_value_nbytes(config):
    """返回生成 backend 为一个配置实际持有的元素存储字节数。"""
    # BF16/FP8 在生成阶段使用可写的中间 dtype，不能直接按逻辑 dtype 计量。
    # 逻辑 dtype 和生成阶段 storage dtype 可能不同，统计必须使用后者。
    name = dtype_name(config.dtype)
    generated_dtype = (
        "float32" if name == "bfloat16" else "float16" if name in FLOAT8_DTYPES else name
    )
    return max(0, config.numel()) * dtype_element_size(generated_dtype)


def _materialization_target_bytes(config):
    # 显式 CPU place 不参与 GPU 预检，即使后续框架会在主存中物化它。
    # CPU 输入不会消耗目标 GPU 显存，预检阶段直接排除。
    if config.place is not None and "cpu" in str(config.place).lower():
        return 0
    return config.nbytes(storage=True)


def _materialization_intermediate_bytes(config):
    # 非连续 Tensor 先创建 flat storage，再通过 view/as_strided 暴露逻辑布局。
    # 非连续布局先经过 flat storage，再创建逻辑 view。
    name = dtype_name(config.dtype)
    intermediate_size = dtype_element_size("float16" if name in FLOAT8_DTYPES else config.dtype)
    return config.storage_numel() * intermediate_size


def build_materialization_plan(config, input_backend, framework, *, input_source_on_gpu):
    """由实际 TensorConfig 物化规则生成 GPU 物化计划。"""
    # 计划只接受已解析的 backend/framework 名称，拒绝隐式降级。
    if input_backend not in {"numpy", "torch", "paddle"}:
        raise ValueError(f"unsupported input backend: {input_backend}")
    if framework not in {"torch", "paddle"}:
        raise ValueError(f"unsupported materialization framework: {framework}")
    if input_backend == "numpy" and input_source_on_gpu:
        raise ValueError("NumPy input source cannot reside on GPU")

    target_bytes = _materialization_target_bytes(config)
    # 零元素或 CPU place 配置无需进入 GPU 物化计划。
    if target_bytes == 0:
        return MaterializationPlan()

    source_bytes = generated_value_nbytes(config) if input_source_on_gpu else 0
    name = dtype_name(config.dtype)
    cast_required = name in CAST_THROUGH_INTERMEDIATE_DTYPES

    if config.is_contiguous:
        # 连续布局可以直接 copy 或复用 source storage。
        if input_backend == "paddle" and framework == "torch":
            transfer_bytes = source_bytes or target_bytes
            if cast_required:
                intermediate_bytes = source_bytes or generated_value_nbytes(config)
                temporary_bytes = transfer_bytes + max(
                    2 * intermediate_bytes, intermediate_bytes + target_bytes
                )
            else:
                temporary_bytes = transfer_bytes + target_bytes
            return MaterializationPlan(target_bytes, temporary_bytes, source_bytes, temporary_bytes)

        reuses_source = input_source_on_gpu and not cast_required
        persistent_bytes = 0 if reuses_source else target_bytes
        temporary_bytes = 0 if reuses_source else target_bytes
        return MaterializationPlan(persistent_bytes, temporary_bytes, source_bytes, temporary_bytes)

    flat_bytes = _materialization_intermediate_bytes(config)
    # 非连续布局需要额外 flat storage，最后再暴露 stride view。
    logical_copy_bytes = (
        max(0, config.numel())
        * dtype_element_size("float16" if name in FLOAT8_DTYPES else config.dtype)
        if not input_source_on_gpu or (input_backend == "paddle" and framework == "torch")
        else 0
    )
    final_cast_bytes = target_bytes if name in FLOAT8_DTYPES else 0
    temporary_bytes = max(flat_bytes + logical_copy_bytes, flat_bytes + final_cast_bytes)
    if input_backend == "paddle" and framework == "torch":
        temporary_bytes += source_bytes or target_bytes
    return MaterializationPlan(target_bytes, temporary_bytes, source_bytes, temporary_bytes)


def iter_unique_tensor_configs(*roots):
    """按对象身份遍历任意配置树中的 TensorConfig。"""
    # 同一个配置对象可能被多个参数引用，按 identity 只计一次。
    seen = set()

    def visit(value):
        if isinstance(value, TensorConfig):
            if id(value) not in seen:
                seen.add(id(value))
                yield value
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from visit(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from visit(item)

    for root in roots:
        yield from visit(root)


def tensor_config_tree_numel(*roots):
    """汇总配置树中唯一 TensorConfig 的逻辑元素数。"""
    # 汇总阶段不触发任何真实 Tensor 分配。
    return sum(config.numel() for config in iter_unique_tensor_configs(*roots))


def tensor_config_tree_nbytes(*roots, storage=True):
    """汇总配置树中唯一 TensorConfig 的逻辑或 storage 字节数。"""
    return sum(config.nbytes(storage=storage) for config in iter_unique_tensor_configs(*roots))


__all__ = [
    "MaterializationPlan",
    "build_materialization_plan",
    "generated_value_nbytes",
    "iter_unique_tensor_configs",
    "tensor_config_tree_nbytes",
    "tensor_config_tree_numel",
]
