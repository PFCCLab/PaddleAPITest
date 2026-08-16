"""TensorConfig 的框架物化计划与配置树资源统计。"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import paddle
from tester.dtype_utils import to_torch_dtype

from .tensor_config import (
    CAST_THROUGH_INTERMEDIATE_DTYPES,
    FLOAT8_DTYPES,
    TensorConfig,
    dtype_element_size,
    dtype_name,
)
from .values import InputValue, attach_input_values, read_input_value


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


def materialize_config_tree(
    value,
    api_config,
    framework,
    *,
    clear_tensor=True,
    convert_dtype=False,
):
    """Materialize one config tree through the owning framework seam."""
    # framework 名称在这里解析，调用方不再复制叶子分支选择。
    if framework not in {"numpy", "torch", "paddle"}:
        raise ValueError(f"unsupported materialization framework: {framework}")
    if isinstance(value, TensorConfig):
        if framework == "numpy":
            # NumPy 输入由生成值拥有，不能伪造一个不存在的框架缓存。
            result = read_input_value(api_config, value)
            if result is None:
                # 缺值必须暴露为配置错误，避免后续 Paddle/Torch 报无关异常。
                raise value._missing_input_error(api_config, "NumPy")
            return result
        # Paddle/Torch 叶子物化和清理必须成对发生，避免缓存跨 case 泄漏。
        result = getattr(value, f"get_{framework}_tensor")(api_config)
        if clear_tensor:
            getattr(value, f"clear_{framework}_tensor")()
        return result
    if isinstance(value, list):
        # 列表容器需要保持原有参数协议，只替换 TensorConfig 叶子。
        return [
            materialize_config_tree(
                item,
                api_config,
                framework,
                clear_tensor=clear_tensor,
                convert_dtype=convert_dtype,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        # tuple 可能承载 API 的 shape 或多输出参数，不能降级成 list。
        return tuple(
            materialize_config_tree(
                item,
                api_config,
                framework,
                clear_tensor=clear_tensor,
                convert_dtype=convert_dtype,
            )
            for item in value
        )
    if isinstance(value, dict):
        # kwargs 和嵌套映射共享同一遍历，避免调用方遗漏 dict 分支。
        return type(value)(
            (
                key,
                materialize_config_tree(
                    item,
                    api_config,
                    framework,
                    clear_tensor=clear_tensor,
                    convert_dtype=convert_dtype,
                ),
            )
            for key, item in value.items()
        )
    if framework == "torch" and (
        convert_dtype or isinstance(value, (paddle.dtype, paddle.base.libpaddle.VarDesc.VarType))
    ):
        # Torch 目标的 dtype 参数必须跨 Paddle 枚举转换，TensorConfig 不处理它。
        return to_torch_dtype(value)
    return value


def clear_tensor_configs(*roots, clear_method, clear_args=()):
    """Clear unique TensorConfig framework caches and report whether any existed."""
    # merged kwargs 与原始 args 可能别名同一叶子，identity 去重是清理协议的一部分。
    configs = tuple(iter_unique_tensor_configs(*roots))
    for config in configs:
        getattr(config, clear_method)(*clear_args)
    return bool(configs)


def reset_tensor_configs(*roots):
    """Drop leaf framework caches before reusing a copied config tree."""
    # 配置副本不能复用源对象的设备 Tensor，否则两次实现会共享可变状态。
    for config in iter_unique_tensor_configs(*roots):
        # 这些字段是 TensorConfig 唯一的框架缓存，不触碰 shape/dtype 元数据。
        config.paddle_tensor = None
        config.torch_tensor = None
        config.cpu_tensor = None
        # 非连续测试的维度扰动属于一次物化状态，复制前必须重新选择。
        config.shuffle_dims = None


def copy_generated_input_values(source_api_config, target_api_config):
    """Copy logical input values and rebuild their target identity index."""
    values = getattr(source_api_config, "_input_generation_values", None)
    if values is None:
        return False
    # APIConfig 的自定义 deepcopy 不复制运行时索引，这里按稳定 path 重建它。
    attach_input_values(
        target_api_config,
        tuple(
            InputValue(item.path, copy.deepcopy(item.generated_value), item.backend_name)
            for item in values
        ),
    )
    return True


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
    "clear_tensor_configs",
    "copy_generated_input_values",
    "generated_value_nbytes",
    "iter_unique_tensor_configs",
    "materialize_config_tree",
    "reset_tensor_configs",
    "tensor_config_tree_nbytes",
    "tensor_config_tree_numel",
]
