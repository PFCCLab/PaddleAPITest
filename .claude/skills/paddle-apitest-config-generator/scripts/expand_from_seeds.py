#!/usr/bin/env python3
"""Expand PaddleAPITest seed configs into 4096, 1M, and 0size corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from apitest_config_utils import (
    SPECS,
    APIConfig,
    CaseRecord,
    TensorConfig,
    anchor_value,
    api_key,
    clone_config,
    config_tensors,
    ensure_zero_dimension,
    parse_config_lines,
    serialize_config,
    write_case_tree,
)


def normalized_axis(shape: list[int], axis: int) -> int:
    # 负轴沿用 Python 约定，统一转换后再访问 shape，错误在生成前暴露。
    # 轴越界说明 seed 与调用参数不匹配，不能静默改用最后一维。
    resolved = axis if axis >= 0 else len(shape) + axis
    if resolved < 0 or resolved >= len(shape):
        raise ValueError(f"anchor axis {axis} is invalid for shape {shape}")
    return resolved


def mutate_seed(
    seed,
    spec: str,
    index: int,
    tensor_type: type,
    anchor_tensor: int,
    anchor_axis: int,
) -> tuple[object, tuple[str, ...]]:
    # 每个规格都从原始 seed 的深拷贝派生，保证不同 case 之间没有共享状态。
    # anchor_tensor/axis 只决定主关联维度，其他维度尽量维持 seed 的结构。
    config = clone_config(seed)
    tensors = config_tensors(config, tensor_type)
    if not tensors:
        raise ValueError(f"seed has no Tensor arguments: {seed}")
    if anchor_tensor < 0 or anchor_tensor >= len(tensors):
        raise ValueError(
            f"anchor tensor {anchor_tensor} is invalid for {len(tensors)} Tensor arguments"
        )
    selected = tensors[anchor_tensor]
    axis = normalized_axis(selected.shape, anchor_axis)
    old_anchor = selected.shape[axis]
    if old_anchor <= 0:
        raise ValueError(f"seed anchor dimension must be positive, got {old_anchor}")

    # target 来自规格边界表，不依赖某个 API 的 hidden size 或实现常量。
    target = anchor_value(spec, index)
    mutations = [f"anchor:{old_anchor}->{target}"]
    original_shapes = [list(tensor.shape) for tensor in tensors]
    # 只替换与旧 anchor 完全相等的维度，保护比例不同的派生维度。
    for tensor, original_shape in zip(tensors, original_shapes):
        tensor.shape = [target if dim == old_anchor else dim for dim in original_shape]

    # 0size 需要显式零维，同时用非 anchor 维度乘数避免输出重复。
    if spec == "0size":
        multiplier = index + 1
        for tensor, original_shape in zip(tensors, original_shapes):
            tensor.shape = [
                dim * multiplier if original != old_anchor and dim > 0 else dim
                for dim, original in zip(tensor.shape, original_shape)
            ]
        mutations.append(f"non_anchor_multiplier:{multiplier}")
        if ensure_zero_dimension(config, tensor_type):
            mutations.append("zero_dimension_appended")
    return config, tuple(mutations)


def generate_records(args: argparse.Namespace) -> list[CaseRecord]:
    # 先解析全部 seed，再按 api 名筛选，保证请求缺失时能给出明确诊断。
    parsed = parse_config_lines(args.seed_file, APIConfig)
    requested = set(args.api or [])
    seeds_by_api: dict[str, list[tuple[Path, int, object]]] = {}
    for path, line_number, config in parsed:
        name = api_key(config)
        if requested and name not in requested:
            continue
        seeds_by_api.setdefault(name, []).append((path, line_number, config))
    missing = requested - set(seeds_by_api)
    if missing:
        raise ValueError(f"no seed configs found for APIs: {sorted(missing)}")
    if not seeds_by_api:
        raise ValueError("no matching seed configs found")

    # 输出按 API、规格、索引排序，便于 manifest 稳定复现和差异审查。
    records = []
    for name, seeds in sorted(seeds_by_api.items()):
        for spec in args.spec:
            seen: set[str] = set()
            # 循环复用 seed，但每次 mutate_seed 都会创建独立配置对象。
            for index in range(args.cases_per_spec):
                path, line_number, seed = seeds[index % len(seeds)]
                config, mutations = mutate_seed(
                    seed,
                    spec,
                    index,
                    TensorConfig,
                    args.anchor_tensor,
                    args.anchor_axis,
                )
                line = serialize_config(config, APIConfig)
                # 某些 seed 的零维变换可能仍碰撞，追加 rank marker 只作为唯一性兜底。
                if line in seen and spec == "0size":
                    first_tensor = config_tensors(config, TensorConfig)[0]
                    first_tensor.shape.append(index + 1)
                    mutations = (*mutations, "rank_marker_appended_for_uniqueness")
                    line = serialize_config(config, APIConfig)
                if line in seen:
                    raise ValueError(
                        f"cannot create unique {name}/{spec} case {index}; "
                        "write an API-aware builder"
                    )
                seen.add(line)
                records.append(
                    CaseRecord(
                        spec=spec,
                        api=name,
                        index=index,
                        category="seed_expansion",
                        violations=mutations,
                        config=config,
                        source=f"{path}:{line_number}",
                    )
                )
    return records


def parse_args() -> argparse.Namespace:
    # CLI 默认生成三类规格；显式 --spec 可缩小范围以便快速验证。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", action="append", help="API name or custom op_name")
    parser.add_argument("--seed-file", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-spec", type=int, default=512)
    parser.add_argument("--spec", action="append", choices=SPECS, default=None)
    parser.add_argument("--anchor-tensor", type=int, default=0)
    parser.add_argument("--anchor-axis", type=int, default=0)
    args = parser.parse_args()
    # argparse 的 None 与空列表语义不同，这里统一成完整默认规格集合。
    args.spec = args.spec or list(SPECS)
    if args.cases_per_spec <= 0:
        parser.error("--cases-per-spec must be positive")
    return args


def main() -> None:
    # 生成和写盘分两步，任何 case 错误都会在目录创建前终止。
    args = parse_args()
    records = generate_records(args)
    write_case_tree(args.output_dir, records)
    print(
        f"generated {len(records)} cases for {len({record.api for record in records})} "
        f"APIs under {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
