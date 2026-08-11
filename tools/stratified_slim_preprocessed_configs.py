#!/usr/bin/env python3
"""Independently slim preprocessed API configuration files by API frequency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

DEFAULT_KEEP_RATIO = 0.10
DEFAULT_SMALL_API_THRESHOLD = 10
DEFAULT_SEED = "stratified-slim-preprocessed-configs-v1"
REMOVED_API_SUFFIXES = (".empty", ".empty_like")


def _api_from_line(line: str, source: Path, line_number: int) -> str:
    # 预处理配置协议以首个左括号分隔 API 名和参数，缺失分隔符视为损坏输入。
    api, separator, _ = line.partition("(")
    if not separator or not api:
        raise ValueError(f"无法解析 API: {source}:{line_number}")
    return api


def _is_removed_api(api: str) -> bool:
    # empty 系列由固定后缀识别，避免误删名称中间恰好包含 empty 的 API。
    return api.endswith(REMOVED_API_SUFFIXES)


def _selection_key(seed: str, relative_path: Path, api: str, occurrence: int, line: str) -> bytes:
    # occurrence 区分同一 API 下的重复行，路径则隔离不同配置集合的抽样空间。
    # 返回摘要字节直接参与排序，选择结果不受字典遍历顺序影响。
    payload = f"{seed}\0{relative_path.as_posix()}\0{api}\0{occurrence}\0{line}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def _prepare_output_dir(path: Path, label: str) -> None:
    # 非空目录意味着可能存在旧结果；拒绝覆盖以维持保留集和补集的配对关系。
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} 已存在且非空: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _is_within(path: Path, parent: Path) -> bool:
    # 调用方传入的路径已 resolve，这里统一处理同目录和子目录两种冲突。
    return path == parent or parent in path.parents


def _api_list_path(preprocessed_path: Path) -> Path:
    # API 清单与预处理文件共享固定命名协议，只替换文件名后缀。
    return preprocessed_path.with_name(
        preprocessed_path.name.replace("_preprocessed.txt", "_api_extracted.txt")
    )


def _write_api_list(source: Path, destination: Path, retained_apis: set[str]) -> None:
    # 优先继承原清单顺序，保证生成结果与已有人工审阅顺序一致。
    ordered_apis = source.read_text(encoding="utf-8").splitlines()
    if len(ordered_apis) != len(set(ordered_apis)):
        raise ValueError(f"API 列表存在重复项: {source}")
    missing = retained_apis - set(ordered_apis)
    # 配置中出现但原清单遗漏的 API 仍需输出，并以排序方式保持可复现。
    ordered_apis.extend(sorted(missing))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(f"{api}\n" for api in ordered_apis if api in retained_apis),
        encoding="utf-8",
    )


def _keep_indices(
    lines: list[str],
    apis: list[str],
    relative_path: Path,
    keep_ratio: float,
    small_api_threshold: int,
    seed: str,
) -> set[int]:
    # 索引而不是文本作为分割单位，因此完全相同的重复配置也能独立抽样。
    indices_by_api: dict[str, list[int]] = defaultdict(list)
    for index, api in enumerate(apis):
        indices_by_api[api].append(index)

    kept: set[int] = set()
    for api, indices in indices_by_api.items():
        # empty 系列进入被裁减补集，不参与低频全保留规则。
        if _is_removed_api(api):
            continue
        # 低频 API 全量保留，防止小样本因比例抽样丢失参数覆盖。
        if len(indices) <= small_api_threshold:
            kept.update(indices)
            continue
        keep_count = math.ceil(len(indices) * keep_ratio)
        # 摘要排序等价于稳定随机排列，ceil 保证正比例下至少保留一个样本。
        ranked = sorted(
            (
                _selection_key(seed, relative_path, api, occurrence, lines[index]),
                index,
            )
            for occurrence, index in enumerate(indices)
        )
        kept.update(index for _, index in ranked[:keep_count])
    return kept


def run(
    *,
    input_dir: Path,
    kept_dir: Path,
    removed_dir: Path,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    small_api_threshold: int = DEFAULT_SMALL_API_THRESHOLD,
    seed: str = DEFAULT_SEED,
) -> dict[str, object]:
    """Create retained and removed config sets without mixing input config files."""
    input_dir = input_dir.resolve()
    kept_dir = kept_dir.resolve()
    removed_dir = removed_dir.resolve()
    # 输入目录不能包含任何输出，否则递归发现范围会被运行中的写入污染。
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在: {input_dir}")
    if not 0 <= keep_ratio <= 1:
        raise ValueError("--keep-ratio 必须位于 0 和 1 之间")
    if small_api_threshold < 0:
        raise ValueError("--small-api-threshold 不能为负数")
    if (
        kept_dir == removed_dir
        or _is_within(kept_dir, input_dir)
        or _is_within(removed_dir, input_dir)
    ):
        raise ValueError("输入、保留和裁减目录必须互不相同")

    _prepare_output_dir(kept_dir, "保留目录")
    _prepare_output_dir(removed_dir, "被裁减目录")
    report: dict[str, object] = {"files": {}}
    preprocessed_paths = sorted(input_dir.rglob("*_preprocessed.txt"))
    # 空输入通常表示路径或上游命名错误，显式失败比生成空报告更易诊断。
    if not preprocessed_paths:
        raise ValueError(f"未找到 *_preprocessed.txt: {input_dir}")

    for source in preprocessed_paths:
        relative_path = source.relative_to(input_dir)
        # 保留行终止符，使两个输出拼接时能够精确恢复源文件的文本边界。
        lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        apis = [_api_from_line(line, source, number) for number, line in enumerate(lines, start=1)]
        kept_indices = _keep_indices(
            lines,
            apis,
            relative_path,
            keep_ratio,
            small_api_threshold,
            seed,
        )
        kept_path = kept_dir / relative_path
        removed_path = removed_dir / relative_path
        kept_path.parent.mkdir(parents=True, exist_ok=True)
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        # 两侧均保持源文件行序，仅成员归属由 kept_indices 决定。
        kept_path.write_text(
            "".join(line for index, line in enumerate(lines) if index in kept_indices),
            encoding="utf-8",
        )
        removed_path.write_text(
            "".join(line for index, line in enumerate(lines) if index not in kept_indices),
            encoding="utf-8",
        )

        api_list = _api_list_path(source)
        # 每份预处理配置必须有配套清单，缺失时不生成语义不完整的输出。
        if not api_list.is_file():
            raise ValueError(f"缺少 API 列表文件: {api_list}")
        kept_apis = {api for index, api in enumerate(apis) if index in kept_indices}
        removed_apis = {api for index, api in enumerate(apis) if index not in kept_indices}
        # 同一 API 若部分保留、部分裁减，会同时出现在两侧清单中，这是补集行级语义。
        _write_api_list(api_list, _api_list_path(kept_path), kept_apis)
        _write_api_list(api_list, _api_list_path(removed_path), removed_apis)

        api_counts: dict[str, dict[str, int]] = {}
        # 报告按 API 汇总原始、保留和裁减计数，用于核验每层策略是否生效。
        for api in sorted(set(apis)):
            original = apis.count(api)
            kept = sum(apis[index] == api for index in kept_indices)
            api_counts[api] = {"original": original, "kept": kept, "removed": original - kept}
        report["files"][relative_path.as_posix()] = {
            "original_lines": len(lines),
            "kept_lines": len(kept_indices),
            "removed_lines": len(lines) - len(kept_indices),
            "api_counts": api_counts,
        }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按配置集与 API 分层随机裁减预处理配置")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--kept-dir", type=Path, required=True)
    parser.add_argument("--removed-dir", type=Path, required=True)
    parser.add_argument("--keep-ratio", type=float, default=DEFAULT_KEEP_RATIO)
    parser.add_argument("--small-api-threshold", type=int, default=DEFAULT_SMALL_API_THRESHOLD)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--report-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(
        input_dir=args.input_dir,
        kept_dir=args.kept_dir,
        removed_dir=args.removed_dir,
        keep_ratio=args.keep_ratio,
        small_api_threshold=args.small_api_threshold,
        seed=args.seed,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_file is not None:
        # 报告路径由调用方控制，不隐式放入任一数据集合目录。
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
