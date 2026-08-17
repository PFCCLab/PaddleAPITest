#!/usr/bin/env python3
"""Randomly retain selected API configurations and write the complementary set."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TARGET_APIS = (
    "fuse_weighted_swiglu_fp8_quant",
    "paddlefleet_fused_swiglu_probs_bwd",
    "fp8_quant_blockwise",
    "fused_act_dequant",
    "moe_permute",
    "moe_unpermute",
)
DEFAULT_KEEP_RATIO = 0.10
DEFAULT_SEED = "random-slim-api-configs-v1"
ALWAYS_REMOVED_APIS = ("paddle.empty_like", "paddle.empty")
# 这些 API 不受 keep_ratio 影响，保证集成入口的固定清理协议。
_DIGEST_MODULUS = 1 << 128


@dataclass
class FileStats:
    original_lines: int = 0
    kept_lines: int = 0
    removed_lines: int = 0
    original_sum: int = 0
    kept_sum: int = 0
    removed_sum: int = 0
    original_xor: int = 0
    kept_xor: int = 0
    removed_xor: int = 0
    api_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _line_digest(raw_line: str) -> int:
    # 摘要保留换行符，因此末行是否带换行也属于分割完整性的一部分。
    return int.from_bytes(hashlib.blake2b(raw_line.encode("utf-8"), digest_size=16).digest(), "big")


def _record_digest(stats: FileStats, group: str, raw_line: str) -> None:
    # 模加和与异或都支持分组后合并，可在不保存全部输入行的前提下校验补集。
    # 这里不使用集合去重，因为重复配置行同样必须按出现次数完整保留。
    digest = _line_digest(raw_line)
    if group == "original":
        stats.original_lines += 1
        stats.original_sum = (stats.original_sum + digest) % _DIGEST_MODULUS
        stats.original_xor ^= digest
    elif group == "kept":
        stats.kept_lines += 1
        stats.kept_sum = (stats.kept_sum + digest) % _DIGEST_MODULUS
        stats.kept_xor ^= digest
    else:
        stats.removed_lines += 1
        stats.removed_sum = (stats.removed_sum + digest) % _DIGEST_MODULUS
        stats.removed_xor ^= digest


def _matched_api(line: str, target_apis: Iterable[str]) -> str | None:
    # 目标列表的顺序也是重叠名称的优先级，报告只把一行归入一个 API。
    for api in target_apis:
        if api in line:
            return api
    return None


def _always_removed_api(line: str) -> str | None:
    # empty 配置不参与随机比例，始终进入裁减集，避免低比例时残留。
    api_name, separator, _ = line.partition("(")
    if not separator:
        return None
    for api in ALWAYS_REMOVED_APIS:
        if api_name.strip() == api:
            return api
    return None


def _should_keep(
    seed: str, relative_path: Path, line_number: int, raw_line: str, ratio: float
) -> bool:
    # 边界比例直接返回，避免浮点比较给 0% 或 100% 留下意外样本。
    if ratio == 0:
        return False
    if ratio == 1:
        return True
    # 路径和行号用于区分内容相同但来源不同的配置，同时保证相同输入可复现。
    # 固定宽度摘要映射到 [0, 1)，使保留比例不依赖 Python 随机数实现。
    payload = f"{seed}\0{relative_path.as_posix()}\0{line_number}\0{raw_line}".encode()
    score = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return score / (1 << 64) < ratio


def _prepare_output_dir(path: Path, label: str) -> None:
    # 拒绝覆盖非空目录，避免新结果与上一次运行残留文件混合。
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{label} 已存在且非空: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _is_within(path: Path, parent: Path) -> bool:
    # resolve 后再判断包含关系，符号链接和相对路径不会绕过目录隔离约束。
    return path == parent or parent in path.parents


def _validate_split(relative_path: Path, stats: FileStats) -> None:
    # 计数、模和与异或摘要共同确认两份输出是输入行的无交集分割。
    if stats.original_lines != stats.kept_lines + stats.removed_lines:
        raise ValueError(f"行数校验失败: {relative_path}")
    if stats.original_sum != (stats.kept_sum + stats.removed_sum) % _DIGEST_MODULUS:
        raise ValueError(f"行集合校验失败: {relative_path}")
    if stats.original_xor != stats.kept_xor ^ stats.removed_xor:
        raise ValueError(f"行交集校验失败: {relative_path}")


def _archive_child_directories(kept_dir: Path, archive_dir: Path) -> list[str]:
    # 每个一级目录单独归档，便于保持现有按配置集合分发的粒度。
    _prepare_output_dir(archive_dir, "归档目录")
    archives: list[str] = []
    for child in sorted(path for path in kept_dir.iterdir() if path.is_dir()):
        archive_path = archive_dir / f"{child.name}.tar"
        # arcname 固定为目录名，避免归档携带调用机器上的绝对路径。
        with tarfile.open(archive_path, "w") as archive:
            archive.add(child, arcname=child.name)
        archives.append(str(archive_path))
    return archives


def run(
    *,
    input_dir: Path,
    kept_dir: Path,
    removed_dir: Path,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    seed: str = DEFAULT_SEED,
    target_apis: Iterable[str] = DEFAULT_TARGET_APIS,
    archive_dir: Path | None = None,
) -> dict[str, object]:
    """Split every text file into kept and removed configuration sets."""
    input_dir = input_dir.resolve()
    kept_dir = kept_dir.resolve()
    removed_dir = removed_dir.resolve()
    target_apis = tuple(target_apis)
    report_apis = tuple(dict.fromkeys((*target_apis, *ALWAYS_REMOVED_APIS)))

    # 三个数据目录必须物理隔离，否则递归扫描可能读到本次运行刚写出的文件。
    if not input_dir.is_dir():
        raise ValueError(f"输入目录不存在: {input_dir}")
    if not 0 <= keep_ratio <= 1:
        raise ValueError("--keep-ratio 必须位于 0 和 1 之间")
    if not target_apis:
        raise ValueError("至少需要一个目标 API")
    if (
        kept_dir == removed_dir
        or _is_within(kept_dir, input_dir)
        or _is_within(removed_dir, input_dir)
    ):
        raise ValueError("输入、保留和裁减目录必须互不相同")

    _prepare_output_dir(kept_dir, "保留目录")
    _prepare_output_dir(removed_dir, "裁减目录")
    # 报告按源文件记录计数，归档列表即使未启用也保持稳定的 JSON 结构。
    report: dict[str, object] = {"files": {}, "archives": []}

    # 先复制空目录，确保两个输出集合与输入保持相同的目录层级。
    for directory in sorted(path for path in input_dir.rglob("*") if path.is_dir()):
        relative_path = directory.relative_to(input_dir)
        (kept_dir / relative_path).mkdir(parents=True, exist_ok=True)
        (removed_dir / relative_path).mkdir(parents=True, exist_ok=True)

    for source in sorted(path for path in input_dir.rglob("*") if path.is_file()):
        relative_path = source.relative_to(input_dir)
        kept_path = kept_dir / relative_path
        removed_path = removed_dir / relative_path
        kept_path.parent.mkdir(parents=True, exist_ok=True)
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        # 每个 API 同时记录两侧计数，便于审计抽样比例和确认非目标行未被裁减。
        stats = FileStats(api_counts={api: {"kept": 0, "removed": 0} for api in report_apis})

        # 单次流式读取同步写入两侧，避免大配置集合常驻内存。
        with (
            source.open(encoding="utf-8") as source_file,
            kept_path.open("w", encoding="utf-8") as kept_file,
            removed_path.open("w", encoding="utf-8") as removed_file,
        ):
            for line_number, raw_line in enumerate(source_file, start=1):
                _record_digest(stats, "original", raw_line)
                # 先判断固定清理项，再执行目标 API 的稳定抽样。
                removed_api = _always_removed_api(raw_line)
                api = removed_api or _matched_api(raw_line, target_apis)
                # empty 系列无条件裁减，其余非目标 API 保留，目标 API 按比例抽样。
                keep = removed_api is None and (
                    api is None
                    or _should_keep(seed, relative_path, line_number, raw_line, keep_ratio)
                )
                if keep:
                    kept_file.write(raw_line)
                    _record_digest(stats, "kept", raw_line)
                    if api is not None:
                        stats.api_counts[api]["kept"] += 1
                else:
                    removed_file.write(raw_line)
                    _record_digest(stats, "removed", raw_line)
                    stats.api_counts[api]["removed"] += 1

        # 校验在每个文件完成后立即执行，使失败位置精确到源文件。
        _validate_split(relative_path, stats)
        report["files"][relative_path.as_posix()] = {
            "original_lines": stats.original_lines,
            "kept_lines": stats.kept_lines,
            "removed_lines": stats.removed_lines,
            "api_counts": stats.api_counts,
        }

    if archive_dir is not None:
        archive_dir = archive_dir.resolve()
        # 归档放入保留目录会在归档过程中改变输入树，必须提前拒绝。
        if archive_dir == kept_dir or kept_dir in archive_dir.parents:
            raise ValueError("归档目录不能位于保留目录内")
        report["archives"] = _archive_child_directories(kept_dir, archive_dir)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机裁减指定 API 配置，并输出保留集与被裁减集")
    parser.add_argument("--input-dir", type=Path, required=True, help="未裁减配置集目录")
    parser.add_argument("--kept-dir", type=Path, required=True, help="保留配置集输出目录")
    parser.add_argument("--removed-dir", type=Path, required=True, help="被裁减配置集输出目录")
    parser.add_argument(
        "--api",
        action="append",
        default=None,
        help="目标 API 名；重复指定会覆盖默认的 6 个 API",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=DEFAULT_KEEP_RATIO,
        help=f"目标 API 行保留比例，默认 {DEFAULT_KEEP_RATIO}",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, help="稳定随机选择的种子")
    parser.add_argument("--archive-dir", type=Path, help="可选：按保留集一级子目录生成 tar 包")
    parser.add_argument("--report-file", type=Path, help="可选：写入 JSON 统计报告")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(
        input_dir=args.input_dir,
        kept_dir=args.kept_dir,
        removed_dir=args.removed_dir,
        keep_ratio=args.keep_ratio,
        seed=args.seed,
        target_apis=tuple(args.api) if args.api is not None else DEFAULT_TARGET_APIS,
        archive_dir=args.archive_dir,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report_file is not None:
        # 报告不属于数据输出目录，可由调用方按流水线约定独立放置。
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
