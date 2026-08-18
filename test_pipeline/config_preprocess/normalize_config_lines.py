#!/usr/bin/env python3
"""将原始配置规范化为每行一个完整的 Paddle API 调用。"""

# 规范化只修复物理换行丢失，不改变调用内容、顺序或重复次数。
# 成功拆出的调用立即写入输出，避免 GB 级输入常驻内存。
# 失败的原始行连同行号写入拒绝文件，供修复后重新处理。
# 严格模式在完整扫描结束后失败，已完成工作和错误证据都会保留。

from __future__ import annotations

import argparse
from pathlib import Path

from test_pipeline.config_preprocess.config_lines import split_top_level_calls


def normalize_file(input_path, output_path, reject_path):
    """流式拆分文件并记录拒绝行。"""
    # 返回计数只描述结构规范化，不做 APIConfig 语义解析或文本去重。
    input_path = Path(input_path)
    output_path = Path(output_path)
    reject_path = Path(reject_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出文件不能相同")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"lines": 0, "calls": 0, "split_lines": 0, "rejected": 0}
    with (
        input_path.open(encoding="utf-8") as source_file,
        output_path.open("w", encoding="utf-8") as normalized_file,
        reject_path.open("w", encoding="utf-8") as reject_file,
    ):
        for line_number, raw_line in enumerate(source_file, start=1):
            counts["lines"] += 1
            config = raw_line.strip()
            if not config:
                continue
            try:
                calls = split_top_level_calls(config)
            except ValueError as error:
                # 错误行完整落盘，后续可以修复后单独重放，不会丢失原始数据。
                counts["rejected"] += 1
                reject_file.write(f"{input_path}:{line_number}\t{error}\t{config}\n")
                continue
            if len(calls) > 1:
                counts["split_lines"] += 1
            for call in calls:
                normalized_file.write(call + "\n")
                counts["calls"] += 1

    if counts["rejected"] == 0:
        reject_path.unlink()
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="原始配置文件")
    parser.add_argument("-o", "--output", required=True, help="规范化配置文件")
    parser.add_argument(
        "--rejects",
        default=None,
        help="拒绝文件路径，默认是 <output>.unparsed.txt",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="完成扫描并写出拒绝文件后，对任何坏行返回非零状态",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reject_path = args.rejects or f"{args.output}.unparsed.txt"
    counts = normalize_file(args.input, args.output, reject_path)
    print(
        f"规范化: {args.input} -> {args.output}，"
        f"输入 {counts['lines']} 行，输出 {counts['calls']} 个调用，"
        f"拆分 {counts['split_lines']} 行，拒绝 {counts['rejected']} 行"
    )
    if counts["rejected"]:
        print(f"警告: 无法拆分的配置已写入 {reject_path}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
