#!/usr/bin/env python3
"""Deduplicate configuration lines while preserving sorted unique output.

Usage:
    python dedup_config.py -i api_config_0_size.txt
    python dedup_config.py -i api_config_0_size.txt -o /output/dir/dedup.txt
"""

from __future__ import annotations

import argparse
import heapq
import os
import tempfile

CHUNK_BYTES = 64 * 1024 * 1024


def _flush_chunk(lines, chunk_dir, chunk_index):
    """将有限大小的去重块排序落盘，避免大文件全部进入内存。"""
    # 块内去重先完成，跨块重复交给最终归并处理。
    if not lines:
        return None
    path = os.path.join(chunk_dir, f"chunk_{chunk_index:06d}.txt")
    with open(path, "w", encoding="utf-8") as output_file:
        for line in sorted(lines):
            output_file.write(line + "\n")
    return path


def _merge_chunks(chunk_paths, output_path):
    """归并分块并去重，保持原脚本的排序输出协议。"""
    # 文件句柄数量等于块数，控制块大小即可控制归并内存。
    streams = [open(path, encoding="utf-8") for path in chunk_paths]
    unique_count = 0
    previous = None
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            for line in heapq.merge(*(stream for stream in streams)):
                if line == previous:
                    continue
                output_file.write(line)
                previous = line
                unique_count += 1
    finally:
        for stream in streams:
            stream.close()
    return unique_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deduplicate configuration lines while preserving sorted unique output.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input config file path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. Default: <input_stem>_dedup.txt in same dir as input",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        stem, ext = os.path.splitext(input_path)
        output_path = f"{stem}_dedup{ext}"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = 0
    blank_lines = 0
    skipped_non_config = 0
    with tempfile.TemporaryDirectory(prefix=".dedup_chunks.", dir=output_dir or ".") as chunk_dir:
        # 只保留当前块的集合，避免与输入文件总大小线性增长。
        chunk_paths = []
        chunk_lines = set()
        chunk_bytes = 0
        chunk_index = 0
        with open(input_path, encoding="utf-8") as input_file:
            for line in input_file:
                total += 1
                line = line.strip()
                if not line:
                    blank_lines += 1
                    continue
                # 与引擎入口保持相同边界，预处理产物只能包含 Paddle API 配置。
                if not line.startswith("paddle."):
                    skipped_non_config += 1
                    continue
                # 不跨块查重，避免维护全局 Python set。
                if line in chunk_lines:
                    continue
                chunk_lines.add(line)
                chunk_bytes += len(line) + 1
                if chunk_bytes >= CHUNK_BYTES:
                    path = _flush_chunk(chunk_lines, chunk_dir, chunk_index)
                    chunk_paths.append(path)
                    chunk_index += 1
                    chunk_lines.clear()
                    chunk_bytes = 0
        path = _flush_chunk(chunk_lines, chunk_dir, chunk_index)
        if path is not None:
            chunk_paths.append(path)
        unique_count = _merge_chunks(chunk_paths, output_path)

    valid_lines = total - blank_lines - skipped_non_config
    print(f"Total lines:   {total}")
    print(f"Blank lines:   {blank_lines}")
    print(f"Non-config:    {skipped_non_config}")
    print(f"Config lines:  {valid_lines}")
    print(f"Unique lines:  {unique_count}")
    print(f"Duplicates:    {valid_lines - unique_count}")
    print(f"Output:        {output_path}")


if __name__ == "__main__":
    main()
