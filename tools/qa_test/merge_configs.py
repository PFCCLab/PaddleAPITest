#!/usr/bin/env python3
"""合并多个 txt 配置文件，剔除空行。

Usage:
    python merge_configs.py -i dir1/ dir2/
    python merge_configs.py -i file1.txt file2.txt file3.txt
    python merge_configs.py -i dir1/ file2.txt -o /output/merged.txt
"""

from __future__ import annotations

import argparse
import glob
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="合并多个 txt 配置文件，剔除空行。",
    )
    parser.add_argument(
        "-i", "--inputs", nargs="+", required=True, help="输入路径（目录或文件），可指定多个"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="输出文件路径。默认：第一个输入目录下 merged.txt"
    )
    return parser.parse_args()


def collect_files(inputs):
    """从输入列表收集所有 txt 文件路径。"""
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.txt"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"警告：路径不存在，跳过 {path}")
    return files


def main():
    args = parse_args()

    if args.output:
        output_file = args.output
    else:
        first_input = args.inputs[0]
        if os.path.isdir(first_input):
            output_file = os.path.join(first_input, "merged.txt")
        else:
            output_file = os.path.join(os.path.dirname(first_input) or ".", "merged.txt")

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    txt_files = collect_files(args.inputs)
    output_abs = os.path.abspath(output_file)

    lines = []
    for filepath in txt_files:
        if os.path.abspath(filepath) == output_abs:
            continue
        with open(filepath) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)

    with open(output_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"合并完成，共 {len(lines)} 行，输出到 {output_file}")


if __name__ == "__main__":
    main()
