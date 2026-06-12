# 合并 API 配置集合小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT_PATHS = ["tester/api_config/api_config_tmp.txt"]
DEFAULT_OUTPUT_DIR = Path("tester/api_config/output")
DEFAULT_MAX_CONFIGS_PER_FILE = 500000


def collect_input_files(input_paths):
    files = []
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            text_files = list(path.rglob("*.txt"))
            files.extend(text_files)
    return files


def _backup_file(input_file):
    backup_file = input_file.with_suffix(input_file.suffix + ".backup")
    backup_file.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created backup: {backup_file}")


def process_api_configs(
    input_paths,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_configs_per_file=DEFAULT_MAX_CONFIGS_PER_FILE,
    inplace=False,
    backup=True,
):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return

    print(f"Processing {len(input_files)} files...")

    api_configs = set()
    total_read = 0

    for input_file in input_files:
        try:
            content = input_file.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            api_configs.update(lines)
            total_read += len(lines)
            print(f"Read {len(lines)} configs from {input_file}")
        except Exception as err:
            print(f"Error reading {input_file}: {err}")
            continue

    if not api_configs:
        print("No valid configs found")
        return

    print(f"Total configs: {total_read}, Unique configs: {len(api_configs)}")

    sorted_configs = sorted(api_configs)

    if inplace:
        merged_content = "\n".join(sorted_configs) + "\n"
        for input_file in input_files:
            try:
                if backup:
                    _backup_file(input_file)
                input_file.write_text(merged_content, encoding="utf-8")
                print(f"Inplace wrote {len(sorted_configs)} configs to {input_file}")
            except Exception as err:
                print(f"Error writing {input_file}: {err}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(sorted_configs) <= max_configs_per_file:
        output_file = output_path / "api_config_merged.txt"
        output_file.write_text("\n".join(sorted_configs) + "\n", encoding="utf-8")
        print(f"Wrote {len(sorted_configs)} configs to {output_file}")
    else:
        for i in range(0, len(sorted_configs), max_configs_per_file):
            chunk = sorted_configs[i : i + max_configs_per_file]
            chunk_num = i // max_configs_per_file + 1
            output_file = output_path / f"api_config_merged_part{chunk_num}.txt"
            output_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
            print(f"Wrote {len(chunk)} configs to {output_file}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="API 配置集合整理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -i file.txt                         # 处理单个文件
  python %(prog)s -i dir/                             # 处理目录下所有 .txt 文件
  python %(prog)s -i . -o output/ --max-configs 100000 # 限制 10 万条/文件
  python %(prog)s -i file.txt -I                      # 原地去重排序，覆盖原文件
  python %(prog)s -i dir/ -I --no-backup              # 原地处理且不创建备份
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=DEFAULT_INPUT_PATHS,
        help="输入路径列表（支持文件或目录）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录路径",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=DEFAULT_MAX_CONFIGS_PER_FILE,
        help="单个输出文件最大配置数量",
    )
    parser.add_argument(
        "--inplace",
        "-I",
        action="store_true",
        default=False,
        help="原地修改：将合并去重排序后的结果写回所有输入文件（忽略 --output-dir）",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="原地修改时不创建备份",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    process_api_configs(
        args.input,
        args.output_dir,
        args.max_configs,
        args.inplace,
        args.backup,
    )


if __name__ == "__main__":
    main()
