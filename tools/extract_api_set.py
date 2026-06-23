# 提取 API 名集合小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT_PATHS = ["tester/api_config/api_config_tmp.txt"]
DEFAULT_OUTPUT_DIR = Path("tester/api_config/output")
OUTPUT_FILE_NAME = "api_extracted.txt"


def collect_input_files(input_paths):
    files = []
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            text_files = list(path.rglob("*.txt"))
            files.extend(text_files)
            print(f"Found {len(text_files)} .txt files in directory: {path}")
        else:
            print(f"Warning: {path} does not exist or is not accessible")
    return files


def extract_apis(input_paths, output_dir=DEFAULT_OUTPUT_DIR):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return

    print(f"Processing {len(input_files)} files...")

    api_names = set()
    total_processed = 0

    for input_file in input_files:
        try:
            content = input_file.read_text(encoding="utf-8")
            file_count = 0

            for line in content.splitlines():
                line = line.strip()
                if line and "(" in line:
                    api_name = line.split("(", 1)[0].strip()
                    if api_name:
                        api_names.add(api_name)
                        file_count += 1
                        total_processed += 1

            print(f"Processed {file_count} APIs from {input_file}")
        except Exception as err:
            print(f"Error reading {input_file}: {err}")
            continue

    if not api_names:
        print("No valid APIs found")
        return

    print(f"Total processed: {total_processed}, Unique APIs: {len(api_names)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sorted_apis = sorted(api_names)

    output_file = output_path / OUTPUT_FILE_NAME
    output_file.write_text("\n".join(sorted_apis) + "\n", encoding="utf-8")
    print(f"Wrote {len(sorted_apis)} API names to {output_file}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="API 提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -i config.txt        # 处理单个配置文件
  python %(prog)s -i configs/          # 处理目录下所有 .txt 文件
  python %(prog)s -i . -o output/      # 当前目录
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extract_apis(args.input, args.output_dir)


if __name__ == "__main__":
    main()
