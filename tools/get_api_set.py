# 获取 api 集合小工具
# @author: cangtianhuang
# @date: 2025-09-26
from __future__ import annotations

import argparse
from pathlib import Path


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


def extract_apis(input_paths, output_dir):
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

    sorted_apis = sorted(api_names)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "api_extracted.txt"
    output_file.write_text("\n".join(sorted_apis) + "\n", encoding="utf-8")
    print(f"Wrote {len(sorted_apis)} API names to {output_file}")


def main():
    default_input = ["tester/api_config/api_config_tmp.txt"]
    default_output = "tester/api_config/output"

    parser = argparse.ArgumentParser(
        description="API提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -i config.txt                 # 处理单个配置文件
  python %(prog)s -i configs/                   # 处理目录下所有.txt文件
  python %(prog)s -i . -o output/                 # 当前目录
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=default_input,
        help="输入路径列表（支持文件或目录）",
    )
    parser.add_argument("--output-dir", "-o", default=default_output, help="输出目录路径")

    args = parser.parse_args()
    extract_apis(args.input, args.output_dir)


if __name__ == "__main__":
    main()
