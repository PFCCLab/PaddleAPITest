# 提取 API 名集合小工具
# @author: cangtianhuang
# @date: 2026-06-11

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


def extract_apis(input_paths, output_file):
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

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_apis = sorted(api_names)

    output_path.write_text("\n".join(sorted_apis) + "\n", encoding="utf-8")
    print(f"Wrote {len(sorted_apis)} API names to {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="API 提取工具：从配置文件中提取唯一 API 名称集合。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -i config.txt -o api_extracted.txt
  python %(prog)s -i configs/ -o /output/dir/apis.txt
  python %(prog)s -i file1.txt file2.txt -o result.txt
        """,
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="输入路径列表（支持文件或目录）",
    )
    parser.add_argument(
        "-o", "--output",
        default="api_extracted.txt",
        help="输出文件路径（默认：当前目录下 api_extracted.txt）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    extract_apis(args.input, args.output)


if __name__ == "__main__":
    main()
