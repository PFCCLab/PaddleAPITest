# 获取 api 配置集合小工具
# @author: cangtianhuang
# @date: 2025-10-29
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
    return files


def process_api_configs(input_paths, output_dir, max_configs_per_file=500000, inplace=False):
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


def main():
    default_input = ["tester/api_config/api_config_tmp.txt"]
    default_output = "tester/api_config/output"

    parser = argparse.ArgumentParser(
        description="API配置集合整理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -i file.txt                    # 处理单个文件
  python %(prog)s -i dir/                        # 处理目录下所有.txt文件
  python %(prog)s -i . -o output/ --max-configs 100000  # 当前目录，限制10万条/文件
  python %(prog)s -i file.txt -I                        # 原地去重排序，覆盖原文件
  python %(prog)s -i dir/ -I                            # 原地处理目录下所有.txt文件
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
    parser.add_argument("--max-configs", type=int, default=500000, help="单个输出文件最大配置数量")
    parser.add_argument(
        "--inplace",
        "-I",
        action="store_true",
        default=False,
        help="原地修改：将合并去重排序后的结果写回所有输入文件（忽略 --output-dir）",
    )

    args = parser.parse_args()
    process_api_configs(args.input, args.output_dir, args.max_configs, args.inplace)


if __name__ == "__main__":
    main()
