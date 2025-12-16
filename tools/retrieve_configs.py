# 召回配置小工具
# @author: cangtianhuang
# @date: 2025-09-26
from __future__ import annotations

import argparse
import re
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


def search_files(input_paths, keywords, output_file, exact_match=False):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return

    if exact_match:
        pattern = re.compile("|".join(rf"\b{re.escape(kw)}\b[^(\n]*\(" for kw in keywords))
    else:
        pattern = re.compile("|".join(rf"^[^(\n]*{re.escape(kw)}[^(\n]*\(" for kw in keywords))

    configs = set()
    prefixes = set()
    count = 0

    for input_file in input_files:
        print(f"Retrieving from {input_file.name}...")
        try:
            content = input_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.rstrip("\n\r")
                if match := pattern.search(line):
                    count += 1
                    configs.add(line)
                    paren_pos = line.find("(", match.start())
                    if paren_pos != -1:
                        prefixes.add(line[:paren_pos].strip())
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"Error reading {input_file}: {e}")
            continue

    print(f"Retrieved {count} configs")
    print(f"Get {len(configs)} unique configs")
    print(f"APIs: {sorted(prefixes)}")

    Path(output_file).write_text("\n".join(sorted(configs)) + "\n", encoding="utf-8")
    print(f"Saved to {output_file}")


def main():
    default_input = ["tester/api_config/5_accuracy"]
    default_keywords = []
    default_output = "tester/api_config/api_config_retrieved.txt"

    parser = argparse.ArgumentParser(
        description="配置文件召回工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -k matmul linear      # 模糊搜索
  python %(prog)s -k paddle.matmul -e   # 精确搜索
""",
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        default=default_input,
        help="输入路径列表（支持文件或目录）",
    )
    parser.add_argument(
        "--keywords",
        "-k",
        nargs="+",
        required=True,
        default=default_keywords,
        help="关键词列表",
    )
    parser.add_argument(
        "--exact",
        "-e",
        action="store_true",
        help="启用精确匹配（匹配完整单词）",
    )
    parser.add_argument("--output", "-o", default=default_output, help="输出文件路径")

    args = parser.parse_args()

    search_files(args.input, args.keywords, args.output, args.exact)


if __name__ == "__main__":
    main()
