# 召回配置集合小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_INPUT_PATHS = ["tester/api_config/5_accuracy"]
DEFAULT_OUTPUT_FILE = Path("tester/api_config/api_config_retrieved.txt")


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


def build_pattern(keywords, exact_match=False):
    if exact_match:
        return re.compile("|".join(rf"\b{re.escape(kw)}\b[^(\n]*\(" for kw in keywords))
    return re.compile("|".join(rf"^[^(\n]*{re.escape(kw)}[^(\n]*\(" for kw in keywords))


def search_files(input_paths, keywords, output_file=DEFAULT_OUTPUT_FILE, exact_match=False):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return

    pattern = build_pattern(keywords, exact_match)
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
        except (UnicodeDecodeError, PermissionError) as err:
            print(f"Error reading {input_file}: {err}")
            continue

    print(f"Retrieved {count} configs")
    print(f"Get {len(configs)} unique configs")
    print(f"APIs: {sorted(prefixes)}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sorted(configs)) + "\n", encoding="utf-8")
    print(f"Saved to {output_path}")


def parse_args(argv=None):
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
        default=DEFAULT_INPUT_PATHS,
        help="输入路径列表（支持文件或目录）",
    )
    parser.add_argument(
        "--keywords",
        "-k",
        nargs="+",
        required=True,
        help="关键词列表",
    )
    parser.add_argument(
        "--exact",
        "-e",
        action="store_true",
        help="启用精确匹配（匹配完整单词）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT_FILE),
        help="输出文件路径",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    search_files(args.input, args.keywords, args.output, args.exact)


if __name__ == "__main__":
    main()
