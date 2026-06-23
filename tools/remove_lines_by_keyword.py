# 按关键词删除配置行小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

DEFAULT_FILE_PATTERN = "tester/api_config/monitor_config/accuracy/GPU/monitoring_configs_*.txt"
DEFAULT_KEYWORD_FILE = Path("kw.txt")


def load_keywords(keyword_file):
    path = Path(keyword_file)
    try:
        return {
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    except FileNotFoundError:
        print(f"错误：关键字文件 {path} 不存在")
        raise


def backup_file(file_path):
    path = Path(file_path)
    backup_path = path.with_suffix(path.suffix + ".backup")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"创建备份: {backup_path}")


def delete_lines_with_keywords(file_pattern, keyword_set, case_sensitive=True, backup=True):
    target_files = sorted(glob.glob(file_pattern))
    if not target_files:
        print(f"警告：未找到匹配 {file_pattern} 的文件")
        return

    flags = 0 if case_sensitive else re.IGNORECASE
    patterns = [re.compile(re.escape(keyword), flags) for keyword in keyword_set]
    total_removed = 0

    for file_path in target_files:
        try:
            path = Path(file_path)
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            original_count = len(lines)
            new_lines = [
                line for line in lines if not any(pattern.search(line) for pattern in patterns)
            ]
            removed_count = original_count - len(new_lines)
            total_removed += removed_count

            if removed_count > 0 and backup:
                backup_file(path)

            path.write_text("".join(new_lines), encoding="utf-8")
            print(
                f"处理 {file_path}: 原始行数 {original_count}, "
                f"删除 {removed_count} 行, 保留 {len(new_lines)} 行"
            )
        except Exception as err:
            print(f"处理文件 {file_path} 时出错: {err!s}")

    print(f"\n处理完成！共处理 {len(target_files)} 个文件, 总计删除 {total_removed} 行")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按关键词删除配置行工具")
    parser.add_argument(
        "--file-pattern",
        "-p",
        default=DEFAULT_FILE_PATTERN,
        help="待处理文件 glob 匹配模式",
    )
    parser.add_argument(
        "--keyword-file",
        "-k",
        default=str(DEFAULT_KEYWORD_FILE),
        help="关键词文件路径，每行一个关键词",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="关键词匹配时忽略大小写",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="不创建备份",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    keywords = load_keywords(args.keyword_file)
    if not keywords:
        print("警告：关键字集为空，未执行任何操作")
        return

    print(
        f"加载 {len(keywords)} 个关键字: {', '.join(sorted(keywords)[:5])}"
        + ("..." if len(keywords) > 5 else "")
    )
    delete_lines_with_keywords(
        args.file_pattern,
        keywords,
        case_sensitive=not args.ignore_case,
        backup=args.backup,
    )


if __name__ == "__main__":
    main()
