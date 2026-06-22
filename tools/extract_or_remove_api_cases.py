# 按 API 提取或删除配置 case 小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG_DIR = Path("tester/api_config/5_accuracy")
DEFAULT_FILE_KEYWORD = "accuracy"
DEFAULT_OUTPUT_FILE = Path("mytmp.txt")


def collect_target_files(config_dir=DEFAULT_CONFIG_DIR, file_keyword=DEFAULT_FILE_KEYWORD):
    path = Path(config_dir)
    if not path.exists():
        print(f"{path} not exists")
        return []
    return sorted(file_path for file_path in path.iterdir() if file_keyword in file_path.name)


def count_unique_apis(config_file):
    api_names = set()
    with Path(config_file).open(encoding="utf-8") as f:
        for line in f:
            api_name = line.split("(", 1)[0]
            if api_name:
                api_names.add(api_name)
    return len(api_names)


def check_config_clean(
    config_prefix, config_dir=DEFAULT_CONFIG_DIR, file_keyword=DEFAULT_FILE_KEYWORD
):
    target_files = collect_target_files(config_dir, file_keyword)
    total_count = 0
    existing_files = set()
    existing_counts = {}

    for target_file in target_files:
        count = 0
        with target_file.open(encoding="utf-8") as f:
            for line in f:
                if config_prefix in line:
                    total_count += 1
                    count += 1
                    existing_files.add(target_file.name)
        existing_counts[target_file.name] = count

    api_name = config_prefix[:-1]
    if total_count:
        print(f"{api_name} is still exist, number of times : {total_count}")
        print(f"{api_name} is still exist in these files : ")
        for file_name in sorted(existing_files):
            print(file_name, existing_counts[file_name])
        return False

    print("clean")
    return True


def append_config_cases(
    config_prefix, output_file, config_dir=DEFAULT_CONFIG_DIR, file_keyword=DEFAULT_FILE_KEYWORD
):
    target_files = collect_target_files(config_dir, file_keyword)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for target_file in target_files:
        lines = target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        matched_lines = [line for line in lines if config_prefix in line]

        with output_path.open("a", encoding="utf-8") as f:
            f.writelines(matched_lines)

        if matched_lines:
            print(
                "add",
                len(matched_lines),
                "lines in",
                target_file.name,
                "to ",
                output_path,
                "successfully",
            )


def backup_file(input_file):
    backup_path = input_file.with_suffix(input_file.suffix + ".backup")
    backup_path.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created backup: {backup_path}")


def remove_config_cases(
    config_prefix, config_dir=DEFAULT_CONFIG_DIR, file_keyword=DEFAULT_FILE_KEYWORD, backup=True
):
    target_files = collect_target_files(config_dir, file_keyword)
    for target_file in target_files:
        lines = target_file.read_text(encoding="utf-8").splitlines(keepends=True)
        count = sum(config_prefix in line for line in lines)
        remaining_lines = [line for line in lines if config_prefix not in line]

        if count and backup:
            backup_file(target_file)

        target_file.write_text("".join(remaining_lines), encoding="utf-8")
        if count:
            print("remove", count, "lines in", target_file.name, "successfully")


def process_api_cases(
    config,
    output_file=DEFAULT_OUTPUT_FILE,
    remove=False,
    config_dir=DEFAULT_CONFIG_DIR,
    file_keyword=DEFAULT_FILE_KEYWORD,
    backup=True,
):
    print(f"开始处理配置：{config}，目标文件：{output_file}")
    config_prefix = f"{config}("

    if not check_config_clean(config_prefix, config_dir, file_keyword):
        append_config_cases(config_prefix, output_file, config_dir, file_keyword)

    if remove:
        print(f"执行删除配置：{config}")
        remove_config_cases(config_prefix, config_dir, file_keyword, backup)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按 API 提取或删除配置 case")
    parser.add_argument("--config", type=str, required=True, help="配置字符串，例如 paddle.numel")
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="临时文件名，默认是 mytmp.txt",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="配置目录路径",
    )
    parser.add_argument(
        "--file-keyword",
        type=str,
        default=DEFAULT_FILE_KEYWORD,
        help="待扫描文件名关键词",
    )
    parser.add_argument("--remove", action="store_true", help="是否执行删除配置")
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="删除时不创建备份",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    process_api_cases(
        args.config,
        args.dst,
        args.remove,
        args.config_dir,
        args.file_keyword,
        args.backup,
    )


if __name__ == "__main__":
    main()
