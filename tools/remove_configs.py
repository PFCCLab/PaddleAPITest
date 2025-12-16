# 移除指定配置小工具
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
    return files


def load_configs_to_remove(remove_config_file):
    configs_to_remove = set()

    path = Path(remove_config_file)
    try:
        content = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        configs_to_remove.update(lines)
        print(f"Loaded {len(configs_to_remove)} configs to remove from {path}")
    except Exception as err:
        print(f"Error reading remove config file {path}: {err}")
        raise

    return configs_to_remove


def remove_configs_from_files(input_paths, remove_config_file, backup=False):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return

    configs_to_remove = load_configs_to_remove(remove_config_file)
    if not configs_to_remove:
        print("No configs to remove found")
        return

    print(f"Processing {len(input_files)} files...")
    print(f"Will remove {len(configs_to_remove)} unique configs")

    total_removed = 0
    files_modified = 0

    for input_file in input_files:
        try:
            content = input_file.read_text(encoding="utf-8")
            original_lines = content.splitlines()

            filtered_lines = []
            removed_count = 0

            for line in original_lines:
                stripped_line = line.strip()
                if stripped_line and stripped_line in configs_to_remove:
                    removed_count += 1
                else:
                    filtered_lines.append(line)

            if removed_count > 0:
                files_modified += 1
                total_removed += removed_count

                if backup:
                    backup_file = input_file.with_suffix(input_file.suffix + ".backup")
                    backup_file.write_text(content, encoding="utf-8")
                    print(f"Created backup: {backup_file}")

                new_content = "\n".join(filtered_lines)
                if new_content and not new_content.endswith("\n"):
                    new_content += "\n"

                input_file.write_text(new_content, encoding="utf-8")

                print(
                    f"Modified {input_file}: removed {removed_count} configs, "
                    f"remaining {len(filtered_lines)} lines"
                )
            else:
                print(f"No configs to remove in {input_file}")

        except Exception as err:
            print(f"Error processing {input_file}: {err}")
            continue

    print("\nSummary:")
    print(f"Files processed: {len(input_files)}")
    print(f"Files modified: {files_modified}")
    print(f"Total configs removed: {total_removed}")


def main():
    parser = argparse.ArgumentParser(
        description="移除指定配置工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python config_remover.py -i input.txt -r remove_configs.txt           # 从单文件删除配置
  python config_remover.py -i file1.txt file2.txt -r remove_configs.txt # 从多文件删除配置
  python config_remover.py -i ./configs/ -r remove_configs.txt          # 从文件夹删除配置
  python config_remover.py -i input.txt -r remove_configs.txt --backup  # 有备份地处理
注意: 所有操作会原地修改文件。使用 --backup 选项可创建备份文件。
        """,
    )

    parser.add_argument("-i", "--input", nargs="+", required=True, help="待处理的文件或目录")
    parser.add_argument("-r", "--remove", required=True, help="包含要删除配置的文件")
    parser.add_argument("-b", "--backup", action="store_true", default=False, help="创建备份文件")

    args = parser.parse_args()

    remove_configs_from_files(args.input, args.remove, args.backup)


if __name__ == "__main__":
    main()
