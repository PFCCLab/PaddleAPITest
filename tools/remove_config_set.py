# 移除指定配置集合小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

BACKUP_DIR_NAME = ".rm_config_backups"
LEGACY_BACKUP_DIR_NAME = "remove_config_set_backups"
BACKUP_DIR_NAMES = (BACKUP_DIR_NAME, LEGACY_BACKUP_DIR_NAME)
BACKUP_SUFFIX = ".backup"
REMOVED_CONFIGS_SUFFIX = ".removed_configs"
CONFIRM_TEXT = "yes"
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}_\d{6}(?:\.\d+)?$")


def print_section(title):
    print(f"\n== {title} ==")


def print_file_item(index, total, path):
    print(f"\n[{index}/{total}] {path}")


def confirm_action(message, dry_run=False):
    if dry_run:
        return True

    answer = input(f"{message}\nType '{CONFIRM_TEXT}' to continue: ").strip()
    if answer != CONFIRM_TEXT:
        print("Canceled")
        return False
    return True


def collect_input_files(input_paths):
    files = []
    seen_files = set()
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_file():
            text_files = [path]
        elif path.is_dir():
            text_files = [
                text_file
                for text_file in sorted(path.rglob("*.txt"))
                if not any(
                    backup_dir_name in text_file.parts for backup_dir_name in BACKUP_DIR_NAMES
                )
            ]
        else:
            print(f"Ignored invalid path: {path}")
            continue

        for text_file in text_files:
            resolved_file = text_file.resolve()
            if resolved_file not in seen_files:
                files.append(text_file)
                seen_files.add(resolved_file)
    return files


def get_timestamped_path(input_file, timestamp, suffix):
    backup_dir = input_file.parent / BACKUP_DIR_NAME
    backup_file = backup_dir / f"{input_file.name}.{timestamp}.{suffix}"
    if not backup_file.exists():
        return backup_file

    index = 1
    while True:
        backup_file = backup_dir / f"{input_file.name}.{timestamp}.{index}.{suffix}"
        if not backup_file.exists():
            return backup_file
        index += 1


def load_configs_to_remove(remove_config_paths):
    configs_to_remove = set()
    remove_config_files = collect_input_files(remove_config_paths)
    if not remove_config_files:
        print("No valid remove config files found")
        return configs_to_remove

    print_section("Load Remove Configs")
    for index, path in enumerate(remove_config_files, start=1):
        try:
            content = path.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            old_count = len(configs_to_remove)
            configs_to_remove.update(lines)
            print(f"[{index}/{len(remove_config_files)}] {path}")
            print(f"  lines: {len(lines)}, new unique: {len(configs_to_remove) - old_count}")
        except Exception as err:
            print(f"Error reading remove config file {path}: {err}")
            raise

    print(f"Total unique configs to remove: {len(configs_to_remove)}")
    return configs_to_remove


def remove_configs_from_files(input_paths, remove_config_paths, backup=True, dry_run=False):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return []

    configs_to_remove = load_configs_to_remove(remove_config_paths)
    if not configs_to_remove:
        print("No configs to remove found")
        return []

    print_section("Process Input Files")
    if dry_run:
        print("Mode: dry-run (no files will be modified)")
    print(f"Files: {len(input_files)}, configs to remove: {len(configs_to_remove)}")

    total_removed = 0
    files_modified = 0
    all_removed_lines = []

    for index, input_file in enumerate(input_files, start=1):
        print_file_item(index, len(input_files), input_file)
        try:
            content = input_file.read_text(encoding="utf-8")
            original_lines = content.splitlines(keepends=True)

            filtered_lines = []
            removed_lines = []

            for line in original_lines:
                stripped_line = line.strip()
                if stripped_line and stripped_line in configs_to_remove:
                    removed_lines.append(line)
                else:
                    filtered_lines.append(line)

            removed_count = len(removed_lines)

            if removed_count > 0:
                files_modified += 1
                total_removed += removed_count
                all_removed_lines.extend(removed_lines)

                if backup:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    backup_file = get_timestamped_path(input_file, timestamp, "backup")
                    removed_backup_file = get_timestamped_path(
                        input_file, timestamp, "removed_configs"
                    )
                    if dry_run:
                        print(f"  backup: {backup_file} (dry-run)")
                        print(f"  removed: {removed_backup_file} (dry-run)")
                    else:
                        backup_file.parent.mkdir(parents=True, exist_ok=True)
                        backup_file.write_text(content, encoding="utf-8")
                        removed_backup_file.write_text("".join(removed_lines), encoding="utf-8")
                        print(f"  backup: {backup_file}")
                        print(f"  removed: {removed_backup_file}")

                if dry_run:
                    print(f"  status: would modify")
                    print(f"  remove: {removed_count} configs")
                    print(f"  remain: {len(filtered_lines)} lines")
                else:
                    input_file.write_text("".join(filtered_lines), encoding="utf-8")
                    print(f"  status: modified")
                    print(f"  removed: {removed_count} configs")
                    print(f"  remain: {len(filtered_lines)} lines")
            else:
                print("  status: unchanged")
                print("  removed: 0 configs")

        except Exception as err:
            print(f"  status: error")
            print(f"  error: {err}")
            continue

    print_section("Summary")
    print(f"Files processed: {len(input_files)}")
    print(f"Files {'would be modified' if dry_run else 'modified'}: {files_modified}")
    print(f"Configs {'would be removed' if dry_run else 'removed'}: {total_removed}")
    return all_removed_lines


def parse_backup_artifact(artifact_path):
    name = artifact_path.name
    if name.endswith(BACKUP_SUFFIX):
        suffix = BACKUP_SUFFIX
    elif name.endswith(REMOVED_CONFIGS_SUFFIX):
        suffix = REMOVED_CONFIGS_SUFFIX
    else:
        return None

    base_name = name[: -len(suffix)]
    dot_index = base_name.rfind(".")
    if dot_index == -1:
        return base_name, suffix, None

    possible_timestamp = base_name[dot_index + 1 :]
    if not TIMESTAMP_RE.match(possible_timestamp):
        return base_name, suffix, None
    return base_name[:dot_index], suffix, possible_timestamp


def collect_compact_groups(backup_dir):
    groups = {}
    for artifact_path in sorted(backup_dir.iterdir()):
        if not artifact_path.is_file():
            continue
        parsed = parse_backup_artifact(artifact_path)
        if parsed is None:
            continue
        original_name, suffix, timestamp = parsed
        group = groups.setdefault(original_name, {"backups": [], "removed": []})
        artifact = {"path": artifact_path, "timestamp": timestamp}
        if suffix == BACKUP_SUFFIX:
            group["backups"].append(artifact)
        else:
            group["removed"].append(artifact)
    return groups


def compact_backup_dir(backup_dir_path, dry_run=False):
    backup_dir = Path(backup_dir_path)
    if not backup_dir.is_dir():
        print(f"Invalid backup directory: {backup_dir}")
        return

    groups = collect_compact_groups(backup_dir)
    print_section("Compact Backup Directory")
    if dry_run:
        print("Mode: dry-run (no files will be modified)")
    print(f"Backup dir: {backup_dir}")
    print(f"Modified files: {len(groups)}")
    if not groups:
        return
    if not confirm_action("This will rewrite backup artifacts in the backup directory.", dry_run):
        return

    groups_compacted = 0
    backups_deleted = 0
    removed_deleted = 0
    for index, (original_name, group) in enumerate(sorted(groups.items()), start=1):
        backups = sorted(
            group["backups"],
            key=lambda artifact: (
                artifact["timestamp"] is None,
                artifact["timestamp"] or "",
                artifact["path"].name,
            ),
        )
        removed_artifacts = sorted(
            group["removed"],
            key=lambda artifact: (
                artifact["timestamp"] is None,
                artifact["timestamp"] or "",
                artifact["path"].name,
            ),
        )
        compact_backup = backup_dir / f"{original_name}{BACKUP_SUFFIX}"
        compact_removed = backup_dir / f"{original_name}{REMOVED_CONFIGS_SUFFIX}"

        print_file_item(index, len(groups), original_name)
        if not backups:
            print("  status: skipped")
            print("  reason: no backup file")
            continue

        first_backup = backups[0]["path"]
        merged_removed = []
        seen_removed = set()
        for artifact in removed_artifacts:
            removed_file = artifact["path"]
            try:
                lines = removed_file.read_text(encoding="utf-8").splitlines()
            except Exception as err:
                print(f"  warning: failed to read {removed_file}: {err}")
                continue
            add_unique_configs(merged_removed, seen_removed, lines)

        print(f"  keep backup: {first_backup.name} -> {compact_backup.name}")
        print(f"  merge removed: {len(removed_artifacts)} files -> {compact_removed.name}")
        print(f"  unique removed configs: {len(merged_removed)}")

        if dry_run:
            print("  status: would compact")
            groups_compacted += 1
            backups_deleted += sum(
                artifact["path"].name != compact_backup.name for artifact in backups[1:]
            )
            removed_deleted += sum(
                artifact["path"].name != compact_removed.name for artifact in removed_artifacts
            )
            continue

        backup_content = first_backup.read_text(encoding="utf-8")
        removed_content = "\n".join(merged_removed)
        if removed_content:
            removed_content += "\n"

        backup_paths = [artifact["path"] for artifact in backups]
        removed_paths = [artifact["path"] for artifact in removed_artifacts]
        for artifact_path in backup_paths + removed_paths:
            if artifact_path.name not in {compact_backup.name, compact_removed.name}:
                artifact_path.unlink()

        compact_backup.write_text(backup_content, encoding="utf-8")
        compact_removed.write_text(removed_content, encoding="utf-8")
        groups_compacted += 1
        backups_deleted += sum(path.name != compact_backup.name for path in backup_paths[1:])
        removed_deleted += sum(path.name != compact_removed.name for path in removed_paths)
        print("  status: compacted")

    print_section("Summary")
    print(f"Groups {'would be compacted' if dry_run else 'compacted'}: {groups_compacted}")
    print(f"Extra backups {'would be removed' if dry_run else 'removed'}: {backups_deleted}")
    print(f"Removed-config files {'would be merged' if dry_run else 'merged'}: {removed_deleted}")


def collect_backup_files(input_file):
    backup_files = []
    for backup_dir_name in BACKUP_DIR_NAMES:
        backup_dir = input_file.parent / backup_dir_name
        if backup_dir.is_dir():
            backup_files.extend(backup_dir.glob(f"{input_file.name}.*.backup"))
    return sorted(backup_files, key=lambda backup_file: backup_file.name)


def revert_files(input_paths, dry_run=False):
    input_files = collect_input_files(input_paths)
    if not input_files:
        print("No valid input files found")
        return

    print_section("Revert Input Files")
    if dry_run:
        print("Mode: dry-run (no files will be modified)")
    print(f"Files: {len(input_files)}")
    if not confirm_action("This will restore each input file from its latest backup.", dry_run):
        return

    files_reverted = 0
    files_would_revert = 0
    for index, input_file in enumerate(input_files, start=1):
        print_file_item(index, len(input_files), input_file)
        backup_files = collect_backup_files(input_file)
        if not backup_files:
            print("  status: no backup found")
            continue

        latest_backup = backup_files[-1]
        if dry_run:
            files_would_revert += 1
            print("  status: would revert")
            print(f"  source: {latest_backup}")
        else:
            input_file.write_text(latest_backup.read_text(encoding="utf-8"), encoding="utf-8")
            files_reverted += 1
            print("  status: reverted")
            print(f"  source: {latest_backup}")

    print_section("Summary")
    print(f"Files processed: {len(input_files)}")
    print(
        f"Files {'would be reverted' if dry_run else 'reverted'}: "
        f"{files_would_revert if dry_run else files_reverted}"
    )


def collect_backup_dirs(input_paths):
    backup_dirs = []
    seen_dirs = set()
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_file():
            candidate_dirs = [path.parent / BACKUP_DIR_NAME]
        elif path.is_dir():
            candidate_dirs = []
            for backup_dir_name in BACKUP_DIR_NAMES:
                candidate_dirs.extend(
                    backup_dir
                    for backup_dir in sorted(path.rglob(backup_dir_name))
                    if backup_dir.is_dir()
                )
                candidate_dirs.append(path / backup_dir_name)
        else:
            continue

        for backup_dir in candidate_dirs:
            if not backup_dir.is_dir():
                continue
            resolved_dir = backup_dir.resolve()
            if resolved_dir not in seen_dirs:
                backup_dirs.append(backup_dir)
                seen_dirs.add(resolved_dir)
    return backup_dirs


def collect_removed_config_files(input_paths):
    removed_config_files = []
    seen_files = set()
    for backup_dir in collect_backup_dirs(input_paths):
        for removed_config_file in sorted(backup_dir.glob("*.removed_configs")):
            resolved_file = removed_config_file.resolve()
            if resolved_file not in seen_files:
                removed_config_files.append(removed_config_file)
                seen_files.add(resolved_file)
    return removed_config_files


def add_unique_configs(configs, seen_configs, lines):
    added_count = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and stripped_line not in seen_configs:
            configs.append(stripped_line)
            seen_configs.add(stripped_line)
            added_count += 1
    return added_count


def merge_removed_configs(input_paths, output_path, extra_removed_lines=None, dry_run=False):
    removed_config_files = collect_removed_config_files(input_paths)
    merged_configs = []
    seen_configs = set()

    print_section("Merge Removed Configs")
    if dry_run:
        print("Mode: dry-run (no files will be modified)")

    for index, removed_config_file in enumerate(removed_config_files, start=1):
        try:
            lines = removed_config_file.read_text(encoding="utf-8").splitlines()
            added_count = add_unique_configs(merged_configs, seen_configs, lines)
            print(f"[{index}/{len(removed_config_files)}] {removed_config_file}")
            print(f"  lines: {len(lines)}, new unique: {added_count}")
        except Exception as err:
            print(f"[{index}/{len(removed_config_files)}] {removed_config_file}")
            print("  status: error")
            print(f"  error: {err}")
            continue

    if extra_removed_lines:
        added_count = add_unique_configs(merged_configs, seen_configs, extra_removed_lines)
        print("[current run]")
        print(f"  lines: {len(extra_removed_lines)}, new unique: {added_count}")

    output_file = Path(output_path)
    output_content = "\n".join(merged_configs)
    if output_content:
        output_content += "\n"

    print_section("Summary")
    if dry_run:
        print(f"Removed-config backups: {len(removed_config_files)}")
        print(f"Unique configs to merge: {len(merged_configs)}")
        print(f"Output: {output_file} (dry-run)")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_content, encoding="utf-8")
    print(f"Removed-config backups: {len(removed_config_files)}")
    print(f"Unique configs merged: {len(merged_configs)}")
    print(f"Output: {output_file}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="移除指定配置工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s -i input.txt -r remove_configs.txt                    # 从单文件删除配置
  python %(prog)s -i file1.txt file2.txt -r remove_configs.txt          # 从多文件删除配置
  python %(prog)s -i ./configs/ -r remove_configs.txt                   # 从文件夹删除配置
  python %(prog)s -i input.txt -r remove1.txt remove2.txt remove_dir/   # 合并多个删除配置集
  python %(prog)s -i configs/ -r remove_configs.txt -n                  # 预览删除与备份
  python %(prog)s -i configs/ -m merged.txt                             # 合并历史剔除配置
  python %(prog)s -i configs/ -R                                         # 按最新备份回退
  python %(prog)s -c configs/.rm_config_backups                         # 精简备份目录
  python %(prog)s -i input.txt -r remove_configs.txt --no-backup        # 不创建备份
注意: 所有删除操作会原地修改文件。默认仅在文件被修改时，将带时间戳的 .backup 原文件备份和 .removed_configs 剔除配置备份写入同目录下的 .rm_config_backups/。
        """,
    )
    parser.add_argument("-i", "--input", nargs="+", help="待处理的文件或目录")
    parser.add_argument("-r", "--remove", nargs="+", help="包含要删除配置的文件或目录")
    parser.add_argument(
        "-m",
        "--merge",
        metavar="OUTPUT",
        help="合并 input 对应备份目录下多次产生的 .removed_configs 到指定文件",
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="仅预览将要执行的修改和输出")
    parser.add_argument(
        "-R", "--revert", action="store_true", help="按最新 .backup 回退 input 文件"
    )
    parser.add_argument("-c", "--compact-backups", metavar="DIR", help="精简指定备份目录")
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="不创建备份文件",
    )
    args = parser.parse_args(argv)
    if args.compact_backups and (args.input or args.remove or args.merge or args.revert):
        parser.error("-c/--compact-backups cannot be used with -i, -r, -m, or -R")
    if args.revert and (args.remove or args.merge):
        parser.error("-R/--revert cannot be used with -r/--remove or -m/--merge")
    if not args.compact_backups and not args.input:
        parser.error("-i/--input is required unless using -c/--compact-backups")
    if not args.remove and not args.merge and not args.revert and not args.compact_backups:
        parser.error(
            "at least one of -r/--remove, -m/--merge, -R/--revert, "
            "or -c/--compact-backups is required"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.compact_backups:
        compact_backup_dir(args.compact_backups, dry_run=args.dry_run)
        return
    if args.revert:
        revert_files(args.input, dry_run=args.dry_run)
        return

    removed_lines = []
    if args.remove:
        removed_lines = remove_configs_from_files(
            args.input,
            args.remove,
            backup=args.backup,
            dry_run=args.dry_run,
        )
    if args.merge:
        merge_removed_configs(
            args.input,
            args.merge,
            extra_removed_lines=removed_lines,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
