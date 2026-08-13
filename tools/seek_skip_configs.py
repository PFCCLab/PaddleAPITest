# 筛选 skip 配置小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TEST_LOG_PATH = Path("tester/api_config/test_log")
DEFAULT_OUTPUT_FILE_NAME = "api_config_skip.txt"
LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "skip": "api_config_skip",
    "paddle_error": "api_config_paddle_error",
    "paddle_accuracy": "api_config_paddle_accuracy",
    "paddle_bitwise": "api_config_paddle_bitwise",
    "paddle_bitwise_knows": "api_config_paddle_bitwise_knows",
    "paddle_cuda": "api_config_paddle_cuda",
    "paddle_crash": "api_config_paddle_crash",
    "oom": "api_config_oom",
    "timeout": "api_config_timeout",
    "torch_error": "api_config_torch_error",
    "config_input": "api_config_config_input",
    "config_parse": "api_config_config_parse",
    "config_convert": "api_config_config_convert",
}


def read_config_set(config_file):
    with Path(config_file).open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def write_config_set(config_file, configs):
    with Path(config_file).open("w", encoding="utf-8") as f:
        f.writelines(f"{line}\n" for line in sorted(configs))


def backup_file(config_file):
    path = Path(config_file)
    backup_path = path.with_suffix(path.suffix + ".backup")
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Created backup: {backup_path}", flush=True)


def seek_skip_configs(
    test_log_path=DEFAULT_TEST_LOG_PATH, output_file=None, update_checkpoint=True, backup=True
):
    test_log_path = Path(test_log_path)
    output_path = (
        Path(output_file) if output_file is not None else test_log_path / DEFAULT_OUTPUT_FILE_NAME
    )

    log_counts = {}
    checkpoint_file = test_log_path / "checkpoint.txt"
    if not checkpoint_file.exists():
        print("No checkpoint file found", flush=True)
        return

    try:
        checkpoint_configs = read_config_set(checkpoint_file)
        log_counts["checkpoint"] = len(checkpoint_configs)
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
        return
    print(f"Read {len(checkpoint_configs)} api configs from checkpoint", flush=True)

    api_configs = checkpoint_configs.copy()
    for log_type, prefix in LOG_PREFIXES.items():
        if log_type == "checkpoint":
            continue
        log_file = test_log_path / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            lines = read_config_set(log_file)
            api_configs -= lines
            log_counts[log_type] = len(lines)
        except Exception as err:
            print(f"Error reading {log_file}: {err}", flush=True)
            return

    if api_configs:
        log_counts["skip"] = len(api_configs)
    else:
        print("No skip configs found", flush=True)

    for log_type, count in log_counts.items():
        print(f"{log_type}: {count}", flush=True)

    if not api_configs:
        return

    try:
        if output_path.exists() and backup:
            backup_file(output_path)
        write_config_set(output_path, api_configs)
    except Exception as err:
        print(f"Error writing to {output_path}: {err}", flush=True)
        return
    print(f"Write {len(api_configs)} skip api configs to {output_path}", flush=True)

    if not update_checkpoint:
        return

    checkpoint_count = len(checkpoint_configs)
    checkpoint_configs -= api_configs
    print(f"checkpoint removed: {checkpoint_count - len(checkpoint_configs)}", flush=True)
    print(f"checkpoint remaining: {len(checkpoint_configs)}", flush=True)
    try:
        if backup:
            backup_file(checkpoint_file)
        write_config_set(checkpoint_file, checkpoint_configs)
    except Exception as err:
        print(f"Error writing {checkpoint_file}: {err}", flush=True)
        return
    print(
        f"Write {len(checkpoint_configs)} checkpoint api configs to {checkpoint_file}",
        flush=True,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="筛选 skip 配置小工具")
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=DEFAULT_TEST_LOG_PATH,
        help="测试日志目录路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="skip 配置输出文件路径（默认写入日志目录 api_config_skip.txt）",
    )
    parser.add_argument(
        "--no-update-checkpoint",
        action="store_false",
        dest="update_checkpoint",
        help="只输出 skip 配置，不修改 checkpoint.txt",
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
    seek_skip_configs(args.path, args.output, args.update_checkpoint, args.backup)


if __name__ == "__main__":
    main()
