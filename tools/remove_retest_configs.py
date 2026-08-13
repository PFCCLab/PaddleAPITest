# 重测配置移除小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LOG_PATH = Path("tester/api_config/test_log")
LOG_PREFIXES = {
    # 独立保留 known 清单，支持针对性复测和清理。
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
DEFAULT_REMOVE_TYPES = ["timeout", "oom", "skip"]


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


def remove_configs(log_path=DEFAULT_LOG_PATH, to_remove=None, backup=True):
    log_path = Path(log_path)
    if to_remove is None:
        to_remove = DEFAULT_REMOVE_TYPES
    if not log_path.exists():
        print(f"{log_path} not exists", flush=True)
        return

    checkpoint_file = log_path / "checkpoint.txt"
    if not checkpoint_file.exists():
        print("No checkpoint file found", flush=True)
        return

    try:
        checkpoint_configs = read_config_set(checkpoint_file)
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
        return
    print(f"Read {len(checkpoint_configs)} api configs from checkpoint", flush=True)

    retest_configs = set()
    valid_remove_types = []
    for log_type in to_remove:
        if log_type not in LOG_PREFIXES:
            print(f"Invalid log type: {log_type}", flush=True)
            continue
        valid_remove_types.append(log_type)
        prefix = LOG_PREFIXES[log_type]
        log_file = log_path / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            lines = read_config_set(log_file)
            retest_configs.update(lines)
            print(f"Read {len(lines)} api configs from {log_file}", flush=True)
        except Exception as err:
            print(f"Error reading {log_file}: {err}", flush=True)
            return

    if retest_configs:
        checkpoint_count = len(checkpoint_configs)
        checkpoint_configs -= retest_configs
        print(
            f"checkpoint removed: {checkpoint_count - len(checkpoint_configs)}",
            flush=True,
        )
        print(f"checkpoint remaining: {len(checkpoint_configs)}", flush=True)
        try:
            if backup:
                backup_file(checkpoint_file)
            write_config_set(checkpoint_file, checkpoint_configs)
        except Exception as err:
            print(f"Error writing {checkpoint_file}: {err}", flush=True)
            return
    else:
        print("No retest configs found", flush=True)

    for log_type in valid_remove_types:
        prefix = LOG_PREFIXES[log_type]
        log_file = log_path / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            if backup:
                backup_file(log_file)
            log_file.unlink()
        except Exception as err:
            print(f"Error removing {log_file}: {err}", flush=True)
            return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="重测配置移除小工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s --path tester/api_config/test_log # 指定测试日志路径
  python %(prog)s --remove timeout oom skip         # 指定需要移除的配置
支持移除的配置集合:
  pass              - api_config_pass
  skip              - api_config_skip
  paddle_error      - api_config_paddle_error
  paddle_accuracy   - api_config_paddle_accuracy
  paddle_bitwise    - api_config_paddle_bitwise
  paddle_bitwise_knows - api_config_paddle_bitwise_knows
  paddle_cuda       - api_config_paddle_cuda
  paddle_crash      - api_config_paddle_crash
  oom               - api_config_oom
  timeout           - api_config_timeout
  torch_error       - api_config_torch_error
  config_input      - api_config_config_input
  config_parse      - api_config_config_parse
  config_convert    - api_config_config_convert
        """,
    )
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="测试日志目录路径",
    )
    parser.add_argument(
        "--remove",
        "-r",
        nargs="+",
        default=DEFAULT_REMOVE_TYPES,
        help="指定需要移除的配置",
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
    remove_configs(args.path, args.remove, args.backup)


if __name__ == "__main__":
    main()
