# 重测配置移除小工具
# @author: cangtianhuang
# @date: 2025-11-11
from __future__ import annotations

import argparse
import os
from pathlib import Path

LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "skip": "api_config_skip",
    "paddle_error": "api_config_paddle_error",
    "paddle_accuracy": "api_config_paddle_accuracy",
    "paddle_bitwise": "api_config_paddle_bitwise",
    "paddle_cuda": "api_config_paddle_cuda",
    "paddle_crash": "api_config_paddle_crash",
    "oom": "api_config_oom",
    "timeout": "api_config_timeout",
    "torch_error": "api_config_torch_error",
    "config_input": "api_config_config_input",
    "config_parse": "api_config_config_parse",
    "config_convert": "api_config_config_convert",
}


def remove_configs(log_path, to_remove):
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"{log_path} not exists", flush=True)
        return

    checkpoint_configs = set()
    checkpoint_file = log_path / "checkpoint.txt"
    if not checkpoint_file.exists():
        print("No checkpoint file found", flush=True)
        return

    try:
        with checkpoint_file.open("r") as f:
            checkpoint_configs = {line.strip() for line in f if line.strip()}
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
        return
    print(f"Read {len(checkpoint_configs)} api configs from checkpoint", flush=True)

    retest_configs = set()
    for log_type in to_remove:
        if log_type not in LOG_PREFIXES:
            print(f"Invalid log type: {log_type}", flush=True)
            continue
        prefix = LOG_PREFIXES[log_type]
        log_file = log_path / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            with log_file.open("r") as f:
                lines = {line.strip() for line in f if line.strip()}
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
            with checkpoint_file.open("w") as f:
                f.writelines(f"{line}\n" for line in sorted(checkpoint_configs))
        except Exception as err:
            print(f"Error writing {checkpoint_file}: {err}", flush=True)
            return
    else:
        print("No retest configs found", flush=True)

    for log_type in to_remove:
        if log_type not in LOG_PREFIXES:
            continue
        prefix = LOG_PREFIXES[log_type]
        log_file = log_path / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            os.remove(log_file)
        except Exception as err:
            print(f"Error removing {log_file}: {err}", flush=True)
            return


def main():
    default_log_path = "tester/api_config/test_log"

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
        type=str,
        default=default_log_path,
        help="测试日志目录路径",
    )
    parser.add_argument(
        "--remove",
        "-r",
        nargs="+",
        help="指定需要移除的配置",
    )
    args = parser.parse_args()
    remove_configs(args.path, args.remove)


if __name__ == "__main__":
    main()
