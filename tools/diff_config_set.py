# 对比 API 配置集合小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_LEFT_PATH = Path("tester/api_config/test_log_cinn_filtered/pass_config.txt")
DEFAULT_RIGHT_PATH = Path("tester/api_config/test_log_cinn/api_config_pass.txt")


def load_config_set(config_file):
    path = Path(config_file)
    content = path.read_text(encoding="utf-8")
    return {line.strip() for line in content.splitlines() if line.strip()}


def diff_config_sets(left_path=DEFAULT_LEFT_PATH, right_path=DEFAULT_RIGHT_PATH):
    left_configs = load_config_set(left_path)
    right_configs = load_config_set(right_path)

    removed_configs = left_configs - right_configs
    added_configs = right_configs - left_configs

    print(f"left configs: {len(left_configs)}")
    print(f"right configs: {len(right_configs)}")
    print(f"removed configs: {len(removed_configs)}")
    for config in sorted(removed_configs):
        print(config)

    print(f"added configs: {len(added_configs)}")
    for config in sorted(added_configs):
        print(config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对比两个 API 配置集合")
    parser.add_argument(
        "--left",
        "-l",
        type=Path,
        default=DEFAULT_LEFT_PATH,
        help="基准配置文件路径",
    )
    parser.add_argument(
        "--right",
        "-r",
        type=Path,
        default=DEFAULT_RIGHT_PATH,
        help="对比配置文件路径",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    diff_config_sets(args.left, args.right)


if __name__ == "__main__":
    main()
