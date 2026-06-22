# 从 CSV 中按 API 提取 case 小工具
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("TotalStableFull.csv")
DEFAULT_OUTPUT_DIR = Path(".")
DIFF_THRESHOLD = 1e-16


def _output_paths(api_name, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return (
        output_path / f"filtered_result_{api_name}.csv",
        output_path / f"error_config_{api_name}.txt",
    )


def extract_cases_for_api(
    api_name, only_diff=False, config_path=DEFAULT_CONFIG_PATH, output_dir=DEFAULT_OUTPUT_DIR
):
    filtered_file, error_config_file = _output_paths(api_name, output_dir)

    with (
        Path(config_path).open(newline="") as infile,
        filtered_file.open("w", newline="") as outfile,
    ):
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            first_col = row[0]
            if api_name not in first_col:
                continue

            last_col = float(row[-1]) if row[-1].strip() else 0
            second_last_col = float(row[-2]) if row[-2].strip() else 0
            if only_diff and last_col < DIFF_THRESHOLD and second_last_col < DIFF_THRESHOLD:
                continue

            writer.writerow(row)

    configs = set()
    with filtered_file.open(newline="") as infile:
        reader = csv.reader(infile)
        next(reader)
        for row in reader:
            configs.add(row[2].replace('""', '"'))

    error_config_file.write_text("\n".join(configs), encoding="utf-8")


def run_extract_cases(
    api_names, only_diff=False, config_path=DEFAULT_CONFIG_PATH, output_dir=DEFAULT_OUTPUT_DIR
):
    for api_name in api_names:
        extract_cases_for_api(api_name, only_diff, config_path, output_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="从 CSV 中按 API 提取 case")
    parser.add_argument("api_names", nargs="+", help="需要提取的 API 名称列表")
    parser.add_argument(
        "--only-diff",
        action="store_true",
        help="仅保留最后两列误差不同时为 0 的记录",
    )
    parser.add_argument(
        "--config-path",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录路径",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_extract_cases(args.api_names, args.only_diff, args.config_path, args.output_dir)


if __name__ == "__main__":
    main()
