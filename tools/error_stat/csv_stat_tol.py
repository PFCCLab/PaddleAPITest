# 整理 tol_*.csv 精度统计数据，产出：tol_full.csv、tol_stat.csv、tol_stat_api.csv
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_TEST_LOG_PATH = Path("tester/api_config/test_log")
GENERATED_FILES = {"tol_stat.csv", "tol_stat_api.csv", "tol_full.csv"}
NUMERIC_COLUMNS = ["max_abs_diff", "max_rel_diff"]
CUSTOM_OP_API = "paddle._C_ops._run_custom_op"
CUSTOM_OP_PATTERN = re.compile(rf"^{re.escape(CUSTOM_OP_API)}\(\s*(['\"])(.*?)\1")


def get_api_key(api, config=None):
    if not isinstance(api, str):
        return api

    match = CUSTOM_OP_PATTERN.match(api)
    if not match and api == CUSTOM_OP_API and isinstance(config, str):
        match = CUSTOM_OP_PATTERN.match(config)
    if not match:
        return api

    op_name = match.group(2)
    return f'{CUSTOM_OP_API}("{op_name}")'


def collect_csv_files(input_path):
    file_pattern = input_path / "tol*.csv"
    file_list = sorted(file_pattern.parent.glob(file_pattern.name))
    file_list = [file_path for file_path in file_list if file_path.name not in GENERATED_FILES]
    if not file_list:
        print(f"No files found matching pattern {file_pattern}")
    return file_list


def load_tol_data(file_list):
    dfs = []
    stats = defaultdict(lambda: defaultdict(list))
    api_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    config_count = 0

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path, on_bad_lines="warn")
            df["API"] = [get_api_key(api, config) for api, config in zip(df["API"], df["config"])]
            dfs.append(df)
            print(f"Read {len(df)} configs in {file_path}")
            config_count += len(df)

            for _, row in df.iterrows():
                api = row["API"]
                dtype = row["dtype"]
                mode = row["mode"]
                max_abs_diff = row["max_abs_diff"]
                max_rel_diff = row["max_rel_diff"]
                stats[(api, dtype, mode)]["abs_diffs"].append(max_abs_diff)
                stats[(api, dtype, mode)]["rel_diffs"].append(max_rel_diff)
                api_stats[api][dtype][mode] += 1
        except Exception as err:
            print(f"Error processing file {file_path}: {err}")

    print(f"\nTotal read {len(stats)} (api, dtype, mode)s, {config_count} configs.")
    return dfs, stats, api_stats


def write_full_csv(dfs, output_path):
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["config", "mode"], keep="last")
    merged_df = merged_df.sort_values(by=["API", "dtype", "config", "mode"], ignore_index=True)
    for col in NUMERIC_COLUMNS:
        merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.6e}")

    output_file = output_path / "tol_full.csv"
    merged_df.to_csv(output_file, index=False, na_rep="nan")


def write_stat_csv(stats, output_path):
    stats_data = []
    for api, dtype, mode in sorted(stats.keys()):
        values = stats[(api, dtype, mode)]
        abs_diffs = values["abs_diffs"]
        rel_diffs = values["rel_diffs"]

        abs_min = min(abs_diffs)
        abs_max = max(abs_diffs)
        abs_mean = sum(abs_diffs) / len(abs_diffs)
        rel_min = min(rel_diffs)
        rel_max = max(rel_diffs)
        rel_mean = sum(rel_diffs) / len(rel_diffs)
        count = len(abs_diffs)

        stats_data.append(
            {
                "API": api,
                "dtype": dtype,
                "mode": mode,
                "abs_min": f"{abs_min:.6e}",
                "abs_max": f"{abs_max:.6e}",
                "abs_mean": f"{abs_mean:.6e}",
                "rel_min": f"{rel_min:.6e}",
                "rel_max": f"{rel_max:.6e}",
                "rel_mean": f"{rel_mean:.6e}",
                "count": count,
            }
        )

    if stats_data:
        df = pd.DataFrame(stats_data)
        output_file = output_path / "tol_stat.csv"
        df.to_csv(output_file, index=False, na_rep="nan")
        print(f"\nStatistics saved to {output_file}")
        print("Sample of the results:")
        print(df.head())
    else:
        print("No data to process.")


def write_api_stat_csv(api_stats, output_path):
    api_stats_data = []
    for api in sorted(api_stats.keys()):
        api_dtype = api_stats[api]
        dtypes = "/".join(sorted(api_dtype.keys()))
        total = sum(
            api_dtype[dtype]["forward"] + api_dtype[dtype]["backward"] for dtype in api_dtype
        )

        api_stats_data.append(
            {
                "API": api,
                "dtype": "dtypes:" + dtypes,
                "mode": "modes:forward/backward",
                "count": total,
                "percentage": 100.0,
            }
        )

        forward_dtypes = []
        forward_total = 0
        backward_dtypes = []
        backward_total = 0
        for dtype, modes in api_dtype.items():
            if "forward" in modes:
                forward_dtypes.append(dtype)
                forward_total += modes["forward"]
            if "backward" in modes:
                backward_dtypes.append(dtype)
                backward_total += modes["backward"]
        forward_dtypes = "/".join(sorted(forward_dtypes))
        backward_dtypes = "/".join(sorted(backward_dtypes))

        api_stats_data.append(
            {
                "API": api,
                "dtype": "dtypes:" + forward_dtypes,
                "mode": "forward",
                "count": forward_total,
                "percentage": round(forward_total / total * 100, 2),
            }
        )
        api_stats_data.append(
            {
                "API": api,
                "dtype": "dtypes:" + backward_dtypes,
                "mode": "backward",
                "count": backward_total,
                "percentage": round(backward_total / total * 100, 2),
            }
        )

        for dtype in sorted(api_dtype.keys()):
            for mode in ["forward", "backward"]:
                count = api_dtype[dtype][mode]
                if count > 0:
                    api_stats_data.append(
                        {
                            "API": api,
                            "dtype": dtype,
                            "mode": mode,
                            "count": count,
                            "percentage": round(count / total * 100, 2),
                        }
                    )

    if api_stats_data:
        df = pd.DataFrame(api_stats_data)
        output_file = output_path / "tol_stat_api.csv"
        df.to_csv(output_file, index=False, na_rep="nan")
        print(f"\nAPI statistics saved to {output_file}")
        print("Sample of API statistics:")
        print(df.head())
    else:
        print("No API statistics to process.")


def run_tol_stat(input_path=DEFAULT_TEST_LOG_PATH, output_path=None):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    file_list = collect_csv_files(input_path)
    if not file_list:
        return

    dfs, stats, api_stats = load_tol_data(file_list)
    if not stats:
        return

    write_full_csv(dfs, output_path)
    write_stat_csv(stats, output_path)
    write_api_stat_csv(api_stats, output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="整理 tol_*.csv 精度统计数据")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_TEST_LOG_PATH),
        help="输入路径，包含 tol*.csv 文件",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径（默认同输入路径）")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_tol_stat(args.input, args.output)


if __name__ == "__main__":
    main()
