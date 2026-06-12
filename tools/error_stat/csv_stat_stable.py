# 整理 stable*.csv 精度统计数据，产出：stable_full.csv、stable_stat.csv、stable_stat_api.csv
# @author: cangtianhuang
# @date: 2026-06-11

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TEST_LOG_PATH = Path("tester/api_config/test_log")
GENERATED_FILES = {"stable_stat.csv", "stable_stat_api.csv", "stable_full.csv"}
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


def list_defaultdict_factory():
    return defaultdict(list)


def int_defaultdict_factory():
    return defaultdict(int)


def nested_int_defaultdict_factory():
    return defaultdict(int_defaultdict_factory)


def collect_csv_files(input_path):
    file_pattern = input_path / "stable*.csv"
    file_list = sorted(file_pattern.parent.glob(file_pattern.name))
    file_list = [file_path for file_path in file_list if file_path.name not in GENERATED_FILES]
    if not file_list:
        print(f"No files found matching pattern {file_pattern}")
    return file_list


def process_chunk(chunk):
    chunk["API"] = [get_api_key(api, config) for api, config in zip(chunk["API"], chunk["config"])]
    stats = defaultdict(list_defaultdict_factory)
    api_stats = defaultdict(nested_int_defaultdict_factory)
    for _, row in chunk.iterrows():
        api = row["API"]
        dtype = row["dtype"]
        comp = row["comp"]
        max_abs_diff = row["max_abs_diff"]
        max_rel_diff = row["max_rel_diff"]

        if np.isinf(max_rel_diff):
            max_rel_diff = max_abs_diff

        stats[(api, dtype, comp)]["abs_diffs"].append(max_abs_diff)
        stats[(api, dtype, comp)]["rel_diffs"].append(max_rel_diff)
        api_stats[api][dtype][comp] += 1
    return stats, api_stats, chunk


def merge_stats(target_stats, source_stats):
    for key in source_stats:
        target_stats[key]["abs_diffs"].extend(source_stats[key]["abs_diffs"])
        target_stats[key]["rel_diffs"].extend(source_stats[key]["rel_diffs"])


def merge_api_stats(target_api_stats, source_api_stats):
    for api in source_api_stats:
        for dtype in source_api_stats[api]:
            for comp in source_api_stats[api][dtype]:
                target_api_stats[api][dtype][comp] += source_api_stats[api][dtype][comp]


def parallel_process_csv(file_path, chunk_size=2000000, max_workers=10):
    stats = defaultdict(list_defaultdict_factory)
    api_stats = defaultdict(nested_int_defaultdict_factory)
    chunks = []
    config_count = 0

    try:
        chunks_iterator = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            on_bad_lines="warn",
            dtype={"max_abs_diff": float, "max_rel_diff": float},
        )
    except Exception as err:
        print(f"Error reading file {file_path} for merging: {err}")
        return stats, api_stats, config_count, chunks

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks_iterator]

        for future in futures:
            chunk_stats, chunk_api_stats, chunk = future.result()
            chunks.append(chunk)
            config_count += len(chunk)
            merge_stats(stats, chunk_stats)
            merge_api_stats(api_stats, chunk_api_stats)

    print(f"Read {config_count} configs in {file_path}")
    return stats, api_stats, config_count, chunks


def load_stable_data(file_list, chunk_size=2000000, max_workers=10):
    stats = defaultdict(list_defaultdict_factory)
    api_stats = defaultdict(nested_int_defaultdict_factory)
    config_count = 0
    dfs = []

    for file_path in file_list:
        file_stats, file_api_stats, file_config_count, file_chunks = parallel_process_csv(
            file_path,
            chunk_size=chunk_size,
            max_workers=max_workers,
        )
        dfs.extend(file_chunks)
        config_count += file_config_count
        merge_stats(stats, file_stats)
        merge_api_stats(api_stats, file_api_stats)

    print(f"\nTotal read {len(stats)} (api, dtype, comp)s, {config_count} configs.")
    return dfs, stats, api_stats


def write_full_csv(dfs, output_path):
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.groupby(["API", "dtype", "config", "comp"], as_index=False)[
        NUMERIC_COLUMNS
    ].mean()
    merged_df = merged_df.sort_values(by=["API", "dtype", "config", "comp"], ignore_index=True)
    for col in NUMERIC_COLUMNS:
        merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.6e}")

    output_file = output_path / "stable_full.csv"
    merged_df.to_csv(output_file, index=False, na_rep="nan")


def write_stat_csv(stats, output_path):
    stats_data = []
    for api, dtype, comp in sorted(stats.keys()):
        abs_diffs = np.array(stats[(api, dtype, comp)]["abs_diffs"], dtype=np.float64)
        rel_diffs = np.array(stats[(api, dtype, comp)]["rel_diffs"], dtype=np.float64)

        count = len(abs_diffs)

        if not np.any(np.isnan(abs_diffs)):
            abs_quantile = np.quantile(abs_diffs, 0.99)
            filtered_abs = abs_diffs[abs_diffs <= abs_quantile]
            abs_diffs = filtered_abs if len(filtered_abs) > 0 else abs_diffs

        if not np.any(np.isnan(rel_diffs)):
            rel_quantile = np.quantile(rel_diffs, 0.99)
            filtered_rel = rel_diffs[rel_diffs <= rel_quantile]
            rel_diffs = filtered_rel if len(filtered_rel) > 0 else rel_diffs

        stats_data.append(
            {
                "API": api,
                "dtype": dtype,
                "comp": comp,
                "abs_min": f"{np.min(abs_diffs):.6e}",
                "abs_max": f"{np.max(abs_diffs):.6e}",
                "abs_mean": f"{np.mean(abs_diffs):.6e}",
                "rel_min": f"{np.min(rel_diffs):.6e}",
                "rel_max": f"{np.max(rel_diffs):.6e}",
                "rel_mean": f"{np.mean(rel_diffs):.6e}",
                "count": count,
            }
        )

    if stats_data:
        df = pd.DataFrame(stats_data)
        output_file = output_path / "stable_stat.csv"
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
        total = sum(api_dtype[dtype][comp] for dtype in api_dtype for comp in api_dtype[dtype])
        all_comps = "/".join(sorted({comp for dtype in api_dtype for comp in api_dtype[dtype]}))

        api_stats_data.append(
            {
                "API": api,
                "dtype": "dtypes:" + dtypes,
                "comp": f"comps:{all_comps}",
                "count": total,
                "percentage": 100.0,
            }
        )

        comp_counts = defaultdict(int)
        for dtype in api_dtype:
            for comp in api_dtype[dtype]:
                comp_counts[comp] += api_dtype[dtype][comp]

        for comp in sorted(comp_counts.keys()):
            comp_total = comp_counts[comp]
            comp_dtypes = "/".join(sorted(dtype for dtype in api_dtype if comp in api_dtype[dtype]))
            api_stats_data.append(
                {
                    "API": api,
                    "dtype": "dtypes:" + comp_dtypes,
                    "comp": comp,
                    "count": comp_total,
                    "percentage": round(comp_total / total * 100, 2),
                }
            )

        for dtype in sorted(api_dtype.keys()):
            for comp in sorted(api_dtype[dtype].keys()):
                count = api_dtype[dtype][comp]
                if count > 0:
                    api_stats_data.append(
                        {
                            "API": api,
                            "dtype": dtype,
                            "comp": comp,
                            "count": count,
                            "percentage": round(count / total * 100, 2),
                        }
                    )

    if api_stats_data:
        df = pd.DataFrame(api_stats_data)
        output_file = output_path / "stable_stat_api.csv"
        df.to_csv(output_file, index=False, na_rep="nan")
        print(f"\nAPI statistics saved to {output_file}")
        print("Sample of API statistics:")
        print(df.head())
    else:
        print("No API statistics to process.")


def run_stable_stat(
    input_path=DEFAULT_TEST_LOG_PATH, output_path=None, chunk_size=2000000, max_workers=10
):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    file_list = collect_csv_files(input_path)
    if not file_list:
        return

    dfs, stats, api_stats = load_stable_data(
        file_list,
        chunk_size=chunk_size,
        max_workers=max_workers,
    )
    if not stats:
        print("No data to process.")
        return

    write_full_csv(dfs, output_path)
    write_stat_csv(stats, output_path)
    write_api_stat_csv(api_stats, output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="整理 stable*.csv 精度统计数据")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_TEST_LOG_PATH),
        help="输入路径，包含 stable*.csv 文件",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径（默认同输入路径）")
    parser.add_argument("--chunk-size", type=int, default=2000000, help="CSV 分块读取行数")
    parser.add_argument("--max-workers", type=int, default=10, help="并行处理进程数")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_stable_stat(
        args.input,
        args.output,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
