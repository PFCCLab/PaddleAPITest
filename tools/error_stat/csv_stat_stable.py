# 整理 stable*.csv 精度统计数据，产出：stable_full.csv、stable_stat.csv、stable_stat_api.csv
# @author: cangtianhuang
# @date: 2025-07-30
from __future__ import annotations

import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = TEST_LOG_PATH
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 查找所有stable*.csv文件
file_pattern = TEST_LOG_PATH / "stable*.csv"
file_list = glob.glob(str(file_pattern))
file_list = [
    f
    for f in file_list
    if Path(f).name not in ["stable_stat.csv", "stable_stat_api.csv", "stable_full.csv"]
]
file_list.sort()
if not file_list:
    print(f"No files found matching pattern {file_pattern}")
    exit(0)


def list_defaultdict_factory():
    return defaultdict(list)


def int_defaultdict_factory():
    return defaultdict(int)


def nested_int_defaultdict_factory():
    return defaultdict(int_defaultdict_factory)


# 读取并处理每个文件
def process_chunk(chunk):
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


# 并行处理 CSV 文件
def parallel_process_csv(file_path, chunk_size=2000000):
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
    except Exception as e:
        print(f"Error reading file {file_path} for merging: {e}")
        return stats, api_stats, config_count, chunks
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks_iterator]

        for future in futures:
            chunk_stats, chunk_api_stats, chunk = future.result()
            chunks.append(chunk)
            config_count += len(chunk)
            for key in chunk_stats:
                stats[key]["abs_diffs"].extend(chunk_stats[key]["abs_diffs"])
                stats[key]["rel_diffs"].extend(chunk_stats[key]["rel_diffs"])
            for api in chunk_api_stats:
                for dtype in chunk_api_stats[api]:
                    for comp in chunk_api_stats[api][dtype]:
                        api_stats[api][dtype][comp] += chunk_api_stats[api][dtype][comp]

    print(f"Read {config_count} configs in {file_path}")
    return stats, api_stats, config_count, chunks


stats = defaultdict(list_defaultdict_factory)
api_stats = defaultdict(nested_int_defaultdict_factory)
config_count = 0
dfs = []
for file_path in file_list:
    # 并行处理统计数据
    file_stats, file_api_stats, file_config_count, file_chunks = parallel_process_csv(file_path)
    dfs.extend(file_chunks)
    config_count += file_config_count
    for key in file_stats:
        stats[key]["abs_diffs"].extend(file_stats[key]["abs_diffs"])
        stats[key]["rel_diffs"].extend(file_stats[key]["rel_diffs"])
    for api in file_api_stats:
        for dtype in file_api_stats[api]:
            for comp in file_api_stats[api][dtype]:
                api_stats[api][dtype][comp] += file_api_stats[api][dtype][comp]

print(f"\nTotal read {len(stats)} (api, dtype, comp)s, {config_count} configs.")
if not stats:
    print("No data to process.")
    exit(0)

# 合并所有DataFrame并保存
merged_df = pd.concat(dfs, ignore_index=True)
numeric_cols = ["max_abs_diff", "max_rel_diff"]
merged_df = merged_df.groupby(["API", "dtype", "config", "comp"], as_index=False)[
    numeric_cols
].mean()
# merged_df = merged_df.drop_duplicates(subset=["config", "comp"], keep="last")
merged_df = merged_df.sort_values(by=["API", "dtype", "config", "comp"], ignore_index=True)
for col in numeric_cols:
    merged_df[col] = merged_df[col].apply(lambda x: f"{float(x):.6e}")
output_file = OUTPUT_PATH / "stable_full.csv"
merged_df.to_csv(output_file, index=False, na_rep="nan")

# 准备结果数据
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

# 转换为DataFrame并保存
if stats_data:
    df = pd.DataFrame(stats_data)
    output_file = OUTPUT_PATH / "stable_stat.csv"
    df.to_csv(output_file, index=False, na_rep="nan")
    print(f"\nStatistics saved to {output_file}")
    print("Sample of the results:")
    print(df.head())
else:
    print("No data to process.")

# 准备统计数据
api_stats_data = []
for api in sorted(api_stats.keys()):
    api_dtype = api_stats[api]
    dtypes = "/".join(sorted(api_dtype.keys()))
    total = sum(api_dtype[dtype][comp] for dtype in api_dtype for comp in api_dtype[dtype])
    all_comps = "/".join(sorted({comp for dtype in api_dtype for comp in api_dtype[dtype]}))

    # 统计所有 comp 的模式
    api_stats_data.append(
        {
            "API": api,
            "dtype": "dtypes:" + dtypes,
            "comp": f"comps:{all_comps}",
            "count": total,
            "percentage": 100.0,
        }
    )

    # 按 comp 统计
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

    # 按 dtype 和 comp 统计
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

# 转换为DataFrame并保存
if api_stats_data:
    df = pd.DataFrame(api_stats_data)
    output_file = OUTPUT_PATH / "stable_stat_api.csv"
    df.to_csv(output_file, index=False, na_rep="nan")
    print(f"\nAPI statistics saved to {output_file}")
    print("Sample of API statistics:")
    print(df.head())
else:
    print("No API statistics to process.")
