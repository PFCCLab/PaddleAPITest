# test_log 一键整理小工具
# @author: cangtianhuang
# @date: 2026-06-11

# 整理效果：pass + skip + Paddle 问题 + 可重测 + 测试侧问题（可按终态类型拆分）
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

DEFAULT_TEST_LOG_PATH = Path("tester/api_config/test_log_big_tensor")
RESULT_DIR_NAME = "error_stat_result"

# 终态分类到结果文件前缀的映射，与 tester/api_config/log_writer.py 中 LOG_PREFIXES 保持一致。
# checkpoint 仅用于读取已完成 case 集合，不作为统计分类输出。
LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "skip": "api_config_skip",
    # Paddle 问题：需重点定位 Paddle bug 或稳定性问题
    "paddle_error": "api_config_paddle_error",
    "paddle_accuracy": "api_config_paddle_accuracy",
    "paddle_bitwise": "api_config_paddle_bitwise",
    "paddle_cuda": "api_config_paddle_cuda",
    "paddle_crash": "api_config_paddle_crash",
    # 可重测：资源或超时，调整资源或放宽时间后可重试
    "oom": "api_config_oom",
    "timeout": "api_config_timeout",
    # 测试侧问题：对照侧或配置问题，不代表 Paddle 本身有 bug
    "torch_error": "api_config_torch_error",
    "config_input": "api_config_config_input",
    "config_parse": "api_config_config_parse",
    "config_convert": "api_config_config_convert",
}

# 默认汇总模式下的输出分组（--split-errors 时按每个终态分类单独输出）：
#   paddle_issue  — Paddle 问题，需重点定位
#   retest        — 资源/超时，主要用于重测筛选
#   test_issue    — 测试侧问题，不直接代表 Paddle bug
SUMMARY_GROUPS = {
    "paddle_issue": (
        "paddle_error",
        "paddle_accuracy",
        "paddle_bitwise",
        "paddle_cuda",
        "paddle_crash",
    ),
    "retest": ("oom", "timeout"),
    "test_issue": (
        "torch_error",
        "config_input",
        "config_parse",
        "config_convert",
    ),
}


def check_count_consistency(parsed_keys, config_keys, prefix):
    # 校验从 log_inorder.log 解析到的 case 数量与结果文件一致，不一致说明日志不完整或被截断
    parsed_len = len(parsed_keys)
    config_len = len(config_keys)
    if parsed_len == config_len:
        return None
    missing_keys = config_keys - parsed_keys
    extra_keys = parsed_keys - config_keys
    return (
        f"[WARNING] {prefix} 数量不一致: "
        f"config={config_len}, parsed={parsed_len}, "
        f"缺失={len(missing_keys)} {sorted(missing_keys)[:3]}, "
        f"多余={len(extra_keys)} {sorted(extra_keys)[:3]}"
    )


def read_configs(file_path):
    if not file_path.exists():
        return set()
    with file_path.open() as f:
        configs = {line.strip() for line in f if line.strip()}
    print(f"Read {len(configs)} configs from {file_path}", flush=True)
    return configs


def load_config_sets(input_path):
    input_path = Path(input_path)
    config_sets = {
        log_type: read_configs(input_path / f"{file_name}.txt")
        for log_type, file_name in LOG_PREFIXES.items()
    }
    return config_sets


def parse_logs(input_path):
    # 从 log_inorder.log 中按 "test begin" 分割出每个 case 的完整日志块。
    # 过滤掉 gpu_resources.cc 和内存等待行，避免干扰 case 边界识别。
    # "Worker PID" 行标志上一个 case 日志块结束（worker 重启分隔符）。
    log_path = Path(input_path) / "log_inorder.log"
    if not log_path.exists():
        print(f"{log_path} not exists", flush=True)
        return []
    with log_path.open("r") as f:
        input_text = f.read()

    logs = []
    in_test_block = False
    current_content = []
    for line in input_text.split("\n"):
        if "gpu_resources.cc" in line or "Waiting for available memory" in line:
            continue
        if "test begin" in line:
            if in_test_block and current_content:
                logs.append("\n".join(current_content))
            in_test_block = True
            current_content = [line]
            continue
        if "Worker PID" in line:
            if in_test_block and current_content:
                logs.append("\n".join(current_content))
            in_test_block = False
            current_content = []
            continue
        if in_test_block:
            current_content.append(line)
    if current_content:
        logs.append("\n".join(current_content))
    print(f"Found {len(logs)} logs", flush=True)
    return logs


def get_sort_key(content):
    match = re.search(r"test begin: (.*)$", content.split("\n", 1)[0])
    return match.group(1).strip() if match else ""


def classify_by_config(logs, config_sets):
    # 将每个日志块按其 config 字符串反查到对应的终态分类。
    # 同一 case 理论上只出现在一个终态文件中；checkpoint 不作为分类输出，跳过。
    classified_logs = {}
    for content in logs:
        key = get_sort_key(content)
        if not key:
            continue
        for log_type, configs in config_sets.items():
            if log_type == "checkpoint" or key not in configs:
                continue
            classified_logs.setdefault(log_type, {})[key] = content
    return classified_logs


def merge_classified_logs(classified_logs, log_types):
    merged_logs = {}
    for log_type in log_types:
        merged_logs.update(classified_logs.get(log_type, {}))
    return merged_logs


def print_consistency_warnings(warnings):
    if not warnings:
        return
    print("\n" + "!" * 50)
    print("WARNING: log count consistency issues were found:")
    for warning in warnings:
        print(f"  {warning}")
    print(
        "Result files have still been generated. "
        "Please check whether logs are incomplete or duplicated."
    )
    print("!" * 50 + "\n")


def write_logs_and_meta(output_path, logs_dict, prefix):
    # 为指定分类写出三个文件：完整日志块、API 名列表、config 字符串列表
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / f"{prefix}_log.log"
    api_file = output_path / f"{prefix}_api.txt"
    config_file = output_path / f"{prefix}_config.txt"

    apis = {config.split("(", 1)[0] for config in logs_dict}

    with open(log_file, "w") as f:
        for key in sorted(logs_dict.keys()):
            f.write(logs_dict[key] + "\n")
    with open(api_file, "w") as f:
        f.writelines(f"{api}\n" for api in sorted(apis))
    with open(config_file, "w") as f:
        f.writelines(f"{cfg}\n" for cfg in sorted(logs_dict.keys()))

    print(f"Write {len(logs_dict)} logs & {len(apis)} apis for {prefix}", flush=True)


def error_state(input_path, output_path, split_errors=False):
    # 结果写入独立子目录，避免与原始日志文件混在同级目录
    output_path = Path(output_path) / RESULT_DIR_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
        print(f"Cleared existing directory: {output_path}", flush=True)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    config_sets = load_config_sets(input_path)
    logs = parse_logs(input_path)
    if not logs:
        return

    classified_logs = classify_by_config(logs, config_sets)
    consistency_warnings = []

    pass_logs = classified_logs.get("pass", {})
    warning = check_count_consistency(set(pass_logs), config_sets["pass"], "pass")
    if warning:
        consistency_warnings.append(warning)
    write_logs_and_meta(output_path, pass_logs, "pass")

    skip_logs = classified_logs.get("skip", {})
    warning = check_count_consistency(set(skip_logs), config_sets["skip"], "skip")
    if warning:
        consistency_warnings.append(warning)
    if skip_logs:
        write_logs_and_meta(output_path, skip_logs, "skip")

    if split_errors:
        # --split-errors：按每个终态分类单独输出，便于精细排查
        for log_type, configs in config_sets.items():
            if log_type in ("checkpoint", "pass", "skip"):
                continue
            category_logs = classified_logs.get(log_type, {})
            warning = check_count_consistency(set(category_logs), configs, log_type)
            if warning:
                consistency_warnings.append(warning)
            if category_logs:
                write_logs_and_meta(output_path, category_logs, log_type)
        print_consistency_warnings(consistency_warnings)
        return

    # 默认汇总模式：按 SUMMARY_GROUPS 合并同组分类后输出
    for group_name, log_types in SUMMARY_GROUPS.items():
        group_logs = merge_classified_logs(classified_logs, log_types)
        group_configs = set().union(*(config_sets[log_type] for log_type in log_types))
        warning = check_count_consistency(set(group_logs), group_configs, group_name)
        if warning:
            consistency_warnings.append(warning)
        if group_logs:
            write_logs_and_meta(output_path, group_logs, group_name)

    print_consistency_warnings(consistency_warnings)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="test_log 分类整理工具（可按终态类型拆分）")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(DEFAULT_TEST_LOG_PATH),
        help="输入路径",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径（默认同输入路径）")
    parser.add_argument(
        "--split-errors",
        "-s",
        action="store_true",
        help="是否按终态分类拆分输出（默认按 paddle_issue/retest/test_issue 汇总）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output if args.output is not None else args.input
    error_state(args.input, output_path, split_errors=args.split_errors)


if __name__ == "__main__":
    main()
