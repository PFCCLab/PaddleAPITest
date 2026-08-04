# test_log 一键整理小工具
# @author: cangtianhuang
# @date: 2026-06-11

# 整理效果：pass + skip + Paddle 问题 + 可重测 + 测试侧问题（可按终态类型拆分）
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from tester.log_writer.log_schema import (
    CASE_BEGIN_TAG,
    CASE_END_TAG,
    LOG_PREFIXES,
)

DEFAULT_TEST_LOG_PATH = Path("tester/api_config/test_log_big_tensor")
RESULT_DIR_NAME = "error_stat_result"

# 默认汇总模式下的输出分组（--split-errors 时按每个终态分类单独输出）：
#   paddle_issue  — Paddle 问题，需重点定位
#   retest        — 资源/超时，主要用于重测筛选
#   test_issue    — 测试侧问题，不直接代表 Paddle bug
SUMMARY_GROUPS = {
    "skipped": ("memory_skip",),
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

CASE_BEGIN_RE = re.compile(rf"^{re.escape(CASE_BEGIN_TAG)}\s+(?P<case_id>[^|\s]+)")
CASE_END_RE = re.compile(rf"^{re.escape(CASE_END_TAG)}\s+(?P<case_id>[^|\s]+)")


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


def _parse_tagged_logs(input_text):
    logs = []
    current_content = []
    current_case_id = None
    begin_count = 0
    end_count = 0
    for line in input_text.splitlines():
        if (
            "gpu_resources.cc:" in line
            and "Please NOTE: device:" in line
            or "Waiting for available memory" in line
        ):
            continue
        begin_match = CASE_BEGIN_RE.match(line)
        if begin_match:
            begin_count += 1
            if current_content:
                logs.append("\n".join(current_content).rstrip())
            current_content = [line]
            current_case_id = begin_match.group("case_id")
            continue
        if current_content:
            end_match = CASE_END_RE.match(line)
            if end_match:
                while len(current_content) > 2 and not current_content[-1].strip():
                    current_content.pop()
                current_content.append(line)
                end_count += 1
                end_case_id = end_match.group("case_id")
                if current_case_id != end_case_id:
                    print(
                        f"[WARNING] case_id mismatch: begin={current_case_id}, end={end_case_id}",
                        flush=True,
                    )
                logs.append("\n".join(current_content).rstrip())
                current_content = []
                current_case_id = None
            else:
                current_content.append(line)
    if current_content:
        logs.append("\n".join(current_content).rstrip())
    if begin_count != end_count:
        print(
            f"[WARNING] CASE tag count mismatch: begin={begin_count}, end={end_count}",
            flush=True,
        )
    return logs


def parse_logs(input_path):
    # 当前日志按 CASE tag 分割。
    log_path = Path(input_path) / "log_inorder.log"
    if not log_path.exists():
        print(f"{log_path} not exists", flush=True)
        return []
    with log_path.open("r") as f:
        input_text = f.read()

    logs = _parse_tagged_logs(input_text)
    print(f"Found {len(logs)} logs", flush=True)
    return logs


def get_config_key(content):
    # CASE block 的第二行固定为 config。
    return content.splitlines()[1].strip()


def classify_by_config(logs, config_sets):
    # 将每个日志块按其 config 字符串反查到对应的终态分类。
    # 同一 case 理论上只出现在一个终态文件中；checkpoint 不作为分类输出，跳过。
    classified_logs = {}
    for content in logs:
        key = get_config_key(content)
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
            f.write(logs_dict[key].rstrip("\n") + "\n\n")
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
