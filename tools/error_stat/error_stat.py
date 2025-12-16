# test_log 一键整理小工具
# @author: cangtianhuang
# @date: 2025-11-11
# 整理效果：pass + error + invalid （可按类型拆分）
from __future__ import annotations

import argparse
import re
from pathlib import Path

SKIP_ERROR_INFO = [
    "(Cannot allocate memory)",
    "(InvalidArgument)",
    "(NotFound)",
    "(ResourceExhausted)",
    "(Unimplemented)",
    "CUDA out of memory",
    "Out of memory error",
    "[Skip]",
    "[match error]",
    "[numpy error]",
    "[paddle_to_torch]",
    "[torch error]",
    "output type diff error",
]

LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
    "timeout": "api_config_timeout",
    "paddle_error": "api_config_paddle_error",
    "accuracy_error": "api_config_accuracy_error",
    "accuracy_diff": "api_config_accuracy_diff",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "match_error": "api_config_match_error",
    "numpy_error": "api_config_numpy_error",
    "cuda_error": "api_config_cuda_error",
}


def check_count_consistency(parsed_keys, config_keys, prefix):
    parsed_len = len(parsed_keys)
    config_len = len(config_keys)
    if parsed_len != config_len:
        missing_keys = config_keys - parsed_keys
        extra_keys = parsed_keys - config_keys
        msg = (
            f"[ASSERT ERROR] {prefix} 数量不一致: "
            f"config={config_len}, parsed={parsed_len}, "
            f"缺失={len(missing_keys)} {sorted(missing_keys)[:3]}, "
            f"多余={len(extra_keys)} {sorted(extra_keys)[:3]}"
        )
        raise AssertionError(msg)


def load_config_sets(input_path):
    config_sets = {}
    for prefix, file_name in LOG_PREFIXES.items():
        file_path = Path(input_path) / f"{file_name}.txt"
        if file_path.exists():
            with open(file_path) as f:
                configs = {line.strip() for line in f if line.strip()}
                config_sets[prefix] = configs
                print(f"Read {len(configs)} configs from {file_path}", flush=True)
        else:
            config_sets[prefix] = set()
    return config_sets


def parse_logs(input_path):
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
    classified_logs = {}
    classified_invalid_logs = {}
    for content in logs:
        key = get_sort_key(content)
        if not key:
            continue
        for prefix, configs in config_sets.items():
            if prefix in ("pass", "checkpoint"):
                continue
            if key in configs:
                if any(info in content for info in SKIP_ERROR_INFO):
                    inv_prefix = f"invalid_{prefix}"
                    classified_invalid_logs.setdefault(inv_prefix, {})[key] = content
                else:
                    classified_logs.setdefault(prefix, {})[key] = content
    return classified_logs, classified_invalid_logs


def write_logs_and_meta(output_path, logs_dict, prefix):
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
    # 读取配置文件
    config_sets = load_config_sets(input_path)
    # 解析日志
    logs = parse_logs(input_path)
    if not logs:
        return

    # 分类 pass
    pass_logs_dict = {
        key: content for content in logs if (key := get_sort_key(content)) in config_sets["pass"]
    }
    check_count_consistency(set(pass_logs_dict.keys()), config_sets["pass"], "pass")
    write_logs_and_meta(output_path, pass_logs_dict, "pass")

    # 处理 error 和 invalid
    error_logs_dict, invalid_logs_dict = classify_by_config(logs, config_sets)
    if split_errors:
        for prefix in config_sets:
            if prefix in ("pass", "checkpoint"):
                continue
            error_keys = set(error_logs_dict.get(prefix, {}).keys())
            invalid_keys = set(invalid_logs_dict.get(f"invalid_{prefix}", {}).keys())
            all_keys = error_keys | invalid_keys
            check_count_consistency(all_keys, config_sets[prefix], prefix)
            if error_keys:
                write_logs_and_meta(output_path, error_logs_dict[prefix], prefix)
            if invalid_keys:
                write_logs_and_meta(
                    output_path,
                    invalid_logs_dict[f"invalid_{prefix}"],
                    f"invalid_{prefix}",
                )
    else:
        error_union = {}
        invalid_union = {}
        error_keys_union = set()
        invalid_keys_union = set()
        for prefix in config_sets:
            if prefix in ("pass", "checkpoint"):
                continue
            error_union.update(error_logs_dict.get(prefix, {}))
            error_keys_union |= set(error_logs_dict.get(prefix, {}).keys())
            inv_prefix = f"invalid_{prefix}"
            invalid_union.update(invalid_logs_dict.get(inv_prefix, {}))
            invalid_keys_union |= set(invalid_logs_dict.get(inv_prefix, {}).keys())
        all_keys = error_keys_union | invalid_keys_union
        all_configs = set().union(
            *(configs for p, configs in config_sets.items() if p not in ("pass", "checkpoint"))
        )
        check_count_consistency(all_keys, all_configs, "error+invalid")
        if error_union:
            write_logs_and_meta(output_path, error_union, "error")
        if invalid_union:
            write_logs_and_meta(output_path, invalid_union, "invalid")


def main():
    parser = argparse.ArgumentParser(description="test_log 整理工具（可按类型拆分）")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="tester/api_config/test_log_big_tensor",
        help="输入路径",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="输出路径（默认同输入路径）")
    parser.add_argument(
        "--split-errors",
        "-s",
        action="store_true",
        help="是否将错误和无效按类型拆分输出",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input
    error_state(args.input, args.output, split_errors=args.split_errors)


if __name__ == "__main__":
    main()
