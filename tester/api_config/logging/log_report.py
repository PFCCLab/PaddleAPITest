"""运行进度和汇总报告。"""

from __future__ import annotations

import os
from datetime import datetime

from .log_schema import LOG_PREFIXES


def _single_line(value):
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def format_duration(seconds):
    """按运行时长选择秒、分钟或小时单位。"""
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.2f} s"


def print_run_header(options, paddle_version):
    """按参数名分组打印一次测试的有效配置。"""
    modes = (
        "accuracy",
        "paddle_only",
        "paddle_cinn",
        "paddle_gpu_performance",
        "torch_gpu_performance",
        "paddle_torch_gpu_performance",
        "accuracy_stable",
        "paddle_custom_device",
        "custom_device_vs_gpu",
    )
    mode = next(name for name in modes if getattr(options, name))
    if options.api_config:
        source_option = ("--api_config", options.api_config)
    elif options.api_config_file:
        source_option = ("--api_config_file", options.api_config_file)
    elif getattr(options, "retest", ""):
        source_option = ("--retest", options.retest)
    else:
        source_option = ("--api_config_file_pattern", options.api_config_file_pattern)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f">>> TEST RUN | {timestamp} | Paddle {paddle_version} | PID {os.getpid()}")
    print("\n--- OPTIONS")
    files = [
        source_option,
        ("--log_dir", options.log_dir),
    ]
    test = [
        (f"--{mode}", True),
        ("--timeout", f"{options.timeout} s"),
        ("--show_runtime_status", options.show_runtime_status),
    ]
    groups = [("Files", files), ("Test", test)]

    if mode in ("accuracy", "custom_device_vs_gpu"):
        groups.append(
            (
                "Accuracy",
                [("--atol", options.atol), ("--rtol", options.rtol)],
            )
        )

    if options.test_cpu:
        compute = [("--test_cpu", True)]
    else:
        if not options.gpu_ids:
            gpu_ids_display = "all visible"
        elif options.gpu_ids == "-1":
            gpu_ids_display = "-1 (all visible)"
        else:
            gpu_ids_display = options.gpu_ids
        compute = [("--gpu_ids", gpu_ids_display)]
        if options.use_gpu_mode:
            compute.extend(
                [
                    ("--use_gpu_mode", True),
                    ("--gpu_memory_policy", options.gpu_memory_policy),
                ]
            )
        elif options.use_cached_numpy:
            compute.append(("--use_cached_numpy", True))
        compute.append(("--num_workers_per_gpu", options.num_workers_per_gpu))
    groups.append(("Compute", compute))
    for group_name, group_options in groups:
        print(group_name)
        for name, value in group_options:
            display_value = str(value).lower() if isinstance(value, bool) else value
            print(f"  {name}: {display_value}")


def print_preparing_summary(
    read_count,
    non_config_count,
    duplicate_count,
    total_case,
    checkpointed_case,
    pending_case,
    *,
    removed_stale_logs,
    retest_types,
):
    """打印配置读取和断点续跑摘要。"""
    print("\n--- PREPARING")
    if removed_stale_logs:
        print(f"Cleanup: {removed_stale_logs} stale result entries removed (not in checkpoint)")
    if retest_types:
        print(f"Retest: {', '.join(retest_types)} | {total_case} selected")
    print(
        f"Configs: {read_count} read | {non_config_count} non-config | {duplicate_count} duplicate"
    )
    print(f"Cases: {total_case} total | {checkpointed_case} checkpointed | {pending_case} pending")


def print_compute_summary(available_gpus, max_workers_per_gpu):
    """打印实际选中的 GPU 和 worker 布局。"""
    total_workers = sum(max_workers_per_gpu.values())
    print(f"Compute: {len(available_gpus)} GPUs | {total_workers} workers")
    layout = " | ".join(
        f"GPU {gpu_id}: {workers}" for gpu_id, workers in sorted(max_workers_per_gpu.items())
    )
    print(f"Layout: {layout}")


def print_case_progress(current, total, status, config, detail):
    """打印紧凑的单行 case 状态和进度百分比。"""
    percent = current / total * 100 if total else 100.0
    detail_field = f"{_single_line(detail)} | " if detail else ""
    print(
        f"[{current}/{total} {percent:.1f}%] {status} | {detail_field}{config}",
        flush=True,
    )


def print_case_notice(status, config, detail):
    """打印不推进完成计数的单行 case 事件。"""
    detail_field = f"{_single_line(detail)} | " if detail else ""
    print(f"[case] {status} | {detail_field}{config}", flush=True)


def print_run_footer(total_case, tested_case, remaining_case, log_counts, elapsed, log_dir):
    """打印统一结果摘要、结束时间和总用时。"""
    _print_log_warnings(log_counts)
    counts = {key: value for key, value in log_counts.items() if not key.startswith("_")}
    paddle_types = (
        "paddle_error",
        "paddle_accuracy",
        "paddle_bitwise",
        "paddle_cuda",
        "paddle_crash",
    )
    test_types = ("torch_error", "config_input", "config_parse", "config_convert")
    paddle_issues = sum(counts.get(key, 0) for key in paddle_types)
    test_issues = sum(counts.get(key, 0) for key in test_types)
    retest = sum(counts.get(key, 0) for key in ("oom", "timeout"))
    outcome = (
        "PASS" if remaining_case == 0 and paddle_issues + test_issues + retest == 0 else "DONE"
    )
    completed_case = counts.get("checkpoint", tested_case)
    overall_total = max(total_case, completed_case + remaining_case)
    failed_case = max(completed_case - counts.get("pass", 0) - counts.get("skip", 0), 0)
    progress = completed_case / overall_total * 100 if overall_total else 100.0
    print("\n--- RESULT")
    print(f"Progress: {completed_case} / {overall_total} | {progress:.1f}%")
    print(
        f"Cases: {counts.get('pass', 0)} pass | {failed_case} fail | "
        f"{counts.get('skip', 0)} skip | {remaining_case} remaining"
    )
    print(f"Issues: {paddle_issues} Paddle | {test_issues} test | {retest} retest")
    print("Classification")
    ordered_types = [*LOG_PREFIXES, "incomplete"]
    for log_type in ordered_types:
        if log_type in counts:
            print(f"  {log_type}: {counts[log_type]}")
    for log_type in sorted(set(counts) - set(ordered_types)):
        print(f"  {log_type}: {counts[log_type]}")
    print(f"Duration: {format_duration(elapsed)}")
    print(f"Logs: {log_dir}")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(
        f"\n<<< TEST RUN | {outcome} | {completed_case}/{overall_total} completed | "
        f"{timestamp} | {format_duration(elapsed)}",
        flush=True,
    )


def _print_duplicate_classifications(integrity_errors):
    """打印主结果或 comp 维度中的重复分类。"""
    for issue in integrity_errors:
        scope = issue["scope"]
        duplicates = issue["duplicates"]
        print("\n" + "!" * 50)
        print(f"WARNING: configs found in multiple log types ({scope}):")
        for config, types in sorted(duplicates.items())[:20]:
            print(f"  {config}")
            print(f"    -> {', '.join(types)}")
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")
        print(f"Found {len(duplicates)} duplicated config(s). Please check log classification.")
        print("!" * 50 + "\n")


def _print_log_warnings(log_counts):
    """打印聚合失败和分类完整性告警。"""
    errors = log_counts.get("_aggregation_errors", [])
    if errors:
        print("\nWARNING: log aggregation was incomplete:")
        for error in errors:
            print(f"  {error}")
    _print_duplicate_classifications(log_counts.get("_integrity_errors", []))
    _print_duplicate_classifications(log_counts.get("_comp_integrity_errors", []))
