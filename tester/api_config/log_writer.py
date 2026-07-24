"""Paddle API 测试日志的写入、结构化边界和聚合。

函数索引：
- get_cfg：返回当前命令行配置。
- set_cfg：保存命令行配置并规范化日志文件 ID 后缀。
- _reset_runtime：关闭缓存句柄并清空进程内日志状态。
- init_log：设置输出目录并初始化主进程或 worker 日志环境。
- get_tmp_log_path：返回 worker 临时日志目录。
- close_process_files：关闭当前进程缓存的所有结果日志句柄。
- has_terminal_log：判断配置是否已有主终态分类。
- get_terminal_log_type：返回配置当前的主终态类型。
- write_checkpoint：记录配置完成并清理其主终态状态。
- write_terminal_log：依次写入终态分类和 checkpoint。
- _write_line：复用缓存句柄向指定结果文件追加一行。
- write_to_log：写入主结果或 worker 结果分片并更新分类状态。
- has_comp_terminal_log：判断配置在指定 comp 维度是否已有分类。
- write_to_comp_log：写入 accuracy-stable comp 维度结果并更新状态。
- read_log：读取一个聚合结果文件中的全部配置。
- parse_retest_types：解析并验证命令行复测分类。
- prepare_retest：加载复测集合并清理其旧结构化结果。
- finish_retest：清除已完成复测的恢复 manifest。
- cleanup_uncheckpointed_result_logs：删除没有对应 checkpoint 的残留结果。
- _read_pending_result_bytes：按结果 offset 读取新增完整行。
- _save_result_offsets：提交结果 offset，并在 cleanup 时删除已消费分片。
- get_sanitizer_case_log_dir：返回 sanitizer case 的隔离日志目录。
- merge_sanitizer_case_logs：将 sanitizer child 结果分片合并回 worker 临时目录。
- clean_sanitizer_case_logs：清理 sanitizer case 隔离目录。
- get_case_id：从配置生成稳定的短 case ID。
- write_case_begin：输出包含运行元数据的结构化 CASE_BEGIN。
- _case_results：读取配置在当前进程产生的全部结果类型。
- write_case_end：输出 CASE_END，flush 后返回 worker 安全 offset。
- get_worker_log_offset：获取当前或指定 worker stdout 文件末尾。
- append_case_end_to_worker_log：为已停止 worker 补写 synthetic CASE_END。
- _copy_inorder_range：复制 worker stdout 的安全字节区间并截断超长行。
- mark_inorder_case_complete：登记 worker 最后完成 case 的安全读取上界。
- flush_completed_inorder_logs：批量追加所有 worker 已完成区间并提交 offset。
- _aggregate_text_logs：合并、去重并排序 worker 文本结果分片。
- _aggregate_result_logs：聚合所有主结果日志类型。
- _aggregate_inorder_logs：结束或 cleanup 时聚合 worker stdout 剩余内容。
- _aggregate_csv_logs：按 offset 合并 worker CSV 分片。
- _sort_csv：按报告字段稳定排序最终 CSV。
- _count_result_logs：统计分类数量并生成未完成配置文件。
- _aggregate_comp_logs：按 accuracy-stable comp 维度聚合结果。
- _find_duplicate_classifications：查找同一目录中的重复分类。
- _read_log_lines：将单个结果文件读取为配置集合。
- _sync_comp_main_summary：将 comp 维度分类合并到主结果摘要。
- _check_log_integrity：检查主目录或 comp 维度的分类完整性。
- aggregate_logs：编排结果、stdout、CSV 和 comp 聚合及最终统计。
- _print_duplicate_classifications：打印主结果重复分类。
- _print_comp_duplicate_classifications：打印 comp 维度重复分类。
- print_log_info：打印最终测试统计和完整性告警。
- limit_worker_layout：按 pending case 数裁剪实际 worker 布局。
- redirect_stdio：将 worker stdout/stderr 行缓冲重定向到临时日志。
- restore_stdio：恢复 worker 原始 stdout/stderr。
- _get_diff：从精度错误消息提取最大绝对和相对误差。
- _append_csv：向 worker CSV 分片追加表头和数据行。
- log_accuracy_tolerance：记录 accuracy tolerance 比较结果。
- log_accuracy_stable：记录 accuracy-stable comp 比较结果。

关键状态：
- _process_file_handlers：当前进程按路径缓存的结果日志句柄，避免每条结果重复
  open/close；_write_line 统一复用句柄并处理写入错误。
- _result_offsets：各 worker 结果文本/CSV 已聚合到的位置。
- _inorder_offsets：各 worker stdout 已写入 log_inorder.log 的位置。
- _inorder_completed_offsets：各 worker 最后一个完整 case 的安全读取上界。
- _process_terminal_configs/_case_result_types：当前 case 的分类和结构化结果。
- _comp_terminal_configs：各 accuracy-stable comp 维度内的分类状态。
- stdout_fd/stderr_fd/orig_*：worker 标准流重定向和恢复所需描述符。
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

# 日志文件路径
DIR_PATH = Path(__file__).resolve()
DIR_PATH = DIR_PATH.parent.parent.parent
TEST_LOG_PATH = DIR_PATH / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"

# 日志类型和对应的文件，可在下方进行注册
LogType = Literal[
    "checkpoint",
    "pass",
    "skip",
    "paddle_error",
    "paddle_accuracy",
    "paddle_bitwise",
    "paddle_cuda",
    "paddle_crash",
    "oom",
    "timeout",
    "torch_error",
    "config_input",
    "config_parse",
    "config_convert",
]

LOG_PREFIXES: dict[LogType, str] = {
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

TERMINAL_LOG_TYPES = frozenset(LOG_PREFIXES) - {"checkpoint"}

# === comp 维度配置（accuracy_stable 模式） ===
COMP_TO_DIMENSION = {
    "P1T1": "accuracy",
    "P2T2": "accuracy",
    "P2T1": "accuracy",
    "P1T2": "accuracy",
    "P1T1B": "accuracy_backward",
    "P2T2B": "accuracy_backward",
    "P2T1B": "accuracy_backward",
    "P1T2B": "accuracy_backward",
    "T1T2": "torch_stable",
    "T1T2B": "torch_stable_backward",
    "P1P2": "paddle_stable",
    "P1P2B": "paddle_stable_backward",
}
COMP_SUMMARY_PAIRS = (
    ("P1T1", "P1T1B"),
    ("P2T2", "P2T2B"),
    ("P2T1", "P2T1B"),
    ("P1T2", "P1T2B"),
    ("T1T2", "T1T2B"),
    ("P1P2", "P1P2B"),
)
ALL_DIMENSIONS = sorted(set(COMP_TO_DIMENSION.values()))
TOL_HEADER = ["API", "config", "dtype", "mode", "max_abs_diff", "max_rel_diff"]
STABLE_HEADER = ["API", "config", "dtype", "comp", "max_abs_diff", "max_rel_diff"]
CASE_BEGIN_TAG = ">>> CASE"
CASE_END_TAG = "<<< CASE"

_use_worker_tmp_logs = False

_process_file_handlers = {}
# 普通结果日志和 inorder stdout 使用不同的 offset 命名空间。
_result_offsets = {}
_inorder_offsets = {}
_inorder_completed_offsets = {}
# 配置 -> 全局终态类型；comp 日志写入后也同步到这里，供 checkpoint 逻辑使用。
_process_terminal_configs = {}
_case_result_types: dict[str, set[str]] = {}
# (comp, result) -> [matched tensor count, total tensor count]
_case_comparisons: dict[tuple[str, str], list[int]] = {}
_case_has_comp_output = False
# dimension -> {config_line -> log_type}，只负责 comp 维度内去重。
_comp_terminal_configs: dict[str, dict[str, str]] = {}
RETEST_PENDING_FILENAME = ".retest_pending.txt"
RETEST_TYPES_FILENAME = ".retest_types.txt"
MAX_CSV_CONFIG_LENGTH = 120000

# 命令行参数配置，由 engine.py 使用
CMD_CONFIG = None


def get_cfg():
    """返回当前命令行配置。"""
    return CMD_CONFIG


def set_cfg(cfg):
    """保存命令行配置，并规范化其文件名后缀。"""
    global CMD_CONFIG
    if cfg.id != "":
        cfg.id = "_" + cfg.id
    CMD_CONFIG = cfg


def _reset_runtime():
    """清理进程内文件句柄和聚合状态。"""
    global _case_has_comp_output
    close_process_files()
    _result_offsets.clear()
    _inorder_offsets.clear()
    _inorder_completed_offsets.clear()
    _process_terminal_configs.clear()
    _case_result_types.clear()
    _case_comparisons.clear()
    _case_has_comp_output = False
    _comp_terminal_configs.clear()


def init_log(log_dir=None, *, worker_tmp_logs=False):
    """初始化日志路径并重置进程内状态。"""
    global TEST_LOG_PATH, TMP_LOG_PATH, _use_worker_tmp_logs
    _reset_runtime()
    if log_dir:
        TEST_LOG_PATH = DIR_PATH / log_dir
    else:
        TEST_LOG_PATH = DIR_PATH / "tester/api_config/test_log"
    TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
    TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"
    _use_worker_tmp_logs = worker_tmp_logs
    if _use_worker_tmp_logs:
        TMP_LOG_PATH.mkdir(exist_ok=True)


def get_tmp_log_path():
    """返回当前临时日志目录。"""
    return TMP_LOG_PATH


def close_process_files():
    """关闭当前进程缓存的结果日志句柄。"""
    global _process_file_handlers
    for handler in _process_file_handlers.values():
        try:
            handler.close()
        except Exception as err:
            print(f"Error closing process file: {err}", flush=True)
    _process_file_handlers = {}


def has_terminal_log(line):
    """判断当前进程是否已分类该配置。"""
    return line.strip() in _process_terminal_configs


def get_terminal_log_type(line):
    """返回当前进程对配置的终态分类。"""
    return _process_terminal_configs.get(line.strip())


def _record_terminal_type(line, log_type):
    """错误终态可覆盖 pass，pass 不覆盖错误终态。"""
    if log_type != "pass" or line not in _process_terminal_configs:
        _process_terminal_configs[line] = log_type


def write_checkpoint(line):
    """写入已完成配置并清理进程内分类状态。"""
    line = line.strip()
    write_to_log("checkpoint", line)
    _process_terminal_configs.pop(line, None)


def write_terminal_log(log_type: LogType, line):
    """写入终态分类并随后写入 checkpoint。"""
    write_to_log(log_type, line)
    write_checkpoint(line)


def _write_line(file_path, line):
    """复用进程文件句柄写一行，集中处理句柄创建、缓存和写入错误。"""
    try:
        handler = _process_file_handlers.get(file_path)
        if handler is None:
            handler = file_path.open("a", buffering=1)
            _process_file_handlers[file_path] = handler
        handler.write(line + "\n")
        return True
    except Exception as err:
        print(f"Error writing to {file_path}: {err}", flush=True)
        return False


def write_to_log(log_type: LogType, line):
    """向主结果日志或 worker 临时结果日志追加一个配置。"""
    line = line.strip()
    if not line:
        return
    terminal_log_type = _process_terminal_configs.get(line)
    if _use_worker_tmp_logs and log_type == "pass" and terminal_log_type not in (None, "pass"):
        return
    prefix = LOG_PREFIXES[log_type]
    if _use_worker_tmp_logs:
        file_path = TMP_LOG_PATH / f"{prefix}_{os.getpid()}.txt"
    else:
        cfg = get_cfg()
        filename = f"{prefix}{cfg.id}.txt" if cfg else f"{prefix}.txt"
        file_path = TEST_LOG_PATH / filename
    if _write_line(file_path, line) and log_type in TERMINAL_LOG_TYPES:
        _case_result_types.setdefault(line, set()).add(log_type)
        if _use_worker_tmp_logs:
            _record_terminal_type(line, log_type)


def has_comp_terminal_log(dimension, line):
    """判断 comp 维度是否已分类该配置。"""
    line = line.strip()
    dim_configs = _comp_terminal_configs.get(dimension)
    if dim_configs is None:
        return False
    return line in dim_configs


def write_to_comp_log(comp, log_type: LogType, line):
    """向一个 comp 维度写入配置并更新分类状态。"""
    dimension = COMP_TO_DIMENSION[comp]
    prefix = LOG_PREFIXES[log_type]
    root = TMP_LOG_PATH if _use_worker_tmp_logs else TEST_LOG_PATH
    comp_dir = root / "comp" / dimension
    comp_dir.mkdir(parents=True, exist_ok=True)
    if _use_worker_tmp_logs:
        file_path = comp_dir / f"{prefix}_{os.getpid()}.txt"
    else:
        cfg = get_cfg()
        file_path = comp_dir / (f"{prefix}{cfg.id}.txt" if cfg else f"{prefix}.txt")
    line = line.strip()
    if not line:
        return

    dim_configs = _comp_terminal_configs.setdefault(dimension, {})
    existing = dim_configs.get(line)
    if existing is not None and (existing != "pass" or log_type == "pass"):
        return

    if not _write_line(file_path, line):
        return

    if log_type in TERMINAL_LOG_TYPES:
        dim_configs[line] = log_type
    _case_result_types.setdefault(line, set()).add(log_type)
    if log_type in TERMINAL_LOG_TYPES and _use_worker_tmp_logs:
        _record_terminal_type(line, log_type)


def read_log(log_type: LogType):
    """读取一个聚合结果日志中的所有非空配置。"""
    cfg = get_cfg()
    prefix = LOG_PREFIXES[log_type]
    filename = f"{prefix}{cfg.id}.txt" if cfg else f"{prefix}.txt"
    file_path = TEST_LOG_PATH / filename
    try:
        with file_path.open("r") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()
    except Exception as err:
        print(f"Error reading {file_path}: {err}", flush=True)
        return set()


def parse_retest_types(value):
    """解析逗号分隔的复测分类，并保持用户给定顺序。"""
    if not value:
        return ()
    valid_types = set(LOG_PREFIXES) - {"checkpoint"}
    retest_types = []
    for raw_type in value.split(","):
        log_type = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        if not log_type:
            raise ValueError("--retest contains an empty classification")
        if log_type not in valid_types:
            choices = ", ".join(sorted(valid_types))
            raise ValueError(
                f"invalid --retest classification '{raw_type.strip()}'; choose: {choices}"
            )
        if log_type not in retest_types:
            retest_types.append(log_type)
    return tuple(retest_types)


def _current_result_file(prefix, root=None):
    root = TEST_LOG_PATH if root is None else root
    cfg = get_cfg()
    suffix = cfg.id if cfg else ""
    return root / f"{prefix}{suffix}.txt"


def _write_lines_atomic(file_path, lines):
    temp_file = file_path.with_name(f".{file_path.name}.tmp")
    try:
        with temp_file.open("w") as target:
            target.writelines(lines)
        os.replace(temp_file, file_path)
    finally:
        temp_file.unlink(missing_ok=True)


def _rewrite_text_excluding(file_path, excluded_lines):
    if not file_path.exists():
        return
    temp_file = file_path.with_name(f".{file_path.name}.retest.tmp")
    try:
        changed = False
        kept = 0
        with file_path.open() as source, temp_file.open("w") as target:
            for line in source:
                if line.strip() in excluded_lines:
                    changed = True
                    continue
                target.write(line)
                kept += 1
        if not changed:
            return
        if kept:
            os.replace(temp_file, file_path)
        else:
            file_path.unlink()
    finally:
        temp_file.unlink(missing_ok=True)


def _rewrite_csv_excluding(file_path, excluded_configs):
    if not file_path.exists():
        return
    temp_file = file_path.with_name(f".{file_path.name}.retest.tmp")
    try:
        with file_path.open(newline="") as source, temp_file.open("w", newline="") as target:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "config" not in reader.fieldnames:
                return
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            changed = False
            for row in reader:
                if row.get("config") in excluded_configs:
                    changed = True
                    continue
                writer.writerow(row)
        if not changed:
            return
        os.replace(temp_file, file_path)
    finally:
        temp_file.unlink(missing_ok=True)


def _restore_truncated_retest_configs(configs, checkpoints):
    truncated = {config for config in configs if len(config) == MAX_CSV_CONFIG_LENGTH}
    if not truncated:
        return set(configs)

    matches_by_prefix = {config: [] for config in truncated}
    for checkpoint in checkpoints:
        if len(checkpoint) < MAX_CSV_CONFIG_LENGTH:
            continue
        prefix = checkpoint[:MAX_CSV_CONFIG_LENGTH]
        if prefix in matches_by_prefix:
            matches_by_prefix[prefix].append(checkpoint)

    restored = set(configs) - truncated
    for config, matches in matches_by_prefix.items():
        if len(matches) > 1:
            raise ValueError(
                "cannot restore truncated retest config: multiple checkpoint entries "
                f"share its {MAX_CSV_CONFIG_LENGTH}-character prefix"
            )
        restored.add(matches[0] if matches else config)
    return restored


def prepare_retest(retest_types):
    """读取复测分类，并清除这些配置的当前结构化结果。"""
    pending_file = TEST_LOG_PATH / RETEST_PENDING_FILENAME
    types_file = TEST_LOG_PATH / RETEST_TYPES_FILENAME
    if pending_file.exists():
        stored_types_text = types_file.read_text().strip() if types_file.exists() else ""
        try:
            stored_types = parse_retest_types(stored_types_text)
        except ValueError:
            stored_types = ()
        if set(stored_types) != set(retest_types):
            expected_types = ",".join(retest_types)
            raise ValueError(
                f"unfinished retest uses '{stored_types_text or 'unknown'}'; "
                f"resume it before starting '{expected_types}'"
            )
        with pending_file.open() as source:
            retest_configs = {line.strip() for line in source if line.strip()}
        retest_configs -= read_log("checkpoint")
        cleanup_configs = set(retest_configs)
        if not retest_configs:
            finish_retest()
            return set()
    else:
        raw_retest_configs = set()
        for log_type in retest_types:
            raw_retest_configs.update(read_log(log_type))
        retest_configs = _restore_truncated_retest_configs(
            raw_retest_configs, read_log("checkpoint")
        )
        cleanup_configs = raw_retest_configs | retest_configs
        if retest_configs:
            _write_lines_atomic(types_file, (",".join(retest_types) + "\n",))
            _write_lines_atomic(
                pending_file,
                (f"{config}\n" for config in sorted(retest_configs)),
            )
    if not retest_configs:
        return set()

    close_process_files()
    for prefix in LOG_PREFIXES.values():
        _rewrite_text_excluding(_current_result_file(prefix), cleanup_configs)

    comp_dir = TEST_LOG_PATH / "comp"
    if comp_dir.exists():
        for dimension_dir in comp_dir.iterdir():
            if not dimension_dir.is_dir():
                continue
            for prefix in LOG_PREFIXES.values():
                _rewrite_text_excluding(
                    _current_result_file(prefix, root=dimension_dir), cleanup_configs
                )

    _rewrite_text_excluding(TEST_LOG_PATH / "api_config_incomplete.txt", cleanup_configs)
    csv_configs = cleanup_configs | {config[:MAX_CSV_CONFIG_LENGTH] for config in cleanup_configs}
    _rewrite_csv_excluding(TEST_LOG_PATH / "tol.csv", csv_configs)
    _rewrite_csv_excluding(TEST_LOG_PATH / "stable.csv", csv_configs)
    return retest_configs


def finish_retest():
    """清除已完成复测的恢复 manifest。"""
    (TEST_LOG_PATH / RETEST_PENDING_FILENAME).unlink(missing_ok=True)
    (TEST_LOG_PATH / RETEST_TYPES_FILENAME).unlink(missing_ok=True)


def cleanup_uncheckpointed_result_logs():
    """删除中断运行中先于 checkpoint 写入的结果记录。"""
    checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
    try:
        with checkpoint_file.open("r") as f:
            checkpoints = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return 0
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
        return 0

    removed = 0
    for log_type, prefix in LOG_PREFIXES.items():
        if log_type == "checkpoint":
            continue
        log_file = TEST_LOG_PATH / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            with log_file.open("r") as f:
                lines = f.readlines()
            new_lines = [line for line in lines if line.strip() in checkpoints]
            removed += len(lines) - len(new_lines)
            if len(new_lines) != len(lines):
                if new_lines:
                    with log_file.open("w") as f:
                        f.writelines(new_lines)
                else:
                    log_file.unlink()
        except Exception as err:
            print(f"Error cleaning {log_file}: {err}", flush=True)
    return removed


def _read_pending_result_bytes(file_path, end=False):
    """读取上次聚合后的新增字节，非 cleanup 模式只读完整行。"""
    offset = _result_offsets.get(file_path, 0)
    file_size = file_path.stat().st_size
    if file_size < offset:
        offset = 0
    if file_size == offset:
        return b"", offset, offset
    with file_path.open("rb") as f:
        f.seek(offset)
        data = f.read(file_size - offset)
    if not end and data and not data.endswith(b"\n"):
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            return b"", offset, offset
        data = data[: last_newline + 1]
        file_size = offset + last_newline + 1
    return data, offset, file_size


def _save_result_offsets(pending_offsets, cleanup):
    """保存读取偏移，并按需删除已消费的临时文件。"""
    for file_path, offset in pending_offsets.items():
        if cleanup:
            _result_offsets.pop(file_path, None)
            file_path.unlink()
        else:
            _result_offsets[file_path] = offset


def get_sanitizer_case_log_dir(slot_index, pid):
    """返回一个 sanitizer worker slot 的隔离输出目录。"""
    return TMP_LOG_PATH / "sanitizer" / f"slot_{slot_index}_{pid}"


def merge_sanitizer_case_logs(case_log_dir):
    """将一个 sanitizer case 的结果文件追加到 worker 临时日志。"""
    case_tmp_dir = case_log_dir / ".tmp"
    if not case_tmp_dir.exists():
        return
    for child_log in case_tmp_dir.iterdir():
        if not child_log.is_file():
            continue
        target_log = TMP_LOG_PATH / child_log.name
        with child_log.open("rb") as in_f, target_log.open("ab") as out_f:
            shutil.copyfileobj(in_f, out_f)


def clean_sanitizer_case_logs():
    """删除所有隔离的 sanitizer case 目录。"""
    shutil.rmtree(TMP_LOG_PATH / "sanitizer", ignore_errors=True)


def get_case_id(api_config_str):
    """生成 case 边界 tag 使用的稳定短 ID。"""
    return hashlib.sha1(api_config_str.encode("utf-8", errors="replace")).hexdigest()[:12]


def write_case_begin(api_config_str, *, worker_pid=None, slot=None, gpu=None, paddle_version=None):
    """写入包含执行元数据的 case 起始行并返回其 ID。"""
    case_id = get_case_id(api_config_str)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    fields = [f"{CASE_BEGIN_TAG} {case_id}", timestamp]
    if paddle_version is not None:
        fields.append(f"Paddle {paddle_version}")
    if gpu is not None:
        fields.append(f"GPU {gpu}")
    if worker_pid is not None:
        fields.append(f"PID {worker_pid}")
    if slot is not None:
        fields.append(f"Slot {slot}")
    print(" | ".join(fields), flush=True)
    print(api_config_str, flush=True)
    return case_id


def _case_results(api_config_str):
    return sorted(_case_result_types.get(api_config_str, set()))


def _record_case_comparison(comp, result, matched, total):
    summary = _case_comparisons.setdefault((comp, result), [0, total])
    summary[0] += matched
    summary[1] = max(summary[1], total)


def _format_case_comparison_summary(comp):
    entries = [
        (result, matched, total)
        for (entry_comp, result), (matched, total) in _case_comparisons.items()
        if entry_comp == comp
    ]
    if not entries:
        return None
    total = max(total for _, _, total in entries)
    matched = sum(matched for _, matched, _ in entries)
    failures = list(dict.fromkeys(result for result, _, _ in entries if result != "Identical"))
    result = "+".join(failures) if failures else "Identical"
    return f"{comp} {result} {matched}/{total}"


def write_case_end(status, case_id=None, api_config_str=None, duration_ms=None, results=None):
    """为指定 ID 或配置写入 case 结束 tag，支持多个终态结果。"""
    global _case_has_comp_output
    if api_config_str is not None:
        case_id = get_case_id(api_config_str)
    if results is None and api_config_str is not None:
        results = _case_results(api_config_str)
    result_types = set(results or ())
    if result_types == {"pass"}:
        outcome = "PASS"
    elif result_types == {"skip"}:
        outcome = "SKIP"
    elif result_types:
        outcome = "FAIL"
    else:
        outcome = status.upper()
    if _case_comparisons:
        ordered_comps = [comp for pair in COMP_SUMMARY_PAIRS for comp in pair]
        known_comps = {comp for comp, _ in _case_comparisons}
        ordered_comps.extend(sorted(known_comps - set(ordered_comps)))
        summaries = {
            comp: summary
            for comp in ordered_comps
            if (summary := _format_case_comparison_summary(comp)) is not None
        }
        summary_lines = []
        for pair in COMP_SUMMARY_PAIRS:
            pair_summaries = [summaries.pop(comp) for comp in pair if comp in summaries]
            if pair_summaries:
                summary_lines.append("> COMP SUMMARY | " + " | ".join(pair_summaries))
        for summary in summaries.values():
            summary_lines.append("> COMP SUMMARY | " + summary)
        print("\n".join(summary_lines), flush=True)
        _case_has_comp_output = True
        _case_comparisons.clear()
    if _case_has_comp_output:
        print(flush=True)
        _case_has_comp_output = False
    fields = [f"{CASE_END_TAG} {case_id}", outcome]
    if duration_ms is not None:
        fields.append(f"{duration_ms} ms")
    print(" | ".join(fields), flush=True)
    return get_worker_log_offset()


def get_worker_log_offset(pid=None):
    """返回 worker 日志当前安全文件末尾；当前 worker 优先避免路径 stat。"""
    worker_pid = os.getpid() if pid is None else pid
    try:
        if worker_pid == os.getpid() and stdout_fd is not None:
            return os.fstat(stdout_fd).st_size
        return (TMP_LOG_PATH / f"log_{worker_pid}.log").stat().st_size
    except (FileNotFoundError, OSError):
        return None


def append_case_end_to_worker_log(pid, status, case_id=None, api_config_str=None):
    """worker 停止后补写 synthetic case 结束 tag。"""
    end_line = f"{CASE_END_TAG} {case_id or get_case_id(api_config_str)} | {status.upper()}"
    try:
        log_file = TMP_LOG_PATH / f"log_{pid}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{end_line}\n")
        return log_file.stat().st_size
    except Exception as err:
        print(f"Error writing case end to worker log {pid}: {err}", flush=True)
        return None


@contextmanager
def suppress_startup_output():
    """在 Paddle 和扩展初始化期间静默 stdout/stderr。"""
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


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
    removed_stale_logs=0,
    retest_types=(),
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


def limit_worker_layout(available_gpus, max_workers_per_gpu, pending_cases):
    """按 pending 数 breadth-first 裁剪每张 GPU 的 worker 数。"""
    if pending_cases <= 0:
        return [], {}
    limited = dict.fromkeys(available_gpus, 0)
    remaining = pending_cases
    while remaining > 0:
        allocated = False
        for gpu_id in available_gpus:
            if limited[gpu_id] >= max_workers_per_gpu[gpu_id]:
                continue
            limited[gpu_id] += 1
            remaining -= 1
            allocated = True
            if remaining == 0:
                break
        if not allocated:
            break
    limited = {gpu_id: workers for gpu_id, workers in limited.items() if workers}
    return list(limited), limited


def print_running_header():
    """标记测试由准备阶段进入实际运行阶段。"""
    print("\n--- RUNNING")


def print_case_progress(current, total, status, config, detail=None):
    """打印紧凑的单行 case 状态和进度百分比。"""
    percent = current / total * 100 if total else 100.0
    detail_field = f"{_single_line(detail)} | " if detail else ""
    print(
        f"[{current}/{total} {percent:.1f}%] {status} | {detail_field}{config}",
        flush=True,
    )


def _single_line(value):
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def print_case_notice(status, config, detail=None):
    """打印不推进完成计数的单行 case 事件。"""
    detail_field = f"{_single_line(detail)} | " if detail else ""
    print(f"[case] {status} | {detail_field}{config}", flush=True)


def classify_worker_exit(exitcode, cuda_exit_code, oom_exit_code, torch_exit_code):
    """将 worker 退出码映射到结果日志类型和显示状态。"""
    return {
        cuda_exit_code: ("paddle_cuda", "PADDLE_CUDA"),
        oom_exit_code: ("oom", "OOM"),
        torch_exit_code: ("torch_error", "TORCH_ERROR"),
    }.get(exitcode, ("paddle_crash", "PADDLE_CRASH"))


def print_run_footer(total_case, tested_case, remaining_case, log_counts, elapsed, log_dir):
    """打印统一结果摘要、结束时间和总用时。"""
    counts = {key: value for key, value in (log_counts or {}).items() if not key.startswith("_")}
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
    if (log_counts or {}).get("_multi_classification") and (log_counts or {}).get(
        "_has_multi_result_overlap"
    ):
        print("  Note: In accuracy_stable mode, one config may appear in multiple result sets.")
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


def _copy_inorder_range(file_path, out_f, start_offset, end_offset):
    """复制已完成字节区间，并限制每个物理行最多 200000 字节。"""
    if end_offset > start_offset:
        with file_path.open("rb") as in_f:
            in_f.seek(start_offset)
            remaining = end_offset - start_offset
            while remaining:
                line = in_f.readline(min(200001, remaining))
                if not line:
                    break
                remaining -= len(line)
                if b"gpu_resources.cc:" in line and b"Please NOTE: device:" in line:
                    continue
                if len(line) <= 200000:
                    out_f.write(line)
                    continue
                out_f.write(line[:200000] + b"\n")
                while remaining and not line.endswith(b"\n"):
                    line = in_f.readline(min(4 * 1024 * 1024, remaining))
                    remaining -= len(line)


def mark_inorder_case_complete(pid, completed_offset):
    """记录 worker 最后一个已完成 case 的安全读取上界。"""
    if not _use_worker_tmp_logs:
        return True
    if completed_offset is None:
        return False
    file_path = TMP_LOG_PATH / f"log_{pid}.log"
    _inorder_completed_offsets[file_path] = max(
        completed_offset, _inorder_completed_offsets.get(file_path, 0)
    )
    return True


def flush_completed_inorder_logs():
    """按 worker 一次读取所有已完成 case，并在成功后推进聚合 offset。"""
    if not _inorder_completed_offsets:
        return True
    out_file = TEST_LOG_PATH / "log_inorder.log"
    try:
        with out_file.open("ab") as out_f:
            for file_path, end_offset in list(_inorder_completed_offsets.items()):
                start_offset = _inorder_offsets.get(file_path, 0)
                if end_offset < start_offset:
                    start_offset = 0
                _copy_inorder_range(file_path, out_f, start_offset, end_offset)
                out_f.flush()
                _inorder_offsets[file_path] = end_offset
                _inorder_completed_offsets.pop(file_path, None)
        return True
    except Exception as err:
        print(f"Error flushing case blocks to {out_file}: {err}", flush=True)
        return False


def _aggregate_text_logs(log_files, out_file, cleanup):
    """合并、去重并排序 worker 文本日志。"""
    if not log_files:
        return True
    all_lines = set()
    pending_offsets = {}
    try:
        for file_path in log_files:
            data, _, end_offset = _read_pending_result_bytes(file_path, end=cleanup)
            pending_offsets[file_path] = end_offset
            all_lines.update(line.strip() for line in data.decode().splitlines() if line.strip())
        if all_lines:
            with out_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        return False
    _save_result_offsets(pending_offsets, cleanup)
    return True


def _aggregate_result_logs(cleanup, tmp_exists):
    """聚合所有主结果日志类型。"""
    success = True
    for prefix in LOG_PREFIXES.values():
        log_files = list(TMP_LOG_PATH.glob(f"{prefix}_*.txt")) if tmp_exists else []
        out_file = TEST_LOG_PATH / f"{prefix}.txt"
        success = _aggregate_text_logs(log_files, out_file, cleanup) and success
    return success


def _aggregate_inorder_logs(cleanup, tmp_exists):
    """刷出安全 block，并聚合已停止 worker 的剩余输出。"""
    if not tmp_exists:
        return True
    if not flush_completed_inorder_logs():
        return False
    log_files = sorted(TMP_LOG_PATH.glob("log_*.log"))
    if not log_files:
        return True

    out_file = TEST_LOG_PATH / "log_inorder.log"
    try:
        with out_file.open("ab") as out_f:
            for file_path in log_files:
                try:
                    start_offset = _inorder_offsets.get(file_path, 0)
                    end_offset = file_path.stat().st_size
                    if end_offset < start_offset:
                        start_offset = 0
                    _copy_inorder_range(file_path, out_f, start_offset, end_offset)
                    out_f.flush()
                    if cleanup:
                        _inorder_offsets.pop(file_path, None)
                        _inorder_completed_offsets.pop(file_path, None)
                        file_path.unlink()
                    else:
                        _inorder_offsets[file_path] = end_offset
                except Exception as err:
                    print(f"Error reading {file_path}: {err}", flush=True)
                    return False
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        return False
    return True


def _aggregate_csv_logs(log_files, out_file, header, cleanup):
    """将 worker CSV 分片中的完整行聚合到一个文件。"""
    if not log_files:
        return True

    pending_offsets = {}
    try:
        is_new = not out_file.exists() or out_file.stat().st_size == 0
        with out_file.open("a", newline="") as out_f:
            writer = csv.writer(out_f)
            if is_new:
                writer.writerow(header)
            for file_path in log_files:
                try:
                    data, start_offset, end_offset = _read_pending_result_bytes(
                        file_path, end=cleanup
                    )
                    pending_offsets[file_path] = end_offset
                    reader = csv.reader(io.StringIO(data.decode()))
                    if start_offset == 0:
                        next(reader, None)
                    for row in reader:
                        if row:
                            writer.writerow(row)
                except Exception as err:
                    print(f"Error reading {file_path}: {err}", flush=True)
                    return False
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        return False

    _save_result_offsets(pending_offsets, cleanup)
    return True


def _sort_csv(file_path, columns):
    """按稳定的报告字段排序聚合 CSV。"""
    if not file_path.exists():
        return
    try:
        df = pd.read_csv(file_path, on_bad_lines="warn")
        df = df.sort_values(by=columns, ignore_index=True)
        df.to_csv(file_path, index=False, na_rep="nan")
    except Exception as err:
        print(f"Error arranging {file_path}: {err}", flush=True)


def _count_result_logs():
    """统计结果分类并写出未完成配置列表。"""
    log_counts = {}
    checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
    api_configs = set()
    try:
        with checkpoint_file.open("r") as f:
            api_configs = {line.strip() for line in f if line.strip()}
            log_counts["checkpoint"] = len(api_configs)
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)

    for log_type, prefix in LOG_PREFIXES.items():
        if log_type == "checkpoint":
            continue
        log_file = TEST_LOG_PATH / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            with log_file.open("r") as f:
                lines = {line.strip() for line in f if line.strip()}
                api_configs -= lines
                log_counts[log_type] = len(lines)
        except Exception as err:
            print(f"Error reading {log_file}: {err}", flush=True)

    incomplete_file = TEST_LOG_PATH / "api_config_incomplete.txt"
    if api_configs:
        log_counts["incomplete"] = len(api_configs)
        try:
            with incomplete_file.open("w") as f:
                f.writelines(f"{line}\n" for line in sorted(api_configs))
        except Exception as err:
            print(f"Error writing to {incomplete_file}: {err}", flush=True)
    else:
        incomplete_file.unlink(missing_ok=True)
    return log_counts


def _aggregate_comp_logs(cleanup, tmp_exists):
    """分别聚合每个 comp 维度的结果日志。"""
    comp_tmp_dir = TMP_LOG_PATH / "comp" if tmp_exists else None
    comp_out_dir = TEST_LOG_PATH / "comp"
    has_comp = (comp_tmp_dir and comp_tmp_dir.exists()) or comp_out_dir.exists()
    if not comp_tmp_dir or not comp_tmp_dir.exists():
        return has_comp

    for dim_dir in sorted(comp_tmp_dir.iterdir()):
        if not dim_dir.is_dir():
            continue
        out_dim_dir = comp_out_dir / dim_dir.name
        out_dim_dir.mkdir(parents=True, exist_ok=True)
        for prefix in LOG_PREFIXES.values():
            log_files = list(dim_dir.glob(f"{prefix}_*.txt"))
            _aggregate_text_logs(log_files, out_dim_dir / f"{prefix}.txt", cleanup)
        if cleanup and dim_dir.exists() and not any(dim_dir.iterdir()):
            dim_dir.rmdir()
    if cleanup and comp_tmp_dir.exists() and not any(comp_tmp_dir.iterdir()):
        comp_tmp_dir.rmdir()
    return has_comp


def _find_duplicate_classifications(log_dir):
    """查找同一目录中被多个结果类型分类的配置。"""
    config_to_types: dict[str, list[str]] = {}
    for log_type, prefix in LOG_PREFIXES.items():
        if log_type == "checkpoint":
            continue
        log_file = log_dir / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            with log_file.open("r") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line:
                        config_to_types.setdefault(line, []).append(log_type)
        except Exception:
            pass
    return {config: types for config, types in config_to_types.items() if len(types) > 1}


def _read_log_lines(log_file):
    """将结果文件读取为集合，文件不存在时返回空集合。"""
    if not log_file.exists():
        return set()
    try:
        with log_file.open("r") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)
        return set()


def _sync_comp_main_summary():
    """将各 comp 维度分类合并到主结果摘要。"""
    comp_out_dir = TEST_LOG_PATH / "comp"
    if not comp_out_dir.exists():
        return

    main_lines_by_type = {
        log_type: _read_log_lines(TEST_LOG_PATH / f"{prefix}.txt")
        for log_type, prefix in LOG_PREFIXES.items()
        if log_type != "checkpoint"
    }
    for dim_dir in sorted(comp_out_dir.iterdir()):
        if not dim_dir.is_dir():
            continue
        for log_type, prefix in LOG_PREFIXES.items():
            if log_type == "checkpoint":
                continue
            main_lines_by_type[log_type].update(_read_log_lines(dim_dir / f"{prefix}.txt"))

    for log_type, lines in main_lines_by_type.items():
        log_file = TEST_LOG_PATH / f"{LOG_PREFIXES[log_type]}.txt"
        try:
            if lines:
                with log_file.open("w") as f:
                    f.writelines(f"{line}\n" for line in sorted(lines))
            else:
                log_file.unlink(missing_ok=True)
        except Exception as err:
            print(f"Error writing to {log_file}: {err}", flush=True)


def _check_log_integrity(log_counts, has_comp):
    """检查重复分类并将诊断信息附加到统计结果。"""
    comp_out_dir = TEST_LOG_PATH / "comp"
    if has_comp:
        log_counts["_multi_classification"] = True
        if _find_duplicate_classifications(TEST_LOG_PATH):
            log_counts["_has_multi_result_overlap"] = True
        for dim_dir in sorted(comp_out_dir.iterdir()) if comp_out_dir.exists() else []:
            if not dim_dir.is_dir():
                continue
            duplicates = _find_duplicate_classifications(dim_dir)
            if duplicates:
                log_counts.setdefault("_comp_integrity_errors", []).append(
                    {"scope": f"comp/{dim_dir.name}", "duplicates": duplicates}
                )
        return
    duplicates = _find_duplicate_classifications(TEST_LOG_PATH)
    if duplicates:
        log_counts["_integrity_errors"] = [
            {"scope": "main log directory", "duplicates": duplicates}
        ]


def aggregate_logs(end=False, cleanup=False):
    """聚合 worker 日志，并按需完成统计和完整性检查。"""
    cleanup_tmp = end or cleanup
    tmp_exists = TMP_LOG_PATH.exists()
    if not tmp_exists and not cleanup_tmp:
        TMP_LOG_PATH.mkdir(exist_ok=True)
        return

    tol_file = TEST_LOG_PATH / "tol.csv"
    stable_file = TEST_LOG_PATH / "stable.csv"
    all_success = _aggregate_result_logs(cleanup_tmp, tmp_exists)
    if cleanup_tmp:
        all_success = _aggregate_inorder_logs(cleanup_tmp, tmp_exists) and all_success
    else:
        all_success = flush_completed_inorder_logs() and all_success
    all_success = (
        _aggregate_csv_logs(
            sorted(TMP_LOG_PATH.glob("tol_*.csv")) if tmp_exists else [],
            tol_file,
            TOL_HEADER,
            cleanup_tmp,
        )
        and all_success
    )
    all_success = (
        _aggregate_csv_logs(
            sorted(TMP_LOG_PATH.glob("stable_*.csv")) if tmp_exists else [],
            stable_file,
            STABLE_HEADER,
            cleanup_tmp,
        )
        and all_success
    )

    if not end:
        if (
            cleanup_tmp
            and all_success
            and TMP_LOG_PATH.exists()
            and not any(TMP_LOG_PATH.iterdir())
        ):
            shutil.rmtree(TMP_LOG_PATH)
        return

    _sort_csv(tol_file, ["API", "dtype", "config", "mode"])
    _sort_csv(stable_file, ["API", "dtype", "config", "comp"])
    has_comp = _aggregate_comp_logs(cleanup_tmp, tmp_exists)
    if cleanup_tmp and all_success and TMP_LOG_PATH.exists() and not any(TMP_LOG_PATH.iterdir()):
        shutil.rmtree(TMP_LOG_PATH)
    if has_comp:
        _sync_comp_main_summary()
    log_counts = _count_result_logs()
    _check_log_integrity(log_counts, has_comp)
    return log_counts


def _print_duplicate_classifications(integrity_errors):
    """打印主结果目录中的重复分类。"""
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
        print(
            f"Found {len(duplicates)} duplicated config(s). "
            "Please check log classification, but the test statistics above are still available."
        )
        print("!" * 50 + "\n")


def _print_comp_duplicate_classifications(comp_integrity_errors):
    """按 comp 维度打印重复分类。"""
    for issue in comp_integrity_errors:
        scope = issue["scope"]
        duplicates = issue["duplicates"]
        print("\n" + "!" * 50)
        print(f"WARNING: configs found in multiple log types within {scope}:")
        for config, types in sorted(duplicates.items())[:20]:
            print(f"  {config}")
            print(f"    -> {', '.join(types)}")
        if len(duplicates) > 20:
            print(f"  ... and {len(duplicates) - 20} more")
        print(
            f"Found {len(duplicates)} duplicated config(s) inside {scope}. "
            "Each comp dimension should still be mutually exclusive."
        )
        print("!" * 50 + "\n")


def print_log_info(remaining_case, log_counts=None):
    """打印最终 case 统计和分类完整性告警。"""
    if log_counts is None:
        log_counts = {}
    integrity_errors = log_counts.get("_integrity_errors", [])
    comp_integrity_errors = log_counts.get("_comp_integrity_errors", [])
    is_multi_classification = log_counts.get("_multi_classification")
    has_multi_result_overlap = log_counts.get("_has_multi_result_overlap")
    counts = {key: value for key, value in log_counts.items() if not key.startswith("_")}
    paddle_types = [
        "paddle_error",
        "paddle_accuracy",
        "paddle_bitwise",
        "paddle_cuda",
        "paddle_crash",
    ]
    test_types = ["torch_error", "config_input", "config_parse", "config_convert"]

    if is_multi_classification:
        _print_comp_duplicate_classifications(comp_integrity_errors)
    else:
        _print_duplicate_classifications(integrity_errors)


stdout_fd = None
stderr_fd = None
orig_stdout_fd = None
orig_stderr_fd = None
log_file = None


def redirect_stdio():
    """将 worker 的 stdout 和 stderr 重定向到临时日志。"""
    global stdout_fd, stderr_fd, orig_stdout_fd, orig_stderr_fd, log_file

    log_path = TMP_LOG_PATH / f"log_{os.getpid()}.log"
    log_file = log_path.open("a", encoding="utf-8")
    log_fd = log_file.fileno()

    import sys

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    orig_stdout_fd = os.dup(stdout_fd)
    orig_stderr_fd = os.dup(stderr_fd)

    os.dup2(log_fd, stdout_fd)
    os.dup2(log_fd, stderr_fd)

    sys.stdout = os.fdopen(stdout_fd, "a", buffering=1)
    sys.stderr = os.fdopen(stderr_fd, "a", buffering=1)

    os.close(log_fd)


def restore_stdio():
    """恢复 redirect_stdio 保存的 stdout 和 stderr 文件描述符。"""
    global stdout_fd, stderr_fd, orig_stdout_fd, orig_stderr_fd, log_file
    if log_file is not None:
        log_file.close()
        log_file = None

    if orig_stdout_fd is not None and stdout_fd is not None:
        os.dup2(orig_stdout_fd, stdout_fd)
        os.close(orig_stdout_fd)
        orig_stdout_fd = None

    if orig_stderr_fd is not None and stderr_fd is not None:
        os.dup2(orig_stderr_fd, stderr_fd)
        os.close(orig_stderr_fd)
        orig_stderr_fd = None


def _get_diff(error_msg, abs_pattern, rel_pattern):
    """从断言消息中提取绝对误差和相对误差。"""
    if error_msg == "Identical":
        return 0.0, 0.0

    abs_match = re.search(abs_pattern, error_msg)
    rel_match = re.search(rel_pattern, error_msg)
    if not abs_match or not rel_match:
        return None, None
    try:
        return float(abs_match.group(1)), float(rel_match.group(1))
    except ValueError:
        return None, None


def _append_csv(output_file, header, row):
    """追加一行 CSV，并在需要时创建表头。"""
    try:
        is_new = not output_file.exists() or output_file.stat().st_size == 0
        with output_file.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as err:
        print(f"Error writing to {output_file}: {err}", flush=True)


def log_accuracy_tolerance(
    error_msg,
    api,
    config,
    dtype,
    is_backward=False,
    *,
    tensor_index=0,
    tensor_count=1,
):
    """记录从 assert-close 失败消息中解析出的容差。"""
    mode = "backward" if is_backward else "forward"
    print(
        f"[tolerance] {mode} | tensor {tensor_index + 1}/{tensor_count} | {config}\n{error_msg}",
        flush=True,
    )
    abs_pattern = (
        r"(?:Absolute|Greatest absolute) difference: "
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
    )
    rel_pattern = (
        r"(?:Relative|Greatest relative) difference: "
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
    )
    max_abs_diff, max_rel_diff = _get_diff(error_msg, abs_pattern, rel_pattern)
    row = [api, config, dtype, mode, str(max_abs_diff), str(max_rel_diff)]
    _append_csv(TMP_LOG_PATH / f"tol_{os.getpid()}.csv", TOL_HEADER, row)


def format_comp_line(
    comp,
    result,
    *,
    tensor_index,
    tensor_count,
    **details,
):
    """构造稳定、单行且便于 grep 的 accuracy_stable 比较标记。"""
    phase = "backward" if comp.endswith("B") else "forward"
    base_comp = comp.removesuffix("B")
    actual_kind, actual_run, expected_kind, expected_run = base_comp
    framework_names = {"P": "Paddle", "T": "Torch"}
    actual_source = f"{framework_names[actual_kind]}#{actual_run}"
    expected_source = f"{framework_names[expected_kind]}#{expected_run}"
    fields = [f"> COMP {comp}", result]
    if tensor_index is not None and tensor_count is not None:
        fields.append(f"tensor {tensor_index + 1}/{tensor_count}")
    fields.extend(
        [
            phase,
            f"{actual_source} vs {expected_source}",
        ]
    )
    for key, value in details.items():
        display_value = _single_line(value)
        if key == "reason":
            fields.append(display_value.replace("_", " "))
        else:
            fields.append(f"{key.replace('_', ' ')} {display_value}")
    return " | ".join(fields)


def print_comp_issue(comp, result, **kwargs):
    """打印非 bitwise 比较问题，格式与详细精度误差保持一致。"""
    global _case_has_comp_output
    print("\n" + format_comp_line(comp, result, **kwargs), flush=True)
    tensor_count = kwargs.get("tensor_count")
    if tensor_count is None:
        tensor_count = max(kwargs.get("actual_count", 1), kwargs.get("expected_count", 1))
    _record_case_comparison(comp, result, 0, tensor_count)
    _case_has_comp_output = True


def log_accuracy_stable(
    error_msg,
    api,
    config,
    dtype,
    comp,
    *,
    tensor_index,
    tensor_count,
):
    """记录一个比较模式下的稳定性误差。"""
    global _case_has_comp_output
    if "\n" in error_msg:
        header = format_comp_line(
            comp,
            "paddle_bitwise",
            tensor_index=tensor_index,
            tensor_count=tensor_count,
        )
        print(f"\n{header}\n{error_msg}", flush=True)
        _record_case_comparison(comp, "paddle_bitwise", 0, tensor_count)
        _case_has_comp_output = True
    else:
        _record_case_comparison(comp, error_msg, 1, tensor_count)
    abs_pattern = (
        r"(?:Absolute|Greatest absolute|Max absolute) difference(?: among violations)?: "
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
    )
    rel_pattern = (
        r"(?:Relative|Greatest relative|Max relative) difference(?: among violations)?: "
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
    )
    max_abs_diff, max_rel_diff = _get_diff(error_msg, abs_pattern, rel_pattern)
    row = [
        api,
        config[:MAX_CSV_CONFIG_LENGTH],
        dtype,
        comp,
        str(max_abs_diff),
        str(max_rel_diff),
    ]
    _append_csv(TMP_LOG_PATH / f"stable_{os.getpid()}.csv", STABLE_HEADER, row)
