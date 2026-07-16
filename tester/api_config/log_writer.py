from __future__ import annotations

import csv
import io
import os
import re
import shutil
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
    "T1P1": "accuracy",
    "T2P2": "accuracy",
    "T1P2": "accuracy",
    "T2P1": "accuracy",
    "T1P1B": "accuracy_backward",
    "T2P2B": "accuracy_backward",
    "T1P2B": "accuracy_backward",
    "T2P1B": "accuracy_backward",
    "T1T2": "torch_stable",
    "T1T2B": "torch_stable_backward",
    "P1P2": "paddle_stable",
    "P1P2B": "paddle_stable_backward",
}
ALL_DIMENSIONS = sorted(set(COMP_TO_DIMENSION.values()))
TOL_HEADER = ["API", "config", "dtype", "mode", "max_abs_diff", "max_rel_diff"]
STABLE_HEADER = ["API", "config", "dtype", "comp", "max_abs_diff", "max_rel_diff"]

_use_worker_tmp_logs = False

_process_file_handlers = {}
_aggregated_offsets = {}
_process_terminal_configs = {}
# 每维度的 terminal configs 追踪: dimension -> {config_line -> log_type}
_comp_terminal_configs: dict[str, dict[str, str]] = {}

# Command line arguments configuration
# Used in engine.py
CMD_CONFIG = None


def get_cfg():
    global CMD_CONFIG
    return CMD_CONFIG


def set_cfg(cfg):
    global CMD_CONFIG
    if cfg.id != "":
        cfg.id = "_" + cfg.id
    CMD_CONFIG = cfg


def _reset_runtime():
    close_process_files()
    _aggregated_offsets.clear()
    _process_terminal_configs.clear()
    _comp_terminal_configs.clear()


def init_log(log_dir=None, *, worker_tmp_logs=False):
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
    return TMP_LOG_PATH


def get_sanitizer_case_log_dir(slot_index, pid):
    return TMP_LOG_PATH / "sanitizer" / f"slot_{slot_index}_{pid}"


def merge_sanitizer_case_logs(case_log_dir):
    case_tmp_dir = case_log_dir / ".tmp"
    if not case_tmp_dir.exists():
        return
    for child_log in case_tmp_dir.iterdir():
        if not child_log.is_file():
            continue
        target_log = TMP_LOG_PATH / child_log.name
        with child_log.open("rb") as in_f, target_log.open("ab") as out_f:
            shutil.copyfileobj(in_f, out_f)


def cleanup_sanitizer_tmp_dir():
    shutil.rmtree(TMP_LOG_PATH / "sanitizer", ignore_errors=True)


def close_process_files():
    """关闭本进程持有的所有文件句柄"""
    global _process_file_handlers
    for handler in _process_file_handlers.values():
        try:
            handler.close()
        except Exception as err:
            print(f"Error closing process file: {err}", flush=True)
    _process_file_handlers = {}


def has_terminal_log(line):
    return line.strip() in _process_terminal_configs


def get_terminal_log_type(line):
    return _process_terminal_configs.get(line.strip())


def write_checkpoint(line):
    line = line.strip()
    write_to_log("checkpoint", line)
    _process_terminal_configs.pop(line, None)


def write_terminal_log(log_type: LogType, line):
    write_to_log(log_type, line)
    write_checkpoint(line)


def _get_log_file(log_type: LogType):
    prefix = LOG_PREFIXES[log_type]
    if not _use_worker_tmp_logs:
        cfg = get_cfg()
        filename = f"{prefix}{cfg.id}.txt" if cfg else f"{prefix}.txt"
        return TEST_LOG_PATH / filename
    pid = os.getpid()
    return TMP_LOG_PATH / f"{prefix}_{pid}.txt"


def _open_handler(file_path):
    if file_path not in _process_file_handlers:
        _process_file_handlers[file_path] = file_path.open("a", buffering=1)
    return _process_file_handlers[file_path]


def _write_line(file_path, line):
    try:
        _open_handler(file_path).write(line + "\n")
        return True
    except Exception as err:
        print(f"Error writing to {file_path}: {err}", flush=True)
        return False


def write_to_log(log_type: LogType, line):
    """添加单条日志到当前进程的日志文件"""
    line = line.strip()
    if not line:
        return
    terminal_log_type = _process_terminal_configs.get(line)
    if _use_worker_tmp_logs and log_type == "pass" and terminal_log_type not in (None, "pass"):
        return
    try:
        file_path = _get_log_file(log_type)
    except Exception as err:
        print(f"Error resolving log file for {log_type}: {err}", flush=True)
        return
    if _write_line(file_path, line) and log_type in TERMINAL_LOG_TYPES and _use_worker_tmp_logs:
        _process_terminal_configs[line] = log_type


def has_comp_terminal_log(dimension, line):
    """检查某个 comp 维度下是否已有终态分类"""
    line = line.strip()
    dim_configs = _comp_terminal_configs.get(dimension)
    if dim_configs is None:
        return False
    return line in dim_configs


def _get_comp_file(dimension, log_type: LogType):
    prefix = LOG_PREFIXES[log_type]
    if _use_worker_tmp_logs:
        comp_dir = TMP_LOG_PATH / "comp" / dimension
        comp_dir.mkdir(parents=True, exist_ok=True)
        return comp_dir / f"{prefix}_{os.getpid()}.txt"
    comp_dir = TEST_LOG_PATH / "comp" / dimension
    comp_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_cfg()
    filename = f"{prefix}{cfg.id}.txt" if cfg else f"{prefix}.txt"
    return comp_dir / filename


def write_to_comp_log(comp, log_type: LogType, line):
    """写入 comp 维度的日志文件，同时更新主 _process_terminal_configs"""
    try:
        dimension = COMP_TO_DIMENSION[comp]
        file_path = _get_comp_file(dimension, log_type)
    except Exception as err:
        print(f"Error resolving comp log file for {comp}/{log_type}: {err}", flush=True)
        return

    line = line.strip()
    if not line:
        return

    dim_configs = _comp_terminal_configs.setdefault(dimension, {})
    existing = dim_configs.get(line)
    if existing is not None and existing != "pass" and log_type != existing:
        return

    if not _write_line(file_path, line):
        return

    if log_type in TERMINAL_LOG_TYPES:
        dim_configs[line] = log_type
    if log_type in TERMINAL_LOG_TYPES and _use_worker_tmp_logs:
        _process_terminal_configs[line] = log_type


def read_log(log_type: LogType):
    """读取文件所有行，返回集合"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
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


def cleanup_uncheckpointed_result_logs():
    """Remove result rows that were written before checkpoint during an interrupted run."""
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
                with log_file.open("w") as f:
                    f.writelines(new_lines)
        except Exception as err:
            print(f"Error cleaning {log_file}: {err}", flush=True)
    return removed


def _read_pending_bytes(file_path, end=False):
    offset = _aggregated_offsets.get(file_path, 0)
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


def _save_offset(file_path, offset, clear=False):
    if clear:
        _aggregated_offsets.pop(file_path, None)
    else:
        _aggregated_offsets[file_path] = offset


def _save_offsets(pending_offsets, cleanup):
    for file_path, offset in pending_offsets.items():
        _save_offset(file_path, offset, clear=cleanup)
        if cleanup:
            file_path.unlink()


def _read_lines(log_files, cleanup):
    all_lines = set()
    pending_offsets = {}
    for file_path in log_files:
        try:
            data, _, end_offset = _read_pending_bytes(file_path, end=cleanup)
            pending_offsets[file_path] = end_offset
            all_lines.update(line.strip() for line in data.decode().splitlines() if line.strip())
        except Exception as err:
            print(f"Error reading {file_path}: {err}", flush=True)
            return set(), {}, False
    return all_lines, pending_offsets, True


def _agg_text(log_files, out_file, cleanup):
    if not log_files:
        return True
    all_lines, pending_offsets, success = _read_lines(log_files, cleanup)
    if not success:
        return False
    try:
        if all_lines:
            with out_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        out_file.unlink(missing_ok=True)
        return False
    _save_offsets(pending_offsets, cleanup)
    return True


def _agg_results(cleanup, tmp_exists):
    success = True
    for prefix in LOG_PREFIXES.values():
        log_files = list(TMP_LOG_PATH.glob(f"{prefix}_*.txt")) if tmp_exists else []
        out_file = TEST_LOG_PATH / f"{prefix}.txt"
        success = _agg_text(log_files, out_file, cleanup) and success
    return success


def _agg_inorder(cleanup, tmp_exists):
    log_files = sorted(TMP_LOG_PATH.glob("log_*.log")) if tmp_exists else []
    if not log_files:
        return True

    out_file = TEST_LOG_PATH / "log_inorder.log"
    pending_offsets = {}
    try:
        with out_file.open("ab") as out_f:
            for file_path in log_files:
                try:
                    data, _, end_offset = _read_pending_bytes(file_path, end=cleanup)
                    pending_offsets[file_path] = end_offset
                    in_f = io.BytesIO(data)
                    while True:
                        lines = in_f.readlines(4 * 1024 * 1024)
                        if not lines:
                            break
                        for line in lines:
                            out_f.write(line[:200000] + b"\n" if len(line) > 200000 else line)
                except Exception as err:
                    print(f"Error reading {file_path}: {err}", flush=True)
                    out_file.unlink(missing_ok=True)
                    return False
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        out_file.unlink(missing_ok=True)
        return False

    _save_offsets(pending_offsets, cleanup)
    return True


def _agg_csv(log_files, out_file, header, cleanup):
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
                    data, start_offset, end_offset = _read_pending_bytes(file_path, end=cleanup)
                    pending_offsets[file_path] = end_offset
                    reader = csv.reader(io.StringIO(data.decode()))
                    if start_offset == 0:
                        next(reader, None)
                    for row in reader:
                        if row:
                            writer.writerow(row)
                except Exception as err:
                    print(f"Error reading {file_path}: {err}", flush=True)
                    out_file.unlink(missing_ok=True)
                    return False
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        out_file.unlink(missing_ok=True)
        return False

    _save_offsets(pending_offsets, cleanup)
    return True


def _sort_csv(file_path, columns):
    if not file_path.exists():
        return
    try:
        df = pd.read_csv(file_path, on_bad_lines="warn")
        df = df.sort_values(by=columns, ignore_index=True)
        df.to_csv(file_path, index=False, na_rep="nan")
    except Exception as err:
        print(f"Error arranging {file_path}: {err}", flush=True)


def _count_logs():
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


def _agg_comp(cleanup, tmp_exists):
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
            _agg_text(log_files, out_dim_dir / f"{prefix}.txt", cleanup)
        if cleanup and dim_dir.exists() and not any(dim_dir.iterdir()):
            dim_dir.rmdir()
    if cleanup and comp_tmp_dir.exists() and not any(comp_tmp_dir.iterdir()):
        comp_tmp_dir.rmdir()
    return has_comp


def _scan_dups(log_dir):
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


def _add_dups(log_counts, scope, duplicates):
    if duplicates:
        log_counts.setdefault("_integrity_errors", []).append(
            {"scope": scope, "duplicates": duplicates}
        )


def _read_log_lines(log_file):
    if not log_file.exists():
        return set()
    try:
        with log_file.open("r") as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)
        return set()


def _sync_comp_main_summary():
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


def _check_logs(log_counts, has_comp):
    comp_out_dir = TEST_LOG_PATH / "comp"
    if has_comp:
        log_counts["_multi_classification"] = True
        if _scan_dups(TEST_LOG_PATH):
            log_counts["_has_multi_result_overlap"] = True
        for dim_dir in sorted(comp_out_dir.iterdir()) if comp_out_dir.exists() else []:
            if not dim_dir.is_dir():
                continue
            duplicates = _scan_dups(dim_dir)
            if duplicates:
                log_counts.setdefault("_comp_integrity_errors", []).append(
                    {"scope": f"comp/{dim_dir.name}", "duplicates": duplicates}
                )
        return
    _add_dups(log_counts, "main log directory", _scan_dups(TEST_LOG_PATH))


def _clean_tmp(cleanup, all_success):
    if cleanup and all_success and TMP_LOG_PATH.exists() and not any(TMP_LOG_PATH.iterdir()):
        shutil.rmtree(TMP_LOG_PATH)


def aggregate_logs(end=False, cleanup=False):
    """聚合所有相同类型的日志文件"""
    cleanup_tmp = end or cleanup
    tmp_exists = TMP_LOG_PATH.exists()
    if not tmp_exists and not cleanup_tmp:
        TMP_LOG_PATH.mkdir(exist_ok=True)
        return

    tol_file = TEST_LOG_PATH / "tol.csv"
    stable_file = TEST_LOG_PATH / "stable.csv"
    all_success = _agg_results(cleanup_tmp, tmp_exists)
    all_success = _agg_inorder(cleanup_tmp, tmp_exists) and all_success
    all_success = (
        _agg_csv(
            sorted(TMP_LOG_PATH.glob("tol_*.csv")) if tmp_exists else [],
            tol_file,
            TOL_HEADER,
            cleanup_tmp,
        )
        and all_success
    )
    all_success = (
        _agg_csv(
            sorted(TMP_LOG_PATH.glob("stable_*.csv")) if tmp_exists else [],
            stable_file,
            STABLE_HEADER,
            cleanup_tmp,
        )
        and all_success
    )

    if not end:
        _clean_tmp(cleanup_tmp, all_success)
        return

    _sort_csv(tol_file, ["API", "dtype", "config", "mode"])
    _sort_csv(stable_file, ["API", "dtype", "config", "comp"])
    has_comp = _agg_comp(cleanup_tmp, tmp_exists)
    _clean_tmp(cleanup_tmp, all_success)
    if has_comp:
        _sync_comp_main_summary()
    log_counts = _count_logs()
    _check_logs(log_counts, has_comp)
    return log_counts


def _visible_counts(log_counts):
    return {key: value for key, value in log_counts.items() if not key.startswith("_")}


def _sum_counts(log_counts, log_types):
    return sum(log_counts.get(log_type, 0) for log_type in log_types)


def _print_dups(integrity_errors):
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


def _print_comp_dups(comp_integrity_errors):
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
    """打印日志统计信息"""
    if log_counts is None:
        log_counts = {}
    integrity_errors = log_counts.get("_integrity_errors", [])
    comp_integrity_errors = log_counts.get("_comp_integrity_errors", [])
    is_multi_classification = log_counts.get("_multi_classification")
    has_multi_result_overlap = log_counts.get("_has_multi_result_overlap")
    counts = _visible_counts(log_counts)
    paddle_types = [
        "paddle_error",
        "paddle_accuracy",
        "paddle_bitwise",
        "paddle_cuda",
        "paddle_crash",
    ]
    test_types = ["torch_error", "config_input", "config_parse", "config_convert"]

    print("\n" + "=" * 50)
    print("Test Case Statistics".center(50))
    print("=" * 50)
    print(f"{'Remaining cases':<30}: {remaining_case:>8}")
    print(f"{'Tested cases':<30}: {counts.get('checkpoint', 0):>8}")
    print(f"{'Pass cases':<30}: {counts.get('pass', 0):>8}")
    print(f"{'Skip cases':<30}: {counts.get('skip', 0):>8}")
    print(f"{'Paddle issue cases':<30}: {_sum_counts(counts, paddle_types):>8}")
    print(f"{'Test issue cases':<30}: {_sum_counts(counts, test_types):>8}")
    print(f"{'Retest cases':<30}: {_sum_counts(counts, ['oom', 'timeout']):>8}")
    if counts:
        print("-" * 50)
        print("Log Type Breakdown:")
        if is_multi_classification and has_multi_result_overlap:
            print("  Note: In accuracy_stable mode, one config may appear in multiple result sets.")
        for log_type, count in counts.items():
            print(f"  {log_type:<28}: {count:>8}")
    print("=" * 50 + "\n")
    if is_multi_classification:
        _print_comp_dups(comp_integrity_errors)
    else:
        _print_dups(integrity_errors)


stdout_fd = None
stderr_fd = None
orig_stdout_fd = None
orig_stderr_fd = None
log_file = None


def redirect_stdio():
    """执行 stdout 和 stderr 的重定向"""
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
    """恢复 stdout 和 stderr 的重定向"""
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
    try:
        is_new = not output_file.exists() or output_file.stat().st_size == 0
        with output_file.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as err:
        print(f"Error writing to {output_file}: {err}", flush=True)


def log_accuracy_tolerance(error_msg, api, config, dtype, is_backward=False):
    """从 torch.testing.assert_close 的异常消息中提取最大绝对误差和相对误差
    将误差数据记录到 CSV 文件
    """
    mode = "backward" if is_backward else "forward"
    print(f"mode={mode} {config}\n{error_msg}", flush=True)
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


def log_accuracy_stable(error_msg, api, config, dtype, comp):
    print(f"comp={comp} {config}\n{error_msg}", flush=True)
    abs_pattern = (
        r"(?:Absolute|Greatest absolute|Max absolute) difference(?: among violations)?: "
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
    )
    rel_pattern = (
        r"(?:Relative|Greatest relative|Max relative) difference(?: among violations)?: "
        r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
    )
    max_abs_diff, max_rel_diff = _get_diff(error_msg, abs_pattern, rel_pattern)
    row = [api, config, dtype, comp, str(max_abs_diff), str(max_rel_diff)]
    _append_csv(TMP_LOG_PATH / f"stable_{os.getpid()}.csv", STABLE_HEADER, row)
