"""Worker 结果、case 边界、退出协议和标准输出重定向。"""

from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import contextmanager
from datetime import datetime

from . import log_runtime as runtime
from .log_schema import (
    ALL_DIMENSIONS,
    CASE_BEGIN_TAG,
    CASE_END_TAG,
    COMP_SUMMARY_PAIRS,
    COMP_TO_DIMENSION,
    FINAL_RESULT_PRIORITY,
    LOG_PREFIXES,
    TERMINAL_LOG_TYPES,
    LogType,
)

_process_terminal_configs = {}
_case_result_types: dict[str, set[str]] = {}
_written_result_types: dict[str, set[str]] = {}
_case_comparisons: dict[tuple[str, str], list[int]] = {}
_comp_terminal_configs: dict[str, dict[str, str]] = {}
_redirected_fds = None

_FATAL_WORKER_EXIT_CODES = {
    ("paddle_cuda", True): 99,
    ("oom", True): 98,
    ("torch_error", True): 97,
    ("paddle_cuda", False): 96,
    ("oom", False): 95,
    ("torch_error", False): 94,
}
_FATAL_PROGRESS_STATUS = {
    "paddle_cuda": "PADDLE_CUDA",
    "oom": "OOM",
    "torch_error": "TORCH_ERROR",
}


def _reset_worker_state():
    _process_terminal_configs.clear()
    _case_result_types.clear()
    _written_result_types.clear()
    _case_comparisons.clear()
    _comp_terminal_configs.clear()


def get_terminal_log_type(line):
    """返回当前进程对配置的终态分类。"""
    return _process_terminal_configs.get(line.strip())


def fatal_exit_code(log_type, terminal_recorded):
    """编码 fatal 类型以及 worker 是否已写入对应终态。"""
    return _FATAL_WORKER_EXIT_CODES[(log_type, bool(terminal_recorded))]


def classify_exit(exitcode):
    """解码 fatal worker 的日志类型、显示状态和终态写入状态。"""
    for (log_type, terminal_recorded), code in _FATAL_WORKER_EXIT_CODES.items():
        if exitcode == code:
            return log_type, _FATAL_PROGRESS_STATUS[log_type], terminal_recorded
    return "paddle_crash", "PADDLE_CRASH", False


def _record_terminal_type(line, log_type):
    """错误终态可覆盖 pass，pass 不覆盖错误终态。"""
    if log_type != "pass" or line not in _process_terminal_configs:
        _process_terminal_configs[line] = log_type


def write_to_log(log_type: LogType, line):
    """向当前进程的结果日志追加一个配置。"""
    line = line.strip()
    if not line:
        return
    written_types = _written_result_types.get(line)
    if log_type in TERMINAL_LOG_TYPES and written_types is not None and log_type in written_types:
        return
    terminal_log_type = _process_terminal_configs.get(line)
    if log_type == "pass" and terminal_log_type not in (None, "pass"):
        return
    prefix = LOG_PREFIXES[log_type]
    file_path = runtime.result_file(prefix)
    if not runtime.write_line(file_path, line):
        return
    if log_type in TERMINAL_LOG_TYPES:
        _written_result_types.setdefault(line, set()).add(log_type)
        _case_result_types.setdefault(line, set()).add(log_type)
        _record_terminal_type(line, log_type)


def write_to_comp_log(comp, log_type: LogType, line):
    """向一个 comp 维度写入配置并更新分类状态。"""
    dimension = COMP_TO_DIMENSION[comp]
    prefix = LOG_PREFIXES[log_type]
    comp_dir = runtime.RESULT_LOG_PATH / "comp" / dimension
    comp_dir.mkdir(parents=True, exist_ok=True)
    file_path = runtime.result_file(prefix, comp_dir)
    line = line.strip()
    if not line:
        return

    dim_configs = _comp_terminal_configs.setdefault(dimension, {})
    existing = dim_configs.get(line)
    if existing is not None and (existing != "pass" or log_type == "pass"):
        return

    if not runtime.write_line(file_path, line):
        return

    if log_type in TERMINAL_LOG_TYPES:
        dim_configs[line] = log_type
    _case_result_types.setdefault(line, set()).add(log_type)
    if log_type in TERMINAL_LOG_TYPES:
        _record_terminal_type(line, log_type)


def write_stable_passes(line):
    """为尚未分类的稳定性维度和主结果写入 pass。"""
    line = line.strip()
    main_terminal_type = get_terminal_log_type(line)
    for dimension in ALL_DIMENSIONS:
        if line not in _comp_terminal_configs.get(dimension, {}):
            representative = next(
                comp
                for comp, comp_dimension in COMP_TO_DIMENSION.items()
                if comp_dimension == dimension
            )
            write_to_comp_log(representative, "pass", line)
    if main_terminal_type is None:
        print(f"[pass] {line}", flush=True)
        write_to_log("pass", line)


def merge_sanitizer_case_logs(case_log_dir):
    """将一个 sanitizer case 的结果文件追加到 worker 临时日志。"""
    case_tmp_dir = case_log_dir / ".tmp"
    if not case_tmp_dir.exists():
        return
    for child_log in case_tmp_dir.rglob("*"):
        if not child_log.is_file():
            continue
        target_log = runtime.TMP_LOG_PATH / child_log.relative_to(case_tmp_dir)
        target_log.parent.mkdir(parents=True, exist_ok=True)
        with child_log.open("rb") as in_f, target_log.open("ab") as out_f:
            shutil.copyfileobj(in_f, out_f)


def clean_sanitizer_case_logs():
    """删除所有隔离的 sanitizer case 目录。"""
    shutil.rmtree(runtime.TMP_LOG_PATH / "sanitizer", ignore_errors=True)


def _get_case_id(api_config_str):
    """生成 case 边界 tag 使用的稳定短 ID。"""
    return hashlib.sha1(api_config_str.encode("utf-8", errors="replace")).hexdigest()[:12]


def write_case_begin(api_config_str, *, worker_pid, gpu, slot=None, paddle_version=None):
    """写入包含执行元数据的 case 起始行并返回其 ID。"""
    case_id = _get_case_id(api_config_str)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    fields = [f"{CASE_BEGIN_TAG} {case_id}", timestamp]
    if paddle_version is not None:
        fields.append(f"Paddle {paddle_version}")
    fields.append(f"GPU {gpu}")
    fields.append(f"PID {worker_pid}")
    if slot is not None:
        fields.append(f"Slot {slot}")
    print(" | ".join(fields), flush=True)
    print(api_config_str, flush=True)


def _clear_case_result_state(api_config_str):
    """释放已完成 case 的进程内分类状态。"""
    _process_terminal_configs.pop(api_config_str, None)
    _case_result_types.pop(api_config_str, None)
    _written_result_types.pop(api_config_str, None)
    for dim_configs in _comp_terminal_configs.values():
        dim_configs.pop(api_config_str, None)


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


def _get_final_case_result(api_config_str, result_types):
    final_result = next(
        (log_type for log_type in FINAL_RESULT_PRIORITY if log_type in result_types),
        None,
    )
    if final_result is None or api_config_str is None:
        return None, []
    dimensions = [
        dimension
        for dimension in ALL_DIMENSIONS
        if _comp_terminal_configs.get(dimension, {}).get(api_config_str) == final_result
    ]
    return final_result, dimensions


def _print_case_comparison_summary():
    if not _case_comparisons:
        return

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
            summary_lines.append("  " + " | ".join(pair_summaries))
    summary_lines.extend(f"  {summary}" for summary in summaries.values())
    print("\n> COMP SUMMARY\n" + "\n".join(summary_lines), flush=True)
    _case_comparisons.clear()


def write_case_end(status, api_config_str, *, duration_ms=None):
    """写入 case 结束 tag，并释放该配置的进程内分类状态。"""
    result_types = set(_case_result_types.get(api_config_str, ()))
    if result_types == {"pass"}:
        outcome = "PASS"
    elif result_types == {"skip"}:
        outcome = "SKIP"
    elif result_types:
        outcome = "FAIL"
    else:
        outcome = status.upper()
    final_result, final_dimensions = _get_final_case_result(api_config_str, result_types)
    _print_case_comparison_summary()
    if final_result not in (None, "pass", "skip") and final_dimensions:
        print(
            f"> FINAL RESULT | {final_result} | dimensions {','.join(final_dimensions)}",
            flush=True,
        )
    fields = [f"{CASE_END_TAG} {_get_case_id(api_config_str)}", outcome]
    if duration_ms is not None:
        fields.append(f"{duration_ms} ms")
    print(" | ".join(fields) + "\n", flush=True)
    completed_offset = get_worker_log_offset()
    _clear_case_result_state(api_config_str)
    return completed_offset


def get_worker_log_offset():
    """返回当前 worker 日志的安全文件末尾。"""
    try:
        if _redirected_fds is not None:
            return os.fstat(_redirected_fds[0]).st_size
        return (runtime.TMP_LOG_PATH / f"log_{os.getpid()}.log").stat().st_size
    except (FileNotFoundError, OSError):
        return None


def append_case_end_to_worker_log(pid, status, api_config_str):
    """worker 停止后补写 synthetic case 结束 tag。"""
    end_line = f"{CASE_END_TAG} {_get_case_id(api_config_str)} | {status.upper()}"
    try:
        log_file = runtime.TMP_LOG_PATH / f"log_{pid}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("ab") as f:
            f.write(f"\n{end_line}\n\n".encode())
            return f.tell()
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


def redirect_stdio():
    """将 worker 的 stdout 和 stderr 重定向到临时日志。"""
    global _redirected_fds

    import sys

    if _redirected_fds is not None:
        raise RuntimeError("stdout/stderr are already redirected")
    sys.stdout.flush()
    sys.stderr.flush()
    target_stdout_fd = sys.stdout.fileno()
    target_stderr_fd = sys.stderr.fileno()
    log_path = runtime.TMP_LOG_PATH / f"log_{os.getpid()}.log"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        saved_stdout_fd = os.dup(target_stdout_fd)
        saved_stderr_fd = os.dup(target_stderr_fd)
        os.dup2(log_fd, target_stdout_fd)
        os.dup2(log_fd, target_stderr_fd)
    except Exception:
        if saved_stdout_fd is not None:
            os.dup2(saved_stdout_fd, target_stdout_fd)
            os.close(saved_stdout_fd)
        if saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, target_stderr_fd)
            os.close(saved_stderr_fd)
        raise
    finally:
        os.close(log_fd)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    _redirected_fds = (
        target_stdout_fd,
        target_stderr_fd,
        saved_stdout_fd,
        saved_stderr_fd,
    )


def restore_stdio():
    """恢复 redirect_stdio 保存的 stdout 和 stderr 文件描述符。"""
    global _redirected_fds

    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    if _redirected_fds is None:
        return
    stdout_fd, stderr_fd, orig_stdout_fd, orig_stderr_fd = _redirected_fds
    _redirected_fds = None
    saved_pairs = ((orig_stdout_fd, stdout_fd), (orig_stderr_fd, stderr_fd))
    first_error = None
    for saved_fd, target_fd in saved_pairs:
        if saved_fd is None or target_fd is None:
            continue
        try:
            os.dup2(saved_fd, target_fd)
        except OSError as err:
            first_error = first_error or err
        finally:
            os.close(saved_fd)
    if first_error is not None:
        raise first_error
