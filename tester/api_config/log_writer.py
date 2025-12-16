from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path

import pandas as pd

# 日志文件路径
DIR_PATH = Path(__file__).resolve()
DIR_PATH = DIR_PATH.parent.parent.parent
TEST_LOG_PATH = DIR_PATH / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"

# 日志类型和对应的文件，可在下方进行注册
LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "numpy_error": "api_config_numpy_error",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "accuracy_diff": "api_config_accuracy_diff",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
    "match_error": "api_config_match_error",
    "cuda_error": "api_config_cuda_error",
}

_is_engineV2 = False

_process_file_handlers = {}

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


def set_test_log_path(log_dir):
    global TEST_LOG_PATH, TMP_LOG_PATH
    TEST_LOG_PATH = DIR_PATH / log_dir
    TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
    TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"


def set_engineV2():
    global _is_engineV2
    _is_engineV2 = True
    TMP_LOG_PATH.mkdir(exist_ok=True)


def close_process_files():
    """关闭本进程持有的所有文件句柄"""
    global _process_file_handlers
    for handler in _process_file_handlers.values():
        try:
            handler.close()
        except Exception as err:
            print(f"Error closing process file: {err}", flush=True)
    _process_file_handlers = {}


def get_log_file(log_type: str):
    """获取指定日志类型和PID对应的日志文件路径"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
    prefix = LOG_PREFIXES[log_type]
    if not _is_engineV2:
        cfg = get_cfg()
        filename = f"{prefix}{cfg.id}.txt" if cfg else f"{prefix}.txt"
        return TEST_LOG_PATH / filename
    pid = os.getpid()
    return TMP_LOG_PATH / f"{prefix}_{pid}.txt"


def write_to_log(log_type, line):
    """添加单条日志到当前进程的日志文件"""
    line = line.strip()
    if not line:
        return
    file_path = get_log_file(log_type)
    try:
        if file_path not in _process_file_handlers:
            _process_file_handlers[file_path] = file_path.open("a", buffering=1)
        handler = _process_file_handlers[file_path]
        handler.write(line + "\n")
    except Exception as err:
        print(f"Error writing to {file_path}: {err}", flush=True)


def read_log(log_type):
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


def aggregate_logs(end=False):
    """聚合所有相同类型的日志文件"""
    if not TMP_LOG_PATH.exists() and not end:
        TMP_LOG_PATH.mkdir(exist_ok=True)
        return

    all_success = True
    for prefix in LOG_PREFIXES.values():
        log_files = list(TMP_LOG_PATH.glob(f"{prefix}_*.txt"))
        if not log_files:
            continue

        prefix_success = True
        all_lines = set()
        for file_path in log_files:
            try:
                with file_path.open("r") as f:
                    all_lines.update(line.strip() for line in f if line.strip())
            except Exception as err:
                print(f"Error reading {file_path}: {err}", flush=True)
                prefix_success = False
                break
        if not prefix_success:
            all_success = False
            continue

        aggregated_file = TEST_LOG_PATH / f"{prefix}.txt"
        try:
            with aggregated_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
        except Exception as err:
            print(f"Error writing to {aggregated_file}: {err}", flush=True)
            prefix_success = False

        if not prefix_success:
            aggregated_file.unlink(missing_ok=True)
            all_success = False
        else:
            for file_path in log_files:
                if end:
                    file_path.unlink()
                else:
                    file_path.write_bytes(b"")

    log_success = True
    log_file = TEST_LOG_PATH / "log_inorder.log"
    tmp_log_files = sorted(TMP_LOG_PATH.glob("log_*.log"))
    BUFFER_SIZE = 4 * 1024 * 1024
    try:
        with log_file.open("ab") as out_f:
            for file_path in tmp_log_files:
                try:
                    with file_path.open("rb") as in_f:
                        while True:
                            lines = in_f.readlines(BUFFER_SIZE)
                            if not lines:
                                break
                            for line in lines:
                                if len(line) > 200000:  # 如果行长度超过200000字节,截断
                                    # print(
                                    #     f"Truncating long line ({len(line)} bytes) in {file_path.name}"
                                    # )
                                    out_f.write(line[:200000] + b"\n")
                                else:
                                    out_f.write(line)
                except Exception as err:
                    print(f"Error reading {file_path}: {err}", flush=True)
                    log_success = False
                    break
    except Exception as err:
        print(f"Error writing to {log_file}: {err}", flush=True)
        log_success = False

    if not log_success:
        log_file.unlink(missing_ok=True)
        all_success = False
    else:
        for file_path in tmp_log_files:
            if end:
                file_path.unlink()
            else:
                file_path.write_bytes(b"")

    tol_success = True
    tol_file = TEST_LOG_PATH / "tol.csv"
    tmp_tol_files = sorted(TMP_LOG_PATH.glob("tol_*.csv"))
    if tmp_tol_files:
        try:
            with tol_file.open("a", newline="") as out_f:
                writer = csv.writer(out_f)
                if not tol_file.exists() or tol_file.stat().st_size == 0:
                    writer.writerow(
                        [
                            "API",
                            "config",
                            "dtype",
                            "mode",
                            "max_abs_diff",
                            "max_rel_diff",
                        ]
                    )
                for file_path in tmp_tol_files:
                    try:
                        with file_path.open("r") as in_f:
                            reader = csv.reader(in_f)
                            next(reader, None)
                            for row in reader:
                                if row:  # 确保行不为空
                                    writer.writerow(row)
                    except Exception as err:
                        print(f"Error reading {file_path}: {err}", flush=True)
                        tol_success = False
                        break
        except Exception as err:
            print(f"Error writing to {tol_file}: {err}", flush=True)
            tol_success = False

        if not tol_success:
            tol_file.unlink(missing_ok=True)
            all_success = False
        else:
            for file_path in tmp_tol_files:
                if end:
                    file_path.unlink()
                else:
                    file_path.write_bytes(b"")

    stable_success = True
    stable_file = TEST_LOG_PATH / "stable.csv"
    tmp_stable_files = sorted(TMP_LOG_PATH.glob("stable_*.csv"))
    if tmp_stable_files:
        try:
            with stable_file.open("a", newline="") as out_f:
                writer = csv.writer(out_f)
                if not stable_file.exists() or stable_file.stat().st_size == 0:
                    writer.writerow(
                        [
                            "API",
                            "config",
                            "dtype",
                            "comp",
                            "max_abs_diff",
                            "max_rel_diff",
                        ]
                    )
                for file_path in tmp_stable_files:
                    try:
                        with file_path.open("r") as in_f:
                            reader = csv.reader(in_f)
                            next(reader, None)
                            for row in reader:
                                if row:  # 确保行不为空
                                    writer.writerow(row)
                    except Exception as err:
                        print(f"Error reading {file_path}: {err}", flush=True)
                        stable_success = False
                        break
        except Exception as err:
            print(f"Error writing to {tol_file}: {err}", flush=True)
            stable_success = False

        if not stable_success:
            stable_file.unlink(missing_ok=True)
            all_success = False
        else:
            for file_path in tmp_stable_files:
                if end:
                    file_path.unlink()
                else:
                    file_path.write_bytes(b"")

    if end:
        if all_success and not os.listdir(TMP_LOG_PATH):
            shutil.rmtree(TMP_LOG_PATH)

        if tol_file.exists():
            try:
                df = pd.read_csv(tol_file, on_bad_lines="warn")
                # df = df.drop_duplicates(subset=["config", "mode"], keep="last")
                df = df.sort_values(by=["API", "dtype", "config", "mode"], ignore_index=True)
                df.to_csv(tol_file, index=False, na_rep="nan")
            except Exception as err:
                print(f"Error arranging {tol_file}: {err}", flush=True)

        if stable_file.exists():
            try:
                df = pd.read_csv(stable_file, on_bad_lines="warn")
                # df = df.drop_duplicates(subset=["config", "comp"], keep="last")
                df = df.sort_values(by=["API", "dtype", "config", "comp"], ignore_index=True)
                df.to_csv(stable_file, index=False, na_rep="nan")
            except Exception as err:
                print(f"Error arranging {stable_file}: {err}", flush=True)

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

        if api_configs:
            log_counts["skip"] = len(api_configs)
            skip_file = TEST_LOG_PATH / "api_config_skip.txt"
            try:
                with skip_file.open("w") as f:
                    f.writelines(f"{line}\n" for line in sorted(api_configs))
            except Exception as err:
                print(f"Error writing to {skip_file}: {err}", flush=True)
        return log_counts


def print_log_info(all_case, log_counts=None):
    """打印日志统计信息"""
    if log_counts is None:
        log_counts = {}
    test_case = log_counts.get("checkpoint", 0)
    pass_case = log_counts.get("pass", 0)
    fail_case = sum(
        log_counts.get(log_type, 0)
        for log_type in [
            "paddle_error",
            "accuracy_error",
            "accuracy_diff",
            "timeout",
            "crash",
            "oom",
            "cuda_error",
        ]
    )
    skip_case = sum(
        log_counts.get(log_type, 0)
        for log_type in [
            "numpy_error",
            "torch_error",
            "paddle_to_torch_failed",
            "match_error",
        ]
    )

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Test Case Statistics".center(50))
    print("=" * 50)
    print(f"{'Total cases':<30}: {all_case}")
    print(f"{'Tested cases':<30}: {test_case}")
    print(f"{'Passed cases':<30}: {pass_case}")
    print(f"{'Failed cases':<30}: {fail_case}")
    print(f"{'Skipped cases':<30}: {skip_case}")
    if log_counts:
        print("-" * 50)
        print("Log Type Breakdown:")
        for log_type, count in log_counts.items():
            print(f"  {log_type:<28}: {count}")
    print("=" * 50 + "\n")


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


def log_accuracy_tolerance(error_msg, api, config, dtype, is_backward=False):
    """从 torch.testing.assert_close 的异常消息中提取最大绝对误差和相对误差
    将误差数据记录到 CSV 文件
    """
    output_file = TMP_LOG_PATH / f"tol_{os.getpid()}.csv"
    mode = "backward" if is_backward else "forward"
    print(f"[{mode}] {config}\n{error_msg}", flush=True)

    if error_msg == "Identical":
        max_abs_diff = 0.0
        max_rel_diff = 0.0
    else:
        max_abs_diff = None
        max_rel_diff = None

        # 使用正则表达式提取误差值
        abs_pattern = (
            r"(?:Absolute|Greatest absolute) difference: (\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
        )
        rel_pattern = (
            r"(?:Relative|Greatest relative) difference: (\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
        )
        abs_match = re.search(abs_pattern, error_msg)
        rel_match = re.search(rel_pattern, error_msg)

        if abs_match and rel_match:
            try:
                max_abs_diff = float(abs_match.group(1))
                max_rel_diff = float(rel_match.group(1))
            except ValueError:
                pass

    row = [api, config, dtype, mode, str(max_abs_diff), str(max_rel_diff)]
    try:
        with open(output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not output_file.exists() or output_file.stat().st_size == 0:
                writer.writerow(
                    [
                        "API",
                        "config",
                        "dtype",
                        "mode",
                        "max_abs_diff",
                        "max_rel_diff",
                    ]
                )
            writer.writerow(row)
    except Exception as err:
        print(f"Error writing to {output_file}: {err}", flush=True)


def log_accuracy_stable(error_msg, api, config, dtype, comp):
    output_file = TMP_LOG_PATH / f"stable_{os.getpid()}.csv"
    print(f"[{comp}] {config}\n{error_msg}", flush=True)

    if error_msg == "Identical":
        max_abs_diff = 0.0
        max_rel_diff = 0.0
    else:
        max_abs_diff = None
        max_rel_diff = None

        # 使用正则表达式提取误差值
        abs_pattern = r"(?:Absolute|Greatest absolute|Max absolute) difference(?: among violations)?: (\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
        rel_pattern = r"(?:Relative|Greatest relative|Max relative) difference(?: among violations)?: (\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
        abs_match = re.search(abs_pattern, error_msg)
        rel_match = re.search(rel_pattern, error_msg)

        if abs_match and rel_match:
            try:
                max_abs_diff = float(abs_match.group(1))
                max_rel_diff = float(rel_match.group(1))
            except ValueError:
                pass

    row = [api, config, dtype, comp, str(max_abs_diff), str(max_rel_diff)]
    try:
        with open(output_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not output_file.exists() or output_file.stat().st_size == 0:
                writer.writerow(
                    [
                        "API",
                        "config",
                        "dtype",
                        "comp",
                        "max_abs_diff",
                        "max_rel_diff",
                    ]
                )
            writer.writerow(row)
    except Exception as err:
        print(f"Error writing to {output_file}: {err}", flush=True)
