import os
import shutil
from pathlib import Path

# 日志文件路径
DIR_PATH = Path(__file__).resolve()
DIR_PATH = DIR_PATH.parent.parent.parent
TEST_LOG_PATH = DIR_PATH / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"

# 日志类型和对应的文件
LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
}

is_engineV2 = False


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
    global is_engineV2
    is_engineV2 = True
    TMP_LOG_PATH.mkdir(exist_ok=True)


def get_log_file(log_type: str):
    """获取指定日志类型和PID对应的日志文件路径"""
    if log_type not in LOG_PREFIXES:
        raise ValueError(f"Invalid log type: {log_type}")
        
    prefix = LOG_PREFIXES[log_type]

    if not is_engineV2:
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
        with file_path.open("a") as f:
            f.write(line + "\n")
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
            return set(line.strip() for line in f if line.strip())
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

    for prefix in LOG_PREFIXES.values():
        log_files = list(TMP_LOG_PATH.glob(f"{prefix}_*.txt"))
        if not log_files:
            continue

        all_lines = set()
        for file_path in log_files:
            try:
                with file_path.open("r") as f:
                    all_lines.update(line.strip() for line in f if line.strip())
                os.remove(str(file_path))
            except Exception as err:
                print(f"Error reading {file_path}: {err}", flush=True)

        aggregated_file = TEST_LOG_PATH / f"{prefix}.txt"
        try:
            with aggregated_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
        except Exception as err:
            print(f"Error writing to {aggregated_file}: {err}", flush=True)

    log_file = TEST_LOG_PATH / f"log_inorder.log"
    tmp_files = sorted(TMP_LOG_PATH.glob(f"log_*.log"))
    try:
        with log_file.open("ab") as out_f:
            for file_path in tmp_files:
                try:
                    with file_path.open("rb") as in_f:
                        for line in in_f:
                            if len(line) > 10000:
                                print(f"Truncating long line ({len(line)} bytes) in {file_path.name}")
                                out_f.write(line[:10000] + b"\n")
                            else:
                                out_f.write(line)
                    if end:
                        file_path.unlink()
                    else:
                        file_path.open("wb").close()
                except Exception as err:
                    print(f"Error reading {file_path}: {err}", flush=True)
    except Exception as err:
        print(f"Error writing to {log_file}: {err}", flush=True)

    if end:
        try:
            shutil.rmtree(TMP_LOG_PATH)
        except OSError:
            pass

        log_counts = {}
        checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
        api_configs = set()
        try:
            with checkpoint_file.open("r") as f:
                api_configs = set(line.strip() for line in f if line.strip())
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
                    lines = set(line.strip() for line in f if line.strip())
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


def print_log_info(all_case, log_counts={}):
    """打印日志统计信息"""
    test_case = log_counts.get("checkpoint", 0)
    fail_case = log_counts.get("crash", 0) + log_counts.get("timeout", 0)
    skip_case = log_counts.get("skip", 0)

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Test Case Statistics".center(50))
    print("=" * 50)
    print(f"{'Total cases':<30}: {all_case}")
    print(f"{'Tested cases':<30}: {test_case}")
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
