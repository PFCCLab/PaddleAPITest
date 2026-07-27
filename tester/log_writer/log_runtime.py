"""日志路径、命令配置和进程级文件资源。"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from .log_schema import LOG_PREFIXES, LogType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_LOG_PATH = PROJECT_ROOT / "tester/api_config/test_log"
TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"
RESULT_LOG_PATH = TEST_LOG_PATH
RESULT_LOG_SUFFIX = ""
MAIN_LOG_SUFFIX = ""
_process_file_handlers = {}


def configure_direct_results(log_id):
    """配置历史单进程引擎使用的主结果文件后缀。"""
    global RESULT_LOG_PATH, RESULT_LOG_SUFFIX, MAIN_LOG_SUFFIX
    suffix = f"_{log_id}" if log_id and not log_id.startswith("_") else log_id
    RESULT_LOG_PATH = TEST_LOG_PATH
    RESULT_LOG_SUFFIX = suffix
    MAIN_LOG_SUFFIX = suffix


def _configure_log(log_dir):
    global TEST_LOG_PATH, TMP_LOG_PATH, RESULT_LOG_PATH, RESULT_LOG_SUFFIX
    global MAIN_LOG_SUFFIX
    close_process_files()
    TEST_LOG_PATH = PROJECT_ROOT / log_dir
    TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
    TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"
    TMP_LOG_PATH.mkdir(exist_ok=True)
    RESULT_LOG_PATH = TMP_LOG_PATH
    RESULT_LOG_SUFFIX = f"_{os.getpid()}"
    MAIN_LOG_SUFFIX = ""


def close_process_files():
    handlers = tuple(_process_file_handlers.values())
    _process_file_handlers.clear()
    for handler in handlers:
        try:
            handler.close()
        except Exception as err:
            print(f"Error closing process file: {err}", flush=True)


def _get_process_file(file_path):
    handler = _process_file_handlers.get(file_path)
    if handler is None:
        handler = file_path.open("a", buffering=1, newline="")
        _process_file_handlers[file_path] = handler
    return handler


def _discard_process_file(file_path):
    handler = _process_file_handlers.pop(file_path, None)
    if handler is not None:
        try:
            handler.close()
        except Exception:
            pass


def write_line(file_path, line):
    try:
        _get_process_file(file_path).write(line + "\n")
        return True
    except Exception as err:
        _discard_process_file(file_path)
        print(f"Error writing to {file_path}: {err}", flush=True)
        return False


def append_csv_row(file_path, header, row):
    """追加一行 CSV，并在首次打开空文件时写入表头。"""
    try:
        handler = _process_file_handlers.get(file_path)
        if handler is None:
            needs_header = not file_path.exists() or file_path.stat().st_size == 0
            handler = _get_process_file(file_path)
        else:
            needs_header = False
        writer = csv.writer(handler)
        if needs_header:
            writer.writerow(header)
        writer.writerow(row)
    except Exception as err:
        _discard_process_file(file_path)
        print(f"Error writing to {file_path}: {err}", flush=True)


def result_file(prefix, root=None):
    directory = RESULT_LOG_PATH if root is None else root
    return directory / f"{prefix}{RESULT_LOG_SUFFIX}.txt"


def read_log_lines(log_file):
    try:
        with log_file.open("r") as source:
            return {line.strip() for line in source if line.strip()}
    except FileNotFoundError:
        return set()


def read_log(log_type: LogType):
    return read_log_lines(TEST_LOG_PATH / f"{LOG_PREFIXES[log_type]}{MAIN_LOG_SUFFIX}.txt")


def write_lines_atomic(file_path, lines):
    temp_file = file_path.with_name(f".{file_path.name}.tmp")
    try:
        with temp_file.open("w") as target:
            target.writelines(lines)
        os.replace(temp_file, file_path)
    finally:
        temp_file.unlink(missing_ok=True)


def write_sorted_lines(file_path, lines):
    if lines:
        write_lines_atomic(file_path, (f"{line}\n" for line in sorted(lines)))
    else:
        file_path.unlink(missing_ok=True)
