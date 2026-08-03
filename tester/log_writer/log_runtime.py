"""日志路径、命令配置和进程级文件资源。"""

from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

from .log_schema import LOG_PREFIXES, LogType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_LOG_PATH = PROJECT_ROOT / "tester/api_config/test_log"
TMP_LOG_PATH = TEST_LOG_PATH / ".tmp"
RESULT_LOG_PATH = TEST_LOG_PATH
RESULT_LOG_SUFFIX = ""
MAIN_LOG_SUFFIX = ""
_process_file_handlers = {}
_main_output_streams = None
_main_output_file = None
_main_output_lock = None


class _TeeStream:
    """Write main-process text output to both the original stream and a log file."""

    def __init__(self, stream, log_file, lock):
        self._stream = stream
        self._log_file = log_file
        self._lock = lock

    def write(self, data):
        if not data:
            return 0
        with self._lock:
            written = self._stream.write(data)
            self._log_file.write(data)
            self._log_file.flush()
        return written

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        with self._lock:
            self._stream.flush()
            self._log_file.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def default_log_dir(*, single=False):
    """Return a project-relative timestamped directory for an unspecified output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "test_log_single" if single else "test_log"
    return Path("logs") / f"{prefix}_{timestamp}"


def configure_direct_results(log_id):
    """配置历史单进程引擎使用的主结果文件后缀。"""
    global RESULT_LOG_PATH, RESULT_LOG_SUFFIX, MAIN_LOG_SUFFIX
    TEST_LOG_PATH.mkdir(parents=True, exist_ok=True)
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


def init_main_output(log_dir):
    """Configure result paths and tee the main process stdout/stderr."""
    global _main_output_streams, _main_output_file, _main_output_lock

    close_main_output()
    _configure_log(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_path = TEST_LOG_PATH / f"log_{timestamp}_{os.getpid()}.log"
    _main_output_file = main_log_path.open("a", encoding="utf-8", buffering=1)
    _main_output_lock = threading.RLock()
    _main_output_streams = (sys.stdout, sys.stderr)
    sys.stdout = _TeeStream(sys.stdout, _main_output_file, _main_output_lock)
    sys.stderr = _TeeStream(sys.stderr, _main_output_file, _main_output_lock)
    return main_log_path


def close_main_output():
    """Restore the original stdout/stderr and close the main output file."""
    global _main_output_streams, _main_output_file, _main_output_lock

    if _main_output_streams is None:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    original_stdout, original_stderr = _main_output_streams
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    if _main_output_file is not None:
        try:
            _main_output_file.close()
        except Exception:
            pass
    _main_output_streams = None
    _main_output_file = None
    _main_output_lock = None


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
