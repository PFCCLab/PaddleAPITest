from __future__ import annotations

import argparse
import atexit
import gc
import importlib
import multiprocessing as mp
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import cpu_count, set_start_method
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pynvml
import yaml

if TYPE_CHECKING:
    import paddle
    import torch
    from tester import (
        APIConfig,
        APITestAccuracy,
        APITestAccuracyStable,
        APITestCINNVSDygraph,
        APITestCustomDeviceVSCPU,
        APITestPaddleDeviceVSGPU,
        APITestPaddleGPUPerformance,
        APITestPaddleOnly,
        APITestPaddleTorchGPUPerformance,
        APITestTorchGPUPerformance,
    )

from tester.dump_writer import (
    dump_enabled,
    parse_strict_bool,
    record_dump_terminal_status,
    resolve_dump_options,
)
from tester.log_writer import (
    init_log,
    log_aggregation,
    log_report,
    log_retest,
    log_runtime,
    log_worker,
)
from tester.runtime_config import (
    limit_worker_layout,
    runtime_config_for_gpu,
)
from tester.sanitizer_output import analyze_sanitizer_output

os.environ["FLAGS_use_system_allocator"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


class GpuMemoryDeferred(Exception):
    """当 GPU 模式下的 case 需要等待更多可用显存时抛出。"""


# 运行时透传给 test class 的选项白名单。
VALID_TEST_ARGS = {
    "test_amp",
    "test_backward",
    "atol",
    "rtol",
    "manual_threshold_config_file",
    "test_tol",
    "operation_mode",
    "bos_path",
    "random_seed",
    "bos_conf_path",
    "bcecmd_path",
    "generate_failed_tests",
    "bitwise_alignment",
    "exit_on_error",
    "use_gpu_mode",
}

SANITIZER_FORWARD_ARGS = {
    "accuracy",
    "paddle_only",
    "paddle_cinn",
    "paddle_gpu_performance",
    "torch_gpu_performance",
    "paddle_torch_gpu_performance",
    "accuracy_stable",
    "accuracy_stable_dual_gpu",
    "paddle_custom_device",
    "custom_device_vs_gpu",
    "custom_device_vs_gpu_mode",
    "test_amp",
    "test_cpu",
    "use_cached_numpy",
    "use_gpu_mode",
    "atol",
    "rtol",
    "manual_threshold_config_file",
    "test_tol",
    "test_backward",
    "show_runtime_status",
    "random_seed",
    "bitwise_alignment",
    "generate_failed_tests",
    "exit_on_error",
}
SANITIZER_FORWARD_ARGS_SORTED = tuple(sorted(SANITIZER_FORWARD_ARGS))

# 运行时错误标记，避免在每个 case 里重复构造。
OOM_ERROR_MARKERS = (
    "cuda out of memory",
    "out of memory error",
    "resourceexhaustederror",
    "out of memory",
    "outofmemoryerror",
    "cannot allocate memory",
    "std::bad_alloc",
    "bad allocation",
    "memoryerror",
    "cublas_status_alloc_failed",
)
CUDA_ERROR_MARKERS = (
    "cuda error",
    "memory corruption",
    "illegal memory access",
    "invalid configuration argument",
    "invalid resource handle",
)
GPU_PERFORMANCE_MODES = (
    "paddle_gpu_performance",
    "torch_gpu_performance",
    "paddle_torch_gpu_performance",
)

# 选择测试类的优先级顺序。
TEST_CLASS_BY_OPTION = (
    ("paddle_only", "APITestPaddleOnly"),
    ("paddle_cinn", "APITestCINNVSDygraph"),
    ("accuracy", "APITestAccuracy"),
    ("paddle_gpu_performance", "APITestPaddleGPUPerformance"),
    ("torch_gpu_performance", "APITestTorchGPUPerformance"),
    ("paddle_torch_gpu_performance", "APITestPaddleTorchGPUPerformance"),
    ("accuracy_stable_dual_gpu", "APITestAccuracyStable"),
    ("accuracy_stable", "APITestAccuracyStable"),
    ("paddle_custom_device", "APITestCustomDeviceVSCPU"),
    ("custom_device_vs_gpu", "APITestPaddleDeviceVSGPU"),
)

# 设备探测命令和缓存状态。
XPU_SMI_COMMAND = "xpu-smi"
XPU_SMI_DEVICE_PATTERN = r"^\|\s*(\d+)\s+\S"
ILUVATAR_SMI_COMMAND = "ixsmi"
ILUVATAR_SMI_DEVICE_PATTERN = r"^\|\s*(\d+)\s+Iluvatar"
DEVICE_TYPE = None
DEVICE_TYPE_DETECTED = False
DEVICE_COUNT = None  # 设备总数
_MEM_SNAPSHOT = None  # gpu_id -> (total_gb, used_gb)
_MEM_SNAPSHOT_TS = 0.0
_NVML_INITIALIZED = False  # 重复显存查询的 NVML 会话。
_MEM_SNAPSHOT_TTL = 2.0  # 秒。

# 调度与重试上限。
MAX_TOTAL_WORKERS = 64
MAX_EXTERNAL_KILL_RETRIES_PER_CASE = 1
MAX_TOTAL_EXTERNAL_KILL_EVENTS = 3
# 初始 warmup 与单个 slot 复活共用的启动超时预算。
WORKER_STARTUP_TIMEOUT = 180
FORECAST_MIN_INTERVAL_SECONDS = 60
FORECAST_MAX_INTERVAL_SECONDS = 30 * 60
FORECAST_TARGET_CASES = 100
FORECAST_INITIAL_MAX_WAIT_SECONDS = 5 * 60


@dataclass
class BatchRetryState:
    per_case_external_kill_retries: dict[str, int] = field(default_factory=dict)
    total_external_kills: int = 0
    unsafe_environment: bool = False


@dataclass
class BatchRunState:
    tested_case: int = 0
    batch_exit_code: int = 0
    shutdown_force: bool = False
    abort_run: bool = False
    active_tasks: int = 0
    test_started_at: float | None = None
    last_forecast_at: float | None = None
    last_forecast_case: int = 0


@dataclass
class CaseRuntimeContext:
    started_at: float
    gpu_id: int
    comparison_gpu_id: int | None
    suppress_case_tags: bool
    runtime_config: object | None = None


@dataclass
class BatchConfigLoadResult:
    api_configs: list[str]
    read_count: int
    skipped_non_config: int
    duplicate_case: int
    finish_case: int
    removed_stale_logs: int

    @property
    def all_case(self):
        return len(self.api_configs)


@dataclass
class BatchMessage:
    msg_type: str
    slot_index: int | None = None
    config: str | None = None
    exitcode: int | None = None
    worker_pid: int | None = None
    completed_offset: int | None = None
    reason: str | None = None
    crash_source: str = "worker"

    @classmethod
    def from_raw(cls, msg):
        """把 batch 原始消息整理成结构化对象。"""
        message = cls(
            msg_type=msg[0],
            slot_index=msg[1] if len(msg) > 1 else None,
            config=msg[2] if len(msg) > 2 else None,
        )
        if message.msg_type == "done":
            message.worker_pid = msg[3] if len(msg) > 3 else None
            message.completed_offset = msg[4] if len(msg) > 4 else None
        elif message.msg_type == "error":
            message.reason = msg[3] if len(msg) > 3 else ""
            message.worker_pid = msg[4] if len(msg) > 4 else None
            message.completed_offset = msg[5] if len(msg) > 5 else None
        elif message.msg_type == "deferred":
            message.reason = msg[3] if len(msg) > 3 else "insufficient GPU memory"
            message.worker_pid = msg[4] if len(msg) > 4 else None
            message.completed_offset = msg[5] if len(msg) > 5 else None
        elif message.msg_type == "crashed":
            message.exitcode = msg[3] if len(msg) > 3 else None
            if len(msg) > 5 and msg[5] == "child":
                message.crash_source = "child"
                message.worker_pid = msg[6] if len(msg) > 6 else None
                message.completed_offset = msg[7] if len(msg) > 7 else None
            else:
                message.worker_pid = msg[4] if len(msg) > 4 else None
        return message


# ─── WorkerPool：每个 worker 独立队列的架构 ───────────────────────────────


@dataclass
class WorkerSlot:
    """表示一个拥有独立输入队列的 worker 进程槽位。"""

    index: int
    gpu_id: int
    comparison_gpu_id: int | None = None
    process: mp.Process | None = None
    input_queue: mp.Queue | None = None
    current_task: str | None = None
    task_start_time: float | None = None
    child_pid: int | None = None
    started_at: float | None = None
    state: str = "dead"  # dead、starting、idle、busy


def _import_optional_runtime_module(module_name):
    try:
        importlib.import_module(module_name)
    except Exception:
        pass


def _init_runtime_modules(options):
    with log_worker.suppress_startup_output():
        import paddle

        globals()["paddle"] = paddle
        if options.test_cpu:
            paddle.device.set_device("cpu")
        elif not getattr(options, "paddle_custom_device", False):
            # CUDA_VISIBLE_DEVICES 只负责限定 slot，Paddle 仍需要显式设置 device。
            paddle.set_device("gpu")
        _import_optional_runtime_module("paddlefleet_ops")
        _import_optional_runtime_module("FusedQuantOps")
        import tester

        test_class = _select_test_class(options)
        globals().update({"APIConfig": tester.APIConfig, test_class.__name__: test_class})


def _visible_gpu_ids(gpu_id, comparison_gpu_id=None):
    if gpu_id is None:
        return None
    if comparison_gpu_id is None:
        return str(gpu_id)
    return f"{gpu_id},{comparison_gpu_id}"


def _init_worker_runtime(
    slot_index,
    gpu_id,
    comparison_gpu_id,
    options,
    *,
    redirect_output,
):
    init_log(options.log_dir)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _visible_gpu_ids(gpu_id, comparison_gpu_id)
        workers_on_gpu = (getattr(options, "gpu_workers_per_gpu_map", {}) or {}).get(gpu_id, 1)
        os.environ["PADDLEAPITEST_WORKERS_ON_GPU"] = str(workers_on_gpu)

    _init_runtime_modules(options)

    if redirect_output:
        log_worker.redirect_stdio()

    if slot_index is not None and gpu_id is not None:
        os.environ["PADDLEAPITEST_WORKER_SLOT"] = str(slot_index)


def _worker_loop(
    slot_index,
    gpu_id,
    comparison_gpu_id,
    input_queue,
    result_queue,
    options,
):
    """常驻 worker 进程，从 input_queue 取任务并把结果写入 result_queue。

    Exit behavior:
        - Normal exit: receives None, releases device resources, and returns gracefully.
        - Fatal CUDA/OOM/Torch errors: run_test_case exits with the centralized fatal protocol.
          The code identifies the result type and whether the worker already wrote it. This
          bypasses Python cleanup; the watchdog detects and respawns the dead worker.
        - Other crashes: any unhandled signal (SIGSEGV etc.) or SIGKILL from Watchdog timeout
          terminates the process. Watchdog detects exitcode != 0 and respawns.

    The main process never dispatches to a dead/restarting worker — upon detecting crash or
    timeout, the next task goes to `pending_dispatch` and is sent after the new worker reports
    "ready".
    """
    # ── GPU 初始化（等价于 init_worker_gpu） ──
    try:
        _init_worker_runtime(
            slot_index,
            gpu_id,
            comparison_gpu_id,
            options,
            redirect_output=True,
        )
    except Exception as e:
        result_queue.put(("init_failed", slot_index, str(e)))
        return

    # ── 通知主进程：ready ──
    result_queue.put(("ready", slot_index))

    # ── 任务循环 ──
    while True:
        try:
            task = input_queue.get()
        except (EOFError, OSError):
            break
        if task is None:  # 毒丸
            break

        api_config_str = task
        result_queue.put(("ack", slot_index, api_config_str))

        try:
            run_test_case(api_config_str, options)
            result_queue.put(
                (
                    "done",
                    slot_index,
                    api_config_str,
                    os.getpid(),
                    log_worker.get_worker_log_offset(),
                )
            )
        except GpuMemoryDeferred as e:
            result_queue.put(
                (
                    "deferred",
                    slot_index,
                    api_config_str,
                    str(e),
                    os.getpid(),
                    log_worker.get_worker_log_offset(),
                )
            )
        except SystemExit:
            # run_test_case 遇到 CUDA 错误时会走 os._exit，这里理论上不应到达；
            # 如果是通过 sys.exit 进入这里，则继续向上抛出。
            raise
        except Exception as e:
            result_queue.put(
                (
                    "error",
                    slot_index,
                    api_config_str,
                    str(e),
                    os.getpid(),
                    log_worker.get_worker_log_offset(),
                )
            )

    # 优雅退出。GPU 模式会跳过逐 case 的收集，因此要在框架 atexit
    # 释放设备管理器之前先清理循环张量图。
    try:
        gc.collect()
        log_runtime.close_process_files()
        log_worker.restore_stdio()
    except Exception:
        pass


def _build_sanitizer_case_command(api_config_str, options, sanitizer_cmd):
    cmd = [
        *sanitizer_cmd,
        sys.executable,
        str(Path(__file__).resolve()),
        f"--api_config={api_config_str}",
        f"--log_dir={options.log_dir}",
        "--_sanitizer_child=True",
    ]
    for key in SANITIZER_FORWARD_ARGS_SORTED:
        value = getattr(options, key, None)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, bool) and not value:
            continue
        formatted = "True" if value is True else "False" if value is False else str(value)
        cmd.append(f"--{key}={formatted}")
    return cmd


def _sanitizer_worker_loop(
    slot_index,
    gpu_id,
    comparison_gpu_id,
    input_queue,
    result_queue,
    options,
):
    init_log(options.log_dir)
    log_worker.redirect_stdio()

    child_process = None
    sanitizer_cmd = getattr(options, "sanitizer_cmd", None) or shlex.split(
        options.sanitizer_command
    )
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = _visible_gpu_ids(gpu_id, comparison_gpu_id)
    child_env["PADDLEAPITEST_SUPPRESS_CASE_TAGS"] = "1"

    def terminate_child(*args):
        if child_process is not None and child_process.poll() is None:
            try:
                os.killpg(child_process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                child_process.kill()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, terminate_child)
    signal.signal(signal.SIGTERM, terminate_child)

    try:
        result_queue.put(("ready", slot_index))

        while True:
            try:
                task = input_queue.get()
            except (EOFError, OSError):
                break
            if task is None:
                break

            api_config_str = task
            result_queue.put(("ack", slot_index, api_config_str))
            log_worker.write_case_begin(
                api_config_str,
                worker_pid=os.getpid(),
                slot=slot_index,
                gpu=gpu_id,
            )
            case_log_dir = (
                log_runtime.TMP_LOG_PATH / "sanitizer" / f"slot_{slot_index}_{os.getpid()}"
            )
            if case_log_dir.exists():
                shutil.rmtree(case_log_dir)
            case_log_dir.mkdir(parents=True, exist_ok=True)
            try:
                cmd = _build_sanitizer_case_command(api_config_str, options, sanitizer_cmd)
            except ValueError as err:
                shutil.rmtree(case_log_dir, ignore_errors=True)
                completed_offset = log_worker.write_case_end("error", api_config_str)
                result_queue.put(
                    ("error", slot_index, api_config_str, str(err), os.getpid(), completed_offset)
                )
                continue

            try:
                child_process = subprocess.Popen(
                    cmd,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as err:
                shutil.rmtree(case_log_dir, ignore_errors=True)
                completed_offset = log_worker.write_case_end("error", api_config_str)
                result_queue.put(
                    ("error", slot_index, api_config_str, str(err), os.getpid(), completed_offset)
                )
                continue
            result_queue.put(("child", slot_index, child_process.pid))
            output_tail = deque(maxlen=40)
            with tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            ) as output_file:
                try:
                    for line in child_process.stdout:
                        output_tail.append(line)
                        output_file.write(line)
                    returncode = child_process.wait()
                finally:
                    if child_process.stdout is not None:
                        child_process.stdout.close()

                child_process = None
                output_file.seek(0)
                if returncode == options.sanitizer_error_exitcode:
                    analysis = analyze_sanitizer_output(
                        output_file.read(), returncode, options.sanitizer_error_exitcode
                    )
                    if analysis.output:
                        print(
                            analysis.output,
                            end="" if analysis.output.endswith("\n") else "\n",
                            flush=True,
                        )
                else:
                    analysis = None
                    shutil.copyfileobj(output_file, sys.stdout)
                    sys.stdout.flush()

                ignored = analysis is not None and analysis.only_ignored_diagnostics
                if returncode in (0, 2) or ignored:
                    log_worker.merge_sanitizer_case_logs(case_log_dir)
                shutil.rmtree(case_log_dir, ignore_errors=True)

                if returncode == 0 or ignored:
                    completed_offset = log_worker.write_case_end("completed", api_config_str)
                    result_queue.put(
                        ("done", slot_index, api_config_str, os.getpid(), completed_offset)
                    )
                elif returncode == 2:
                    completed_offset = log_worker.write_case_end("error", api_config_str)
                    result_queue.put(
                        (
                            "error",
                            slot_index,
                            api_config_str,
                            f"child exited with {returncode}",
                            os.getpid(),
                            completed_offset,
                        )
                    )
                else:
                    completed_offset = log_worker.write_case_end("crashed", api_config_str)
                    result_queue.put(
                        (
                            "crashed",
                            slot_index,
                            api_config_str,
                            returncode,
                            "".join(output_tail),
                            "child",
                            os.getpid(),
                            completed_offset,
                        )
                    )
    finally:
        if child_process is not None and child_process.poll() is None:
            try:
                os.killpg(child_process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                child_process.kill()
            try:
                child_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            log_runtime.close_process_files()
            log_worker.restore_stdio()
        except Exception:
            pass


class WorkerPool:
    """用于公平 GPU 调度的自定义进程池，每个 worker 对应一个队列。"""

    def __init__(self, available_gpus, max_workers_per_gpu, options, *, gpu_total_memory_map=None):
        # 将 argparse.Namespace 转成 SimpleNamespace，便于 worker 进程更干净地 pickle。
        if isinstance(options, argparse.Namespace):
            self.options = SimpleNamespace(**vars(options))
        else:
            self.options = options
        self.options.gpu_workers_per_gpu_map = dict(max_workers_per_gpu)
        if gpu_total_memory_map is None:
            # 允许外部预先收集，避免主流程和进程池重复探测同一批 GPU。
            gpu_total_memory_map = _build_gpu_total_memory_map(available_gpus)
        self.options.gpu_total_memory_map = dict(gpu_total_memory_map)
        self.result_queue = mp.Queue()
        self.slots: list[WorkerSlot] = []
        self._shutdown_event = threading.Event()
        self._watchdog_thread = None
        self._lock = threading.Lock()  # 保护 slot 状态修改
        self._spawn_lock = threading.Lock()
        self._closed = False

        # 构建 worker 槽位：按 GPU 或 GPU 对确定性分配。
        idx = 0
        if getattr(self.options, "accuracy_stable_dual_gpu", False):
            for pair_index in range(0, len(available_gpus), 2):
                slot = WorkerSlot(
                    index=idx,
                    gpu_id=available_gpus[pair_index],
                    comparison_gpu_id=available_gpus[pair_index + 1],
                )
                self.slots.append(slot)
                idx += 1
        else:
            for gpu_id in available_gpus:
                for _ in range(max_workers_per_gpu[gpu_id]):
                    slot = WorkerSlot(index=idx, gpu_id=gpu_id)
                    self.slots.append(slot)
                    idx += 1

    @property
    def total_workers(self):
        return len(self.slots)

    def start(self):
        """并行启动所有 worker 进程，然后启动 watchdog 线程。"""
        for slot in self.slots:
            self._spawn_worker(slot)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="pool-watchdog"
        )
        self._watchdog_thread.start()

    def _close_queue(self, q, *, cancel_join=False):
        """关闭 multiprocessing 队列，避免清理错误掩盖测试结果。"""
        if q is None:
            return
        try:
            if cancel_join:
                q.cancel_join_thread()
        except Exception:
            pass
        try:
            q.close()
        except Exception:
            pass
        if not cancel_join:
            try:
                q.join_thread()
            except Exception:
                pass

    def _spawn_worker(self, slot):
        """为指定 slot 拉起一个新的 worker 进程。"""
        with self._spawn_lock:
            if self._closed or self._shutdown_event.is_set():
                return False
            if slot.process is not None and slot.process.is_alive():
                return False
            if slot.process is not None:
                self._join_process(slot.process, timeout=1)
            self._close_queue(slot.input_queue, cancel_join=True)
            slot.input_queue = mp.Queue()
            worker_target = (
                _sanitizer_worker_loop
                if getattr(self.options, "use_compute_sanitizer", False)
                else _worker_loop
            )
            p = mp.Process(
                target=worker_target,
                args=(
                    slot.index,
                    slot.gpu_id,
                    slot.comparison_gpu_id,
                    slot.input_queue,
                    self.result_queue,
                    self.options,
                ),
                daemon=True,
            )
            p.start()
            slot.process = p
            slot.state = "starting"
            slot.current_task = None
            slot.task_start_time = None
            slot.child_pid = None
            slot.started_at = time.monotonic()
            return True

    def _startup_timeout(self):
        return getattr(self.options, "worker_startup_timeout", WORKER_STARTUP_TIMEOUT)

    def _check_starting_worker(self, slot, *, now=None):
        """重启一个在启动阶段失败或卡住的 worker。"""
        if slot.state != "starting" or slot.process is None or slot.started_at is None:
            return False
        now = time.monotonic() if now is None else now
        if now - slot.started_at < self._startup_timeout():
            return False
        if slot.process.is_alive():
            print(
                f"[worker] INIT_TIMEOUT | slot {slot.index} | timeout {self._startup_timeout()} s",
                flush=True,
            )
            self._kill_process(slot.process)
        else:
            print(
                f"[worker] INIT_CRASH | slot {slot.index} | exit {slot.process.exitcode}",
                flush=True,
            )
            self._join_process(slot.process, timeout=1)
        self._spawn_worker(slot)
        return True

    def warmup(self, timeout=None):
        """等待所有 worker 就绪，或直到启动超时。"""
        if timeout is None:
            timeout = self._startup_timeout()
        ready_slots = {slot.index for slot in self.slots if slot.state == "idle"}
        deadline = time.monotonic() + timeout

        while len(ready_slots) < self.total_workers and time.monotonic() < deadline:
            try:
                remaining = max(0.1, deadline - time.monotonic())
                msg = self.result_queue.get(timeout=min(5.0, remaining))
                if self._handle_worker_control_message(msg):
                    if msg[0] == "ready" and self.slots[msg[1]].state == "idle":
                        ready_slots.add(msg[1])
                    continue
            except queue.Empty:
                now = time.monotonic()
                for slot in self.slots:
                    self._check_starting_worker(slot, now=now)

        ready_count = sum(slot.state == "idle" for slot in self.slots)
        if ready_count != self.total_workers:
            print(
                f"[workers] READY_TIMEOUT | {ready_count}/{self.total_workers} ready | "
                f"timeout {timeout} s",
                flush=True,
            )
        return ready_count

    def _handle_worker_control_message(self, msg):
        """处理来自 worker 的 ready、init_failed 和 child 账务消息。"""
        msg_type = msg[0]
        if msg_type == "ready":
            slot_idx = msg[1]
            with self._lock:
                slot = self.slots[slot_idx]
                slot.state = "idle"
                slot.started_at = None
            return True
        if msg_type == "init_failed":
            slot_idx = msg[1]
            error_msg = msg[2]
            print(
                f"[worker] INIT_FAILED | slot {slot_idx} | {error_msg}",
                flush=True,
            )
            slot = self.slots[slot_idx]
            self._join_process(slot.process, timeout=1)
            self._spawn_worker(slot)
            return True
        if msg_type == "child":
            slot_idx = msg[1]
            child_pid = msg[2]
            with self._lock:
                self.slots[slot_idx].child_pid = child_pid
            return True
        return False

    def dispatch(self, slot_index, config):
        """向指定 worker slot 派发任务。"""
        slot = self.slots[slot_index]
        with self._lock:
            slot.current_task = config
            slot.task_start_time = None  # 在 ack 到来时再记录
            slot.state = "busy"
        slot.input_queue.put(config)

    def collect_one(self, timeout=5.0):
        """从 result_queue 取一条消息，超时则返回 None。"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def idle_slots(self):
        """遍历当前处于空闲状态的 slot。"""
        for slot in self.slots:
            if slot.state == "idle":
                yield slot

    def mark_idle(self, slot_index):
        """在任务完成后将 worker slot 标记为空闲。"""
        with self._lock:
            slot = self.slots[slot_index]
            slot.state = "idle"
            slot.current_task = None
            slot.task_start_time = None
            slot.child_pid = None
            slot.started_at = None

    def _watchdog_loop(self):
        """周期性检查超时和非预期死亡的 worker。"""
        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(1.0):
                break
            now = time.time()

            for slot in self.slots:
                if self._shutdown_event.is_set():
                    break
                self._watchdog_tick_slot(slot, now=now)

    def _watchdog_tick_slot(self, slot, *, now):
        """在持有 slot 锁时评估单个 slot 并执行恢复。"""
        with self._lock:
            if self._shutdown_event.is_set():
                return
            # 在 slot 快照仍然一致时直接执行恢复动作。
            if slot.state == "starting":
                self._check_starting_worker(slot, now=time.monotonic())
                return
            if (
                slot.state == "busy"
                and slot.task_start_time is not None
                and now - slot.task_start_time > self.options.timeout
            ):
                self._handle_timeout(slot)
                return
            if (
                slot.state in ("busy", "idle")
                and slot.process is not None
                and not slot.process.is_alive()
            ):
                self._handle_crash(slot)

    def _handle_timeout(self, slot):
        """终止超时 worker，并写入 timeout 结果。"""
        if self._closed or self._shutdown_event.is_set():
            return
        config = slot.current_task
        old_pid = slot.process.pid if slot.process else None
        self._kill_slot_child(slot)
        self._kill_process(slot.process)
        if old_pid is not None and config is not None:
            completed_offset = log_worker.append_case_end_to_worker_log(
                old_pid, "timeout", api_config_str=config
            )
            log_aggregation.mark_inorder_case_complete(old_pid, completed_offset)
        if self._closed or self._shutdown_event.is_set():
            return
        self.result_queue.put(("timeout", slot.index, config))
        self._spawn_worker(slot)

    def _handle_crash(self, slot):
        """处理非预期死亡的 worker。"""
        if self._closed or self._shutdown_event.is_set():
            return
        exitcode = slot.process.exitcode if slot.process else None
        config = slot.current_task
        if self._closed or self._shutdown_event.is_set():
            return
        if config is not None:
            completed_offset = log_worker.append_case_end_to_worker_log(
                slot.process.pid, "crashed", api_config_str=config
            )
            log_aggregation.mark_inorder_case_complete(slot.process.pid, completed_offset)
            self.result_queue.put(("crashed", slot.index, config, exitcode))
        else:
            print(
                f"[worker] PADDLE_CRASH | slot {slot.index} | exit {exitcode}",
                flush=True,
            )
        self._spawn_worker(slot)

    def _kill_process_group(self, pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def _kill_slot_child(self, slot):
        if slot.child_pid is not None:
            self._kill_process_group(slot.child_pid)
            slot.child_pid = None

    def _sigkill_process(self, process):
        """向进程发送 SIGKILL，但不等待其退出。"""
        try:
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    def _join_process(self, process, timeout=5):
        """等待进程退出，并忽略清理阶段的失败。"""
        try:
            process.join(timeout=timeout)
        except Exception:
            pass

    def _kill_process(self, process):
        """SIGKILL 一个进程（CUDA 死锁进程通常不会响应 SIGTERM）。"""
        self._sigkill_process(process)
        self._join_process(process, timeout=5)

    def shutdown(self, force=False):
        """停止所有 worker 并释放 multiprocessing 队列。"""
        if self._closed:
            return
        self._closed = True
        self._shutdown_event.set()

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=3)

        try:
            if not force:
                # 优雅退出：发送毒丸
                for slot in self.slots:
                    if slot.input_queue is not None:
                        try:
                            slot.input_queue.put(None)
                        except (OSError, EOFError, ValueError):
                            pass
                for slot in self.slots:
                    if slot.process is not None:
                        slot.process.join(timeout=10)
                        if slot.process.is_alive():
                            self._kill_process(slot.process)
            else:
                # 强制退出：先对所有 worker 发 SIGKILL，再统一 join。
                # 这样在存在大量 CUDA 死锁 worker 时不会串行等待太久。
                for slot in self.slots:
                    self._kill_slot_child(slot)
                    if slot.process is not None:
                        self._sigkill_process(slot.process)
                for slot in self.slots:
                    if slot.process is not None:
                        self._join_process(slot.process, timeout=3)

        finally:
            for slot in self.slots:
                self._close_queue(slot.input_queue, cancel_join=force)
                slot.input_queue = None
            self._close_queue(self.result_queue, cancel_join=force)


def _smi_output(command):
    return subprocess.check_output([command], text=True, stderr=subprocess.STDOUT)


def _command_has_device(command, device_pattern):
    if not shutil.which(command):
        return False
    try:
        out = _smi_output(command)
    except Exception:
        return False
    return any(re.match(device_pattern, line) for line in out.splitlines())


def _count_smi_devices(command, device_pattern, *, stop_at_processes=False):
    ids = set()
    for line in _smi_output(command).splitlines():
        if stop_at_processes and "Processes:" in line:
            break
        m = re.match(device_pattern, line)
        if m:
            ids.add(int(m.group(1)))
    return len(ids)


def _read_smi_memory_snapshot(command, device_pattern):
    snapshot = {}
    lines = _smi_output(command).splitlines()
    for i, line in enumerate(lines):
        m = re.match(device_pattern, line)
        if not m:
            continue
        dev_id = int(m.group(1))
        for mem_line in lines[i + 1 : min(i + 8, len(lines))]:
            mm = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", mem_line)
            if mm:
                used_mib = int(mm.group(1))
                total_mib = int(mm.group(2))
                snapshot[dev_id] = (total_mib / 1024.0, used_mib / 1024.0)
                break
    return snapshot


def detect_device_type() -> str:
    global DEVICE_TYPE, DEVICE_TYPE_DETECTED, _NVML_INITIALIZED
    if DEVICE_TYPE_DETECTED:
        return DEVICE_TYPE

    # 探测顺序决定运行后端优先级：NVIDIA GPU > XPU > Iluvatar > CPU。
    try:
        if not _NVML_INITIALIZED:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        if pynvml.nvmlDeviceGetCount() > 0:
            DEVICE_TYPE = "gpu"
            DEVICE_TYPE_DETECTED = True
            return DEVICE_TYPE
    except Exception:
        # 没有 NVML 或不是 NVIDIA 环境时继续探测其他后端。
        pass

    for device_type, command, device_pattern in (
        ("xpu", XPU_SMI_COMMAND, XPU_SMI_DEVICE_PATTERN),
        ("iluvatar_gpu", ILUVATAR_SMI_COMMAND, ILUVATAR_SMI_DEVICE_PATTERN),
    ):
        if _command_has_device(command, device_pattern):
            DEVICE_TYPE = device_type
            DEVICE_TYPE_DETECTED = True
            return DEVICE_TYPE

    DEVICE_TYPE = "cpu"
    DEVICE_TYPE_DETECTED = True
    return DEVICE_TYPE


def get_device_count() -> int:
    """获取可用设备（加速器）数量。"""
    global DEVICE_COUNT, _NVML_INITIALIZED
    if DEVICE_COUNT is not None:
        return DEVICE_COUNT

    device_type = detect_device_type()

    if device_type == "gpu":
        if not _NVML_INITIALIZED:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        count = pynvml.nvmlDeviceGetCount()
        DEVICE_COUNT = count
        return count

    if device_type == "xpu":
        DEVICE_COUNT = _count_smi_devices(
            XPU_SMI_COMMAND,
            XPU_SMI_DEVICE_PATTERN,
            stop_at_processes=True,
        )
        return DEVICE_COUNT

    if device_type == "iluvatar_gpu":
        DEVICE_COUNT = _count_smi_devices(ILUVATAR_SMI_COMMAND, ILUVATAR_SMI_DEVICE_PATTERN)
        return DEVICE_COUNT

    # CPU 场景 / 无加速器场景
    DEVICE_COUNT = 0
    return 0


def _refresh_snapshot(device_type):
    global _MEM_SNAPSHOT, _MEM_SNAPSHOT_TS

    now = time.time()
    if now - _MEM_SNAPSHOT_TS < _MEM_SNAPSHOT_TTL and _MEM_SNAPSHOT is not None:
        return

    if device_type == "xpu":
        snapshot = _read_smi_memory_snapshot(XPU_SMI_COMMAND, XPU_SMI_DEVICE_PATTERN)
    elif device_type == "iluvatar_gpu":
        snapshot = _read_smi_memory_snapshot(
            ILUVATAR_SMI_COMMAND,
            ILUVATAR_SMI_DEVICE_PATTERN,
        )
    else:
        # NVIDIA GPU 场景不使用快照，直接调用 NVML。
        _MEM_SNAPSHOT = None
        _MEM_SNAPSHOT_TS = now
        return

    _MEM_SNAPSHOT = snapshot
    _MEM_SNAPSHOT_TS = now


def get_memory_info(gpu_id):
    """返回加速器设备的 (total_memory, used_memory)，单位 GB。"""
    global _NVML_INITIALIZED
    device_type = detect_device_type()

    if device_type == "gpu":
        if not _NVML_INITIALIZED:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem_info.total) / (1024**3), int(mem_info.used) / (1024**3)

    if device_type in ("xpu", "iluvatar_gpu"):
        _refresh_snapshot(device_type)
        if _MEM_SNAPSHOT is None or gpu_id not in _MEM_SNAPSHOT:
            raise RuntimeError(f"Failed to get memory info for {device_type} device {gpu_id}")
        return _MEM_SNAPSHOT[gpu_id]

    raise RuntimeError("No supported accelerator (GPU / XPU / Iluvatar) detected.")


def _build_gpu_total_memory_map(available_gpus):
    gpu_total_memory_map = {}
    for gpu_id in available_gpus:
        try:
            gpu_total_memory_map[gpu_id] = get_memory_info(gpu_id)[0]
        except Exception:
            pass
    return gpu_total_memory_map


ARGUMENT_ERROR_PREFIX = "[argument error]"
ARGUMENT_WARNING_PREFIX = "[argument warning]"
TEST_MODE_ERROR = (
    "specify exactly one test mode: --accuracy, --paddle_only, --paddle_cinn, "
    "--paddle_gpu_performance, --torch_gpu_performance, "
    "--paddle_torch_gpu_performance, --accuracy_stable, "
    "--accuracy_stable_dual_gpu, --paddle_custom_device, --custom_device_vs_gpu"
)


def _argument_error(message):
    print(f"{ARGUMENT_ERROR_PREFIX} {message}", flush=True)
    return 2


def _mode_uses_torch(options):
    return any(
        getattr(options, opt, False)
        for opt in (
            "accuracy",
            "paddle_cinn",
            "paddle_gpu_performance",
            "torch_gpu_performance",
            "paddle_torch_gpu_performance",
            "accuracy_stable",
            "accuracy_stable_dual_gpu",
            "paddle_custom_device",
            "custom_device_vs_gpu",
        )
    )


def _select_test_class(options):
    import tester

    class_name = next(
        (
            class_name
            for option, class_name in TEST_CLASS_BY_OPTION
            if getattr(options, option, False)
        ),
        "APITestAccuracy",
    )
    return getattr(tester, class_name)


def _clear_device_cache(options):
    import paddle

    if _mode_uses_torch(options):
        import torch

        torch.cuda.empty_cache()
    paddle.device.cuda.empty_cache()


def _parse_gpu_ids(gpu_ids_arg, device_count):
    gpu_ids = []
    for raw_part in gpu_ids_arg.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part == "-1":
            gpu_ids.append(-1)
            continue
        if "-" in part:
            try:
                start, end = map(int, part.split("-", 1))
            except ValueError:
                raise ValueError(
                    f"invalid --gpu_ids='{gpu_ids_arg}': expected integers or ranges like '0,2,4-7'"
                ) from None
            if start > end:
                raise ValueError(f"invalid --gpu_ids='{gpu_ids_arg}': range start must be <= end")
            gpu_ids.extend(range(start, end + 1))
            continue
        try:
            gpu_ids.append(int(part))
        except ValueError:
            raise ValueError(
                f"invalid --gpu_ids='{gpu_ids_arg}': expected integers or ranges like '0,2,4-7'"
            ) from None

    if not gpu_ids:
        raise ValueError(f"invalid --gpu_ids='{gpu_ids_arg}': expected at least one GPU id")
    seen_gpu_ids = set()
    for gpu_id in gpu_ids:
        if gpu_id in seen_gpu_ids:
            raise ValueError(f"invalid --gpu_ids='{gpu_ids_arg}': duplicate GPU id {gpu_id}")
        seen_gpu_ids.add(gpu_id)
    if len(gpu_ids) > 1 and -1 in gpu_ids:
        raise ValueError(
            f"invalid --gpu_ids='{gpu_ids_arg}': -1 cannot be combined with explicit GPU IDs"
        )
    if gpu_ids != [-1] and not all(0 <= gpu_id < device_count for gpu_id in gpu_ids):
        raise ValueError(
            f"invalid --gpu_ids='{gpu_ids_arg}': valid GPU id range is [0, {device_count})"
        )
    return tuple(sorted(gpu_ids))


def normalize_accuracy_stable_dual_gpu_options(options):
    """让双 GPU 标志自洽地进入 accuracy-stable GPU 模式。"""
    if not getattr(options, "accuracy_stable_dual_gpu", False):
        return
    if not getattr(options, "use_gpu_mode", False):
        print(
            f"{ARGUMENT_WARNING_PREFIX} "
            "--accuracy_stable_dual_gpu=True implies --use_gpu_mode=True; enabling GPU mode",
            flush=True,
        )
        options.use_gpu_mode = True
    options.accuracy_stable = True


def validate_gpu_options(options) -> tuple:
    """校验并规范化 GPU 相关参数。"""
    normalize_accuracy_stable_dual_gpu_options(options)
    device_count = get_device_count()
    if device_count == 0:
        raise ValueError("no accelerator devices were found")

    gpu_ids = _parse_gpu_ids(options.gpu_ids, device_count) if options.gpu_ids else (-1,)
    if options.num_gpus < -1 or options.num_gpus == 0 or options.num_gpus > device_count:
        raise ValueError(
            f"invalid --num_gpus={options.num_gpus}: expected -1 or a value in [1, {device_count}]"
        )
    if options.num_gpus == -1:
        options.num_gpus = device_count if gpu_ids == (-1,) else len(gpu_ids)
    if gpu_ids == (-1,):
        gpu_ids = tuple(range(options.num_gpus))
    elif len(gpu_ids) != options.num_gpus:
        raise ValueError(
            f"invalid --num_gpus={options.num_gpus}: expected {len(gpu_ids)} "
            f"to match --gpu_ids={gpu_ids}"
        )
    if options.num_workers_per_gpu < -1 or options.num_workers_per_gpu == 0:
        raise ValueError(
            f"invalid --num_workers_per_gpu={options.num_workers_per_gpu}: "
            "expected -1 or a positive integer"
        )
    if getattr(options, "accuracy_stable_dual_gpu", False):
        if getattr(options, "test_cpu", False):
            raise ValueError("--accuracy_stable_dual_gpu=True does not support --test_cpu=True")
        if options.num_gpus < 2 or options.num_gpus % 2:
            raise ValueError("--accuracy_stable_dual_gpu=True requires an even --num_gpus")
        if options.num_workers_per_gpu != 1:
            raise ValueError("--accuracy_stable_dual_gpu=True requires --num_workers_per_gpu=1")
    return tuple(gpu_ids)


def _resolve_dump_options(parser, options):
    try:
        options.use_dump, options.dump_dir = resolve_dump_options(
            options.use_dump, options.dump_dir
        )
    except ValueError as err:
        parser.error(str(err))
    os.environ["USE_DUMP"] = str(options.use_dump)
    os.environ["DUMP_DIR"] = options.dump_dir


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in ("true", "1", "yes", "y"):
            return True
        if normalized in ("false", "0", "no", "n"):
            return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _apply_single_config_gpu_defaults(options):
    if not options.gpu_ids and options.num_gpus == -1:
        if getattr(options, "accuracy_stable_dual_gpu", False):
            options.gpu_ids = "0,1"
            options.num_gpus = 2
        else:
            options.gpu_ids = "0"
            options.num_gpus = 1


def _prepare_single_config_gpu(options):
    normalize_accuracy_stable_dual_gpu_options(options)
    if getattr(options, "accuracy_stable_dual_gpu", False) and getattr(options, "test_cpu", False):
        raise ValueError("--accuracy_stable_dual_gpu=True does not support --test_cpu=True")
    if options.test_cpu:
        options.gpu_workers_per_gpu_map = {}
        options.gpu_total_memory_map = {}
        return None

    _apply_single_config_gpu_defaults(options)
    gpu_ids = validate_gpu_options(options)
    expected_gpu_count = 2 if getattr(options, "accuracy_stable_dual_gpu", False) else 1
    if len(gpu_ids) != expected_gpu_count:
        raise ValueError(
            f"single --api_config run requires exactly {expected_gpu_count} GPU(s); "
            f"got {len(gpu_ids)} GPUs: {gpu_ids}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    options.gpu_total_memory_map = _build_gpu_total_memory_map(gpu_ids)
    options.gpu_workers_per_gpu_map = dict.fromkeys(gpu_ids, 1)
    return gpu_ids


def _validate_sanitizer_command(command):
    try:
        sanitizer_cmd = shlex.split(command)
    except ValueError as err:
        print(
            f"{ARGUMENT_ERROR_PREFIX} invalid --sanitizer_command: {err}",
            flush=True,
        )
        return None
    if not sanitizer_cmd:
        print(
            f"{ARGUMENT_ERROR_PREFIX} invalid --sanitizer_command: command cannot be empty",
            flush=True,
        )
        return None
    if shutil.which(sanitizer_cmd[0]) is None:
        print(
            f"{ARGUMENT_ERROR_PREFIX} sanitizer executable not found: {sanitizer_cmd[0]}",
            flush=True,
        )
        return None
    return sanitizer_cmd


def _run_single_config_with_sanitizer(options):
    sanitizer_cmd = _validate_sanitizer_command(options.sanitizer_command)
    if sanitizer_cmd is None:
        return 2

    try:
        gpu_ids = _prepare_single_config_gpu(options)
    except ValueError as err:
        return _argument_error(str(err))

    api_config = options.api_config.strip()
    cmd = _build_sanitizer_case_command(
        api_config,
        options,
        sanitizer_cmd,
    )
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    result = subprocess.run(
        cmd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    raw_output = f"{result.stdout or ''}{result.stderr or ''}"
    analysis = analyze_sanitizer_output(
        raw_output, result.returncode, options.sanitizer_error_exitcode
    )
    if analysis.output:
        print(
            analysis.output,
            end="" if analysis.output.endswith("\n") else "\n",
            flush=True,
        )
    if analysis.only_ignored_diagnostics:
        return 0
    if result.returncode == options.sanitizer_error_exitcode:
        print(
            f"[error] compute-sanitizer reported errors for {api_config} "
            f"(exit {result.returncode})",
            flush=True,
        )
    return result.returncode


def check_gpu_memory(gpu_ids, num_workers_per_gpu):
    assert isinstance(gpu_ids, tuple) and len(gpu_ids) > 0
    available_gpus = []
    max_workers_per_gpu = {}

    for gpu_id in gpu_ids:
        try:
            get_memory_info(gpu_id)
        except Exception as e:
            print(
                f"[warn] Failed to check accelerator {gpu_id}: {type(e).__name__}: {e!s}",
                flush=True,
            )
            continue
        available_gpus.append(gpu_id)
        max_workers_per_gpu[gpu_id] = 1 if num_workers_per_gpu == -1 else num_workers_per_gpu

    return available_gpus, max_workers_per_gpu


def limit_dual_gpu_worker_layout(available_gpus, pending_cases):
    """根据待处理 case 数限制完整 GPU 对的数量。"""
    if len(available_gpus) % 2:
        raise ValueError("dual-GPU worker layout requires complete GPU pairs")
    pair_budget = max(0, pending_cases)
    pair_count = min(len(available_gpus) // 2, pair_budget)
    selected_gpus = list(available_gpus[: pair_count * 2])
    return selected_gpus, dict.fromkeys(selected_gpus, 1)


def _handle_external_kill_retry(
    retry_state,
    pending_dispatch,
    api_config_str,
    *,
    max_case_retries=MAX_EXTERNAL_KILL_RETRIES_PER_CASE,
    max_total_external_kills=None,
):
    """每个 case 只重试一次；若外部 kill 持续发生，则标记运行环境不安全。"""
    retry_state.total_external_kills += 1
    if (
        max_total_external_kills is not None
        and retry_state.total_external_kills > max_total_external_kills
    ):
        retry_state.unsafe_environment = True
        return False

    retry_count = retry_state.per_case_external_kill_retries.get(api_config_str, 0)
    if retry_count < max_case_retries:
        retry_state.per_case_external_kill_retries[api_config_str] = retry_count + 1
        pending_dispatch.appendleft(api_config_str)
        return True

    retry_state.unsafe_environment = True
    return False


def resolve_batch_worker_layout(
    available_gpus,
    max_workers_per_gpu,
    pending_cases,
    *,
    dual_gpu=False,
):
    """校验并裁剪当前 batch 的 worker 布局。"""
    if dual_gpu:
        available_gpus, max_workers_per_gpu = limit_dual_gpu_worker_layout(
            available_gpus,
            pending_cases,
        )
        configured_worker_count = len(available_gpus) // 2
        if configured_worker_count > MAX_TOTAL_WORKERS:
            raise ValueError(
                f"configured worker count {configured_worker_count} exceeds the engine limit "
                f"{MAX_TOTAL_WORKERS}"
            )
        gpu_pairs = list(zip(available_gpus[::2], available_gpus[1::2], strict=True))
    else:
        available_gpus, max_workers_per_gpu = limit_worker_layout(
            available_gpus,
            max_workers_per_gpu,
            pending_cases,
        )
        configured_worker_count = sum(max_workers_per_gpu.values())
        if configured_worker_count > MAX_TOTAL_WORKERS:
            raise ValueError(
                f"configured worker count {configured_worker_count} exceeds the engine limit "
                f"{MAX_TOTAL_WORKERS}"
            )
        gpu_pairs = None
    return available_gpus, max_workers_per_gpu, gpu_pairs


def _fill_idle_workers(pool, pending_dispatch, config_iter):
    """优先用 pending 队列补满空闲 worker，再消费新的 case。"""
    dispatched_count = 0
    while True:
        slot = next(pool.idle_slots(), None)
        if slot is None:
            break
        if pending_dispatch:
            config = pending_dispatch.popleft()
        else:
            config = next(config_iter, None)
        if config is None:
            break
        pool.dispatch(slot.index, config)
        dispatched_count += 1
    return dispatched_count


def _handle_batch_result(
    *,
    pool,
    options,
    all_case,
    checkpointed_case,
    batch_state,
    retry_state,
    pending_dispatch,
    msg,
    max_total_external_kills,
):
    """处理单条 case 终态消息，并维护批处理状态。"""
    message = BatchMessage.from_raw(msg)
    msg_type = message.msg_type
    config = message.config
    slot_index = message.slot_index
    exitcode = message.exitcode
    crash_source = message.crash_source
    reason = message.reason

    if message.worker_pid is not None:
        log_aggregation.mark_inorder_case_complete(
            message.worker_pid,
            message.completed_offset,
        )

    worker_reusable = msg_type in ("done", "error", "deferred") or (
        msg_type == "crashed" and options.use_compute_sanitizer and crash_source == "child"
    )
    external_kill = msg_type == "crashed" and exitcode in (-signal.SIGKILL, -signal.SIGTERM)
    batch_state.active_tasks -= 1

    if external_kill:
        if _handle_external_kill_retry(
            retry_state,
            pending_dispatch,
            config,
            max_case_retries=MAX_EXTERNAL_KILL_RETRIES_PER_CASE,
            max_total_external_kills=max_total_external_kills,
        ):
            log_report.print_case_notice("RETRY", config, f"exit {exitcode}")
            if worker_reusable:
                pool.mark_idle(slot_index)
        else:
            log_report.print_case_notice(
                "ABORT",
                config,
                f"exit {exitcode} | unsafe environment",
            )
            batch_state.batch_exit_code = 1
            batch_state.shutdown_force = True
            batch_state.abort_run = True
            pending_dispatch.clear()
        return

    if worker_reusable:
        pool.mark_idle(slot_index)

    if msg_type == "deferred":
        pending_dispatch.append(config)
        log_report.print_case_notice("DEFERRED", config, reason)
        return

    batch_state.tested_case += 1
    progress_status = "DONE"
    progress_detail = None

    if msg_type == "timeout":
        log_worker.write_to_log("timeout", config)
        progress_status = "TIMEOUT"
    elif msg_type == "crashed":
        log_type, progress_status, terminal_recorded = log_worker.classify_exit(exitcode)
        if crash_source == "child":
            terminal_recorded = False
        if (
            progress_status == "PADDLE_CRASH"
            and options.use_compute_sanitizer
            and exitcode == options.sanitizer_error_exitcode
        ):
            log_type = "paddle_cuda"
            progress_status = "PADDLE_CUDA"
            terminal_recorded = False
            progress_detail = f"sanitizer exit {exitcode}"
        elif progress_status == "PADDLE_CRASH":
            progress_detail = f"exit {exitcode}"
        if not terminal_recorded:
            log_worker.write_to_log(log_type, config)
    elif msg_type == "error":
        log_worker.write_to_log("config_parse", config)
        progress_status = "CONFIG_PARSE"
        progress_detail = reason

    if (
        options.show_runtime_status
        or batch_state.tested_case % 10000 == 0
        or progress_status != "DONE"
    ):
        log_report.print_case_progress(
            checkpointed_case + batch_state.tested_case,
            checkpointed_case + all_case,
            progress_status,
            config,
            progress_detail,
        )

    if (
        options.show_runtime_status
        and batch_state.tested_case < all_case
        and batch_state.test_started_at is not None
        and batch_state.last_forecast_at is not None
    ):
        now = time.monotonic()
        elapsed = now - batch_state.test_started_at
        rate = batch_state.tested_case / elapsed
        if batch_state.last_forecast_case == 0:
            forecast_due = elapsed >= FORECAST_MIN_INTERVAL_SECONDS and (
                batch_state.tested_case >= FORECAST_TARGET_CASES
                or elapsed >= FORECAST_INITIAL_MAX_WAIT_SECONDS
            )
        else:
            forecast_interval = max(
                FORECAST_MIN_INTERVAL_SECONDS,
                min(FORECAST_MAX_INTERVAL_SECONDS, FORECAST_TARGET_CASES / rate),
            )
            forecast_due = now - batch_state.last_forecast_at >= forecast_interval
        if forecast_due:
            eta = (all_case - batch_state.tested_case) / rate
            log_report.print_batch_forecast(
                checkpointed_case + batch_state.tested_case,
                checkpointed_case + all_case,
                rate,
                elapsed,
                eta,
            )
            batch_state.last_forecast_at = now
            batch_state.last_forecast_case = batch_state.tested_case

    log_worker.write_to_log("checkpoint", config)

    if batch_state.tested_case % 1000 == 0:
        log_aggregation.aggregate_logs()


def _install_batch_signal_handlers(cleanup_handler):
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, cleanup_handler)
    return previous_handlers


def _restore_batch_signal_handlers(previous_handlers):
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)


def _run_batch_mode(
    *,
    options,
    api_configs,
    all_case,
    checkpointed_case,
    available_gpus,
    max_workers_per_gpu,
    start_time,
):
    batch_state = BatchRunState()
    retry_state = BatchRetryState()
    max_total_external_kills = MAX_TOTAL_EXTERNAL_KILL_EVENTS
    pool = None
    previous_signal_handlers = {}

    try:
        # batch 执行分为三段：
        # 启动 worker、处理结果、收尾日志和信号处理器。
        log_report.print_running_banner()
        # 批量路径只在这里统一收集一次，避免池内再次探测同一批 GPU。
        gpu_total_memory_map = _build_gpu_total_memory_map(available_gpus)

        pool = WorkerPool(
            available_gpus,
            max_workers_per_gpu,
            options,
            gpu_total_memory_map=gpu_total_memory_map,
        )

        def cleanup_handler(*args):
            print(f"\n{datetime.now()} Cleanup started", flush=True)
            if pool is not None:
                try:
                    pool.shutdown(force=True)
                except Exception as e:
                    print(f"{datetime.now()} Error shutting down pool: {e}", flush=True)
            print(f"{datetime.now()} Cleanup completed", flush=True)
            sys.exit(1)

        previous_signal_handlers = _install_batch_signal_handlers(cleanup_handler)

        worker_start_time = time.monotonic()
        print(f"Workers: starting | {len(pool.slots)} requested", flush=True)
        pool.start()
        ready_workers = pool.warmup()
        print(
            f"Workers: ready | {ready_workers} online | {len(pool.slots)} requested | "
            f"{log_report.format_duration(time.monotonic() - worker_start_time)}",
            flush=True,
        )

        if ready_workers != len(pool.slots):
            print(
                "Workers: failed | startup barrier incomplete; no cases will be dispatched",
                flush=True,
            )
            batch_state.batch_exit_code = 1
            batch_state.shutdown_force = True
            batch_state.abort_run = True

        config_iter = iter(api_configs)
        pending_dispatch = deque()

        def refill_idle_workers():
            if not batch_state.abort_run:
                batch_state.active_tasks += _fill_idle_workers(
                    pool,
                    pending_dispatch,
                    config_iter,
                )

        # 先把当前已经空闲的 worker 填满，让首轮尽快开始。
        if not batch_state.abort_run:
            refill_idle_workers()
            if batch_state.active_tasks:
                batch_state.test_started_at = time.monotonic()
                batch_state.last_forecast_at = batch_state.test_started_at

        # 主循环只负责收消息、补空闲 worker、以及把结果交给专门的处理逻辑。
        while (batch_state.active_tasks > 0 or pending_dispatch) and not batch_state.abort_run:
            msg = pool.collect_one(timeout=5.0)
            if msg is not None and not pool._handle_worker_control_message(msg):
                msg_type = msg[0]
                if msg_type == "ack":
                    slot_idx = msg[1]
                    with pool._lock:
                        pool.slots[slot_idx].task_start_time = time.time()
                elif msg_type in ("done", "error", "timeout", "deferred", "crashed"):
                    _handle_batch_result(
                        pool=pool,
                        options=options,
                        all_case=all_case,
                        checkpointed_case=checkpointed_case,
                        batch_state=batch_state,
                        retry_state=retry_state,
                        pending_dispatch=pending_dispatch,
                        msg=msg,
                        max_total_external_kills=max_total_external_kills,
                    )
            refill_idle_workers()

    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
        batch_state.batch_exit_code = 1
        batch_state.shutdown_force = True
        batch_state.abort_run = True
    finally:
        # 在进入 pool 清理前恢复进程级信号处理器。
        _restore_batch_signal_handlers(previous_signal_handlers)
        if pool is not None:
            pool.shutdown(force=batch_state.shutdown_force)
        if options.use_compute_sanitizer:
            log_worker.clean_sanitizer_case_logs()
        log_counts = log_aggregation.finalize_logs()
        if (
            options.retest
            and batch_state.batch_exit_code == 0
            and batch_state.tested_case == all_case
        ):
            log_retest.finish_retest()
        log_report.print_run_footer(
            all_case,
            batch_state.tested_case,
            max(all_case - batch_state.tested_case, 0),
            log_counts,
            time.time() - start_time,
            options.log_dir,
        )
    return batch_state.batch_exit_code


def _build_case_runtime_context(api_config_str, options):
    started_at = time.monotonic()
    visible_gpu_ids = tuple(
        int(value) for value in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    )
    gpu_id = visible_gpu_ids[0]
    comparison_gpu_id = (
        visible_gpu_ids[1]
        if getattr(options, "accuracy_stable_dual_gpu", False) and len(visible_gpu_ids) > 1
        else None
    )
    suppress_case_tags = os.environ.get("PADDLEAPITEST_SUPPRESS_CASE_TAGS") == "1"
    if not suppress_case_tags:
        log_worker.write_case_begin(
            api_config_str,
            worker_pid=os.getpid(),
            slot=os.environ.get("PADDLEAPITEST_WORKER_SLOT"),
            gpu=gpu_id,
            paddle_version=options.paddle_version,
        )
    return CaseRuntimeContext(
        started_at=started_at,
        gpu_id=gpu_id,
        comparison_gpu_id=comparison_gpu_id,
        suppress_case_tags=suppress_case_tags,
    )


def _handle_case_exception(api_config_str, err):
    err_msg = str(err).lower()
    terminal_log_type = log_worker.get_terminal_log_type(api_config_str)
    fatal_log_type = None
    if any(marker in err_msg for marker in OOM_ERROR_MARKERS):
        fatal_log_type = "oom"
    elif terminal_log_type == "torch_error" and any(
        marker in err_msg for marker in CUDA_ERROR_MARKERS
    ):
        fatal_log_type = "torch_error"
    elif any(marker in err_msg for marker in CUDA_ERROR_MARKERS):
        fatal_log_type = "paddle_cuda"
    if fatal_log_type is not None:
        exit_code = log_worker.fatal_exit_code(fatal_log_type, terminal_log_type == fatal_log_type)
        if dump_enabled():
            record_dump_terminal_status("engine_fatal", exit_code=exit_code, error=str(err))
        try:
            log_runtime.close_process_files()
        finally:
            try:
                log_worker.restore_stdio()
            finally:
                os._exit(exit_code)
    if terminal_log_type is not None:
        return True
    print(f"[test error] {api_config_str}: {err}", flush=True)
    return False


def _cleanup_case_runtime(options):
    if not getattr(options, "use_gpu_mode", False):
        gc.collect()
    if not any(getattr(options, opt) for opt in GPU_PERFORMANCE_MODES) and not getattr(
        options, "use_gpu_mode", False
    ):
        _clear_device_cache(options)


def _validate_input_sources(options):
    input_sources = (
        bool(options.api_config),
        bool(options.api_config_file),
        bool(options.api_config_file_pattern),
        bool(options.retest),
    )
    if sum(input_sources) != 1:
        return _argument_error(
            "exactly one of --api_config, --api_config_file, "
            "--api_config_file_pattern, or --retest is required"
        )
    return None


def _validate_test_mode(options):
    mode = [
        options.accuracy,
        options.paddle_only,
        options.paddle_cinn,
        options.paddle_gpu_performance,
        options.torch_gpu_performance,
        options.paddle_torch_gpu_performance,
        options.accuracy_stable,
        options.paddle_custom_device,
        options.custom_device_vs_gpu,
    ]
    if len([m for m in mode if m is True]) != 1:
        return _argument_error(TEST_MODE_ERROR)
    return None


def _load_custom_device_options(options):
    bos_config_path = Path("tester/bos_config.yaml")
    if not bos_config_path.exists():
        print(f"BOS config file not found: {bos_config_path}", flush=True)
        return 2
    try:
        with open(bos_config_path, encoding="utf-8") as f:
            bos_config_data = yaml.safe_load(f)
        if not bos_config_data:
            print(f"BOS config file is empty: {bos_config_path}", flush=True)
            return 2
        required_keys = ["bos_path", "bos_conf_path", "bcecmd_path"]
        missing_keys = [key for key in required_keys if key not in bos_config_data]
        if missing_keys:
            print(f"Missing required keys in BOS config: {missing_keys}", flush=True)
            return 2
        options.operation_mode = options.custom_device_vs_gpu_mode
        options.bos_path = bos_config_data["bos_path"]
        options.bos_conf_path = bos_config_data["bos_conf_path"]
        options.bcecmd_path = bos_config_data["bcecmd_path"]
    except Exception as err:
        print(f"Failed to load BOS config file {bos_config_path}: {err}", flush=True)
        return 2
    return None


def _apply_runtime_environment_flags(options):
    if options.use_gpu_mode and options.use_cached_numpy:
        print(
            f"{ARGUMENT_WARNING_PREFIX} "
            "--use_cached_numpy=True is ignored because --use_gpu_mode=True uses GPU "
            "tensor generation",
            flush=True,
        )
        options.use_cached_numpy = False
    os.environ["USE_CACHED_NUMPY"] = str(options.use_cached_numpy)
    os.environ["USE_GPU_MODE"] = str(options.use_gpu_mode)
    if options.bitwise_alignment:
        options.atol = 0.0
        options.rtol = 0.0


def _detect_paddle_version():
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("paddlepaddle-gpu")
    except Exception:
        try:
            from importlib.metadata import version as _pkg_version

            return _pkg_version("paddlepaddle")
        except Exception:
            return "unknown"


def run_test_case(api_config_str, options):
    """运行指定 API 配置的单个测试 case。"""
    case_context = _build_case_runtime_context(api_config_str, options)
    test_class = api_config = case = None
    case_status = "done"
    try:
        case_context.runtime_config = runtime_config_for_gpu(
            options,
            case_context.gpu_id,
            comparison_gpu_id=case_context.comparison_gpu_id,
        )
        try:
            api_config = APIConfig(api_config_str)
        except Exception as err:
            log_worker.emit_case_result("config_parse", api_config_str, message=str(err))
            case_status = "error"
            return

        test_class = _select_test_class(options)
        kwargs = {k: v for k, v in vars(options).items() if k in VALID_TEST_ARGS}
        kwargs["runtime_config"] = case_context.runtime_config
        case = test_class(api_config, **kwargs)
        try:
            if dump_enabled():
                case.run_with_dump()
            else:
                case.test()
        except Exception as err:
            if _handle_case_exception(api_config_str, err):
                return
            raise
        finally:
            del test_class, api_config, case
            _cleanup_case_runtime(options)

    except GpuMemoryDeferred:
        case_status = "deferred"
        raise
    except BaseException:
        case_status = "error"
        raise
    finally:
        if not case_context.suppress_case_tags:
            log_worker.write_case_end(
                case_status,
                api_config_str=api_config_str,
                duration_ms=round((time.monotonic() - case_context.started_at) * 1000),
            )


def _prepare_common_options(options):
    try:
        options.retest_types = log_retest.parse_retest_types(options.retest)
    except ValueError as err:
        return _argument_error(str(err))

    normalize_accuracy_stable_dual_gpu_options(options)
    if options.api_config and not options.test_cpu:
        _apply_single_config_gpu_defaults(options)

    common_error = _validate_input_sources(options)
    if common_error is not None:
        return common_error

    mode_error = _validate_test_mode(options)
    if mode_error is not None:
        return mode_error

    if not options._sanitizer_child:
        log_report.print_run_header(options, options.paddle_version)

    if options.use_dump:
        if not options.api_config or options.api_config_file or options.api_config_file_pattern:
            return _argument_error("dump only supports single --api_config runs")
        if not (options.accuracy or options.paddle_only):
            return _argument_error("dump currently supports only --accuracy or --paddle_only")

    if options.custom_device_vs_gpu:
        custom_device_error = _load_custom_device_options(options)
        if custom_device_error is not None:
            return custom_device_error

    if options.test_tol and not options.accuracy:
        print(
            f"{ARGUMENT_WARNING_PREFIX} --test_tol takes effect only when --accuracy=True",
            flush=True,
        )
    if options.test_backward and not options.paddle_cinn:
        print(
            f"{ARGUMENT_WARNING_PREFIX} --test_backward takes effect only when --paddle_cinn=True",
            flush=True,
        )
    _apply_runtime_environment_flags(options)
    return None


def _run_sanitizer_child_mode(options):
    try:
        _init_worker_runtime(None, None, None, options, redirect_output=False)
        options.api_config = options.api_config.strip()
        run_test_case(options.api_config, options)
    except SystemExit:
        raise
    except Exception as err:
        print(f"[test error] {options.api_config}: {err}", flush=True)
        return 2
    finally:
        try:
            log_runtime.close_process_files()
        except Exception:
            pass
    return 0


def _load_retest_configs(options):
    api_configs = log_retest.prepare_retest(options.retest_types)
    removed_stale_logs = log_retest.cleanup_uncheckpointed_result_logs()
    finish_configs = log_runtime.read_log("checkpoint")
    api_config_count = len(api_configs)
    skipped_non_config = 0
    dup_case = 0
    read_count = api_config_count
    api_configs = sorted(api_configs - finish_configs)
    finish_case = api_config_count - len(api_configs)
    return BatchConfigLoadResult(
        api_configs=api_configs,
        read_count=read_count,
        skipped_non_config=skipped_non_config,
        duplicate_case=dup_case,
        finish_case=finish_case,
        removed_stale_logs=removed_stale_logs,
    )


def _load_file_configs(options, finish_configs, removed_stale_logs):
    if options.api_config_file_pattern:
        import glob

        config_files = []
        patterns = options.api_config_file_pattern.split(",")
        for pattern in patterns:
            pattern = pattern.strip()
            config_files.extend(glob.glob(pattern))
        if not config_files:
            raise FileNotFoundError(f"No config files found: {options.api_config_file_pattern}")
        config_files.sort()
        print("Config files to be tested:", flush=True)
        for i, config_file in enumerate(config_files, 1):
            print(f"{i}. {config_file}", flush=True)
    else:
        if not os.path.exists(options.api_config_file):
            raise FileNotFoundError(f"No config file found: {options.api_config_file}")
        config_files = [options.api_config_file]

    api_config_count = 0
    skipped_non_config = 0
    api_configs = set()
    for config_file in config_files:
        try:
            with open(config_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if not line.startswith("paddle."):
                        skipped_non_config += 1
                        continue
                    api_config_count += 1
                    api_configs.add(line)
        except Exception as err:
            raise OSError(f"Failed to read config file {config_file}: {err}") from err

    dup_case = api_config_count - len(api_configs)
    read_count = api_config_count + skipped_non_config
    api_config_count = len(api_configs)
    api_configs = sorted(api_configs - finish_configs)
    finish_case = api_config_count - len(api_configs)
    return BatchConfigLoadResult(
        api_configs=api_configs,
        read_count=read_count,
        skipped_non_config=skipped_non_config,
        duplicate_case=dup_case,
        finish_case=finish_case,
        removed_stale_logs=removed_stale_logs,
    )


def _load_batch_configs(options):
    if options.retest:
        return _load_retest_configs(options)
    removed_stale_logs = log_retest.cleanup_uncheckpointed_result_logs()
    finish_configs = log_runtime.read_log("checkpoint")
    return _load_file_configs(options, finish_configs, removed_stale_logs)


def _run_single_case_mode(options, start_time):
    if options.use_compute_sanitizer:
        return _run_single_config_with_sanitizer(options)

    try:
        _prepare_single_config_gpu(options)
    except ValueError as err:
        return _argument_error(str(err))

    # 单 case 执行与 worker 复用同样的静默 Paddle/bootstrap 路径。
    _init_runtime_modules(options)
    init_log(options.log_dir)

    options.api_config = options.api_config.strip()
    single_case_error = None
    try:
        run_test_case(options.api_config, options)
        log_worker.write_to_log("checkpoint", options.api_config)
    except Exception as err:
        single_case_error = err
        print(f"[test error] {options.api_config}: {err}", flush=True)
    finally:
        log_runtime.close_process_files()
        log_counts = log_aggregation.finalize_logs()
        completed_case = log_counts.get("checkpoint", 0)
        remaining_case = max(1 - completed_case, 0)
        log_report.print_run_footer(
            1,
            completed_case,
            remaining_case,
            log_counts,
            time.time() - start_time,
            options.log_dir,
        )
    if single_case_error is not None:
        return 1
    return 0


def _run_batch_case_mode(options, start_time):
    init_log(options.log_dir)

    # 批量任务重启时，从 .tmp 目录恢复已有 worker 日志。
    if not log_aggregation.recover_logs():
        return _argument_error(
            "failed to recover worker logs; fix the reported log error before retrying"
        )
    if options.use_compute_sanitizer:
        log_worker.clean_sanitizer_case_logs()
    try:
        batch_configs = _load_batch_configs(options)
    except (OSError, ValueError) as err:
        if options.retest:
            return _argument_error(str(err))
        print(str(err), flush=True)
        return 2

    log_report.print_preparing_summary(
        batch_configs.read_count,
        batch_configs.skipped_non_config,
        batch_configs.duplicate_case,
        batch_configs.all_case + batch_configs.finish_case,
        batch_configs.finish_case,
        batch_configs.all_case,
        removed_stale_logs=batch_configs.removed_stale_logs,
        retest_types=options.retest_types,
    )

    api_configs = batch_configs.api_configs
    all_case = batch_configs.all_case
    if not api_configs:
        if options.retest:
            log_retest.finish_retest()
        log_report.print_running_banner()
        print("Workers: skipped | 0 pending", flush=True)
        log_counts = log_aggregation.finalize_logs()
        log_report.print_run_footer(
            0,
            0,
            0,
            log_counts,
            time.time() - start_time,
            options.log_dir,
        )
        return 0

    # 校验 GPU 可见性并推导每张 GPU 的 worker 数量。
    gpu_ids = validate_gpu_options(options)
    available_gpus, max_workers_per_gpu = check_gpu_memory(gpu_ids, options.num_workers_per_gpu)
    if not available_gpus:
        print("No usable GPUs available.", flush=True)
        return 2
    if options.accuracy_stable_dual_gpu and len(available_gpus) != len(gpu_ids):
        print("Not all selected GPUs are usable; no complete dual-GPU layout.", flush=True)
        return 2

    try:
        available_gpus, max_workers_per_gpu, gpu_pairs = resolve_batch_worker_layout(
            available_gpus,
            max_workers_per_gpu,
            all_case,
            dual_gpu=options.accuracy_stable_dual_gpu,
        )
    except ValueError as err:
        return _argument_error(str(err))

    if options.use_compute_sanitizer:
        sanitizer_cmd = _validate_sanitizer_command(options.sanitizer_command)
        if sanitizer_cmd is None:
            return 2
        options.sanitizer_cmd = sanitizer_cmd

    log_report.print_compute_summary(
        available_gpus,
        max_workers_per_gpu,
        gpu_pairs=gpu_pairs,
    )

    if options.test_cpu:
        print(f"CPU: {cpu_count()} available | Paddle CPU mode", flush=True)

    return _run_batch_mode(
        options=options,
        api_configs=api_configs,
        all_case=all_case,
        checkpointed_case=batch_configs.finish_case,
        available_gpus=available_gpus,
        max_workers_per_gpu=max_workers_per_gpu,
        start_time=start_time,
    )


def _build_argument_parser():
    parser = argparse.ArgumentParser(description="Run Paddle API test cases", allow_abbrev=False)
    parser.add_argument(
        "--api_config_file",
        default="",
        help=(
            "Path to a config file. Mutually exclusive with "
            "--api_config_file_pattern, --api_config, and --retest."
        ),
    )
    parser.add_argument(
        "--api_config_file_pattern",
        default="",
        help="Glob pattern(s) for config files; comma-separated patterns are supported.",
    )
    parser.add_argument(
        "--api_config",
        default="",
        help=(
            "Run one API config string directly. Single-case mode uses one GPU, or one "
            "GPU pair with --accuracy_stable_dual_gpu=True."
        ),
    )
    parser.add_argument(
        "--retest",
        default="",
        help=(
            "Retest classifications from --log_dir, e.g. config_input or "
            "config_input,timeout. Mutually exclusive with other config inputs."
        ),
    )
    parser.add_argument(
        "--paddle_only",
        type=parse_bool,
        default=False,
        help="Run Paddle-only API support checks.",
    )
    parser.add_argument(
        "--paddle_cinn",
        type=parse_bool,
        default=False,
        help="Run Paddle dynamic graph vs CINN checks.",
    )
    parser.add_argument(
        "--accuracy",
        type=parse_bool,
        default=False,
        help="Run Paddle vs corresponding Torch accuracy checks.",
    )
    parser.add_argument(
        "--paddle_gpu_performance",
        type=parse_bool,
        default=False,
        help="Run Paddle GPU performance checks.",
    )
    parser.add_argument(
        "--torch_gpu_performance",
        type=parse_bool,
        default=False,
        help="Run Torch GPU performance checks.",
    )
    parser.add_argument(
        "--paddle_torch_gpu_performance",
        type=parse_bool,
        default=False,
        help="Run Paddle and Torch GPU performance checks.",
    )
    parser.add_argument(
        "--accuracy_stable",
        type=parse_bool,
        default=False,
        help="Run stable Paddle vs corresponding Torch accuracy checks.",
    )
    parser.add_argument(
        "--accuracy_stable_dual_gpu",
        type=parse_bool,
        default=False,
        help=("Use one compute GPU and one full-result comparison GPU per accuracy-stable worker."),
    )
    parser.add_argument(
        "--paddle_custom_device",
        type=parse_bool,
        default=False,
        help="Run Paddle custom device vs CPU checks.",
    )
    parser.add_argument(
        "--test_amp",
        type=parse_bool,
        default=False,
        help="Enable auto mixed precision (AMP) checks.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use. Use -1 for all selected GPUs.",
    )
    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=1,
        help="Workers per GPU. In gpu_mode, -1 uses one worker per GPU.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="GPU IDs to use, e.g. '0', '0,2', '0-3'. Use '-1' for all GPUs.",
    )
    parser.add_argument(
        "--test_cpu",
        type=parse_bool,
        default=False,
        help="Run Paddle in CPU mode only; Torch reference still runs on GPU.",
    )
    parser.add_argument(
        "--use_cached_numpy",
        type=parse_bool,
        default=False,
        help="Reuse cached NumPy inputs when available.",
    )
    parser.add_argument(
        "--use_gpu_mode",
        type=parse_bool,
        default=False,
        help="Enable GPU tensor generation, GPU compare, and CUDA allocator reuse for speed.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Directory for test logs; default is logs/test_log_<timestamp>.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for accuracy checks.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for accuracy checks.",
    )
    parser.add_argument(
        "--manual_threshold_config_file",
        type=str,
        default="",
        help="YAML file with per-API manual accuracy thresholds",
    )
    parser.add_argument(
        "--test_tol",
        type=parse_bool,
        default=False,
        help="Enable tolerance range checks in accuracy mode.",
    )
    parser.add_argument(
        "--test_backward",
        type=parse_bool,
        default=False,
        help="Enable backward checks in paddle_cinn mode.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per test case, in seconds.",
    )
    parser.add_argument(
        "--show_runtime_status",
        type=parse_bool,
        default=True,
        help="Show real-time progress; when False, only failed cases are printed.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="NumPy random seed.",
    )
    parser.add_argument(
        "--custom_device_vs_gpu",
        type=parse_bool,
        default=False,
        help="Run Paddle custom device vs GPU checks.",
    )
    parser.add_argument(
        "--custom_device_vs_gpu_mode",
        type=str,
        choices=["upload", "download"],
        default="upload",
        help="Operation mode for custom_device_vs_gpu.",
    )
    parser.add_argument(
        "--bitwise_alignment",
        type=parse_bool,
        default=False,
        help="Use bitwise alignment for accuracy checks.",
    )
    parser.add_argument(
        "--generate_failed_tests",
        type=parse_bool,
        default=False,
        help="Generate reproducible test files for failed cases.",
    )
    parser.add_argument(
        "--exit_on_error",
        type=parse_bool,
        default=False,
        help="Exit the process when a paddle_error occurs.",
    )
    parser.add_argument(
        "--use_dump",
        type=parse_strict_bool,
        default=None,
        help="Enable dump tracing (True or False). Overrides USE_DUMP.",
    )
    parser.add_argument(
        "--dump_dir",
        default=None,
        help="Dump output directory. Overrides DUMP_DIR; empty uses the default directory.",
    )
    parser.add_argument(
        "--use_compute_sanitizer",
        type=parse_bool,
        default=False,
        help="Run each case in a compute-sanitizer wrapped subprocess.",
    )
    parser.add_argument(
        "--sanitizer_command",
        type=str,
        default="compute-sanitizer --target-processes all --error-exitcode=86",
        help="Command prefix used when --use_compute_sanitizer=True.",
    )
    parser.add_argument(
        "--sanitizer_error_exitcode",
        type=int,
        default=86,
        help="Exit code used by compute-sanitizer when it reports errors.",
    )
    parser.add_argument(
        "--_sanitizer_child",
        type=parse_bool,
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser


def main():
    start_time = time.time()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    paddle_version = _detect_paddle_version()
    parser = _build_argument_parser()
    options = parser.parse_args()
    options.paddle_version = paddle_version
    _resolve_dump_options(parser, options)
    if not options.log_dir:
        options.log_dir = str(log_runtime.default_log_dir(single=bool(options.api_config)))
    if not options._sanitizer_child:
        log_runtime.init_main_output(options.log_dir)
        atexit.register(log_runtime.close_main_output)
    if options.random_seed != parser.get_default("random_seed"):
        np.random.seed(options.random_seed)
    common_error = _prepare_common_options(options)
    if common_error is not None:
        return common_error

    if options._sanitizer_child:
        return _run_sanitizer_child_mode(options)

    if options.api_config:
        return _run_single_case_mode(options, start_time)
    if options.api_config_file or options.api_config_file_pattern or options.retest:
        return _run_batch_case_mode(options, start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
