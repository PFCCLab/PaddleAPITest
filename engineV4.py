from __future__ import annotations

import argparse
import gc
import importlib
import multiprocessing as mp
import os
import queue
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
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import cpu_count, set_start_method
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated.*",
        category=FutureWarning,
    )
    import pynvml

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

from tester.api_config.dump_writer import (
    dump_enabled,
    parse_strict_bool,
    record_dump_terminal_status,
    resolve_dump_options,
)
from tester.api_config.log_writer import *
from tester.api_config.sanitizer_output import analyze_sanitizer_output
from tester.runtime_config import (
    GPU_MEMORY_POLICY_ENV,
    TestRuntimeConfig,
    resolve_gpu_memory_policy,
    runtime_config_for_gpu,
)

os.environ["FLAGS_use_system_allocator"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

FATAL_CUDA_EXIT_CODE = 99
FATAL_OOM_EXIT_CODE = 98
FATAL_TORCH_EXIT_CODE = 97


class GpuMemoryDeferred(Exception):
    """Raised when a GPU-mode case should wait for more free memory."""


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
    "runtime_config",
}

SANITIZER_FORWARD_ARGS = {
    "accuracy",
    "paddle_only",
    "paddle_cinn",
    "paddle_gpu_performance",
    "torch_gpu_performance",
    "paddle_torch_gpu_performance",
    "accuracy_stable",
    "paddle_custom_device",
    "custom_device_vs_gpu",
    "custom_device_vs_gpu_mode",
    "test_amp",
    "test_cpu",
    "use_cached_numpy",
    "log_dir",
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

DEVICE_TYPE = None
DEVICE_TYPE_DETECTED = False
DEVICE_COUNT = None  # total number of devices
_MEM_SNAPSHOT = None  # dict: gpu_id -> (total_gb, used_gb)
_MEM_SNAPSHOT_TS = 0.0
_NVML_INITIALIZED = False  # persistent NVML session for repeated memory queries
_MEM_SNAPSHOT_TTL = 2.0  # seconds — snapshot cache ttl


def cleanup(pool):
    print(f"{datetime.now()} Cleanup started", flush=True)
    if pool is not None:
        try:
            pool.shutdown(force=True)
        except Exception as e:
            print(f"{datetime.now()} Error shutting down pool: {e}", flush=True)
    print(f"{datetime.now()} Cleanup completed", flush=True)


# ─── WorkerPool: per-worker queue architecture ───────────────────────────────


@dataclass
class WorkerSlot:
    """Represents one worker process slot with its own input queue."""

    index: int
    gpu_id: int
    process: mp.Process | None = None
    input_queue: mp.Queue | None = None
    current_task: str | None = None
    task_start_time: float | None = None
    child_pid: int | None = None
    state: str = "dead"  # dead, starting, idle, busy


def _import_optional_runtime_module(module_name):
    try:
        importlib.import_module(module_name)
    except Exception:
        pass


def _init_runtime_modules(options):
    with suppress_startup_output():
        import paddle

        globals()["paddle"] = paddle
        if options.test_cpu:
            paddle.device.set_device("cpu")
        elif not getattr(options, "paddle_custom_device", False):
            # CUDA_VISIBLE_DEVICES assigns the worker slot; Paddle still needs an explicit device.
            paddle.set_device("gpu")
        _import_optional_runtime_module("paddlefleet_ops")
        _import_optional_runtime_module("FusedQuantOps")
        globals().update(_load_test_classes(options))


def _init_worker_runtime(slot_index, gpu_id, options, *, redirect_output):
    init_log(options.log_dir, worker_tmp_logs=True)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    _init_runtime_modules(options)

    if redirect_output:
        redirect_stdio()

    if slot_index is not None and gpu_id is not None:
        os.environ["PADDLEAPITEST_WORKER_SLOT"] = str(slot_index)


def _worker_loop(slot_index, gpu_id, input_queue, result_queue, options):
    """Long-running worker process. Receives tasks from input_queue, sends results to result_queue.

    Exit behavior:
        - Normal exit: receives None, releases device resources, and returns gracefully.
        - Fatal CUDA error: run_test_case calls os._exit(99) for unrecoverable CUDA errors
          (corruption, device-side asserts). This bypasses Python cleanup — the main process
          Watchdog detects the dead process via is_alive() check and respawns a new worker.
        - OOM: os._exit(98) for CUDA out-of-memory. Same recovery path as above.
        - Other crashes: any unhandled signal (SIGSEGV etc.) or SIGKILL from Watchdog timeout
          terminates the process. Watchdog detects exitcode != 0 and respawns.

    The main process never dispatches to a dead/restarting worker — upon detecting crash or
    timeout, the next task goes to `pending_dispatch` and is sent after the new worker reports
    "ready".
    """
    # ── GPU initialization (equivalent to init_worker_gpu) ──
    try:
        _init_worker_runtime(slot_index, gpu_id, options, redirect_output=True)
    except Exception as e:
        result_queue.put(("init_failed", slot_index, str(e)))
        return

    # ── Notify main: ready ──
    result_queue.put(("ready", slot_index))

    # ── Task loop ──
    while True:
        try:
            task = input_queue.get()
        except (EOFError, OSError):
            break
        if task is None:  # poison pill
            break

        api_config_str = task
        result_queue.put(("ack", slot_index, api_config_str))

        try:
            run_test_case(api_config_str, options)
            result_queue.put(
                ("done", slot_index, api_config_str, os.getpid(), get_worker_log_offset())
            )
        except GpuMemoryDeferred as e:
            result_queue.put(
                (
                    "deferred",
                    slot_index,
                    api_config_str,
                    str(e),
                    os.getpid(),
                    get_worker_log_offset(),
                )
            )
        except SystemExit:
            # run_test_case calls os._exit for CUDA errors, this shouldn't reach here
            # but if it does via sys.exit, let it propagate
            raise
        except Exception as e:
            result_queue.put(
                (
                    "error",
                    slot_index,
                    api_config_str,
                    str(e),
                    os.getpid(),
                    get_worker_log_offset(),
                )
            )

    # Graceful exit. GPU mode skips per-case collection, so collect its cyclic
    # tensor graphs before framework atexit handlers tear down the device manager.
    try:
        gc.collect()
        close_process_files()
        restore_stdio()
    except Exception:
        pass


def _format_cli_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _build_sanitizer_case_command(api_config_str, options, log_dir, sanitizer_cmd=None):
    if sanitizer_cmd is None:
        sanitizer_cmd = shlex.split(options.sanitizer_command)
    cmd = [
        *sanitizer_cmd,
        sys.executable,
        str(Path(__file__).resolve()),
        f"--api_config={api_config_str}",
        f"--log_dir={log_dir}",
        "--_sanitizer_child=True",
    ]
    for key in SANITIZER_FORWARD_ARGS_SORTED:
        if key == "log_dir":
            continue
        value = getattr(options, key, None)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, bool) and not value:
            continue
        cmd.append(f"--{key}={_format_cli_value(value)}")
    return cmd


def _sanitizer_worker_loop(slot_index, gpu_id, input_queue, result_queue, options):
    init_log(options.log_dir, worker_tmp_logs=True)
    redirect_stdio()

    child_process = None
    sanitizer_cmd = shlex.split(options.sanitizer_command)
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
            case_id = write_case_begin(
                api_config_str,
                worker_pid=os.getpid(),
                slot=slot_index,
                gpu=gpu_id,
            )
            case_log_dir = get_sanitizer_case_log_dir(slot_index, os.getpid())
            if case_log_dir.exists():
                shutil.rmtree(case_log_dir)
            case_log_dir.mkdir(parents=True, exist_ok=True)
            try:
                cmd = _build_sanitizer_case_command(
                    api_config_str, options, str(case_log_dir), sanitizer_cmd
                )
            except ValueError as err:
                shutil.rmtree(case_log_dir, ignore_errors=True)
                completed_offset = write_case_end("error", case_id=case_id)
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
                completed_offset = write_case_end("error", case_id=case_id)
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
                    merge_sanitizer_case_logs(case_log_dir)
                shutil.rmtree(case_log_dir, ignore_errors=True)

                if returncode == 0 or ignored:
                    completed_offset = write_case_end("completed", case_id=case_id)
                    result_queue.put(
                        ("done", slot_index, api_config_str, os.getpid(), completed_offset)
                    )
                elif returncode == 2:
                    completed_offset = write_case_end("error", case_id=case_id)
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
                    completed_offset = write_case_end("crashed", case_id=case_id)
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
            close_process_files()
            restore_stdio()
        except Exception:
            pass


class WorkerPool:
    """Custom process pool with per-worker queues for fair GPU scheduling."""

    def __init__(self, available_gpus, max_workers_per_gpu, options):
        # Convert argparse.Namespace to SimpleNamespace(dict) for cleaner pickling to workers
        if isinstance(options, argparse.Namespace):
            self.options = SimpleNamespace(**vars(options))
        else:
            self.options = options
        self.options.gpu_workers_per_gpu_map = dict(max_workers_per_gpu)
        gpu_total_memory_map = {}
        for gpu_id in available_gpus:
            try:
                gpu_total_memory_map[gpu_id] = get_memory_info(gpu_id)[0]
            except Exception:
                pass
        self.options.gpu_total_memory_map = gpu_total_memory_map
        self.options.runtime_config = TestRuntimeConfig.from_options(self.options)
        self.result_queue = mp.Queue()
        self.slots: list[WorkerSlot] = []
        self._shutdown_event = threading.Event()
        self._watchdog_thread = None
        self._lock = threading.Lock()  # protects slot state modifications
        self._closed = False

        # Build worker slots: deterministic GPU assignment
        idx = 0
        for gpu_id in available_gpus:
            for _ in range(max_workers_per_gpu[gpu_id]):
                slot = WorkerSlot(index=idx, gpu_id=gpu_id)
                self.slots.append(slot)
                idx += 1

    @property
    def total_workers(self):
        return len(self.slots)

    def start(self):
        """Spawn all worker processes and start watchdog thread."""
        for slot in self.slots:
            self._spawn_worker(slot)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="pool-watchdog"
        )
        self._watchdog_thread.start()

    def _close_queue(self, q, *, cancel_join=False):
        """Close a multiprocessing queue without letting cleanup errors mask test results."""
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
        """Spawn a new worker process for the given slot."""
        if self._closed or self._shutdown_event.is_set():
            return False
        if slot.process is not None and not slot.process.is_alive():
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
            args=(slot.index, slot.gpu_id, slot.input_queue, self.result_queue, self.options),
            daemon=True,
        )
        p.start()
        slot.process = p
        slot.state = "starting"
        slot.current_task = None
        slot.task_start_time = None
        slot.child_pid = None
        return True

    def warmup(self, timeout=180):
        """Wait for all workers to report ready."""
        ready_slots = set()
        deadline = time.time() + timeout

        while len(ready_slots) < self.total_workers and time.time() < deadline:
            try:
                remaining = max(0.1, deadline - time.time())
                msg = self.result_queue.get(timeout=min(5.0, remaining))
                msg_type = msg[0]
                if msg_type == "ready":
                    slot_idx = msg[1]
                    with self._lock:
                        self.slots[slot_idx].state = "idle"
                    ready_slots.add(slot_idx)
                elif msg_type == "init_failed":
                    slot_idx = msg[1]
                    error_msg = msg[2]
                    print(
                        f"[worker] INIT_FAILED | slot {slot_idx} | {error_msg}",
                        flush=True,
                    )
                    slot = self.slots[slot_idx]
                    self._join_process(slot.process, timeout=1)
                    self._spawn_worker(slot)
            except queue.Empty:
                # Check for dead processes
                for slot in self.slots:
                    if slot.state == "starting" and slot.process and not slot.process.is_alive():
                        print(
                            f"[worker] INIT_CRASH | slot {slot.index} | "
                            f"exit {slot.process.exitcode}",
                            flush=True,
                        )
                        self._spawn_worker(slot)

        ready_count = len(ready_slots)
        if ready_count < self.total_workers:
            print(
                f"[workers] READY_TIMEOUT | {ready_count}/{self.total_workers} ready | "
                f"timeout {timeout} s",
                flush=True,
            )
        return ready_count

    def dispatch(self, slot_index, config):
        """Send a task to a specific worker slot."""
        slot = self.slots[slot_index]
        with self._lock:
            slot.current_task = config
            slot.task_start_time = None  # set when ack received
            slot.state = "busy"
        slot.input_queue.put(config)

    def collect_one(self, timeout=5.0):
        """Get one message from result_queue. Returns None on timeout."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def idle_slots(self):
        """Yield slots that are currently idle."""
        for slot in self.slots:
            if slot.state == "idle":
                yield slot

    def mark_idle(self, slot_index):
        """Mark a worker slot as idle after task completion."""
        with self._lock:
            slot = self.slots[slot_index]
            slot.state = "idle"
            slot.current_task = None
            slot.task_start_time = None
            slot.child_pid = None

    def _watchdog_loop(self):
        """Periodically check for timeouts and unexpectedly dead workers."""
        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(1.0):
                break
            now = time.time()

            for slot in self.slots:
                if self._shutdown_event.is_set():
                    break
                with self._lock:
                    if self._shutdown_event.is_set():
                        break
                    # Check timeout
                    if (
                        slot.state == "busy"
                        and slot.task_start_time is not None
                        and now - slot.task_start_time > self.options.timeout
                    ):
                        self._handle_timeout(slot)
                        continue

                    if self._shutdown_event.is_set():
                        break

                    # Check unexpected death
                    if (
                        slot.state in ("busy", "idle")
                        and slot.process is not None
                        and not slot.process.is_alive()
                    ):
                        self._handle_crash(slot)

    def _handle_timeout(self, slot):
        """Kill timed-out worker and enqueue timeout result."""
        if self._closed or self._shutdown_event.is_set():
            return
        config = slot.current_task
        old_pid = slot.process.pid if slot.process else None
        self._kill_slot_child(slot)
        self._kill_process(slot.process)
        if old_pid is not None and config is not None:
            completed_offset = append_case_end_to_worker_log(
                old_pid, "timeout", api_config_str=config
            )
            mark_inorder_case_complete(old_pid, completed_offset)
        if self._closed or self._shutdown_event.is_set():
            return
        self.result_queue.put(("timeout", slot.index, config, old_pid))
        self._spawn_worker(slot)

    def _handle_crash(self, slot):
        """Handle unexpectedly dead worker."""
        if self._closed or self._shutdown_event.is_set():
            return
        exitcode = slot.process.exitcode if slot.process else None
        config = slot.current_task
        if self._closed or self._shutdown_event.is_set():
            return
        if config is not None:
            completed_offset = append_case_end_to_worker_log(
                slot.process.pid, "crashed", api_config_str=config
            )
            mark_inorder_case_complete(slot.process.pid, completed_offset)
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
        """Send SIGKILL to a process without waiting for it to exit."""
        try:
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    def _join_process(self, process, timeout=5):
        """Wait for a process to exit, ignoring cleanup-time failures."""
        try:
            process.join(timeout=timeout)
        except Exception:
            pass

    def _kill_process(self, process):
        """SIGKILL a process (CUDA-deadlocked processes don't respond to SIGTERM)."""
        self._sigkill_process(process)
        self._join_process(process, timeout=5)

    def shutdown(self, force=False):
        """Stop all workers and release multiprocessing queues."""
        if self._closed:
            return
        self._closed = True
        self._shutdown_event.set()

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=3)

        try:
            if not force:
                # Graceful: send poison pills
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
                # Force: send SIGKILL to all workers first, then join all.
                # This avoids serial join latency when many CUDA-deadlocked workers exist.
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


def detect_device_type() -> str:
    global DEVICE_TYPE, DEVICE_TYPE_DETECTED, _NVML_INITIALIZED
    if DEVICE_TYPE_DETECTED:
        return DEVICE_TYPE

    # 优先尝试 NVML（NVIDIA GPU）
    try:
        if not _NVML_INITIALIZED:
            pynvml.nvmlInit()
            _NVML_INITIALIZED = True
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            DEVICE_TYPE = "gpu"
            DEVICE_TYPE_DETECTED = True
            return DEVICE_TYPE
    except Exception:
        # 没有 NVML 或不是 NVIDIA，忽略错误，继续往下探测
        pass

    # 再尝试 XPU
    if shutil.which("xpu-smi"):
        try:
            out = subprocess.check_output(["xpu-smi"], text=True, stderr=subprocess.STDOUT)
            if any(re.match(r"^\|\s*\d+\s+\S", line) for line in out.splitlines()):
                DEVICE_TYPE = "xpu"
                DEVICE_TYPE_DETECTED = True
                return DEVICE_TYPE
        except Exception:
            pass

    # 再尝试 Iluvatar
    if shutil.which("ixsmi"):
        try:
            out = subprocess.check_output(["ixsmi"], text=True, stderr=subprocess.STDOUT)
            if any(re.match(r"^\|\s*\d+\s+Iluvatar", line) for line in out.splitlines()):
                DEVICE_TYPE = "iluvatar_gpu"
                DEVICE_TYPE_DETECTED = True
                return DEVICE_TYPE
        except Exception:
            pass

    # 都没有就是 CPU
    DEVICE_TYPE = "cpu"
    DEVICE_TYPE_DETECTED = True
    return DEVICE_TYPE


def get_device_count() -> int:
    """Get the number of available devices (accelerators)."""
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
        out = subprocess.check_output(["xpu-smi"], text=True, stderr=subprocess.STDOUT)
        ids = set()
        for line in out.splitlines():
            if "Processes:" in line:
                break
            m = re.match(r"^\|\s*(\d+)\s+\S", line)
            if m:
                ids.add(int(m.group(1)))
        DEVICE_COUNT = len(ids)
        return DEVICE_COUNT

    if device_type == "iluvatar_gpu":
        out = subprocess.check_output(["ixsmi"], text=True, stderr=subprocess.STDOUT)
        ids = set()
        for line in out.splitlines():
            m = re.match(r"^\|\s*(\d+)\s+Iluvatar", line)
            if m:
                ids.add(int(m.group(1)))
        DEVICE_COUNT = len(ids)
        return DEVICE_COUNT

    # CPU case／no accelerator
    DEVICE_COUNT = 0
    return 0


def _refresh_snapshot(device_type):
    global _MEM_SNAPSHOT, _MEM_SNAPSHOT_TS

    now = time.time()
    if now - _MEM_SNAPSHOT_TS < _MEM_SNAPSHOT_TTL and _MEM_SNAPSHOT is not None:
        return

    snapshot = {}
    if device_type == "xpu":
        out = subprocess.check_output(["xpu-smi"], text=True, stderr=subprocess.STDOUT)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\|\s*(\d+)\s+\S", line)
            if m:
                dev_id = int(m.group(1))
                for j in range(i + 1, min(i + 8, len(lines))):
                    mm = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", lines[j])
                    if mm:
                        used_mib = int(mm.group(1))
                        total_mib = int(mm.group(2))
                        snapshot[dev_id] = (total_mib / 1024.0, used_mib / 1024.0)
                        break

    elif device_type == "iluvatar_gpu":
        out = subprocess.check_output(["ixsmi"], text=True, stderr=subprocess.STDOUT)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\|\s*(\d+)\s+Iluvatar", line)
            if m:
                dev_id = int(m.group(1))
                for j in range(i + 1, min(i + 8, len(lines))):
                    mm = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", lines[j])
                    if mm:
                        used_mib = int(mm.group(1))
                        total_mib = int(mm.group(2))
                        snapshot[dev_id] = (total_mib / 1024.0, used_mib / 1024.0)
                        break

    else:
        # GPU (NVIDIA) case does not use snapshot (use NVML directly)
        _MEM_SNAPSHOT = None
        _MEM_SNAPSHOT_TS = now
        return

    _MEM_SNAPSHOT = snapshot
    _MEM_SNAPSHOT_TS = now


def get_memory_info(gpu_id):
    """Return (total_memory, used_memory) in GB for accelerator device."""
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


ARGUMENT_ERROR_PREFIX = "[argument error]"
ARGUMENT_WARNING_PREFIX = "[argument warning]"
TEST_MODE_ERROR = (
    "specify exactly one test mode: --accuracy, --paddle_only, --paddle_cinn, "
    "--paddle_gpu_performance, --torch_gpu_performance, "
    "--paddle_torch_gpu_performance, --accuracy_stable, --paddle_custom_device, "
    "--custom_device_vs_gpu"
)


def _print_argument(prefix, message):
    print(f"{prefix} {message}", flush=True)


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
            "paddle_custom_device",
            "custom_device_vs_gpu",
        )
    )


def _load_test_classes(options):
    from tester import APIConfig, APITestPaddleOnly

    test_classes = {
        "APIConfig": APIConfig,
        "APITestPaddleOnly": APITestPaddleOnly,
    }
    if _mode_uses_torch(options):
        from tester import (
            APITestAccuracy,
            APITestAccuracyStable,
            APITestCINNVSDygraph,
            APITestCustomDeviceVSCPU,
            APITestPaddleDeviceVSGPU,
            APITestPaddleGPUPerformance,
            APITestPaddleTorchGPUPerformance,
            APITestTorchGPUPerformance,
        )

        test_classes.update(
            {
                "APITestAccuracy": APITestAccuracy,
                "APITestCINNVSDygraph": APITestCINNVSDygraph,
                "APITestPaddleGPUPerformance": APITestPaddleGPUPerformance,
                "APITestTorchGPUPerformance": APITestTorchGPUPerformance,
                "APITestPaddleTorchGPUPerformance": APITestPaddleTorchGPUPerformance,
                "APITestAccuracyStable": APITestAccuracyStable,
                "APITestCustomDeviceVSCPU": APITestCustomDeviceVSCPU,
                "APITestPaddleDeviceVSGPU": APITestPaddleDeviceVSGPU,
            }
        )
    return test_classes


def _select_test_class(options):
    test_classes = _load_test_classes(options)
    option_to_class_name = {
        "paddle_only": "APITestPaddleOnly",
        "paddle_cinn": "APITestCINNVSDygraph",
        "accuracy": "APITestAccuracy",
        "paddle_gpu_performance": "APITestPaddleGPUPerformance",
        "torch_gpu_performance": "APITestTorchGPUPerformance",
        "paddle_torch_gpu_performance": "APITestPaddleTorchGPUPerformance",
        "accuracy_stable": "APITestAccuracyStable",
        "paddle_custom_device": "APITestCustomDeviceVSCPU",
        "custom_device_vs_gpu": "APITestPaddleDeviceVSGPU",
    }
    for opt, class_name in option_to_class_name.items():
        if getattr(options, opt, False):
            return test_classes[class_name]
    return test_classes["APITestAccuracy"]


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


def validate_gpu_options(options) -> tuple:
    """Validate and normalize GPU-related options."""
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
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        elif value in ["false", "0", "no", "n"]:
            return False
    else:
        raise ValueError(f"Invalid boolean value: {value} parsed from command line")


def _apply_single_config_gpu_defaults(options):
    if not options.gpu_ids and options.num_gpus == -1:
        options.gpu_ids = "0"
        options.num_gpus = 1


def _prepare_single_config_gpu(options):
    if options.test_cpu:
        options.gpu_workers_per_gpu_map = {}
        options.gpu_total_memory_map = {}
        options.runtime_config = TestRuntimeConfig.from_options(options)
        return None

    _apply_single_config_gpu_defaults(options)
    gpu_ids = validate_gpu_options(options)
    if len(gpu_ids) != 1:
        raise ValueError(
            f"single --api_config run supports exactly one GPU; got {len(gpu_ids)} GPUs: {gpu_ids}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    try:
        options.gpu_total_memory_map = {gpu_ids[0]: get_memory_info(gpu_ids[0])[0]}
    except Exception:
        options.gpu_total_memory_map = {}
    options.gpu_workers_per_gpu_map = {gpu_ids[0]: 1}
    options.runtime_config = TestRuntimeConfig.from_options(options)
    options.runtime_config = runtime_config_for_gpu(options, gpu_ids[0])
    return gpu_ids[0]


def _validate_sanitizer_command(command):
    try:
        sanitizer_cmd = shlex.split(command)
    except ValueError as err:
        _print_argument(ARGUMENT_ERROR_PREFIX, f"invalid --sanitizer_command: {err}")
        return None
    if not sanitizer_cmd:
        _print_argument(
            ARGUMENT_ERROR_PREFIX,
            "invalid --sanitizer_command: command cannot be empty",
        )
        return None
    if shutil.which(sanitizer_cmd[0]) is None:
        _print_argument(
            ARGUMENT_ERROR_PREFIX,
            f"sanitizer executable not found: {sanitizer_cmd[0]}",
        )
        return None
    return sanitizer_cmd


def _run_single_config_with_sanitizer(options):
    sanitizer_cmd = _validate_sanitizer_command(options.sanitizer_command)
    if sanitizer_cmd is None:
        return 2

    try:
        gpu_id = _prepare_single_config_gpu(options)
    except ValueError as err:
        _print_argument(ARGUMENT_ERROR_PREFIX, str(err))
        return 2

    api_config = options.api_config.strip()
    cmd = _build_sanitizer_case_command(api_config, options, sanitizer_cmd)
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        except pynvml.NVMLError as e:
            print(f"[warn] Failed to check GPU {gpu_id}: {e!s}", flush=True)
            continue
        available_gpus.append(gpu_id)
        max_workers_per_gpu[gpu_id] = 1 if num_workers_per_gpu == -1 else num_workers_per_gpu

    return available_gpus, max_workers_per_gpu


def run_test_case(api_config_str, options):
    """Run a single test case for the given API configuration."""
    started_at = time.monotonic()
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    suppress_case_tags = os.environ.get("PADDLEAPITEST_SUPPRESS_CASE_TAGS") == "1"
    case_id = None
    if not suppress_case_tags:
        case_id = write_case_begin(
            api_config_str,
            worker_pid=os.getpid(),
            slot=os.environ.get("PADDLEAPITEST_WORKER_SLOT"),
            gpu=gpu_id,
            paddle_version=options.paddle_version,
        )
    case_status = "done"
    try:
        runtime_config = runtime_config_for_gpu(options, gpu_id)

        try:
            api_config = APIConfig(api_config_str)
        except Exception as err:
            print(f"[config_parse] {api_config_str} {err!s}", flush=True)
            write_to_log("config_parse", api_config_str)
            case_status = "error"
            return

        test_class = _select_test_class(options)
        kwargs = {k: v for k, v in vars(options).items() if k in VALID_TEST_ARGS}
        kwargs["runtime_config"] = runtime_config
        case = test_class(api_config, **kwargs)
        try:
            if dump_enabled():
                case.run_with_dump()
            else:
                case.test()
        except Exception as err:
            err_msg = str(err).lower()
            terminal_log_type = get_terminal_log_type(api_config_str)
            oom_markers = (
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
            cuda_markers = (
                "cuda error",
                "memory corruption",
                "illegal memory access",
                "invalid configuration argument",
                "invalid resource handle",
            )
            exit_code = None
            if any(marker in err_msg for marker in oom_markers):
                exit_code = FATAL_OOM_EXIT_CODE
            elif terminal_log_type == "torch_error" and any(
                marker in err_msg for marker in cuda_markers
            ):
                exit_code = FATAL_TORCH_EXIT_CODE
            elif any(marker in err_msg for marker in cuda_markers):
                exit_code = FATAL_CUDA_EXIT_CODE
            if exit_code is not None:
                if dump_enabled():
                    record_dump_terminal_status("engine_fatal", exit_code=exit_code, error=str(err))
                if has_terminal_log(api_config_str):
                    write_checkpoint(api_config_str)
                try:
                    close_process_files()
                finally:
                    try:
                        restore_stdio()
                    finally:
                        os._exit(exit_code)
            if has_terminal_log(api_config_str):
                write_checkpoint(api_config_str)
                return
            # if not fatal error, subprocess will be alive and report error
            print(f"[test error] {api_config_str}: {err}", flush=True)
            raise
        finally:
            del test_class, api_config, case
            if not getattr(options, "use_gpu_mode", False):
                gc.collect()
            if not any(
                getattr(options, opt)
                for opt in (
                    "paddle_gpu_performance",
                    "torch_gpu_performance",
                    "paddle_torch_gpu_performance",
                )
            ) and not getattr(options, "use_gpu_mode", False):
                _clear_device_cache(options)

    except GpuMemoryDeferred:
        case_status = "deferred"
        raise
    except BaseException:
        case_status = "error"
        raise
    finally:
        if not suppress_case_tags:
            write_case_end(
                case_status,
                case_id=case_id,
                api_config_str=api_config_str,
                duration_ms=round((time.monotonic() - started_at) * 1000),
            )


def main():
    start_time = time.time()
    set_start_method("spawn")

    try:
        from importlib.metadata import version as _pkg_version

        paddle_version = _pkg_version("paddlepaddle-gpu")
    except Exception:
        try:
            from importlib.metadata import version as _pkg_version

            paddle_version = _pkg_version("paddlepaddle")
        except Exception:
            paddle_version = "unknown"

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
        help="Run one API config string directly. Single-case mode supports at most one GPU.",
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
        help="Run Paddle in CPU mode.",
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
        help="Directory for test logs.",
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
        type=bool,
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

    options = parser.parse_args()
    options.paddle_version = paddle_version
    _resolve_dump_options(parser, options)
    try:
        options.gpu_memory_policy = resolve_gpu_memory_policy()
    except ValueError as err:
        parser.error(str(err))
    if options.random_seed != parser.get_default("random_seed"):
        np.random.seed(options.random_seed)
    try:
        options.retest_types = parse_retest_types(options.retest)
    except ValueError as err:
        _print_argument(ARGUMENT_ERROR_PREFIX, str(err))
        return

    input_sources = (
        bool(options.api_config),
        bool(options.api_config_file),
        bool(options.api_config_file_pattern),
        bool(options.retest),
    )
    if sum(input_sources) != 1:
        _print_argument(
            ARGUMENT_ERROR_PREFIX,
            "exactly one of --api_config, --api_config_file, "
            "--api_config_file_pattern, or --retest is required",
        )
        return
    if options.api_config and not options.test_cpu:
        _apply_single_config_gpu_defaults(options)

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
        _print_argument(ARGUMENT_ERROR_PREFIX, TEST_MODE_ERROR)
        return
    if not options._sanitizer_child:
        print_run_header(options, paddle_version)
    if options.use_dump:
        if not options.api_config or options.api_config_file or options.api_config_file_pattern:
            _print_argument(ARGUMENT_ERROR_PREFIX, "dump only supports single --api_config runs")
            return
        if not (options.accuracy or options.paddle_only):
            _print_argument(
                ARGUMENT_ERROR_PREFIX, "dump currently supports only --accuracy or --paddle_only"
            )
            return

    # 处理 custom_device_vs_gpu 模式的配置
    bos_config_data = None
    if options.custom_device_vs_gpu:
        # 读取 BOS 配置文件（固定路径：tester/bos_config.yaml）
        bos_config_path = Path("tester/bos_config.yaml")
        if not bos_config_path.exists():
            print(f"BOS config file not found: {bos_config_path}", flush=True)
            return

        try:
            with open(bos_config_path, encoding="utf-8") as f:
                bos_config_data = yaml.safe_load(f)

            if not bos_config_data:
                print(f"BOS config file is empty: {bos_config_path}", flush=True)
                return

            # 验证必需的配置项
            required_keys = ["bos_path", "bos_conf_path", "bcecmd_path"]
            missing_keys = [key for key in required_keys if key not in bos_config_data]
            if missing_keys:
                print(f"Missing required keys in BOS config: {missing_keys}", flush=True)
                return

            # 将配置添加到 options 中，以便传递给测试类
            options.operation_mode = options.custom_device_vs_gpu_mode
            options.bos_path = bos_config_data["bos_path"]
            options.bos_conf_path = bos_config_data["bos_conf_path"]
            options.bcecmd_path = bos_config_data["bcecmd_path"]

        except Exception as e:
            print(f"Failed to load BOS config file {bos_config_path}: {e}", flush=True)
            return

    if options.test_tol and not options.accuracy:
        _print_argument(
            ARGUMENT_WARNING_PREFIX, "--test_tol takes effect only when --accuracy=True"
        )
    if options.test_backward and not options.paddle_cinn:
        _print_argument(
            ARGUMENT_WARNING_PREFIX, "--test_backward takes effect only when --paddle_cinn=True"
        )
    if options.use_gpu_mode and options.use_cached_numpy:
        _print_argument(
            ARGUMENT_WARNING_PREFIX,
            "--use_cached_numpy=True is ignored because --use_gpu_mode=True uses GPU "
            "tensor generation",
        )
        options.use_cached_numpy = False
    os.environ["USE_CACHED_NUMPY"] = str(options.use_cached_numpy)
    os.environ["USE_GPU_MODE"] = str(options.use_gpu_mode)
    os.environ[GPU_MEMORY_POLICY_ENV] = options.gpu_memory_policy
    if options.bitwise_alignment:
        options.atol = 0.0
        options.rtol = 0.0

    if options._sanitizer_child:
        try:
            _init_worker_runtime(None, None, options, redirect_output=False)
            options.api_config = options.api_config.strip()
            run_test_case(options.api_config, options)
        except SystemExit:
            raise
        except Exception as err:
            print(f"[test error] {options.api_config}: {err}", flush=True)
            sys.exit(2)
        finally:
            try:
                close_process_files()
            except Exception:
                pass
        return

    if options.api_config:
        if options.use_compute_sanitizer:
            sys.exit(_run_single_config_with_sanitizer(options))
        try:
            _prepare_single_config_gpu(options)
        except ValueError as err:
            _print_argument(ARGUMENT_ERROR_PREFIX, str(err))
            return

        # Single config execution uses the same quiet Paddle/bootstrap path as workers.
        _init_runtime_modules(options)

        init_log(options.log_dir, worker_tmp_logs=True)

        options.api_config = options.api_config.strip()
        try:
            run_test_case(options.api_config, options)
        except Exception as err:
            if (
                "Tensor-likes are not equal" in str(err)
                or "Mismatched elements" in str(err)
                or "Error Message Summary" in str(err)
            ):
                exit(1)
            print(f"[test error] {options.api_config}: {err}", flush=True)
    elif options.api_config_file or options.api_config_file_pattern or options.retest:
        # get config files
        if options.retest:
            config_files = []
        elif options.api_config_file_pattern:
            import glob

            config_files = []
            patterns = options.api_config_file_pattern.split(",")
            for pattern in patterns:
                pattern = pattern.strip()
                config_files.extend(glob.glob(pattern))
            if not config_files:
                print(
                    f"No config files found: {options.api_config_file_pattern}",
                    flush=True,
                )
                return
            config_files.sort()
            print("Config files to be tested:", flush=True)
            for i, config_file in enumerate(config_files, 1):
                print(f"{i}. {config_file}", flush=True)
        else:
            if not os.path.exists(options.api_config_file):
                print(f"No config file found: {options.api_config_file}", flush=True)
                return
            config_files = [options.api_config_file]

        init_log(options.log_dir, worker_tmp_logs=True)

        # when engineV2 was interrupted, resume from .tmp dir
        aggregate_logs(cleanup=True)
        if options.use_compute_sanitizer:
            clean_sanitizer_case_logs()
        removed_stale_logs = 0 if options.retest else cleanup_uncheckpointed_result_logs()

        if options.retest:
            try:
                api_configs = prepare_retest(options.retest_types)
            except (OSError, ValueError) as err:
                _print_argument(ARGUMENT_ERROR_PREFIX, str(err))
                return
            removed_stale_logs = cleanup_uncheckpointed_result_logs()
            api_config_count = len(api_configs)
            skipped_non_config = 0
            finish_configs = read_log("checkpoint")
        else:
            finish_configs = read_log("checkpoint")
            api_config_count = 0
            skipped_non_config = 0
            api_configs = set()
            for config_file in config_files:
                try:
                    with open(config_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if not line.startswith("paddle."):
                                skipped_non_config += 1
                                continue
                            api_config_count += 1
                            api_configs.add(line)
                except Exception as e:
                    print(f"Failed to read config file {config_file}: {e}", flush=True)
                    return
        dup_case = api_config_count - len(api_configs)
        read_count = api_config_count + skipped_non_config
        api_config_count = len(api_configs)
        api_configs = sorted(api_configs - finish_configs)
        all_case = len(api_configs)
        finish_case = api_config_count - all_case
        print_preparing_summary(
            read_count,
            skipped_non_config,
            dup_case,
            api_config_count,
            finish_case,
            all_case,
            removed_stale_logs=removed_stale_logs,
            retest_types=options.retest_types,
        )
        del api_config_count, dup_case, finish_case, read_count

        if not api_configs:
            if options.retest:
                finish_retest()
            print_running_header()
            print("Workers: skipped | 0 pending", flush=True)
            log_counts = aggregate_logs(end=True)
            print_log_info(0, log_counts)
            print_run_footer(
                0,
                0,
                0,
                log_counts,
                time.time() - start_time,
                options.log_dir,
            )
            return

        # validate GPU visibility and derive per-GPU worker counts
        gpu_ids = validate_gpu_options(options)
        available_gpus, max_workers_per_gpu = check_gpu_memory(gpu_ids, options.num_workers_per_gpu)
        if not available_gpus:
            print("No usable GPUs available.", flush=True)
            return
        available_gpus, max_workers_per_gpu = limit_worker_layout(
            available_gpus, max_workers_per_gpu, all_case
        )

        if (
            options.use_compute_sanitizer
            and _validate_sanitizer_command(options.sanitizer_command) is None
        ):
            return

        total_workers = sum(max_workers_per_gpu.values())
        print_compute_summary(available_gpus, max_workers_per_gpu)

        if options.test_cpu:
            print(f"CPU: {cpu_count()} available | Paddle CPU mode", flush=True)

        print_running_header()

        # initialize worker pool (per-worker queue architecture)
        pool = WorkerPool(available_gpus, max_workers_per_gpu, options)

        def cleanup_handler(*args):
            cleanup(pool)
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

        worker_start_time = time.monotonic()
        print(f"Workers: starting | {total_workers} requested", flush=True)
        pool.start()
        ready_workers = pool.warmup(timeout=180)
        requested_field = f" | {total_workers} requested" if ready_workers != total_workers else ""
        print(
            f"Workers: ready | {ready_workers} online{requested_field} | "
            f"{format_duration(time.monotonic() - worker_start_time)}",
            flush=True,
        )

        # dispatch tasks using per-worker queue round-robin
        tested_case = 0
        try:
            config_iter = iter(api_configs)
            active_tasks = 0
            pending_dispatch = []  # configs waiting for a free worker

            # Initial dispatch: one task per idle worker
            for slot in pool.idle_slots():
                config = next(config_iter, None)
                if config is None:
                    break
                pool.dispatch(slot.index, config)
                active_tasks += 1

            # Main loop: collect results and dispatch next
            while active_tasks > 0 or pending_dispatch:
                msg = pool.collect_one(timeout=5.0)
                if msg is None:
                    if pending_dispatch:
                        for slot in pool.idle_slots():
                            if not pending_dispatch:
                                break
                            config = pending_dispatch.pop(0)
                            pool.dispatch(slot.index, config)
                            active_tasks += 1
                    continue  # watchdog handles timeouts/crashes

                msg_type = msg[0]

                if msg_type == "ack":
                    # Worker started processing — record start time
                    slot_idx = msg[1]
                    with pool._lock:
                        pool.slots[slot_idx].task_start_time = time.time()
                    continue

                if msg_type == "ready":
                    # A respawned worker is ready — dispatch pending task if any
                    slot_idx = msg[1]
                    with pool._lock:
                        pool.slots[slot_idx].state = "idle"
                    if pending_dispatch:
                        config = pending_dispatch.pop(0)
                        pool.dispatch(slot_idx, config)
                        active_tasks += 1
                    continue

                if msg_type == "child":
                    slot_idx = msg[1]
                    child_pid = msg[2]
                    with pool._lock:
                        pool.slots[slot_idx].child_pid = child_pid
                    continue

                # Task completed (done/error/timeout/crashed)
                slot_idx = msg[1]
                config = msg[2]
                exitcode = msg[3] if msg_type == "crashed" and len(msg) > 3 else None
                crash_source = "worker"
                worker_pid = None
                completed_offset = None
                if msg_type == "done":
                    worker_pid = msg[3] if len(msg) > 3 else None
                    completed_offset = msg[4] if len(msg) > 4 else None
                elif msg_type == "error":
                    worker_pid = msg[4] if len(msg) > 4 else None
                    completed_offset = msg[5] if len(msg) > 5 else None
                elif msg_type == "timeout":
                    worker_pid = msg[3] if len(msg) > 3 else None
                elif msg_type == "deferred":
                    worker_pid = msg[4] if len(msg) > 4 else None
                    completed_offset = msg[5] if len(msg) > 5 else None
                elif msg_type == "crashed":
                    if len(msg) > 5 and msg[5] == "child":
                        crash_source = "child"
                        worker_pid = msg[6] if len(msg) > 6 else None
                        completed_offset = msg[7] if len(msg) > 7 else None
                    else:
                        worker_pid = msg[4] if len(msg) > 4 else None
                if worker_pid is not None:
                    mark_inorder_case_complete(worker_pid, completed_offset)
                worker_reusable = msg_type in ("done", "error", "deferred") or (
                    msg_type == "crashed"
                    and options.use_compute_sanitizer
                    and crash_source == "child"
                )
                external_kill = msg_type == "crashed" and exitcode in (
                    -signal.SIGKILL,
                    -signal.SIGTERM,
                )
                active_tasks -= 1

                if external_kill:
                    print_case_notice("RETRY", config, f"exit {exitcode}")
                    if worker_reusable:
                        pool.mark_idle(slot_idx)
                    next_config = next(config_iter, None)
                    if next_config is not None:
                        if not worker_reusable:
                            pending_dispatch.append(next_config)
                        else:
                            pool.dispatch(slot_idx, next_config)
                            active_tasks += 1
                    continue

                if worker_reusable:
                    pool.mark_idle(slot_idx)

                if msg_type == "deferred":
                    reason = msg[3] if len(msg) > 3 else "insufficient GPU memory"
                    pending_dispatch.append(config)
                    print_case_notice("DEFERRED", config, reason)
                    continue

                tested_case += 1
                progress_status = "DONE"
                progress_detail = None

                if msg_type == "timeout":
                    write_to_log("timeout", config)
                    progress_status = "TIMEOUT"
                elif msg_type == "crashed":
                    log_type, progress_status = classify_worker_exit(
                        exitcode,
                        FATAL_CUDA_EXIT_CODE,
                        FATAL_OOM_EXIT_CODE,
                        FATAL_TORCH_EXIT_CODE,
                    )
                    if (
                        progress_status == "PADDLE_CRASH"
                        and options.use_compute_sanitizer
                        and exitcode == options.sanitizer_error_exitcode
                    ):
                        log_type = "paddle_cuda"
                        progress_status = "PADDLE_CUDA"
                        progress_detail = f"sanitizer exit {exitcode}"
                    elif progress_status == "PADDLE_CRASH":
                        progress_detail = f"exit {exitcode}"
                    write_to_log(log_type, config)
                elif msg_type == "error":
                    error_msg = msg[3] if len(msg) > 3 else ""
                    write_to_log("config_parse", config)
                    progress_status = "CONFIG_PARSE"
                    progress_detail = error_msg

                if (
                    options.show_runtime_status
                    or tested_case % 10000 == 0
                    or progress_status != "DONE"
                ):
                    print_case_progress(
                        tested_case,
                        all_case,
                        progress_status,
                        config,
                        progress_detail,
                    )

                write_to_log("checkpoint", config)

                # 先派发下一项，再做周期日志聚合，避免空闲 worker 等待共享存储 I/O。
                next_config = next(config_iter, None)
                if next_config is not None:
                    # For timeout/non-sanitizer crashed: worker is restarting, queue for later dispatch
                    if not worker_reusable:
                        pending_dispatch.append(next_config)
                    else:
                        # Worker is alive and ready for next task
                        pool.dispatch(slot_idx, next_config)
                        active_tasks += 1

                # Periodic log aggregation
                if tested_case % 1000 == 0:
                    aggregate_logs()

            pool.shutdown()
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)
            cleanup(pool)
        finally:
            pool.shutdown()
            if options.use_compute_sanitizer:
                clean_sanitizer_case_logs()
            log_counts = aggregate_logs(end=True)
            if options.retest and tested_case == all_case:
                finish_retest()
            print_log_info(max(all_case - tested_case, 0), log_counts)
            end_time = time.time()
            total_time = end_time - start_time
            print_run_footer(
                all_case,
                tested_case,
                max(all_case - tested_case, 0),
                log_counts,
                total_time,
                options.log_dir,
            )


if __name__ == "__main__":
    main()
