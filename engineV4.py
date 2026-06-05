from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
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

from tester.api_config.log_writer import *

os.environ["FLAGS_use_system_allocator"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

VALID_TEST_ARGS = {
    "test_amp",
    "test_backward",
    "atol",
    "rtol",
    "test_tol",
    "operation_mode",
    "bos_path",
    "random_seed",
    "bos_conf_path",
    "bcecmd_path",
    "generate_failed_tests",
    "bitwise_alignment",
    "exit_on_error",
}

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
    # Aggregate .tmp logs before exit so results are not lost
    try:
        aggregate_logs(end=True)
    except Exception as e:
        print(f"{datetime.now()} Error aggregating logs: {e}", flush=True)
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
    state: str = "dead"  # dead, starting, idle, busy


def _worker_loop(slot_index, gpu_id, input_queue, result_queue, options):
    """Long-running worker process. Receives tasks from input_queue, sends results to result_queue.

    Exit behavior:
        - Normal exit: receives None (poison pill) from input_queue, returns gracefully.
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
    if options.log_dir:
        set_test_log_path(options.log_dir)
    set_engineV2()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        import paddle
        import torch

        try:
            import paddlefleet_ops  # noqa: F401
        except ImportError:
            pass
        try:
            import FusedQuantOps  # noqa: F401
        except ImportError:
            pass

        globals()["torch"] = torch
        globals()["paddle"] = paddle

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

        test_classes = {
            "APIConfig": APIConfig,
            "APITestAccuracy": APITestAccuracy,
            "APITestCINNVSDygraph": APITestCINNVSDygraph,
            "APITestPaddleOnly": APITestPaddleOnly,
            "APITestPaddleGPUPerformance": APITestPaddleGPUPerformance,
            "APITestTorchGPUPerformance": APITestTorchGPUPerformance,
            "APITestPaddleTorchGPUPerformance": APITestPaddleTorchGPUPerformance,
            "APITestAccuracyStable": APITestAccuracyStable,
            "APITestCustomDeviceVSCPU": APITestCustomDeviceVSCPU,
            "APITestPaddleDeviceVSGPU": APITestPaddleDeviceVSGPU,
        }
        globals().update(test_classes)

        if options.test_cpu:
            paddle.device.set_device("cpu")

        redirect_stdio()

        print(
            f"{datetime.now()} Worker PID: {os.getpid()}, Slot: {slot_index}, GPU: {gpu_id}",
            flush=True,
        )
    except Exception as e:
        print(f"{datetime.now()} Worker {os.getpid()} init failed: {e}", flush=True)
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
            result_queue.put(("done", slot_index, api_config_str))
        except SystemExit:
            # run_test_case calls os._exit for CUDA errors, this shouldn't reach here
            # but if it does via sys.exit, let it propagate
            raise
        except Exception as e:
            result_queue.put(("error", slot_index, api_config_str, str(e)))

    # Graceful exit
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
        self.result_queue = mp.Queue()
        self.slots: list[WorkerSlot] = []
        self._shutdown_event = threading.Event()
        self._watchdog_thread = None
        self._lock = threading.Lock()  # protects slot state modifications

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

    def _spawn_worker(self, slot):
        """Spawn a new worker process for the given slot."""
        slot.input_queue = mp.Queue()
        p = mp.Process(
            target=_worker_loop,
            args=(slot.index, slot.gpu_id, slot.input_queue, self.result_queue, self.options),
            daemon=True,
        )
        p.start()
        slot.process = p
        slot.state = "starting"
        slot.current_task = None
        slot.task_start_time = None

    def warmup(self, timeout=180):
        """Wait for all workers to report ready."""
        ready_count = 0
        deadline = time.time() + timeout
        failed_slots = []

        while ready_count < self.total_workers and time.time() < deadline:
            try:
                remaining = max(0.1, deadline - time.time())
                msg = self.result_queue.get(timeout=min(5.0, remaining))
                msg_type = msg[0]
                if msg_type == "ready":
                    slot_idx = msg[1]
                    with self._lock:
                        self.slots[slot_idx].state = "idle"
                    ready_count += 1
                elif msg_type == "init_failed":
                    slot_idx = msg[1]
                    error_msg = msg[2]
                    failed_slots.append((slot_idx, error_msg))
                    ready_count += 1  # count as handled
                    print(
                        f"{datetime.now()} Worker slot {slot_idx} init failed: {error_msg}",
                        flush=True,
                    )
            except queue.Empty:
                # Check for dead processes
                for slot in self.slots:
                    if slot.state == "starting" and slot.process and not slot.process.is_alive():
                        print(
                            f"{datetime.now()} Worker slot {slot.index} died during init (exit={slot.process.exitcode})",
                            flush=True,
                        )
                        self._spawn_worker(slot)

        if ready_count < self.total_workers:
            print(
                f"{datetime.now()} WARNING: Only {ready_count}/{self.total_workers} workers ready after {timeout}s",
                flush=True,
            )

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

    def _watchdog_loop(self):
        """Periodically check for timeouts and unexpectedly dead workers."""
        while not self._shutdown_event.is_set():
            time.sleep(1.0)
            now = time.time()

            for slot in self.slots:
                with self._lock:
                    # Check timeout
                    if (
                        slot.state == "busy"
                        and slot.task_start_time is not None
                        and now - slot.task_start_time > self.options.timeout
                    ):
                        self._handle_timeout(slot)
                        continue

                    # Check unexpected death
                    if (
                        slot.state in ("busy", "idle")
                        and slot.process is not None
                        and not slot.process.is_alive()
                    ):
                        self._handle_crash(slot)

    def _handle_timeout(self, slot):
        """Kill timed-out worker and enqueue timeout result."""
        config = slot.current_task
        print(
            f"{datetime.now()} Watchdog: slot {slot.index} timeout, killing PID {slot.process.pid}",
            flush=True,
        )
        self._kill_process(slot.process)
        self.result_queue.put(("timeout", slot.index, config))
        self._spawn_worker(slot)

    def _handle_crash(self, slot):
        """Handle unexpectedly dead worker."""
        exitcode = slot.process.exitcode if slot.process else None
        config = slot.current_task
        print(
            f"{datetime.now()} Watchdog: slot {slot.index} died (exit={exitcode})",
            flush=True,
        )
        if config is not None:
            self.result_queue.put(("crashed", slot.index, config, exitcode))
        self._spawn_worker(slot)

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
        """Stop all workers."""
        self._shutdown_event.set()

        if not force:
            # Graceful: send poison pills
            for slot in self.slots:
                if slot.input_queue is not None:
                    try:
                        slot.input_queue.put(None)
                    except (OSError, EOFError):
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
                if slot.process is not None:
                    self._sigkill_process(slot.process)
            for slot in self.slots:
                if slot.process is not None:
                    self._join_process(slot.process, timeout=3)

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=3)


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


def validate_gpu_options(options) -> tuple:
    """Validate and normalize GPU-related options."""
    device_count = get_device_count()
    if device_count == 0:
        raise ValueError("No devices found")
    if options.gpu_ids:
        try:
            gpu_ids = []
            for part in options.gpu_ids.split(","):
                part = part.strip()
                if not part:
                    continue
                if part.startswith("-") and part[1:].isdigit():
                    gpu_ids.append(int(part))
                elif "-" in part and not part.startswith("-"):
                    start, end = map(int, part.split("-"))
                    if start > end:
                        raise ValueError(f"Invalid range: {part} (start > end)")
                    gpu_ids.extend(range(start, end + 1))
                else:
                    gpu_ids.append(int(part))
        except ValueError:
            raise ValueError(
                f"Invalid gpu_ids: {options.gpu_ids} (int or range expected)"
            ) from None
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"Invalid gpu_ids: {options.gpu_ids} (duplicates)")
        gpu_ids = sorted(set(gpu_ids))
        if len(gpu_ids) > 1 and -1 in gpu_ids:
            raise ValueError(f"Invalid gpu_ids: {options.gpu_ids} (-1 allowed only)")
        if gpu_ids != [-1] and not all(0 <= id < device_count for id in gpu_ids):
            raise ValueError(
                f"Invalid gpu_ids: {options.gpu_ids} (valid range [0, {device_count}))"
            )
    else:
        gpu_ids = [-1]
    if options.num_gpus < -1 or options.num_gpus == 0 or options.num_gpus > device_count:
        raise ValueError(f"Invalid num_gpus: {options.num_gpus}")
    if options.num_gpus == -1:
        options.num_gpus = device_count if gpu_ids == [-1] else len(gpu_ids)
    if gpu_ids == [-1]:
        gpu_ids = list(range(options.num_gpus))
    elif len(gpu_ids) != options.num_gpus:
        raise ValueError(f"num_gpus {options.num_gpus} mismatches gpu_ids {gpu_ids}")
    if options.num_workers_per_gpu < -1 or options.num_workers_per_gpu == 0:
        raise ValueError(f"Invalid num_workers_per_gpu: {options.num_workers_per_gpu}")
    if options.required_memory <= 0:
        raise ValueError(f"Invalid required_memory: {options.required_memory}")
    return tuple(gpu_ids)


def parse_bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        elif value in ["false", "0", "no", "n"]:
            return False
    else:
        raise ValueError(f"Invalid boolean value: {value} parsed from command line")


def check_gpu_memory(gpu_ids, num_workers_per_gpu, required_memory):  # required_memory in GB
    assert isinstance(gpu_ids, tuple) and len(gpu_ids) > 0
    available_gpus = []
    max_workers_per_gpu = {}

    for gpu_id in gpu_ids:
        try:
            total_memory, used_memory = get_memory_info(gpu_id)
            free_memory = total_memory - used_memory
            max_workers = int(free_memory // required_memory)
            if max_workers >= 1:
                available_gpus.append(gpu_id)
                max_workers_per_gpu[gpu_id] = (
                    max_workers
                    if num_workers_per_gpu == -1
                    else min(max_workers, num_workers_per_gpu)
                )
        except pynvml.NVMLError as e:
            print(f"[WARNING] Failed to check GPU {gpu_id}: {e!s}", flush=True)
            continue

    return available_gpus, max_workers_per_gpu


def run_test_case(api_config_str, options):
    """Run a single test case for the given API configuration."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_id = int(cuda_visible.split(",")[0])

    write_to_log("checkpoint", api_config_str)
    print(
        f"{datetime.now()} GPU {gpu_id} {os.getpid()} [paddle {options.paddle_version}] test begin: {api_config_str}",
        flush=True,
    )

    max_memory_wait = 30  # max 30 iterations × 10s = 5 minutes
    for wait_iter in range(max_memory_wait):
        total_memory, used_memory = get_memory_info(gpu_id)
        free_memory = total_memory - used_memory

        if free_memory >= options.required_memory:
            break

        if wait_iter % 6 == 0:  # log every ~60s (every 6th iteration)
            print(
                f"{datetime.now()} device {gpu_id} Free: {free_memory:.1f} GB, "
                f"Required: {options.required_memory:.1f} GB. "
                f"Waiting for available memory... (attempt {wait_iter + 1}/{max_memory_wait})",
                flush=True,
            )
        time.sleep(10)
    else:
        print(
            f"{datetime.now()} device {gpu_id} Memory wait timeout after {max_memory_wait * 10}s. "
            f"Proceeding anyway (free={free_memory:.1f} GB, required={options.required_memory:.1f} GB).",
            flush=True,
        )

    try:
        api_config = APIConfig(api_config_str)
    except Exception as err:
        print(f"[config parse error] {api_config_str} {err!s}", flush=True)
        return

    option_to_class = {
        "paddle_only": APITestPaddleOnly,
        "paddle_cinn": APITestCINNVSDygraph,
        "accuracy": APITestAccuracy,
        "paddle_gpu_performance": APITestPaddleGPUPerformance,
        "torch_gpu_performance": APITestTorchGPUPerformance,
        "paddle_torch_gpu_performance": APITestPaddleTorchGPUPerformance,
        "accuracy_stable": APITestAccuracyStable,
        "paddle_custom_device": APITestCustomDeviceVSCPU,
        "custom_device_vs_gpu": APITestPaddleDeviceVSGPU,
    }

    test_class = next(
        (cls for opt, cls in option_to_class.items() if getattr(options, opt, False)),
        APITestAccuracy,  # default fallback
    )
    kwargs = {k: v for k, v in vars(options).items() if k in VALID_TEST_ARGS}
    case = test_class(api_config, **kwargs)
    try:
        case.test()
    except Exception as err:
        # if fatal error happens, subprocess need to exit with non-zero status
        if "CUDA error" in str(err) or "memory corruption" in str(err):
            os._exit(99)
        if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
            os._exit(98)
        if "AssertionError" in str(err) or "Tensor-likes are not equal" in str(err):
            os._exit(1)
        # if not fatal error, subprocess will be alive and report error
        print(f"[test error] {api_config_str}: {err}", flush=True)
        raise
    finally:
        del test_class, api_config, case
        gc.collect()
        if not any(
            getattr(options, opt)
            for opt in (
                "paddle_gpu_performance",
                "torch_gpu_performance",
                "paddle_torch_gpu_performance",
            )
        ):
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()


def main():
    start_time = time.time()
    print(f"Main process id: {os.getpid()}")
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

    parser = argparse.ArgumentParser(description="API Test")
    parser.add_argument("--api_config_file", default="")
    parser.add_argument(
        "--api_config_file_pattern",
        default="",
        help="Pattern to match multiple config files (e.g., 'tester/api_config/api_config_support2torch_*.txt')",
    )
    parser.add_argument("--api_config", default="")
    parser.add_argument(
        "--paddle_only",
        type=parse_bool,
        default=False,
        help="test paddle api only to figure out whether the api is supported",
    )
    parser.add_argument(
        "--paddle_cinn",
        type=parse_bool,
        default=False,
        help="test paddle api in dynamic graph mode and cinn mode",
    )
    parser.add_argument(
        "--accuracy",
        type=parse_bool,
        default=False,
        help="test paddle api to corespoding torch api",
    )
    parser.add_argument(
        "--paddle_gpu_performance",
        type=parse_bool,
        default=False,
        help="test paddle api performance",
    )
    parser.add_argument(
        "--torch_gpu_performance",
        type=parse_bool,
        default=False,
        help="test torch api performance",
    )
    parser.add_argument(
        "--paddle_torch_gpu_performance",
        type=parse_bool,
        default=False,
        help="test paddle and torch api performance",
    )
    parser.add_argument(
        "--accuracy_stable",
        type=parse_bool,
        default=False,
        help="test paddle api to corespoding torch api steadily",
    )
    parser.add_argument(
        "--paddle_custom_device",
        type=parse_bool,
        default=False,
        help="test paddle api on custom device vs CPU",
    )
    parser.add_argument(
        "--test_amp",
        type=parse_bool,
        default=False,
        help="Whether to test in auto mixed precision (AMP) mode",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use, -1 to use all available",
    )
    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=1,
        help="Number of workers per GPU, -1 to maximize based on memory",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="GPU IDs to use ('-1' for all available). "
        "Accepts comma-separated values and/or ranges (e.g., '0-3,6,7')",
    )
    parser.add_argument(
        "--required_memory",
        type=float,
        default=10.0,
        help="Required memory per worker in GB",
    )
    parser.add_argument(
        "--test_cpu",
        type=parse_bool,
        default=False,
        help="Whether to test CPU mode",
    )
    parser.add_argument("--use_cached_numpy", type=bool, default=False)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Log directory",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for accuracy tests",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for accuracy tests",
    )
    parser.add_argument(
        "--test_tol",
        type=parse_bool,
        default=False,
        help="Whether to test tolerance range in accuracy mode",
    )
    parser.add_argument(
        "--test_backward",
        type=parse_bool,
        default=False,
        help="Whether to test backward in paddle_cinn mode",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout setting for a single test case, in seconds",
    )
    parser.add_argument(
        "--show_runtime_status",
        type=parse_bool,
        default=True,
        help="Whether to show the current test progress in real-time. If set to False, only failed cases will be output",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="The numpy random seed ",
    )
    parser.add_argument(
        "--custom_device_vs_gpu",
        type=parse_bool,
        default=False,
        help="test paddle api on custom device vs GPU",
    )
    parser.add_argument(
        "--custom_device_vs_gpu_mode",
        type=str,
        choices=["upload", "download"],
        default="upload",
        help="operation mode for custom_device_vs_gpu: 'upload' or 'download'",
    )
    parser.add_argument(
        "--bitwise_alignment",
        type=bool,
        default=False,
        help="Whether to using bitwise alignment when run accuracy test",
    )
    parser.add_argument(
        "--generate_failed_tests",
        type=parse_bool,
        default=False,
        help="Whether to generate reproducible test files for failed cases",
    )
    parser.add_argument(
        "--exit_on_error",
        type=parse_bool,
        default=False,
        help="Whether to exit the process when a paddle_error occurs.",
    )

    options = parser.parse_args()
    options.paddle_version = paddle_version
    print(f"Options: {vars(options)}", flush=True)
    print(f"PaddlePaddle version: {paddle_version}", flush=True)
    if options.random_seed != parser.get_default("random_seed"):
        np.random.seed(options.random_seed)

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
        print(
            "Specify only one test mode:"
            "--accuracy,"
            "--paddle_only,"
            "--paddle_cinn,"
            "--paddle_gpu_performance,"
            "--torch_gpu_performance,"
            "--paddle_torch_gpu_performance"
            "--accuracy_stable"
            "--paddle_custom_device"
            "--custom_device_vs_gpu",
            flush=True,
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
        print("--test_tol takes effect when --accuracy is True.", flush=True)
    if options.test_backward and not options.paddle_cinn:
        print("--test_backward takes effect when --paddle_cinn is True.", flush=True)
    os.environ["USE_CACHED_NUMPY"] = str(options.use_cached_numpy)
    if options.bitwise_alignment:
        options.atol = 0.0
        options.rtol = 0.0
    if options.log_dir:
        set_test_log_path(options.log_dir)

    if options.api_config:
        # Single config execution
        # Load custom ops from paddlefleet to register _run_custom_op operators
        try:
            import paddlefleet_ops
        except ImportError:
            pass
        try:
            import FusedQuantOps
        except ImportError:
            pass

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

        # set log_writer
        set_engineV2()

        options.api_config = options.api_config.strip()
        print(
            f"{datetime.now()} [paddle {paddle_version}] test begin: {options.api_config}",
            flush=True,
        )
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print(f"[config parse error] {options.api_config} {err!s}", flush=True)
            return

        option_to_class = {
            "paddle_only": APITestPaddleOnly,
            "paddle_cinn": APITestCINNVSDygraph,
            "accuracy": APITestAccuracy,
            "paddle_gpu_performance": APITestPaddleGPUPerformance,
            "torch_gpu_performance": APITestTorchGPUPerformance,
            "paddle_torch_gpu_performance": APITestPaddleTorchGPUPerformance,
            "accuracy_stable": APITestAccuracyStable,
            "paddle_custom_device": APITestCustomDeviceVSCPU,
            "custom_device_vs_gpu": APITestPaddleDeviceVSGPU,
        }

        test_class = next(
            (cls for opt, cls in option_to_class.items() if getattr(options, opt, False)),
            APITestAccuracy,  # default fallback
        )

        if options.test_cpu:
            import paddle

            paddle.device.set_device("cpu")
        if options.custom_device_vs_gpu:
            # custom_device_vs_gpu 模式需要传递额外参数
            case = test_class(
                api_config,
                operation_mode=options.operation_mode,
                bos_path=options.bos_path,
                bos_conf_path=options.bos_conf_path,
                bcecmd_path=options.bcecmd_path,
                random_seed=options.random_seed,
                atol=options.atol,
                rtol=options.rtol,
            )
        elif options.accuracy:
            case = test_class(
                api_config,
                test_amp=options.test_amp,
                atol=options.atol,
                rtol=options.rtol,
                test_tol=options.test_tol,
                bitwise_alignment=options.bitwise_alignment,
                exit_on_error=options.exit_on_error,
            )
        else:
            case = test_class(api_config, test_amp=options.test_amp)
        try:
            case.test()
        except Exception as err:
            if (
                "Tensor-likes are not equal" in str(err)
                or "Mismatched elements" in str(err)
                or "Tensor-likes are not equal" in str(err)
                or "Error Message Summary" in str(err)
            ):
                exit(1)
            print(f"[test error] {options.api_config}: {err}", flush=True)
        finally:
            case.clear_tensor()
            del case
    elif options.api_config_file or options.api_config_file_pattern:
        # validate GPU options
        gpu_ids = validate_gpu_options(options)

        # get config files
        if options.api_config_file_pattern:
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

        # when engineV2 was interrupted, resume from .tmp dir
        aggregate_logs()

        # read checkpoint
        finish_configs = read_log("checkpoint")
        print(len(finish_configs), "cases in checkpoint.", flush=True)

        api_config_count = 0
        api_configs = set()
        for config_file in config_files:
            try:
                with open(config_file) as f:
                    lines = [line.strip() for line in f if line.strip()]
                    api_config_count += len(lines)
                    api_configs.update(lines)
            except Exception as e:
                print(f"Failed to read config file {config_file}: {e}", flush=True)
                return
        print(api_config_count, "cases in total.", flush=True)
        dup_case = api_config_count - len(api_configs)
        if dup_case > 0:
            print(dup_case, "cases are duplicates and removed.", flush=True)

        api_config_count = len(api_configs)
        api_configs = sorted(api_configs - finish_configs)
        all_case = len(api_configs)
        finish_case = api_config_count - all_case
        if finish_case:
            print(finish_case, "cases already tested.", flush=True)
        print(all_case, "cases will be tested.", flush=True)
        del api_config_count, dup_case, finish_case

        # validate GPU memory
        available_gpus, max_workers_per_gpu = check_gpu_memory(
            gpu_ids, options.num_workers_per_gpu, options.required_memory
        )
        if not available_gpus:
            print(
                f"No GPUs with sufficient memory available. Current memory constraint is {options.required_memory} GB.",
                flush=True,
            )
            return

        total_workers = sum(max_workers_per_gpu.values())
        print(
            f"Using {len(available_gpus)} GPU(s) with max workers per GPU: {max_workers_per_gpu}. Total workers: {total_workers}.",
            flush=True,
        )

        if options.test_cpu:
            print(f"Using {cpu_count()} CPU(s) for paddle in CPU mode.", flush=True)

        # set log_writer
        if options.log_dir:
            set_test_log_path(options.log_dir)
        set_engineV2()

        # initialize worker pool (per-worker queue architecture)
        pool = WorkerPool(available_gpus, max_workers_per_gpu, options)

        def cleanup_handler(*args):
            cleanup(pool)
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

        print(f"{datetime.now()} Starting {total_workers} workers...", flush=True)
        pool.start()
        pool.warmup(timeout=180)
        print(f"{datetime.now()} All workers ready.", flush=True)

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

                # Task completed (done/error/timeout/crashed)
                slot_idx = msg[1]
                config = msg[2]
                if msg_type in ("done", "error"):
                    pool.mark_idle(slot_idx)
                active_tasks -= 1
                tested_case += 1

                if options.show_runtime_status or tested_case % 10000 == 0:
                    print(
                        f"[{tested_case}/{all_case}] Testing {config}",
                        flush=True,
                    )

                if msg_type == "done":
                    if options.show_runtime_status or tested_case % 10000 == 0:
                        print(f"[info] Test case succeeded for {config}", flush=True)
                elif msg_type == "timeout":
                    write_to_log("timeout", config)
                    print(
                        f"[error] Test case timed out for {config}",
                        flush=True,
                    )
                elif msg_type == "crashed":
                    exitcode = msg[3] if len(msg) > 3 else None
                    if exitcode == 99:
                        print(
                            f"[error] CUDA error for {config}",
                            flush=True,
                        )
                    elif exitcode == 98:
                        print(
                            f"[error] CUDA out of memory for {config}",
                            flush=True,
                        )
                    else:
                        write_to_log("crash", config)
                        print(
                            f"[fatal] Worker crashed for {config} (exit={exitcode})",
                            flush=True,
                        )
                elif msg_type == "error":
                    error_msg = msg[3] if len(msg) > 3 else ""
                    print(
                        f"[warn] Test case failed for {config}: {error_msg}",
                        flush=True,
                    )

                # Get next config to dispatch
                next_config = next(config_iter, None)
                if next_config is None:
                    continue

                # For timeout/crashed: worker is restarting, queue for later dispatch
                if msg_type in ("timeout", "crashed"):
                    pending_dispatch.append(next_config)
                else:
                    # Worker is alive and ready for next task
                    pool.dispatch(slot_idx, next_config)
                    active_tasks += 1

                # Periodic log aggregation
                if tested_case % 1000 == 0:
                    aggregate_logs()

            aggregate_logs()
            pool.shutdown()
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)
            cleanup(pool)
            total_time = time.time() - start_time
            print(f"Test time: {round(total_time / 60, 3)} minutes.", flush=True)
        finally:
            print(f"{tested_case} cases have been tested.", flush=True)
            log_counts = aggregate_logs(end=True)
            print_log_info(all_case, log_counts)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Test time: {round(total_time / 60, 3)} minutes.", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
