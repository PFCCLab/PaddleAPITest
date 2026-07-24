from __future__ import annotations

import argparse
import errno
import gc
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import TimeoutError, as_completed
from datetime import datetime
from multiprocessing import Lock, Manager, cpu_count, set_start_method
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pynvml
import yaml
from pebble import ProcessExpired, ProcessPool

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
from tester.runtime_config import (
    GPU_MEMORY_POLICY_ENV,
    TestRuntimeConfig,
    resolve_gpu_memory_policy,
    runtime_config_for_gpu,
)

os.environ["FLAGS_use_system_allocator"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

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

DEVICE_TYPE = None
DEVICE_TYPE_DETECTED = False
DEVICE_COUNT = None  # total number of devices
_MEM_SNAPSHOT = None  # dict: gpu_id -> (total_gb, used_gb)
_MEM_SNAPSHOT_TS = 0.0
_MEM_SNAPSHOT_TTL = 2.0  # seconds — snapshot cache ttl
FATAL_CUDA_EXIT_CODE = 99
FATAL_OOM_EXIT_CODE = 98
FATAL_TORCH_EXIT_CODE = 97
MEMORY_WAIT_SECONDS = 10
MEMORY_WAIT_LOG_INTERVAL = 60


class GpuMemoryDeferred(Exception):
    """Raised when a GPU-mode case should wait for more free memory."""


def cleanup(pool):
    print(f"\n{datetime.now()} Cleanup started", flush=True)
    if pool is not None:
        try:
            if pool.active:
                pool.stop()
                pool.join(timeout=5)
        except Exception as e:
            print(f"{datetime.now()} Error shutting down executor: {e}", flush=True)
    print(f"{datetime.now()} Cleanup completed", flush=True)


def estimate_timeout(api_config) -> float:
    """Estimate timeout based on tensor size in APIConfig."""
    # TIMEOUT_STEPS = (
    #     (1e4, 10),
    #     (1e5, 30),
    #     (1e6, 90),
    #     (1e7, 300),
    #     (1e8, 1800),
    #     (float("inf"), 3600),
    # )
    # try:
    #     api_config = APIConfig(api_config)
    #     first = None
    #     if api_config.args:
    #         first = api_config.args[0]
    #     elif api_config.kwargs:
    #         first = next(iter(api_config.kwargs.values()))
    #     if first is not None and hasattr(first, "shape"):
    #         total_elements = math.prod(first.shape)
    #         for threshold, timeout in TIMEOUT_STEPS:
    #             if total_elements <= threshold:
    #                 return timeout
    # except Exception:
    #     pass
    # return TIMEOUT_STEPS[-1][1]
    return 1800


def detect_device_type() -> str:
    global DEVICE_TYPE, DEVICE_TYPE_DETECTED
    if DEVICE_TYPE_DETECTED:
        return DEVICE_TYPE

    # 优先尝试 NVML（NVIDIA GPU）
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
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
    global DEVICE_COUNT
    if DEVICE_COUNT is not None:
        return DEVICE_COUNT

    device_type = detect_device_type()

    if device_type == "gpu":
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
        finally:
            pynvml.nvmlShutdown()
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
    device_type = detect_device_type()

    if device_type == "gpu":
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(mem_info.total) / (1024**3), int(mem_info.used) / (1024**3)
        finally:
            pynvml.nvmlShutdown()

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


def init_worker_gpu(gpu_worker_list, lock, available_gpus, max_workers_per_gpu, options):
    init_log(options.log_dir, worker_tmp_logs=True)
    my_pid = os.getpid()

    def pid_exists(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError as e:
            return e.errno == errno.EPERM

    try:
        with lock:
            assigned_gpu = -1
            max_available_slots = -1
            for gpu_id in available_gpus:
                workers = gpu_worker_list[gpu_id]
                workers[:] = [pid for pid in workers if pid_exists(pid)]
                available_slots = max_workers_per_gpu[gpu_id] - len(workers)
                if available_slots > max_available_slots:
                    max_available_slots = available_slots
                    assigned_gpu = gpu_id

            if assigned_gpu == -1:
                raise RuntimeError(f"Worker {my_pid} could not be assigned a GPU.")

            gpu_worker_list[assigned_gpu].append(my_pid)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
        with suppress_startup_output():
            import paddle

            globals()["paddle"] = paddle
            if options.test_cpu:
                paddle.device.set_device("cpu")
            elif not getattr(options, "paddle_custom_device", False):
                # CUDA_VISIBLE_DEVICES assigns the worker slot; Paddle still needs an explicit device.
                paddle.set_device("gpu")
            try:
                import paddlefleet_ops
            except ImportError:
                pass
            try:
                import FusedQuantOps
            except ImportError:
                pass
            globals().update(_load_test_classes(options))

            def signal_handler(*args):
                _clear_device_cache(options)
                restore_stdio()
                close_process_files()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            redirect_stdio()

    except Exception as e:
        print(f"[worker] INIT_FAILED | PID {my_pid} | {e}", flush=True)
        raise


def run_test_case(api_config_str, options):
    """Run a single test case for the given API configuration."""
    completion = [os.getpid(), None]
    started_at = time.monotonic()
    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    case_id = write_case_begin(
        api_config_str,
        worker_pid=os.getpid(),
        gpu=gpu_id,
        paddle_version=options.paddle_version,
    )
    runtime_config = runtime_config_for_gpu(options, gpu_id)
    case_status = "done"
    try:
        if options.show_runtime_status:
            total_memory, used_memory_before = get_memory_info(gpu_id)
            print(
                f"{datetime.now()} GPU {gpu_id} memory before: used={used_memory_before:.1f} GB, "
                f"free={total_memory - used_memory_before:.1f} GB",
                flush=True,
            )

        api_config = None
        case = None
        try:
            api_config = APIConfig(api_config_str)
        except Exception as err:
            print(f"[config_parse] {api_config_str} {err!s}", flush=True)
            write_terminal_log("config_parse", api_config_str)
            case_status = "error"
            return completion

        test_class = _select_test_class(options)
        kwargs = {k: v for k, v in vars(options).items() if k in VALID_TEST_ARGS}
        kwargs["runtime_config"] = runtime_config
        case = test_class(api_config, **kwargs)
        try:
            if dump_enabled():
                case.run_with_dump()
            else:
                case.test()
            if has_terminal_log(api_config_str):
                write_checkpoint(api_config_str)
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
                return completion
            # if not fatal error, subprocess will be alive and report error
            print(f"[error] {api_config_str}: {err}", flush=True)
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
            if options.show_runtime_status:
                try:
                    total_memory, used_memory_after = get_memory_info(gpu_id)
                    print(
                        f"{datetime.now()} GPU {gpu_id} memory after cleanup: used={used_memory_after:.1f} GB, "
                        f"free={total_memory - used_memory_after:.1f} GB",
                        flush=True,
                    )
                except Exception as err:
                    print(
                        f"{datetime.now()} Failed to read GPU {gpu_id} memory after cleanup: {err}",
                        flush=True,
                    )

        return completion
    except GpuMemoryDeferred:
        case_status = "deferred"
        raise
    except BaseException:
        case_status = "error"
        raise
    finally:
        completion[1] = write_case_end(
            case_status,
            case_id=case_id,
            api_config_str=api_config_str,
            duration_ms=round((time.monotonic() - started_at) * 1000),
        )


def main():
    start_time = time.time()
    set_start_method("spawn")

    try:
        import paddle as _paddle

        paddle_version = _paddle.__version__
        del _paddle
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

    if options.api_config:
        try:
            _prepare_single_config_gpu(options)
        except ValueError as err:
            _print_argument(ARGUMENT_ERROR_PREFIX, str(err))
            return

        # Single config execution
        import paddle

        globals()["paddle"] = paddle
        if options.test_cpu:
            paddle.device.set_device("cpu")
        elif not getattr(options, "paddle_custom_device", False):
            # CUDA_VISIBLE_DEVICES assigns the worker slot; Paddle still needs an explicit device.
            paddle.set_device("gpu")

        # Load custom ops from paddlefleet to register _run_custom_op operators
        try:
            import paddlefleet_ops
        except ImportError:
            pass
        try:
            import FusedQuantOps
        except ImportError:
            pass

        globals().update(_load_test_classes(options))

        init_log(options.log_dir, worker_tmp_logs=True)

        options.api_config = options.api_config.strip()
        single_case_error = None
        try:
            run_test_case(options.api_config, options)
        except Exception as err:
            single_case_error = err
            print(f"[error] {options.api_config}: {err}", flush=True)
        finally:
            close_process_files()
            log_counts = aggregate_logs(end=True)
            completed_case = (log_counts or {}).get("checkpoint", 0)
            remaining_case = max(1 - completed_case, 0)
            print_log_info(remaining_case, log_counts)
            print_run_footer(
                1,
                completed_case,
                remaining_case,
                log_counts,
                time.time() - start_time,
                options.log_dir,
            )
        if single_case_error is not None:
            sys.exit(1)
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

        total_workers = sum(max_workers_per_gpu.values())
        gpu_total_memory_map = {}
        for gpu_id in available_gpus:
            try:
                gpu_total_memory_map[gpu_id] = get_memory_info(gpu_id)[0]
            except Exception:
                pass
        options.gpu_workers_per_gpu_map = dict(max_workers_per_gpu)
        options.gpu_total_memory_map = gpu_total_memory_map
        options.runtime_config = TestRuntimeConfig.from_options(options)
        print_compute_summary(available_gpus, max_workers_per_gpu)

        if options.test_cpu:
            print(f"CPU: {cpu_count()} available | Paddle CPU mode", flush=True)

        print_running_header()

        # initialize process pool
        manager = Manager()
        gpu_worker_list = manager.dict({gpu_id: manager.list() for gpu_id in available_gpus})
        lock = Lock()

        pool = ProcessPool(
            max_workers=total_workers,
            initializer=init_worker_gpu,
            initargs=[
                gpu_worker_list,
                lock,
                available_gpus,
                max_workers_per_gpu,
                options,
            ],
        )

        def cleanup_handler(*args):
            cleanup(pool)
            sys.exit(1)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

        # batch test
        tested_case = 0
        try:
            BATCH_SIZE = 20000
            for batch_start in range(0, len(api_configs), BATCH_SIZE):
                batch = api_configs[batch_start : batch_start + BATCH_SIZE]
                futures = {}

                def schedule_config(config):
                    # timeout = estimate_timeout(config)
                    timeout = options.timeout
                    future = pool.schedule(
                        run_test_case,
                        [config, options],
                        timeout=timeout,
                    )
                    futures[future] = config

                for config in batch:
                    schedule_config(config)

                while futures:
                    for future in list(as_completed(futures)):
                        config = futures.pop(future)
                        checkpoint_ready = True
                        worker_pid = None
                        progress_status = "DONE"
                        progress_detail = None
                        try:
                            worker_pid, completed_offset = future.result()
                            mark_inorder_case_complete(worker_pid, completed_offset)
                        except TimeoutError as err:
                            write_terminal_log("timeout", config)
                            progress_status = "TIMEOUT"
                            worker_pid = getattr(err, "pid", None)
                            if worker_pid is not None:
                                completed_offset = append_case_end_to_worker_log(
                                    worker_pid, "timeout", api_config_str=config
                                )
                                mark_inorder_case_complete(worker_pid, completed_offset)
                        except ProcessExpired as err:
                            worker_pid = getattr(err, "pid", None)
                            if err.exitcode in (-signal.SIGKILL, -signal.SIGTERM):
                                expired_status = "timeout"
                                checkpoint_ready = False
                                progress_status = "RETRY"
                                progress_detail = f"exit {err.exitcode}"
                            else:
                                expired_status, progress_status = classify_worker_exit(
                                    err.exitcode,
                                    FATAL_CUDA_EXIT_CODE,
                                    FATAL_OOM_EXIT_CODE,
                                    FATAL_TORCH_EXIT_CODE,
                                )
                                write_terminal_log(expired_status, config)
                                if progress_status == "PADDLE_CRASH":
                                    progress_detail = f"exit {err.exitcode}"
                            if worker_pid is not None:
                                completed_offset = append_case_end_to_worker_log(
                                    worker_pid, expired_status, api_config_str=config
                                )
                                mark_inorder_case_complete(worker_pid, completed_offset)
                        except GpuMemoryDeferred as err:
                            checkpoint_ready = False
                            progress_status = "DEFERRED"
                            progress_detail = str(err)
                            schedule_config(config)
                        except Exception as err:
                            write_terminal_log("config_parse", config)
                            checkpoint_ready = False
                            progress_status = "CONFIG_PARSE"
                            progress_detail = str(err)
                        if checkpoint_ready:
                            tested_case += 1
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
                        elif progress_status in ("RETRY", "DEFERRED", "CONFIG_PARSE"):
                            print_case_notice(progress_status, config, progress_detail)
                aggregate_logs()
            pool.close()
            pool.join()
        except Exception as e:
            print(f"Unexpected error: {e}", flush=True)
            cleanup(pool)
        finally:
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
