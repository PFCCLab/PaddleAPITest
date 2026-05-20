"""HTTP server for remote API test execution.

Runs on the remote machine. Accepts API config strings via HTTP POST,
executes them on the local device (GPU/XPU/etc.) using a multi-GPU
process pool, and returns the pdtensor result bytes.

Usage:
    python -m tester.http_server --host 0.0.0.0 --port 8089 --num_gpus=-1
    python -m tester.http_server --host 0.0.0.0 --port 8089 --gpu_ids=6,7
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import secrets
import signal
import sys
import threading
import time
from concurrent.futures import TimeoutError
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Lock, Manager, set_start_method
from pathlib import Path

import numpy as np
from pebble import ProcessExpired, ProcessPool

# Add project root to path so we can import engineV2 utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engineV2 import (
    check_gpu_memory,
    detect_device_type,
    get_memory_info,
    validate_gpu_options,
)
from tester.api_config.log_writer import set_engineV2

# Global process pool, set during server startup
_pool = None
_server_timeout = 1800  # default timeout per task
_concurrency_semaphore = None  # limits queued + in-flight requests

# Admin interface: set via --admin_token; empty string means disabled
_admin_token: str = ""
REPO_ROOT = Path(__file__).resolve().parent.parent


def init_server_worker(gpu_worker_list, lock, available_gpus, max_workers_per_gpu):
    """Initialize a worker process with GPU assignment.

    Similar to engineV2.init_worker_gpu but without test-specific options.
    """
    import errno

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

        # Import paddle/torch AFTER setting CUDA_VISIBLE_DEVICES so that
        # the CUDA context is created on the correct device, not GPU 0.
        import paddle
        import torch

        globals()["torch"] = torch
        globals()["paddle"] = paddle

        from tester import APIConfig, APITestPaddleDeviceVSGPU
        from tester.paddle_device_vs_gpu import _PaddleSkipError

        globals()["APIConfig"] = APIConfig
        globals()["APITestPaddleDeviceVSGPU"] = APITestPaddleDeviceVSGPU
        globals()["_PaddleSkipError"] = _PaddleSkipError

        print(
            f"{datetime.now()} Server worker PID: {my_pid}, Assigned GPU ID: {assigned_gpu}",
            flush=True,
        )
    except Exception as e:
        print(f"{datetime.now()} Server worker {my_pid} init failed: {e}", flush=True)
        raise


def run_single_api(api_config_str, random_seed, test_amp=False):
    """Execute a single API test and return the pdtensor bytes.

    Runs inside a worker process. Returns bytes on success, raises on failure.
    """
    from engineV2 import detect_device_type

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_id = int(cuda_visible.split(",")[0])

    print(
        f"{datetime.now()} GPU {gpu_id} {os.getpid()} server exec: {api_config_str}",
        flush=True,
    )

    # Wait for sufficient GPU memory
    for _ in range(60):  # max ~60 minutes wait
        total_memory, used_memory = get_memory_info(gpu_id)
        free_memory = total_memory - used_memory
        if free_memory >= 2.0:  # minimal 2GB required
            break
        time.sleep(60)
    else:
        raise RuntimeError(f"GPU {gpu_id} insufficient memory after waiting")

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    paddle.seed(random_seed)

    api_config = APIConfig(api_config_str)

    # Create a minimal instance to reuse _run_paddle
    tester = APITestPaddleDeviceVSGPU(
        api_config,
        operation_mode="upload",  # dummy, we won't call test()
        random_seed=random_seed,
        test_amp=test_amp,
    )

    device_type = detect_device_type()
    try:
        output, grads = tester._run_paddle(device_type)
    except _PaddleSkipError as e:
        # Re-raise as a tagged RuntimeError so pebble can serialize it across
        # the process boundary and the handler can detect it.
        raise RuntimeError(f"__SKIP__:{e}") from None

    if output is None:
        real_error = getattr(tester, "_last_error", None)
        raise RuntimeError(real_error if real_error else f"API execution returned None for {api_config_str}")

    # Normalize output: paddle.save does not support named tuples (e.g.
    # CummaxRetType, TopKRetType). Convert them to plain tuple recursively.
    # Also eagerly evaluate lazy Jacobian/Hessian objects so that paddle.save
    # can serialize the result as a plain Tensor.
    def _normalize(obj):
        if isinstance(obj, paddle.Tensor):
            return obj
        # Evaluate lazy Jacobian/Hessian objects (Hessian inherits Jacobian)
        try:
            from paddle.autograd.autograd import Jacobian
            if isinstance(obj, Jacobian):
                return obj[:]  # triggers full evaluation, returns Tensor
        except ImportError:
            pass
        if isinstance(obj, (list, tuple)):
            items = [_normalize(x) for x in obj]
            return items if isinstance(obj, list) else tuple(items)
        return obj

    output = _normalize(output)

    # Serialize to pdtensor bytes
    save_data = {"output": output}
    if grads is not None:
        save_data["grads"] = grads

    buffer = io.BytesIO()
    paddle.save(save_data, buffer)
    result_bytes = buffer.getvalue()

    # Cleanup
    tester.clear_tensor()
    del tester, api_config, output, grads, save_data
    gc.collect()
    torch = globals().get("torch")
    if torch is not None:
        torch.cuda.empty_cache()
    paddle.device.cuda.empty_cache()

    print(
        f"{datetime.now()} GPU {gpu_id} {os.getpid()} server done: {api_config_str} ({len(result_bytes)} bytes)",
        flush=True,
    )
    return result_bytes


class APITestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for API test execution."""

    def log_message(self, format, *args):
        """Override to use flush for immediate output."""
        print(f"{datetime.now()} {self.client_address[0]} - {format % args}", flush=True)

    def _send_json_response(self, status_code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes_response(self, data, device_type="unknown"):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Device-Type", device_type)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            device_type = detect_device_type()
            self._send_json_response(
                200,
                {
                    "status": "ok",
                    "device_type": device_type,
                    "paddle_version": __import__("paddle").__version__,
                },
            )
        else:
            self._send_json_response(404, {"error": "not_found"})

    def do_POST(self):
        if self.path == "/run_api_test":
            self._handle_run_api_test()
        elif self.path == "/admin/upload_file":
            self._handle_admin_upload_file()
        elif self.path == "/admin/delete_file":
            self._handle_admin_delete_file()
        elif self.path == "/admin/restart":
            self._handle_admin_restart()
        else:
            self._send_json_response(404, {"error": "not_found"})

    def _check_admin_token(self) -> bool:
        if not _admin_token:
            self._send_json_response(503, {"error": "admin_disabled"})
            return False
        provided = self.headers.get("X-Admin-Token", "")
        if not secrets.compare_digest(provided, _admin_token):
            self._send_json_response(403, {"error": "forbidden"})
            return False
        return True

    def _handle_run_api_test(self):
        # Parse request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json_response(
                400,
                {
                    "error": "bad_request",
                    "detail": f"Invalid JSON: {e}",
                },
            )
            return

        api_config_str = data.get("api_config")
        random_seed = data.get("random_seed", 0)
        test_amp = data.get("test_amp", False)

        if not api_config_str:
            self._send_json_response(
                400,
                {
                    "error": "bad_request",
                    "detail": "Missing field: api_config",
                },
            )
            return

        # Submit to process pool
        global _pool, _server_timeout, _concurrency_semaphore
        if _pool is None:
            self._send_json_response(
                503,
                {
                    "error": "service_unavailable",
                    "detail": "Process pool not initialized",
                    "api_config": api_config_str,
                },
            )
            return

        # Block until a slot opens. The client's http_timeout is the ultimate
        # backstop — if we wait too long here the client will close the
        # connection and this thread will get a BrokenPipeError on write,
        # which is caught by the outer except.
        if _concurrency_semaphore is not None:
            _concurrency_semaphore.acquire()

        try:
            future = _pool.schedule(
                run_single_api,
                [api_config_str, random_seed, test_amp],
                timeout=_server_timeout,
            )
            result_bytes = future.result()
            device_type = detect_device_type()
            self._send_bytes_response(result_bytes, device_type)

        except TimeoutError:
            self._send_json_response(
                504,
                {
                    "error": "timeout",
                    "detail": f"Execution timed out after {_server_timeout}s",
                    "api_config": api_config_str,
                },
            )

        except ProcessExpired as e:
            if e.exitcode == 99:
                error_type = "cuda_error"
            elif e.exitcode == 98:
                error_type = "oom"
            else:
                error_type = "crash"
            self._send_json_response(
                500,
                {
                    "error": error_type,
                    "detail": str(e),
                    "api_config": api_config_str,
                },
            )

        except Exception as e:
            detail = str(e)
            if detail.startswith("__SKIP__:"):
                self._send_json_response(
                    422,
                    {
                        "error": "skip",
                        "detail": detail[len("__SKIP__:") :],
                        "api_config": api_config_str,
                    },
                )
            else:
                self._send_json_response(
                    500,
                    {
                        "error": "remote_error",
                        "detail": detail,
                        "api_config": api_config_str,
                    },
                )

        finally:
            if _concurrency_semaphore is not None:
                _concurrency_semaphore.release()

    def _handle_admin_upload_file(self):
        if not self._check_admin_token():
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            rel_path = data["path"]
            content = data["content"]
        except Exception as e:
            self._send_json_response(400, {"error": "bad_request", "detail": str(e)})
            return
        # Security: reject path traversal
        target = (REPO_ROOT / rel_path).resolve()
        if not str(target).startswith(str(REPO_ROOT)):
            self._send_json_response(403, {"error": "forbidden", "detail": "path traversal"})
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self._send_json_response(200, {"status": "ok", "path": rel_path})
        except Exception as e:
            self._send_json_response(500, {"status": "error", "detail": str(e)})

    def _handle_admin_delete_file(self):
        if not self._check_admin_token():
            return
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            rel_path = data["path"]
        except Exception as e:
            self._send_json_response(400, {"error": "bad_request", "detail": str(e)})
            return
        # Security: reject path traversal
        target = (REPO_ROOT / rel_path).resolve()
        if not str(target).startswith(str(REPO_ROOT)):
            self._send_json_response(403, {"error": "forbidden", "detail": "path traversal"})
            return
        try:
            if target.exists():
                target.unlink()
                # Also remove the corresponding .pyc from __pycache__ if it exists
                pyc = target.parent / "__pycache__" / (target.stem + ".cpython-310.pyc")
                if pyc.exists():
                    pyc.unlink()
            self._send_json_response(200, {"status": "ok", "path": rel_path})
        except Exception as e:
            self._send_json_response(500, {"status": "error", "detail": str(e)})

    def _handle_admin_restart(self):
        if not self._check_admin_token():
            return
        self._send_json_response(200, {"status": "restarting"})

        def do_restart():
            time.sleep(0.5)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        threading.Thread(target=do_restart, daemon=True).start()


def parse_bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        elif value in ["false", "0", "no", "n"]:
            return False
    raise ValueError(f"Invalid boolean value: {value}")


def main():
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="HTTP server for remote API test execution")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8089, help="Server port")
    parser.add_argument(
        "--num_gpus", type=int, default=-1, help="Number of GPUs to use, -1 for all"
    )
    parser.add_argument("--num_workers_per_gpu", type=int, default=1, help="Workers per GPU")
    parser.add_argument(
        "--required_memory", type=float, default=10.0, help="Required memory per worker in GB"
    )
    parser.add_argument("--gpu_ids", type=str, default="", help="GPU IDs (e.g., '0,1,2' or '0-3')")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-task timeout in seconds")
    parser.add_argument(
        "--admin_token",
        type=str,
        default="",
        help="If non-empty, enables /admin/upload_file and /admin/restart endpoints "
             "protected by X-Admin-Token header",
    )

    args = parser.parse_args()

    global _server_timeout, _admin_token
    _server_timeout = args.timeout
    _admin_token = args.admin_token

    # Detect device
    device_type = detect_device_type()
    print(f"Detected device type: {device_type}", flush=True)

    # Build a minimal options namespace for validate_gpu_options
    gpu_options = argparse.Namespace(
        num_gpus=args.num_gpus,
        num_workers_per_gpu=args.num_workers_per_gpu,
        required_memory=args.required_memory,
        gpu_ids=args.gpu_ids,
    )

    gpu_ids = validate_gpu_options(gpu_options)
    available_gpus, max_workers_per_gpu = check_gpu_memory(
        gpu_ids, gpu_options.num_workers_per_gpu, gpu_options.required_memory
    )

    if not available_gpus:
        print(
            f"No GPUs with sufficient memory. Required: {gpu_options.required_memory} GB.",
            flush=True,
        )
        sys.exit(1)

    total_workers = sum(max_workers_per_gpu.values())
    print(
        f"Using {len(available_gpus)} GPU(s) with workers: {max_workers_per_gpu}. "
        f"Total workers: {total_workers}.",
        flush=True,
    )

    # Limit concurrent requests to 2x workers — enough to keep the pool saturated
    # without unbounded queuing. Excess requests get 503 immediately.
    global _concurrency_semaphore
    _concurrency_semaphore = threading.Semaphore(total_workers * 2)

    # Initialize process pool
    manager = Manager()
    gpu_worker_list = manager.dict({gid: manager.list() for gid in available_gpus})
    lock = Lock()

    global _pool
    _pool = ProcessPool(
        max_workers=total_workers,
        initializer=init_server_worker,
        initargs=[gpu_worker_list, lock, available_gpus, max_workers_per_gpu],
    )

    # Start HTTP server
    server = ThreadingHTTPServer((args.host, args.port), APITestHandler)

    def shutdown_handler(*_):
        print(f"\n{datetime.now()} Shutting down...", flush=True)
        if _pool is not None:
            try:
                if _pool.active:
                    _pool.stop()
                    _pool.join(timeout=5)
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(
        f"{datetime.now()} HTTP server listening on {args.host}:{args.port} "
        f"(device: {device_type}, workers: {total_workers})",
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown_handler()


if __name__ == "__main__":
    main()
