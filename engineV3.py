#!/bin/env python3
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
import zipfile
from datetime import datetime

os.environ["FLAGS_USE_SYSTEM_ALLOCATOR"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


##### v5 #####
# 增加缓冲队列 ，增加结束后打包功能
# python3.10 workerfarm.py   --api_config_file=error.txt   --gpus=0,1,2,3 --per_gpu_concurrency=1   --timeout=1800   --accuracy


class LoggerRedirector:
    """同时重定向 sys.stdout/sys.stderr 和 底层文件描述符的日志器"""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = None
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_file = open(self.log_file_path, "a")

        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)

        os.dup2(self.log_file.fileno(), self.stdout_fd)
        os.dup2(self.log_file.fileno(), self.stderr_fd)

        sys.stdout = os.fdopen(self.stdout_fd, "a", buffering=1)
        sys.stderr = os.fdopen(self.stderr_fd, "a", buffering=1)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.saved_stdout_fd:
                os.dup2(self.saved_stdout_fd, self.stdout_fd)
                os.close(self.saved_stdout_fd)
            if self.saved_stderr_fd:
                os.dup2(self.saved_stderr_fd, self.stderr_fd)
                os.close(self.saved_stderr_fd)

            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        except Exception as e:
            print(
                f"[LoggerRedirector] Error restoring stdout/stderr: {e}",
                file=sys.__stderr__,
            )

        if self.log_file:
            self.log_file.close()


def worker_process(gpu_id, task_queue, result_queue, idx, args_mode):
    log_path = os.path.join("worker_logs", f"worker_{idx}_log.txt")
    with LoggerRedirector(log_path):
        try:
            print(f"[Worker {idx}] Started on GPU {gpu_id}")
            test_class = None

            while True:
                task = task_queue.get()
                if task is None:
                    print(f"[Worker {idx}] Received stop signal.")
                    break

                task_id, api_config_str = task

                # 通知master真正开始执行
                result_queue.put((task_id, "start", idx))

                print(f"[Worker {idx}] Processing Task {task_id}: {api_config_str}")

                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                    if test_class is None:
                        from tester import (
                            APIConfig,
                            APITestAccuracy,
                            APITestCINNVSDygraph,
                            APITestPaddleOnly,
                        )

                        mode_to_test_class = {
                            "paddle_only": APITestPaddleOnly,
                            "paddle_cinn": APITestCINNVSDygraph,
                            "accuracy": APITestAccuracy,
                        }
                        if args_mode not in mode_to_test_class:
                            raise ValueError(f"Unknown mode: {args_mode}")
                        test_class = mode_to_test_class[args_mode]

                    import paddle
                    import torch

                    api_config = APIConfig(api_config_str.strip())
                    case = test_class(api_config)
                    case.test()
                    case.clear_tensor()

                    del case
                    del api_config

                    torch.cuda.empty_cache()
                    paddle.device.cuda.empty_cache()

                    result_queue.put((task_id, "success"))
                    print(f"[Worker {idx}] Completed Task {task_id}")

                except Exception as e:
                    print(f"[Worker {idx}] Error on Task {task_id}: {e}")
                    result_queue.put((task_id, "error", str(e)))
                    if (
                        "CUDA error" in str(e)
                        or "memory corruption" in str(e)
                        or "CUDA out of memory" in str(e)
                    ):
                        print(f"[Worker {idx}] OOM on Task {task_id}: {e}")
                        exit(1)

        except Exception as e:
            print(f"[Worker {idx}] Fatal Error: {e}")
            result_queue.put(("worker_init_error", "error", str(e)))


def spawn_worker(gpu_list, concurrency, idx, task_queue, result_queue, selected_mode):
    gpu_id = gpu_list[idx % len(gpu_list)]
    p = mp.Process(
        target=worker_process,
        args=(gpu_id, task_queue, result_queue, idx, selected_mode),
    )
    p.start()
    print(f"[Master] Spawned Worker {idx} on GPU {gpu_id} (PID={p.pid})")
    return p


def kill_worker(p):
    try:
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
            p.join(timeout=1)
    except Exception as e:
        print(f"[Master] Failed to kill worker {p.pid}: {e}")


def pack():
    pack_name = f"result_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(pack_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in ["logs", "worker_logs"]:
            if os.path.exists(folder):
                for root, _dirs, files in os.walk(folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=os.path.dirname(folder))
                        zipf.write(file_path, arcname)
    print(f"[Master] Packed logs into {pack_name}")


def main():
    parser = argparse.ArgumentParser(description="Workerfarm 并行 API 测试运行器")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--paddle_only", action="store_true", help="只测试 Paddle")
    group.add_argument("--paddle_cinn", action="store_true", help="测试 Paddle CINN")
    group.add_argument("--accuracy", action="store_true", help="测试精度")

    parser.add_argument("--api_config_file", required=True, help="API 配置文件路径")
    parser.add_argument("--gpus", required=True, help="用逗号分隔的 GPU ID 列表")
    parser.add_argument("--per_gpu_concurrency", type=int, default=1, help="每块 GPU 并发数")
    parser.add_argument("--timeout", type=int, default=1800, help="每个任务超时时间（秒）")

    args = parser.parse_args()

    if os.path.exists("worker_logs"):
        shutil.rmtree("worker_logs")
        print("[Master] Deleted existing worker_logs directory.")

    if args.paddle_only:
        selected_mode = "paddle_only"
    elif args.paddle_cinn:
        selected_mode = "paddle_cinn"
    elif args.accuracy:
        selected_mode = "accuracy"
    else:
        raise ValueError("必须指定 --paddle_only 或 --paddle_cinn 或 --accuracy")

    gpu_list = [int(g) for g in args.gpus.split(",")]
    concurrency = args.per_gpu_concurrency
    timeout = args.timeout

    os.makedirs("logs", exist_ok=True)
    master_log = open(os.path.join("logs", "master.log"), "w")
    timeout_cases = open(os.path.join("logs", "timeout_cases.txt"), "w")
    error_cases = open(os.path.join("logs", "error_cases.txt"), "w")

    # 加载API配置
    with open(args.api_config_file) as f:
        pending_api_configs = [line.strip() for line in f if line.strip()]

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    workers = {}
    worker_last_task_time = {}

    total_worker_num = len(gpu_list) * concurrency
    pending_task_index = 0  # 下一个要塞的任务索引
    task_id_counter = 0
    total_tasks = 0  # ���新增，动态记录总任务数
    max_queue_size = total_worker_num * 3  # 队列最大保持一定量
    print(f"Launching {total_worker_num} workers...")
    master_log.write(f"{datetime.now()} Launching {total_worker_num} workers...\n")
    master_log.flush()

    for idx in range(total_worker_num):
        p = spawn_worker(gpu_list, concurrency, idx, task_queue, result_queue, selected_mode)
        workers[idx] = p
        worker_last_task_time[idx] = time.time()

    task_status = {}
    task_id_to_worker_idx = {}
    completed_task_ids = set()

    # ====== 首次预塞一批任务 ======
    for _ in range(min(max_queue_size, len(pending_api_configs))):
        api_config_str = pending_api_configs[pending_task_index]
        task_queue.put((task_id_counter, api_config_str))
        task_status[task_id_counter] = (None, api_config_str)
        pending_task_index += 1
        task_id_counter += 1
        total_tasks += 1

    success_count = 0
    error_count = 0
    timeout_count = 0

    # ====== 主循环 ======
    while len(completed_task_ids) < total_tasks:
        now = time.time()
        try:
            task_id, status, *others = result_queue.get(timeout=5)

            if task_id == "worker_init_error":
                raise RuntimeError(f"Worker initialization failed: {others[0]}")

            if status == "start":
                if task_id in task_status:
                    _, api_config_str = task_status[task_id]
                    task_status[task_id] = (time.time(), api_config_str)
                worker_idx = others[0]
                task_id_to_worker_idx[task_id] = worker_idx
                continue

            completed_task_ids.add(task_id)
            start_time, api_config_str = task_status.pop(task_id)

            if status == "success":
                success_count += 1
                master_log.write(f"{datetime.now()} Task {task_id} SUCCESS: {api_config_str}\n")
            elif status == "error":
                error_count += 1
                error_msg = others[0] if others else "Unknown error"
                master_log.write(
                    f"{datetime.now()} Task {task_id} ERROR: {api_config_str}, {error_msg}\n"
                )
                error_cases.write(f"{api_config_str}\n")

            master_log.flush()

        except Exception:
            pass  # 超时，下面补充任务 + 手动检查超时

        # 补充新任务：保持队列里有任务在跑
        while pending_task_index < len(pending_api_configs) and task_queue.qsize() < max_queue_size:
            api_config_str = pending_api_configs[pending_task_index]
            task_queue.put((task_id_counter, api_config_str))
            task_status[task_id_counter] = (None, api_config_str)
            pending_task_index += 1
            task_id_counter += 1
            total_tasks += 1

        # 超时检测
        for tid, (start_time, api_config_str) in list(task_status.items()):
            if start_time is None:
                continue  # 还没真正开始执行
            if tid not in completed_task_ids and (now - start_time) > timeout:
                print(f"[Timeout] Task {tid} after {timeout}s")
                master_log.write(f"{datetime.now()} Task {tid} TIMEOUT: {api_config_str}\n")
                timeout_cases.write(f"{api_config_str}\n")
                master_log.flush()
                timeout_cases.flush()
                completed_task_ids.add(tid)
                timeout_count += 1

                worker_idx = task_id_to_worker_idx.get(tid)
                if worker_idx is not None:
                    p = workers.get(worker_idx)
                    # if p and p.is_alive():
                    if p:
                        print(
                            f"[Master] Killing Worker {worker_idx} (PID={p.pid}) due to task timeout..."
                        )
                        kill_worker(p)
                        workers[worker_idx] = spawn_worker(
                            gpu_list,
                            concurrency,
                            worker_idx,
                            task_queue,
                            result_queue,
                            selected_mode,
                        )
                        worker_last_task_time[worker_idx] = time.time()

        # 检查worker意外挂掉  这里之前有个bug，就是如果hang了，很多时候进程会自己kill，然后到超时的时候任务会二次kill进程，导致正常任务被杀死，所以禁用了这个检测
        for idx, p in list(workers.items()):
            if not p.is_alive():
                print(f"[Master] Worker {idx} (PID={p.pid}) died unexpectedly, restarting...")

                # 找出这个worker正在执行的任务
                tasks_of_dead_worker = [
                    tid
                    for tid, worker_idx in task_id_to_worker_idx.items()
                    if worker_idx == idx and tid not in completed_task_ids
                ]

                for tid in tasks_of_dead_worker:
                    start_time, api_config_str = task_status.pop(tid, (None, None))
                    if api_config_str:
                        completed_task_ids.add(tid)
                        error_count += 1
                        master_log.write(
                            f"{datetime.now()} Task {tid} ERROR (Worker died): {api_config_str}\n"
                        )
                        error_cases.write(f"{api_config_str}\n")
                        master_log.flush()
                        error_cases.flush()

                # 重启新的worker
                workers[idx] = spawn_worker(
                    gpu_list, concurrency, idx, task_queue, result_queue, selected_mode
                )
                worker_last_task_time[idx] = time.time()

    # 收尾
    for _ in range(total_worker_num):
        task_queue.put(None)

    for p in workers.values():
        p.join()

    print(
        f"{datetime.now()} Summary: {success_count} success, {error_count} error, {timeout_count} timeout."
    )
    master_log.write(
        f"{datetime.now()} Summary: {success_count} success, {error_count} error, {timeout_count} timeout.\n"
    )

    master_log.close()
    timeout_cases.close()
    error_cases.close()
    pack()


if __name__ == "__main__":
    main()
