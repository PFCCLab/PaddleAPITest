from __future__ import annotations

import argparse
import atexit
import gc
import heapq
import importlib
import itertools
import math
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
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import cpu_count, set_start_method
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pynvml
import yaml
from sanitizer_session import (
    SanitizerSession,
    encode_ready,
    encode_result,
    parse_event,
)
from tester.dump_writer import (
    dump_enabled,
    parse_strict_bool,
    record_dump_terminal_status,
    resolve_dump_options,
)
from tester.gpu_memory_preflight import (
    GpuMemoryDeferred,
    estimate_gpu_memory,
    should_check_grad,
)

GIB = 1024**3
# GPU 超时只由批次入口读取，避免单 case 和批次形成不同的环境变量优先级。
GPU_PRESSURE_TIMEOUT_ENV_VAR = "PADDLEAPITEST_GPU_PRESSURE_TIMEOUT_SECONDS"
# 600 秒是显存压力保护，不是单 case 的执行超时。
DEFAULT_GPU_PRESSURE_TIMEOUT_SECONDS = 600.0
# slot 尚未 ready 的初始化态：批处理主循环据此判断是否仍有进展中 worker。
_INITIALIZING_SLOT_STATES = frozenset({"starting", "loaded", "preparing"})
from tester.log_writer import (
    init_log,
    log_aggregation,
    log_report,
    log_retest,
    log_runtime,
    log_worker,
)
from tester.runtime_config import (
    TestRuntimeConfig,
    limit_worker_layout,
    runtime_config_for_gpu,
)
from tester.sanitizer_output import analyze_sanitizer_output

os.environ["FLAGS_use_system_allocator"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


@dataclass(frozen=True)
class GpuMemorySnapshot:
    # snapshot 是调度器唯一认可的物理显存输入，不保存进程级占用猜测。
    """一次物理显存采样；free 已包含外部进程和遗留 CUDA context。"""

    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class CaseGpuEstimate:
    # compute/comparison 分开记录，双卡模式不能用计算卡峰值代替对比卡预算。
    compute_bytes: int = 0
    comparison_bytes: int = 0


@dataclass
class GpuReclaimTracker:
    """用两个独立物理采样确认异步释放已经停止。"""

    tolerance_bytes: int = 64 * 1024**2
    sample_interval_seconds: float = 1.0
    _last_free_bytes: int | None = None
    _last_sample_at: float | None = None

    def reset(self):
        # 新的 release_pending 周期必须重新建立物理显存基线。
        self._last_free_bytes = None
        self._last_sample_at = None

    def observe(self, snapshot, *, now):
        # NVML 快照可能在多个主循环迭代中重复，间隔不足时不算独立样本。
        free_bytes = max(0, int(snapshot.free_bytes))
        if self._last_sample_at is None:
            self._last_free_bytes = free_bytes
            self._last_sample_at = float(now)
            return False
        if float(now) - self._last_sample_at < self.sample_interval_seconds:
            return False
        stable = abs(free_bytes - int(self._last_free_bytes or 0)) <= self.tolerance_bytes
        self._last_free_bytes = free_bytes
        self._last_sample_at = float(now)
        return stable


@dataclass
class GpuPressureTimeout:
    """记录 pending 在没有任何持久化进展时的连续阻塞时长。"""

    timeout_seconds: float
    blocked_since: float | None = None

    def update(self, *, blocked, now):
        # 任意真实派发或持久化终态都会清零连续阻塞窗口。
        if not blocked:
            self.blocked_since = None
            return False
        if self.blocked_since is None:
            self.blocked_since = float(now)
        return float(now) - self.blocked_since >= self.timeout_seconds


def read_gpu_pressure_timeout(environ=None):
    # 非法环境变量在批次启动阶段失败，不能运行到压力循环才静默采用默认值。
    source = os.environ if environ is None else environ
    raw_value = source.get(
        GPU_PRESSURE_TIMEOUT_ENV_VAR,
        str(DEFAULT_GPU_PRESSURE_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"{GPU_PRESSURE_TIMEOUT_ENV_VAR} must be a finite non-negative number, "
            f"got {raw_value!r}"
        ) from err
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError(
            f"{GPU_PRESSURE_TIMEOUT_ENV_VAR} must be a finite non-negative number, "
            f"got {raw_value!r}"
        )
    return timeout


def parse_sanitizer_timing_file(path):
    # 观测文件损坏只丢弃样本，不能改变 sanitizer 的业务结果。
    values = {}
    try:
        with open(path, encoding="utf-8") as timing_file:
            for line in timing_file:
                # 只读取制表符分隔的 phase，普通输出不会进入统计。
                phase, separator, raw_duration = line.rstrip("\n").partition("\t")
                if not separator:
                    continue
                try:
                    duration = float(raw_duration)
                except ValueError:
                    continue
                if duration >= 0:
                    values[phase] = values.get(phase, 0.0) + duration
    except OSError:
        return {}
    return values


@dataclass(frozen=True)
class GpuSchedulingPolicy:
    safety_reserve_bytes_min: int = 2 * GIB
    safety_reserve_fraction: float = 0.05
    minimum_case_bytes: int = 1 * GIB
    case_margin_bytes: int = 512 * 1024**2
    case_multiplier: float = 1.25

    def safety_reserve_bytes(self, snapshot):
        # 安全余量同时覆盖小卡固定开销和大卡按比例增长的 workspace。
        return max(
            self.safety_reserve_bytes_min,
            int(max(0, snapshot.total_bytes) * self.safety_reserve_fraction),
        )

    def case_admission_bytes(self, estimated_peak_bytes):
        # 未知配置仍需占用最小预算，估算值较大时再叠加边际和倍率。
        estimate = max(0, int(estimated_peak_bytes))
        return max(
            self.minimum_case_bytes,
            estimate + self.case_margin_bytes,
            int(estimate * self.case_multiplier),
        )


@dataclass
class GpuReservation:
    # state 保留 active 与 release_pending 的差异，后者仍然占用承诺。
    assignment_id: int
    slot_index: int
    device_bytes: dict[int, int]
    state: str = "active"


class GpuReservationLedger:
    """按 assignment 持有显存承诺，release_pending 期间禁止复用。"""

    def __init__(self, device_ids, *, snapshots, max_workers, policy=None):
        # 一个 group 要么是一张计算卡，要么是不可拆分的双卡 pair。
        if not device_ids or len(device_ids) > 2:
            raise ValueError("a reservation group must contain one or two devices")
        self.device_ids = tuple(device_ids)
        self.max_workers = max(0, int(max_workers))
        self.policy = policy or GpuSchedulingPolicy()
        self._reservations = {}
        self._committed = dict.fromkeys(self.device_ids, 0)
        self._trackers = {gpu_id: GpuReclaimTracker() for gpu_id in self.device_ids}
        self._pending_release = set()
        self._claimed = set()
        # 从单 worker 起步，稳定完成后再逐步扩容，避免启动阶段瞬间压满显存。
        self.target_workers = min(self.max_workers, 1)
        self._success_streak = 0
        # 初始化快照后才允许 reserve，避免无采样状态被当成无限容量。
        self.update_snapshots(snapshots)

    def update_snapshots(self, snapshots):
        # 只保留本 group 的设备，双卡确认必须使用同一轮调用传入的快照。
        self._snapshots = {gpu_id: snapshots[gpu_id] for gpu_id in self.device_ids}

    def _available(self, gpu_id):
        # free 已含外部占用和遗留 context，不再按进程归属做推断。
        snapshot = self._snapshots[gpu_id]
        return max(0, snapshot.free_bytes - self.policy.safety_reserve_bytes(snapshot))

    def _requested(self, estimates):
        # 双卡 reservation 同时扣计算卡和 comparison 卡的预算。
        requested = {self.device_ids[0]: self.policy.case_admission_bytes(estimates.compute_bytes)}
        if len(self.device_ids) == 2:
            requested[self.device_ids[1]] = self.policy.case_admission_bytes(
                estimates.comparison_bytes
            )
        return requested

    def reserve(self, *, assignment_id, slot_index, estimates):
        # 常驻 worker 的旧 lease 可以被同一 slot 的新 assignment 原子替换。
        if assignment_id in self._reservations:
            return None
        requested = self._requested(estimates)
        old_lease = next(
            (
                reservation
                for reservation in self._reservations.values()
                if reservation.slot_index == slot_index and reservation.state == "release_pending"
            ),
            None,
        )
        active_count = sum(
            reservation.state == "active" for reservation in self._reservations.values()
        )
        if active_count >= self.target_workers:
            return None
        committed = dict(self._committed)
        if old_lease is not None:
            for gpu_id, amount in old_lease.device_bytes.items():
                committed[gpu_id] -= amount
        if any(
            committed[gpu_id] + requested[gpu_id] > self._available(gpu_id)
            for gpu_id in self.device_ids
        ):
            return None
        if old_lease is not None:
            self._pending_release.discard(old_lease.assignment_id)
            self._reservations.pop(old_lease.assignment_id, None)
            self._claimed.discard(old_lease.assignment_id)
            self._committed.update(committed)
            # pending lease 被同 slot 接管时，旧回收样本不能污染下一次 reclaim 基线。
            for tracker in self._trackers.values():
                tracker.reset()
        reservation = GpuReservation(assignment_id, slot_index, requested)
        self._reservations[assignment_id] = reservation
        for gpu_id, amount in requested.items():
            self._committed[gpu_id] += amount
        return reservation

    def record_result(self, msg_type):
        """按结果调整 group 并发上限；异常只收缩，不改变用户硬上限。"""
        if msg_type == "done":
            self._success_streak += 1
            if self._success_streak >= 2 and self.target_workers < self.max_workers:
                self.target_workers += 1
                self._success_streak = 0
            return
        if msg_type in {"deferred", "timeout", "crashed"}:
            self._success_streak = 0
            self.target_workers = max(1, min(self.target_workers, self.max_workers) - 1)

    def confirm(self, assignment_id, snapshots):
        # worker bootstrap 可能改变 free，派发前必须做一次独立物理确认。
        reservation = self._reservations.get(assignment_id)
        if reservation is None or reservation.state != "active":
            return False
        self.update_snapshots(snapshots)
        # 确认检查整个 group 的承诺，避免多个延迟启动的 worker 分别通过后合计超卖。
        if any(self._committed[gpu_id] > self._available(gpu_id) for gpu_id in self.device_ids):
            self.mark_release_pending(assignment_id)
            return False
        return True

    def mark_release_pending(self, assignment_id):
        # 终态只开启回收观察，不直接释放显存承诺。
        reservation = self._reservations.get(assignment_id)
        if reservation is None:
            return False
        if reservation.state == "active":
            reservation.state = "release_pending"
            self._pending_release.add(assignment_id)
        return True

    def advance_reclaim(self, snapshots, *, now):
        # 同组所有设备稳定后批量释放，避免双卡只回收一半就重新派发。
        if not self._pending_release:
            return 0
        self.update_snapshots(snapshots)
        stable = all(
            self._trackers[gpu_id].observe(self._snapshots[gpu_id], now=now)
            for gpu_id in self.device_ids
        )
        if not stable:
            # 任一设备仍在变化，保留全部 pending reservation。
            return 0
        released = []
        for assignment_id in tuple(self._pending_release):
            reservation = self._reservations.pop(assignment_id, None)
            self._pending_release.discard(assignment_id)
            if reservation is None:
                continue
            for gpu_id, amount in reservation.device_bytes.items():
                self._committed[gpu_id] -= amount
            released.append(assignment_id)
        for tracker in self._trackers.values():
            # 下一批回收必须使用新的基线，不能复用上一批的稳定样本。
            tracker.reset()
        return tuple(released)

    def claim_terminal(self, assignment_id):
        # claimed 集合防止 timeout/crash 与 worker 正常终态重复结算。
        reservation = self._reservations.get(assignment_id)
        if reservation is None or assignment_id in self._claimed:
            return False
        self._claimed.add(assignment_id)
        self.mark_release_pending(assignment_id)
        return True


# 运行时透传给 test class 的选项白名单。
VALID_TEST_ARGS = {
    "test_amp",
    "test_backward",
    "atol",
    "rtol",
    "manual_threshold_config_file",
    # tester 只在严格比较失败后读取该文件。
    "accuracy_manual_file",
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
    "accuracy_dual_gpu",
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
    "accuracy_manual_file",
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
# 主模式互斥校验只看这些开关；dual 标志会先展开成对应主模式。
PRIMARY_TEST_MODES = (
    "paddle_only",
    "paddle_cinn",
    "accuracy",
    "paddle_gpu_performance",
    "torch_gpu_performance",
    "paddle_torch_gpu_performance",
    "accuracy_stable",
    "paddle_custom_device",
    "custom_device_vs_gpu",
)
TORCH_REFERENCE_MODES = (
    "accuracy",
    "accuracy_stable",
    "accuracy_dual_gpu",
    "accuracy_stable_dual_gpu",
    "torch_gpu_performance",
    "paddle_torch_gpu_performance",
)
TORCH_UTILITY_MODES = TORCH_REFERENCE_MODES + (
    "paddle_cinn",
    "paddle_gpu_performance",
    "paddle_custom_device",
    "custom_device_vs_gpu",
)
GPU_MEMORY_PREFLIGHT_MODES = (
    "accuracy_stable_dual_gpu",
    "accuracy_dual_gpu",
    "accuracy_stable",
    "accuracy",
    "paddle_only",
)
_BYTES_PER_GIB = 1024**3

# 选择测试类的优先级顺序。
TEST_CLASS_BY_OPTION = (
    ("paddle_only", "APITestPaddleOnly"),
    ("paddle_cinn", "APITestCINNVSDygraph"),
    ("accuracy_dual_gpu", "APITestAccuracy"),
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
GPU_MEMORY_DEFER_INITIAL_BACKOFF_SECONDS = 1.0
GPU_MEMORY_DEFER_MAX_BACKOFF_SECONDS = 30.0
# 首次 deferred 只清理并复用常驻 worker，连续失败才退役进程。
GPU_MEMORY_DEFER_RETIRE_AFTER = 2
SANITIZER_COMPUTE_BUDGET_ENV = "PADDLEAPITEST_SANITIZER_COMPUTE_BUDGET_GIB"
SANITIZER_COMPARISON_BUDGET_ENV = "PADDLEAPITEST_SANITIZER_COMPARISON_BUDGET_GIB"
SANITIZER_TIMING_FILE_ENV = "PADDLEAPITEST_SANITIZER_TIMING_FILE"
# 调度只需要有限候选；窗口随最大并发放大，保留跳过大 case 的能力。
CANDIDATE_WINDOW_PER_WORKER = 32
CANDIDATE_WINDOW_MIN = 64
# 窗口内全部装不下时放大一次扫描范围，但仍然有界：无界扫描在百万级 pending 上
# 单轮就要十几秒，比它要修的问题更慢。超出该上界的 case 靠队首消费自然前移。
EXTENDED_SCAN_MAX_CANDIDATES = 8192
# 扩展扫描的最小间隔，从上一次扫描“结束”开始计时，避免扫描本身把间隔耗尽。
EXTENDED_SCAN_MIN_INTERVAL_SECONDS = 5.0
WORKER_PREPARE_RUNTIME = "prepare_runtime"


def _option_enabled(options, name):
    return bool(getattr(options, name, False))


def _memory_defer_delay(retry_count):
    retry_count = max(0, int(retry_count))
    return min(
        GPU_MEMORY_DEFER_MAX_BACKOFF_SECONDS,
        GPU_MEMORY_DEFER_INITIAL_BACKOFF_SECONDS * (2 ** min(retry_count, 5)),
    )


def _collect_wait_timeout(pending_dispatch, default=0.5):
    pending_delay = pending_dispatch.earliest_delay()
    if pending_delay is None:
        return default
    return min(default, max(0.1, pending_delay))


def _fatal_log_type_for_error(error, terminal_log_type=None):
    """按普通运行路径的优先级识别 fatal 类型。"""
    error_text = str(error).lower()
    if any(marker in error_text for marker in OOM_ERROR_MARKERS):
        return "oom"
    if terminal_log_type == "torch_error" and any(
        marker in error_text for marker in CUDA_ERROR_MARKERS
    ):
        return "torch_error"
    if any(marker in error_text for marker in CUDA_ERROR_MARKERS):
        return "paddle_cuda"
    return None


def _normalize_sanitizer_exitcode(output, *, returncode, sanitizer_error_exitcode):
    """把 sanitizer 专用退出码还原为普通 worker 的 fatal 退出协议。"""
    if returncode != sanitizer_error_exitcode:
        return returncode
    # 仅把 sanitizer 的协议码转换为统一 fatal 码，普通 child 退出码保持原语义。
    # Torch 已写出的终态会出现在 child 输出中；OOM 仍由共享文本规则优先识别。
    terminal_log_type = "torch_error" if "[torch_error]" in output.lower() else None
    fatal_log_type = _fatal_log_type_for_error(output, terminal_log_type)
    return log_worker.fatal_exit_code(fatal_log_type or "paddle_cuda", False)


def _sanitizer_analysis_is_ignored(analysis):
    """仅在过滤后没有应用 fatal 证据时忽略 sanitizer 噪声。"""
    return bool(
        analysis.only_ignored_diagnostics and _fatal_log_type_for_error(analysis.output) is None
    )


def _sanitizer_session_output_has_report(output):
    """识别 session 当前 request 是否包含 compute-sanitizer 诊断块。"""
    # 普通 Paddle 输出不会使用 sanitizer 的固定分隔线；按 request 片段检查可恢复逐 case 归因。
    if "========= " not in output:
        return False
    if "Error:" in output or "Program hit" in output:
        return True
    summary_counts = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", output)
    return any(int(count) > 0 for count in summary_counts)


@dataclass
class GpuEstimateFailureReport:
    """汇总显存估算失败。

    估算失败会让 case 退化到 1 GiB 准入下界：既可能过量准入导致真实 OOM，也可能
    在 worker 侧被误判为 skip。逐条打印会在大批量失败时刷屏，因此只保留首条诊断
    加总数，足以区分“个别 API 没有模型”和“估算器整体坏了”。
    """

    total: int = 0
    first_config: str | None = None
    first_error: str | None = None
    error_counts: dict[str, int] = field(default_factory=dict)

    def record(self, config, err):
        self.total += 1
        error_kind = type(err).__name__
        self.error_counts[error_kind] = self.error_counts.get(error_kind, 0) + 1
        if self.first_config is None:
            self.first_config = config
            self.first_error = f"{type(err).__name__}: {err}"

    def emit(self, all_case):
        if not self.total:
            return
        print(
            f"[gpu] ESTIMATE_FALLBACK | fallback={self.total}/{all_case} | "
            f"types={self.error_counts} | first={self.first_config} | "
            f"error={self.first_error}",
            flush=True,
        )


@dataclass
class BatchRetryState:
    per_case_external_kill_retries: dict[str, int] = field(default_factory=dict)
    per_case_memory_defer_retries: dict[str, int] = field(default_factory=dict)
    case_memory_estimates: dict[str, CaseGpuEstimate] = field(default_factory=dict)
    slot_memory_defer_retries: dict[int, int] = field(default_factory=dict)
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


@dataclass(frozen=True)
class PendingCase:
    """待重试 case 及其最早可再次派发时间。"""

    config: str
    ready_at: float = 0.0
    compute_estimate_bytes: int = 0
    comparison_estimate_bytes: int = 0

    @property
    def gpu_estimate(self):
        return CaseGpuEstimate(self.compute_estimate_bytes, self.comparison_estimate_bytes)


class PendingQueue:
    """待派发 case 队列：ready 段保持 FIFO，延迟重试段按 ready_at 组织为最小堆。

    协议边界：
    - 所有读取入口内部先 promote 到期项，调用方无需关心两段结构的同步顺序。
    - appendleft 用于整波回滚与外部 kill 重试；仅当 ready_at 已到期才真正插到队首。
    - 与旧的单一 deque 实现的差异：未到期 case 不再占据 ready 段，因此取任务和
      计算最近可派发时间都不需要遍历整个队列。
    """

    __slots__ = (
        "_delayed",
        "_ready",
        "_sequence",
        "_ready_version",
        "_ready_snapshot_cache",
        "_ready_snapshot_iterator",
        "_ready_snapshot_version",
    )

    def __init__(self, cases=()):
        # OrderedDict 同时提供稳定 FIFO 和按 config 的 O(1) 删除。
        self._ready = OrderedDict()
        self._delayed = []
        # 单调序号让相同 ready_at 的堆序稳定，同时避免堆比较落到 PendingCase 上。
        self._sequence = 0
        self._ready_version = 0
        self._ready_snapshot_cache = None
        self._ready_snapshot_iterator = None
        self._ready_snapshot_version = -1
        # 构造阶段没有并发读者，批量填充后一次发布版本，避免逐项失效扫描缓存。
        for case in cases:
            if case.ready_at <= 0.0:
                self._ready[case.config] = case
            else:
                self._push_delayed(case)
        if self._ready:
            self._ready_version = 1

    def __len__(self):
        return len(self._ready) + len(self._delayed)

    def __bool__(self):
        return bool(self._ready) or bool(self._delayed)

    @property
    def ready_count(self):
        """返回已到期段长度；扩展扫描不应把延迟 case 算入候选窗口。"""
        self._promote(time.monotonic())
        return len(self._ready)

    def _invalidate_ready_snapshot(self):
        self._ready_version += 1
        self._ready_snapshot_cache = None
        self._ready_snapshot_iterator = None
        self._ready_snapshot_version = -1

    def _ready_snapshot_to(self, end):
        """按需缓存扫描前缀，避免百万级 ready 队列首次扩展时整队复制。"""
        if self._ready_snapshot_version != self._ready_version:
            self._ready_snapshot_cache = []
            self._ready_snapshot_iterator = iter(self._ready.values())
            self._ready_snapshot_version = self._ready_version
        cache = self._ready_snapshot_cache
        if cache is None or self._ready_snapshot_iterator is None:
            return []
        missing = max(0, int(end) - len(cache))
        if missing:
            cache.extend(itertools.islice(self._ready_snapshot_iterator, missing))
        return cache

    def _push_delayed(self, case):
        self._sequence += 1
        heapq.heappush(self._delayed, (case.ready_at, self._sequence, case))

    def _promote(self, now):
        promoted = []
        while self._delayed and self._delayed[0][0] <= now:
            promoted.append(heapq.heappop(self._delayed)[2])
        if promoted:
            # 重试 case 到期后优先于尚未派发的新 case；逆序扩展才能保留到期顺序。
            for case in reversed(promoted):
                self._ready[case.config] = case
                self._ready.move_to_end(case.config, last=False)
            self._invalidate_ready_snapshot()

    def append(self, case, *, now=None):
        now = time.monotonic() if now is None else now
        if case.ready_at <= now:
            self._ready[case.config] = case
            self._invalidate_ready_snapshot()
        else:
            self._push_delayed(case)

    def appendleft(self, case, *, now=None):
        now = time.monotonic() if now is None else now
        if case.ready_at <= now:
            self._ready[case.config] = case
            self._ready.move_to_end(case.config, last=False)
            self._invalidate_ready_snapshot()
        else:
            self._push_delayed(case)

    def clear(self):
        self._ready.clear()
        self._delayed.clear()
        self._invalidate_ready_snapshot()

    def earliest_delay(self, now=None):
        """返回最近一次可派发的剩余等待秒数；队列为空返回 None。

        读取路径会顺带把到期的延迟 case 迁入 ready 段。
        """
        if not self:
            return None
        now = time.monotonic() if now is None else now
        self._promote(now)
        if self._ready:
            return 0.0
        return max(0.0, self._delayed[0][0] - now)

    def pop_ready(self, now=None):
        """取出一个已到期 case 的配置字符串；没有已到期 case 时返回 None。"""
        now = time.monotonic() if now is None else now
        self._promote(now)
        if not self._ready:
            return None
        _, case = self._ready.popitem(last=False)
        self._invalidate_ready_snapshot()
        return case.config

    def candidate_window(self, limit, now=None):
        """返回队首至多 limit 个已到期 case；limit 为 None 表示全部 ready case。

        返回的是队首快照副本，不消费队列；真正取走由 take_case_selection 完成。
        读取路径会顺带把到期的延迟 case 迁入 ready 段。
        """
        now = time.monotonic() if now is None else now
        self._promote(now)
        if limit is None or limit >= len(self._ready):
            return list(self._ready.values())
        return list(itertools.islice(self._ready.values(), limit))

    def scan_window(self, limit, *, cursor=0, now=None):
        """从 ready 段的游标位置取有界窗口，并返回下一次扫描游标。

        ready 队列不变时复用一次快照，避免为了跨过一批不可准入 case 而重复从队首
        扫描；队列结构变化后会让调用方重新从当前窗口起点开始。
        """
        now = time.monotonic() if now is None else now
        self._promote(now)
        ready_count = len(self._ready)
        if not ready_count:
            return [], 0
        start = int(cursor) % ready_count
        end = min(start + max(0, int(limit)), ready_count)
        ready = self._ready_snapshot_to(end)
        next_cursor = end if end < ready_count else 0
        return ready[start:end], next_cursor

    def take_case_selection(self, candidates, selected_indices):
        """从任意扫描窗口中取走选中的 case，并保持其余 ready 顺序。"""
        selected = [candidates[index] for index in selected_indices if 0 <= index < len(candidates)]
        if not selected:
            return []
        # 只按 key 删除选中项，延迟堆和未选中的 ready 顺序都不变。
        for case in selected:
            self._ready.pop(case.config, None)
        self._invalidate_ready_snapshot()
        return selected

    def iter_all(self):
        """遍历全部 case，仅用于构建批次级快照。"""
        yield from self._ready.values()
        for _, _, case in self._delayed:
            yield case


@dataclass
class TerminalClaims:
    """case 终态的一次性认领表，CPU 和 GPU 路径共用同一协议。"""

    tokens: dict[int, tuple[int | None, str]] = field(default_factory=dict)

    def register(self, slot_index, worker_pid, config):
        self.tokens[slot_index] = (worker_pid, config)

    def claim(self, msg):
        message = BatchMessage.from_raw(msg)
        expected_token = self.tokens.get(message.slot_index)
        if expected_token is None or expected_token != (message.worker_pid, message.config):
            return False
        del self.tokens[message.slot_index]
        return True


@dataclass(frozen=True)
class WorkerTask:
    """主调度器传给 worker 的 case 及本波实际显存预算。"""

    config: str
    workers_on_gpu: int
    compute_budget_gib: float
    comparison_budget_gib: float = 0.0


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
    # watchdog 生成的 timeout/crash 携带检测时刻，避免队列等待污染执行耗时。
    terminal_timestamp: float | None = None
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
        if message.msg_type in {"done", "timeout"}:
            message.worker_pid = msg[3] if len(msg) > 3 else None
            message.completed_offset = msg[4] if len(msg) > 4 else None
            message.terminal_timestamp = msg[5] if len(msg) > 5 else None
        elif message.msg_type in {"deferred", "error"}:
            message.reason = msg[3] if len(msg) > 3 else ""
            message.worker_pid = msg[4] if len(msg) > 4 else None
            message.completed_offset = msg[5] if len(msg) > 5 else None
        elif message.msg_type == "crashed":
            message.exitcode = msg[3] if len(msg) > 3 else None
            if len(msg) > 5 and msg[5] == "session":
                message.crash_source = "session"
                message.worker_pid = msg[6] if len(msg) > 6 else None
                message.completed_offset = msg[7] if len(msg) > 7 else None
            else:
                message.worker_pid = msg[4] if len(msg) > 4 else None
                message.completed_offset = msg[5] if len(msg) > 5 else None
                message.terminal_timestamp = msg[6] if len(msg) > 6 else None
        return message


# ─── WorkerPool：每个 worker 独立队列的架构 ───────────────────────────────


@dataclass
class WorkerSlot:
    """表示一个拥有独立输入队列的 worker 进程槽位。"""

    index: int
    gpu_id: int | None
    comparison_gpu_id: int | None = None
    process: mp.Process | None = None
    input_queue: mp.Queue | None = None
    current_task: str | None = None
    task_start_time: float | None = None
    child_pid: int | None = None
    started_at: float | None = None
    state: str = "dead"  # dead、starting、loaded、preparing、idle、busy、suspended


def _import_optional_runtime_module(module_name):
    try:
        importlib.import_module(module_name)
    except Exception:
        pass


def _init_runtime_modules(options, *, prepare_runtime=True):
    with log_worker.suppress_startup_output():
        import paddle

        globals()["paddle"] = paddle
        if options.test_cpu:
            paddle.device.set_device("cpu")
        elif not _option_enabled(options, "paddle_custom_device"):
            # CUDA_VISIBLE_DEVICES 只负责限定 slot，Paddle 仍需要显式设置 device。
            paddle.set_device("gpu")
        _import_optional_runtime_module("paddlefleet_ops")
        _import_optional_runtime_module("FusedQuantOps")
        import tester

        if prepare_runtime:
            # 单 case 入口没有主进程握手，仍在模块加载后立即完成通用 preparation。
            tester.prepare_process_runtime(options)
        test_class = _select_test_class(options)
        globals().update({"APIConfig": tester.APIConfig, test_class.__name__: test_class})
        return tester


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
    prepare_runtime=True,
    redirect_output=False,
):
    init_log(options.log_dir)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _visible_gpu_ids(gpu_id, comparison_gpu_id)
        workers_on_gpu = (getattr(options, "gpu_workers_per_gpu_map", {}) or {}).get(gpu_id, 1)
        os.environ["PADDLEAPITEST_WORKERS_ON_GPU"] = str(workers_on_gpu)
    if slot_index is not None and gpu_id is not None:
        # backend 导入和准备阶段即可读取稳定 slot，复活 worker 也遵循同一初始化协议。
        os.environ["PADDLEAPITEST_WORKER_SLOT"] = str(slot_index)

    tester = _init_runtime_modules(options, prepare_runtime=prepare_runtime)

    if redirect_output:
        log_worker.redirect_stdio()
    return tester


def _await_runtime_preparation(input_queue):
    """等待主进程放行 preparation；None 表示 slot 已被正常回收。"""
    command = input_queue.get()
    if command is None:
        return False
    if command != WORKER_PREPARE_RUNTIME:
        raise RuntimeError(f"unexpected worker initialization command: {command!r}")
    return True


def _apply_worker_task_runtime_budget(options, task, gpu_id, comparison_gpu_id=None):
    """让 worker 内预检使用本波实际预算，而不是配置的最大逻辑并发。"""
    # 旧预检会再除以 workers_on_gpu，因此计算卡保存整波总承诺。
    workers_on_gpu = max(1, int(task.workers_on_gpu))
    workers_map = dict(getattr(options, "gpu_workers_per_gpu_map", {}) or {})
    total_memory_map = dict(getattr(options, "gpu_total_memory_map", {}) or {})
    workers_map[gpu_id] = workers_on_gpu
    total_memory_map[gpu_id] = max(0.0, float(task.compute_budget_gib)) * workers_on_gpu
    if comparison_gpu_id is not None:
        total_memory_map[comparison_gpu_id] = max(0.0, float(task.comparison_budget_gib))
    options.gpu_workers_per_gpu_map = workers_map
    options.gpu_total_memory_map = total_memory_map
    os.environ["PADDLEAPITEST_WORKERS_ON_GPU"] = str(workers_on_gpu)


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
          bypasses Python cleanup; the watchdog detects and suspends the dead worker.
        - Other crashes: any unhandled signal (SIGSEGV etc.) or SIGKILL from Watchdog timeout
          terminates the process. Watchdog detects exitcode != 0 and suspends the slot.

    The main process never dispatches to a dead/starting worker. After a crash or timeout, the
    case returns to `pending_dispatch`; a later admission may create a replacement worker.
    """
    # 模块加载与设备 preparation 分成两个阶段，主进程据 slot 拓扑控制第二阶段并发。
    try:
        tester_module = _init_worker_runtime(
            slot_index,
            gpu_id,
            comparison_gpu_id,
            options,
            prepare_runtime=False,
        )
    except Exception as e:
        result_queue.put(("init_failed", slot_index, os.getpid(), "module_load", str(e)))
        return

    result_queue.put(("loaded", slot_index, os.getpid()))
    try:
        if not _await_runtime_preparation(input_queue):
            return
        with log_worker.suppress_startup_output():
            tester_module.prepare_process_runtime(options)
        log_worker.redirect_stdio()
    except Exception as e:
        result_queue.put(("init_failed", slot_index, os.getpid(), "preparation", str(e)))
        return

    # ready 只表示模块、设备 context 和输入 backend 物化通道均已可用。
    result_queue.put(("ready", slot_index, os.getpid()))

    # ── 任务循环 ──
    while True:
        try:
            task = input_queue.get()
        except (EOFError, OSError):
            break
        if task is None:  # 毒丸
            break

        if isinstance(task, WorkerTask):
            _apply_worker_task_runtime_budget(options, task, gpu_id, comparison_gpu_id)
            api_config_str = task.config
        else:
            api_config_str = task
        result_queue.put(("ack", slot_index, os.getpid(), api_config_str))

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


def _append_sanitizer_forward_args(cmd, options):
    """把主流程的测试选项统一转发给唯一的 sanitizer session 入口。"""
    for key in SANITIZER_FORWARD_ARGS_SORTED:
        value = getattr(options, key, None)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, bool) and not value:
            continue
        if value is True:
            formatted = "True"
        elif value is False:
            formatted = "False"
        else:
            formatted = str(value)
        cmd.append(f"--{key}={formatted}")
    return cmd


def _build_sanitizer_session_command(options, sanitizer_cmd):
    """构造常驻 sanitizer/Paddle session；case 配置通过 stdin 逐条传入。"""
    cmd = [
        *sanitizer_cmd,
        sys.executable,
        str(Path(__file__).resolve()),
        f"--log_dir={options.log_dir}",
        "--_sanitizer_session=True",
    ]
    return _append_sanitizer_forward_args(cmd, options)


def _apply_sanitizer_budget(options, environ=None):
    """从 session request 恢复 wrapper 为当前 request 选择的显存预算。"""
    source = os.environ if environ is None else environ
    if SANITIZER_COMPUTE_BUDGET_ENV not in source:
        return False

    def read_nonnegative_float(name):
        try:
            value = float(source[name])
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"invalid internal sanitizer budget {name}") from err
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid internal sanitizer budget {name}")
        return value

    try:
        workers_on_gpu = int(source["PADDLEAPITEST_WORKERS_ON_GPU"])
    except (KeyError, TypeError, ValueError) as err:
        raise ValueError("invalid internal sanitizer worker count") from err
    if workers_on_gpu <= 0:
        raise ValueError("invalid internal sanitizer worker count")

    visible_gpu_ids = tuple(int(value) for value in source["CUDA_VISIBLE_DEVICES"].split(","))
    if not visible_gpu_ids:
        raise ValueError("sanitizer session requires a visible compute GPU")
    compute_budget_gib = read_nonnegative_float(SANITIZER_COMPUTE_BUDGET_ENV)
    comparison_budget_gib = read_nonnegative_float(SANITIZER_COMPARISON_BUDGET_ENV)
    compute_gpu_id = visible_gpu_ids[0]
    workers_map = dict(getattr(options, "gpu_workers_per_gpu_map", {}) or {})
    total_memory_map = dict(getattr(options, "gpu_total_memory_map", {}) or {})
    workers_map[compute_gpu_id] = workers_on_gpu
    # runtime_config_for_gpu 会按 worker 数切分，因此这里恢复整波计算卡总承诺。
    total_memory_map[compute_gpu_id] = compute_budget_gib * workers_on_gpu
    if len(visible_gpu_ids) > 1:
        total_memory_map[visible_gpu_ids[1]] = comparison_budget_gib
    options.gpu_workers_per_gpu_map = workers_map
    options.gpu_total_memory_map = total_memory_map
    return True


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

    sanitizer_cmd = getattr(options, "sanitizer_cmd", None) or shlex.split(
        options.sanitizer_command
    )
    child_env = os.environ.copy()
    visible_gpu_ids = _visible_gpu_ids(gpu_id, comparison_gpu_id)
    if visible_gpu_ids is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = visible_gpu_ids
    child_env["PADDLEAPITEST_SUPPRESS_CASE_TAGS"] = "1"
    session = SanitizerSession(_build_sanitizer_session_command(options, sanitizer_cmd), child_env)

    def terminate_child(*args):
        session.close()
        raise SystemExit(1)

    signal.signal(signal.SIGINT, terminate_child)
    signal.signal(signal.SIGTERM, terminate_child)

    def start_session():
        """启动并握手一个 session；失败只影响当前请求，wrapper 可继续重建。"""
        try:
            ready_event = session.start(
                on_output=lambda line: print(line, end="", flush=True),
                on_started=lambda child_pid: result_queue.put(
                    ("child", slot_index, os.getpid(), child_pid)
                ),
            )
        except OSError as err:
            print(f"[sanitizer session] spawn failed: {err}", flush=True)
            return False
        except (EOFError, ValueError):
            return False
        if ready_event is not None:
            result_queue.put(
                (
                    "sanitizer_session_ready",
                    slot_index,
                    os.getpid(),
                    float(ready_event.get("framework_ms", 0.0)),
                )
            )
        return True

    try:
        result_queue.put(("loaded", slot_index, os.getpid()))
        if not _await_runtime_preparation(input_queue):
            return
        # session 在 ready 前完成框架初始化，首个 case 的 timeout 不包含 bootstrap 假成功。
        if not start_session():
            result_queue.put(("init_failed", slot_index, os.getpid(), "module_load", "session"))
            return
        result_queue.put(("ready", slot_index, os.getpid()))

        request_id = 0
        while True:
            try:
                task = input_queue.get()
            except (EOFError, OSError):
                break
            if task is None:
                break

            api_config_str = task.config if isinstance(task, WorkerTask) else task
            result_queue.put(("ack", slot_index, os.getpid(), api_config_str))
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
            if not start_session():
                shutil.rmtree(case_log_dir, ignore_errors=True)
                result_queue.put(
                    (
                        # session 尚未接受该 case，不能伪造 sanitizer terminal；
                        # 交给既有 deferred 路径回队并在新 slot 中重试。
                        "deferred",
                        slot_index,
                        api_config_str,
                        "sanitizer session unavailable",
                        os.getpid(),
                        log_worker.get_worker_log_offset(),
                    )
                )
                continue

            request_id += 1
            result_queue.put(("child", slot_index, os.getpid(), session.pid))
            output_tail = deque(maxlen=40)
            with tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            ) as output_file:
                timing_path = case_log_dir / "timing.tsv"
                worker_task = (
                    task
                    if isinstance(task, WorkerTask)
                    else WorkerTask(
                        config=api_config_str,
                        workers_on_gpu=1,
                        compute_budget_gib=0.0,
                    )
                )

                def capture_output(line):
                    output_tail.append(line)
                    output_file.write(line)

                request_result = session.run_request(
                    request_id,
                    api_config_str,
                    timing_path,
                    on_output=capture_output,
                    workers_on_gpu=worker_task.workers_on_gpu,
                    compute_budget_gib=worker_task.compute_budget_gib,
                    comparison_budget_gib=worker_task.comparison_budget_gib,
                )
                status = request_result.status
                returncode = request_result.returncode
                if request_result.diagnostic:
                    output_tail.append(request_result.diagnostic)
                # 先读取隔离 timing 文件，再删除目录；消息不是 case 终态。
                # timing 归属 wrapper 执行阶段，不能与 request 的最终状态竞争 terminal claim。
                for phase, duration in parse_sanitizer_timing_file(timing_path).items():
                    result_queue.put(("sanitizer_timing", slot_index, os.getpid(), phase, duration))
                output_file.seek(0)
                raw_output = output_file.read()
                output_file.seek(0)
                if status == "crashed":
                    analysis = None
                elif returncode == options.sanitizer_error_exitcode or (
                    status == "done" and _sanitizer_session_output_has_report(raw_output)
                ):
                    analysis = analyze_sanitizer_output(
                        raw_output,
                        options.sanitizer_error_exitcode if status == "done" else returncode,
                        options.sanitizer_error_exitcode,
                    )
                    if analysis.output:
                        print(
                            analysis.output,
                            end="" if analysis.output.endswith("\n") else "\n",
                            flush=True,
                        )
                elif status in {"error", "done"}:
                    analysis = None
                    shutil.copyfileobj(output_file, sys.stdout)
                    sys.stdout.flush()
                else:
                    analysis = None

                if status == "done" and analysis is not None:
                    # 常驻 sanitizer 不在每个 request 返回 86，按当前输出片段恢复原有分类。
                    if not _sanitizer_analysis_is_ignored(analysis):
                        status = "crashed"
                        returncode = _normalize_sanitizer_exitcode(
                            analysis.output,
                            returncode=options.sanitizer_error_exitcode,
                            sanitizer_error_exitcode=options.sanitizer_error_exitcode,
                        )

                ignored = analysis is not None and _sanitizer_analysis_is_ignored(analysis)
                if status not in {"crashed", "error"} and not ignored:
                    returncode = _normalize_sanitizer_exitcode(
                        analysis.output if analysis is not None else "",
                        returncode=returncode,
                        sanitizer_error_exitcode=options.sanitizer_error_exitcode,
                    )
                if status == "done" or ignored:
                    log_worker.merge_sanitizer_case_logs(case_log_dir)
                shutil.rmtree(case_log_dir, ignore_errors=True)

                if status == "crashed":
                    result_queue.put(
                        (
                            "crashed",
                            slot_index,
                            api_config_str,
                            returncode,
                            "".join(output_tail),
                            "session",
                            os.getpid(),
                            None,
                        )
                    )
                elif status == "done" or ignored:
                    completed_offset = log_worker.write_case_end("completed", api_config_str)
                    result_queue.put(
                        (
                            "done",
                            slot_index,
                            api_config_str,
                            os.getpid(),
                            completed_offset,
                        )
                    )
                else:
                    completed_offset = log_worker.write_case_end("error", api_config_str)
                    result_queue.put(
                        (
                            "error",
                            slot_index,
                            api_config_str,
                            f"session case exited with {returncode}",
                            os.getpid(),
                            completed_offset,
                        )
                    )
    finally:
        session.close(wait=True)
        try:
            log_runtime.close_process_files()
            log_worker.restore_stdio()
        except Exception:
            pass


class WorkerPool:
    """用于公平 GPU 调度的自定义进程池，每个 worker 对应一个队列。"""

    def __init__(
        self,
        available_gpus,
        max_workers_per_gpu,
        options,
        *,
        gpu_total_memory_map=None,
        cpu_worker_count=0,
    ):
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
        # CPU worker 使用 gpu_id=None；只有需要 GPU 运行时时才建立 GPU 槽位。
        idx = 0
        if cpu_worker_count:
            for _ in range(cpu_worker_count):
                self.slots.append(WorkerSlot(index=idx, gpu_id=None))
                idx += 1
        elif _dual_gpu_mode_enabled(self.options):
            for pair_index in range(0, len(available_gpus), 2):
                slot = WorkerSlot(
                    index=idx,
                    gpu_id=available_gpus[pair_index],
                    comparison_gpu_id=available_gpus[pair_index + 1],
                )
                self.slots.append(slot)
                idx += 1
        else:
            # breadth-first 只调整启动先后，不改变每卡 slot 数与设备映射。
            max_rounds = max(max_workers_per_gpu.values(), default=0)
            for worker_round in range(max_rounds):
                for gpu_id in available_gpus:
                    if worker_round >= max_workers_per_gpu[gpu_id]:
                        continue
                    slot = WorkerSlot(index=idx, gpu_id=gpu_id)
                    self.slots.append(slot)
                    idx += 1

    @property
    def total_workers(self):
        return len(self.slots)

    def slot_devices(self):
        """返回 scheduler 需要的设备拓扑，不暴露 WorkerSlot 实例。"""
        return tuple((slot.index, slot.gpu_id, slot.comparison_gpu_id) for slot in self.slots)

    def slot_can_schedule(self, slot_index):
        # scheduler 只依赖可调度语义，不读取可变的 WorkerSlot 状态对象。
        with self._lock:
            return self.slots[slot_index].state in {"idle", "dead", "suspended"}

    def slot_can_start(self, slot_index):
        # 仅离线 slot 可启动；idle slot 已持有进程，不能创建第二代 worker。
        with self._lock:
            return self.slots[slot_index].state in {"dead", "suspended"}

    def slot_is_idle(self, slot_index):
        # idle 是 dispatch 的必要前置条件，状态检查由 pool 串行化。
        with self._lock:
            return self.slots[slot_index].state == "idle"

    def worker_pid(self, slot_index):
        # PID 是终态令牌的一部分，scheduler 不缓存进程对象。
        with self._lock:
            process = self.slots[slot_index].process
            return process.pid if process is not None else None

    def has_initializing_slots(self):
        """是否仍有未 ready 的 slot；批处理循环据此判断生命周期是否结束。"""
        return any(slot.state in _INITIALIZING_SLOT_STATES for slot in self.slots)

    def start(self):
        """CPU worker 立即启动；GPU worker 等显存准入后按需创建。"""
        # 纯 CPU 没有显存准入阶段，需要先建立进程再进入普通空闲队列派发。
        for slot in self.slots:
            if slot.gpu_id is None:
                self.start_worker(slot.index)
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="pool-watchdog"
        )
        self._watchdog_thread.start()

    def warmup_cpu_workers(self, timeout=None):
        """等待纯 CPU worker 全部就绪，启动失败时保持批次级失败语义。"""
        if timeout is None:
            timeout = self._startup_timeout()
        # CPU 首次启动也保留模块加载屏障，避免部分进程提前 preparation 争抢主机资源。
        deadline = time.monotonic() + timeout
        preparation_released = False
        while time.monotonic() < deadline:
            ready_count = sum(slot.state == "idle" for slot in self.slots)
            if ready_count == self.total_workers:
                return ready_count
            if all(slot.state in {"loaded", "preparing", "idle"} for slot in self.slots):
                if not preparation_released:
                    # 模块加载与 preparation 各自拥有完整 startup timeout。
                    deadline = time.monotonic() + timeout
                    preparation_released = True
                self.start_loaded_preparations(range(self.total_workers))
            msg = self.collect_one(timeout=min(0.5, max(0.1, deadline - time.monotonic())))
            if msg is not None:
                self.handle_message(msg)
            # 任一 slot 进入 suspended 都表示初始化已经失败，继续等待不会增加 ready 数量。
            if any(slot.state == "suspended" for slot in self.slots):
                break
        return sum(slot.state == "idle" for slot in self.slots)

    def _suspend_slot(self, slot):
        """挂起需要回收观察的 slot，后续只能由主调度循环恢复。"""
        slot.state = "suspended"
        slot.current_task = None
        slot.task_start_time = None
        slot.child_pid = None
        slot.started_at = None

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

    def start_worker(self, slot_index):
        """为指定 slot 拉起一个新的 worker 进程。"""
        slot = self.slots[slot_index]
        with self._spawn_lock:
            # 只有主调度线程在显存准入后调用；锁只防止关闭流程并发穿透。
            if self._closed or self._shutdown_event.is_set():
                return None
            if slot.process is not None and slot.process.is_alive():
                return None
            if slot.process is not None:
                self._join_process(slot.process, timeout=1)
            self._close_queue(slot.input_queue, cancel_join=True)
            slot.input_queue = mp.Queue()
            worker_target = (
                _sanitizer_worker_loop
                if getattr(self.options, "use_compute_sanitizer", False)
                else _worker_loop
            )
            # sanitizer wrapper 也是 slot 的 worker，本层不额外预建其 CUDA child。
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
            inherited_visibility = _visible_gpu_ids(slot.gpu_id, slot.comparison_gpu_id)
            previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
            spawn_started_at = time.monotonic()
            try:
                # spawn 会先导入本模块；必须在 start 前让子解释器继承 slot 映射。
                if inherited_visibility is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = inherited_visibility
                p.start()
            finally:
                if previous_visibility is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_visibility
            slot.process = p
            slot.state = "starting"
            slot.current_task = None
            slot.task_start_time = None
            slot.child_pid = None
            slot.started_at = spawn_started_at
            # spawn 起点必须早于 p.start，才能覆盖 Python spawn 和 import 开销。
            return p.pid

    def start_loaded_preparations(self, slot_indices):
        """按物理设备放行已完成模块加载的 worker preparation。"""
        started_count = 0
        with self._lock:
            # 同卡 preparation 串行，双卡 slot 必须同时占住其计算与对比设备。
            preparing_gpu_ids = {
                gpu_id
                for slot in self.slots
                if slot.state == "preparing"
                for gpu_id in (slot.gpu_id, slot.comparison_gpu_id)
                if gpu_id is not None
            }
            for slot_index in slot_indices:
                slot = self.slots[slot_index]
                if slot.state != "loaded" or slot.input_queue is None:
                    continue
                slot_gpu_ids = {
                    gpu_id for gpu_id in (slot.gpu_id, slot.comparison_gpu_id) if gpu_id is not None
                }
                if slot_gpu_ids & preparing_gpu_ids:
                    continue
                slot.state = "preparing"
                slot.started_at = time.monotonic()
                try:
                    slot.input_queue.put(WORKER_PREPARE_RUNTIME)
                except (OSError, EOFError, ValueError) as err:
                    # 握手队列损坏时交给统一退役路径，不能让单 slot 失败中止整批。
                    self._suspend_slot(slot)
                    print(
                        f"[worker] PREPARE_DISPATCH_FAILED | slot {slot.index} | "
                        f"{type(err).__name__}: {err}",
                        flush=True,
                    )
                    continue
                preparing_gpu_ids.update(slot_gpu_ids)
                started_count += 1
        return started_count

    def retire_slots(self, slot_indices):
        """停止空闲或启动中的进程；显存是否回收由 GPU 调度器另行确认。"""
        for slot_index in slot_indices:
            slot = self.slots[slot_index]
            with self._lock:
                # 先在锁内定格状态并置为 suspended，watchdog 的 busy/idle 分支
                # 随即不再命中该 slot，避免主动退役被误记为 PADDLE_CRASH。
                was_idle = slot.state == "idle"
                process = slot.process
                input_queue = slot.input_queue
                # child 可能持有主要 CUDA context，必须在 child_pid 被清空前终止。
                self._kill_slot_child(slot)
                # 进程退出只结束生命周期，不能据此宣告物理显存已经恢复。
                slot.input_queue = None
                slot.process = None
                self._suspend_slot(slot)
            # 阻塞式回收放在锁外，避免 watchdog 被 join/SIGKILL 长时间挡住。
            if process is not None and process.is_alive():
                if was_idle and input_queue is not None:
                    try:
                        input_queue.put(None)
                    except (OSError, EOFError, ValueError):
                        # 队列已损坏时跳过优雅退出，下面仍会强制终止 worker。
                        pass
                    else:
                        self._join_process(process, timeout=2)
                if process.is_alive():
                    self._kill_process(process)
            elif process is not None:
                self._join_process(process, timeout=1)
            self._close_queue(input_queue, cancel_join=True)

    def _startup_timeout(self):
        return getattr(self.options, "worker_startup_timeout", WORKER_STARTUP_TIMEOUT)

    def handle_message(self, msg):
        """处理来自 worker 的生命周期、任务确认和 child 账务消息。"""
        # 控制消息只推进 worker 状态；case 终态必须由 batch coordinator 认领。
        msg_type = msg[0]
        if msg_type not in {
            "loaded",
            "ready",
            "init_failed",
            "ack",
            "child",
            "sanitizer_timing",
            "sanitizer_session_ready",
        }:
            return False
        slot_idx = msg[1]
        worker_pid = msg[2]
        late_child_pid = None
        with self._lock:
            slot = self.slots[slot_idx]
            # 共享结果队列可能晚到已回收 worker 的消息，PID 是 slot 生命周期令牌。
            if slot.process is None or slot.process.pid != worker_pid:
                return True
            if msg_type == "loaded":
                if slot.state == "starting":
                    slot.state = "loaded"
                    slot.started_at = None
                return True
            if msg_type == "ready":
                if slot.state == "preparing":
                    slot.state = "idle"
                    slot.started_at = None
                return True
            if msg_type == "ack":
                config = msg[3]
                if slot.state == "busy" and slot.current_task == config:
                    # 超时只依赖经过时长，不能受系统墙上时钟校正影响。
                    slot.task_start_time = time.monotonic()
                return True
            if msg_type == "child":
                # slot 已挂起时忽略迟到登记，否则 wrapper 被杀后 child 会重新进入账本。
                if slot.state in _INITIALIZING_SLOT_STATES or slot.state == "busy":
                    slot.child_pid = msg[3]
                    return True
                # wrapper PID 已校验，迟到 child 仍属于这一代 slot，交给锁外清理。
                late_child_pid = msg[3]
            if msg_type == "sanitizer_timing":
                return True
            if msg_type == "sanitizer_session_ready":
                return True

            if msg_type == "init_failed":
                phase = msg[3]
                error_msg = msg[4]
                expected_state = "starting" if phase == "module_load" else "preparing"
                if slot.state != expected_state:
                    return True
                process = slot.process
                child_pid = slot.child_pid
                self._suspend_slot(slot)

        if late_child_pid is not None:
            self._kill_process_group(late_child_pid)
            return True
        if msg_type == "init_failed":
            print(
                f"[worker] INIT_FAILED | slot {slot_idx} | phase {phase} | {error_msg}",
                flush=True,
            )
            if child_pid is not None:
                self._kill_process_group(child_pid)
            self._join_process(process, timeout=1)
            # init_failed 常见于 bootstrap OOM，slot 必须退出 planned 启动屏障。
            return True
        return True

    def dispatch(self, slot_index, config):
        """向指定 worker slot 派发任务，并返回该 slot 当前 worker 的 PID token。"""
        slot = self.slots[slot_index]
        accounting_config = config.config if isinstance(config, WorkerTask) else config
        with self._lock:
            previous_state = slot.state
            previous_task = slot.current_task
            previous_start = slot.task_start_time
            slot.current_task = accounting_config
            # 派发时刻即开始计时；ack 只刷新起点，避免 ack 丢失后超时永远不触发。
            slot.task_start_time = time.monotonic()
            slot.state = "busy"
            worker_pid = slot.process.pid if slot.process is not None else None
            input_queue = slot.input_queue
            if input_queue is None or slot.process is None:
                # 退役与派发必须互斥；缺少队列表示 slot 已失效，不能留下一个
                # 已计入 active_tasks 却永远收不到终态的幽灵任务。
                slot.current_task = None
                slot.task_start_time = None
                slot.state = "suspended"
                raise RuntimeError(f"slot {slot_index} is not dispatchable")
            try:
                # 投递也在同一把锁内，避免 retire_slots 清空并关闭旧队列后才 put。
                input_queue.put(config)
            except Exception:
                # put 失败必须回滚账本，否则调用方会计入幽灵任务且再也收不到终态。
                slot.current_task = previous_task
                slot.task_start_time = previous_start
                slot.state = previous_state
                raise
            return worker_pid

    def collect_one(self, timeout=5.0):
        """从 result_queue 取一条消息，超时则返回 None。"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def idle_slot_indices(self):
        """返回当前空闲 slot 的稳定快照，调用方不接触 WorkerSlot。"""
        with self._lock:
            idle_indices = tuple(slot.index for slot in self.slots if slot.state == "idle")
        return iter(idle_indices)

    def suspended_slot_indices(self):
        # 返回索引快照，避免调用方遍历 watchdog 正在更新的 slot。
        with self._lock:
            suspended_indices = tuple(
                slot.index for slot in self.slots if slot.state == "suspended"
            )
        return suspended_indices

    def mark_idle(self, slot_index, *, worker_pid):
        """仅在终态仍属于当前存活 worker 时把 busy slot 标记为空闲。"""
        with self._lock:
            slot = self.slots[slot_index]
            if (
                slot.state != "busy"
                or slot.process is None
                or slot.process.pid != worker_pid
                or not slot.process.is_alive()
            ):
                # watchdog 已挂起或 slot 已换代时，迟到结果不能复活旧进程。
                return False
            slot.state = "idle"
            slot.current_task = None
            slot.task_start_time = None
            slot.child_pid = None
            slot.started_at = None
            return True

    def _watchdog_loop(self):
        """周期性检查超时和非预期死亡的 worker。"""
        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(1.0):
                break
            self._watchdog_pass(now=time.monotonic())

    def _watchdog_pass(self, *, now):
        """遍历所有 slot 做一轮检查，单个 slot 的失败不影响其余 slot。"""
        for slot in self.slots:
            if self._shutdown_event.is_set():
                break
            try:
                self._watchdog_tick_slot(slot, now=now)
            except Exception as err:
                # watchdog 是超时与崩溃检测的唯一来源。线程一旦退出，挂死的 case
                # 会让 active_tasks 永远回不到 0，连 PRESSURE_TIMEOUT 都不会触发，
                # 因此任何单 slot 异常都必须就地记录并继续。
                if self._closed or self._shutdown_event.is_set():
                    # 关闭期间队列已释放，异常属于预期竞态，不再刷屏。
                    continue
                try:
                    print(
                        f"[worker] WATCHDOG_ERROR | slot {slot.index} | "
                        f"{type(err).__name__}: {err}",
                        flush=True,
                    )
                except Exception:
                    # stdout 断开或解释器退出时，诊断失败也不能终止检测线程。
                    pass

    def _watchdog_tick_slot(self, slot, *, now):
        """只在锁内做快照，进程终止和 join 必须在锁外执行。"""
        action = None
        with self._lock:
            if self._shutdown_event.is_set():
                return
            if slot.state in _INITIALIZING_SLOT_STATES:
                action = "initializing"
            elif (
                slot.state == "busy"
                and slot.task_start_time is not None
                and now - slot.task_start_time > self.options.timeout
            ):
                action = "timeout"
            elif (
                slot.state in ("busy", "idle")
                and slot.process is not None
                and not slot.process.is_alive()
            ):
                action = "crash"
        # watchdog 的锁只保护快照；SIGKILL/join 放锁外避免阻塞派发和结果处理。
        if action is not None:
            self._handle_lifecycle_event(slot, action, now=now)

    def _handle_lifecycle_event(self, slot, event, *, now):
        """统一处理 worker 初始化失败、超时和异常死亡。"""
        # 先在锁内摘除 slot 令牌，再在锁外执行阻塞式进程回收。
        with self._lock:
            if self._closed or self._shutdown_event.is_set():
                return False
            if event == "initializing":
                if slot.state not in _INITIALIZING_SLOT_STATES or slot.process is None:
                    return False
                phase = "module_load" if slot.state == "starting" else "preparation"
                process = slot.process
                if not process.is_alive():
                    failure = "crash"
                elif slot.state == "loaded" or slot.started_at is None:
                    # loaded 表示 worker 正等待主进程按设备调度，不应消耗 preparation 超时。
                    return False
                elif now - slot.started_at < self._startup_timeout():
                    return False
                else:
                    failure = "timeout"
                child_pid = slot.child_pid
                self._suspend_slot(slot)
            elif event == "timeout":
                if slot.state != "busy" or slot.process is None:
                    return False
                config = slot.current_task
                process = slot.process
                old_pid = process.pid
                child_pid = slot.child_pid
                terminal_timestamp = float(now)
                self._suspend_slot(slot)
            elif event == "crash":
                if (
                    slot.state not in {"busy", "idle"}
                    or slot.process is None
                    or slot.process.is_alive()
                ):
                    return False
                process = slot.process
                exitcode = process.exitcode
                worker_pid = process.pid
                config = slot.current_task
                terminal_timestamp = float(now)
                child_pid = slot.child_pid
                self._suspend_slot(slot)
            else:
                raise ValueError(f"unknown worker lifecycle event: {event!r}")

        if child_pid is not None:
            self._kill_process_group(child_pid)
        if event == "initializing":
            if failure == "crash":
                print(
                    f"[worker] INIT_CRASH | slot {slot.index} | phase {phase} | "
                    f"exit {process.exitcode}",
                    flush=True,
                )
                self._join_process(process, timeout=1)
            else:
                print(
                    f"[worker] INIT_TIMEOUT | slot {slot.index} | phase {phase} | "
                    f"timeout {self._startup_timeout()} s",
                    flush=True,
                )
                self._kill_process(process)
            # 不在 watchdog 内重建；下一轮先取消 assignment 并观察显存稳定。
            return True
        if event == "timeout":
            self._kill_process(process)
            if self._closed or self._shutdown_event.is_set():
                return False
            # watchdog 只上报候选终态，日志边界必须等主循环成功认领后再写。
            self.result_queue.put(
                ("timeout", slot.index, config, old_pid, None, terminal_timestamp)
            )
            return True
        if self._closed or self._shutdown_event.is_set():
            return False
        if config is not None:
            # 与 timeout 相同，crash 日志边界只能由胜出的终态消息写入。
            self.result_queue.put(
                (
                    "crashed",
                    slot.index,
                    config,
                    exitcode,
                    worker_pid,
                    None,
                    terminal_timestamp,
                )
            )
        else:
            print(
                f"[worker] PADDLE_CRASH | slot {slot.index} | exit {exitcode}",
                flush=True,
            )
        return True

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
                    # wrapper 正常退出也可能来不及执行自身 finally，
                    # child 的回收不能依赖 wrapper 的存活状态。
                    self._kill_slot_child(slot)
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
    # 厂商探测工具失控时必须让设备探测回退到下一个后端。
    return subprocess.check_output([command], text=True, stderr=subprocess.STDOUT, timeout=5)


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
    global DEVICE_TYPE, DEVICE_TYPE_DETECTED
    if DEVICE_TYPE_DETECTED:
        return DEVICE_TYPE

    # 探测顺序决定运行后端优先级：NVIDIA GPU > XPU > Iluvatar > CPU。
    try:
        _ensure_nvml()
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
        # 厂商命令缺失或异常时继续尝试下一个后端，最终回退 CPU。
        if not shutil.which(command):
            continue
        try:
            has_device = _count_smi_devices(command, device_pattern) > 0
        except Exception:
            has_device = False
        if has_device:
            DEVICE_TYPE = device_type
            DEVICE_TYPE_DETECTED = True
            return DEVICE_TYPE

    DEVICE_TYPE = "cpu"
    DEVICE_TYPE_DETECTED = True
    return DEVICE_TYPE


def get_device_count() -> int:
    """获取可用设备（加速器）数量。"""
    global DEVICE_COUNT
    if DEVICE_COUNT is not None:
        return DEVICE_COUNT

    device_type = detect_device_type()
    if device_type == "gpu":
        _ensure_nvml()
        DEVICE_COUNT = pynvml.nvmlDeviceGetCount()
    elif device_type == "xpu":
        DEVICE_COUNT = _count_smi_devices(
            XPU_SMI_COMMAND,
            XPU_SMI_DEVICE_PATTERN,
            stop_at_processes=True,
        )
    elif device_type == "iluvatar_gpu":
        DEVICE_COUNT = _count_smi_devices(ILUVATAR_SMI_COMMAND, ILUVATAR_SMI_DEVICE_PATTERN)
    else:
        DEVICE_COUNT = 0
    return DEVICE_COUNT


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
    total_bytes, used_bytes = _read_device_memory_bytes(gpu_id)
    return total_bytes / _BYTES_PER_GIB, used_bytes / _BYTES_PER_GIB


def _ensure_nvml():
    global _NVML_INITIALIZED
    if not _NVML_INITIALIZED:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True


def _read_device_memory_bytes(gpu_id):
    """读取设备显存，单位字节，避免 GB 往返带来的取整误差。"""
    device_type = detect_device_type()
    if device_type == "gpu":
        _ensure_nvml()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem_info.total), int(mem_info.used)
    if device_type in ("xpu", "iluvatar_gpu"):
        _refresh_snapshot(device_type)
        if _MEM_SNAPSHOT is None or gpu_id not in _MEM_SNAPSHOT:
            raise RuntimeError(f"Failed to get memory info for {device_type} device {gpu_id}")
        total_gib, used_gib = _MEM_SNAPSHOT[gpu_id]
        return int(total_gib * _BYTES_PER_GIB), int(used_gib * _BYTES_PER_GIB)
    raise RuntimeError("No supported accelerator (GPU / XPU / Iluvatar) detected.")


def _read_gpu_memory_snapshots(gpu_ids):
    snapshots = {}
    for gpu_id in gpu_ids:
        # 始终重新读取物理 used；外部进程和退出进程的遗留显存都会反映在 free。
        total_bytes, used_bytes = _read_device_memory_bytes(gpu_id)
        snapshots[gpu_id] = GpuMemorySnapshot(
            total_bytes=total_bytes,
            free_bytes=max(0, total_bytes - used_bytes),
        )
    return snapshots


def _memoized_snapshot_reader(snapshot_reader):
    """把一轮调度内的显存采样收敛为每设备一次。

    同一轮内不同阶段（reclaim 观察、规划、压力日志）本来就应该看到
    一致的快照；重复 NVML 查询既浪费又会让同轮决策基于互相矛盾的 free 值。
    派发确认必须另取真实采样，不能复用这份缓存。
    """
    cache = {}

    def read(gpu_ids):
        # 快照读取器的输入保持 tuple，避免缓存层改变既有调用方的参数契约。
        gpu_ids = tuple(gpu_ids)
        missing = tuple(gpu_id for gpu_id in gpu_ids if gpu_id not in cache)
        if missing:
            cache.update(snapshot_reader(missing))
        return {gpu_id: cache[gpu_id] for gpu_id in gpu_ids}

    return read


def _build_gpu_total_memory_map(available_gpus):
    gpu_total_memory_map = {}
    for gpu_id in available_gpus:
        try:
            gpu_total_memory_map[gpu_id] = _read_device_memory_bytes(gpu_id)[0] / _BYTES_PER_GIB
        except Exception:
            pass
    return gpu_total_memory_map


ARGUMENT_ERROR_PREFIX = "[argument error]"
ARGUMENT_WARNING_PREFIX = "[argument warning]"
TEST_MODE_ERROR = (
    "specify exactly one test mode: --accuracy, --paddle_only, --paddle_cinn, "
    "--paddle_gpu_performance, --torch_gpu_performance, "
    "--paddle_torch_gpu_performance, --accuracy_stable, "
    "--accuracy_dual_gpu, --accuracy_stable_dual_gpu, "
    "--paddle_custom_device, --custom_device_vs_gpu"
)


def _argument_error(message):
    print(f"{ARGUMENT_ERROR_PREFIX} {message}", flush=True)
    return 2


def _select_test_class(options):
    import tester

    class_name = next(
        (
            class_name
            for option, class_name in TEST_CLASS_BY_OPTION
            if _option_enabled(options, option)
        ),
        "APITestAccuracy",
    )
    return getattr(tester, class_name)


def _clear_device_cache(options):
    import paddle

    if any(_option_enabled(options, opt) for opt in TORCH_UTILITY_MODES):
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


def _dual_gpu_mode_enabled(options):
    """双卡判断只属于引擎资源协议，不承载 tester 的显存治理策略。"""
    return _option_enabled(options, "accuracy_dual_gpu") or _option_enabled(
        options, "accuracy_stable_dual_gpu"
    )


def _normalize_dual_gpu_option(options, option_name, base_mode):
    if not _option_enabled(options, option_name):
        return
    # 先把组合开关展开为既有主模式，再进入统一的模式互斥校验。
    # dual 标志本身不作为第二个主模式重复计数。
    if not _option_enabled(options, "use_gpu_mode"):
        print(
            f"{ARGUMENT_WARNING_PREFIX} "
            f"--{option_name}=True implies --use_gpu_mode=True; enabling GPU mode",
            flush=True,
        )
        options.use_gpu_mode = True
    setattr(options, base_mode, True)


def normalize_dual_gpu_options(options):
    _normalize_dual_gpu_option(options, "accuracy_dual_gpu", "accuracy")
    _normalize_dual_gpu_option(options, "accuracy_stable_dual_gpu", "accuracy_stable")


def _mode_runs_torch_gpu_reference(options):
    """只有真实执行 Torch reference 的模式才要求为其保留计算卡。"""
    # Paddle/CINN 等仅借用 Torch 工具的模式不属于 reference GPU 执行。
    return any(_option_enabled(options, mode) for mode in TORCH_REFERENCE_MODES)


def _requires_gpu_runtime(options):
    """GPU 算子或 GPU 生成/比较任一启用时，都必须准备 GPU 运行时。"""
    # test_cpu 只移走 Paddle kernel；Torch reference、GPU 生成和比较仍各自需要 GPU。
    return bool(
        not _option_enabled(options, "test_cpu")
        or _option_enabled(options, "use_gpu_mode")
        or _mode_runs_torch_gpu_reference(options)
    )


def validate_gpu_options(options) -> tuple:
    """校验并规范化 GPU 相关参数。"""
    normalize_dual_gpu_options(options)
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
    if _dual_gpu_mode_enabled(options):
        # 一对卡是不可拆分的 worker slot，禁止共享其中任意一张卡。
        if options.num_gpus < 2 or options.num_gpus % 2:
            raise ValueError("dual-GPU accuracy modes require an even --num_gpus")
        if options.num_workers_per_gpu != 1:
            raise ValueError("dual-GPU accuracy modes require --num_workers_per_gpu=1")
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
        # 双卡单 case 必须形成完整 pair，不能沿用普通模式的一卡默认值。
        if _dual_gpu_mode_enabled(options):
            options.gpu_ids = "0,1"
            options.num_gpus = 2
        else:
            options.gpu_ids = "0"
            options.num_gpus = 1


def _prepare_single_config_gpu(options):
    normalize_dual_gpu_options(options)
    if not _requires_gpu_runtime(options):
        options.gpu_workers_per_gpu_map = {}
        options.gpu_total_memory_map = {}
        return None

    _apply_single_config_gpu_defaults(options)
    gpu_ids = validate_gpu_options(options)
    expected_gpu_count = 2 if _dual_gpu_mode_enabled(options) else 1
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
    cmd = _build_sanitizer_session_command(options, sanitizer_cmd)
    env = os.environ.copy()
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)

    session = SanitizerSession(cmd, env)
    raw_lines = []
    status = "crashed"
    returncode = -1
    with tempfile.TemporaryDirectory(prefix="sanitizer_single_") as timing_dir:
        timing_path = Path(timing_dir) / "timing.tsv"
        try:
            session.start(on_output=raw_lines.append)
            request_result = session.run_request(
                0,
                api_config,
                timing_path,
                on_output=raw_lines.append,
            )
            status = request_result.status
            returncode = request_result.returncode
            if request_result.diagnostic:
                raw_lines.append(request_result.diagnostic)
        except (EOFError, OSError, ValueError) as err:
            raw_lines.append(f"[sanitizer session] {err}\n")
            status = "crashed"
            returncode = -1
        finally:
            session.close(wait=True)

    raw_output = "".join(raw_lines)
    if status == "done" and _sanitizer_session_output_has_report(raw_output):
        returncode = options.sanitizer_error_exitcode
    analysis = analyze_sanitizer_output(raw_output, returncode, options.sanitizer_error_exitcode)
    if analysis.output:
        print(
            analysis.output,
            end="" if analysis.output.endswith("\n") else "\n",
            flush=True,
        )
    if _sanitizer_analysis_is_ignored(analysis):
        return 0
    normalized_returncode = _normalize_sanitizer_exitcode(
        raw_output,
        returncode=returncode,
        sanitizer_error_exitcode=options.sanitizer_error_exitcode,
    )
    if returncode == options.sanitizer_error_exitcode:
        print(
            f"[error] compute-sanitizer reported errors for {api_config} (exit {returncode})",
            flush=True,
        )
    return normalized_returncode


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
        pending_dispatch.appendleft(_pending_case_for_retry(retry_state, api_config_str))
        return True

    retry_state.unsafe_environment = True
    return False


def _pending_case_for_retry(retry_state, config, *, ready_at=0.0):
    estimate = retry_state.case_memory_estimates.get(config, CaseGpuEstimate())
    return PendingCase(
        config=config,
        ready_at=ready_at,
        compute_estimate_bytes=estimate.compute_bytes,
        comparison_estimate_bytes=estimate.comparison_bytes,
    )


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


def _resolve_cpu_worker_count(options, pending_cases):
    """纯 CPU batch 沿用 worker 数参数，但不执行任何 GPU 探测或绑定。"""
    requested = 1 if options.num_workers_per_gpu == -1 else options.num_workers_per_gpu
    if requested <= 0:
        raise ValueError("--num_workers_per_gpu must be -1 or a positive integer")
    return min(requested, pending_cases, MAX_TOTAL_WORKERS)


def _fill_idle_workers(pool, pending_dispatch, *, on_dispatch):
    """用 pending 队列补满空闲 worker；首次任务与重试任务遵循同一顺序。"""
    dispatched_count = 0
    # 重试任务优先于新任务，保证外部 kill 或显存 defer 不会长期饥饿。
    while True:
        slot_index = next(pool.idle_slot_indices(), None)
        if slot_index is None:
            break
        config = pending_dispatch.pop_ready()
        if config is None:
            break
        try:
            # dispatch 在状态锁内读取 PID，作为终态认领的 slot 生命周期令牌。
            worker_pid = pool.dispatch(slot_index, config)
        except Exception as err:
            # 派发失败必须把 case 还回队首，否则这条配置会从批次中消失。
            pending_dispatch.appendleft(PendingCase(config=config))
            print(
                f"[worker] DISPATCH_FAILED | slot {slot_index} | {type(err).__name__}: {err}",
                flush=True,
            )
            break
        on_dispatch(slot_index, worker_pid, config)
        dispatched_count += 1
    return dispatched_count


def _estimate_case_gpu_memory(api_config_str, options, *, failure_report):
    mode = next(
        (mode for mode in GPU_MEMORY_PREFLIGHT_MODES if _option_enabled(options, mode)),
        None,
    )
    if mode is None:
        return CaseGpuEstimate()
    try:
        from tester import APIConfig

        api_config = APIConfig(api_config_str)
        runtime_config = options.runtime_config
        estimate = estimate_gpu_memory(
            api_config,
            mode,
            check_grad=should_check_grad(api_config),
            input_backend=runtime_config.input_backend_resolved,
            input_source_on_gpu=(
                runtime_config.input_backend_resolved != "numpy"
                and runtime_config.input_logical_device != "cpu"
            ),
            # 执行设备分别建模，不能用 use_gpu_mode 代替 Paddle/Torch 的真实 place。
            paddle_kernel_on_gpu=not options.test_cpu,
            torch_operator_on_gpu=_mode_runs_torch_gpu_reference(options),
        )
    except Exception as err:
        # 配置解析和不支持 dtype 仍由 worker 分类；调度器使用最小准入成本兜底，
        # 但兜底必须留下诊断，否则无法区分“无模型”与“估算器故障”。
        failure_report.record(api_config_str, err)
        return CaseGpuEstimate()
    compute_bytes = max(
        # 多阶段执行采用峰值而非求和；阶段之间不会同时持有全部临时张量。
        (stage.total_bytes for stage in estimate.stages if stage.device == "compute"),
        default=0,
    )
    comparison_bytes = max(
        (stage.total_bytes for stage in estimate.stages if stage.device == "comparison"),
        default=0,
    )
    return CaseGpuEstimate(compute_bytes, comparison_bytes)


def _build_pending_cases(api_configs, options, *, failure_report):
    pending_cases = PendingQueue()
    for config in api_configs:
        estimate = _estimate_case_gpu_memory(config, options, failure_report=failure_report)
        pending_cases.append(
            PendingCase(
                config=config,
                compute_estimate_bytes=estimate.compute_bytes,
                comparison_estimate_bytes=estimate.comparison_bytes,
            ),
            now=0.0,
        )
    return pending_cases


class BatchTerminalCoordinator:
    """统一认领并结算 CPU/GPU worker 的 case 终态。"""

    def __init__(
        self,
        *,
        pool,
        options,
        all_case,
        checkpointed_case,
        batch_state,
        retry_state,
        pending_dispatch,
        max_total_external_kills,
    ):
        self.pool = pool
        self.options = options
        self.all_case = all_case
        self.checkpointed_case = checkpointed_case
        self.batch_state = batch_state
        self.retry_state = retry_state
        self.pending_dispatch = pending_dispatch
        self.max_total_external_kills = max_total_external_kills
        # CPU/GPU 共用同一令牌协议，避免 done 与 watchdog 终态重复结算。
        self.claims = TerminalClaims()

    def register(self, slot_index, worker_pid, config):
        # dispatch 成功后才注册；排队中的 case 不能拥有终态令牌。
        self.claims.register(slot_index, worker_pid, config)

    def claim(self, msg):
        # claim 是日志、retry 和 checkpoint 之前的唯一幂等门。
        return self.claims.claim(msg)

    def process(self, msg):
        # tested_case 的变化代表产生了可 checkpoint 的业务终态。
        before = self.batch_state.tested_case
        _handle_batch_result(
            pool=self.pool,
            options=self.options,
            all_case=self.all_case,
            checkpointed_case=self.checkpointed_case,
            batch_state=self.batch_state,
            retry_state=self.retry_state,
            pending_dispatch=self.pending_dispatch,
            msg=msg,
            max_total_external_kills=self.max_total_external_kills,
        )
        return self.batch_state.tested_case > before


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

    if (
        msg_type in {"timeout", "crashed"}
        and crash_source == "worker"
        and message.worker_pid is not None
        and message.completed_offset is None
    ):
        # 认领完成后才补 abrupt exit 日志，迟到的竞争终态不会产生第二个边界。
        message.completed_offset = log_worker.append_case_end_to_worker_log(
            message.worker_pid,
            msg_type,
            api_config_str=config,
        )
    if message.worker_pid is not None:
        log_aggregation.mark_inorder_case_complete(
            message.worker_pid,
            message.completed_offset,
        )

    worker_reusable = msg_type in ("done", "error") or (
        msg_type == "crashed" and options.use_compute_sanitizer and crash_source == "session"
    )
    external_kill = msg_type == "crashed" and exitcode in (
        -signal.SIGKILL,
        -signal.SIGTERM,
    )
    batch_state.active_tasks -= 1

    if external_kill:
        # external kill 同时是 crash 样本和可能的 retry 触发原因。
        if _handle_external_kill_retry(
            retry_state,
            pending_dispatch,
            config,
            max_case_retries=MAX_EXTERNAL_KILL_RETRIES_PER_CASE,
            max_total_external_kills=max_total_external_kills,
        ):
            log_report.print_case_notice("RETRY", config, f"exit {exitcode}")
            if worker_reusable:
                pool.mark_idle(slot_index, worker_pid=message.worker_pid)
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
        pool.mark_idle(slot_index, worker_pid=message.worker_pid)
        if msg_type != "deferred":
            retry_state.slot_memory_defer_retries.pop(slot_index, None)

    if msg_type == "deferred":
        # deferred 是显存准入退避，不属于 external-kill retry 计数。
        # 首次 deferred 保留常驻 worker，只有连续失败才退役并强制回收 context。
        pool.mark_idle(slot_index, worker_pid=message.worker_pid)
        defer_count = retry_state.slot_memory_defer_retries.get(slot_index, 0) + 1
        retry_state.slot_memory_defer_retries[slot_index] = defer_count
        if defer_count >= GPU_MEMORY_DEFER_RETIRE_AFTER:
            pool.retire_slots((slot_index,))
            retry_state.slot_memory_defer_retries.pop(slot_index, None)
        retry_count = retry_state.per_case_memory_defer_retries.get(config, 0)
        retry_state.per_case_memory_defer_retries[config] = retry_count + 1
        delay = _memory_defer_delay(retry_count)
        pending_dispatch.append(
            _pending_case_for_retry(
                retry_state,
                config,
                ready_at=time.monotonic() + delay,
            )
        )
        log_report.print_case_notice(
            "DEFER",
            config,
            f"retry {retry_count + 1} in {delay:.1f}s | {reason}",
        )
        return

    batch_state.tested_case += 1
    progress_status = "DONE"
    progress_detail = None

    if msg_type == "timeout":
        log_worker.write_to_log("timeout", config)
        progress_status = "TIMEOUT"
    elif msg_type == "crashed":
        # 只有已认领的 crash 才写终态，避免 watchdog/终态竞争双计数。
        log_type, progress_status, terminal_recorded = log_worker.classify_exit(exitcode)
        if crash_source == "session":
            terminal_recorded = False
        if progress_status == "PADDLE_CRASH":
            progress_detail = f"exit {exitcode}"
        if not terminal_recorded:
            log_worker.write_to_log(log_type, config)
    elif msg_type == "error":
        # worker 的 error 通道覆盖配置解析失败和未分类异常，不能一律记成 parse。
        log_type = "config_parse" if "APIConfig" in (reason or "") else "paddle_error"
        log_worker.write_to_log(log_type, config)
        progress_status = "CONFIG_PARSE" if log_type == "config_parse" else "PADDLE_ERROR"
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


def _run_cpu_batch_loop(
    pool,
    options,
    api_configs,
    all_case,
    checkpointed_case,
    batch_state,
    retry_state,
    max_total_external_kills,
):
    """运行不依赖 GPU 准入和显存回收状态机的纯 CPU 批处理。"""
    ready_workers = pool.warmup_cpu_workers()
    if ready_workers != pool.total_workers:
        print(
            "Workers: failed | CPU startup barrier incomplete; no cases will be dispatched",
            flush=True,
        )
        batch_state.batch_exit_code = 1
        batch_state.shutdown_force = True
        batch_state.abort_run = True
        return

    pending_dispatch = PendingQueue(PendingCase(config) for config in api_configs)
    terminal = BatchTerminalCoordinator(
        pool=pool,
        options=options,
        all_case=all_case,
        checkpointed_case=checkpointed_case,
        batch_state=batch_state,
        retry_state=retry_state,
        pending_dispatch=pending_dispatch,
        max_total_external_kills=max_total_external_kills,
    )

    def refill_idle_workers():
        if batch_state.abort_run:
            return
        dispatched = _fill_idle_workers(
            pool,
            pending_dispatch,
            on_dispatch=terminal.register,
        )
        batch_state.active_tasks += dispatched
        if dispatched and batch_state.test_started_at is None:
            batch_state.test_started_at = time.monotonic()
            batch_state.last_forecast_at = batch_state.test_started_at

    refill_idle_workers()
    # 初始化中 slot 纳入循环存活条件，避免重建进程的生命周期消息无人接收。
    while (
        batch_state.active_tasks > 0 or pending_dispatch or pool.has_initializing_slots()
    ) and not batch_state.abort_run:
        msg = pool.collect_one(timeout=_collect_wait_timeout(pending_dispatch))
        if msg is not None and not pool.handle_message(msg):
            # 控制消息只改变 slot 生命周期；case 终态才允许推进计数和 checkpoint。
            # 认领失败的迟到/重复终态没有任何批次副作用。
            if msg[0] in ("done", "deferred", "error", "timeout", "crashed") and terminal.claim(
                msg
            ):
                terminal.process(msg)

        # timeout、crash 和 deferred 会挂起原 slot；纯 CPU 可直接重建，无需等待显存回收。
        if pending_dispatch and not batch_state.abort_run:
            for slot_index in pool.suspended_slot_indices():
                pool.start_worker(slot_index)
        # CPU replacement 不再参与首次全量屏障，模块加载完成后即可独立 preparation。
        pool.start_loaded_preparations(range(pool.total_workers))
        refill_idle_workers()


@dataclass
class ContinuousAssignment:
    # assignment 在 reservation、worker PID 和业务 case 间建立唯一关联。
    assignment_id: int
    group_key: tuple[int, ...]
    slot_index: int
    pending: PendingCase
    reservation: GpuReservation
    worker_pid: int | None = None
    dispatched: bool = False


class ContinuousGpuBatchScheduler:
    """聚合 GPU group 的准入、worker 启动、派发和终态认领。"""

    def __init__(self, pool, snapshots):
        # 相同设备拓扑共享账本；不同双卡 pair 之间不能互相扣额度。
        grouped_slots = {}
        for slot_index, gpu_id, comparison_gpu_id in pool.slot_devices():
            device_ids = (gpu_id, comparison_gpu_id) if comparison_gpu_id is not None else (gpu_id,)
            grouped_slots.setdefault(device_ids, []).append(slot_index)
        self.groups = {
            device_ids: {
                "ledger": GpuReservationLedger(
                    device_ids,
                    snapshots={gpu_id: snapshots[gpu_id] for gpu_id in device_ids},
                    max_workers=len(slot_indices),
                ),
                "slot_indices": tuple(slot_indices),
                "last_extended_scan": 0.0,
            }
            for device_ids, slot_indices in grouped_slots.items()
        }
        self.pool = pool
        # assignments 覆盖 reservation 到结果结算的完整生命周期。
        self.assignments = {}
        # running 映射阻止 slot 复用后的迟到终态认领新 assignment。
        self._running_assignments = {}
        self._next_assignment_id = 1

    def has_assignments(self):
        # 未派发 reservation 也属于活跃 assignment，批次不能提前退出。
        return bool(self.assignments)

    def cancel_failed_startup(self, pending_dispatch):
        # 初始化失败只回队当前 slot，其他在途 assignment 仍可继续完成。
        for assignment_id, assignment in tuple(self.assignments.items()):
            if assignment.dispatched or not self.pool.slot_can_start(assignment.slot_index):
                continue
            self.groups[assignment.group_key]["ledger"].mark_release_pending(assignment_id)
            # 未形成业务终态的 assignment 必须回队，不能只释放 reservation。
            pending_dispatch.appendleft(assignment.pending)
            del self.assignments[assignment_id]

    def reclaim(self, snapshots_reader, *, now):
        # release_pending 期间只观察物理回收，不创建替代 worker。
        # 物理显存稳定前不能用 worker 退出事件提前释放 lease。
        for group_key, group in self.groups.items():
            group["ledger"].advance_reclaim(snapshots_reader(group_key), now=now)

    def schedule(self, pending_dispatch, *, snapshots_reader, now):
        """按 group 批量规划可准入 case，并保留 pending 队列顺序。"""
        # reservation 建立后 assignment 才离开 pending 队列。
        scheduled = 0
        for group_key, group in self.groups.items():
            ledger = group["ledger"]
            ledger.update_snapshots(snapshots_reader(group_key))
            idle_slots = [
                slot_index
                for slot_index in group["slot_indices"]
                if self.pool.slot_can_schedule(slot_index)
                and not any(
                    assignment.slot_index == slot_index for assignment in self.assignments.values()
                )
            ]
            if not idle_slots or not pending_dispatch:
                continue

            base_limit = max(
                CANDIDATE_WINDOW_MIN,
                len(group["slot_indices"]) * CANDIDATE_WINDOW_PER_WORKER,
            )
            candidates = pending_dispatch.candidate_window(base_limit, now=now)
            extended = []
            selected_indices = []
            selected_configs = set()
            planned = []

            def plan_from_window(window, offset=0):
                # 已规划 case 留在内存集合中，直到整组规划结束才从队列删除。
                nonlocal scheduled
                for slot_index in idle_slots:
                    if any(item[0] == slot_index for item in planned):
                        continue
                    for local_index, pending in enumerate(window):
                        if pending.config in selected_configs:
                            continue
                        assignment_id = self._next_assignment_id
                        reservation = ledger.reserve(
                            assignment_id=assignment_id,
                            slot_index=slot_index,
                            estimates=pending.gpu_estimate,
                        )
                        if reservation is None:
                            continue
                        planned.append((slot_index, pending, assignment_id, reservation))
                        selected_indices.append(offset + local_index)
                        selected_configs.add(pending.config)
                        self._next_assignment_id += 1
                        scheduled += 1
                        break

            plan_from_window(candidates)
            if (
                len(planned) < len(idle_slots)
                and pending_dispatch.ready_count > len(candidates)
                and now - group["last_extended_scan"] >= EXTENDED_SCAN_MIN_INTERVAL_SECONDS
            ):
                extended, _ = pending_dispatch.scan_window(
                    EXTENDED_SCAN_MAX_CANDIDATES,
                    cursor=len(candidates),
                    now=now,
                )
                if extended:
                    group["last_extended_scan"] = now
                    # scan_window 从 base_limit 之后开始，合并后索引可一次删除。
                    plan_from_window(extended, offset=len(candidates))

            if not planned:
                continue
            selected_cases = pending_dispatch.take_case_selection(
                candidates + extended,
                selected_indices,
            )
            selected_by_config = {case.config for case in selected_cases}
            for slot_index, pending, assignment_id, reservation in planned:
                if pending.config not in selected_by_config:
                    # 队列被外部路径修改时撤销孤立 reservation，不能留下幽灵 lease。
                    ledger.mark_release_pending(assignment_id)
                    scheduled -= 1
                    continue
                self.assignments[assignment_id] = ContinuousAssignment(
                    assignment_id,
                    group_key,
                    slot_index,
                    pending,
                    reservation,
                )
        return scheduled

    def start_workers(self):
        """只为已有 reservation 的 assignment 启动 worker。"""
        # spawn 不参与显存决策，只消费 schedule 已建立的准入承诺。
        for assignment in self.assignments.values():
            # 已有 PID 表示 spawn 已提交，重复启动会破坏 slot 的生命周期令牌。
            if assignment.dispatched or assignment.worker_pid is not None:
                continue
            if not self.pool.slot_can_start(assignment.slot_index):
                continue
            worker_pid = self.pool.start_worker(assignment.slot_index)
            if worker_pid is not None:
                assignment.worker_pid = worker_pid
            else:
                # 关闭或 spawn 失败时交给统一的 startup 取消路径回队。
                self.groups[assignment.group_key]["ledger"].mark_release_pending(
                    assignment.assignment_id
                )

    def dispatch_ready(self, *, snapshots_reader, terminal_coordinator):
        # 只有 worker ready 且二次显存确认成功后，才把 case 投入 worker 队列。
        dispatched = 0
        for assignment in tuple(self.assignments.values()):
            # tuple 快照避免 dispatch/retire 过程中修改 assignments 影响当前扫描。
            if assignment.dispatched:
                continue
            if assignment.worker_pid is None:
                assignment.worker_pid = self.pool.worker_pid(assignment.slot_index)
            if not self.pool.slot_is_idle(assignment.slot_index) or assignment.worker_pid is None:
                continue
            group = self.groups[assignment.group_key]
            if not group["ledger"].confirm(
                assignment.assignment_id, snapshots_reader(assignment.group_key)
            ):
                # 二次确认失败时 worker 只退役，case 留到下一次稳定回收后重试。
                self.pool.retire_slots((assignment.slot_index,))
                continue
            workers_on_gpu = max(
                1,
                sum(
                    other.group_key == assignment.group_key and other.dispatched
                    for other in self.assignments.values()
                )
                + 1,
            )
            policy = group["ledger"].policy
            dual_gpu = len(assignment.group_key) == 2
            try:
                # 预算传给 worker 后由其本地预检使用，不能只依赖主进程估算。
                worker_pid = self.pool.dispatch(
                    assignment.slot_index,
                    WorkerTask(
                        config=assignment.pending.config,
                        workers_on_gpu=workers_on_gpu,
                        compute_budget_gib=(
                            policy.case_admission_bytes(assignment.pending.compute_estimate_bytes)
                            / _BYTES_PER_GIB
                        ),
                        comparison_budget_gib=(
                            policy.case_admission_bytes(
                                assignment.pending.comparison_estimate_bytes
                            )
                            / _BYTES_PER_GIB
                            if dual_gpu
                            else 0.0
                        ),
                    ),
                )
            except Exception as err:
                # put/queue 失败不能留下 active_tasks 幽灵计数或占用 slot。
                print(
                    f"[gpu] ASSIGNMENT_DISPATCH_FAILED | slot={assignment.slot_index} | "
                    f"{type(err).__name__}: {err}",
                    flush=True,
                )
                group["ledger"].mark_release_pending(assignment.assignment_id)
                self.pool.retire_slots((assignment.slot_index,))
                continue
            assignment.worker_pid = worker_pid
            assignment.dispatched = True
            terminal_coordinator.register(
                assignment.slot_index,
                worker_pid,
                assignment.pending.config,
            )
            self._running_assignments[assignment.slot_index] = assignment.assignment_id
            dispatched += 1
        return dispatched

    def claim_terminal(self, msg, terminal_coordinator):
        # 终态先做 assignment 级认领，再进入日志、checkpoint 和 retry 处理。
        message = BatchMessage.from_raw(msg)
        if message.slot_index is None:
            return None
        assignment = next(
            (
                item
                for item in self.assignments.values()
                if item.slot_index == message.slot_index and item.dispatched
            ),
            None,
        )
        if assignment is None:
            return None
        if self._running_assignments.get(message.slot_index) != assignment.assignment_id:
            return None
        # 通用令牌继续校验 PID/config，两个账本共同阻断跨代消息。
        if not terminal_coordinator.claim(msg):
            return None
        group = self.groups[assignment.group_key]
        if not group["ledger"].claim_terminal(assignment.assignment_id):
            return None
        del self._running_assignments[assignment.slot_index]
        return assignment

    def record_result(self, assignment, msg_type):
        self.groups[assignment.group_key]["ledger"].record_result(msg_type)

    def finish(self, assignment):
        """在批处理结果完成 retry/checkpoint 后移除 assignment。"""
        # 业务结算先于删除，retry 路径仍可读取原 assignment 上下文。
        current = self.assignments.pop(assignment.assignment_id, None)
        if current is not assignment:
            raise RuntimeError(f"assignment {assignment.assignment_id} is not active")


def _run_continuous_gpu_batch_loop(
    *,
    pool,
    options,
    api_configs,
    all_case,
    checkpointed_case,
    batch_state,
    retry_state,
    max_total_external_kills,
):
    # 主循环顺序固定为回收、准入、启动、派发、收终态，避免状态逆向迁移。
    estimate_failures = GpuEstimateFailureReport()
    pending_dispatch = _build_pending_cases(api_configs, options, failure_report=estimate_failures)
    estimate_failures.emit(all_case)
    retry_state.case_memory_estimates = {
        pending.config: pending.gpu_estimate for pending in pending_dispatch.iter_all()
    }
    initial_snapshots = _read_gpu_memory_snapshots(
        tuple(
            gpu_id
            for _, gpu_id, comparison_gpu_id in pool.slot_devices()
            for gpu_id in (gpu_id, comparison_gpu_id)
            if gpu_id is not None
        )
    )
    scheduler = ContinuousGpuBatchScheduler(pool, initial_snapshots)
    terminal = BatchTerminalCoordinator(
        pool=pool,
        options=options,
        all_case=all_case,
        checkpointed_case=checkpointed_case,
        batch_state=batch_state,
        retry_state=retry_state,
        pending_dispatch=pending_dispatch,
        max_total_external_kills=max_total_external_kills,
    )
    # scheduler 以 assignment_id 为主键，避免 slot 复用产生跨代串消息。
    pressure_timeout = GpuPressureTimeout(options.gpu_pressure_timeout)

    while (
        pending_dispatch
        or scheduler.has_assignments()
        or batch_state.active_tasks
        or pool.has_initializing_slots()
    ) and not batch_state.abort_run:
        # 每轮重新采样，不能把上一轮的 free 快照当作当前显存承诺。
        now = time.monotonic()
        planning_snapshots_reader = _memoized_snapshot_reader(_read_gpu_memory_snapshots)
        scheduler.cancel_failed_startup(pending_dispatch)
        scheduler.reclaim(planning_snapshots_reader, now=now)
        scheduler.schedule(
            pending_dispatch,
            snapshots_reader=planning_snapshots_reader,
            now=now,
        )
        scheduler.start_workers()
        pool.start_loaded_preparations(range(pool.total_workers))
        # worker preparation 可能改变显存，confirm 必须使用新的一轮采样。
        confirm_snapshots_reader = _memoized_snapshot_reader(_read_gpu_memory_snapshots)
        dispatched = scheduler.dispatch_ready(
            snapshots_reader=confirm_snapshots_reader,
            terminal_coordinator=terminal,
        )
        if dispatched:
            # active_tasks 只统计已真正 put 到 worker 队列的 case。
            batch_state.active_tasks += dispatched
            if batch_state.test_started_at is None:
                batch_state.test_started_at = now
                batch_state.last_forecast_at = now

        blocked = (
            bool(pending_dispatch)
            and batch_state.active_tasks == 0
            and not scheduler.has_assignments()
            and not pool.has_initializing_slots()
        )
        # startup assignment 和初始化 slot 都属于进展中状态，不应触发压力超时。
        if pressure_timeout.update(blocked=blocked, now=now):
            print(
                f"[gpu] PRESSURE_TIMEOUT | {options.gpu_pressure_timeout:.1f} s without "
                f"checkpoint progress | {len(pending_dispatch)} pending",
                flush=True,
            )
            batch_state.batch_exit_code = 1
            batch_state.abort_run = True
            break

        msg = pool.collect_one(timeout=_collect_wait_timeout(pending_dispatch))
        if msg is None:
            # 没有消息时下一轮仍需处理 watchdog 状态和显存回收。
            continue
        if pool.handle_message(msg):
            continue
        if msg[0] not in {"done", "deferred", "error", "timeout", "crashed"}:
            # 控制消息已在上方处理，未知消息不能改变 active_tasks。
            continue
        assignment = scheduler.claim_terminal(msg, terminal)
        if assignment is None:
            # 迟到或重复终态不应再次写 checkpoint、计数或触发 retry。
            continue
        scheduler.record_result(assignment, msg[0])
        progressed = terminal.process(msg)
        scheduler.finish(assignment)
        # 删除 assignment 必须发生在统一结果处理之后，便于 retry 读取原 case。
        if progressed:
            pressure_timeout.update(blocked=False, now=time.monotonic())
        elif msg[0] == "deferred":
            pressure_timeout.update(blocked=True, now=time.monotonic())


def _run_batch_mode(
    *,
    options,
    api_configs,
    all_case,
    checkpointed_case,
    available_gpus,
    max_workers_per_gpu,
    gpu_total_memory_map,
    cpu_worker_count,
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
        pool = WorkerPool(
            available_gpus,
            max_workers_per_gpu,
            options,
            gpu_total_memory_map=gpu_total_memory_map,
            cpu_worker_count=cpu_worker_count,
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

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signal_handlers[sig] = signal.signal(sig, cleanup_handler)

        print(f"Workers: lazy | {pool.total_workers} logical slots", flush=True)
        pool.start()
        if cpu_worker_count:
            _run_cpu_batch_loop(
                pool,
                options,
                api_configs,
                all_case,
                checkpointed_case,
                batch_state,
                retry_state,
                max_total_external_kills,
            )
            return batch_state.batch_exit_code

        _run_continuous_gpu_batch_loop(
            pool=pool,
            options=options,
            api_configs=api_configs,
            all_case=all_case,
            checkpointed_case=checkpointed_case,
            batch_state=batch_state,
            retry_state=retry_state,
            max_total_external_kills=max_total_external_kills,
        )

    except Exception as e:
        print(f"Unexpected error: {e}", flush=True)
        batch_state.batch_exit_code = 1
        batch_state.shutdown_force = True
        batch_state.abort_run = True
    finally:
        # 在进入 pool 清理前恢复进程级信号处理器。
        for sig, handler in previous_signal_handlers.items():
            signal.signal(sig, handler)
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
    # 纯 CPU 模式不借用虚拟 GPU id，日志和显存预算也保持 CPU 语义。
    visible_gpu_ids = ()
    if _requires_gpu_runtime(options):
        visible_gpu_ids = tuple(
            int(value) for value in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        )
    gpu_id = visible_gpu_ids[0] if visible_gpu_ids else None
    comparison_gpu_id = (
        visible_gpu_ids[1] if _dual_gpu_mode_enabled(options) and len(visible_gpu_ids) > 1 else None
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
    terminal_log_type = log_worker.get_terminal_log_type(api_config_str)
    fatal_log_type = _fatal_log_type_for_error(err, terminal_log_type)
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
    if (
        _requires_gpu_runtime(options)
        and not any(getattr(options, opt) for opt in GPU_PERFORMANCE_MODES)
        and not getattr(options, "use_gpu_mode", False)
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
    enabled_modes = [mode for mode in PRIMARY_TEST_MODES if _option_enabled(options, mode)]
    if len(enabled_modes) != 1:
        return _argument_error(TEST_MODE_ERROR)
    # GPU 性能和自定义设备模式具有固定设备协议，不能被 test_cpu 改写为 CPU kernel。
    # 对矛盾组合在参数层 fail fast，避免 worker 启动后才触发硬编码 CUDA 同步错误。
    # accuracy/stable/paddle_only/CINN 未列入此集合，继续支持正交的四种设备组合。
    cpu_incompatible_modes = GPU_PERFORMANCE_MODES + (
        "paddle_custom_device",
        "custom_device_vs_gpu",
    )
    if options.test_cpu and any(getattr(options, mode, False) for mode in cpu_incompatible_modes):
        return _argument_error(
            "--test_cpu=True is incompatible with GPU performance and custom-device modes"
        )
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
    cache_requested = bool(options.use_cached_numpy)
    requested_backend = (
        (
            getattr(options, "input_backend_requested", None)
            or os.environ.get("PADDLEAPITEST_INPUT_BACKEND")
            or ""
        )
        .strip()
        .lower()
    )
    # 警告只在主进程规范化阶段输出，避免每个 spawn worker 重复报告同一参数冲突。
    if options.use_gpu_mode and cache_requested:
        print(
            f"{ARGUMENT_WARNING_PREFIX} --use_cached_numpy=True is ignored when "
            "--use_gpu_mode=True",
            flush=True,
        )
        options.use_cached_numpy = False
    elif cache_requested and requested_backend in {"torch", "paddle"}:
        print(
            f"{ARGUMENT_WARNING_PREFIX} PADDLEAPITEST_INPUT_BACKEND={requested_backend} is "
            "ignored when --use_cached_numpy=True; using NumPy backend",
            flush=True,
        )
    # 环境变量传播规范化后的有效 cache 状态，worker 不再自行解释冲突组合。
    os.environ["USE_CACHED_NUMPY"] = str(options.use_cached_numpy)
    os.environ["USE_GPU_MODE"] = str(options.use_gpu_mode)
    # 主进程只解析一次终态，spawn worker、预检和具体 tester 共享同一份策略。
    options.runtime_config = TestRuntimeConfig.from_options(options)
    policy = options.runtime_config.input_backend_policy
    options.input_backend_requested = policy.requested
    options.input_backend_resolved = policy.resolved
    options.input_logical_device = policy.logical_device
    if options.bitwise_alignment:
        options.atol = 0.0
        options.rtol = 0.0


def _detect_paddle_version():
    from importlib.metadata import version

    try:
        return version("paddlepaddle-gpu")
    except Exception:
        try:
            return version("paddlepaddle")
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
        except GpuMemoryDeferred:
            raise
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

    normalize_dual_gpu_options(options)
    if options.api_config and _requires_gpu_runtime(options):
        _apply_single_config_gpu_defaults(options)

    if not options._sanitizer_session:
        common_error = _validate_input_sources(options)
        if common_error is not None:
            return common_error

    mode_error = _validate_test_mode(options)
    if mode_error is not None:
        return mode_error

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
    try:
        _apply_runtime_environment_flags(options)
    except ValueError as err:
        return _argument_error(str(err))
    if not options._sanitizer_session:
        log_report.print_run_header(options, options.paddle_version)
    return None


def _write_sanitizer_case_timing(duration):
    """把 session case 时长写入隔离文件，避免把内部标记混入 sanitizer 输出。"""
    timing_path = os.environ.get(SANITIZER_TIMING_FILE_ENV)
    if not timing_path:
        # 单 case 普通运行没有隔离目录，保持零开销返回。
        return
    try:
        with open(timing_path, "a", encoding="utf-8") as timing_file:
            timing_file.write(f"case_execution\t{float(duration):.9f}\n")
    except (OSError, TypeError, ValueError):
        # timing 文件不可写时不影响 sanitizer return code 和日志合并。
        # 观测目录不可写时继续沿用原有 sanitizer 结果协议。
        return


def _run_sanitizer_case_in_session(options, api_config):
    """在已初始化的 runtime 中执行一个 case，保留 deferred 重试协议。"""
    options.api_config = str(api_config).strip()
    defer_retry_count = 0
    case_started_at = time.monotonic()
    try:
        while True:
            try:
                run_test_case(options.api_config, options)
                return "done"
            except GpuMemoryDeferred as err:
                delay = _memory_defer_delay(defer_retry_count)
                defer_retry_count += 1
                print(
                    f"[DEFER] {options.api_config} | retry {defer_retry_count} "
                    f"in {delay:.1f}s | {err}",
                    flush=True,
                )
                time.sleep(delay)
    finally:
        # 每个 request 都单独结算，session 复用不能把多个 case 合并成一个样本。
        _write_sanitizer_case_timing(time.monotonic() - case_started_at)


def _run_sanitizer_session_mode(options):
    """初始化一次 Python/Paddle/CUDA runtime，并按协议顺序处理全部 case。"""
    framework_started_at = time.monotonic()
    previous_timing_path = os.environ.pop(SANITIZER_TIMING_FILE_ENV, None)
    try:
        _apply_sanitizer_budget(options)
        _init_worker_runtime(None, None, None, options, redirect_output=False)
        framework_duration = (time.monotonic() - framework_started_at) * 1000.0
        print(encode_ready(framework_duration), end="", flush=True)
        for line in sys.stdin:
            try:
                event = parse_event(line)
            except ValueError as err:
                print(f"[sanitizer session] protocol error: {err}", flush=True)
                return 2
            if event["event"] != "request":
                print("[sanitizer session] expected request event", flush=True)
                return 2
            case_id = int(event["case_id"])
            timing_path = str(event["timing_path"])
            budget_environment = os.environ.copy()
            if "workers_on_gpu" in event:
                budget_environment["PADDLEAPITEST_WORKERS_ON_GPU"] = str(event["workers_on_gpu"])
            if "compute_budget_gib" in event:
                budget_environment[SANITIZER_COMPUTE_BUDGET_ENV] = str(event["compute_budget_gib"])
            if "comparison_budget_gib" in event:
                budget_environment[SANITIZER_COMPARISON_BUDGET_ENV] = str(
                    event["comparison_budget_gib"]
                )
            _apply_sanitizer_budget(options, budget_environment)
            os.environ[SANITIZER_TIMING_FILE_ENV] = timing_path
            try:
                status = _run_sanitizer_case_in_session(options, str(event["config"]))
            except SystemExit:
                raise
            except Exception as err:
                print(f"[test error] {options.api_config}: {err}", flush=True)
                status = "error"
            finally:
                if previous_timing_path is None:
                    os.environ.pop(SANITIZER_TIMING_FILE_ENV, None)
                else:
                    os.environ[SANITIZER_TIMING_FILE_ENV] = previous_timing_path
            print(encode_result(case_id, status), end="", flush=True)
    except SystemExit:
        raise
    except Exception as err:
        print(f"[sanitizer session] init error: {err}", flush=True)
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

    log_report.print_running_banner()

    # 单 case 执行与 worker 复用同样的静默 Paddle/bootstrap 路径。
    _init_runtime_modules(options)
    init_log(options.log_dir)

    options.api_config = options.api_config.strip()
    single_case_error = None
    defer_retry_count = 0
    try:
        while True:
            try:
                run_test_case(options.api_config, options)
            except GpuMemoryDeferred as err:
                delay = _memory_defer_delay(defer_retry_count)
                defer_retry_count += 1
                log_report.print_case_notice(
                    "DEFER",
                    options.api_config,
                    f"retry {defer_retry_count} in {delay:.1f}s | {err}",
                )
                time.sleep(delay)
                continue
            log_worker.write_to_log("checkpoint", options.api_config)
            break
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

    try:
        options.gpu_pressure_timeout = read_gpu_pressure_timeout()
    except ValueError as err:
        return _argument_error(str(err))

    available_gpus = []
    max_workers_per_gpu = {}
    gpu_pairs = None
    gpu_total_memory_map = {}
    cpu_worker_count = 0
    if _requires_gpu_runtime(options):
        # GPU 算子与 GPU mode 共用真实 GPU slot，但用途由各自配置独立决定。
        # 此分支同时覆盖 GPU-kernel/CPU-compare 与 CPU-kernel/GPU-compare。
        # GPU 可见性只负责生成、执行、比较所需的 CUDA runtime 边界。
        # kernel 的最终 place 仍由 worker 内的 test_cpu 单独设置。
        gpu_ids = validate_gpu_options(options)
        available_gpus, max_workers_per_gpu = check_gpu_memory(gpu_ids, options.num_workers_per_gpu)
        if not available_gpus:
            print("No usable GPUs available.", flush=True)
            return 2
        if _dual_gpu_mode_enabled(options) and len(available_gpus) != len(gpu_ids):
            print(
                "Not all selected GPUs are usable; no complete dual-GPU layout.",
                flush=True,
            )
            return 2
        try:
            (
                available_gpus,
                max_workers_per_gpu,
                gpu_pairs,
            ) = resolve_batch_worker_layout(
                available_gpus,
                max_workers_per_gpu,
                all_case,
                dual_gpu=_dual_gpu_mode_enabled(options),
            )
        except ValueError as err:
            return _argument_error(str(err))
        # 批量路径统一收集一次，避免 worker pool 再次探测同一批 GPU。
        gpu_total_memory_map = _build_gpu_total_memory_map(available_gpus)
    else:
        # 纯 CPU 分支不调用 validate_gpu_options/check_gpu_memory。
        # cpu_worker_count 直接生成 gpu_id=None 的 worker slot。
        # 这样 runtime config、case tag 和显存预算都不会携带伪 GPU id。
        try:
            cpu_worker_count = _resolve_cpu_worker_count(options, all_case)
        except ValueError as err:
            return _argument_error(str(err))

    if options.use_compute_sanitizer:
        sanitizer_cmd = _validate_sanitizer_command(options.sanitizer_command)
        if sanitizer_cmd is None:
            return 2
        options.sanitizer_cmd = sanitizer_cmd

    if available_gpus:
        log_report.print_compute_summary(
            available_gpus,
            max_workers_per_gpu,
            gpu_pairs=gpu_pairs,
        )
    if cpu_worker_count:
        print(f"CPU: {cpu_count()} available | {cpu_worker_count} workers", flush=True)

    return _run_batch_mode(
        options=options,
        api_configs=api_configs,
        all_case=all_case,
        checkpointed_case=batch_configs.finish_case,
        available_gpus=available_gpus,
        max_workers_per_gpu=max_workers_per_gpu,
        gpu_total_memory_map=gpu_total_memory_map,
        cpu_worker_count=cpu_worker_count,
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
            "GPU pair with a dual-GPU accuracy mode."
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
        "--accuracy_dual_gpu",
        type=parse_bool,
        default=False,
        help="Use one input/compute GPU and one full-result comparison GPU per accuracy worker.",
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
        help="Run only Paddle forward/backward on CPU; Torch reference remains on GPU.",
    )
    parser.add_argument(
        "--use_cached_numpy",
        type=parse_bool,
        default=False,
        help=(
            "Reuse NumPy-backend output gradients and force the NumPy input backend; "
            "ignored in GPU mode."
        ),
    )
    parser.add_argument(
        "--use_gpu_mode",
        type=parse_bool,
        default=False,
        help=(
            "Enable GPU tensor generation, comparison, and allocator reuse; "
            "does not change the Paddle kernel or Torch reference devices."
        ),
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
        "--accuracy_manual_file",
        dest="accuracy_manual_file",
        type=str,
        default="",
        # 该参数不改变全局 atol/rtol，只提供 known API 的二次复核。
        help="YAML file with per-API thresholds for strict accuracy fallback",
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
        help="Run all cases through a reusable compute-sanitizer session.",
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
        "--_sanitizer_session",
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
    if not options._sanitizer_session:
        log_runtime.init_main_output(options.log_dir)
        atexit.register(log_runtime.close_main_output)
    if options.random_seed != parser.get_default("random_seed"):
        np.random.seed(options.random_seed)
    common_error = _prepare_common_options(options)
    if common_error is not None:
        return common_error

    if options._sanitizer_session:
        return _run_sanitizer_session_mode(options)

    if options.api_config:
        return _run_single_case_mode(options, start_time)
    if options.api_config_file or options.api_config_file_pattern or options.retest:
        return _run_batch_case_mode(options, start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
