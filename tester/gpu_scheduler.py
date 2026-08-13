"""主进程使用的 GPU 显存调度策略。"""

from __future__ import annotations

import math
import os

# 使用集合抽象类型，兼容运行时实例检查与类型标注。
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

GIB = 1024**3
GPU_PRESSURE_TIMEOUT_ENV_VAR = "PADDLEAPITEST_GPU_PRESSURE_TIMEOUT_SECONDS"
DEFAULT_GPU_PRESSURE_TIMEOUT_SECONDS = 600.0


def read_gpu_pressure_timeout(environ: Mapping[str, str] | None = None) -> float:
    """读取批次无显存进展超时；该协议刻意不暴露命令行参数。"""
    # 环境变量是唯一入口，避免 CLI 与回归脚本产生两套优先级规则。
    # 零值表示首次确认持续阻塞时立即终止，并不代表关闭超时保护。
    # 这里仅解析策略，checkpoint 的保留和非零退出由批次主循环负责。
    source = os.environ if environ is None else environ
    raw_value = source.get(
        GPU_PRESSURE_TIMEOUT_ENV_VAR,
        str(DEFAULT_GPU_PRESSURE_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError) as err:
        # 启动阶段直接拒绝非法值，不能在批次运行后静默退回默认值。
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


@dataclass(frozen=True)
class GpuMemorySnapshot:
    """一次物理显存采样；外部占用和遗留显存均已反映在 free 中。"""

    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class GpuWaveAdmission:
    selected_indices: tuple[int, ...] = ()
    committed_bytes: int = 0


@dataclass(frozen=True)
class CaseGpuEstimate:
    compute_bytes: int = 0
    comparison_bytes: int = 0


@dataclass
class GpuReclaimTracker:
    """等待物理 free 稳定，避免在驱动延迟回收期间创建替代进程。"""

    tolerance_bytes: int = 64 * 1024**2
    sample_interval_seconds: float = 1.0
    _last_free_bytes: int | None = None
    _last_sample_at: float | None = None

    def reset(self) -> None:
        self._last_free_bytes = None
        self._last_sample_at = None

    def observe(self, snapshot: GpuMemorySnapshot, *, now: float) -> bool:
        free_bytes = max(0, int(snapshot.free_bytes))
        # tracker 只判断连续物理快照是否稳定，不推断显存属于本批次还是外部进程。
        # 因此 worker 已退出也不能跳过该状态，CUDA context 可能仍在异步释放。
        # 首个样本只建立基线，单次 free 上升不能证明 allocator 已经稳定。
        if self._last_sample_at is None:
            self._last_free_bytes = free_bytes
            self._last_sample_at = float(now)
            return False
        if float(now) - self._last_sample_at < self.sample_interval_seconds:
            # 高频轮询不算独立样本，防止同一个驱动快照被误判为已回收。
            return False

        # 大幅变化意味着回收仍在继续，本次采样成为新的稳定性基线。
        stable = abs(free_bytes - int(self._last_free_bytes or 0)) <= self.tolerance_bytes
        self._last_free_bytes = free_bytes
        self._last_sample_at = float(now)
        return stable


@dataclass
class GpuPressureTimeout:
    """记录整个批次连续无法取得调度进展的时长。"""

    timeout_seconds: float
    blocked_since: float | None = None

    def update(self, *, blocked: bool, now: float) -> bool:
        if not blocked:
            # 派发或完成任一 case 都会结束“连续无进展”区间。
            self.blocked_since = None
            return False
        if self.blocked_since is None:
            self.blocked_since = float(now)
        return float(now) - self.blocked_since >= self.timeout_seconds


@dataclass(frozen=True)
class GpuSchedulingPolicy:
    safety_reserve_bytes_min: int = 2 * GIB
    safety_reserve_fraction: float = 0.05
    minimum_case_bytes: int = 1 * GIB
    case_margin_bytes: int = 512 * 1024**2
    case_multiplier: float = 1.25

    def safety_reserve_bytes(self, snapshot: GpuMemorySnapshot) -> int:
        # 小卡使用绝对保留量，大卡使用比例保留量，兼顾启动和临时 workspace。
        proportional_reserve = int(max(0, snapshot.total_bytes) * self.safety_reserve_fraction)
        return max(self.safety_reserve_bytes_min, proportional_reserve)

    def case_admission_bytes(self, estimated_peak_bytes: int) -> int:
        estimate = max(0, int(estimated_peak_bytes))
        # 三种下界同时覆盖未知配置、小配置固定开销和大配置估算误差。
        return max(
            self.minimum_case_bytes,
            estimate + self.case_margin_bytes,
            int(estimate * self.case_multiplier),
        )


class GpuWaveController:
    """维护一个 GPU 或双 GPU 对的波次准入与回收边界。"""

    def __init__(
        self,
        *,
        device_ids: tuple[int, ...],
        max_workers: int,
        policy: GpuSchedulingPolicy | None = None,
    ):
        if not device_ids or len(device_ids) > 2:
            raise ValueError("GPU wave requires one compute GPU and at most one comparison GPU")
        self.device_ids = tuple(device_ids)
        self.max_workers = max(0, int(max_workers))
        self.policy = policy or GpuSchedulingPolicy()
        self.state = "ready"
        self.active_cases = 0
        # planned commitment 只在主进程持有，worker 不独立争抢同一份 free。
        self._planned_commitments: dict[int, int] = {}
        self._reclaim_trackers = {gpu_id: GpuReclaimTracker() for gpu_id in self.device_ids}

    def _available_bytes(self, snapshots: Mapping[int, GpuMemorySnapshot]) -> dict[int, int]:
        available = {}
        for gpu_id in self.device_ids:
            snapshot = snapshots[gpu_id]
            # NVML free 已包含外部进程和本批次遗留占用，不再按 PID 做归属推断。
            available[gpu_id] = max(
                0,
                snapshot.free_bytes - self.policy.safety_reserve_bytes(snapshot),
            )
        return available

    def plan(
        self,
        estimates: Sequence[CaseGpuEstimate],
        snapshots: Mapping[int, GpuMemorySnapshot],
    ) -> GpuWaveAdmission:
        # plan 只写入主进程的逻辑承诺，不创建 worker，也不改变物理显存。
        # 调用方必须在延迟创建完成后再次采样，并通过 confirm 才能派发。
        if self.state not in {"ready", "pressure"} or self.max_workers <= 0:
            # planned/running/reclaim_pending 状态都禁止波次中途补位。
            return GpuWaveAdmission()

        available = self._available_bytes(snapshots)
        reserves = {
            gpu_id: self.policy.safety_reserve_bytes(snapshots[gpu_id])
            for gpu_id in self.device_ids
        }
        commitments = dict.fromkeys(self.device_ids, 0)
        selected_indices = []
        for index, estimate in enumerate(estimates):
            if len(selected_indices) >= self.max_workers:
                break
            requested = {
                self.device_ids[0]: self.policy.case_admission_bytes(estimate.compute_bytes)
            }
            if len(self.device_ids) == 2:
                # 双卡 case 必须同时承诺计算卡和对比卡，任一卡不足都不准入。
                requested[self.device_ids[1]] = self.policy.case_admission_bytes(
                    estimate.comparison_bytes
                )
            exclusive = any(requested[gpu_id] >= reserves[gpu_id] for gpu_id in self.device_ids)
            if exclusive and selected_indices:
                # 大 case 留到独占波次，避免多个进程的真实峰值同时超过静态下界。
                continue
            if any(
                commitments[gpu_id] + requested[gpu_id] > available[gpu_id]
                for gpu_id in self.device_ids
            ):
                # 当前 case 留给后续 GPU 或波次，不能因为大 case 阻塞整个队列扫描。
                continue
            selected_indices.append(index)
            for gpu_id in self.device_ids:
                commitments[gpu_id] += requested[gpu_id]
            if exclusive:
                break

        if not selected_indices:
            # pressure 仍允许后续重新 plan，但不会触发任何进程创建。
            self.state = "pressure"
            self._planned_commitments.clear()
            return GpuWaveAdmission()
        self.state = "planned"
        self._planned_commitments = commitments
        return GpuWaveAdmission(
            tuple(selected_indices),
            commitments[self.device_ids[0]],
        )

    def confirm_planned_wave(self, snapshots: Mapping[int, GpuMemorySnapshot]) -> bool:
        if self.state != "planned":
            return False
        # 二次采样覆盖 worker 初始化成本以及采样间隙中新出现的外部占用。
        # 任一设备失约时整波作废，不能用部分 worker 执行部分 case。
        available = self._available_bytes(snapshots)
        if all(
            self._planned_commitments[gpu_id] <= available[gpu_id] for gpu_id in self.device_ids
        ):
            # worker bootstrap 后仍满足原承诺，才允许把整波任务原子派发出去。
            return True

        # worker 初始化已经发生时必须先观察物理显存回收，不能直接重新规划。
        self.state = "reclaim_pending"
        self._planned_commitments.clear()
        for tracker in self._reclaim_trackers.values():
            tracker.reset()
        return False

    def cancel_planned_wave(self) -> None:
        if self.state != "planned":
            return
        # cancel 只撤销尚未派发的逻辑波次，调用方仍需终止已创建的进程。
        self.state = "reclaim_pending"
        # 即使 worker 未执行 case，启动阶段也可能留下 CUDA context 显存。
        self._planned_commitments.clear()
        for tracker in self._reclaim_trackers.values():
            tracker.reset()

    def mark_dispatched(self, case_count: int) -> None:
        if self.state != "planned" or case_count <= 0:
            raise RuntimeError("cannot dispatch a GPU wave that has not passed admission")
        self.active_cases = int(case_count)
        # running 计数属于整波，单个 case 完成不会恢复 ready。
        self.state = "running"

    def mark_completed(self) -> None:
        if self.state != "running" or self.active_cases <= 0:
            raise RuntimeError("cannot complete a case outside a running GPU wave")
        self.active_cases -= 1
        if self.active_cases:
            # 不在半波状态下补位，否则多个 worker 会竞争未回收显存。
            return
        self.state = "reclaim_pending"
        self._planned_commitments.clear()
        for tracker in self._reclaim_trackers.values():
            tracker.reset()

    def observe_reclaim(
        self,
        snapshots: Mapping[int, GpuMemorySnapshot],
        *,
        now: float,
    ) -> bool:
        if self.state != "reclaim_pending":
            return self.state == "ready"
        stable_by_device = [
            self._reclaim_trackers[gpu_id].observe(snapshots[gpu_id], now=now)
            for gpu_id in self.device_ids
        ]
        # 双 GPU 必须在同一次观察轮次都稳定，不能只恢复计算卡。
        if all(stable_by_device):
            self.state = "ready"
            return True
        return False


def select_admissible_wave(
    snapshot: GpuMemorySnapshot,
    estimated_peak_bytes: Sequence[int],
    max_workers: int,
    policy: GpuSchedulingPolicy | None = None,
) -> GpuWaveAdmission:
    """按输入顺序形成一个原子波次，不能放入的 case 留给其他 GPU。"""
    # 该纯函数供无进程生命周期的调用方复用，不承担二次采样和回收等待。
    # 返回下标引用原输入序列，未选项的保存和重排由上层队列负责。
    policy = policy or GpuSchedulingPolicy()
    available_bytes = max(0, snapshot.free_bytes - policy.safety_reserve_bytes(snapshot))
    # 先扣安全保留量，再在本地累加承诺，保证一次 plan 内没有超卖。
    selected_indices = []
    committed_bytes = 0

    # worker 上限只是逻辑容量；物理显存不足时实际波次会自然缩小到零。
    for index, estimated_peak in enumerate(estimated_peak_bytes):
        if len(selected_indices) >= max(0, int(max_workers)):
            break
        case_bytes = policy.case_admission_bytes(estimated_peak)
        if committed_bytes + case_bytes > available_bytes:
            continue
        selected_indices.append(index)
        committed_bytes += case_bytes

    return GpuWaveAdmission(tuple(selected_indices), committed_bytes)
