"""engineV4 调度观测的纯 Python 聚合器。"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# worker 阶段只统计主进程能够确认起止点的完整样本。
_WORKER_STAGES = ("module_load", "preparation", "ready")
# 这些计数用于解释 pending 没有前进的原因，不参与退出码分类。
_COUNTERS = ("deferred", "retry", "crash")
# 浮点时间戳允许极小的队列传输误差，避免把合法事件判成倒序。
_EPSILON = 1e-9


@dataclass
class TimingSamples:
    """保存完整样本的总和与边界，缺失阶段不会进入分母。"""

    _count: int = 0
    _total: float = 0.0
    _minimum: float | None = None
    _maximum: float | None = None

    def add(self, value: float) -> None:
        # 非有限值和负时长都来自观测故障，不能污染平均值分母。
        value = float(value)
        if not math.isfinite(value) or value < 0:
            return
        self._count += 1
        self._total += value
        self._minimum = value if self._minimum is None else min(self._minimum, value)
        self._maximum = value if self._maximum is None else max(self._maximum, value)

    @property
    def count(self) -> int:
        # 调用方用样本数判断边界字段是否可以输出。
        return self._count

    @property
    def minimum(self) -> float | None:
        # 缺少样本时保持 None，摘要层会省略对应指标。
        return self._minimum

    @property
    def maximum(self) -> float | None:
        # 最大值用于暴露单个长尾 case 或启动异常。
        return self._maximum

    @property
    def average(self) -> float | None:
        # 只对已经通过 add 校验的完整样本求平均。
        return self._total / self.count if self.count else None


@dataclass
class WorkerObservation:
    """一个 slot 生命周期内的启动阶段时间点。"""

    gpu_id: int | None
    spawn_at: float
    device_key: str
    loaded_at: float | None = None
    preparation_started_at: float | None = None
    ready_at: float | None = None


@dataclass
class WaveObservation:
    """一个逻辑 GPU 波次的短生命周期事件快照。"""

    wave_id: int
    device_key: str
    planned_count: int
    planned_at: float
    dispatch_started_at: float | None = None
    dispatched_at: float | None = None
    reclaim_started_at: float | None = None
    reclaim_ready_at: float | None = None
    cancelled: bool = False
    # 只保留当前未回收波次的槽位时间点，回收后立即释放。
    case_starts: dict[int, float] = field(default_factory=dict)
    case_terminals: dict[int, float] = field(default_factory=dict)


def parse_sanitizer_timing_file(path: Path) -> dict[str, float]:
    """读取 sanitizer session 的 case 计时文件，损坏样本只影响观测。"""

    # timing.tsv 位于 case 隔离目录，session crash 时文件可能不存在。
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeError, TypeError, ValueError):
        return {}
    timings = {}
    for line in lines:
        # 只接受 case phase，避免 sanitizer 输出或用户内容注入摘要字段。
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] != "case_execution":
            continue
        try:
            value = float(fields[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            # 同一 phase 的最后一条完整记录代表最终重试后的耗时。
            timings[fields[0]] = value
    return timings


class SchedulingObserver:
    """收集调度事件并在批次结束时生成稳定摘要。"""

    def __init__(self, *, clock=None):
        # observer 使用独立时钟注入点，测试可复现且不依赖墙上时钟。
        self._clock = clock or time.monotonic
        # worker 记录按 slot 生命周期覆盖，阶段样本在写入时立即聚合。
        self._workers: dict[int, WorkerObservation] = {}
        # 只保留未完成波次；已回收波次的 case 起止点会立即释放。
        self._waves: dict[int, WaveObservation] = {}
        # 防止迟到终态把 case 记到错误波次，当前只保留活跃 slot 映射。
        self._slot_wave: dict[int, int] = {}
        # sanitizer session 退出前可能没有 terminal 消息，因此单独保存 session 起点。
        self._sanitizer_session_starts: dict[int, float] = {}
        self._next_wave_id = 1
        # 阶段统计按设备和名字隔离，避免多 GPU 启动耗时互相稀释。
        self._worker_stage_samples: dict[str, dict[str, TimingSamples]] = defaultdict(
            lambda: {stage: TimingSamples() for stage in _WORKER_STAGES}
        )
        # 设备级指标采用流式聚合，避免大批次保留完整样本列表。
        self._devices: dict[str, dict[str, object]] = {}
        self._sanitizer_samples = {
            "case_execution": TimingSamples(),
            "wrapper_session": TimingSamples(),
            "session_ready": TimingSamples(),
        }
        # defaultdict 保证新批次没有事件时仍能输出零计数。
        self._counters = defaultdict(int)
        self._error_count = 0
        self._error_methods: dict[str, int] = defaultdict(int)

    def _now(self):
        return float(self._clock())

    @staticmethod
    def _duration(start, end):
        # reclaim 未稳定时 end 为空是正常状态，不应让摘要收尾失败。
        if start is None or end is None:
            return None
        duration = float(end) - float(start)
        return duration if math.isfinite(duration) and duration >= 0 else None

    @staticmethod
    def _device_key(device_ids):
        return "+".join(str(device_id) for device_id in device_ids)

    def _device_stats(self, device_key):
        stats = self._devices.get(device_key)
        if stats is None:
            stats = {
                "waves_planned": 0,
                "waves_dispatched": 0,
                "waves_cancelled": 0,
                "cases_planned": 0,
                "cases": 0,
                "plan": TimingSamples(),
                "dispatch": TimingSamples(),
                "wave": TimingSamples(),
                "case": TimingSamples(),
                "reclaim": TimingSamples(),
                "barrier_idle": 0.0,
            }
            self._devices[device_key] = stats
        return stats

    def record_worker_spawn(self, slot_index, gpu_id, timestamp=None, *, comparison_gpu_id=None):
        # 新 PID 复用同一 slot 时覆盖旧起点，避免跨代混算启动耗时。
        timestamp = self._now() if timestamp is None else float(timestamp)
        device_ids = (gpu_id,) if comparison_gpu_id is None else (gpu_id, comparison_gpu_id)
        self._workers[int(slot_index)] = WorkerObservation(
            gpu_id, timestamp, self._device_key(device_ids)
        )

    def record_worker_loaded(self, slot_index, timestamp=None):
        # loaded 只能消费当前 spawn 代的第一次消息。
        worker = self._workers.get(int(slot_index))
        if worker is None or worker.loaded_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        duration = self._duration(worker.spawn_at, timestamp)
        if duration is not None:
            self._worker_stage_samples[worker.device_key]["module_load"].add(duration)
        worker.loaded_at = timestamp

    def record_worker_preparation_start(self, slot_index, timestamp=None):
        # preparation 必须发生在 loaded 之后，倒序消息只被忽略。
        worker = self._workers.get(int(slot_index))
        if worker is None or worker.preparation_started_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        if worker.loaded_at is None or timestamp + _EPSILON < worker.loaded_at:
            return
        worker.preparation_started_at = timestamp

    def record_worker_ready(self, slot_index, timestamp=None):
        # 没有 preparation 起点时不猜测时长，保留后续重建机会。
        worker = self._workers.get(int(slot_index))
        if worker is None or worker.ready_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        duration = self._duration(worker.preparation_started_at, timestamp)
        if duration is None:
            return
        device_key = worker.device_key
        self._worker_stage_samples[device_key]["preparation"].add(duration)
        ready_duration = self._duration(worker.spawn_at, timestamp)
        if ready_duration is not None:
            self._worker_stage_samples[device_key]["ready"].add(ready_duration)
        worker.ready_at = timestamp

    def record_wave_planned(self, device_ids, planned_count, timestamp=None):
        # planned 是逻辑承诺建立点，尚未代表进程已经启动或 case 已派发。
        timestamp = self._now() if timestamp is None else float(timestamp)
        wave_id = self._next_wave_id
        self._next_wave_id += 1
        device_key = self._device_key(device_ids)
        self._waves[wave_id] = WaveObservation(
            wave_id=wave_id,
            device_key=device_key,
            planned_count=max(0, int(planned_count)),
            planned_at=timestamp,
        )
        stats = self._device_stats(device_key)
        stats["waves_planned"] += 1
        stats["cases_planned"] += max(0, int(planned_count))
        return wave_id

    def record_wave_dispatched(self, wave_id, timestamp=None):
        # 只有原子派发完成后才把波次计入 dispatched 分母。
        wave = self._waves.get(int(wave_id))
        if wave is None or wave.cancelled or wave.dispatched_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        if timestamp + _EPSILON < wave.planned_at:
            return
        wave.dispatched_at = timestamp
        stats = self._device_stats(wave.device_key)
        stats["waves_dispatched"] += 1
        plan_duration = self._duration(wave.planned_at, timestamp)
        if plan_duration is not None:
            stats["plan"].add(plan_duration)
        dispatch_duration = self._duration(wave.dispatch_started_at, timestamp)
        if dispatch_duration is not None:
            stats["dispatch"].add(dispatch_duration)

    def record_wave_dispatch_start(self, wave_id, timestamp=None):
        # 派发耗时只覆盖 put/claim 操作，不把 worker 执行时间混入。
        wave = self._waves.get(int(wave_id))
        if wave is None or wave.cancelled or wave.dispatch_started_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        if timestamp + _EPSILON < wave.planned_at:
            return
        wave.dispatch_started_at = timestamp

    def record_wave_cancelled(self, wave_id):
        # 已经部分派发的波次不能标为取消，否则会掩盖真实在途 case。
        wave = self._waves.get(int(wave_id))
        if wave is not None and wave.dispatched_at is None:
            wave.cancelled = True
            self._device_stats(wave.device_key)["waves_cancelled"] += 1
            self._waves.pop(wave.wave_id, None)
            return True
        return False

    def record_case_dispatched(self, wave_id, slot_index, config, timestamp=None):
        # case 起点在 terminal claim 前登记，便于 timeout/crash 也结算耗时。
        del config
        wave = self._waves.get(int(wave_id))
        if wave is None or wave.cancelled or wave.dispatched_at is None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        if timestamp + _EPSILON < wave.dispatched_at:
            return
        wave.case_starts.setdefault(int(slot_index), timestamp)
        self._slot_wave[int(slot_index)] = wave.wave_id

    def record_sanitizer_session_start(self, slot_index, timestamp=None):
        """保存 wrapper 观察到的 session 启动点，终态时结算 session 总耗时。"""

        # wrapper 的 child PID 消息是父进程可见的最早启动边界。
        timestamp = self._now() if timestamp is None else float(timestamp)
        self._sanitizer_session_starts[int(slot_index)] = timestamp

    def record_case_terminal(self, wave_id, slot_index, timestamp=None):
        # 终态认领由 engineV4 先完成；此方法只接收胜出的消息。
        wave = self._waves.get(int(wave_id))
        slot_index = int(slot_index)
        if (
            wave is None
            or self._slot_wave.get(slot_index) != wave.wave_id
            or slot_index not in wave.case_starts
        ):
            return False
        if slot_index in wave.case_terminals:
            return False
        timestamp = self._now() if timestamp is None else float(timestamp)
        if timestamp + _EPSILON < wave.case_starts[slot_index]:
            return False
        wave.case_terminals[slot_index] = timestamp
        stats = self._device_stats(wave.device_key)
        stats["cases"] += 1
        duration = self._duration(wave.case_starts[slot_index], timestamp)
        if duration is not None:
            stats["case"].add(duration)
        session_started_at = self._sanitizer_session_starts.pop(slot_index, None)
        session_duration = self._duration(session_started_at, timestamp)
        if session_duration is not None:
            self._sanitizer_samples["wrapper_session"].add(session_duration)
        self._slot_wave.pop(slot_index, None)
        return True

    def record_reclaim_started(self, wave_id, timestamp=None):
        # 最后一个 terminal 到达才开启物理显存回收计时。
        wave = self._waves.get(int(wave_id))
        if wave is None or wave.reclaim_started_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        if len(wave.case_terminals) != len(wave.case_starts):
            return
        wave.reclaim_started_at = timestamp
        stats = self._device_stats(wave.device_key)
        if wave.case_terminals:
            last_terminal = max(wave.case_terminals.values())
            wave_duration = self._duration(wave.dispatched_at, last_terminal)
            if wave_duration is not None:
                stats["wave"].add(wave_duration)
            stats["barrier_idle"] += sum(
                max(0.0, last_terminal - terminal_at)
                for terminal_at in wave.case_terminals.values()
            )

    def record_reclaim_ready(self, wave_id, timestamp=None):
        # ready 只能由稳定采样确认，单次 free 上升不进入样本。
        # planned wave 被取消后仍需等待物理回收，但没有可结算的 logical wave。
        if wave_id is None:
            return
        wave = self._waves.get(int(wave_id))
        if wave is None or wave.reclaim_started_at is None or wave.reclaim_ready_at is not None:
            return
        timestamp = self._now() if timestamp is None else float(timestamp)
        if timestamp + _EPSILON < wave.reclaim_started_at:
            return
        wave.reclaim_ready_at = timestamp
        duration = self._duration(wave.reclaim_started_at, timestamp)
        if duration is not None:
            self._device_stats(wave.device_key)["reclaim"].add(duration)
        self._waves.pop(wave.wave_id, None)

    def record_sanitizer_phase(self, phase, duration):
        # child timing 文件只允许设计中的 phase，未知字段静默丢弃。
        samples = self._sanitizer_samples.get(phase)
        if samples is not None:
            samples.add(duration)

    def record_sanitizer_session_ready(self, duration):
        # session 初始化只发生在 session spawn，不能误计入每个 case 的 child 时长。
        self._sanitizer_samples["session_ready"].add(duration / 1000.0)

    def record_counter(self, name, amount=1):
        # 计数器只表达已经处理的事件，不允许 retry 负向抵消。
        amount = int(amount)
        if amount < 0:
            raise ValueError("observation counter cannot be negative")
        self._counters[name] += amount

    def record_error(self, method="unknown"):
        # 观测错误保留方法名，批次结束时一次性报告，避免刷屏。
        self._error_count += 1
        self._error_methods[str(method)] += 1

    @staticmethod
    def _format_sample(prefix, samples):
        # 空样本不输出 min/max/avg，防止 None 参与格式化。
        if not samples.count:
            return []
        return [
            f"{prefix}_samples={samples.count}",
            f"{prefix}_ms_min={samples.minimum * 1000:.1f}",
            f"{prefix}_ms_max={samples.maximum * 1000:.1f}",
            f"{prefix}_ms_avg={samples.average * 1000:.1f}",
        ]

    def summary_lines(self):
        # 摘要按设备排序，便于多 GPU 批次的日志 diff 和脚本解析。
        lines = []
        for device_key in sorted(self._devices):
            stats = self._devices[device_key]
            fields = [
                "[gpu] OBSERVE",
                f"devices={device_key}",
                f"waves_planned={stats['waves_planned']}",
                f"waves_dispatched={stats['waves_dispatched']}",
                f"waves_cancelled={stats['waves_cancelled']}",
                f"cases_planned={stats['cases_planned']}",
                f"cases={stats['cases']}",
            ]
            for prefix in ("plan", "dispatch", "wave", "case", "reclaim"):
                fields.extend(self._format_sample(prefix, stats[prefix]))
            if stats["cases"]:
                fields.append(f"barrier_idle_ms={stats['barrier_idle'] * 1000:.1f}")
            lines.append(" | ".join(fields))

        for device_key in sorted(self._worker_stage_samples):
            for stage in _WORKER_STAGES:
                fields = self._format_sample(stage, self._worker_stage_samples[device_key][stage])
                if fields:
                    lines.append(
                        " | ".join(
                            [
                                "[gpu] OBSERVE",
                                f"devices={device_key}",
                                f"worker_stage={stage}",
                                *fields,
                            ]
                        )
                    )
        for phase, samples in self._sanitizer_samples.items():
            fields = self._format_sample("sanitizer_" + phase, samples)
            if fields:
                lines.append(" | ".join(["[gpu] OBSERVE", *fields]))
        if self._devices or any(self._counters[name] for name in _COUNTERS):
            lines.append(
                " | ".join(
                    [
                        "[gpu] OBSERVE",
                        "counters=global",
                        f"deferred={self._counters['deferred']}",
                        f"retries={self._counters['retry']}",
                        f"crashes={self._counters['crash']}",
                    ]
                )
            )
        if self._error_count:
            methods = ",".join(
                f"{method}={count}" for method, count in sorted(self._error_methods.items())
            )
            lines.append(f"[gpu] OBSERVE_ERROR | count={self._error_count} | methods={methods}")
        return lines
