"""Compute-sanitizer session protocol and process lifecycle."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
from dataclasses import dataclass

# 控制行使用专用前缀，普通 sanitizer 输出不会被误解析为协议事件。
SESSION_EVENT_PREFIX = "__PADDLEAPITEST_SANITIZER_SESSION__ "
# EOF 表示 session 级崩溃，不允许伪造成 request result。
_TERMINAL_STATUSES = frozenset({"done", "error", "crashed"})


def _encode_event(payload):
    return (
        SESSION_EVENT_PREFIX + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    )


def _encode_nonnegative_finite(value, field):
    # 编码端和解析端拒绝同一组非法预算，避免本地生成不可回读的协议。
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"{field} must be non-negative")
    return float(value)


def encode_ready(framework_ms=None):
    # ready 只表示框架初始化完成，不代表已有 request 可以乱序执行。
    payload = {"event": "ready"}
    if framework_ms is not None:
        payload["framework_ms"] = _encode_nonnegative_finite(framework_ms, "framework_ms")
    return _encode_event(payload)


def encode_request(
    case_id,
    config,
    timing_path,
    *,
    workers_on_gpu=None,
    compute_budget_gib=None,
    comparison_budget_gib=None,
):
    # timing_path 属于 wrapper 与 session 的文件协议，必须保持字符串类型。
    # case_id 不借助 int() 放宽类型，否则编码端会接受解析端拒绝的字符串。
    if (
        not isinstance(case_id, int)
        or isinstance(case_id, bool)
        or case_id < 0
        or not isinstance(config, str)
        or not config.strip()
        or not isinstance(timing_path, str)
        or not timing_path
    ):
        raise ValueError("invalid sanitizer session request")
    payload = {
        "event": "request",
        "case_id": case_id,
        "config": config,
        "timing_path": timing_path,
    }
    if workers_on_gpu is not None:
        if (
            not isinstance(workers_on_gpu, int)
            or isinstance(workers_on_gpu, bool)
            or workers_on_gpu <= 0
        ):
            raise ValueError("workers_on_gpu must be positive")
        payload["workers_on_gpu"] = workers_on_gpu
    if compute_budget_gib is not None:
        payload["compute_budget_gib"] = _encode_nonnegative_finite(
            compute_budget_gib, "compute_budget_gib"
        )
    if comparison_budget_gib is not None:
        payload["comparison_budget_gib"] = _encode_nonnegative_finite(
            comparison_budget_gib, "comparison_budget_gib"
        )
    return _encode_event(payload)


def encode_result(case_id, status):
    # result 必须匹配现存 request；session 崩溃由 wrapper 的 EOF 分支归类。
    if (
        not isinstance(case_id, int)
        or isinstance(case_id, bool)
        or case_id < 0
        or status not in _TERMINAL_STATUSES
    ):
        raise ValueError("invalid sanitizer session result")
    return _encode_event({"event": "result", "case_id": case_id, "status": status})


def parse_event(line):
    # 子进程输出属于不可信协议输入，marker、JSON 和字段类型都需要验证。
    if not isinstance(line, str) or not line.startswith(SESSION_EVENT_PREFIX):
        raise ValueError("invalid sanitizer session marker")
    try:
        payload = json.loads(line[len(SESSION_EVENT_PREFIX) :])
    except (TypeError, ValueError) as err:
        raise ValueError("invalid sanitizer session JSON") from err
    if not isinstance(payload, dict):
        raise ValueError("sanitizer session event must be an object")

    event = payload.get("event")
    if event == "ready":
        # ready 的扩展字段必须显式加入白名单，避免协议漂移被静默吞掉。
        if set(payload) - {"event", "framework_ms"}:
            raise ValueError("ready event has unexpected fields")
        if "framework_ms" in payload and (
            not isinstance(payload["framework_ms"], (int, float))
            or isinstance(payload["framework_ms"], bool)
            or not math.isfinite(float(payload["framework_ms"]))
            or payload["framework_ms"] < 0
        ):
            raise ValueError("ready framework_ms must be non-negative")
        return payload

    if event == "request":
        # request 是唯一允许携带预算和 timing 文件位置的消息类型。
        required = {"event", "case_id", "config", "timing_path"}
        if not required.issubset(payload) or set(payload) - required - {
            "workers_on_gpu",
            "compute_budget_gib",
            "comparison_budget_gib",
        }:
            raise ValueError("request event has unexpected fields")
        if (
            not isinstance(payload["case_id"], int)
            or isinstance(payload["case_id"], bool)
            or payload["case_id"] < 0
        ):
            raise ValueError("request case_id must be non-negative")
        if not isinstance(payload["config"], str) or not payload["config"].strip():
            raise ValueError("request config must be non-empty")
        if not isinstance(payload["timing_path"], str) or not payload["timing_path"]:
            raise ValueError("request timing_path must be non-empty")
        if "workers_on_gpu" in payload and (
            not isinstance(payload["workers_on_gpu"], int)
            or isinstance(payload["workers_on_gpu"], bool)
            or payload["workers_on_gpu"] <= 0
        ):
            raise ValueError("request workers_on_gpu must be positive")
        for name in ("compute_budget_gib", "comparison_budget_gib"):
            # NaN/Inf 会绕过普通非负比较，随后污染显存预检环境变量。
            if name in payload and (
                not isinstance(payload[name], (int, float))
                or isinstance(payload[name], bool)
                or not math.isfinite(float(payload[name]))
                or payload[name] < 0
            ):
                raise ValueError(f"request {name} must be non-negative")
        return payload

    if event == "result":
        # bool 是 int 子类，必须显式拒绝，避免 case_id 令牌错配。
        if (
            set(payload) != {"event", "case_id", "status"}
            or not isinstance(payload["case_id"], int)
            or isinstance(payload["case_id"], bool)
            or payload["case_id"] < 0
            or payload.get("status") not in _TERMINAL_STATUSES
        ):
            raise ValueError("invalid sanitizer result event")
        return payload
    raise ValueError(f"unknown sanitizer session event: {event!r}")


@dataclass(frozen=True)
class SanitizerSessionResult:
    # diagnostic 只承载 session 协议故障，不混入普通 sanitizer 输出流。
    status: str
    returncode: int
    diagnostic: str = ""


class SanitizerSession:
    """Manage one reusable sanitizer subprocess and its sequential requests."""

    def __init__(self, command, environ):
        # command/env 在 session 生命周期内固定，request 只改变协议 payload。
        self.command = tuple(command)
        self.environ = environ
        self.process = None

    @property
    def pid(self):
        # 调用方只拿 PID 做 watchdog 账务，不依赖 subprocess 对象。
        return self.process.pid if self.process is not None else None

    def start(self, *, on_output, on_started=None):
        """Ensure the session is ready; return the ready event for a new session."""
        # 已 ready 的 session 直接复用，不重复触发框架初始化和 ready 计时。
        if self.process is not None and self.process.poll() is None:
            return None
        # 已退出进程先清账，新的 Popen 才能成为唯一 session 句柄。
        self.close()
        self.process = subprocess.Popen(
            self.command,
            env=self.environ,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        # 子进程组在框架初始化前登记，startup timeout 才能完整回收它。
        if on_started is not None:
            on_started(self.process.pid)
        while True:
            # 普通输出交给调用方归集，marker 才属于 session 控制协议。
            line = self.process.stdout.readline()
            if not line:
                self.close(wait=True)
                raise EOFError("sanitizer session exited before ready")
            if not line.startswith(SESSION_EVENT_PREFIX):
                on_output(line)
                continue
            try:
                event = parse_event(line)
            except ValueError:
                self.close(wait=True)
                raise
            if event["event"] != "ready":
                self.close(wait=True)
                raise ValueError("sanitizer session did not send ready")
            return event

    def run_request(
        self,
        case_id,
        config,
        timing_path,
        *,
        on_output,
        workers_on_gpu=None,
        compute_budget_gib=None,
        comparison_budget_gib=None,
    ):
        """Send one request and return its matching terminal result."""
        # session 不可用时返回可分类的 crashed，不让调用方接触进程细节。
        process = self.process
        if process is None or process.poll() is not None:
            return SanitizerSessionResult("crashed", -1, "sanitizer session unavailable")
        try:
            # request 由统一编码器生成，两个入口不能产生不同预算协议。
            process.stdin.write(
                encode_request(
                    case_id,
                    config,
                    str(timing_path),
                    workers_on_gpu=workers_on_gpu,
                    compute_budget_gib=compute_budget_gib,
                    comparison_budget_gib=comparison_budget_gib,
                )
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError, AttributeError) as err:
            # 写入失败表示 request 是否被接受未知，不能在同一 session 重发。
            self.close(wait=True)
            return SanitizerSessionResult("crashed", -1, str(err))

        while True:
            line = process.stdout.readline()
            if not line:
                # EOF 后不能复用半关闭 stdout；close 同时回收 sanitizer 进程组。
                returncode = process.poll()
                self.close(wait=returncode is None)
                return SanitizerSessionResult("crashed", -1 if returncode is None else returncode)
            if not line.startswith(SESSION_EVENT_PREFIX):
                on_output(line)
                continue
            try:
                event = parse_event(line)
            except ValueError as err:
                # 非法 marker 会破坏流同步，诊断返回调用方后立即关闭 session。
                self.close(wait=True)
                return SanitizerSessionResult(
                    "crashed", -1, f"[sanitizer session] protocol error: {err}\n"
                )
            if event["event"] != "result" or int(event["case_id"]) != case_id:
                # 结果错配意味着协议失去同步，必须整组退役。
                self.close(wait=True)
                return SanitizerSessionResult(
                    "crashed", -1, "[sanitizer session] unexpected result marker\n"
                )
            status = str(event["status"])
            return SanitizerSessionResult(status, 0 if status == "done" else 2)

    def close(self, *, wait=False):
        # close 幂等，异常路径和 signal handler 可以安全重复调用。
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            # 独立进程组必须同时回收 sanitizer 和其目标进程。
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    process.kill()
                except (ProcessLookupError, OSError, AttributeError):
                    pass
        if wait:
            # wait 只用于确定性清理路径；signal handler 保持非阻塞退出。
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, AttributeError):
                pass
