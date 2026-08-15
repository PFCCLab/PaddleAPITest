"""compute-sanitizer 常驻 session 的窄协议。"""

from __future__ import annotations

import json

SESSION_EVENT_PREFIX = "__PADDLEAPITEST_SANITIZER_SESSION__ "
_TERMINAL_STATUSES = frozenset({"done", "error", "crashed"})


def _encode_event(payload: dict[str, object]) -> str:
    # marker 与 JSON 分开，避免 sanitizer 输出中的普通文本伪造控制消息。
    return (
        SESSION_EVENT_PREFIX + json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    )


def encode_ready(framework_ms: float | None = None) -> str:
    """编码 session 完成一次框架初始化后的握手消息。"""

    payload: dict[str, object] = {"event": "ready"}
    if framework_ms is not None:
        if float(framework_ms) < 0:
            raise ValueError("framework_ms must be non-negative")
        payload["framework_ms"] = float(framework_ms)
    return _encode_event(payload)


def encode_request(
    case_id: int,
    config: str,
    timing_path: str,
    *,
    workers_on_gpu: int | None = None,
    compute_budget_gib: float | None = None,
    comparison_budget_gib: float | None = None,
) -> str:
    """编码一个严格按序执行的 case 请求。"""

    if int(case_id) < 0:
        raise ValueError("case_id must be non-negative")
    if not isinstance(config, str) or not config.strip():
        raise ValueError("config must be a non-empty string")
    if not isinstance(timing_path, str) or not timing_path:
        raise ValueError("timing_path must be a non-empty string")
    payload: dict[str, object] = {
        "event": "request",
        "case_id": int(case_id),
        "config": config,
        "timing_path": timing_path,
    }
    if workers_on_gpu is not None:
        payload["workers_on_gpu"] = int(workers_on_gpu)
    if compute_budget_gib is not None:
        payload["compute_budget_gib"] = float(compute_budget_gib)
    if comparison_budget_gib is not None:
        payload["comparison_budget_gib"] = float(comparison_budget_gib)
    return _encode_event(payload)


def encode_result(case_id: int, status: str) -> str:
    """编码当前 case 的终态；session 级 crash 不发送伪造结果。"""

    if int(case_id) < 0:
        raise ValueError("case_id must be non-negative")
    if status not in _TERMINAL_STATUSES:
        raise ValueError(f"invalid session case status: {status!r}")
    return _encode_event({"event": "result", "case_id": int(case_id), "status": status})


def parse_event(line: str) -> dict[str, object]:
    """解析一行 session 控制消息，并拒绝不完整或跨类型字段。"""

    if not isinstance(line, str) or not line.startswith(SESSION_EVENT_PREFIX):
        raise ValueError("invalid sanitizer session marker")
    try:
        payload = json.loads(line[len(SESSION_EVENT_PREFIX) :])
    except (TypeError, ValueError, json.JSONDecodeError) as err:
        raise ValueError("invalid sanitizer session JSON") from err
    if not isinstance(payload, dict):
        raise ValueError("sanitizer session event must be an object")
    event = payload.get("event")
    if event == "ready":
        if set(payload) not in ({"event"}, {"event", "framework_ms"}):
            raise ValueError("ready event has unexpected fields")
        if "framework_ms" in payload and (
            not isinstance(payload["framework_ms"], (int, float))
            or isinstance(payload["framework_ms"], bool)
            or payload["framework_ms"] < 0
        ):
            raise ValueError("ready framework_ms must be non-negative")
        return payload
    if event == "request":
        _validate_request(payload)
        return payload
    if event == "result":
        _validate_result(payload)
        return payload
    raise ValueError(f"unknown sanitizer session event: {event!r}")


def _validate_request(payload: dict[str, object]) -> None:
    # 请求字段固定，防止未来把未审计对象直接传入 worker runtime。
    allowed = {
        "event",
        "case_id",
        "config",
        "timing_path",
        "workers_on_gpu",
        "compute_budget_gib",
        "comparison_budget_gib",
    }
    if not set(payload).issubset(allowed) or not {
        "event",
        "case_id",
        "config",
        "timing_path",
    }.issubset(payload):
        raise ValueError("request event has unexpected fields")
    case_id = payload.get("case_id")
    if not isinstance(case_id, int) or isinstance(case_id, bool) or case_id < 0:
        raise ValueError("request case_id must be a non-negative integer")
    if not isinstance(payload.get("config"), str) or not str(payload["config"]).strip():
        raise ValueError("request config must be a non-empty string")
    if not isinstance(payload.get("timing_path"), str) or not payload["timing_path"]:
        raise ValueError("request timing_path must be a non-empty string")
    if "workers_on_gpu" in payload and (
        not isinstance(payload["workers_on_gpu"], int)
        or isinstance(payload["workers_on_gpu"], bool)
        or payload["workers_on_gpu"] <= 0
    ):
        raise ValueError("request workers_on_gpu must be positive")
    for name in ("compute_budget_gib", "comparison_budget_gib"):
        if name in payload and (
            not isinstance(payload[name], (int, float))
            or isinstance(payload[name], bool)
            or payload[name] < 0
        ):
            raise ValueError(f"request {name} must be non-negative")


def _validate_result(payload: dict[str, object]) -> None:
    # result 只允许终态，session 崩溃由 EOF 语义表达而不是伪造结果。
    if set(payload) != {"event", "case_id", "status"}:
        raise ValueError("result event has unexpected fields")
    case_id = payload.get("case_id")
    if not isinstance(case_id, int) or isinstance(case_id, bool) or case_id < 0:
        raise ValueError("result case_id must be a non-negative integer")
    if payload.get("status") not in _TERMINAL_STATUSES:
        raise ValueError("result status is not terminal")
