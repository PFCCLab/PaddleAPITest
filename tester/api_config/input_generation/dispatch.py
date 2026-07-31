"""输入生成调度与事件记录。"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .registry import API_RULE_REGISTRY
from .signature_binding import build_ctx

INPUT_GENERATOR_ENV = "PADDLEAPITEST_INPUT_GENERATOR"
INPUT_GENERATOR_MODES = frozenset({"v2"})
DEFAULT_INPUT_GENERATOR_MODE = "v2"
NO_RULE_ID = "unregistered"


@dataclass(frozen=True)
class LegacyGenerationEvent:
    rule_id: str
    api_name: str
    arg_path: str


@dataclass(frozen=True)
class DispatchEvent:
    api_name: str
    runtime_mode: str
    outcome: str
    rule_id: str
    fallback_reason: str | None = None


_legacy_event_buffer: ContextVar[list[LegacyGenerationEvent] | None] = ContextVar(
    "input_generation_event_buffer", default=None
)
_dispatch_event_buffer: ContextVar[list[DispatchEvent] | None] = ContextVar(
    "input_generation_dispatch_event_buffer", default=None
)


def _format_arg_path(index=None, key=None, list_index=None):
    if index is not None:
        path = f"args[{index}]"
    elif key is not None:
        path = f"kwargs.{key}"
    else:
        path = "unknown"
    for item_index in list_index or ():
        path += f"[{item_index}]"
    return path


def record_legacy(api_config, index=None, key=None, list_index=None):
    # 旧事件只为兼容验证器保留，生产 v2 仅记录调度事件。
    buffer = _legacy_event_buffer.get()
    if buffer is None:
        return
    api_name = api_config.api_name
    buffer.append(
        LegacyGenerationEvent(
            rule_id=f"legacy.api.{api_name}",
            api_name=api_name,
            arg_path=_format_arg_path(index=index, key=key, list_index=list_index),
        )
    )


@contextmanager
def capture_legacy() -> Iterator[list[LegacyGenerationEvent]]:
    # 事件缓冲区是上下文局部的，避免并发 case 互相污染。
    events = []
    token = _legacy_event_buffer.set(events)
    try:
        yield events
    finally:
        _legacy_event_buffer.reset(token)


def record_dispatch(
    api_name: str,
    runtime_mode: str,
    outcome: str,
    rule_id: str,
    fallback_reason: str | None = None,
) -> None:
    # 遥测是按需开启的，正常运行不承担缓冲开销。
    buffer = _dispatch_event_buffer.get()
    if buffer is None:
        return
    buffer.append(
        DispatchEvent(
            api_name=api_name,
            runtime_mode=runtime_mode,
            outcome=outcome,
            rule_id=rule_id,
            fallback_reason=fallback_reason,
        )
    )


@contextmanager
def capture_dispatch() -> Iterator[list[DispatchEvent]]:
    events = []
    token = _dispatch_event_buffer.set(events)
    try:
        yield events
    finally:
        _dispatch_event_buffer.reset(token)


def resolve_mode(value: str | None = None) -> str:
    # 环境变量解析保持极小化：当前只支持 v2。
    raw_value = os.environ.get(INPUT_GENERATOR_ENV, "") if value is None else value
    mode = str(raw_value or DEFAULT_INPUT_GENERATOR_MODE).strip().lower()
    if mode not in INPUT_GENERATOR_MODES:
        choices = ", ".join(sorted(INPUT_GENERATOR_MODES))
        raise ValueError(f"invalid {INPUT_GENERATOR_ENV}={raw_value!r}: expected one of {choices}")
    return mode


def dispatch_input(
    api_test,
    rules_by_api: Mapping[str, object] | None = None,
) -> bool:
    """通过 v2 规则注册表调度一个 APIConfig。"""
    # 调度逻辑保持短小，确保失败早于昂贵的绑定流程。
    mode = resolve_mode()
    api_config = api_test.api_config
    api_name = api_config.api_name

    rules_by_api = API_RULE_REGISTRY if rules_by_api is None else rules_by_api
    rule = rules_by_api.get(api_name)
    if rule is None:
        # 未注册 API 直接失败，避免旧时代的静默回退。
        record_dispatch(
            api_name,
            mode,
            "blocked",
            NO_RULE_ID,
            fallback_reason="no-registered-rule",
        )
        raise RuntimeError(f"no v2 input-generation rule registered for {api_name}")

    # 绑定是刻意延迟的：默认路径不会额外引入 import、签名解析、框架调用或 RNG 消耗。
    runtime_config = api_test.runtime_config
    context = build_ctx(
        api_config,
        seed=runtime_config.random_seed,
        runtime_mode=mode,
        use_torch=api_config.use_torch,
        gpu_enabled=runtime_config.gpu_mode.enabled,
    )
    fallback_reason = getattr(rule, "fallback_reason", lambda _context: None)(context)
    if fallback_reason:
        # 阻断规则在生成前失败，保证 RNG 状态不被污染。
        record_dispatch(
            api_name,
            mode,
            "blocked",
            rule.rule_id,
            fallback_reason=fallback_reason,
        )
        raise RuntimeError(f"v2 input-generation rule blocked for {api_name}: {fallback_reason}")
    record_dispatch(api_name, mode, "rule", rule.rule_id)
    return rule.generate(context, api_config)


# 兼容别名保留给包外旧导入，包内新代码应使用短命名。
record_legacy_generation = record_legacy
capture_legacy_generation = capture_legacy
resolve_input_generator_mode = resolve_mode
dispatch_input_generation = dispatch_input
