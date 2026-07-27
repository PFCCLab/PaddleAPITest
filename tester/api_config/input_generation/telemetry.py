"""Opt-in, context-local input generation telemetry."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


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


def record_legacy_generation(api_config, index=None, key=None, list_index=None):
    """Record one legacy generator call when capture is explicitly enabled."""
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
def capture_legacy_generation() -> Iterator[list[LegacyGenerationEvent]]:
    """Capture events in the current context without process-global mutation."""
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
