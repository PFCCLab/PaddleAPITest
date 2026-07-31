"""Compatibility dispatcher for legacy and case-level input generation."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .registry import API_RULE_REGISTRY
from .telemetry import record_dispatch

INPUT_GENERATOR_ENV = "PADDLEAPITEST_INPUT_GENERATOR"
INPUT_GENERATOR_MODES = frozenset({"v2"})
DEFAULT_INPUT_GENERATOR_MODE = "v2"
LEGACY_RULE_ID = "legacy"


def resolve_input_generator_mode(value: str | None = None) -> str:
    raw_value = os.environ.get(INPUT_GENERATOR_ENV, "") if value is None else value
    mode = str(raw_value or DEFAULT_INPUT_GENERATOR_MODE).strip().lower()
    if mode not in INPUT_GENERATOR_MODES:
        choices = ", ".join(sorted(INPUT_GENERATOR_MODES))
        raise ValueError(f"invalid {INPUT_GENERATOR_ENV}={raw_value!r}: expected one of {choices}")
    return mode


def dispatch_input_generation(
    api_test,
    rules_by_api: Mapping[str, object] | None = None,
) -> bool:
    """Dispatch one APIConfig through the v2 rule registry."""
    mode = resolve_input_generator_mode()
    api_config = api_test.api_config
    api_name = api_config.api_name

    rules_by_api = API_RULE_REGISTRY if rules_by_api is None else rules_by_api
    rule = rules_by_api.get(api_name)
    if rule is None:
        record_dispatch(
            api_name,
            mode,
            "blocked",
            LEGACY_RULE_ID,
            fallback_reason="no-registered-rule",
        )
        raise RuntimeError(f"no v2 input-generation rule registered for {api_name}")

    # Binding is intentionally lazy: default mode and v2 fallback do not add
    # imports, signature inspection, framework calls, or RNG consumption.
    from .binding import build_generation_context

    runtime_config = api_test.runtime_config
    context = build_generation_context(
        api_config,
        seed=runtime_config.random_seed,
        runtime_mode=mode,
        use_torch=api_config.use_torch,
        gpu_enabled=runtime_config.gpu_mode.enabled,
    )
    fallback_reason = getattr(rule, "fallback_reason", lambda _context: None)(context)
    if fallback_reason:
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
