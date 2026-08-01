"""输入生成调度。"""

from __future__ import annotations

from collections.abc import Mapping

from .input_bind import build_input_context
from .registry import API_RULE_REGISTRY, DEFAULT_INPUT_GENERATION_RULE


def dispatch_input(
    api_test,
    rules_by_api: Mapping[str, object] | None = None,
) -> bool:
    """通过规则注册表调度一个 APIConfig。"""

    api_config = api_test.api_config
    api_name = api_config.api_name

    rules_by_api = API_RULE_REGISTRY if rules_by_api is None else rules_by_api
    rule = rules_by_api.get(api_name)
    if rule is None:
        rule = DEFAULT_INPUT_GENERATION_RULE

    context = build_input_context(
        api_config,
        seed=api_test.runtime_config.random_seed,
        use_torch=api_config.use_torch,
        gpu_enabled=api_test.runtime_config.gpu_mode.enabled,
    )
    block_reason = getattr(rule, "block_reason", lambda _context: None)(context)
    if block_reason:
        raise RuntimeError(f"input-generation rule blocked for {api_name}: {block_reason}")

    return rule.generate(context, api_config)
