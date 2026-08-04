"""输入生成调度。"""

from __future__ import annotations

from .input_binding import build_input_context
from .input_registry import rules


def dispatch_input(api_test) -> bool:
    """通过规则注册表调度一个 APIConfig。"""

    api_config = api_test.api_config
    api_name = api_config.api_name

    rule = rules.resolve(api_name)

    context = build_input_context(api_config, seed=api_test.runtime_config.random_seed)
    return rule.generate(context, api_config)
