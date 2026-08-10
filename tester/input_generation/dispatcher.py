"""输入生成调度。"""

from __future__ import annotations

from .binding import build_input_generation_context
from .generation_rules import input_rules

# dispatcher 只连接测试对象、绑定上下文和规则注册表，不拥有任何值域逻辑。
# 规则执行失败时异常原样上抛，由上层测试模式统一记录失败分类。


def dispatch_input_generation(api_test_case) -> bool:
    """通过规则注册表调度一个 APIConfig。"""

    api_config = api_test_case.api_config
    api_name = api_config.api_name

    input_rule = input_rules.resolve(api_name)

    # dispatcher 只传递已解析策略，规则不再拥有 backend 选择权。
    input_generation_context = build_input_generation_context(
        api_config,
        seed=api_test_case.runtime_config.random_seed,
        input_backend_policy=api_test_case.runtime_config.input_backend,
        # 旧开关在非 GPU mode 下缓存 NumPy canonical input；GPU-native 生成保持原语义。
        cache_enabled=(
            api_test_case.runtime_config.use_cached_numpy
            and not api_test_case.runtime_config.gpu_mode.enabled
        ),
    )
    return input_rule.generate(input_generation_context, api_config)
