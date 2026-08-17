"""输入物化阶段的原地修改隔离策略。"""

from __future__ import annotations


def requires_inplace_input_copy(api_config):
    """判断 API 是否可能原地修改输入，因此必须断开框架间 storage。"""
    # 判定需与 accuracy 的 build_*_input 生命周期保持一致，避免输入别名泄漏。
    api_name = getattr(api_config, "api_name", "")
    return (api_name.endswith("_") and not api_name.endswith("__")) or api_name == (
        "paddle.Tensor.__setitem__"
    )


__all__ = ["requires_inplace_input_copy"]
