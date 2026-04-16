"""
非确定性 API 的跳过注册。

以下 API 的输出在两端之间天然不一致，不应进行精度对比：

- paddle.empty / paddle.Tensor.empty_like：分配未初始化内存，两端垃圾值不同。
- paddle.empty_like：同上。
- paddle.multinomial：随机采样，结果天然不确定。
"""

from . import register_skip


@register_skip(
    "paddle.empty",
    "paddle.empty_like",
    "paddle.Tensor.empty_like",
    "paddle.multinomial",
)
def _skip():
    ...
