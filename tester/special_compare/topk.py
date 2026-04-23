"""
paddle.topk / paddle.Tensor.topk 的特殊对比逻辑。

问题背景
--------
topk 在输入存在相同值（ties）时，XPU 和 GPU 打破平局的方式不同，会导致：
1. 两侧 values 输出中相同 slot 存放的是不同（但等价）的元素——尤其在 sorted=False 时。
2. 两侧 indices 输出永远可能不同（tie 时合法索引不唯一）。

对比策略
--------
**Forward**：
  - 对两侧的 values 输出沿 topk 轴排序（升序），再逐元素比较。
    排序后，两侧的合法 top-k 集合应当完全相同（bitwise），因为值来自
    同一输入张量，只是顺序/选取位置不同。
  - indices 输出完全忽略（tie 时合法索引不唯一）。

**Backward**：
  - topk 的反向梯度是将 upstream_grad scatter 回 x 的对应位置。
    当 tie 导致两侧选取不同索引时，梯度也会 scatter 到不同位置，
    逐元素对比会失败。
  - 解决方法：对两侧输入梯度沿 topk 轴排序后比较。
    如果两侧 top-k 集合相同（Forward 已验证），则排序后的梯度也应相同。
"""

from __future__ import annotations

import numpy as np

from . import register_backward, register_forward


def _get_topk_axis(api_config, input_ndim: int) -> int:
    """从 api_config 中提取 topk 的 axis/dim 参数，归一化为非负整数。

    paddle.topk 签名：topk(x, k, axis=None, largest=True, sorted=True, ...)
      - axis=None 时默认取最后一轴（等价于 axis=-1）
      - 也接受 dim 作为别名（Tensor.topk / torch-style alias）

    对 paddle.Tensor.topk，args 的位置参数顺序为：
      args[0]=x, args[1]=k, args[2]=axis, args[3]=largest, args[4]=sorted
    """
    kwargs = api_config.kwargs
    args = api_config.args

    # 优先从 kwargs 中读取（两个可能的名称：axis / dim）
    axis = kwargs.get("axis", None)
    if axis is None:
        axis = kwargs.get("dim", None)

    # 其次从位置参数中读取（args[2] 对应 axis，适用于 Tensor.topk 的 positional 形式）
    if axis is None and len(args) > 2:
        candidate = args[2]
        if isinstance(candidate, int):
            axis = candidate

    # 默认最后一轴
    if axis is None:
        axis = -1

    # 将负数轴转换为非负数
    if isinstance(axis, int) and axis < 0:
        axis = axis + input_ndim

    return int(axis)


@register_forward("paddle.topk", "paddle.Tensor.topk")
def compare_topk_forward(local_output, remote_output, api_config, tester):
    """
    topk 前向特殊对比：对两侧的 values 输出沿 topk 轴排序后比较，忽略 indices。

    local_output / remote_output 均为 TopKRetType（长度为 2 的具名元组），
    其中 [0] 为 values，[1] 为 indices。
    """
    local_vals = local_output[0]
    remote_vals = remote_output[0]

    local_np = local_vals.numpy()
    remote_np = remote_vals.numpy()

    # 0-dim 或 empty tensor：直接比较值（无 tie 问题）
    if local_np.ndim == 0 or local_np.size == 0:
        np.testing.assert_array_equal(
            local_np,
            remote_np,
            err_msg=(
                f"topk 特殊对比失败（0-dim/empty tensor）：{api_config.config}"
            ),
        )
        return

    axis = _get_topk_axis(api_config, local_np.ndim)

    # 沿 topk 轴排序（升序），消除 tie-breaking 和 sorted=False 引入的顺序差异
    local_sorted = np.sort(local_np, axis=axis)
    remote_sorted = np.sort(remote_np, axis=axis)

    # values 来自同一输入，bitwise 应完全一致（gather 不涉及浮点运算）
    np.testing.assert_array_equal(
        local_sorted,
        remote_sorted,
        err_msg=(
            f"topk 特殊对比失败（排序后的 values 不一致）：{api_config.config}\n"
            "这说明两侧的 top-k 集合不同，可能存在真实精度问题（非 tie-breaking 引起）。"
        ),
    )


@register_backward("paddle.topk", "paddle.Tensor.topk")
def compare_topk_backward(local_grads, remote_grads, api_config, tester):
    """
    topk 反向特殊对比：对两侧输入梯度沿 topk 轴排序后比较。

    topk 的反向梯度是将 upstream_grad scatter 回 x 的对应位置。
    Tie-breaking 导致两侧 scatter 位置不同，但排序后应相同。

    local_grads / remote_grads 是 list[Tensor]，通常只有一个元素（x 的梯度）。
    """
    # 统一为列表方便逐一处理
    if isinstance(local_grads, (list, tuple)):
        local_list = list(local_grads)
        remote_list = list(remote_grads)
    else:
        local_list = [local_grads]
        remote_list = [remote_grads]

    for i, (local_g, remote_g) in enumerate(zip(local_list, remote_list)):
        # 允许其中一侧梯度为 None（unused input）
        if local_g is None and remote_g is None:
            continue
        if local_g is None or remote_g is None:
            # 一侧有梯度一侧没有，属于真实差异
            raise AssertionError(
                f"topk backward 特殊对比失败（梯度[{i}] 一侧为 None，另一侧不为 None）："
                f"{api_config.config}"
            )

        local_np = local_g.numpy()
        remote_np = remote_g.numpy()

        # 0-dim 或 empty：直接数值对比
        if local_np.ndim == 0 or local_np.size == 0:
            atol, rtol = tester.atol, tester.rtol
            np.testing.assert_allclose(
                local_np,
                remote_np,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                err_msg=(
                    f"topk backward 特殊对比失败（梯度[{i}]，0-dim/empty）：{api_config.config}"
                ),
            )
            continue

        axis = _get_topk_axis(api_config, local_np.ndim)

        # 沿 topk 轴排序：scatter 位置不同但值集合相同时，排序后应一致
        local_sorted = np.sort(local_np, axis=axis)
        remote_sorted = np.sort(remote_np, axis=axis)

        atol, rtol = tester.atol, tester.rtol
        np.testing.assert_allclose(
            local_sorted,
            remote_sorted,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
            err_msg=(
                f"topk backward 特殊对比失败（梯度[{i}] 排序后不一致）：{api_config.config}\n"
                "这说明两侧 scatter 的梯度值集合不同，可能存在真实精度问题。"
            ),
        )
