"""
paddle.nn.functional.max_pool1d / max_pool2d 的特殊对比逻辑。

问题背景
--------
max pooling 在 pooling window 内存在相同最大值（tie）时，XPU 和 GPU 选择不同位置
作为"赢家"。前向输出（池化值本身）完全相同，但反向梯度流向不同位置，导致
backward max_rel_diff=2.0，被误判为 accuracy_error。

对比策略
--------
Forward：
  - 当 return_mask=True 时，输出是 (out, mask)；只比较 out[0]（池化值）。
  - 当 return_mask=False 时，输出是普通 Tensor，直接 bitwise 比较。
  - 使用 np.testing.assert_array_equal（无需容差，两边池化值应完全一致）。

Backward：
  - local_grads / remote_grads 是只含一个元素的 list，元素形状与输入相同。
  - Tie-breaking 只改变梯度落点，不改变梯度总量。对于每个 pooling window，
    XPU 和 GPU 分配的梯度总量应该相同（只是落在不同的 tied 位置上）。
  - 策略：将展平后的梯度排序后进行对比。当所有 tied 元素接收到来自输出侧
    相同的梯度时，排序后的梯度向量在两边应该完全一致。
  - 使用 tester.atol / tester.rtol 作为容差。
"""

import numpy as np

from . import register_backward, register_forward


@register_forward(
    "paddle.nn.functional.max_pool1d",
    "paddle.nn.functional.max_pool2d",
)
def compare_max_pool_forward(local_output, remote_output, api_config, tester):
    """
    max_pool1d / max_pool2d 前向特殊对比。

    当 return_mask=True 时，输出为 (out, mask)；forward 值 out[0] 应 bitwise 相同。
    当 return_mask=False 时，输出为普通 Tensor；同样应 bitwise 相同。
    mask 不做对比（tie-breaking 导致两端 mask 不同是预期行为）。
    """
    # 提取 pooled 值（忽略 mask）
    if isinstance(local_output, (tuple, list)):
        local_out = local_output[0]
    else:
        local_out = local_output

    if isinstance(remote_output, (tuple, list)):
        remote_out = remote_output[0]
    else:
        remote_out = remote_output

    local_np = local_out.numpy()
    remote_np = remote_out.numpy()

    np.testing.assert_array_equal(
        local_np,
        remote_np,
        err_msg=(
            f"max_pool 前向对比失败（池化值不一致，非 tie-breaking 问题）："
            f"{api_config.config}"
        ),
    )


@register_backward(
    "paddle.nn.functional.max_pool1d",
    "paddle.nn.functional.max_pool2d",
)
def compare_max_pool_backward(local_grads, remote_grads, api_config, tester):
    """
    max_pool1d / max_pool2d 反向特殊对比。

    Tie-breaking 只影响梯度落点，排序后两端梯度应一致。
    local_grads / remote_grads 均为 list[Tensor]，每个 Tensor 形状与输入相同。
    """
    # 支持 list/tuple 和单个 Tensor
    if isinstance(local_grads, (list, tuple)):
        local_grad_list = list(local_grads)
    else:
        local_grad_list = [local_grads]

    if isinstance(remote_grads, (list, tuple)):
        remote_grad_list = list(remote_grads)
    else:
        remote_grad_list = [remote_grads]

    for i, (local_g, remote_g) in enumerate(zip(local_grad_list, remote_grad_list)):
        local_np = local_g.numpy()
        remote_np = remote_g.numpy()

        # 排序后对比：tie-breaking 只改变梯度落点，排好序后应一致
        local_sorted = np.sort(local_np.flatten())
        remote_sorted = np.sort(remote_np.flatten())

        atol = tester.atol
        rtol = tester.rtol

        max_abs_diff = np.max(np.abs(local_sorted - remote_sorted))
        tol = atol + rtol * np.abs(remote_sorted)
        if np.any(np.abs(local_sorted - remote_sorted) > tol):
            raise AssertionError(
                f"max_pool 反向对比失败（排序后梯度不一致，非 tie-breaking 问题）："
                f"{api_config.config}\n"
                f"grad[{i}] sorted max_abs_diff={max_abs_diff:.6g}, "
                f"atol={atol:.6g}, rtol={rtol:.6g}"
            )
