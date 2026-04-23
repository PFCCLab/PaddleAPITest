"""
paddle.sort / paddle.Tensor.sort 的特殊对比逻辑。

问题背景
--------
sort 在输入存在相同值（ties）时，XPU 和 GPU 打破平局的方式不同，会产生
合法但不同的排序索引。前向输出（排序后的值本身）两边完全一致。但反向时，
梯度通过 sort indices 散射回输入位置：tied 位置的索引不同，导致梯度落在
不同位置，被误判为 accuracy_error。

API 说明
--------
paddle.sort(x, axis=-1, ...) 返回单个 Tensor（排序后的值），不返回索引。
paddle.Tensor.sort 是同一函数的 Tensor 方法形式。

对比策略
--------
Forward：
  - 两边均为排序后的值 Tensor，应 bitwise 完全相同。
  - 使用 np.testing.assert_array_equal（无容差）。

Backward：
  - local_grads / remote_grads 是 list[Tensor]，每个 Tensor 形状与输入相同。
  - Tie-breaking 只改变梯度落点，不改变梯度值集合。
  - 策略：将梯度沿 sort axis 排序后对比。
    对于每个 "fiber"（沿 axis 切出的一维切片），排序后的梯度值集合在两边应一致。
  - 若输入为 0-dim tensor（ndim==0），直接对比，无需排序。
"""

import numpy as np

from . import register_backward, register_forward


@register_forward("paddle.sort", "paddle.Tensor.sort")
def compare_sort_forward(local_output, remote_output, api_config, tester):
    """
    sort 前向特殊对比：直接比较排序后的值（bitwise）。

    paddle.sort 返回排序后的值 Tensor，两边输入相同，排序值应完全一致。
    """
    # Handle tuple/list output defensively (in case of future API changes)
    if isinstance(local_output, (tuple, list)):
        local_vals = local_output[0]
    else:
        local_vals = local_output

    if isinstance(remote_output, (tuple, list)):
        remote_vals = remote_output[0]
    else:
        remote_vals = remote_output

    local_np = local_vals.numpy()
    remote_np = remote_vals.numpy()

    np.testing.assert_array_equal(
        local_np,
        remote_np,
        err_msg=(
            f"sort 前向对比失败（排序值不一致，可能存在真实精度问题）："
            f"{api_config.config}"
        ),
    )


@register_backward("paddle.sort", "paddle.Tensor.sort")
def compare_sort_backward(local_grads, remote_grads, api_config, tester):
    """
    sort 反向特殊对比：沿 sort axis 排序梯度后对比。

    Tie-breaking 只影响梯度落点，不改变梯度值集合。对每个 fiber 排序后，
    两边应拥有完全相同的梯度值集合。
    """
    # 获取原始输入张量（用于 ndim 信息）
    if tester.paddle_args:
        input_tensor = tester.paddle_args[0]
    elif "x" in tester.paddle_kwargs:
        input_tensor = tester.paddle_kwargs["x"]
    else:
        raise AssertionError(
            "compare_sort_backward: paddle_args 为空且 paddle_kwargs 中没有 'x'，"
            "无法获取原始输入张量"
        )

    input_ndim = input_tensor.numpy().ndim

    # 提取 axis 参数
    # paddle.sort 签名：sort(x, axis=-1, descending=False, stable=False)
    axis = api_config.kwargs.get("axis", None)
    if axis is None and len(api_config.args) > 1 and isinstance(api_config.args[1], int):
        axis = api_config.args[1]
    if axis is None:
        axis = -1
    if isinstance(axis, int) and axis < 0:
        axis = axis + input_ndim

    # 标准化 local_grads / remote_grads 为 list
    if isinstance(local_grads, (list, tuple)):
        local_grad_list = list(local_grads)
    else:
        local_grad_list = [local_grads]

    if isinstance(remote_grads, (list, tuple)):
        remote_grad_list = list(remote_grads)
    else:
        remote_grad_list = [remote_grads]

    for i, (local_g, remote_g) in enumerate(zip(local_grad_list, remote_grad_list)):
        if local_g is None and remote_g is None:
            continue
        if local_g is None or remote_g is None:
            raise AssertionError(
                f"sort 反向对比失败：grad[{i}] 一侧为 None，另一侧非 None；"
                f"{api_config.config}"
            )

        local_np = local_g.numpy()
        remote_np = remote_g.numpy()

        # 0-dim tensor：直接对比，无需排序
        if local_np.ndim == 0:
            atol = tester.atol
            rtol = tester.rtol
            if not np.allclose(local_np, remote_np, atol=atol, rtol=rtol, equal_nan=True):
                raise AssertionError(
                    f"sort 反向对比失败（0-dim gradient 不一致）："
                    f"{api_config.config}"
                )
            continue

        # 沿 sort axis 排序梯度后对比
        # np.sort 对每个 fiber（axis 方向的切片）独立排序
        local_sorted = np.sort(local_np, axis=axis)
        remote_sorted = np.sort(remote_np, axis=axis)

        atol = tester.atol
        rtol = tester.rtol

        abs_diff = np.abs(local_sorted.astype(np.float64) - remote_sorted.astype(np.float64))
        max_abs_diff = float(np.nanmax(abs_diff)) if abs_diff.size > 0 else 0.0
        tol = atol + rtol * np.abs(remote_sorted.astype(np.float64))

        if abs_diff.size > 0 and np.any(abs_diff > tol):
            raise AssertionError(
                f"sort 反向对比失败（排序后梯度不一致，非 tie-breaking 问题）："
                f"{api_config.config}\n"
                f"grad[{i}] sorted max_abs_diff={max_abs_diff:.6g}, "
                f"atol={atol:.6g}, rtol={rtol:.6g}"
            )
