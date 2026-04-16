"""
paddle.argsort / paddle.Tensor.argsort 的特殊对比逻辑。

问题背景
--------
argsort 在输入存在相同值（ties）时，XPU 和 GPU 打破平局的方式不同，会产生
合法但不同的排序索引，导致逐元素对比误判为 accuracy_error。

对比策略
--------
不直接比较索引，而是用两边的索引分别 gather 原始输入的值，再对比 gather 后的值。
如果两边的排序都是合法的，gather 后的值应完全一致（bitwise），因为是对同一数组的
两种等价排列。
"""

import numpy as np

from . import register_forward


@register_forward("paddle.argsort", "paddle.Tensor.argsort")
def compare_argsort_forward(local_output, remote_output, api_config, tester):
    """
    argsort 前向特殊对比：gather 原始输入值后比较，而非直接比较索引。

    tester.paddle_args[0] 是排序前的输入张量，在 _run_paddle 执行后仍保留在实例上。
    由于两侧使用相同 random_seed 生成输入，GPU 侧与 XPU 侧输入数据完全相同，
    因此可用 XPU 侧的输入张量作为两边 gather 的数据源。
    """
    if not tester.paddle_args:
        raise AssertionError(
            "compare_argsort_forward: tester.paddle_args 为空，无法获取原始输入张量"
        )

    input_np = tester.paddle_args[0].cpu().numpy()

    # 提取 axis 参数
    # paddle.argsort 签名：argsort(x, axis=-1, descending=False, stable=False)
    axis = api_config.kwargs.get("axis", None)
    if axis is None and len(api_config.args) > 1 and isinstance(api_config.args[1], int):
        axis = api_config.args[1]
    if axis is None:
        axis = -1
    if isinstance(axis, int) and axis < 0:
        axis = axis + input_np.ndim

    local_indices_np = local_output.numpy()    # int64，形状与输入相同
    remote_indices_np = remote_output.numpy()  # int64，形状与输入相同

    # 用各自的索引 gather 原始输入值
    local_vals = np.take_along_axis(input_np, local_indices_np, axis=axis)
    remote_vals = np.take_along_axis(input_np, remote_indices_np, axis=axis)

    # 两边都是对同一数组的合法排序，gather 后的值应 bitwise 完全相同
    # （gather 本身只是索引查找，不涉及浮点运算，无需容差）
    np.testing.assert_array_equal(
        local_vals,
        remote_vals,
        err_msg=(
            f"argsort 特殊对比失败（gather values 不一致）：{api_config.config}\n"
            "这说明两侧排序结果不是同一输入的等价排列，可能存在真实精度问题。"
        ),
    )


# argsort 输出为整数索引，不涉及浮点梯度，无需注册 backward 特殊对比。
