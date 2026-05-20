"""
paddle.max / paddle.Tensor.max / paddle.Tensor.min 的特殊对比逻辑。

问题背景
--------
paddle.max(x, axis=...) 和 paddle.Tensor.max(axis=...) 对指定轴做 reduce_max，
返回一个值 Tensor（不是 (values, indices) 元组）。

当输入中存在多个相同的最大/最小值（ties）时，XPU 和 GPU 对反传梯度的处理方式
不同：
  - GPU (CUDA)：将 output_grad / k 均匀分配到所有 k 个 tied 位置
  - XPU：将 output_grad 广播到所有 tied 位置（每个都获得完整梯度）

两种做法都是合法的次梯度（subgradient），不影响最优化的正确性，但数值不同，
会导致直接比较误报。

对比策略
--------
Forward：
  - paddle.max 始终返回单个 Tensor（values），无平局歧义，用默认逻辑对比。

Backward：
  - 基于前向输出 max_val 和原始输入 x，找到所有 tied 位置（x == max_val）。
  - 验证两侧梯度的非零值都在 tied 位置上（不在 tied 位置的非零梯度 = 真实 bug）。
  - 不比较梯度的具体数值（不同实现对 tied 位置给出不同数值，均合法）。
"""

from __future__ import annotations

import numpy as np

from . import register_backward

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

_DYNAMIC_AXIS = object()  # sentinel: axis 是动态 Tensor，无法静态确定


def _get_axis(api_config):
    """从 api_config 中解析 axis 参数（返回原始值，含 int/list/tuple/None）。

    支持的配置格式：
      paddle.max(Tensor(...), axis=1, ...)        → kwargs["axis"] = 1
      paddle.max(Tensor(...), 1, ...)             → args[1] = 1
      paddle.Tensor.max(Tensor(...), -2, ...)     → args[1] = -2
      paddle.Tensor.max(Tensor(...), axis=1, ...) → kwargs["axis"] = 1
      paddle.max(Tensor(...), axis=None, ...)     → None（全局 reduce）
      paddle.max(Tensor(...), Tensor([2],"int64"), ...) → _DYNAMIC_AXIS（无法静态分析）
    """
    # 检查 kwargs["axis"] 是否是 TensorConfig（动态 axis）
    from tester.api_config.config_analyzer import TensorConfig

    if "axis" in api_config.kwargs:
        axis = api_config.kwargs["axis"]
        if isinstance(axis, TensorConfig):
            return _DYNAMIC_AXIS
        return axis

    if len(api_config.args) > 1:
        # args[0] is x (or self for Tensor methods), args[1] might be axis
        candidate = api_config.args[1]
        if isinstance(candidate, (int, list, tuple)):
            return candidate
        if isinstance(candidate, TensorConfig):
            return _DYNAMIC_AXIS

    return None  # 全局 reduce


def _normalize_axis(axis, ndim: int):
    """将 axis 统一为非负整数列表。

    返回：
      - list[int]  → 有效的 reduce 轴列表
      - None        → 全局 reduce（axis=None）
      - _DYNAMIC_AXIS → 动态 axis（无法静态分析，应跳过比较）
    """
    if axis is _DYNAMIC_AXIS:
        return _DYNAMIC_AXIS
    if axis is None:
        return list(range(ndim))  # 全局 reduce
    if isinstance(axis, int):
        axes = [axis]
    elif isinstance(axis, (list, tuple)):
        # 过滤掉 axis 列表中含 TensorConfig 的元素（动态 axis 无法静态分析）
        int_axes = []
        for a in axis:
            if isinstance(a, int):
                int_axes.append(a)
            else:
                # Contains non-int → fall back to dynamic
                return _DYNAMIC_AXIS
        axes = int_axes
    else:
        return _DYNAMIC_AXIS
    return [a % ndim if ndim > 0 else a for a in axes]


def _check_grads_at_tied_positions(
    local_np: np.ndarray,
    remote_np: np.ndarray,
    input_np: np.ndarray,
    reduce_axes: list[int],
    atol: float,
    label: str,
    api_config_str: str,
    is_min: bool = False,
):
    """
    验证两侧的梯度非零值都落在 tied（最大/最小值）位置上。

    策略：
    1. 沿 reduce_axes 计算 input_np 的 max/min，得到 extreme_vals（keepdim=True）。
    2. 计算 tied_mask = (input_np == extreme_vals)，形状同 input_np。
    3. 检查 local_np 和 remote_np 中所有 "显著非零" 的位置是否都在 tied_mask 内。
       "显著非零" = |grad| > atol（避免浮点噪声干扰）。
    """
    # 计算 extreme_vals
    if is_min:
        extreme_vals = np.min(input_np, axis=tuple(reduce_axes), keepdims=True)
    else:
        extreme_vals = np.max(input_np, axis=tuple(reduce_axes), keepdims=True)

    tied_mask = (input_np == extreme_vals)  # bool array, same shape as input_np

    # 找出显著非零的梯度位置
    # 使用 atol 作为"显著"阈值，避免 float16 舍入误差产生的小噪声
    threshold = max(atol, 1e-7)

    local_nonzero = np.abs(local_np.astype(np.float64)) > threshold
    remote_nonzero = np.abs(remote_np.astype(np.float64)) > threshold

    # 非零梯度不在 tied 位置 = 真实 bug
    local_invalid = local_nonzero & ~tied_mask
    remote_invalid = remote_nonzero & ~tied_mask

    if np.any(local_invalid):
        bad_val = float(np.max(np.abs(local_np[local_invalid].astype(np.float64))))
        count = int(np.sum(local_invalid))
        raise AssertionError(
            f"{label} [local/XPU] 梯度非零位置不在 tied 区域（count={count}, "
            f"max_abs={bad_val:.6g}）：{api_config_str}\n"
            "这表明 XPU 将梯度传到了非最大值位置，可能存在真实 bug。"
        )

    if np.any(remote_invalid):
        bad_val = float(np.max(np.abs(remote_np[remote_invalid].astype(np.float64))))
        count = int(np.sum(remote_invalid))
        raise AssertionError(
            f"{label} [remote/GPU] 梯度非零位置不在 tied 区域（count={count}, "
            f"max_abs={bad_val:.6g}）：{api_config_str}\n"
            "这表明 GPU 将梯度传到了非最大值位置，可能存在真实 bug。"
        )


# ---------------------------------------------------------------------------
# Backward 特殊对比
# ---------------------------------------------------------------------------

@register_backward("paddle.max", "paddle.Tensor.max", "paddle.Tensor.min")
def compare_reduce_max_min_backward(local_grads, remote_grads, api_config, tester):
    """
    paddle.max / Tensor.max / Tensor.min 反向特殊对比。

    验证两侧梯度的非零值都在 tied 位置（最大/最小值所在位置），
    不比较具体数值（不同平台对 tied 位置的分配策略不同，但都合法）。
    """
    is_min = "min" in api_config.api_name

    # 获取输入张量
    if tester.paddle_args:
        input_tensor = tester.paddle_args[0]
    elif "x" in tester.paddle_kwargs:
        input_tensor = tester.paddle_kwargs["x"]
    else:
        raise AssertionError(
            "compare_reduce_max_min_backward: 无法获取原始输入张量"
        )

    input_np = input_tensor.numpy()
    ndim = input_np.ndim

    # 解析 axis
    raw_axis = _get_axis(api_config)
    reduce_axes = _normalize_axis(raw_axis, ndim)

    # local_grads / remote_grads 是 list[Tensor | None]
    if not isinstance(local_grads, (list, tuple)):
        local_grads = [local_grads]
    if not isinstance(remote_grads, (list, tuple)):
        remote_grads = [remote_grads]

    import paddle as _paddle

    for i, (local_g, remote_g) in enumerate(zip(local_grads, remote_grads)):
        if local_g is None and remote_g is None:
            continue
        if local_g is None or remote_g is None:
            raise AssertionError(
                f"reduce_max_min backward gradient[{i}] 一侧为 None，另一侧不为 None：{api_config.config}"
            )
        if not isinstance(local_g, _paddle.Tensor):
            continue

        local_np = local_g.numpy()
        remote_np = remote_g.numpy()

        dtype_str = str(local_g.dtype).split(".")[-1]
        atol, _ = tester._resolve_atol_rtol(dtype_str)

        if reduce_axes is _DYNAMIC_AXIS:
            # 动态 axis（TensorConfig），无法静态分析，跳过梯度比较
            print(
                f"[compare] reduce_max_min backward gradient[{i}] 动态 axis，跳过比较：{api_config.config}",
                flush=True,
            )
            continue

        if local_np.ndim == 0 or not reduce_axes:
            # 0-dim 或无 reduce 轴：直接比较（无 tied 问题）
            a = local_np.astype(np.float64)
            b = remote_np.astype(np.float64)
            abs_diff = np.abs(a - b)
            if np.any(abs_diff > atol):
                max_abs = float(np.nanmax(abs_diff))
                raise AssertionError(
                    f"reduce_max_min backward gradient[{i}] 直接比较失败 "
                    f"(max_abs_diff={max_abs:.6g})：{api_config.config}"
                )
        else:
            _check_grads_at_tied_positions(
                local_np=local_np,
                remote_np=remote_np,
                input_np=input_np,
                reduce_axes=reduce_axes,
                atol=atol,
                label=f"Backward gradient[{i}]",
                api_config_str=api_config.config,
                is_min=is_min,
            )
