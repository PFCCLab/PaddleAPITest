"""
随机/非确定性算子的跳过逻辑。

包含两类处理策略：

1. 无条件跳过（unconditional skip）
   以下 API 在 XPU 与 GPU 之间使用各自独立的随机数生成器，输出天然不一致，
   无论参数如何均应跳过精度对比：

   - paddle.normal / paddle.standard_normal / paddle.log_normal
   - paddle.poisson / paddle.bernoulli / paddle.standard_gamma / paddle.binomial
   - paddle.Tensor.normal_ / paddle.Tensor.exponential_ / paddle.Tensor.cauchy_
   - paddle.Tensor.geometric_ / paddle.Tensor.log_normal_ / paddle.Tensor.bernoulli_
   - paddle.nn.functional.gumbel_softmax
   - paddle.geometric.sample_neighbors
   - paddle.nn.functional.fractional_max_pool2d / fractional_max_pool3d

2. 条件跳过（conditional skip）
   仅当 training=True 时输出随机，training=False 时完全确定：

   - paddle.nn.functional.dropout
   - paddle.nn.functional.dropout2d
   - paddle.nn.functional.dropout3d
   - paddle.nn.functional.alpha_dropout

   对于 paddle.incubate.nn.functional.fused_dropout_add，仅当
   training=True 且 p > 0 时跳过。
"""

from __future__ import annotations

import numpy as np

from . import SkipComparison, register_backward, register_forward, register_skip

# ---------------------------------------------------------------------------
# 1. 无条件跳过
# ---------------------------------------------------------------------------


@register_skip(
    "paddle.normal",
    "paddle.standard_normal",
    "paddle.log_normal",
    "paddle.poisson",
    "paddle.bernoulli",
    "paddle.standard_gamma",
    "paddle.binomial",
    "paddle.Tensor.normal_",
    "paddle.Tensor.exponential_",
    "paddle.Tensor.cauchy_",
    "paddle.Tensor.geometric_",
    "paddle.Tensor.log_normal_",
    "paddle.Tensor.bernoulli_",
    "paddle.nn.functional.gumbel_softmax",
    "paddle.geometric.sample_neighbors",
    "paddle.nn.functional.fractional_max_pool2d",
    "paddle.nn.functional.fractional_max_pool3d",
)
def _skip_random_ops():
    ...


# ---------------------------------------------------------------------------
# 2. 条件跳过辅助函数
# ---------------------------------------------------------------------------


def _get_training(api_config, training_arg_index: int) -> bool:
    """
    从 api_config 中读取 training 参数的布尔值。

    优先从 kwargs 中读取；若不存在则尝试 args[training_arg_index]；
    若两者均无则按默认值 True 处理（所有 dropout 变体的默认值均为 True）。

    参数
    ----
    api_config        : APIConfig 实例
    training_arg_index: training 在位置参数列表中的 0-base 下标
    """
    if "training" in api_config.kwargs:
        return bool(api_config.kwargs["training"])
    if len(api_config.args) > training_arg_index:
        return bool(api_config.args[training_arg_index])
    # 默认 True
    return True


def _get_p(api_config, p_arg_index: int) -> float:
    """
    从 api_config 中读取 p 参数的浮点值。

    若 p 是 Tensor（TensorConfig 实例），无法静态判断大小，按 p > 0 处理（保守策略）。
    若两者均无则按默认值 0.5 处理。
    """
    p_val = api_config.kwargs.get("p", None)
    if p_val is None and len(api_config.args) > p_arg_index:
        p_val = api_config.args[p_arg_index]
    if p_val is None:
        return 0.5  # 默认值
    # 若 p 是 TensorConfig，无法静态判断，保守地视为 > 0
    try:
        return float(p_val)
    except (TypeError, ValueError):
        return 0.5  # 保守策略：视为 0.5 > 0


def _compare_tensors(local_output, remote_output, api_config, tester):
    """
    对两个 Tensor（或 Tensor 列表/元组）执行 assert_allclose 对比。
    """
    def _to_np(t):
        if hasattr(t, "numpy"):
            return t.numpy()
        return np.array(t)

    if isinstance(local_output, (list, tuple)):
        if len(local_output) != len(remote_output):
            raise AssertionError(
                f"输出数量不一致: local={len(local_output)}, remote={len(remote_output)}, "
                f"config={api_config.config}"
            )
        for i, (l, r) in enumerate(zip(local_output, remote_output)):
            np.testing.assert_allclose(
                _to_np(l),
                _to_np(r),
                atol=tester.atol,
                rtol=tester.rtol,
                err_msg=f"输出[{i}] 不一致: {api_config.config}",
            )
    else:
        np.testing.assert_allclose(
            _to_np(local_output),
            _to_np(remote_output),
            atol=tester.atol,
            rtol=tester.rtol,
            err_msg=f"输出不一致: {api_config.config}",
        )


# ---------------------------------------------------------------------------
# 3. paddle.nn.functional.dropout
#    签名: dropout(x, p=0.5, axis=None, training=True, inplace=False,
#                  mode='upscale_in_train', name=None)
#    training 在位置参数列表中的 0-base 下标 = 3
# ---------------------------------------------------------------------------

_DROPOUT_TRAINING_IDX = 3


@register_forward(
    "paddle.nn.functional.dropout",
)
def _compare_dropout_forward(local_output, remote_output, api_config, tester):
    if _get_training(api_config, _DROPOUT_TRAINING_IDX):
        raise SkipComparison(
            "paddle.nn.functional.dropout forward: training=True 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_output, remote_output, api_config, tester)


@register_backward(
    "paddle.nn.functional.dropout",
)
def _compare_dropout_backward(local_grads, remote_grads, api_config, tester):
    if _get_training(api_config, _DROPOUT_TRAINING_IDX):
        raise SkipComparison(
            "paddle.nn.functional.dropout backward: training=True 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_grads, remote_grads, api_config, tester)


# ---------------------------------------------------------------------------
# 4. paddle.nn.functional.dropout2d / dropout3d
#    签名: dropout2d(x, p=0.5, training=True, data_format='NCHW', name=None)
#           dropout3d(x, p=0.5, training=True, data_format='NCDHW', name=None)
#    training 位于 0-base 下标 = 2
# ---------------------------------------------------------------------------

_DROPOUT2D3D_TRAINING_IDX = 2


@register_forward(
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
)
def _compare_dropout2d3d_forward(local_output, remote_output, api_config, tester):
    if _get_training(api_config, _DROPOUT2D3D_TRAINING_IDX):
        raise SkipComparison(
            f"{api_config.api_name} forward: training=True 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_output, remote_output, api_config, tester)


@register_backward(
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
)
def _compare_dropout2d3d_backward(local_grads, remote_grads, api_config, tester):
    if _get_training(api_config, _DROPOUT2D3D_TRAINING_IDX):
        raise SkipComparison(
            f"{api_config.api_name} backward: training=True 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_grads, remote_grads, api_config, tester)


# ---------------------------------------------------------------------------
# 5. paddle.nn.functional.alpha_dropout
#    签名: alpha_dropout(x, p=0.5, training=True, name=None)
#    training 位于 0-base 下标 = 2
# ---------------------------------------------------------------------------

_ALPHA_DROPOUT_TRAINING_IDX = 2


@register_forward(
    "paddle.nn.functional.alpha_dropout",
)
def _compare_alpha_dropout_forward(local_output, remote_output, api_config, tester):
    if _get_training(api_config, _ALPHA_DROPOUT_TRAINING_IDX):
        raise SkipComparison(
            "paddle.nn.functional.alpha_dropout forward: training=True 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_output, remote_output, api_config, tester)


@register_backward(
    "paddle.nn.functional.alpha_dropout",
)
def _compare_alpha_dropout_backward(local_grads, remote_grads, api_config, tester):
    if _get_training(api_config, _ALPHA_DROPOUT_TRAINING_IDX):
        raise SkipComparison(
            "paddle.nn.functional.alpha_dropout backward: training=True 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_grads, remote_grads, api_config, tester)


# ---------------------------------------------------------------------------
# 6. paddle.incubate.nn.functional.fused_dropout_add
#    签名: fused_dropout_add(x, y, p=0.5, training=True, mode=..., name=None)
#    training 位于 0-base 下标 = 3
#    p 位于 0-base 下标 = 2
#    仅当 training=True 且 p > 0 时跳过。
# ---------------------------------------------------------------------------

_FUSED_DA_TRAINING_IDX = 3
_FUSED_DA_P_IDX = 2


def _fused_dropout_add_is_random(api_config) -> bool:
    training = _get_training(api_config, _FUSED_DA_TRAINING_IDX)
    if not training:
        return False
    p = _get_p(api_config, _FUSED_DA_P_IDX)
    return p > 0.0


@register_forward(
    "paddle.incubate.nn.functional.fused_dropout_add",
)
def _compare_fused_dropout_add_forward(local_output, remote_output, api_config, tester):
    if _fused_dropout_add_is_random(api_config):
        raise SkipComparison(
            "paddle.incubate.nn.functional.fused_dropout_add forward: "
            "training=True 且 p>0 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_output, remote_output, api_config, tester)


@register_backward(
    "paddle.incubate.nn.functional.fused_dropout_add",
)
def _compare_fused_dropout_add_backward(local_grads, remote_grads, api_config, tester):
    if _fused_dropout_add_is_random(api_config):
        raise SkipComparison(
            "paddle.incubate.nn.functional.fused_dropout_add backward: "
            "training=True 且 p>0 时随机掩码不可跨设备对比"
        )
    _compare_tensors(local_grads, remote_grads, api_config, tester)
