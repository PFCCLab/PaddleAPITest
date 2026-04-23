"""
特征向量 / 奇异向量符号歧义的特殊对比逻辑。

问题背景
--------
对于对称矩阵的特征向量 v，若 A @ v = λ·v，则 -v 也满足同样的方程。
因此特征向量的符号是任意的（v 和 -v 均合法）。
类似地，对于奇异值分解 A = U @ diag(S) @ Vh，若将 U 的某列乘以 -1，
只需同时将 Vh 对应行也乘以 -1，分解结果保持不变。

XPU 和 GPU 独立选择各自的符号约定，导致逐元素比较时差异高达 2.0（diff = v - (-v) = 2v），
即使两侧结果都完全正确。

涉及 API
--------
- paddle.linalg.eigh(x, UPLO) → (eigenvalues, eigenvectors)
    eigenvalues：实数标量，符号确定，正常比较。
    eigenvectors：列向量符号任意，对比 |V| 而非 V。

- paddle.linalg.svd(x, full_matrices) → (U, S, Vh)
    S：奇异值，非负且唯一，正常比较。
    U：列向量符号任意，对比 |U|。
    Vh：行向量符号任意，对比 |Vh|。
    Backward：SVD 反向梯度依赖 U/Vh 的符号约定，XPU/GPU 符号不同时梯度不同。
              → 反向梯度跳过比较（skip）。

- paddle.linalg.svd_lowrank(x, q, niter, M) → (U, S, Vh)
    svd_lowrank 使用随机化投影算法，输出本身是不确定的近似值。
    奇异值、U、Vh 在 XPU/GPU 上使用独立的随机数，结果无法对比。
    → 无条件跳过。
"""

from __future__ import annotations

import numpy as np

from . import SkipComparison, register_backward, register_forward, register_skip


def _to_np(t) -> np.ndarray:
    """paddle.Tensor → numpy array，复数转 complex128，实数转 float64。"""
    arr = t.numpy()
    if np.issubdtype(arr.dtype, np.complexfloating):
        return arr.astype(np.complex128)
    return arr.astype(np.float64)


def _assert_allclose_abs(local_t, remote_t, atol: float, rtol: float, label: str, config_str: str):
    """比较 |local_t| 与 |remote_t|，容差 atol/rtol。"""
    local_np = np.abs(_to_np(local_t))
    remote_np = np.abs(_to_np(remote_t))
    try:
        np.testing.assert_allclose(local_np, remote_np, atol=atol, rtol=rtol)
    except AssertionError as e:
        raise AssertionError(
            f"{label} |absolute value| 比较失败（atol={atol}, rtol={rtol}）："
            f"{config_str}\n{e}"
        ) from None


def _assert_allclose(local_t, remote_t, atol: float, rtol: float, label: str, config_str: str):
    """直接比较 local_t 与 remote_t，容差 atol/rtol。"""
    local_np = _to_np(local_t)
    remote_np = _to_np(remote_t)
    try:
        np.testing.assert_allclose(local_np, remote_np, atol=atol, rtol=rtol)
    except AssertionError as e:
        raise AssertionError(
            f"{label} 精度比较失败（atol={atol}, rtol={rtol}）："
            f"{config_str}\n{e}"
        ) from None


# ---------------------------------------------------------------------------
# paddle.linalg.svd_lowrank → 无条件跳过（随机化算法）
# ---------------------------------------------------------------------------

@register_skip("paddle.linalg.svd_lowrank")
def _skip_svd_lowrank():
    ...


# ---------------------------------------------------------------------------
# paddle.linalg.eigh → (eigenvalues, eigenvectors)
# ---------------------------------------------------------------------------

@register_forward("paddle.linalg.eigh")
def compare_eigh_forward(local_output, remote_output, api_config, tester):
    """
    eigh 前向特殊对比。

    输出格式：(eigenvalues: Tensor, eigenvectors: Tensor)
      - eigenvalues：实数，符号确定，正常比较。
      - eigenvectors：符号任意，对比 |V|。
    """
    if not isinstance(local_output, (list, tuple)) or len(local_output) < 2:
        dtype_str = str(local_output.dtype).split(".")[-1]
        atol, rtol = tester._resolve_atol_rtol(dtype_str)
        _assert_allclose(local_output, remote_output, atol, rtol,
                         "eigh single output", api_config.config)
        return

    eigenvalues_local, eigenvectors_local = local_output[0], local_output[1]
    eigenvalues_remote, eigenvectors_remote = remote_output[0], remote_output[1]

    dtype_str = str(eigenvalues_local.dtype).split(".")[-1]
    atol, rtol = tester._resolve_atol_rtol(dtype_str)

    # 1. 特征值：直接比较
    _assert_allclose(eigenvalues_local, eigenvalues_remote, atol, rtol,
                     "eigh eigenvalues", api_config.config)

    # 2. 特征向量：|V| 比较（消除符号歧义）
    _assert_allclose_abs(eigenvectors_local, eigenvectors_remote, atol, rtol,
                         "eigh eigenvectors", api_config.config)


# ---------------------------------------------------------------------------
# paddle.linalg.svd → (U, S, Vh)
# ---------------------------------------------------------------------------

@register_forward("paddle.linalg.svd")
def compare_svd_forward(local_output, remote_output, api_config, tester):
    """
    svd 前向特殊对比。

    输出格式：(U: Tensor, S: Tensor, Vh: Tensor)
      - S：奇异值，非负且唯一，正常比较。
      - U：列向量符号任意，对比 |U|。
      - Vh：行向量符号任意，对比 |Vh|。
    """
    if not isinstance(local_output, (list, tuple)) or len(local_output) < 3:
        dtype_str = str(local_output.dtype).split(".")[-1]
        atol, rtol = tester._resolve_atol_rtol(dtype_str)
        _assert_allclose(local_output, remote_output, atol, rtol,
                         "svd single output", api_config.config)
        return

    U_local, S_local, Vh_local = local_output[0], local_output[1], local_output[2]
    U_remote, S_remote, Vh_remote = remote_output[0], remote_output[1], remote_output[2]

    dtype_str = str(S_local.dtype).split(".")[-1]
    atol, rtol = tester._resolve_atol_rtol(dtype_str)

    # 1. 奇异值：直接比较（非负，无歧义）
    _assert_allclose(S_local, S_remote, atol, rtol,
                     "svd singular values S", api_config.config)

    # 2. U：|U| 比较
    _assert_allclose_abs(U_local, U_remote, atol, rtol,
                         "svd U", api_config.config)

    # 3. Vh：|Vh| 比较
    _assert_allclose_abs(Vh_local, Vh_remote, atol, rtol,
                         "svd Vh", api_config.config)


@register_backward("paddle.linalg.svd")
def compare_svd_backward(local_grads, remote_grads, api_config, tester):
    """
    svd 反向特殊对比。

    SVD 的反向梯度 ∂L/∂x 依赖前向的 U 和 Vh 的具体符号约定。
    当 XPU 与 GPU 对某列/行选取了相反的符号时，该列对应的梯度分量方向也相反，
    导致 max_abs_diff ≈ 2 * |梯度值|，属于合法的符号歧义传播，而非真实 Bug。

    处理策略：跳过反向梯度比较（与 sort/topk 的 tie-breaking backward 策略一致）。
    """
    raise SkipComparison(
        "svd backward gradient depends on sign convention of U/Vh; "
        "XPU and GPU may choose opposite signs for tied singular vectors — skip."
    )
