from __future__ import annotations

import paddle
import torch

__all__ = ["process_grad_output", "process_output"]


def paddle_tensor_to_torch(value):
    return torch.utils.dlpack.from_dlpack(
        paddle.utils.dlpack.to_dlpack(value.detach())  # type: ignore[reportGeneralTypeIssues]
    )


def process_output(api_config, paddle_output, torch_output):
    # accuracy 与 accuracy_stable 共用输出裁剪规则，避免两个模式的精度语义分叉。
    if api_config.api_name == "paddle.unique":
        if "return_index=True" in api_config.config:
            paddle_output = list(paddle_output)
            paddle_output.pop(1)
    elif api_config.api_name in {
        "paddle.mode",
        "paddle.Tensor.mode",
        "paddle.incubate.nn.functional.fused_layer_norm",
        "paddle.incubate.nn.functional.fused_rms_norm",
        "paddle.kthvalue",
        "paddle.Tensor.kthvalue",
    }:
        paddle_output = paddle_output[:1]
        torch_output = torch_output[:1]
    elif api_config.api_name in {
        "paddle.strided_slice",
        "paddle.vander",
    }:
        if any(s < 0 for s in paddle_output.strides):
            # torch's from_dlpack now don't support negative strides
            paddle_output = paddle_output.contiguous()
    elif api_config.api_name == "paddle.linalg.eigh":
        # The output of eigen vectors are not unique, because multiplying an eigen vector by -1 in the real case
        # or by e^(i*\theta) in the complex case produces another set of valid eigen vectors of the matrix.
        # So we test whether the elements of each coef_vector (i.e. paddle_output / torch_output for each eigen vector)
        # are all the same and whether the |coef| == 1 for simplicity.
        paddle_output, torch_output = list(paddle_output), list(torch_output)
        eigvector_len = paddle_output[1].shape[-2]
        paddle_eigvectors = paddle_output.pop(1).matrix_transpose().reshape([-1, eigvector_len])
        torch_eigvectors = torch_output.pop(1).transpose(-1, -2).reshape((-1, eigvector_len))
        paddle_output, torch_output = [], []
        for i in range(paddle_eigvectors.shape[0]):
            paddle_vector = paddle_tensor_to_torch(paddle_eigvectors[i]).to(
                device=torch_eigvectors.device
            )
            coef_vector = paddle_vector / torch_eigvectors[i]
            if coef_vector.is_complex():
                coef_vector = torch.complex(
                    coef_vector.real.round(decimals=2),
                    coef_vector.imag.round(decimals=2),
                )
            else:
                coef_vector = coef_vector.round(decimals=2)
            coef_vector_approx = torch.ones_like(coef_vector) * coef_vector[0]
            abs_coef = coef_vector.abs().to(dtype=torch.float64)[0]
            one = torch.ones_like(abs_coef)
            paddle_output.append([coef_vector, abs_coef])
            torch_output.append([coef_vector_approx, one])
    elif api_config.api_name == "paddle._C_ops.fused_linear_param_grad_add":
        # When has_bias=False, Paddle returns an uninitialized tensor for dbias (2nd output).
        # Only compare the first output (dweight).
        if isinstance(paddle_output, (list, tuple)) and len(paddle_output) > 1:
            paddle_output = paddle_output[:1]
        if isinstance(torch_output, (list, tuple)) and len(torch_output) > 1:
            torch_output = torch_output[:1]
    elif api_config.api_name == "paddle._C_ops.swiglu_grad":
        # When y is None, Paddle returns an uninitialized placeholder tensor for dy.
        # Only compare dx to avoid converting the uninitialized tensor to DLPack.
        if len(api_config.args) > 1 and api_config.args[1] is None:
            if isinstance(paddle_output, (list, tuple)) and len(paddle_output) > 1:
                paddle_output = paddle_output[:1]
            if isinstance(torch_output, (list, tuple)) and len(torch_output) > 1:
                torch_output = torch_output[:1]
    elif (
        api_config.api_name == "paddle._C_ops._run_custom_op"
        and api_config.args
        and api_config.args[0]
        in {
            "fuse_weighted_swiglu_fp8_quant",
            "fuse_weighted_swiglu_fp8_quant_clamp",
        }
    ):
        # fp8 量化算子输出是 [fp8码, scale]，真实值 = 码×scale。scale 是 pow2，边界处
        # kernel 与参考可能选相邻档位，raw 码会差约 2×（fp8 仅 3 位尾数，放大成夸张差异）。
        # 这里改比"反量化值 码×scale"：档位差被抵消，只留下真实的量化误差（≤1 fp8 ULP）。
        # use_ue8m0 时 scale 是打包 int32 指数位，需先解包成 2^(e-127) 再反量化。
        use_ue8m0 = len(api_config.args) > 4 and bool(api_config.args[4])
        if (
            isinstance(paddle_output, (list, tuple))
            and len(paddle_output) >= 2
            and isinstance(torch_output, (list, tuple))
            and len(torch_output) >= 2
        ):
            paddle_output = [
                _dequant_fp8_block(paddle_output[0], paddle_output[1], "paddle", use_ue8m0)
            ]
            torch_output = [
                _dequant_fp8_block(torch_output[0], torch_output[1], "torch", use_ue8m0)
            ]
    return paddle_output, torch_output


def _unpack_ue8m0_scale(packed, nblocks):
    # 打包规则（见 rules.py）：每个 int32 的低到高 4 字节 = 连续 4 个 block 的 inv_scale 指数位。
    # 这里反向拆回 [rows, nblocks] 的指数，再还原成 2^(e-127)（inv_scale 为 pow2，尾数恒 0）。
    p = packed.to(torch.int64) & 0xFFFFFFFF
    bytes4 = torch.stack([(p >> (8 * k)) & 0xFF for k in range(4)], dim=-1)
    exp = bytes4.reshape(*packed.shape[:-1], -1)[..., :nblocks].to(torch.float32)
    return torch.exp2(exp - 127.0)


def _dequant_fp8_block(code, scale, framework, use_ue8m0=False, tile=128):
    # 把每 128 列一个的 block scale 沿列展开到与 fp8 码同宽，再逐元素相乘还原真实值。
    if use_ue8m0:
        # ue8m0 需按位解包，统一转到 torch 处理（paddle 位运算在此路径更易错）。
        if framework == "paddle":
            code = paddle_tensor_to_torch(code)
            scale = paddle_tensor_to_torch(scale)
        width = code.shape[-1]
        nblocks = (width + tile - 1) // tile
        scale_exp = _unpack_ue8m0_scale(scale, nblocks).repeat_interleave(tile, dim=-1)
        return code.float() * scale_exp[..., :width]
    width = code.shape[-1]
    if framework == "paddle":
        scale_exp = paddle.repeat_interleave(scale.astype("float32"), tile, axis=-1)
        return code.astype("float32") * scale_exp[..., :width]
    scale_exp = scale.float().repeat_interleave(tile, dim=-1)
    return code.float() * scale_exp[..., :width]


def process_grad_output(api_config, paddle_out_grads, torch_out_grads):
    # 仅保留两端都有可比语义的梯度输出；不参与 torch 对齐的配置应同步到错误清单。
    if api_config.api_name in {
        "paddle.nn.functional.scaled_dot_product_attention",
    }:
        paddle_out_grads = paddle_out_grads[:3]
        torch_out_grads = torch_out_grads[:3]
    elif api_config.api_name in {
        "paddle.lerp",
        "paddle.tensordot",
    }:
        paddle_out_grads = paddle_out_grads[:2]
        torch_out_grads = torch_out_grads[:2]
    elif api_config.api_name in {
        "paddle.Tensor.__setitem__",
        "paddle.Tensor.fill_diagonal_tensor",
        "paddle.diagonal_scatter",
        "paddle.incubate.softmax_mask_fuse",
        "paddle.nn.functional.binary_cross_entropy",
        "paddle.nn.functional.binary_cross_entropy_with_logits",
        "paddle.nn.functional.cross_entropy",
        "paddle.nn.functional.gaussian_nll_loss",
        "paddle.nn.functional.kl_div",
        "paddle.nn.functional.sigmoid_focal_loss",
        "paddle.scale",
    }:
        paddle_out_grads = paddle_out_grads[:1]
        torch_out_grads = torch_out_grads[:1]
    elif api_config.api_name in {
        "paddle.combinations",
        "paddle.nn.utils.parameters_to_vector",
        "paddle.cdist",
    }:
        paddle_out_grads = []
        torch_out_grads = []
    elif api_config.api_name == "paddle.linalg.cholesky_solve":
        if len(api_config.args) > 2:
            is_upper = api_config.args[2]
        elif "is_upper" in api_config.kwargs:
            is_upper = api_config.kwargs["is_upper"]
        else:
            is_upper = False
        torch_out_grads = list(torch_out_grads)
        torch_out_grads[1] = (
            torch.triu(torch_out_grads[1]) if is_upper else torch.tril(torch_out_grads[1])
        )
    elif api_config.api_name == "paddle.incubate.nn.functional.fused_rotary_position_embedding":
        # Paddle only has 3 outputs/grads Q, K, V
        valid_out_num = len([out for out in paddle_out_grads if out is not None])
        paddle_out_grads = paddle_out_grads[:valid_out_num]
        torch_out_grads = torch_out_grads[:valid_out_num]
    return paddle_out_grads, torch_out_grads
