"""Explicit per-API slimming models used by the conservative slimmer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApiModel:
    name: str
    strategy: str
    retention_rate: float = 1.0
    minimum_cases: int = 1
    feature_profile: str = "generic"


PRESERVE = ApiModel("preserve", "preserve")
NEAR = ApiModel("numeric_near", "near")
SIMPLE = ApiModel("simple_coverage", "coverage", 0.10, 4)
CAST = ApiModel("cast_coverage", "coverage", 0.20, 8)
ELEMENTWISE = ApiModel("elementwise_coverage", "coverage", 0.40, 12)
BROADCAST = ApiModel("broadcast_coverage", "coverage", 0.70, 24)
LINEAR_ALGEBRA = ApiModel("linear_algebra_coverage", "coverage", 0.75, 32)
MOE_PERMUTE = ApiModel("moe_permute", "coverage", 0.60, 32, "moe_permute")
MOE_UNPERMUTE = ApiModel("moe_unpermute", "coverage", 0.60, 32, "moe_unpermute")
FP8_BLOCKWISE = ApiModel("fp8_blockwise", "coverage", 0.60, 32, "fp8_blockwise")
FUSED_DEQUANT = ApiModel("fused_act_dequant", "coverage", 0.60, 32, "fused_act_dequant")
CUSTOM = ApiModel("custom_coverage", "coverage", 0.70, 64, "custom")
UNMODELED = ApiModel("unmodeled_preserve", "preserve")


_MODELED_API_NAMES = frozenset(
    """
paddle.Tensor.__add__
paddle.Tensor.__eq__
paddle.Tensor.__ge__
paddle.Tensor.__getitem__
paddle.Tensor.__gt__
paddle.Tensor.__len__
paddle.Tensor.__mul__
paddle.Tensor.__ne__
paddle.Tensor.__neg__
paddle.Tensor.__nonzero__
paddle.Tensor.__radd__
paddle.Tensor.__rmul__
paddle.Tensor.__rpow__
paddle.Tensor.__rsub__
paddle.Tensor.__rtruediv__
paddle.Tensor.__setitem__
paddle.Tensor.__sub__
paddle.Tensor.__truediv__
paddle.Tensor.all
paddle.Tensor.astype
paddle.Tensor.cast
paddle.Tensor.clone
paddle.Tensor.cos
paddle.Tensor.detach
paddle.Tensor.dim
paddle.Tensor.expand
paddle.Tensor.flatten
paddle.Tensor.item
paddle.Tensor.max
paddle.Tensor.mean
paddle.Tensor.reshape
paddle.Tensor.sigmoid
paddle.Tensor.sin
paddle.Tensor.square
paddle.Tensor.squeeze
paddle.Tensor.sum
paddle.Tensor.tolist
paddle.Tensor.transpose
paddle.Tensor.unsqueeze
paddle.Tensor.view
paddle.Tensor.zero_
paddle._C_ops._run_custom_op
paddle._C_ops.adamw_
paddle._C_ops.add_
paddle._C_ops.bitwise_not
paddle._C_ops.flatten_
paddle._C_ops.full_
paddle._C_ops.fused_linear_param_grad_add
paddle._C_ops.gaussian
paddle._C_ops.matmul_grad
paddle._C_ops.multiply_
paddle._C_ops.numel
paddle._C_ops.put_along_axis_
paddle._C_ops.scale_
paddle._C_ops.subtract_
paddle._C_ops.swiglu_grad
paddle._C_ops.transpose
paddle._C_ops.uniform
paddle.add_n
paddle.addmm
paddle.arange
paddle.assign
paddle.baddbmm
paddle.broadcast_to
paddle.cast
paddle.cat
paddle.chunk
paddle.clamp
paddle.clip
paddle.compat.nn.functional.linear
paddle.concat
paddle.cos
paddle.divide
paddle.einsum
paddle.empty
paddle.empty_like
paddle.full
paddle.gather
paddle.incubate.nn.functional.fp8_quant_blockwise
paddle.incubate.nn.functional.fused_act_dequant
paddle.isfinite
paddle.lerp
paddle.matmul
paddle.max
paddle.maximum
paddle.mean
paddle.median
paddle.min
paddle.nn.clip._squared_l2_norm
paddle.nn.functional.dropout
paddle.nn.functional.embedding
paddle.nn.functional.linear
paddle.nn.functional.moe_permute
paddle.nn.functional.moe_unpermute
paddle.nn.functional.normalize
paddle.nn.functional.pad
paddle.nn.functional.rms_norm
paddle.nn.functional.sigmoid
paddle.nn.functional.softmax
paddle.nn.functional.swiglu
paddle.nn.init.trunc_normal_
paddle.nonzero
paddle.outer
paddle.repeat_interleave
paddle.reshape
paddle.rsqrt
paddle.sign
paddle.sin
paddle.split
paddle.sqrt
paddle.stack
paddle.sum
paddle.transpose
paddle.unbind
paddle.var
paddle.where
paddle.zeros
paddle.zeros_like
""".split()
)


_SIMPLE_APIS = frozenset(
    {
        "paddle.Tensor.clone",
        "paddle.Tensor.detach",
        "paddle.Tensor.dim",
        "paddle.Tensor.item",
        "paddle.Tensor.tolist",
        "paddle.Tensor.zero_",
        "paddle._C_ops.full_",
        "paddle._C_ops.numel",
        "paddle.arange",
        "paddle.assign",
        "paddle.empty",
        "paddle.empty_like",
        "paddle.full",
        "paddle.zeros",
        "paddle.zeros_like",
    }
)

_CAST_APIS = frozenset(
    {
        "paddle.Tensor.astype",
        "paddle.Tensor.cast",
        "paddle.cast",
    }
)

_ELEMENTWISE_APIS = frozenset(
    {
        "paddle.Tensor.__neg__",
        "paddle.Tensor.cos",
        "paddle.Tensor.sigmoid",
        "paddle.Tensor.sin",
        "paddle.Tensor.square",
        "paddle._C_ops.bitwise_not",
        "paddle.clamp",
        "paddle.clip",
        "paddle.cos",
        "paddle.isfinite",
        "paddle.nn.functional.sigmoid",
        "paddle.rsqrt",
        "paddle.sign",
        "paddle.sin",
        "paddle.sqrt",
    }
)

_BROADCAST_APIS = frozenset(
    {
        "paddle.Tensor.__add__",
        "paddle.Tensor.__mul__",
        "paddle.Tensor.__sub__",
        "paddle.Tensor.__truediv__",
        "paddle._C_ops.add_",
        "paddle._C_ops.multiply_",
        "paddle._C_ops.scale_",
        "paddle._C_ops.subtract_",
        "paddle.divide",
        "paddle.lerp",
        "paddle.maximum",
    }
)

_LINEAR_ALGEBRA_APIS = frozenset(
    {
        "paddle._C_ops.fused_linear_param_grad_add",
        "paddle._C_ops.matmul_grad",
        "paddle.addmm",
        "paddle.baddbmm",
        "paddle.compat.nn.functional.linear",
        "paddle.einsum",
        "paddle.matmul",
        "paddle.nn.functional.linear",
        "paddle.outer",
    }
)

_PRESERVE_APIS = frozenset(
    {
        "paddle.Tensor.transpose",
        "paddle.Tensor.__getitem__",
        "paddle.Tensor.__nonzero__",
        "paddle.Tensor.__setitem__",
        "paddle.Tensor.all",
        "paddle.Tensor.expand",
        "paddle.Tensor.max",
        "paddle.Tensor.mean",
        "paddle.Tensor.sum",
        "paddle._C_ops.adamw_",
        "paddle._C_ops.gaussian",
        "paddle._C_ops.put_along_axis_",
        "paddle._C_ops.swiglu_grad",
        "paddle._C_ops.transpose",
        "paddle._C_ops.uniform",
        "paddle.add_n",
        "paddle.broadcast_to",
        "paddle.cat",
        "paddle.chunk",
        "paddle.concat",
        "paddle.gather",
        "paddle.max",
        "paddle.mean",
        "paddle.median",
        "paddle.min",
        "paddle.nn.clip._squared_l2_norm",
        "paddle.nn.functional.dropout",
        "paddle.nn.functional.embedding",
        "paddle.nn.functional.normalize",
        "paddle.nn.functional.pad",
        "paddle.nn.functional.rms_norm",
        "paddle.nn.functional.softmax",
        "paddle.nn.functional.swiglu",
        "paddle.nn.init.trunc_normal_",
        "paddle.nonzero",
        "paddle.repeat_interleave",
        "paddle.split",
        "paddle.stack",
        "paddle.sum",
        "paddle.transpose",
        "paddle.unbind",
        "paddle.var",
        "paddle.where",
    }
)


_NUMERIC_NEAR_APIS = frozenset(
    {
        "paddle.Tensor.__eq__",
        "paddle.Tensor.__ge__",
        "paddle.Tensor.__gt__",
        "paddle.Tensor.__len__",
        "paddle.Tensor.__ne__",
        "paddle.Tensor.__radd__",
        "paddle.Tensor.__rmul__",
        "paddle.Tensor.__rpow__",
        "paddle.Tensor.__rsub__",
        "paddle.Tensor.__rtruediv__",
        "paddle.Tensor.flatten",
        "paddle.Tensor.reshape",
        "paddle.Tensor.squeeze",
        "paddle.Tensor.unsqueeze",
        "paddle.Tensor.view",
        "paddle._C_ops.flatten_",
        "paddle.reshape",
    }
)


API_MODEL_REGISTRY: dict[str, ApiModel] = {}
API_MODEL_REGISTRY.update(dict.fromkeys(_NUMERIC_NEAR_APIS, NEAR))
API_MODEL_REGISTRY.update(dict.fromkeys(_PRESERVE_APIS, PRESERVE))
API_MODEL_REGISTRY.update(dict.fromkeys(_SIMPLE_APIS, SIMPLE))
API_MODEL_REGISTRY.update(dict.fromkeys(_CAST_APIS, CAST))
API_MODEL_REGISTRY.update(dict.fromkeys(_ELEMENTWISE_APIS, ELEMENTWISE))
API_MODEL_REGISTRY.update(dict.fromkeys(_BROADCAST_APIS, BROADCAST))
API_MODEL_REGISTRY.update(dict.fromkeys(_LINEAR_ALGEBRA_APIS, LINEAR_ALGEBRA))
API_MODEL_REGISTRY.update(
    {
        "paddle.nn.functional.moe_permute": MOE_PERMUTE,
        "paddle.nn.functional.moe_unpermute": MOE_UNPERMUTE,
        "paddle.incubate.nn.functional.fp8_quant_blockwise": FP8_BLOCKWISE,
        "paddle.incubate.nn.functional.fused_act_dequant": FUSED_DEQUANT,
        "paddle._C_ops._run_custom_op": CUSTOM,
    }
)

CUSTOM_OP_MODEL_REGISTRY: dict[str, ApiModel] = {
    "fuse_weighted_swiglu_fp8_quant": CUSTOM,
    "paddlefleet_fused_swiglu_probs_bwd": CUSTOM,
    "fuse_stack_fp8_quant": PRESERVE,
    "fuse_stack_transpose_fp8_quant": PRESERVE,
}


def resolve_api_model(api_name: str | None, custom_name: str | None) -> ApiModel:
    if api_name is None:
        return UNMODELED
    if api_name == "paddle._C_ops._run_custom_op":
        return CUSTOM_OP_MODEL_REGISTRY.get(custom_name or "", UNMODELED)
    return API_MODEL_REGISTRY.get(api_name, UNMODELED)


def modeled_api_names() -> frozenset[str]:
    return _MODELED_API_NAMES


assert _SIMPLE_APIS <= _MODELED_API_NAMES
assert _CAST_APIS <= _MODELED_API_NAMES
assert _ELEMENTWISE_APIS <= _MODELED_API_NAMES
assert _BROADCAST_APIS <= _MODELED_API_NAMES
assert _LINEAR_ALGEBRA_APIS <= _MODELED_API_NAMES
assert _PRESERVE_APIS <= _MODELED_API_NAMES
assert _NUMERIC_NEAR_APIS <= _MODELED_API_NAMES
assert set(API_MODEL_REGISTRY) == set(_MODELED_API_NAMES)
assert (
    sum(
        len(group)
        for group in (
            _SIMPLE_APIS,
            _CAST_APIS,
            _ELEMENTWISE_APIS,
            _BROADCAST_APIS,
            _LINEAR_ALGEBRA_APIS,
            _PRESERVE_APIS,
            _NUMERIC_NEAR_APIS,
        )
    )
    == len(_MODELED_API_NAMES) - 5
)
