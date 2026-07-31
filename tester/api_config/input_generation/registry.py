"""输入生成规则的装饰器注册中心。"""

from __future__ import annotations

import math
import numbers
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy

from .input_bind import InputGenerationContext
from .input_values import InputValue, attach_values
from .input_values import input_value as read_input_value
from .tensor_config import not_zero_apis
from .tensor_view import TensorView
from .value_sample import (
    create_case_rng,
    generate_abs_plus_one,
    generate_binary01,
    generate_default,
    generate_dropout_prob,
    generate_empty_shape,
    generate_fill_value,
    generate_hinge_label,
    generate_int_64,
    generate_int_128,
    generate_int_1024,
    generate_int_2048,
    generate_int_2048_raw,
    generate_int_65535_raw,
    generate_int_or_default,
    generate_int_or_unit,
    generate_multiply,
    generate_nonzero,
    generate_normal_std,
    generate_ones_shape,
    generate_quantile,
    generate_random_range,
    generate_remainder,
    generate_signed_half,
    generate_uniform,
    generate_unit_interval,
    generate_unit_plus_one,
)

ValueGenerator = Callable[..., numpy.ndarray]
CaseValueGenerator = Callable[[object], numpy.ndarray]
RuleFunction = Callable[[InputGenerationContext, "RuleCase"], None]
BlockerFunction = Callable[[InputGenerationContext], str | None]
_RAW_WRITE = object()


# 这些描述符字符串要保持稳定，因为规则体直接依赖它们。
_VALUE_GENERATORS: dict[str, ValueGenerator] = {
    "default": lambda spec, low, high, rng: generate_default(spec, rng),
    "nonzero": lambda spec, low, high, rng: generate_nonzero(spec, rng),
    "unit_interval": lambda spec, low, high, rng: generate_unit_interval(spec, rng),
    "multiply": lambda spec, low, high, rng: generate_multiply(spec, rng),
    "unit_interval_plus_one": lambda spec, low, high, rng: generate_unit_plus_one(spec, rng),
    "signed_half_interval": lambda spec, low, high, rng: generate_signed_half(spec, rng),
    "normal_std": lambda spec, low, high, rng: generate_normal_std(spec, rng),
    "dropout_probability": lambda spec, low, high, rng: generate_dropout_prob(spec, rng),
    "full_fill_value": lambda spec, low, high, rng: generate_fill_value(spec, rng),
    "quantile_q": lambda spec, low, high, rng: generate_quantile(spec, rng),
    "remainder_rhs": lambda spec, low, high, rng: generate_remainder(spec, rng),
    "int_zero_1024": lambda spec, low, high, rng: generate_int_1024(spec, rng),
    "int_zero_64": lambda spec, low, high, rng: generate_int_64(spec, rng),
    "int_zero_2048_no_cast": lambda spec, low, high, rng: generate_int_2048_raw(spec, rng),
    "empty_shape": lambda spec, low, high, rng: generate_empty_shape(spec, rng),
    "int_one_128": lambda spec, low, high, rng: generate_int_128(spec, rng),
    "int_one_2048": lambda spec, low, high, rng: generate_int_2048(spec, rng),
    "int_one_65535_no_cast": lambda spec, low, high, rng: generate_int_65535_raw(spec, rng),
    "ones_shape": lambda spec, low, high, rng: generate_ones_shape(spec, rng),
    "int_zero_65535_else_unit": lambda spec, low, high, rng: generate_int_or_unit(spec, rng),
    "int_minus127_127_else_default": lambda spec, low, high, rng: generate_int_or_default(
        spec, rng
    ),
    "binary_0_1": lambda spec, low, high, rng: generate_binary01(spec, rng),
    "hinge_labels": lambda spec, low, high, rng: generate_hinge_label(spec, rng),
    "abs_unit_plus_one": lambda spec, low, high, rng: generate_abs_plus_one(spec, rng),
    "uniform": lambda spec, low, high, rng: generate_uniform(spec, low, high, rng),
    "random_range": lambda spec, low, high, rng: generate_random_range(spec, low, high, rng),
}


@dataclass(frozen=True)
class RegisteredRule:
    """一条通过装饰器注册的 case 级规则。

    这里保存的是规则的元信息和执行入口，不保存 API 约束逻辑本身。
    规则函数负责描述参数关系，`RegisteredRule` 负责门控、完整性检查和提交。
    """

    api_names: tuple[str, ...]
    function: RuleFunction
    block_key: str
    blocker: BlockerFunction | None = None
    allow_gpu: bool = True
    allow_cached: bool = True

    def block_reason(self, context: InputGenerationContext) -> str | None:
        from .tensor_config import USE_CACHED_NUMPY

        if context.gpu_enabled and not self.allow_gpu:
            return f"{self.block_key}-gpu-blocked"

        if USE_CACHED_NUMPY and not self.allow_cached:
            return f"{self.block_key}-cache-blocked"
        if self.blocker is not None:
            return self.blocker(context)
        return None

    def generate(self, context: InputGenerationContext, case: object) -> bool:
        rule_case = RuleCase(context, case)
        self.function(context, rule_case)
        rule_case.require_complete()
        attach_values(case, rule_case.payload_items())
        rule_case.rng.commit()
        return True


class RuleCase:
    """规则编写时可操作的 case 视图。

    `RuleCase` 负责把一次 API 调用中的所有 TensorConfig、随机状态和逻辑值操作串起来。
    它让规则作者只关心“生成什么值”，而不需要关心 payload 存储、去重和提交时机。
    """

    def __init__(self, context: InputGenerationContext, raw_case: object):
        self.context = context
        self.raw_case = raw_case
        self.rng = create_case_rng(context)
        # 用路径去重，避免同一个 Tensor 被重复写入。
        self._generated_paths = set()
        # payload 以 InputPath 为键，物化层只消费逻辑值。
        self._payload_by_path: dict[object, InputValue] = {}

    def generate_all(self, generator: str, low=None, high=None):
        for binding in self.context.call.tensors:
            self._generate_binding(binding, generator, low, high)

    def generate(
        self,
        parameter_name: str | tuple[str, ...],
        generator: str | CaseValueGenerator,
        low=None,
        high=None,
        required: bool = True,
    ):
        names = (parameter_name,) if isinstance(parameter_name, str) else tuple(parameter_name)
        matched = False
        for binding in self.context.call.tensors:
            if binding.parameter_name in names:
                self._generate_binding(binding, generator, low, high)
                matched = True
        if required and not matched:
            raise ValueError(
                f"rule {self.context.call.api_name} did not find parameter {parameter_name!r}"
            )

    def generate_remaining(self, generator: str | CaseValueGenerator, low=None, high=None):
        for binding in self.context.call.tensors:
            if binding.path not in self._generated_paths:
                self._generate_binding(binding, generator, low, high)

    def generate_by_parameter(
        self, parameter_generators, default: str | CaseValueGenerator | None = None
    ):
        normalized = []
        for parameter_name, generator in parameter_generators:
            names = (parameter_name,) if isinstance(parameter_name, str) else tuple(parameter_name)
            normalized.append((names, generator))
        for binding in self.context.call.tensors:
            generator = None
            for names, candidate in normalized:
                if binding.parameter_name in names:
                    generator = candidate
                    break
            if generator is None:
                generator = default
            if generator is not None:
                self._generate_binding(binding, generator)

    def require_complete(self):
        missing = [
            str(binding.path)
            for binding in self.context.call.tensors
            if binding.path not in self._generated_paths
        ]
        if missing:
            raise ValueError(
                f"rule {self.context.call.api_name} left tensors ungenerated: {missing}"
            )

    def is_generated(self, binding):
        return binding.path in self._generated_paths

    def _generate_binding(self, binding, generator: str | CaseValueGenerator, low=None, high=None):
        if binding.path in self._generated_paths:
            raise ValueError(f"rule generated tensor twice: {binding.path}")
        if callable(generator):
            value = generator(binding)
            if value is _RAW_WRITE:
                return
        else:
            generate_value = _VALUE_GENERATORS[generator]
            value = generate_value(binding.spec, low, high, self.rng)
        self.set_value(binding, value)

    def set_value(self, binding, value):
        if binding.path in self._generated_paths:
            raise ValueError(f"rule generated tensor twice: {binding.path}")
        self._payload_by_path[binding.path] = InputValue(binding.path, value)
        _apply_value(self.raw_case, binding.path, value)
        self._generated_paths.add(binding.path)

    def rewrite_value(self, binding, value):
        self._payload_by_path[binding.path] = InputValue(binding.path, value)
        _apply_value(self.raw_case, binding.path, value)
        self._generated_paths.add(binding.path)

    def set_value_raw(self, binding, value):
        if binding.path in self._generated_paths:
            raise ValueError(f"rule generated tensor twice: {binding.path}")
        self._payload_by_path[binding.path] = InputValue(binding.path, value, update_metadata=False)
        _apply_value_raw(self.raw_case, binding.path, value, update_metadata=False)
        self._generated_paths.add(binding.path)
        return _RAW_WRITE

    def find(self, parameter_name: str):
        for binding in self.context.call.tensors:
            if binding.parameter_name == parameter_name:
                return binding
        return None

    def binding_for_config(self, tensor_config):
        for binding in self.context.call.tensors:
            if _tensor_config_at(self.raw_case, binding.path) is tensor_config:
                return binding
        return None

    def value(self, binding):
        payload = self.payload(binding)
        if payload is not None:
            return payload.value
        config = _tensor_config_at(self.raw_case, binding.path)
        return read_input_value(self.raw_case, config)

    def payload(self, binding):
        return self._payload_by_path.get(binding.path)

    def payload_items(self):
        return tuple(self._payload_by_path.values())

    @property
    def api_name(self):
        return self.context.call.api_name

    def arg(self, position, name, default=None):
        return _api_arg(self.raw_case, position, name, default)

    def has_kwarg(self, name):
        return name in self.raw_case.kwargs

    def kwarg(self, name, default=None):
        return self.raw_case.kwargs.get(name, default)

    def argument_values(self):
        return (*self.raw_case.args, *self.raw_case.kwargs.values())

    def binding_list_index(self, binding, default=None):
        config = _tensor_config_at(self.raw_case, binding.path)
        return getattr(config, "list_index", default)

    def is_tensor_config(self, value):
        return _is_tensor_config(value)

    def value_domain(self, generator: str, binding, low=None, high=None):
        return _VALUE_GENERATORS[generator](binding.spec, low, high, self.rng)

    def random(self, size=None, dtype=None):
        value = self.rng.random(size)
        return value.astype(dtype) if dtype is not None else value

    def uniform(self, low=0.0, high=1.0, size=None, dtype=None):
        value = self.rng.uniform(low=low, high=high, size=size)
        return value.astype(dtype) if dtype is not None else value

    def randn(self, *shape):
        return self.rng.randn(*shape)

    def randint(self, low, high=None, size=None, dtype=None):
        kwargs = {"size": size}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self.rng.randint(low, high, **kwargs)

    def array(self, value, dtype=None, copy=True, order="K"):
        return numpy.array(value, dtype=dtype, copy=copy, order=order)

    def arange(self, *args, **kwargs):
        return numpy.arange(*args, **kwargs)

    def zeros(self, shape, dtype=None):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return numpy.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        return numpy.full(shape, fill_value, dtype=dtype)

    def choice(self, values, size=None, replace=True, p=None):
        return self.rng.choice(values, size=size, replace=replace, p=p)

    def python_choice(self, values):
        return random.choice(values)

    def where(self, condition, x, y):
        return numpy.where(condition, x, y)

    def minimum(self, x, y):
        return numpy.minimum(x, y)

    def maximum(self, x, y):
        return numpy.maximum(x, y)

    def abs(self, value):
        return numpy.abs(value)

    def sort(self, value):
        return numpy.sort(value)

    def cumsum(self, value):
        return numpy.cumsum(value)

    def sum(self, value):
        return numpy.sum(value)

    def count_nonzero(self, value):
        return numpy.count_nonzero(value)

    def nonzero(self, value):
        return numpy.nonzero(value)

    def einsum(self, expression, *operands):
        return numpy.einsum(expression, *operands)

    def dot(self, left, right):
        return numpy.dot(left, right)

    def matmul(self, left, right):
        return numpy.matmul(left, right)

    def swapaxes(self, value, axis1, axis2):
        return numpy.swapaxes(value, axis1, axis2)

    def triu(self, value, k=0):
        return numpy.triu(value, k=k)

    def tril(self, value, k=0):
        return numpy.tril(value, k=k)

    def conj(self, value):
        return numpy.conj(value)

    def eye(self, size, dtype=None):
        return numpy.eye(size, dtype=dtype)

    def ascontiguousarray(self, value):
        return numpy.ascontiguousarray(value)

    def dtype_max(self, dtype):
        return numpy.finfo(dtype).max

    def dtype_min(self, dtype):
        return numpy.finfo(dtype).min

    def dtype_eps(self, dtype):
        return numpy.finfo(dtype).eps


class RuleRegistry:
    """失败即止的装饰器注册表。"""

    def __init__(self):
        self._rules: list[RegisteredRule] = []
        self._by_api: dict[str, RegisteredRule] = {}

    def register(
        self,
        *api_names: str,
        aliases: tuple[str, ...] = (),
        block_key: str | None = None,
        blocker: BlockerFunction | None = None,
        allow_gpu: bool = True,
        allow_cached: bool = True,
    ):
        names = _normalize_names((*api_names, *aliases), "api_names")
        if not names:
            raise ValueError("registered rule must declare at least one API")

        def decorator(function: RuleFunction):
            for api_name in names:
                if api_name in self._by_api:
                    raise ValueError(f"input-generation API overlap: {api_name}")
            rule = RegisteredRule(
                api_names=names,
                function=function,
                block_key=block_key or names[0],
                blocker=blocker,
                allow_gpu=allow_gpu,
                allow_cached=allow_cached,
            )
            self._rules.append(rule)
            for api_name in names:
                self._by_api[api_name] = rule
            return function

        return decorator

    @property
    def rules(self) -> tuple[RegisteredRule, ...]:
        return tuple(self._rules)

    @property
    def by_api(self) -> dict[str, RegisteredRule]:
        return dict(self._by_api)


def _normalize_names(values, field_name: str) -> tuple[str, ...]:
    names = tuple(values)
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        if name != name.strip():
            raise ValueError(f"{field_name} entry has surrounding whitespace: {name!r}")
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    if duplicates:
        raise ValueError(f"duplicate {field_name}: {duplicates}")
    return names


def _max_unpool_input_size(
    api_name: str,
    x_shape,
    unpool_output_size,
    kernel_size,
    stride,
    padding,
):
    ndim = 1
    if "max_unpool2d" in api_name:
        ndim = 2
    elif "max_unpool3d" in api_name:
        ndim = 3
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * ndim
    if isinstance(stride, int):
        stride = [stride] * ndim
    if isinstance(padding, int):
        padding = [padding] * ndim

    pool_input_size = unpool_output_size
    if pool_input_size is None:
        if ndim == 1:
            w_in = x_shape[-1]
            w_out = (w_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
            pool_input_size = [*x_shape[:-1], w_out]
        elif ndim == 2:
            h_in, w_in = x_shape[-2], x_shape[-1]
            h_out = (h_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
            w_out = (w_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
            pool_input_size = [*x_shape[:-2], h_out, w_out]
        else:
            d_in, h_in, w_in = (
                x_shape[-3],
                x_shape[-2],
                x_shape[-1],
            )
            d_out = (d_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
            h_out = (h_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
            w_out = (w_in - 1) * stride[2] - 2 * padding[2] + kernel_size[2]
            pool_input_size = [*x_shape[:-3], d_out, h_out, w_out]
    elif len(pool_input_size) == ndim:
        pool_input_size = [*x_shape[:-ndim], *pool_input_size[-ndim:]]
    elif len(pool_input_size) != len(x_shape):
        raise ValueError(
            f"invalid output_size for {api_name}, len(output_size) should be {ndim} or "
            f"{len(x_shape)} or output_size == None, got len(output_size)="
            f"{len(pool_input_size)} and output_size={unpool_output_size}"
        )
    return kernel_size, stride, padding, pool_input_size


def _optimizer_beta_pow(case: RuleCase, binding, beta, step):
    import paddle

    use_accuracy_compatible = paddle.get_flags("FLAGS_use_accuracy_compatible_kernel")[
        "FLAGS_use_accuracy_compatible_kernel"
    ]
    if use_accuracy_compatible:
        beta_pow_value = beta**step
    else:
        beta_pow_value = numpy.power(numpy.float32(beta), numpy.float32(step)).item()
    return case.full(binding.spec.shape, beta_pow_value, dtype=binding.spec.dtype)


def _is_tensor_config(value) -> bool:
    return hasattr(value, "numpy_tensor") and hasattr(value, "shape") and hasattr(value, "dtype")


rules = RuleRegistry()


@rules.register("paddle.add", "paddle.logical_not", "paddle.concat")
def default_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("default")


@rules.register("paddle.incubate.nn.functional.fused_act_dequant")
def fused_act_dequant_values(ctx: InputGenerationContext, case: RuleCase):
    def x_scale_value(binding):
        if binding.spec.dtype == "int32":
            exponent = case.randint(120, 128, size=binding.spec.shape, dtype="int32")
            return exponent * case.array(0x01010101, dtype="int32")
        return case.value_domain("default", binding)

    case.generate_by_parameter((("x_scale", x_scale_value),), default="default")


@rules.register("paddle.incubate.nn.functional.variable_length_memory_efficient_attention")
def variable_length_memory_efficient_attention_values(ctx: InputGenerationContext, case: RuleCase):
    def seq_lens_value(binding):
        query = case.arg(0, "query")
        q_seq_len = query.shape[2]
        return case.value_domain("random_range", binding, 1, q_seq_len)

    def kv_seq_lens_value(binding):
        key = case.arg(1, "key")
        value = case.arg(2, "value")
        max_seq_len = min(key.shape[2], value.shape[2])
        return case.value_domain("random_range", binding, 1, max_seq_len)

    def mask_value(binding):
        return case.randint(0, 2, size=binding.spec.shape).astype(
            binding.spec.dtype
        ) * case.dtype_min(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("seq_lens", seq_lens_value),
            ("kv_seq_lens", kv_seq_lens_value),
            ("mask", mask_value),
        ),
        default="default",
    )


@rules.register(
    "paddle.incubate.nn.functional.block_multihead_attention",
)
def block_multihead_attention_values(ctx: InputGenerationContext, case: RuleCase):
    qkv = case.arg(0, "qkv")
    seq_lens_encoder = case.arg(3, "seq_lens_encoder")
    batch_size = seq_lens_encoder.shape[0]
    seq_len = qkv.shape[0] // batch_size

    zero_parameters = {
        "key_cache",
        "value_cache",
        "seq_lens_decoder",
        "block_tables",
        "max_dec_len_this_time",
    }
    positive_range_parameters = {
        "cache_k_quant_scales",
        "cache_v_quant_scales",
        "cache_k_dequant_scales",
        "cache_v_dequant_scales",
        "qkv_out_scale",
        "out_smooth",
    }

    def seq_len_array(binding):
        return case.array([seq_len] * batch_size, dtype=binding.spec.dtype)

    def set_padding_offsets(binding):
        seq_lens_this_time = case.value(case.find("seq_lens_this_time"))
        cum_offsets_now = case.cumsum(seq_len - seq_lens_this_time)
        cum_offsets_binding = case.find("cum_offsets")
        cu_seqlens_q_binding = case.find("cu_seqlens_q")
        cu_seqlens_k_binding = case.find("cu_seqlens_k")
        cum_offsets = case.zeros((batch_size + 1,), dtype=cum_offsets_binding.spec.dtype)
        cum_offsets[1:] = cum_offsets_now
        token_num = case.sum(seq_lens_this_time)
        padding_offsets = case.zeros((token_num,), dtype=binding.spec.dtype)
        cu_seqlens_q = case.zeros((batch_size + 1,), dtype=cu_seqlens_q_binding.spec.dtype)
        cu_seqlens_k = case.zeros((batch_size + 1,), dtype=cu_seqlens_k_binding.spec.dtype)
        for batch_index in range(batch_size):
            seq_len_now = int(seq_lens_this_time[batch_index])
            cum_offset = int(cum_offsets[batch_index])
            for token_index in range(seq_len_now):
                padding_offsets[batch_index * seq_len - cum_offset + token_index] = cum_offset
            cum_seq_len = (batch_index + 1) * seq_len - cum_offsets[batch_index + 1]
            cu_seqlens_q[batch_index + 1] = cum_seq_len
            cu_seqlens_k[batch_index + 1] = cum_seq_len
        case.set_value(cum_offsets_binding, cum_offsets[:-1])
        case.set_value(cu_seqlens_q_binding, cu_seqlens_q)
        case.set_value(cu_seqlens_k_binding, cu_seqlens_k)
        case.set_value(binding, padding_offsets)

    for binding in ctx.call.tensors:
        if case.is_generated(binding):
            continue
        if binding.parameter_name in zero_parameters:
            case.set_value(binding, case.zeros(binding.spec.shape, dtype=binding.spec.dtype))
        elif binding.parameter_name == "seq_lens_encoder":
            case.set_value(binding, seq_len_array(binding))
        elif binding.parameter_name == "seq_lens_this_time":
            case.set_value(binding, case.value(case.find("seq_lens_encoder")))
        elif binding.parameter_name == "padding_offsets":
            set_padding_offsets(binding)
        elif binding.parameter_name in positive_range_parameters:
            case.set_value(binding, case.value_domain("random_range", binding, low=0))
        elif binding.parameter_name == "max_enc_len_this_time":
            case.set_value(binding, seq_len_array(binding))
        elif binding.parameter_name in {"mask", "tgt_mask"}:
            case.set_value(
                binding,
                case.value_domain("random_range", binding, high=case.dtype_eps(binding.spec.dtype)),
            )
        else:
            case.set_value(binding, case.value_domain("default", binding))


@rules.register(
    "paddle._C_ops.adam_",
    "paddle._C_ops.adamw_",
    "paddle._C_ops.merged_adam_",
    block_key="optimizer",
)
def optimizer_values(ctx: InputGenerationContext, case: RuleCase):
    zero_parameters = {"moment1", "moment2", "moment2_max"}
    optimizer_step = None

    def generate_value(binding):
        nonlocal optimizer_step
        if binding.parameter_name in zero_parameters:
            return case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        if case.api_name == "paddle._C_ops.adamw_" and binding.parameter_name in {
            "beta1_pow",
            "beta2_pow",
        }:
            if optimizer_step is None:
                optimizer_step = case.randint(1, 101)
            beta = case.arg(10, "beta1")
            if binding.parameter_name == "beta2_pow":
                beta = case.arg(11, "beta2")
            return _optimizer_beta_pow(case, binding, beta, optimizer_step)
        return case.value_domain("default", binding)

    case.generate_all(generate_value)


@rules.register(
    "paddle.nn.functional.max_unpool1d",
    "paddle.nn.functional.max_unpool2d",
    "paddle.nn.functional.max_unpool3d",
)
def max_unpool_values(ctx: InputGenerationContext, case: RuleCase):
    import paddle

    x_binding = case.find("x")
    indices_binding = case.find("indices")
    if x_binding is None or indices_binding is None:
        raise ValueError(f"rule {case.api_name} requires x and indices tensors")

    kernel_size = case.arg(2, "kernel_size")
    stride = case.arg(3, "stride")
    padding = case.arg(4, "padding")
    output_size = case.arg(5, "output_size")
    kernel_size, stride, padding, pool_input_size = _max_unpool_input_size(
        case.api_name,
        x_binding.spec.shape,
        output_size,
        kernel_size,
        stride,
        padding,
    )
    data_type = "float64" if x_binding.spec.dtype == "int64" else x_binding.spec.dtype
    pool_input_spec = TensorView(
        shape=tuple(pool_input_size),
        dtype=data_type,
        place=x_binding.spec.place,
        is_contiguous=x_binding.spec.is_contiguous,
        strides=x_binding.spec.strides,
    )
    pool_input = generate_random_range(pool_input_spec, low=-5, high=5, rng=case.rng)
    x = paddle.to_tensor(pool_input)
    pool_name = case.api_name.rsplit(".", 1)[-1].replace("max_unpool", "max_pool")
    max_poolxd_func = getattr(paddle.nn.functional, pool_name)
    x, indices = max_poolxd_func(x, kernel_size, stride, padding, return_mask=True)
    case.set_value(x_binding, x.numpy())
    case.set_value(indices_binding, indices.numpy())


@rules.register("paddle.arange")
def arange_values(ctx: InputGenerationContext, case: RuleCase):
    def tensor_binding(value):
        return case.binding_for_config(value) if case.is_tensor_config(value) else None

    def rewrite_tensor(value, tensor_value):
        binding = tensor_binding(value)
        if binding is not None:
            case.rewrite_value(binding, tensor_value)

    def generate_step_tensor(step_config, is_positive):
        if "int" in step_config.dtype:
            if is_positive:
                return case.randint(1, 10, size=step_config.shape).astype(step_config.dtype)
            return case.randint(-10, -1, size=step_config.shape).astype(step_config.dtype)
        if is_positive:
            return case.uniform(0.1, 5.0, size=step_config.shape).astype(step_config.dtype)
        return case.uniform(-5.0, -0.1, size=step_config.shape).astype(step_config.dtype)

    def safe_range(low, high):
        max_range = 100
        if high - low > max_range:
            if low < 0:
                high = low + max_range
            else:
                low = high - max_range
        if low >= high:
            low = high - 10
        return max(low, -1000), min(high, 1000)

    def random_range(tensor_config, low, high):
        if "int" in tensor_config.dtype:
            return case.randint(low, high, size=tensor_config.shape).astype(tensor_config.dtype)
        return case.uniform(low, high, size=tensor_config.shape).astype(tensor_config.dtype)

    def handle_arange_relation():
        start_val = case.arg(0, "start", 0)
        end_val = case.arg(1, "end", None)
        step_val = case.arg(2, "step", 1)

        if case.is_tensor_config(start_val):
            if case.is_tensor_config(end_val):
                if case.is_tensor_config(step_val):
                    flag = case.choice([True, False])
                    rewrite_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                rewrite_tensor(start_val, random_range(start_val, -50, 50))
                start = case.value(case.find("start")).item()
                if flag:
                    low, high = safe_range(start + 1, start + 50)
                else:
                    low, high = safe_range(start - 50, start - 1)
                rewrite_tensor(end_val, random_range(end_val, low, high))
            elif end_val is None:
                if case.is_tensor_config(step_val):
                    flag = case.choice([True, False])
                    rewrite_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                if flag:
                    if "int" in start_val.dtype:
                        value = case.randint(1, 50, size=start_val.shape).astype(start_val.dtype)
                    else:
                        value = case.uniform(0.1, 50.0, size=start_val.shape).astype(
                            start_val.dtype
                        )
                elif "int" in start_val.dtype:
                    value = case.randint(-50, -1, size=start_val.shape).astype(start_val.dtype)
                else:
                    value = case.uniform(-50.0, -0.1, size=start_val.shape).astype(start_val.dtype)
                rewrite_tensor(start_val, value)
            else:
                if case.is_tensor_config(step_val):
                    flag = case.choice([True, False])
                    rewrite_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                if flag:
                    low, high = safe_range(end_val - 50, end_val - 1)
                else:
                    low, high = safe_range(end_val + 1, end_val + 50)
                rewrite_tensor(start_val, random_range(start_val, low, high))
        elif case.is_tensor_config(end_val):
            if case.is_tensor_config(step_val):
                flag = case.choice([True, False])
                rewrite_tensor(step_val, generate_step_tensor(step_val, flag))
            else:
                flag = step_val > 0
            if flag:
                low, high = safe_range(start_val + 1, start_val + 50)
            else:
                low, high = safe_range(start_val - 50, start_val - 1)
            rewrite_tensor(end_val, random_range(end_val, low, high))
        elif end_val is None:
            if case.is_tensor_config(step_val):
                flag = start_val > 0
                rewrite_tensor(step_val, generate_step_tensor(step_val, flag))
        elif case.is_tensor_config(step_val):
            flag = start_val < end_val
            rewrite_tensor(step_val, generate_step_tensor(step_val, flag))

    for binding in ctx.call.tensors:
        if case.value(binding) is None:
            handle_arange_relation()


@rules.register("paddle.nn.functional.moe_permute")
def moe_permute_values(ctx: InputGenerationContext, case: RuleCase):
    def expert_routemap_value(binding):
        num_experts = case.arg(4, "num_experts", 32)
        hidden_states = case.arg(0, "hidden_states")
        scale = case.arg(1, "scale")
        expert_prob = case.arg(3, "expert_prob_topk")
        tokens_per_expert = case.arg(5, "tokens_per_expert")
        padding_alignment = case.arg(6, "padding_alignment")
        using_ue8m0_scale = case.arg(8, "using_ue8m0_scale", False)
        if (
            not isinstance(num_experts, int)
            or isinstance(num_experts, bool)
            or not 1 <= num_experts <= 64
        ):
            raise ValueError("num_experts must be an integer in [1, 64]")
        if (
            not isinstance(padding_alignment, int)
            or isinstance(padding_alignment, bool)
            or padding_alignment <= 0
            or padding_alignment & (padding_alignment - 1)
        ):
            raise ValueError("padding_alignment must be a positive power of 2")
        if not case.is_tensor_config(hidden_states) or (
            len(hidden_states.shape) != 2
            or hidden_states.dtype not in {"bfloat16", "float32", "float8_e4m3fn"}
        ):
            raise ValueError("hidden_states must be a rank-2 bfloat16 or float8_e4m3fn tensor")
        if binding.spec.dtype != "int32":
            raise ValueError("expert_routemap_topk dtype must be int32")
        if not case.is_tensor_config(expert_prob) or (
            len(expert_prob.shape) != 2 or expert_prob.dtype != "float32"
        ):
            raise ValueError("expert_prob_topk must be a rank-2 float32 tensor")
        seqlen, topk = binding.spec.shape[0], binding.spec.shape[1]
        if not (hidden_states.shape[0] == seqlen and tuple(expert_prob.shape) == (seqlen, topk)):
            raise ValueError(
                "hidden_states, expert_routemap_topk, and expert_prob_topk "
                "must share sequence_length and top_k dimensions"
            )
        if hidden_states.dtype == "float8_e4m3fn":
            expected_scale_width = (hidden_states.shape[1] + 127) // 128
            expected_scale_dtype = "float32"
            if using_ue8m0_scale:
                expected_scale_width = (expected_scale_width + 3) // 4
                expected_scale_dtype = "int32"
            if not (
                case.is_tensor_config(scale)
                and tuple(scale.shape) == (seqlen, expected_scale_width)
                and scale.dtype == expected_scale_dtype
            ):
                raise ValueError(
                    "float8 hidden_states requires scale with shape "
                    f"[{seqlen}, {expected_scale_width}] and dtype {expected_scale_dtype}"
                )
        elif scale is not None:
            raise ValueError("scale must be None when hidden_states dtype is bfloat16")
        routemap = case.full((seqlen, topk), -1, dtype="int32")
        if topk == 0:
            raise ValueError("topk should be greater than 0")
        if not isinstance(tokens_per_expert, list):
            raise ValueError("tokens_per_expert must be a list of integers")
        if len(tokens_per_expert) != num_experts:
            raise ValueError("tokens_per_expert length must equal num_experts")
        if any(
            not isinstance(count, int) or isinstance(count, bool) for count in tokens_per_expert
        ):
            raise ValueError("tokens_per_expert must be a list of integers")
        total_assignments = sum(tokens_per_expert)
        representable = total_assignments <= seqlen * topk and not any(
            count < 0 or count > seqlen for count in tokens_per_expert
        )
        if not representable:
            raise ValueError(
                "tokens_per_expert cannot be represented by the expert_routemap_topk shape"
            )
        cursor = 0
        for expert, count in enumerate(tokens_per_expert):
            positions = case.arange(cursor, cursor + count, dtype="int64")
            rows = positions % seqlen
            columns = (positions // seqlen) % topk
            routemap[rows, columns] = expert
            cursor += count
        return routemap

    def expert_prob_value(binding):
        routemap_binding = case.find("expert_routemap_topk")
        probs = case.zeros(binding.spec.shape, dtype="float32")
        if routemap_binding is not None and case.value(routemap_binding) is not None:
            mask = case.value(routemap_binding) >= 0
            raw = case.random(binding.spec.shape).astype("float32") * mask
            row_sums = raw.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = raw / row_sums
        else:
            probs = case.random(binding.spec.shape).astype("float32")
            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums
        return probs

    case.generate_by_parameter(
        (
            ("expert_routemap_topk", expert_routemap_value),
            ("expert_prob_topk", expert_prob_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.moe_unpermute")
def moe_unpermute_values(ctx: InputGenerationContext, case: RuleCase):
    def expert_routemap_value(binding):
        num_experts = case.arg(5, "num_experts", 32)
        total_zipped_tokens = case.arg(4, "total_zipped_tokens")
        hidden_config = case.arg(0, "hidden_states_unzipped")
        rowmap_config = case.arg(1, "zipped_expertwise_rowmap")
        prob_config = case.arg(3, "token_prob_unzipped")
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if (
            not isinstance(total_zipped_tokens, int)
            or isinstance(total_zipped_tokens, bool)
            or total_zipped_tokens < 0
        ):
            raise ValueError("total_zipped_tokens must be a non-negative integer")
        if not (
            case.is_tensor_config(hidden_config)
            and len(hidden_config.shape) == 2
            and hidden_config.dtype in {"bfloat16", "float32"}
        ):
            raise ValueError("hidden_states_unzipped must be a rank-2 bfloat16 tensor")
        if not (
            case.is_tensor_config(rowmap_config)
            and len(rowmap_config.shape) == 2
            and rowmap_config.dtype == "int32"
            and tuple(rowmap_config.shape) == (total_zipped_tokens, num_experts)
        ):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        if not (
            case.is_tensor_config(prob_config)
            and len(prob_config.shape) in (1, 2)
            and prob_config.shape[0] == hidden_config.shape[0]
            and (len(prob_config.shape) == 1 or prob_config.shape[1] == 1)
            and prob_config.dtype == "float32"
        ):
            raise ValueError(
                "token_prob_unzipped must have shape "
                "[seqlen_broadcasted] or [seqlen_broadcasted, 1] and dtype float32"
            )
        if binding.spec.dtype != "int32" or len(binding.spec.shape) != 2:
            raise ValueError("expert_routemap_topk must be a rank-2 int32 tensor")
        seqlen, topk = binding.spec.shape[0], binding.spec.shape[1]
        if seqlen != total_zipped_tokens:
            raise ValueError("expert_routemap_topk sequence length must equal total_zipped_tokens")
        if topk <= 0:
            raise ValueError("topk should be greater than 0")
        routemap = case.full(binding.spec.shape, -1, dtype="int32")
        max_assign = min(topk, num_experts)
        route_count = min(hidden_config.shape[0], seqlen * max_assign)
        positions = case.arange(route_count, dtype="int64")
        rows = positions % seqlen
        columns = positions // seqlen
        routemap[rows, columns] = (rows + columns) % num_experts
        return routemap

    def rowmap_value(binding):
        routemap_binding = case.find("expert_routemap_topk")
        if routemap_binding is not None and not case.is_generated(routemap_binding):
            case.set_value(routemap_binding, expert_routemap_value(routemap_binding))
        routemap_config = case.arg(2, "expert_routemap_topk")
        num_experts = case.arg(5, "num_experts", 32)
        total_zipped_tokens = case.arg(4, "total_zipped_tokens")
        hidden_config = case.arg(0, "hidden_states_unzipped")
        seqlen = total_zipped_tokens
        unzipped_seqlen = hidden_config.shape[0] if case.is_tensor_config(hidden_config) else seqlen
        if binding.spec.dtype != "int32" or tuple(binding.spec.shape) != (seqlen, num_experts):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        rowmap = case.full(binding.spec.shape, -1, dtype="int32")
        if case.is_tensor_config(routemap_config) and routemap_binding is not None:
            routemap = case.value(routemap_binding)
            expert_counts = case.array(
                [case.count_nonzero(routemap == expert) for expert in range(num_experts)],
                dtype="int64",
            )
            if int(expert_counts.sum()) > unzipped_seqlen:
                raise ValueError("routemap assignments exceed hidden_states_unzipped capacity")
            expert_offsets = case.zeros(num_experts, dtype="int64")
            expert_offsets[1:] = case.cumsum(expert_counts[:-1])
            expert_counters = case.zeros(num_experts, dtype="int64")
            for row_index in range(seqlen):
                for expert in range(num_experts):
                    positions = case.nonzero(routemap[row_index] == expert)[0]
                    if positions.size == 0:
                        continue
                    rowmap[row_index, expert] = expert_offsets[expert] + expert_counters[expert]
                    expert_counters[expert] += 1
        return rowmap

    def token_prob_value(binding):
        hidden_config = case.arg(0, "hidden_states_unzipped")
        if not (
            binding.spec.dtype == "float32"
            and len(binding.spec.shape) in (1, 2)
            and case.is_tensor_config(hidden_config)
            and binding.spec.shape[0] == hidden_config.shape[0]
            and (len(binding.spec.shape) == 1 or binding.spec.shape[1] == 1)
        ):
            raise ValueError(
                "token_prob_unzipped must match the broadcasted sequence "
                "length and have dtype float32"
            )
        return case.random(binding.spec.shape).astype("float32")

    for binding in ctx.call.tensors:
        if case.is_generated(binding):
            continue
        if binding.parameter_name == "expert_routemap_topk":
            case.set_value(binding, expert_routemap_value(binding))
        elif binding.parameter_name == "zipped_expertwise_rowmap":
            case.set_value(binding, rowmap_value(binding))
        elif binding.parameter_name == "token_prob_unzipped":
            case.set_value(binding, token_prob_value(binding))
        else:
            case.set_value(binding, case.value_domain("default", binding))


@rules.register(*tuple(sorted(not_zero_apis)))
def nonzero_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("nonzero")


@rules.register("paddle.bernoulli")
def bernoulli_probability(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("unit_interval")


@rules.register("paddle.standard_gamma")
def standard_gamma_unit_interval(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("unit_interval")


@rules.register("paddle.poisson")
def poisson_unit_interval(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("unit_interval")


@rules.register(
    "paddle.sqrt",
    aliases=("paddle.Tensor.sqrt",),
)
def sqrt_nonnegative(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("uniform", low=0, high=1000)


@rules.register(
    "paddle.rsqrt",
    aliases=("paddle.Tensor.rsqrt",),
)
def rsqrt_positive(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("uniform", low=1e-7, high=1000)


@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(ctx: InputGenerationContext, case: RuleCase):
    x_binding = case.find("x")
    min_binding = case.find("min")
    max_binding = case.find("max")
    min_config = case.arg(1, "min")
    max_config = case.arg(2, "max")

    if case.is_tensor_config(min_config) and case.is_tensor_config(max_config):
        min_value = case.value_domain("random_range", min_binding)
        max_value = case.value_domain("random_range", max_binding, low=min_value)
        case.set_value(min_binding, min_value)
        case.set_value(max_binding, max_value)
    elif min_config is not None and max_config is not None:
        if case.is_tensor_config(min_config) and isinstance(max_config, (int, float)):
            min_value = case.value_domain("random_range", min_binding, high=max_config)
            case.set_value(min_binding, min_value)
        elif case.is_tensor_config(max_config) and isinstance(min_config, (int, float)):
            max_value = case.value_domain("random_range", max_binding, low=min_config)
            case.set_value(max_binding, max_value)

    if x_binding is not None:
        case.set_value(
            x_binding,
            case.value_domain("random_range", x_binding),
        )
    case.generate_remaining("default")


@rules.register("paddle.multiply")
def multiply_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("multiply")


@rules.register(
    "paddle.nn.functional.binary_cross_entropy",
)
def binary_cross_entropy_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("unit_interval")


@rules.register("paddle.nn.functional.alpha_dropout")
def alpha_dropout_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("x", "unit_interval"),), default="default")


@rules.register("paddle.nn.functional.conv2d_transpose")
def conv2d_transpose_values(ctx: InputGenerationContext, case: RuleCase):
    def tensor_value(binding):
        if "int" in binding.spec.dtype:
            return case.randint(-65535, 65535, size=binding.spec.shape).astype(binding.spec.dtype)
        return (case.random(binding.spec.shape) - 0.5).astype(binding.spec.dtype)

    case.generate_by_parameter(
        ((("x", "weight", "bias"), tensor_value),),
        default="default",
    )


@rules.register("paddle.vision.ops.distribute_fpn_proposals")
def distribute_fpn_proposals_values(ctx: InputGenerationContext, case: RuleCase):
    state = {"num": None}

    def fpn_rois_value(binding):
        num = binding.spec.shape[0]
        state["num"] = num
        rois = case.randint(1, 1024, size=[num, 4])
        rois[:, 0] = rois[:, 0] + case.random([num])
        rois[:, 1] = rois[:, 1] + case.random([num])
        rois[:, 2] = rois[:, 0] + case.randint(1, 1024, size=[num]) + case.random([num])
        rois[:, 3] = rois[:, 1] + case.randint(1, 1024, size=[num]) + case.random([num])
        return rois

    def rois_num_value(binding):
        if state["num"] is None:
            fpn_rois = case.arg(0, "fpn_rois")
            state["num"] = fpn_rois.shape[0]
        num = state["num"]
        remaining = binding.spec.shape[0]
        result = case.zeros(binding.spec.shape)
        if num > 4096 or remaining > 4096:
            if num < remaining:
                result[:num] = 1
            else:
                result += num // remaining
                result[: num % remaining] += 1
        elif num < remaining:
            indices = case.choice(remaining, num, replace=False)
            result[indices] = 1
        else:
            for index in range(binding.spec.shape[0] - 1):
                result[index] = case.randint(1, num - remaining + 2)
                num -= result[index]
                remaining -= 1
            result[binding.spec.shape[0] - 1] = num
        return result

    case.generate_by_parameter(
        (
            ("fpn_rois", fpn_rois_value),
            ("rois_num", rois_num_value),
        )
    )


@rules.register("paddle.vision.ops.generate_proposals")
def generate_proposals_values(ctx: InputGenerationContext, case: RuleCase):
    def random_value(binding):
        return case.random(binding.spec.shape, dtype=binding.spec.dtype)

    def img_size_value(binding):
        return case.randint(0, 1024, size=binding.spec.shape).astype(binding.spec.dtype)

    def anchors_value(binding):
        anchors = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        width = binding.spec.shape[0]
        height = binding.spec.shape[1]
        for index in range(binding.spec.shape[0]):
            anchors[index][0] = case.random() * width
            anchors[index][1] = case.random() * height
            anchors[index][2] = (
                case.random() * (width - anchors[index][0] + 1) + anchors[index][0] + 1
            )
            anchors[index][3] = (
                case.random() * (height - anchors[index][1] + 1) + anchors[index][1] + 1
            )
        return anchors

    for binding in ctx.call.tensors:
        if binding.parameter_name in {"scores", "bbox_deltas"}:
            case.set_value(binding, random_value(binding))
        elif binding.parameter_name == "img_size":
            case.set_value(binding, img_size_value(binding))
        elif binding.parameter_name == "anchors":
            case.set_value(binding, anchors_value(binding))
        else:
            case.set_value(binding, case.value_domain("default", binding))


@rules.register("paddle.vision.ops.nms")
def nms_values(ctx: InputGenerationContext, case: RuleCase):
    def boxes_value(binding):
        boxes = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(binding.spec.shape[0]):
            boxes[index][0] = case.random() * 1023
            boxes[index][1] = case.random() * 1023
            boxes[index][2] = case.random() * (1024 - boxes[index][0] + 1) + boxes[index][0] + 1
            boxes[index][3] = case.random() * (1024 - boxes[index][1] + 1) + boxes[index][1] + 1
        return boxes

    def scores_value(binding):
        return case.random(binding.spec.shape, dtype=binding.spec.dtype)

    def default_vision_value(binding):
        return case.randint(0, 1024, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("boxes", boxes_value),
            ("scores", scores_value),
        ),
        default=default_vision_value,
    )


@rules.register(
    "paddle.vision.ops.roi_align",
    "paddle.vision.ops.roi_pool",
    "paddle.vision.ops.psroi_pool",
)
def roi_pool_values(ctx: InputGenerationContext, case: RuleCase):
    state = {"x_shape": None, "boxes_shape": None}

    def x_value(binding):
        state["x_shape"] = binding.spec.shape
        return (case.random(binding.spec.shape) * 255).astype(binding.spec.dtype)

    def boxes_value(binding):
        if state["x_shape"] is None:
            x = case.arg(0, "x")
            state["x_shape"] = x.shape
        state["boxes_shape"] = binding.spec.shape
        boxes = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(binding.spec.shape[0]):
            boxes[index][0] = case.random() * (state["x_shape"][2] - 2)
            boxes[index][1] = case.random() * (state["x_shape"][3] - 2)
            boxes[index][2] = (
                case.random() * (state["x_shape"][2] - 1 - boxes[index][0] + 1)
                + boxes[index][0]
                + 1
            )
            boxes[index][3] = (
                case.random() * (state["x_shape"][3] - 1 - boxes[index][1] + 1)
                + boxes[index][1]
                + 1
            )
        return boxes

    def boxes_num_value(binding):
        if state["boxes_shape"] is None:
            boxes = case.arg(1, "boxes")
            state["boxes_shape"] = boxes.shape
        boxes_remaining = state["boxes_shape"][0]
        result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        numel = math.prod(binding.spec.shape)
        for index in range(numel - 1):
            if boxes_remaining < numel:
                result[index] = 0
            else:
                result[index] = case.randint(1, boxes_remaining - (numel - 1 - index) + 1)
                boxes_remaining -= result[index]
        result[numel - 1] = boxes_remaining
        return result

    case.generate_by_parameter(
        (
            ("x", x_value),
            ("boxes", boxes_value),
            ("boxes_num", boxes_num_value),
        ),
        default="default",
    )


@rules.register(
    "paddle.gammainc",
    "paddle.gammaincc",
    "paddle.linspace",
)
def zero_65535_or_unit_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("int_zero_65535_else_unit")


@rules.register("paddle.dot")
def dot_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("int_minus127_127_else_default")


@rules.register("paddle.normal")
def normal_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter(
        (
            ("mean", "signed_half_interval"),
            ("std", "normal_std"),
        ),
        default="int_zero_1024",
    )


@rules.register("paddle.ones")
def ones_shape(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("ones_shape")


@rules.register("paddle.zeros")
def zeros_shape(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("int_zero_2048_no_cast")


@rules.register("paddle.eye")
def eye_shape(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("int_zero_2048_no_cast")


@rules.register(
    "paddle.nn.functional.interpolate",
    "paddle.Tensor.tile",
    "paddle.tile",
)
def shape_parameter_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter(
        ((("size", "scale_factor", "repeat_times"), "int_one_128"),),
        default="default",
    )


@rules.register("paddle.nn.functional.upsample")
def upsample_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter(
        (
            ("size", "int_one_128"),
            ("scale_factor", "abs_unit_plus_one"),
        ),
        default="default",
    )


@rules.register(
    "paddle.nn.functional.gaussian_nll_loss",
)
def gaussian_nll_loss_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter(
        ((("var", "variance"), "unit_interval_plus_one"),),
        default="default",
    )


@rules.register(
    "paddle.nn.functional.hinge_embedding_loss",
)
def hinge_embedding_loss_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("label", "hinge_labels"),), default="default")


@rules.register(
    "paddle.nn.functional.sigmoid_focal_loss",
)
def sigmoid_focal_loss_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("label", "binary_0_1"),), default="default")


@rules.register("paddle.full")
def full_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter(
        (
            ("shape", "int_zero_64"),
            ("fill_value", "full_fill_value"),
        ),
        default="int_zero_64",
    )


@rules.register("paddle.standard_normal")
def standard_normal_shape(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("shape", "int_one_128"),), default="default")


@rules.register("paddle.logspace")
def logspace_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("num", "int_one_65535_no_cast"),), default="default")


@rules.register("paddle.quantile")
def quantile_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("q", "quantile_q"),), default="default")


@rules.register(
    "paddle.remainder",
    aliases=("paddle.Tensor.remainder",),
)
def remainder_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("y", "remainder_rhs"),), default="default")


@rules.register(
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
)
def dropout_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("p", "dropout_probability"),), default="default")


@rules.register("paddle.atan2")
def atan2_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("unit_interval_plus_one")


@rules.register("paddle.bincount")
def bincount_values(ctx: InputGenerationContext, case: RuleCase):
    def integer_value(binding):
        return case.randint(0, 65535, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("x", integer_value),
            ("minlength", integer_value),
        ),
        default="default",
    )


@rules.register(
    "paddle.nn.functional.adaptive_avg_pool2d", "paddle.nn.functional.adaptive_avg_pool3d"
)
def adaptive_avg_pool_values(ctx: InputGenerationContext, case: RuleCase):
    def output_size(binding):
        x_shape = case.arg(0, "x").shape
        return case.randint(1, 2 * max(x_shape), size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (("output_size", output_size),),
        default="default",
    )


@rules.register("paddle.empty")
def empty_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_by_parameter((("shape", "empty_shape"),), default="default")


@rules.register(
    "paddle.repeat_interleave",
    aliases=("paddle.Tensor.repeat_interleave",),
)
def repeat_interleave_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x = case.arg(0, "x")
        input_dims = len(x.shape)
        if len(binding.spec.shape) == 0:
            return case.array(case.randint(-input_dims, input_dims), dtype=binding.spec.dtype)
        return case.randint(-input_dims, input_dims, size=binding.spec.shape).astype(
            binding.spec.dtype
        )

    case.generate_by_parameter(
        (
            ("repeats", "int_one_2048"),
            ("axis", axis_value),
        ),
        default="default",
    )


@rules.register(
    "paddle.put_along_axis",
    aliases=(
        "paddle.Tensor.put_along_axis",
        "paddle.put_along_axis_",
        "paddle.Tensor.put_along_axis_",
        "paddle._C_ops.put_along_axis",
        "paddle._C_ops.put_along_axis_",
        "paddle._C_ops.Tensor.put_along_axis",
        "paddle._C_ops.Tensor.put_along_axis_",
    ),
)
def put_along_axis_values(ctx: InputGenerationContext, case: RuleCase):
    def random_tensor_value(binding, shape):
        scalar_spec = TensorView(
            shape=tuple(shape),
            dtype=binding.spec.dtype,
            place=binding.spec.place,
            is_contiguous=binding.spec.is_contiguous,
            strides=binding.spec.strides,
        )
        return generate_random_range(scalar_spec, rng=case.rng)

    def indices_value(binding):
        x_tensor = case.arg(0, "arr", case.arg(0, "x"))
        x_shape = tuple(x_tensor.shape) if x_tensor is not None else ()
        x_dims = len(x_shape)
        current_shape = tuple(binding.spec.shape)
        if len(current_shape) != x_dims:
            new_shape = [current_shape[i] if i < len(current_shape) else 1 for i in range(x_dims)]
            indices = case.zeros(new_shape, dtype="int64")
            for axis in range(x_dims):
                if axis < len(current_shape):
                    dim_size = x_shape[axis]
                    if dim_size > 0:
                        axis_indices = case.choice(
                            dim_size, size=new_shape[axis], replace=False
                        ).astype("int64")
                        idx_tuple = tuple(
                            [slice(None)] * axis
                            + [slice(None, new_shape[axis])]
                            + [slice(None)] * (x_dims - axis - 1)
                        )
                        indices[idx_tuple] = axis_indices.reshape([-1] + [1] * (x_dims - axis - 1))
            return indices
        axis = case.arg(3, "axis", 0)
        axis = axis if isinstance(axis, int) else 0
        axis = axis if axis >= 0 else axis + x_dims
        indices = case.zeros(current_shape, dtype="int64")
        if 0 <= axis < x_dims:
            dim_size = x_shape[axis]
            for idx in numpy.ndindex(tuple(current_shape[:-1])):
                indices[idx] = case.choice(dim_size, size=current_shape[-1], replace=False)
        return indices

    def values_value(binding):
        indices_binding = case.find("indices")
        if indices_binding is not None:
            indices = case.value(indices_binding)
            if tuple(indices.shape) != tuple(binding.spec.shape):
                if numpy.prod(binding.spec.shape) == 1:
                    return case.set_value_raw(
                        binding,
                        case.full(
                            indices.shape,
                            random_tensor_value(binding, ())[()],
                            dtype=binding.spec.dtype,
                        ),
                    )
                return case.set_value_raw(binding, random_tensor_value(binding, indices.shape))
            return random_tensor_value(binding, binding.spec.shape)
        return case.value_domain("default", binding)

    case.generate_by_parameter(
        (("indices", indices_value), ("values", values_value)), default="default"
    )


@rules.register("paddle.matrix_transpose")
def matrix_transpose_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        shape = binding.spec.shape if len(binding.spec.shape) >= 2 else (2, 2)
        dtype = binding.spec.dtype
        if "int" in dtype:
            return case.randint(-65535, 65535, size=shape).astype(dtype)
        return (case.random(shape) - 0.5).astype(dtype)

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.nn.functional.softmax")
def softmax_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        return case.value_domain("uniform", binding, low=-len(x_shape), high=len(x_shape))

    case.generate_by_parameter(
        (
            ("x", "random_range"),
            ("axis", axis_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.zeropad2d")
def zeropad2d_values(ctx: InputGenerationContext, case: RuleCase):
    def padding_value(binding):
        return case.value_domain("uniform", binding, low=0, high=10)

    case.generate_by_parameter(
        (
            ("x", "random_range"),
            ("padding", padding_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.pad")
def pad_values(ctx: InputGenerationContext, case: RuleCase):
    def pad_value(binding):
        x_shape = case.arg(0, "x").shape
        return case.value_domain("uniform", binding, low=0, high=min(x_shape))

    case.generate_by_parameter((("pad", pad_value),), default="default")


@rules.register("paddle.nn.functional.class_center_sample")
def class_center_sample_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        num_classes = case.arg(1, "num_classes")
        return case.randint(0, num_classes, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.shard_index")
def shard_index_values(ctx: InputGenerationContext, case: RuleCase):
    def input_binding(binding):
        index_num = case.arg(1, "index_num")
        if index_num is None:
            index_num = case.randint(1, 1000)
        return case.randint(0, index_num, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("input", input_binding),), default="default")


@rules.register("paddle.incubate.nn.functional.masked_multihead_attention")
def masked_multihead_attention_values(ctx: InputGenerationContext, case: RuleCase):
    def sequence_lengths(binding):
        return case.value_domain("random_range", binding, low=1)

    def rotary_tensor(binding):
        return case.value_domain("uniform", binding, low=0, high=1000)

    case.generate_by_parameter(
        (
            ("sequence_lengths", sequence_lengths),
            ("rotary_tensor", rotary_tensor),
        ),
        default="default",
    )


@rules.register(
    "paddle.argmax",
    "paddle.argmin",
    aliases=("paddle.Tensor.argmax", "paddle.Tensor.argmin"),
)
def argminmax_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        min_dim = len(x_shape)
        return case.randint(-min_dim, min_dim - 1, size=binding.spec.shape).astype("int64")

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register("paddle.cumsum", aliases=("paddle.Tensor.cumsum",))
def cumsum_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        return case.randint(-len(x_shape), len(x_shape), size=binding.spec.shape)

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register(
    "paddle.mean", "paddle.max", "paddle.min", "paddle.prod", "paddle.sum", "paddle.squeeze"
)
def reduction_axis_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        max_dim = max(len(x_shape), 1)
        if len(binding.spec.shape) == 0:
            dim = case.randint(0, max_dim)
            if case.random() > 0.5:
                dim -= max_dim
            return case.array(dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1:
            dims = case.choice(max_dim, size=binding.spec.shape[0], replace=False)
            mask = case.random(binding.spec.shape[0]) > 0.5
            dims = case.where(mask, dims - max_dim, dims)
            return case.array(dims, dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {case.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register("paddle.unsqueeze")
def unsqueeze_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        max_dim = len(x_shape) + 1
        if len(binding.spec.shape) == 0:
            dim = case.randint(0, max_dim)
            if case.random() > 0.5:
                dim -= max_dim
            return case.array(dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1:
            dims = case.choice(max_dim, size=binding.spec.shape[0], replace=False)
            mask = case.random(binding.spec.shape[0]) > 0.5
            dims = case.where(mask, dims - max_dim, dims)
            return case.array(dims, dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register("paddle.unflatten", aliases=("paddle.Tensor.unflatten",))
def unflatten_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        return case.randint(0, len(x_shape), size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register("paddle.topk", aliases=("paddle.Tensor.topk",))
def topk_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        dtype = binding.spec.dtype
        if dtype == "bfloat16" or dtype in {"float8_e4m3fn", "float8_e5m2"}:
            dtype = "float32" if dtype == "bfloat16" else "float16"
        if dtype in {"float32", "float64"}:
            return ((case.random(binding.spec.shape) - 0.5) * 1.2).astype(dtype)
        if dtype == "float16":
            return (case.randn(*binding.spec.shape).astype(dtype) * 1e-3).astype(dtype)
        if dtype in {"int32", "int64"}:
            return case.randint(-10, 10, size=binding.spec.shape).astype(dtype)
        raise ValueError(
            f"Unsupported dtype {binding.spec.dtype} for paddle.topk / paddle.Tensor.topk"
        )

    def k_value(binding):
        x_config = case.arg(0, "x")
        axis = case.arg(2, "axis", -1)
        max_k_value = 1
        if x_config is not None and x_config.shape:
            max_k_value = x_config.shape[axis] if len(x_config.shape) > 0 else 1
        if not binding.spec.shape:
            return case.array(case.randint(1, max_k_value + 1), dtype=binding.spec.dtype)
        return case.randint(1, max_k_value + 1, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("x", x_value),
            ("k", k_value),
        ),
        default="default",
    )


@rules.register("paddle.index_sample")
def index_sample_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        x_dim = case.arg(0, "x").shape[1]
        return case.randint(0, x_dim, size=binding.spec.shape)

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.Tensor.__getitem__")
def tensor_getitem_values(ctx: InputGenerationContext, case: RuleCase):
    def source_binding():
        binding = case.find("arr") or case.find("x") or case.find("self")
        if binding is None:
            raise ValueError("Tensor.__getitem__ rule could not find source tensor")
        return binding

    def item_value(binding):
        min_dim = min(source_binding().spec.shape)
        numel = math.prod(binding.spec.shape)
        if binding.spec.dtype == "bool":
            indices = case.choice([0, 1], size=numel)
        else:
            indices = case.randint(0, min_dim, size=numel)
        return indices.reshape(binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("item", item_value),), default="default")


@rules.register("paddle.Tensor.__setitem__")
def tensor_setitem_values(ctx: InputGenerationContext, case: RuleCase):
    def source_binding():
        binding = case.find("arr") or case.find("x") or case.find("self")
        if binding is None:
            raise ValueError("Tensor.__setitem__ rule could not find source tensor")
        return binding

    def item_value(binding):
        min_dim = min(source_binding().spec.shape)
        numel = math.prod(binding.spec.shape)
        if binding.spec.dtype == "bool":
            value = case.arg(2, "value")
            if value is not None and hasattr(value, "shape"):
                indices = case.zeros(numel, dtype="int64")
                num_true = min(value.shape[0], numel)
                true_indices = case.choice(numel, size=num_true, replace=False)
                indices[true_indices] = 1
            else:
                indices = case.choice([0, 1], size=numel)
        else:
            indices = case.randint(0, min_dim, size=numel)
        return indices.reshape(binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("item", item_value),), default="default")


@rules.register("paddle.index_add", "paddle.index_fill")
def index_update_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        axis = case.arg(2, "axis")
        if axis is None:
            raise ValueError("Axis is None")
        x_shape = case.arg(0, "x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
        if len(binding.spec.shape) >= 1:
            return case.randint(0, x_shape[axis], size=binding.spec.shape).astype(
                binding.spec.dtype
            )
        raise ValueError(
            f"Invalid shape for 'index' Tensor in {case.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.take")
def take_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        x = case.arg(0, "x")
        dim_size = math.prod(x.shape)
        return case.randint(0, dim_size, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.gather", aliases=("paddle.Tensor.gather",))
def gather_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        x = case.arg(0, "x")
        if case.has_kwarg("axis"):
            axis = case.arg(2, "axis")
            if hasattr(axis, "shape"):
                axis = axis.shape[0]
        else:
            axis = 0
        return case.randint(0, x.shape[axis], size=binding.spec.shape).astype(binding.spec.dtype)

    def axis_value(binding):
        return case.randint(0, 2, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("index", index_value),
            ("axis", axis_value),
        ),
        default="default",
    )


@rules.register("paddle.gather_nd", aliases=("paddle.Tensor.gather_nd",))
def gather_nd_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        x_shape = case.arg(0, "x").shape
        index_shape = case.arg(1, "index").shape
        result = case.zeros(index_shape, dtype=binding.spec.dtype)
        for index in range(index_shape[-1]):
            result[..., index] = case.randint(0, x_shape[index], size=result[..., index].shape)
        return result

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.index_select", aliases=("paddle.Tensor.index_select",))
def index_select_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        axis = case.arg(2, "axis")
        if axis is None:
            axis = 0
        x = case.arg(0, "x")
        if x.shape[axis] == 0:
            return case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return case.randint(0, x.shape[axis], size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.take_along_axis", aliases=("paddle.Tensor.take_along_axis",))
def take_along_axis_values(ctx: InputGenerationContext, case: RuleCase):
    def indices_value(binding):
        arr_shape = case.arg(0, "arr").shape
        axis = case.arg(2, "axis")
        axis_value = axis if axis >= 0 else axis + len(arr_shape)
        dim_size = arr_shape[axis_value]
        dtype = binding.spec.dtype if binding.spec.dtype in {"int32", "int64"} else "int64"
        num_elements = math.prod(binding.spec.shape)
        if num_elements == 0:
            indices = case.array([], dtype=dtype)
        elif dim_size == 1:
            indices = case.zeros(num_elements, dtype=dtype)
        elif num_elements == 1:
            indices = case.array([0], dtype=dtype)
        else:
            indices = case.randint(0, dim_size, size=num_elements).astype(dtype)
            positions_to_replace = case.choice(num_elements, size=2, replace=False)
            flat_indices = indices.flatten()
            flat_indices[positions_to_replace[0]] = 0
            flat_indices[positions_to_replace[1]] = dim_size - 1
            indices = flat_indices
        return indices.reshape(binding.spec.shape)

    case.generate_by_parameter((("indices", indices_value),), default="default")


@rules.register("paddle.index_put", aliases=("paddle.Tensor.index_put",))
def index_put_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("default")


@rules.register("paddle.multiplex")
def multiplex_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        inputs = case.arg(0, "inputs")
        return case.randint(0, len(inputs), size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register(
    "paddle.geometric.segment_sum",
    "paddle.geometric.segment_max",
    "paddle.geometric.segment_mean",
    "paddle.geometric.segment_min",
    "paddle.incubate.segment_sum",
    "paddle.incubate.segment_max",
    "paddle.incubate.segment_mean",
    "paddle.incubate.segment_min",
)
def segment_values(ctx: InputGenerationContext, case: RuleCase):
    def segment_ids_value(binding):
        batch_size = case.arg(0, "x").shape[0]
        max_segments = case.randint(1, batch_size + 1)
        values = case.randint(0, max_segments, size=binding.spec.shape).astype(binding.spec.dtype)
        return case.sort(values)

    case.generate_by_parameter((("segment_ids", segment_ids_value),), default="default")


@rules.register(
    "paddle.geometric.send_u_recv",
    "paddle.geometric.send_uv",
    "paddle.geometric.send_ue_recv",
)
def geometric_send_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        num_nodes = case.arg(0, "x").shape[0]
        return case.randint(0, num_nodes, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        ((("src_index", "dst_index"), index_value),),
        default="default",
    )


@rules.register("paddle.geometric.sample_neighbors")
def sample_neighbors_values(ctx: InputGenerationContext, case: RuleCase):
    def row_value(binding):
        colptr_shape = case.arg(1, "colptr").shape
        num_nodes = colptr_shape[0] - 1
        return case.randint(0, num_nodes, size=binding.spec.shape, dtype=binding.spec.dtype)

    def colptr_value(binding):
        row = case.arg(0, "row")
        num_edges = row.shape[0]
        num_nodes = binding.spec.shape[0] - 1
        colptr = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        if num_nodes > 0 and num_edges > 0:
            splits = case.choice(
                case.arange(num_edges + 1),
                size=num_nodes - 1,
                replace=True,
            )
            splits.sort()
            colptr[1:num_nodes] = splits
            colptr[num_nodes] = num_edges
        return colptr

    def input_nodes_value(binding):
        num_nodes = binding.spec.shape[0] - 1
        return case.randint(0, num_nodes, size=binding.spec.shape, dtype=binding.spec.dtype)

    def edge_order_value(binding):
        num_edges = case.arg(0, "row").shape[0]
        return case.arange(num_edges, dtype=binding.spec.dtype).reshape(binding.spec.shape)

    case.generate_by_parameter(
        (
            ("row", row_value),
            ("colptr", colptr_value),
            ("input_nodes", input_nodes_value),
            (("eids", "perm_buffer"), edge_order_value),
        ),
        default="default",
    )


@rules.register("paddle.reshape", aliases=("paddle.Tensor.reshape",))
def reshape_values(ctx: InputGenerationContext, case: RuleCase):
    state = {
        "shape": None,
        "maxvalue": None,
        "tensornum": None,
    }

    def initialize_from_x(binding):
        shape = binding.spec.shape
        if 0 not in shape and state["shape"] is None:
            state["shape"] = shape
            state["maxvalue"] = math.prod(shape)
            state["tensornum"] = 0
            for candidate in case.argument_values():
                if isinstance(candidate, (list, tuple)):
                    for index, item in enumerate(candidate):
                        if isinstance(item, numbers.Integral):
                            if item == 0:
                                state["maxvalue"] //= shape[index]
                            elif item != -1:
                                state["maxvalue"] //= int(item)
                        elif case.is_tensor_config(item):
                            state["tensornum"] += 1
        return case.value_domain("default", binding)

    def shape_value(binding):
        if state["tensornum"] == 0:
            state["tensornum"] = 1
        dtype = "int32"
        shape = binding.spec.shape
        maxvalue = state["maxvalue"]
        if shape not in ((), (1,)):
            result = case.zeros(shape, dtype=dtype)
            for index in range(shape[0]):
                if index < shape[0] - 1:
                    result[index] = case.randint(1, maxvalue + 1)
                    while maxvalue % result[index]:
                        result[index] = case.randint(1, maxvalue + 1)
                    maxvalue //= result[index]
                else:
                    result[index] = maxvalue
            state["maxvalue"] = maxvalue
            return result
        if state["tensornum"] == 1:
            return case.randint(maxvalue, maxvalue + 1, size=shape).astype(dtype)
        state["tensornum"] -= 1
        result = case.randint(1, maxvalue + 1, size=shape).astype(dtype)
        while maxvalue % result:
            result = case.randint(1, maxvalue + 1, size=shape).astype(dtype)
        state["maxvalue"] = maxvalue // result
        return result

    case.generate_by_parameter(
        (
            ("x", initialize_from_x),
            ("shape", shape_value),
        ),
        default="default",
    )


@rules.register("paddle.slice")
def slice_values(ctx: InputGenerationContext, case: RuleCase):
    state = {
        "shape": None,
        "indice": 0,
        "start": [],
        "index": 0,
    }

    def axes():
        return case.arg(1, "axes")

    def input_binding(binding):
        if state["shape"] is None:
            state["shape"] = binding.spec.shape
        return case.value_domain("default", binding)

    def starts_value(binding):
        dim_sizes = [state["shape"][axis] for axis in axes()]
        if binding.spec.shape == ():
            coin = case.randint(0, 2)
            if coin == 0:
                value = case.randint(0, dim_sizes[state["indice"]] - 1, binding.spec.shape)
            else:
                value = case.randint(-65535, -1, binding.spec.shape)
            state["start"].append(value)
            state["indice"] += 1
            return case.array(value, dtype=binding.spec.dtype)
        result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(math.prod(binding.spec.shape)):
            coin = case.randint(0, 2)
            if coin == 0:
                result[index] = case.randint(0, dim_sizes[state["indice"]] - 1)
            else:
                result[index] = case.randint(-65535, -1)
            state["start"].append(result[index])
            state["indice"] += 1
        return result

    def ends_value(binding):
        if not state["start"]:
            start_arg = case.arg(2, "starts")
            state["start"] = list(
                start_arg if isinstance(start_arg, (list, tuple)) else [start_arg]
            )
        dim_sizes = [state["shape"][axis] for axis in axes()]
        start = state["start"]
        for index, item in enumerate(start):
            if item < 0:
                item = item if item > -dim_sizes[index] else -dim_sizes[index]
                start[index] = item + dim_sizes[index]
        if binding.spec.shape == ():
            coin = case.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                value = case.randint(current + 1, 65535, binding.spec.shape)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                value = case.randint(min(current - dim_sizes[index] + 1, -1), 0, binding.spec.shape)
            state["index"] += 1
            return case.array(value, dtype=binding.spec.dtype)
        result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(math.prod(binding.spec.shape)):
            coin = case.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                result[index] = case.randint(current + 1, 65535)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                result[index] = case.randint(current - dim_sizes[state["index"]] + 1, 0)
            state["index"] += 1
        return result

    case.generate_by_parameter(
        (
            ("input", input_binding),
            ("starts", starts_value),
            ("ends", ends_value),
        ),
        default="default",
    )


@rules.register("paddle.scatter")
def scatter_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        x = case.arg(0, "x")
        first_dim = x.shape[0]
        overwrite = case.arg(3, "overwrite")
        if (overwrite is None or overwrite is True) and (
            binding.spec.shape == () or binding.spec.shape[0]
        ) <= first_dim:
            return case.choice(first_dim, size=binding.spec.shape, replace=False).astype(
                binding.spec.dtype
            )
        return case.randint(0, first_dim, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.scatter_nd")
def scatter_nd_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        output_shape = case.arg(2, "shape")
        if output_shape and len(output_shape):
            result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
            for axis in range(len(output_shape)):
                if axis >= binding.spec.shape[-1]:
                    break
                result[..., axis] = case.randint(
                    -output_shape[axis],
                    output_shape[axis],
                    size=result[..., axis].shape,
                ).astype(binding.spec.dtype)
            return result
        return case.value_domain("default", binding)

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.scatter_nd_add")
def scatter_nd_add_values(ctx: InputGenerationContext, case: RuleCase):
    def index_value(binding):
        x_shape = case.arg(0, "x").shape
        result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for axis in range(binding.spec.shape[-1]):
            result[..., axis] = case.randint(
                -x_shape[axis],
                x_shape[axis],
                size=result[..., axis].shape,
            ).astype(binding.spec.dtype)
        return result

    case.generate_by_parameter((("index", index_value),), default="default")


@rules.register("paddle.strided_slice")
def strided_slice_values(ctx: InputGenerationContext, case: RuleCase):
    def axes_value(binding):
        x = case.arg(0, "x")
        return case.randint(0, len(x.shape), size=binding.spec.shape).astype(binding.spec.dtype)

    def list_value(binding):
        x = case.arg(0, "x")
        axes_arg = case.arg(1, "axes")
        axes = axes_arg
        if not isinstance(axes, list):
            axes = case.value(case.find("axes"))
        list_index_path = case.binding_list_index(binding)
        if list_index_path is None:
            return case.value_domain("default", binding)
        list_index = list_index_path[0]
        parameter = binding.parameter_name
        if parameter == "starts":
            return case.randint(0, x.shape[axes[list_index]] - 1, size=binding.spec.shape).astype(
                binding.spec.dtype
            )
        if parameter == "ends":
            return case.randint(
                case.value(case.find("starts"))[list_index] + 1,
                x.shape[axes[list_index]],
                size=binding.spec.shape,
            ).astype(binding.spec.dtype)
        if parameter == "strides":
            return case.randint(1, x.shape[axes[list_index]], size=binding.spec.shape).astype(
                binding.spec.dtype
            )
        return case.value_domain("default", binding)

    case.generate_by_parameter(
        (
            ("axes", axes_value),
            (("starts", "ends", "strides"), list_value),
        ),
        default="default",
    )


@rules.register("paddle.tensordot")
def tensordot_values(ctx: InputGenerationContext, case: RuleCase):
    state = {"shape1": None, "shape2": None, "tensor1": None}

    def x_value(binding):
        if state["shape1"] is None:
            state["shape1"] = binding.spec.shape
        return case.value_domain("default", binding)

    def y_value(binding):
        if state["shape2"] is None:
            state["shape2"] = binding.spec.shape
        return case.value_domain("default", binding)

    def axes_value(binding):
        axes_arg = case.arg(2, "axes")
        rank = len(state["shape1"])
        if isinstance(axes_arg, (list, tuple)):
            if state["tensor1"] is None:
                result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
                used = []
                for index in range(math.prod(binding.spec.shape)):
                    result[index] = case.randint(0, rank)
                    while (
                        state["shape1"][result[index]] not in state["shape2"]
                        or result[index] in used
                    ):
                        result[index] = case.randint(0, rank)
                    used.append(result[index])
                state["tensor1"] = result
                return result
            result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
            used = []
            for index in range(math.prod(binding.spec.shape)):
                result[index] = case.randint(0, rank)
                while (
                    state["shape2"][result[index]] != state["shape1"][state["tensor1"][index]]
                    or result[index] in used
                ):
                    result[index] = case.randint(0, rank)
                used.append(result[index])
            return result
        if binding.spec.shape == () or math.prod(binding.spec.shape) == 1:
            case.randint(0, 2, size=binding.spec.shape).astype(binding.spec.dtype)
            case.randint(0, rank, size=binding.spec.shape).astype(binding.spec.dtype)
            candidates = [
                index
                for index in range(min(len(state["shape1"]), len(state["shape2"])))
                if state["shape1"][index] == state["shape2"][index]
            ]
            if not candidates:
                raise ValueError(
                    f"No valid axis found for tensordot,x shape {state['shape1']}, "
                    f"y shape {state['shape2']},axes {axes_arg}"
                )
            return case.array([case.python_choice(candidates)], dtype=binding.spec.dtype)
        result = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        used1 = []
        used2 = []
        for index in range(binding.spec.shape[0]):
            result[0][index] = case.randint(0, rank)
            result[1][index] = case.randint(0, rank)
            while (
                state["shape1"][result[0][index]] != state["shape2"][result[1][index]]
                or result[0][index] in used1
                or result[1][index] in used2
            ):
                result[0][index] = case.randint(0, rank)
                result[1][index] = case.randint(0, rank)
            used1.append(result[0][index])
            used2.append(result[1][index])
        return result

    case.generate_by_parameter(
        (
            ("x", x_value),
            ("y", y_value),
            ("axes", axes_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.embedding")
def embedding_values(ctx: InputGenerationContext, case: RuleCase):
    def ids_value(binding):
        weight_config = case.arg(1, "weight")
        vocab_size = case.randint(10, 1000)
        if weight_config is not None and weight_config.shape:
            vocab_size = weight_config.shape[0]
        if vocab_size == 0:
            return case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return case.randint(0, vocab_size, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("x", ids_value),
            ("ids", ids_value),
            ("weight", "multiply"),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.affine_grid")
def affine_grid_values(ctx: InputGenerationContext, case: RuleCase):
    def out_shape_value(binding):
        theta_shape = case.arg(0, "theta").shape
        values = case.randint(1, 128, size=binding.spec.shape).astype(binding.spec.dtype)
        values[0] = theta_shape[0]
        return values

    case.generate_by_parameter((("out_shape", out_shape_value),), default="default")


@rules.register("paddle.nn.functional.hsigmoid_loss")
def hsigmoid_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        num_classes = case.arg(2, "num_classes")
        return case.randint(0, num_classes, size=binding.spec.shape).astype(binding.spec.dtype)

    def path_table_value(binding):
        weight = case.arg(3, "weight")
        return case.randint(0, weight.shape[0], size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("label", label_value),
            ("path_table", path_table_value),
            ("path_code", "binary_0_1"),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.margin_cross_entropy")
def margin_cross_entropy_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        logits = case.arg(0, "logits")
        return case.randint(0, logits.shape[1], size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.nn.functional.multi_margin_loss")
def multi_margin_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        logits = case.arg(0, "input")
        return case.randint(0, logits.shape[1], size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.nn.functional.dice_loss")
def dice_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        tensor = case.arg(0, "input")
        return case.randint(0, tensor.shape[-1], size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.nn.functional.nll_loss")
def nll_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        input_config = case.arg(0, "input")
        n_classes = case.randint(5, 50) if input_config is None else input_config.shape[1]
        return case.randint(0, n_classes, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.nn.functional.adaptive_log_softmax_with_loss")
def adaptive_log_softmax_with_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        cutoffs = case.arg(4, "cutoffs")
        n_classes = cutoffs[-1]
        generation_size = binding.spec.shape
        if len(binding.spec.shape) == 0:
            generation_size = 1
        if n_classes == 1:
            return case.zeros(generation_size, dtype=binding.spec.dtype)
        return case.randint(0, n_classes, size=generation_size, dtype=binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.nn.functional.cross_entropy")
def cross_entropy_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        input_shape = case.arg(0, "input").shape
        axis = case.arg(7, "axis", -1)
        num_classes = input_shape[axis]
        soft_label = case.arg(5, "soft_label", False)
        label_smoothing = case.arg(6, "label_smoothing", 0.0)
        if (label_smoothing > 0 and list(binding.spec.shape) == list(input_shape)) or (
            label_smoothing == 0 and soft_label
        ):
            soft_labels = case.random(binding.spec.shape)
            soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
            return soft_labels.astype(binding.spec.dtype)
        if num_classes == 0:
            return case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return case.randint(0, num_classes, size=binding.spec.shape).astype(binding.spec.dtype)

    def weight_value(binding):
        values = case.random(binding.spec.shape)
        return values / values.sum()

    case.generate_by_parameter(
        (
            ("label", label_value),
            ("weight", weight_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.ctc_loss")
def ctc_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def labels_value(binding):
        num_classes = case.arg(0, "log_probs").shape[2] - 1
        blank = case.arg(4, "blank", 0)
        valid_label_indices = [index for index in range(num_classes + 1) if index != blank]
        if not valid_label_indices:
            return case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return case.choice(valid_label_indices, size=binding.spec.shape, replace=True).astype(
            binding.spec.dtype
        )

    def input_lengths_value(binding):
        max_logit_length = case.arg(0, "log_probs").shape[0]
        return case.randint(
            1,
            max_logit_length + 1,
            size=binding.spec.shape,
            dtype=binding.spec.dtype,
        )

    def label_lengths_value(binding):
        max_label_length = case.arg(1, "labels").shape[1]
        max_logit_length = case.arg(0, "log_probs").shape[0]
        cand_label_lengths = case.randint(
            1,
            max_label_length + 1,
            size=binding.spec.shape,
            dtype=binding.spec.dtype,
        )
        compatible_input_lengths = case.randint(
            1,
            max_logit_length + 1,
            size=binding.spec.shape,
            dtype=binding.spec.dtype,
        )
        final_label_lengths = case.minimum(cand_label_lengths, compatible_input_lengths)
        return case.maximum(final_label_lengths, 1)

    case.generate_by_parameter(
        (
            ("labels", labels_value),
            ("input_lengths", input_lengths_value),
            ("label_lengths", label_lengths_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.sequence_mask")
def sequence_mask_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        maxlen_config = case.arg(1, "maxlen")
        provided_maxlen = None
        if isinstance(maxlen_config, int):
            provided_maxlen = max(1, maxlen_config)
        if provided_maxlen is not None:
            return case.randint(0, provided_maxlen + 1, size=binding.spec.shape).astype(
                binding.spec.dtype
            )
        high_value = case.randint(1, 2048)
        values = case.randint(0, high_value, size=binding.spec.shape).astype(binding.spec.dtype)
        if values.size > 0 and values.max() == 0:
            fix_value = case.randint(1, max(2, high_value))
            values.flat[0] = fix_value
        return values

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.nn.functional.softmax_with_cross_entropy")
def softmax_with_cross_entropy_values(ctx: InputGenerationContext, case: RuleCase):
    def label_value(binding):
        logits = case.arg(0, "logits")
        if not hasattr(logits, "shape"):
            logits = case.kwarg("logits")
        num_classes = 10
        if logits is not None:
            axis = case.kwarg("axis", -1)
            axis = axis if axis >= 0 else len(logits.shape) + axis
            if 0 <= axis < len(logits.shape):
                num_classes = logits.shape[axis]
        else:
            num_classes = case.randint(5, 20)
        return case.randint(0, num_classes, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("label", label_value),), default="default")


@rules.register("paddle.linalg.cholesky")
def cholesky_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        if len(binding.spec.shape) < 2 or binding.spec.shape[-1] != binding.spec.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = binding.spec.shape[:-2]
        matrix_dim = binding.spec.shape[-1]
        matrix = case.random([*batch_dims, matrix_dim, matrix_dim], dtype=binding.spec.dtype)
        if len(batch_dims) > 0:
            tensor = case.einsum("...ij,...kj->...ik", matrix, matrix)
        else:
            tensor = case.dot(matrix, matrix.T)
        tensor += case.eye(matrix_dim, dtype=binding.spec.dtype) * 10000
        print("cholesky tensor", tensor)
        return tensor

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.linalg.cov")
def covariance_values(ctx: InputGenerationContext, case: RuleCase):
    def observation_count():
        x_shape = case.arg(0, "x").shape
        rowvar = case.arg(1, "rowvar")
        if rowvar is None:
            rowvar = True
        return (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]

    def x_value(binding):
        if len(binding.spec.shape) < 1 or len(binding.spec.shape) > 2:
            raise ValueError("Shape must have 1 or 2 dimensions for covariance input")
        tensor = case.random(binding.spec.shape, dtype=binding.spec.dtype)
        tensor += case.random(binding.spec.shape, dtype=binding.spec.dtype) * 1e-6
        return tensor

    def fweights_value(binding):
        return case.randint(1, 11, size=(observation_count(),)).astype(binding.spec.dtype)

    def aweights_value(binding):
        if binding.spec.dtype in ["float32", "float64"]:
            return case.uniform(0.1, 1.0, size=(observation_count(),), dtype=binding.spec.dtype)
        return case.randint(1, 11, size=(observation_count(),)).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("x", x_value),
            ("fweights", fweights_value),
            ("aweights", aweights_value),
        ),
        default="default",
    )


@rules.register("paddle.linalg.eigh", "paddle.linalg.eigvalsh")
def eigen_symmetric_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        if len(binding.spec.shape) < 2 or binding.spec.shape[-1] != binding.spec.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = binding.spec.shape[:-2]
        matrix_dim = binding.spec.shape[-1]
        matrix = case.random([*batch_dims, matrix_dim, matrix_dim], dtype=binding.spec.dtype)
        if binding.spec.dtype in ["complex64", "complex128"]:
            matrix = matrix + 1j * case.random(
                [*batch_dims, matrix_dim, matrix_dim],
                dtype=binding.spec.dtype,
            )
            tensor = matrix + case.swapaxes(matrix, -1, -2).conj()
        elif len(batch_dims) > 0:
            tensor = case.einsum("...ij,...kj->...ik", matrix, matrix)
        else:
            tensor = case.dot(matrix, matrix.T)
        tensor += case.eye(matrix_dim, dtype=binding.spec.dtype) * 1e-6
        return tensor

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.linalg.lstsq")
def lstsq_values(ctx: InputGenerationContext, case: RuleCase):
    def matrix_value(binding):
        if len(binding.spec.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for lstsq x")
        batch_dims = binding.spec.shape[:-2]
        rows, cols = binding.spec.shape[-2], binding.spec.shape[-1]
        return case.random([*batch_dims, rows, cols], dtype=binding.spec.dtype)

    case.generate_by_parameter(((("x", "y"), matrix_value),), default="default")


@rules.register("paddle.linalg.lu_unpack")
def lu_unpack_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        if len(binding.spec.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for LU matrix")
        tensor = case.random(binding.spec.shape, dtype=binding.spec.dtype)
        diagonal_size = min(binding.spec.shape[-2], binding.spec.shape[-1])
        tensor[..., range(diagonal_size), range(diagonal_size)] += 1e-6
        return tensor

    def pivot_value(binding):
        row_count = case.arg(0, "x").shape[-2]
        return case.randint(1, row_count + 1, size=binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter(
        (
            ("x", x_value),
            (("pivot", "y"), pivot_value),
        ),
        default="default",
    )


@rules.register("paddle.linalg.cond")
def condition_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        matrix_size = binding.spec.shape[-1]
        tensor = case.random(binding.spec.shape, dtype=binding.spec.dtype)
        tensor += matrix_size * case.eye(matrix_size, dtype=binding.spec.dtype)
        return tensor

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.linalg.det", "paddle.linalg.slogdet")
def determinant_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        if len(binding.spec.shape) < 2:
            raise AssertionError("Input must be at least 2D.")
        if binding.spec.shape[-1] != binding.spec.shape[-2]:
            raise AssertionError("Input must be square matrices.")
        matrix_size = binding.spec.shape[-1]
        is_complex = binding.spec.dtype.startswith("complex")
        if is_complex:
            real_dtype = "float32" if binding.spec.dtype == "complex64" else "float64"
            real = case.uniform(0.5, 1.0, size=binding.spec.shape, dtype=real_dtype)
            imag = case.uniform(0.5, 1.0, size=binding.spec.shape, dtype=real_dtype)
            matrix = (real + 1j * imag).astype(binding.spec.dtype)
            matrix_h = case.conj(matrix).swapaxes(-1, -2)
        else:
            matrix = case.uniform(0.5, 1.0, size=binding.spec.shape, dtype=binding.spec.dtype)
            matrix_h = case.swapaxes(matrix, -1, -2)
        return case.matmul(matrix, matrix_h) + case.eye(matrix_size, dtype=binding.spec.dtype)

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.linalg.pca_lowrank")
def pca_lowrank_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        return case.randn(*binding.spec.shape).astype(binding.spec.dtype)

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.linalg.corrcoef")
def corrcoef_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        if binding.spec.dtype == "float16":
            return case.randn(*binding.spec.shape).astype(binding.spec.dtype) * 1e-3
        return case.value_domain("default", binding)

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register(
    "paddle.linalg.matrix_norm",
    "paddle.linalg.matrix_rank",
    "paddle.linalg.lu",
    "paddle.linalg.multi_dot",
    "paddle.linalg.norm",
    "paddle.linalg.matrix_transpose",
    "paddle.linalg.matrix_power",
    "paddle.linalg.svd",
    "paddle.linalg.svdvals",
    "paddle.linalg.eig",
    "paddle.linalg.eigvals",
    "paddle.linalg.svd_lowrank",
    "paddle.linalg.solve",
    "paddle.linalg.triangular_solve",
    "paddle.linalg.inv",
    "paddle.linalg.qr",
    "paddle.linalg.vector_norm",
)
def linalg_default_values(ctx: InputGenerationContext, case: RuleCase):
    case.generate_all("default")


@rules.register("paddle.linalg.pinv")
def pinv_values(ctx: InputGenerationContext, case: RuleCase):
    hermitian = bool(case.arg(2, " hermitian", False))
    if not hermitian:
        case.generate_all("default")
        return

    def x_value(binding):
        if len(binding.spec.shape) not in [2, 3]:
            raise ValueError("pinv only supports 2D or 3D tensors")
        if binding.spec.dtype.startswith("complex"):
            real_dtype = "float32" if binding.spec.dtype == "complex64" else "float64"
            real = case.randn(*binding.spec.shape).astype(real_dtype)
            imag = case.randn(*binding.spec.shape).astype(real_dtype)
            matrix = (real + 1j * imag).astype(binding.spec.dtype)
        else:
            matrix = case.randn(*binding.spec.shape).astype(binding.spec.dtype)
        if len(binding.spec.shape) == 2:
            matrix_t = matrix.conj().T if binding.spec.dtype.startswith("complex") else matrix.T
        else:
            matrix_t = (
                case.conj(matrix).swapaxes(-2, -1)
                if binding.spec.dtype.startswith("complex")
                else case.swapaxes(matrix, -2, -1)
            )
        return (matrix + matrix_t) / 2

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register("paddle.linalg.cholesky_solve", aliases=("paddle.Tensor.cholesky_solve",))
def cholesky_solve_values(ctx: InputGenerationContext, case: RuleCase):
    if case.api_name == "paddle.linalg.cholesky_solve":
        case.generate_all("default")
        return

    def y_value(binding):
        value = case.value_domain("random_range", binding)
        if case.arg(2, "upper"):
            return case.triu(value)
        return case.tril(value)

    case.generate_by_parameter((("y", y_value),), default="default")


@rules.register("paddle.view", aliases=("paddle.Tensor.view",))
def view_values(ctx: InputGenerationContext, case: RuleCase):
    def x_value(binding):
        if binding.spec.dtype == "uint8":
            target = str(case.arg(1, "shape_or_dtype", ""))
            nbytes = math.prod(binding.spec.shape)
            itemsize = {
                "paddle.bfloat16": 2,
                "paddle.float16": 2,
                "paddle.float32": 4,
                "paddle.float64": 8,
            }.get(target)
            if itemsize is not None and nbytes % itemsize == 0:
                numel = nbytes // itemsize
                if target == "paddle.bfloat16":
                    finite_f32 = ((case.random(numel) - 0.5) * 1.2).astype("float32")
                    return (finite_f32.view("uint32") >> 16).astype("uint16").view("uint8")
                finite = ((case.random(numel) - 0.5) * 1.2).astype(target.replace("paddle.", ""))
                return case.ascontiguousarray(finite).view("uint8")
        return case.value_domain("default", binding)

    case.generate_by_parameter((("x", x_value),), default="default")


@rules.register(
    "paddle.pow",
    aliases=("paddle.Tensor.pow", "paddle.Tensor.__rpow__", "paddle.Tensor.__pow__"),
)
def pow_values(ctx: InputGenerationContext, case: RuleCase):
    def get_base_max(value, dtype_max, default_max=5):
        value_max = default_max
        if value <= 0:
            return value_max
        if value < 1:
            value = 1 / value
        ln_value = math.log(value)
        output_max = dtype_max / max(1, ln_value)
        value_max = math.log(output_max) / ln_value
        if isinstance(value, int):
            value_max = math.floor(value_max)
        return value_max

    def get_exponent_max(value, dtype_max, default_max=5):
        value_max = default_max
        if isinstance(value, numbers.Number):
            if value <= 2:
                return value_max
            value_max = math.pow(dtype_max / value, 1 / value)
            if isinstance(value, int):
                value_max = math.floor(value_max)
        return value_max

    def value(binding):
        api_name = case.api_name
        dtype = binding.spec.dtype
        if api_name == "paddle.Tensor.__rpow__":
            is_base_arg = binding.parameter_name in {"other", "y"} or str(binding.path) == "args[1]"
            if is_base_arg:
                const = case.arg(0, "self")
                get_max = get_base_max
                default_max = 10
            else:
                const = case.arg(1, "other")
                get_max = get_exponent_max
                default_max = 5
        else:
            is_base_arg = binding.parameter_name in {"self", "x"}
            if is_base_arg:
                const = case.arg(1, "other", case.arg(1, "y"))
                get_max = get_base_max
                default_max = 10
            else:
                const = case.arg(0, "self", case.arg(0, "x"))
                get_max = get_exponent_max
                default_max = 5
        if isinstance(const, numbers.Number):
            value_max = get_max(const, case.dtype_max(dtype), default_max)
            if is_base_arg and int(const) != const:
                return case.value_domain("random_range", binding, low=0, high=value_max)
            return case.value_domain("random_range", binding, low=-value_max, high=value_max)
        if is_base_arg:
            return case.value_domain("random_range", binding, low=0, high=default_max)
        return case.value_domain("random_range", binding, low=-default_max, high=default_max)

    case.generate_all(value)


@rules.register("paddle.nn.functional.rnnt_loss")
def rnnt_loss_values(ctx: InputGenerationContext, case: RuleCase):
    def logits(binding):
        shape = binding.spec.shape if len(binding.spec.shape) == 4 else (3, 4, 3, 5)
        return case.random(shape, dtype=binding.spec.dtype)

    def labels(binding):
        shape = binding.spec.shape if len(binding.spec.shape) == 2 else (3, 2)
        return case.randint(1, 4, size=shape).astype(binding.spec.dtype)

    def lengths(max_possible_length):
        def generate(binding):
            shape = binding.spec.shape if len(binding.spec.shape) == 1 else (3,)
            return case.ones(shape, dtype=binding.spec.dtype) * max_possible_length

        return generate

    case.generate_by_parameter(
        (
            (("input", "logits"), logits),
            (("label", "labels"), labels),
            ("input_lengths", lengths(4)),
            ("label_lengths", lengths(2)),
        ),
        default="default",
    )


@rules.register("paddle.chunk")
def chunk_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_tensor = case.arg(0, "x")
        chunks = case.arg(1, "chunks")
        valid_axes = [
            index for index, dim_size in enumerate(x_tensor.shape) if dim_size % chunks == 0
        ]
        if not valid_axes:
            raise ValueError(
                f"No valid axis found in x.shape = {x_tensor.shape} for chunks = {chunks}. "
                f"Each dim must be divisible by chunks."
            )
        chosen_axis = random.choice(valid_axes)
        if len(binding.spec.shape) == 0:
            return case.array(chosen_axis, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1 and binding.spec.shape[0] == 1:
            return case.array([chosen_axis], dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register("paddle.split")
def split_values(ctx: InputGenerationContext, case: RuleCase):
    def axis_value(binding):
        x_shape = case.arg(0, "x").shape
        num_or_sections = case.arg(1, "num_or_sections")
        if isinstance(num_or_sections, (list, tuple)):
            neg_one_count = sum(1 for item in num_or_sections if item == -1)
            if neg_one_count > 1:
                raise ValueError(
                    f"num_or_sections can contain at most one -1, but got {num_or_sections}"
                )
            num_splits = len(num_or_sections)
            known_size = sum(num_or_sections) + neg_one_count
        elif isinstance(num_or_sections, int):
            num_splits = num_or_sections
            known_size = None
        else:
            raise ValueError(
                f"num_or_sections must be an int, list, or tuple, but got {type(num_or_sections)}"
            )

        target_dim = None
        if len(x_shape) == 0:
            target_dim = case.randint(-1, 0)
        else:
            for dim, dim_size in enumerate(x_shape):
                if isinstance(num_or_sections, int) and dim_size % num_splits == 0:
                    target_dim = dim
                elif isinstance(num_or_sections, (list, tuple)):
                    if (neg_one_count == 0 and dim_size == known_size) or (
                        neg_one_count == 1 and dim_size > known_size
                    ):
                        target_dim = dim
        if target_dim is None:
            raise ValueError(
                f"No valid axis found for paddle.split with x.shape={x_shape} "
                f"and num_or_sections={num_or_sections}"
            )
        if len(binding.spec.shape) == 0:
            return case.array(target_dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1 and binding.spec.shape[0] == 1:
            return case.array([target_dim], dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.split. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    case.generate_by_parameter((("axis", axis_value),), default="default")


@rules.register("paddle.expand", aliases=("paddle.Tensor.expand",))
def expand_values(ctx: InputGenerationContext, case: RuleCase):
    def shape_value(binding):
        x_shape = case.arg(0, "x").shape
        shape_index = binding.path.indices[0] if binding.path.indices else 0
        if len(x_shape) == 0 or shape_index > len(x_shape) - 1 or x_shape[shape_index] == 1:
            return case.randint(1, 127, size=binding.spec.shape).astype(binding.spec.dtype)
        if len(binding.spec.shape) == 0 or binding.spec.shape[0] == 1:
            return case.array(x_shape[shape_index])
        values = case.randint(1, 127, size=binding.spec.shape).astype(binding.spec.dtype)
        offset = binding.spec.shape[0] - len(x_shape)
        for index in range(binding.spec.shape[0]):
            if index >= offset and x_shape[index - offset] != 1:
                values[index] = x_shape[index - offset]
        return values

    case.generate_by_parameter((("shape", shape_value),), default="default")


@rules.register("paddle.nn.functional.gather_tree")
def gather_tree_values(ctx: InputGenerationContext, case: RuleCase):
    def parents_value(binding):
        sequences = case.arg(0, "sequences")
        if hasattr(sequences, "shape") and len(sequences.shape) >= 3:
            beam_size = sequences.shape[2]
        else:
            beam_size = binding.spec.shape[2] if len(binding.spec.shape) >= 3 else 4
        beam_size = 1 if beam_size < 1 else beam_size
        parents = case.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for time_index in range(binding.spec.shape[0]):
            for batch_index in range(binding.spec.shape[1]):
                for beam_index in range(binding.spec.shape[2]):
                    parents[time_index, batch_index, beam_index] = case.randint(0, beam_size)
        return parents

    case.generate_by_parameter((("parents", parents_value),), default="default")


@rules.register("paddle.multinomial")
def multinomial_values(ctx: InputGenerationContext, case: RuleCase):
    x_binding = case.find("x")
    num_samples_binding = case.find("num_samples")
    if x_binding is not None:
        values = case.abs(case.random(x_binding.spec.shape)).astype(x_binding.spec.dtype)
        case.set_value(x_binding, values)
    if num_samples_binding is not None:
        replacement = case.arg(2, "replacement")
        if case.has_kwarg("replacement") and replacement is True:
            max_allow = 1024
        else:
            inputs = case.value(x_binding)
            max_allow = (inputs > 0).sum().item()
        case.set_value(
            num_samples_binding,
            case.randint(
                1,
                max_allow + 1,
                size=num_samples_binding.spec.shape,
            ).astype(num_samples_binding.spec.dtype),
        )
    case.generate_remaining("default")


@rules.register("paddle.nn.functional.one_hot")
def one_hot_values(ctx: InputGenerationContext, case: RuleCase):
    x_binding = case.find("x")
    num_classes_binding = case.find("num_classes")
    num_classes_config = case.arg(1, "num_classes")
    default_random_num_classes = case.randint(1, 65535)
    if isinstance(num_classes_config, int):
        determined_num_classes = num_classes_config
    elif case.is_tensor_config(num_classes_config):
        if num_classes_binding is not None and num_classes_config.numel() in {0, 1}:
            case.set_value(
                num_classes_binding,
                case.array([default_random_num_classes], dtype="int64"),
            )
        determined_num_classes = case.value(num_classes_binding).item()
    else:
        determined_num_classes = default_random_num_classes
    if x_binding is not None:
        case.set_value(
            x_binding,
            case.randint(
                0,
                determined_num_classes,
                size=x_binding.spec.shape,
                dtype=x_binding.spec.dtype,
            ),
        )
    case.generate_remaining("default")


INPUT_GENERATION_RULES = rules.rules
API_RULE_REGISTRY = rules.by_api


def build_registry(rule_records=INPUT_GENERATION_RULES) -> dict[str, RegisteredRule]:
    registry = {}
    for rule in rule_records:
        for api_name in rule.api_names:
            if api_name in registry:
                raise ValueError(f"input-generation API overlap: {api_name}")
            registry[api_name] = rule
    return registry


def get_rule(api_name: str) -> RegisteredRule | None:
    return API_RULE_REGISTRY.get(api_name)


def _tensor_config_at(case, path):
    value = case.args[path.key] if path.root == "args" else case.kwargs[path.key]
    for index in path.indices:
        value = value[index]
    return value


def _api_arg(case, position, name, default=None):
    if 0 <= position < len(case.args):
        return case.args[position]
    return case.kwargs.get(name, default)


def _apply_value(case, path, value):
    _apply_value_raw(case, path, value, update_metadata=True)


def _apply_value_raw(case, path, value, update_metadata):
    config = _tensor_config_at(case, path)
    if path.root == "args":
        config.index = path.key
    else:
        config.key = path.key
    if path.indices:
        config.list_index = list(path.indices)
    array = numpy.array(value, copy=True, order="K")
    config.numpy_tensor = array
    if update_metadata:
        config.dtype = str(array.dtype)
        config.shape = list(array.shape)
