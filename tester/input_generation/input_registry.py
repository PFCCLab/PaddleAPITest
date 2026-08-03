"""输入生成规则的装饰器注册中心。"""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy

from .input_backend import create_input_backend
from .input_binding import InputContext
from .input_data import InputData, attach_values
from .input_data import input_value as read_input_value
from .tensor_config import CAST_THROUGH_INTERMEDIATE_DTYPES, TensorConfig, not_zero_apis
from .tensor_spec import TensorSpec
from .value_gen import (
    create_config_rng,
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

# Core protocols and value-domain descriptors.
ValueGenerator = Callable[..., object]
ConfigValueGenerator = Callable[[object], object]
RuleFunction = Callable[["ConfigView", "ValueFactory", "InputWriter"], None]
BlockerFunction = Callable[[InputContext], str | None]


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


# Rule execution and the three author-facing roles.
@dataclass(frozen=True)
class RegisteredRule:
    """一条通过装饰器注册的输入生成规则。

    这里保存的是规则的元信息和执行入口，不保存 API 约束逻辑本身。
    规则函数负责描述参数关系，`RegisteredRule` 负责门控、完整性检查和提交。
    """

    api_names: tuple[str, ...]
    function: RuleFunction
    block_key: str
    blocker: BlockerFunction | None = None
    allow_gpu: bool = True
    allow_cached: bool = True

    def block_reason(self, context: InputContext) -> str | None:
        if context.gpu_enabled and not self.allow_gpu:
            return f"{self.block_key}-gpu-blocked"

        use_cached_numpy = os.getenv("USE_CACHED_NUMPY", "False").lower() in {
            "true",
            "1",
            "yes",
            "y",
        }
        if use_cached_numpy and not self.allow_cached:
            return f"{self.block_key}-cache-blocked"
        if self.blocker is not None:
            return self.blocker(context)
        return None

    def generate(self, context: InputContext, api_config: object) -> bool:
        rng = create_config_rng(context)
        backend = create_input_backend(rng)
        config = ConfigView(context.call, api_config)
        values = ValueFactory(backend)
        writer = InputWriter(api_config, backend)
        self.function(config, values, writer)
        attach_values(api_config, writer.finish(config))
        backend.commit()
        return True


class ConfigView:
    """规则侧只读 config 视图。"""

    def __init__(self, call, raw_config: object):
        self._call = call
        self._raw_config = raw_config

    def bindings(self, parameter_name: str):
        return tuple(
            binding for binding in self._call.tensors if binding.parameter_name == parameter_name
        )

    def binding(self, parameter_name: str):
        matches = self.bindings(parameter_name)
        if len(matches) > 1:
            raise ValueError(
                f"rule {self.api_name} found multiple tensors for parameter "
                f"{parameter_name!r}: {[str(binding.path) for binding in matches]}"
            )
        return matches[0] if matches else None

    def binding_for_value(self, value):
        if not self.is_tensor_config(value):
            return None
        for binding in self._call.tensors:
            if _tensor_config_at(self._raw_config, binding.path) is value:
                return binding
        return None

    @property
    def api_name(self):
        return self._call.api_name

    @property
    def tensors(self):
        return self._call.tensors

    def arg(self, position, name, default=None):
        return _api_arg(self._raw_config, position, name, default)

    def has_kwarg(self, name):
        return name in self._raw_config.kwargs

    def kwarg(self, name, default=None):
        return self._raw_config.kwargs.get(name, default)

    def argument_values(self):
        return (*self._raw_config.args, *self._raw_config.kwargs.values())

    def is_tensor_config(self, value):
        return isinstance(value, TensorConfig)


class ValueFactory:
    """规则侧数值生成入口。"""

    def __init__(self, backend):
        self.backend = backend

    def domain(self, generator: str, binding, low=None, high=None):
        generate_value = _VALUE_GENERATORS.get(generator)
        if generate_value is None:
            raise ValueError(f"unknown input value generator {generator!r} for {binding.path}")
        return generate_value(binding.spec, low, high, self.backend)

    def dtype_eps(self, dtype):
        dtype = self._numeric_dtype(dtype)
        if dtype == "bool" or "int" in dtype:
            return 0
        return numpy.finfo(self._real_dtype(dtype)).eps

    def dtype_max(self, dtype):
        dtype = self._numeric_dtype(dtype)
        if dtype == "bool":
            return 1
        if "int" in dtype:
            return numpy.iinfo(dtype).max
        return numpy.finfo(self._real_dtype(dtype)).max

    def dtype_min(self, dtype):
        dtype = self._numeric_dtype(dtype)
        if dtype == "bool":
            return 0
        if "int" in dtype:
            return numpy.iinfo(dtype).min
        return numpy.finfo(self._real_dtype(dtype)).min

    def _numeric_dtype(self, dtype):
        return self.backend.generation_dtype(str(dtype).replace("paddle.", ""))

    def _real_dtype(self, dtype):
        if dtype == "complex64":
            return "float32"
        if dtype == "complex128":
            return "float64"
        return dtype


class InputWriter:
    """规则侧输入写入与完整性状态。"""

    def __init__(self, raw_config: object, backend):
        self._raw_config = raw_config
        self._backend = backend
        self._generated_paths = set()
        self._input_data_by_path: dict[object, InputData] = {}
        self._update_config_by_path: dict[object, bool] = {}

    def finish(self, config: ConfigView):
        missing = [
            str(binding.path)
            for binding in config.tensors
            if binding.path not in self._generated_paths
        ]
        if missing:
            raise ValueError(f"rule {config.api_name} left tensors ungenerated: {missing}")
        for path, data in self._input_data_by_path.items():
            _apply_input_data(
                self._raw_config,
                data,
                update_config=self._update_config_by_path[path],
            )
        return tuple(self._input_data_by_path.values())

    def is_generated(self, binding):
        return binding.path in self._generated_paths

    def set_value(self, binding, value):
        if binding.path in self._generated_paths:
            raise ValueError(f"rule generated tensor twice: {binding.path}")
        self._write_value(binding, value, update_config=True)

    def set_value_preserving_spec(self, binding, value):
        if binding.path in self._generated_paths:
            raise ValueError(f"rule generated tensor twice: {binding.path}")
        self._write_value(binding, value, update_config=False)

    def value(self, binding):
        data = self._input_data_by_path.get(binding.path)
        if data is not None:
            return data.value
        config = _tensor_config_at(self._raw_config, binding.path)
        return read_input_value(self._raw_config, config)

    def _write_value(self, binding, value, update_config):
        storage_value = self._backend.asarray(value, copy=True)
        self._input_data_by_path[binding.path] = InputData(
            binding.path,
            storage_value,
            self._backend.name,
        )
        self._update_config_by_path[binding.path] = update_config
        self._generated_paths.add(binding.path)


# Generic rule-generation helpers.
def generate_all(
    config: ConfigView,
    values: ValueFactory,
    writer: InputWriter,
    generator,
    low=None,
    high=None,
):
    for binding in config.tensors:
        _generate_binding(values, writer, binding, generator, low, high)


def generate_remaining(config: ConfigView, values: ValueFactory, writer: InputWriter, generator):
    for binding in config.tensors:
        if not writer.is_generated(binding):
            _generate_binding(values, writer, binding, generator)


def generate_by_parameter(
    config: ConfigView,
    values: ValueFactory,
    writer: InputWriter,
    parameter_generators,
    default: str | ConfigValueGenerator | None = None,
):
    normalized = []
    for parameter_name, generator in parameter_generators:
        names = (parameter_name,) if isinstance(parameter_name, str) else tuple(parameter_name)
        normalized.append((names, generator))
    for binding in config.tensors:
        generator = None
        for names, candidate in normalized:
            if binding.parameter_name in names:
                generator = candidate
                break
        if generator is None:
            generator = default
        if generator is not None:
            _generate_binding(values, writer, binding, generator)


def _generate_binding(
    values: ValueFactory,
    writer: InputWriter,
    binding,
    generator,
    low=None,
    high=None,
):
    if callable(generator):
        value = generator(binding)
    else:
        value = values.domain(generator, binding, low, high)
    writer.set_value(binding, value)


# Registration infrastructure.
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
        names = (*api_names, *aliases)
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("api_names must contain non-empty strings")
            if name != name.strip():
                raise ValueError(f"api_names entry has surrounding whitespace: {name!r}")
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"duplicate api_names: {duplicates}")
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


# Registered API rules remain together in this file for centralized lookup.
rules = RuleRegistry()


@rules.register(
    "paddle.add",
    "paddle.logical_not",
    "paddle.concat",
    aliases=("paddle.Tensor.__add__", "paddle.Tensor.add", "paddle.Tensor.__radd__"),
)
@rules.register(
    "paddle.subtract",
    aliases=("paddle.Tensor.__sub__", "paddle.Tensor.sub", "paddle.Tensor.__rsub__"),
)
@rules.register("paddle.all", aliases=("paddle.Tensor.all",))
@rules.register("paddle.transpose", aliases=("paddle.Tensor.transpose",))
@rules.register("paddle.cast", aliases=("paddle.Tensor.cast",))
@rules.register("paddle.assign")
@rules.register("paddle.exp")
@rules.register("paddle.nn.functional.sigmoid")
def default_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "default")


@rules.register("paddle.incubate.nn.functional.fused_act_dequant")
def fused_act_dequant_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_scale_value(binding):
        if binding.spec.dtype == "int32":
            exponent = values.backend.randint(120, 128, shape=binding.spec.shape, dtype="int32")
            return exponent * values.backend.asarray(0x01010101, dtype="int32")
        return values.domain("default", binding)

    generate_by_parameter(config, values, writer, (("x_scale", x_scale_value),), default="default")


@rules.register("paddle.incubate.nn.functional.variable_length_memory_efficient_attention")
def variable_length_memory_efficient_attention_values(
    config: ConfigView, values: ValueFactory, writer: InputWriter
):
    def seq_lens_value(binding):
        query = config.arg(0, "query")
        q_seq_len = query.shape[2]
        return values.domain("random_range", binding, 1, q_seq_len)

    def kv_seq_lens_value(binding):
        key = config.arg(1, "key")
        value = config.arg(2, "value")
        max_seq_len = min(key.shape[2], value.shape[2])
        return values.domain("random_range", binding, 1, max_seq_len)

    def mask_value(binding):
        return values.backend.cast(
            values.backend.randint(0, 2, shape=binding.spec.shape),
            binding.spec.dtype,
        ) * values.dtype_min(binding.spec.dtype)

    generate_by_parameter(
        config,
        values,
        writer,
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
def block_multihead_attention_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    qkv = config.arg(0, "qkv")
    seq_lens_encoder = config.arg(3, "seq_lens_encoder")
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
        return values.backend.asarray([seq_len] * batch_size, dtype=binding.spec.dtype)

    def set_padding_offsets(binding):
        seq_lens_this_time = writer.value(config.binding("seq_lens_this_time"))
        cum_offsets_now = values.backend.cumsum(seq_len - seq_lens_this_time)
        cum_offsets_binding = config.binding("cum_offsets")
        cu_seqlens_q_binding = config.binding("cu_seqlens_q")
        cu_seqlens_k_binding = config.binding("cu_seqlens_k")
        cum_offsets = values.backend.zeros((batch_size + 1,), dtype=cum_offsets_binding.spec.dtype)
        cum_offsets[1:] = cum_offsets_now
        token_num = values.backend.sum(seq_lens_this_time)
        padding_offsets = values.backend.zeros((token_num,), dtype=binding.spec.dtype)
        cu_seqlens_q = values.backend.zeros(
            (batch_size + 1,), dtype=cu_seqlens_q_binding.spec.dtype
        )
        cu_seqlens_k = values.backend.zeros(
            (batch_size + 1,), dtype=cu_seqlens_k_binding.spec.dtype
        )
        for batch_index in range(batch_size):
            seq_len_now = int(seq_lens_this_time[batch_index])
            cum_offset = int(cum_offsets[batch_index])
            for token_index in range(seq_len_now):
                padding_offsets[batch_index * seq_len - cum_offset + token_index] = cum_offset
            cum_seq_len = (batch_index + 1) * seq_len - cum_offsets[batch_index + 1]
            cu_seqlens_q[batch_index + 1] = cum_seq_len
            cu_seqlens_k[batch_index + 1] = cum_seq_len
        writer.set_value(cum_offsets_binding, cum_offsets[:-1])
        writer.set_value(cu_seqlens_q_binding, cu_seqlens_q)
        writer.set_value(cu_seqlens_k_binding, cu_seqlens_k)
        writer.set_value(binding, padding_offsets)

    for binding in config.tensors:
        if writer.is_generated(binding):
            continue
        if binding.parameter_name in zero_parameters:
            writer.set_value(
                binding, values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
            )
        elif binding.parameter_name == "seq_lens_encoder":
            writer.set_value(binding, seq_len_array(binding))
        elif binding.parameter_name == "seq_lens_this_time":
            writer.set_value(binding, writer.value(config.binding("seq_lens_encoder")))
        elif binding.parameter_name == "padding_offsets":
            set_padding_offsets(binding)
        elif binding.parameter_name in positive_range_parameters:
            writer.set_value(binding, values.domain("random_range", binding, low=0))
        elif binding.parameter_name == "max_enc_len_this_time":
            writer.set_value(binding, seq_len_array(binding))
        elif binding.parameter_name in {"mask", "tgt_mask"}:
            writer.set_value(
                binding,
                values.domain("random_range", binding, high=values.dtype_eps(binding.spec.dtype)),
            )
        else:
            writer.set_value(binding, values.domain("default", binding))


@rules.register(
    "paddle._C_ops.adam_",
    "paddle._C_ops.adamw_",
    "paddle._C_ops.merged_adam_",
    block_key="optimizer",
)
def optimizer_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    zero_parameters = {"moment1", "moment2", "moment2_max"}
    optimizer_step = None

    def beta_pow_value(binding, beta, step):
        import paddle

        use_accuracy_compatible = paddle.get_flags("FLAGS_use_accuracy_compatible_kernel")[
            "FLAGS_use_accuracy_compatible_kernel"
        ]
        if use_accuracy_compatible:
            beta_pow = beta**step
        else:
            # Preserve the float32 power semantics without passing GPU scalars to NumPy.
            step = step.item() if hasattr(step, "item") else step
            beta_pow = values.backend.power(
                numpy.float32(beta),
                numpy.float32(int(step)),
            )
            beta_pow = beta_pow.item() if hasattr(beta_pow, "item") else beta_pow
        return values.backend.full(
            binding.spec.shape,
            beta_pow,
            dtype=binding.spec.dtype,
        )

    def generate_value(binding):
        nonlocal optimizer_step
        if binding.parameter_name in zero_parameters:
            return values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        if config.api_name == "paddle._C_ops.adamw_" and binding.parameter_name in {
            "beta1_pow",
            "beta2_pow",
        }:
            if optimizer_step is None:
                optimizer_step = values.backend.randint(1, 101)
            beta = config.arg(10, "beta1")
            if binding.parameter_name == "beta2_pow":
                beta = config.arg(11, "beta2")
            return beta_pow_value(binding, beta, optimizer_step)
        return values.domain("default", binding)

    generate_all(config, values, writer, generate_value)


@rules.register(
    "paddle.nn.functional.max_unpool1d",
    "paddle.nn.functional.max_unpool2d",
    "paddle.nn.functional.max_unpool3d",
)
def max_unpool_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def resolve_parameters(x_shape, output_size, kernel_size, stride, padding):
        dimensions = {
            "paddle.nn.functional.max_unpool1d": 1,
            "paddle.nn.functional.max_unpool2d": 2,
            "paddle.nn.functional.max_unpool3d": 3,
        }[config.api_name]
        if isinstance(kernel_size, numbers.Integral):
            kernel_size = [kernel_size] * dimensions
        else:
            kernel_size = list(kernel_size)
        if stride is None:
            stride = list(kernel_size)
        elif isinstance(stride, numbers.Integral):
            stride = [stride] * dimensions
        else:
            stride = list(stride)
        if isinstance(padding, numbers.Integral):
            padding = [padding] * dimensions
        else:
            padding = list(padding)

        if output_size is None:
            spatial_size = [
                (x_shape[-dimensions + index] - 1) * stride[index]
                - 2 * padding[index]
                + kernel_size[index]
                for index in range(dimensions)
            ]
            pool_input_size = [*x_shape[:-dimensions], *spatial_size]
        elif len(output_size) == dimensions:
            pool_input_size = [*x_shape[:-dimensions], *output_size]
        elif len(output_size) == len(x_shape):
            pool_input_size = list(output_size)
        else:
            raise ValueError(
                f"invalid output_size for {config.api_name}, len(output_size) should be "
                f"{dimensions} or {len(x_shape)} or output_size == None, got "
                f"len(output_size)={len(output_size)} and output_size={output_size}"
            )
        return kernel_size, stride, padding, pool_input_size

    x_binding = config.binding("x")
    indices_binding = config.binding("indices")
    if x_binding is None or indices_binding is None:
        raise ValueError(f"rule {config.api_name} requires x and indices tensors")

    kernel_size = config.arg(2, "kernel_size")
    stride = config.arg(3, "stride")
    padding = config.arg(4, "padding")
    output_size = config.arg(5, "output_size")
    kernel_size, stride, padding, pool_input_size = resolve_parameters(
        x_binding.spec.shape,
        output_size,
        kernel_size,
        stride,
        padding,
    )
    data_type = "float64" if x_binding.spec.dtype == "int64" else x_binding.spec.dtype
    pool_input_spec = TensorSpec(
        shape=tuple(pool_input_size),
        dtype=data_type,
        place=x_binding.spec.place,
        is_contiguous=x_binding.spec.is_contiguous,
        strides=x_binding.spec.strides,
    )
    pool_input = generate_random_range(pool_input_spec, low=-5, high=5, rng=values.backend)
    pool_name = config.api_name.rsplit(".", 1)[-1].replace("max_unpool", "max_pool")
    if values.backend.name == "torch":
        import torch.nn.functional as torch_functional

        max_poolxd_func = getattr(torch_functional, pool_name)
        x, indices = max_poolxd_func(
            pool_input,
            kernel_size,
            stride,
            padding,
            return_indices=True,
        )
        writer.set_value(x_binding, x)
        writer.set_value(indices_binding, indices)
        return

    import paddle

    max_poolxd_func = getattr(paddle.nn.functional, pool_name)
    x, indices = max_poolxd_func(
        paddle.to_tensor(pool_input),
        kernel_size,
        stride,
        padding,
        return_mask=True,
    )
    if values.backend.name == "paddle":
        writer.set_value(x_binding, x)
        writer.set_value(indices_binding, indices)
        return
    writer.set_value(x_binding, x.numpy())
    writer.set_value(indices_binding, indices.numpy())


@rules.register("paddle.arange")
def arange_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def tensor_binding(value):
        return config.binding_for_value(value)

    def set_tensor(value, tensor_value):
        binding = tensor_binding(value)
        if binding is not None:
            writer.set_value(binding, tensor_value)

    def generate_step_tensor(step_config, is_positive):
        if "int" in step_config.dtype:
            if is_positive:
                return values.backend.cast(
                    values.backend.randint(1, 10, shape=step_config.shape),
                    step_config.dtype,
                )
            return values.backend.cast(
                values.backend.randint(-10, -1, shape=step_config.shape),
                step_config.dtype,
            )
        if is_positive:
            return values.backend.cast(
                values.backend.uniform(0.1, 5.0, shape=step_config.shape),
                step_config.dtype,
            )
        return values.backend.cast(
            values.backend.uniform(-5.0, -0.1, shape=step_config.shape),
            step_config.dtype,
        )

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
            return values.backend.cast(
                values.backend.randint(low, high, shape=tensor_config.shape),
                tensor_config.dtype,
            )
        return values.backend.cast(
            values.backend.uniform(low, high, shape=tensor_config.shape),
            tensor_config.dtype,
        )

    def handle_arange_relation():
        start_val = config.arg(0, "start", 0)
        end_val = config.arg(1, "end", None)
        step_val = config.arg(2, "step", 1)

        if config.is_tensor_config(start_val):
            if config.is_tensor_config(end_val):
                if config.is_tensor_config(step_val):
                    flag = values.backend.choice([True, False])
                    set_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                set_tensor(start_val, random_range(start_val, -50, 50))
                start = writer.value(config.binding("start")).item()
                if flag:
                    low, high = safe_range(start + 1, start + 50)
                else:
                    low, high = safe_range(start - 50, start - 1)
                set_tensor(end_val, random_range(end_val, low, high))
            elif end_val is None:
                if config.is_tensor_config(step_val):
                    flag = values.backend.choice([True, False])
                    set_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                if flag:
                    if "int" in start_val.dtype:
                        value = values.backend.cast(
                            values.backend.randint(1, 50, shape=start_val.shape),
                            start_val.dtype,
                        )
                    else:
                        value = values.backend.cast(
                            values.backend.uniform(0.1, 50.0, shape=start_val.shape),
                            start_val.dtype,
                        )
                elif "int" in start_val.dtype:
                    value = values.backend.cast(
                        values.backend.randint(-50, -1, shape=start_val.shape),
                        start_val.dtype,
                    )
                else:
                    value = values.backend.cast(
                        values.backend.uniform(-50.0, -0.1, shape=start_val.shape),
                        start_val.dtype,
                    )
                set_tensor(start_val, value)
            else:
                if config.is_tensor_config(step_val):
                    flag = values.backend.choice([True, False])
                    set_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                if flag:
                    low, high = safe_range(end_val - 50, end_val - 1)
                else:
                    low, high = safe_range(end_val + 1, end_val + 50)
                set_tensor(start_val, random_range(start_val, low, high))
        elif config.is_tensor_config(end_val):
            if config.is_tensor_config(step_val):
                flag = values.backend.choice([True, False])
                set_tensor(step_val, generate_step_tensor(step_val, flag))
            else:
                flag = step_val > 0
            if flag:
                low, high = safe_range(start_val + 1, start_val + 50)
            else:
                low, high = safe_range(start_val - 50, start_val - 1)
            set_tensor(end_val, random_range(end_val, low, high))
        elif end_val is None:
            if config.is_tensor_config(step_val):
                flag = start_val > 0
                set_tensor(step_val, generate_step_tensor(step_val, flag))
        elif config.is_tensor_config(step_val):
            flag = start_val < end_val
            set_tensor(step_val, generate_step_tensor(step_val, flag))

    for binding in config.tensors:
        if writer.value(binding) is None:
            handle_arange_relation()


@rules.register("paddle.nn.functional.moe_permute")
def moe_permute_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def expert_routemap_value(binding):
        num_experts = config.arg(4, "num_experts", 32)
        hidden_states = config.arg(0, "hidden_states")
        scale = config.arg(1, "scale")
        expert_prob = config.arg(3, "expert_prob_topk")
        tokens_per_expert = config.arg(5, "tokens_per_expert")
        padding_alignment = config.arg(6, "padding_alignment")
        using_ue8m0_scale = config.arg(8, "using_ue8m0_scale", False)
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
        if not config.is_tensor_config(hidden_states) or (
            len(hidden_states.shape) != 2
            or hidden_states.dtype not in {"bfloat16", "float32", "float8_e4m3fn"}
        ):
            raise ValueError("hidden_states must be a rank-2 bfloat16 or float8_e4m3fn tensor")
        if binding.spec.dtype != "int32":
            raise ValueError("expert_routemap_topk dtype must be int32")
        if not config.is_tensor_config(expert_prob) or (
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
                config.is_tensor_config(scale)
                and tuple(scale.shape) == (seqlen, expected_scale_width)
                and scale.dtype == expected_scale_dtype
            ):
                raise ValueError(
                    "float8 hidden_states requires scale with shape "
                    f"[{seqlen}, {expected_scale_width}] and dtype {expected_scale_dtype}"
                )
        elif scale is not None:
            raise ValueError("scale must be None when hidden_states dtype is bfloat16")
        routemap = values.backend.full((seqlen, topk), -1, dtype="int32")
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
            positions = values.backend.arange(cursor, cursor + count, dtype="int64")
            rows = positions % seqlen
            columns = (positions // seqlen) % topk
            routemap[rows, columns] = values.backend.cast(
                values.backend.full(rows.shape, expert, dtype="int32"), "int32"
            )
            cursor += count
        return routemap

    def expert_prob_value(binding):
        routemap_binding = config.binding("expert_routemap_topk")
        probs = values.backend.zeros(binding.spec.shape, dtype="float32")
        if routemap_binding is not None and writer.value(routemap_binding) is not None:
            mask = writer.value(routemap_binding) >= 0
            raw = values.backend.cast(values.backend.random(binding.spec.shape), "float32") * mask
            row_sums = values.backend.sum(raw, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = raw / row_sums
        else:
            probs = values.backend.cast(values.backend.random(binding.spec.shape), "float32")
            row_sums = values.backend.sum(probs, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums
        return probs

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("expert_routemap_topk", expert_routemap_value),
            ("expert_prob_topk", expert_prob_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.moe_unpermute")
def moe_unpermute_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def expert_routemap_value(binding):
        num_experts = config.arg(5, "num_experts", 32)
        total_zipped_tokens = config.arg(4, "total_zipped_tokens")
        hidden_config = config.arg(0, "hidden_states_unzipped")
        rowmap_config = config.arg(1, "zipped_expertwise_rowmap")
        prob_config = config.arg(3, "token_prob_unzipped")
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if (
            not isinstance(total_zipped_tokens, int)
            or isinstance(total_zipped_tokens, bool)
            or total_zipped_tokens < 0
        ):
            raise ValueError("total_zipped_tokens must be a non-negative integer")
        if not (
            config.is_tensor_config(hidden_config)
            and len(hidden_config.shape) == 2
            and hidden_config.dtype in {"bfloat16", "float32"}
        ):
            raise ValueError("hidden_states_unzipped must be a rank-2 bfloat16 tensor")
        if not (
            config.is_tensor_config(rowmap_config)
            and len(rowmap_config.shape) == 2
            and rowmap_config.dtype == "int32"
            and tuple(rowmap_config.shape) == (total_zipped_tokens, num_experts)
        ):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        if not (
            config.is_tensor_config(prob_config)
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
        routemap = values.backend.full(binding.spec.shape, -1, dtype="int32")
        max_assign = min(topk, num_experts)
        route_count = min(hidden_config.shape[0], seqlen * max_assign)
        positions = values.backend.arange(route_count, dtype="int64")
        rows = positions % seqlen
        columns = positions // seqlen
        routemap[rows, columns] = values.backend.cast((rows + columns) % num_experts, "int32")
        return routemap

    def rowmap_value(binding):
        routemap_binding = config.binding("expert_routemap_topk")
        if routemap_binding is not None and not writer.is_generated(routemap_binding):
            writer.set_value(routemap_binding, expert_routemap_value(routemap_binding))
        routemap_config = config.arg(2, "expert_routemap_topk")
        num_experts = config.arg(5, "num_experts", 32)
        total_zipped_tokens = config.arg(4, "total_zipped_tokens")
        hidden_config = config.arg(0, "hidden_states_unzipped")
        seqlen = total_zipped_tokens
        unzipped_seqlen = (
            hidden_config.shape[0] if config.is_tensor_config(hidden_config) else seqlen
        )
        if binding.spec.dtype != "int32" or tuple(binding.spec.shape) != (seqlen, num_experts):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        rowmap = values.backend.full(binding.spec.shape, -1, dtype="int32")
        if config.is_tensor_config(routemap_config) and routemap_binding is not None:
            routemap = writer.value(routemap_binding)
            expert_counts = values.backend.asarray(
                [values.backend.count_nonzero(routemap == expert) for expert in range(num_experts)],
                dtype="int64",
            )
            if int(values.backend.sum(expert_counts)) > unzipped_seqlen:
                raise ValueError("routemap assignments exceed hidden_states_unzipped capacity")
            expert_offsets = values.backend.zeros(num_experts, dtype="int64")
            expert_offsets[1:] = values.backend.cumsum(expert_counts[:-1])
            expert_counters = values.backend.zeros(num_experts, dtype="int64")
            for row_index in range(seqlen):
                for expert in range(num_experts):
                    positions = values.backend.nonzero(routemap[row_index] == expert)[0]
                    if values.backend.prod(positions.shape) == 0:
                        continue
                    rowmap[row_index, expert] = values.backend.cast(
                        expert_offsets[expert] + expert_counters[expert], "int32"
                    )
                    expert_counters[expert] += 1
        return rowmap

    def token_prob_value(binding):
        hidden_config = config.arg(0, "hidden_states_unzipped")
        if not (
            binding.spec.dtype == "float32"
            and len(binding.spec.shape) in (1, 2)
            and config.is_tensor_config(hidden_config)
            and binding.spec.shape[0] == hidden_config.shape[0]
            and (len(binding.spec.shape) == 1 or binding.spec.shape[1] == 1)
        ):
            raise ValueError(
                "token_prob_unzipped must match the broadcasted sequence "
                "length and have dtype float32"
            )
        return values.backend.cast(values.backend.random(binding.spec.shape), "float32")

    for binding in config.tensors:
        if writer.is_generated(binding):
            continue
        if binding.parameter_name == "expert_routemap_topk":
            writer.set_value(binding, expert_routemap_value(binding))
        elif binding.parameter_name == "zipped_expertwise_rowmap":
            writer.set_value(binding, rowmap_value(binding))
        elif binding.parameter_name == "token_prob_unzipped":
            writer.set_value(binding, token_prob_value(binding))
        else:
            writer.set_value(binding, values.domain("default", binding))


@rules.register(*tuple(sorted(not_zero_apis)))
def nonzero_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "nonzero")


@rules.register("paddle.bernoulli")
def bernoulli_probability(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "unit_interval")


@rules.register("paddle.standard_gamma")
def standard_gamma_unit_interval(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "unit_interval")


@rules.register("paddle.poisson")
def poisson_unit_interval(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "unit_interval")


@rules.register(
    "paddle.sqrt",
    aliases=("paddle.Tensor.sqrt",),
)
def sqrt_nonnegative(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "uniform", low=0, high=1000)


@rules.register(
    "paddle.rsqrt",
    aliases=("paddle.Tensor.rsqrt",),
)
def rsqrt_positive(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "uniform", low=1e-7, high=1000)


@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    x_binding = config.binding("x")
    min_binding = config.binding("min")
    max_binding = config.binding("max")
    min_config = config.arg(1, "min")
    max_config = config.arg(2, "max")

    if config.is_tensor_config(min_config) and config.is_tensor_config(max_config):
        min_value = values.domain("random_range", min_binding)
        max_value = values.domain("random_range", max_binding, low=min_value)
        writer.set_value(min_binding, min_value)
        writer.set_value(max_binding, max_value)
    elif min_config is not None and max_config is not None:
        if config.is_tensor_config(min_config) and isinstance(max_config, (int, float)):
            min_value = values.domain("random_range", min_binding, high=max_config)
            writer.set_value(min_binding, min_value)
        elif config.is_tensor_config(max_config) and isinstance(min_config, (int, float)):
            max_value = values.domain("random_range", max_binding, low=min_config)
            writer.set_value(max_binding, max_value)

    if x_binding is not None:
        writer.set_value(
            x_binding,
            values.domain("random_range", x_binding),
        )
    generate_remaining(config, values, writer, "default")


@rules.register(
    "paddle.multiply",
    aliases=("paddle.Tensor.__mul__", "paddle.Tensor.multiply", "paddle.Tensor.__rmul__"),
)
def multiply_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "multiply")


@rules.register(
    "paddle.nn.functional.binary_cross_entropy",
)
def binary_cross_entropy_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "unit_interval")


@rules.register("paddle.nn.functional.alpha_dropout")
def alpha_dropout_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("x", "unit_interval"),), default="default")


@rules.register("paddle.nn.functional.conv2d_transpose")
def conv2d_transpose_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def tensor_value(binding):
        if "int" in binding.spec.dtype:
            return values.backend.cast(
                values.backend.randint(-65535, 65535, shape=binding.spec.shape),
                binding.spec.dtype,
            )
        return values.backend.cast(
            values.backend.random(binding.spec.shape) - 0.5,
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        ((("x", "weight", "bias"), tensor_value),),
        default="default",
    )


@rules.register("paddle.vision.ops.distribute_fpn_proposals")
def distribute_fpn_proposals_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    state = {"num": None}

    def fpn_rois_value(binding):
        num = binding.spec.shape[0]
        state["num"] = num
        rois = values.backend.randint(1, 1024, shape=[num, 4])
        rois[:, 0] = rois[:, 0] + values.backend.random([num])
        rois[:, 1] = rois[:, 1] + values.backend.random([num])
        rois[:, 2] = (
            rois[:, 0] + values.backend.randint(1, 1024, shape=[num]) + values.backend.random([num])
        )
        rois[:, 3] = (
            rois[:, 1] + values.backend.randint(1, 1024, shape=[num]) + values.backend.random([num])
        )
        return rois

    def rois_num_value(binding):
        if state["num"] is None:
            fpn_rois = config.arg(0, "fpn_rois")
            state["num"] = fpn_rois.shape[0]
        num = state["num"]
        remaining = binding.spec.shape[0]
        result = values.backend.zeros(binding.spec.shape)
        if num > 4096 or remaining > 4096:
            if num < remaining:
                result[:num] = 1
            else:
                result += num // remaining
                result[: num % remaining] += 1
        elif num < remaining:
            indices = values.backend.choice(remaining, num, replace=False)
            result[indices] = 1
        else:
            for index in range(binding.spec.shape[0] - 1):
                result[index] = values.backend.randint(1, num - remaining + 2)
                num -= result[index]
                remaining -= 1
            result[binding.spec.shape[0] - 1] = num
        return result

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("fpn_rois", fpn_rois_value),
            ("rois_num", rois_num_value),
        ),
    )


@rules.register("paddle.vision.ops.generate_proposals")
def generate_proposals_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def random_value(binding):
        return values.backend.random(binding.spec.shape, dtype=binding.spec.dtype)

    def img_size_value(binding):
        return values.backend.cast(
            values.backend.randint(0, 1024, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    def anchors_value(binding):
        anchors = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        width = binding.spec.shape[0]
        height = binding.spec.shape[1]
        for index in range(binding.spec.shape[0]):
            anchors[index][0] = values.backend.random() * width
            anchors[index][1] = values.backend.random() * height
            anchors[index][2] = (
                values.backend.random() * (width - anchors[index][0] + 1) + anchors[index][0] + 1
            )
            anchors[index][3] = (
                values.backend.random() * (height - anchors[index][1] + 1) + anchors[index][1] + 1
            )
        return anchors

    for binding in config.tensors:
        if binding.parameter_name in {"scores", "bbox_deltas"}:
            writer.set_value(binding, random_value(binding))
        elif binding.parameter_name == "img_size":
            writer.set_value(binding, img_size_value(binding))
        elif binding.parameter_name == "anchors":
            writer.set_value(binding, anchors_value(binding))
        else:
            writer.set_value(binding, values.domain("default", binding))


@rules.register("paddle.vision.ops.nms")
def nms_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def boxes_value(binding):
        boxes = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(binding.spec.shape[0]):
            boxes[index][0] = values.backend.random() * 1023
            boxes[index][1] = values.backend.random() * 1023
            boxes[index][2] = (
                values.backend.random() * (1024 - boxes[index][0] + 1) + boxes[index][0] + 1
            )
            boxes[index][3] = (
                values.backend.random() * (1024 - boxes[index][1] + 1) + boxes[index][1] + 1
            )
        return boxes

    def scores_value(binding):
        return values.backend.random(binding.spec.shape, dtype=binding.spec.dtype)

    def default_vision_value(binding):
        return values.backend.cast(
            values.backend.randint(0, 1024, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
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
def roi_pool_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    state = {"x_shape": None, "boxes_shape": None}

    def x_value(binding):
        state["x_shape"] = binding.spec.shape
        return values.backend.cast(
            values.backend.random(binding.spec.shape) * 255,
            binding.spec.dtype,
        )

    def boxes_value(binding):
        if state["x_shape"] is None:
            x = config.arg(0, "x")
            state["x_shape"] = x.shape
        state["boxes_shape"] = binding.spec.shape
        boxes = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(binding.spec.shape[0]):
            boxes[index][0] = values.backend.random() * (state["x_shape"][2] - 2)
            boxes[index][1] = values.backend.random() * (state["x_shape"][3] - 2)
            boxes[index][2] = (
                values.backend.random() * (state["x_shape"][2] - 1 - boxes[index][0] + 1)
                + boxes[index][0]
                + 1
            )
            boxes[index][3] = (
                values.backend.random() * (state["x_shape"][3] - 1 - boxes[index][1] + 1)
                + boxes[index][1]
                + 1
            )
        return boxes

    def boxes_num_value(binding):
        if state["boxes_shape"] is None:
            boxes = config.arg(1, "boxes")
            state["boxes_shape"] = boxes.shape
        boxes_remaining = state["boxes_shape"][0]
        result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        numel = math.prod(binding.spec.shape)
        for index in range(numel - 1):
            if boxes_remaining < numel:
                result[index] = 0
            else:
                result[index] = values.backend.randint(1, boxes_remaining - (numel - 1 - index) + 1)
                boxes_remaining -= result[index]
        result[numel - 1] = boxes_remaining
        return result

    generate_by_parameter(
        config,
        values,
        writer,
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
def zero_65535_or_unit_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "int_zero_65535_else_unit")


@rules.register("paddle.dot")
def dot_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "int_minus127_127_else_default")


@rules.register("paddle.normal")
def normal_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("mean", "signed_half_interval"),
            ("std", "normal_std"),
        ),
        default="int_zero_1024",
    )


@rules.register("paddle.ones")
def ones_shape(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "ones_shape")


@rules.register("paddle.zeros")
def zeros_shape(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "int_zero_2048_no_cast")


@rules.register("paddle.eye")
def eye_shape(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "int_zero_2048_no_cast")


@rules.register(
    "paddle.nn.functional.interpolate",
    "paddle.Tensor.tile",
    "paddle.tile",
)
def shape_parameter_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config,
        values,
        writer,
        ((("size", "scale_factor", "repeat_times"), "int_one_128"),),
        default="default",
    )


@rules.register("paddle.nn.functional.upsample")
def upsample_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("size", "int_one_128"),
            ("scale_factor", "abs_unit_plus_one"),
        ),
        default="default",
    )


@rules.register(
    "paddle.nn.functional.gaussian_nll_loss",
)
def gaussian_nll_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config,
        values,
        writer,
        ((("var", "variance"), "unit_interval_plus_one"),),
        default="default",
    )


@rules.register(
    "paddle.nn.functional.hinge_embedding_loss",
)
def hinge_embedding_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("label", "hinge_labels"),), default="default")


@rules.register(
    "paddle.nn.functional.sigmoid_focal_loss",
)
def sigmoid_focal_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("label", "binary_0_1"),), default="default")


@rules.register("paddle.full")
def full_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("shape", "int_zero_64"),
            ("fill_value", "full_fill_value"),
        ),
        default="int_zero_64",
    )


@rules.register("paddle.standard_normal")
def standard_normal_shape(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("shape", "int_one_128"),), default="default")


@rules.register("paddle.logspace")
def logspace_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config, values, writer, (("num", "int_one_65535_no_cast"),), default="default"
    )


@rules.register("paddle.quantile")
def quantile_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("q", "quantile_q"),), default="default")


@rules.register(
    "paddle.remainder",
    aliases=("paddle.Tensor.remainder",),
)
def remainder_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("y", "remainder_rhs"),), default="default")


@rules.register(
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
)
def dropout_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(
        config, values, writer, (("p", "dropout_probability"),), default="default"
    )


@rules.register("paddle.atan2")
def atan2_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "unit_interval_plus_one")


@rules.register("paddle.bincount")
def bincount_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def integer_value(binding):
        return values.backend.cast(
            values.backend.randint(0, 65535, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", integer_value),
            ("minlength", integer_value),
        ),
        default="default",
    )


@rules.register(
    "paddle.nn.functional.adaptive_avg_pool2d", "paddle.nn.functional.adaptive_avg_pool3d"
)
def adaptive_avg_pool_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def output_size(binding):
        x_shape = config.arg(0, "x").shape
        return values.backend.cast(
            values.backend.randint(1, 2 * max(x_shape), shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (("output_size", output_size),),
        default="default",
    )


@rules.register("paddle.empty")
def empty_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_by_parameter(config, values, writer, (("shape", "empty_shape"),), default="default")


@rules.register(
    "paddle.repeat_interleave",
    aliases=("paddle.Tensor.repeat_interleave",),
)
def repeat_interleave_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x = config.arg(0, "x")
        input_dims = len(x.shape)
        if len(binding.spec.shape) == 0:
            return values.backend.asarray(
                values.backend.randint(-input_dims, input_dims), dtype=binding.spec.dtype
            )
        return values.backend.cast(
            values.backend.randint(-input_dims, input_dims, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
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
def put_along_axis_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def random_tensor_value(binding, shape):
        scalar_spec = TensorSpec(
            shape=tuple(shape),
            dtype=binding.spec.dtype,
            place=binding.spec.place,
            is_contiguous=binding.spec.is_contiguous,
            strides=binding.spec.strides,
        )
        return generate_random_range(scalar_spec, rng=values.backend)

    def indices_value(binding):
        x_tensor = config.arg(0, "arr", config.arg(0, "x"))
        x_shape = tuple(x_tensor.shape) if x_tensor is not None else ()
        x_dims = len(x_shape)
        current_shape = tuple(binding.spec.shape)
        if len(current_shape) != x_dims:
            new_shape = [current_shape[i] if i < len(current_shape) else 1 for i in range(x_dims)]
            indices = values.backend.zeros(new_shape, dtype="int64")
            for axis in range(x_dims):
                if axis < len(current_shape):
                    dim_size = x_shape[axis]
                    if dim_size > 0:
                        axis_indices = values.backend.choice(
                            dim_size, shape=new_shape[axis], replace=False
                        )
                        axis_indices = values.backend.cast(axis_indices, "int64")
                        idx_tuple = tuple(
                            [slice(None)] * axis
                            + [slice(None, new_shape[axis])]
                            + [slice(None)] * (x_dims - axis - 1)
                        )
                        indices[idx_tuple] = values.backend.reshape(
                            axis_indices,
                            [-1] + [1] * (x_dims - axis - 1),
                        )
            return indices
        axis = config.arg(3, "axis", 0)
        axis = axis if isinstance(axis, int) else 0
        axis = axis if axis >= 0 else axis + x_dims
        indices = values.backend.zeros(current_shape, dtype="int64")
        if 0 <= axis < x_dims:
            dim_size = x_shape[axis]
            for idx in values.backend.ndindex(tuple(current_shape[:-1])):
                indices[idx] = values.backend.choice(
                    dim_size, shape=current_shape[-1], replace=False
                )
        return indices

    def write_values(binding):
        indices_binding = config.binding("indices")
        if indices_binding is not None:
            indices = writer.value(indices_binding)
            if tuple(indices.shape) != tuple(binding.spec.shape):
                if values.backend.prod(binding.spec.shape) == 1:
                    writer.set_value_preserving_spec(
                        binding,
                        values.backend.full(
                            indices.shape,
                            random_tensor_value(binding, ())[()],
                            dtype=binding.spec.dtype,
                        ),
                    )
                else:
                    writer.set_value_preserving_spec(
                        binding,
                        random_tensor_value(binding, indices.shape),
                    )
                return
            writer.set_value(binding, random_tensor_value(binding, binding.spec.shape))
            return
        writer.set_value(binding, values.domain("default", binding))

    for binding in config.tensors:
        if binding.parameter_name == "indices":
            writer.set_value(binding, indices_value(binding))
        elif binding.parameter_name == "values":
            write_values(binding)
        else:
            writer.set_value(binding, values.domain("default", binding))


@rules.register("paddle.matrix_transpose")
def matrix_transpose_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        shape = binding.spec.shape if len(binding.spec.shape) >= 2 else (2, 2)
        dtype = binding.spec.dtype
        if "int" in dtype:
            return values.backend.cast(values.backend.randint(-65535, 65535, shape=shape), dtype)
        return values.backend.cast(values.backend.random(shape) - 0.5, dtype)

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.nn.functional.softmax")
def softmax_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_shape = config.arg(0, "x").shape
        return values.domain("uniform", binding, low=-len(x_shape), high=len(x_shape))

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", "random_range"),
            ("axis", axis_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.zeropad2d")
def zeropad2d_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def padding_value(binding):
        return values.domain("uniform", binding, low=0, high=10)

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", "random_range"),
            ("padding", padding_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.pad")
def pad_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def pad_value(binding):
        x_shape = config.arg(0, "x").shape
        return values.domain("uniform", binding, low=0, high=min(x_shape))

    generate_by_parameter(config, values, writer, (("pad", pad_value),), default="default")


@rules.register("paddle.nn.functional.class_center_sample")
def class_center_sample_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def label_value(binding):
        num_classes = config.arg(1, "num_classes")
        return values.backend.cast(
            values.backend.randint(0, num_classes, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.shard_index")
def shard_index_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def input_binding(binding):
        index_num = config.arg(1, "index_num")
        if index_num is None:
            index_num = values.backend.randint(1, 1000)
        return values.backend.cast(
            values.backend.randint(0, index_num, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("input", input_binding),), default="default")


@rules.register("paddle.incubate.nn.functional.masked_multihead_attention")
def masked_multihead_attention_values(
    config: ConfigView, values: ValueFactory, writer: InputWriter
):
    def sequence_lengths(binding):
        return values.domain("random_range", binding, low=1)

    def rotary_tensor(binding):
        return values.domain("uniform", binding, low=0, high=1000)

    generate_by_parameter(
        config,
        values,
        writer,
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
def argminmax_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_shape = config.arg(0, "x").shape
        min_dim = len(x_shape)
        return values.backend.cast(
            values.backend.randint(-min_dim, min_dim - 1, shape=binding.spec.shape),
            "int64",
        )

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register("paddle.cumsum", aliases=("paddle.Tensor.cumsum",))
def cumsum_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_shape = config.arg(0, "x").shape
        return values.backend.randint(-len(x_shape), len(x_shape), shape=binding.spec.shape)

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register(
    "paddle.mean", "paddle.max", "paddle.min", "paddle.prod", "paddle.sum", "paddle.squeeze"
)
def reduction_axis_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    used_list_axes = None

    def init_used_list_axes(x_shape, axis_arg):
        used_axes = set()
        max_dim = max(len(x_shape), 1)
        if isinstance(axis_arg, (list, tuple)):
            for item in axis_arg:
                if config.is_tensor_config(item):
                    continue
                if not isinstance(item, int):
                    raise ValueError(f"Invalid item type for axis: {type(item)}")
                if not (-max_dim <= item < max_dim):
                    raise ValueError(f"Axis value {item} out of range [-{max_dim}, {max_dim})")
                positive_axis = item + max_dim if item < 0 else item
                if positive_axis in used_axes:
                    raise ValueError(f"Duplicate axis value: {item}")
                used_axes.add(positive_axis)
        return used_axes

    def axis_value(binding):
        nonlocal used_list_axes
        x_shape = config.arg(0, "x").shape
        max_dim = max(len(x_shape), 1)
        axis_arg = config.arg(1, "axis", None)
        if isinstance(axis_arg, (list, tuple)) and binding.path.indices:
            if binding.spec.shape not in [(), (1,)]:
                raise ValueError(
                    f"Invalid TensorConfig for axis: shape {binding.spec.shape} "
                    f"or dtype {binding.spec.dtype}"
                )
            if binding.spec.dtype not in {"int32", "int64"}:
                raise ValueError(
                    f"Invalid TensorConfig for axis: shape {binding.spec.shape} "
                    f"or dtype {binding.spec.dtype}"
                )
            if used_list_axes is None:
                used_list_axes = init_used_list_axes(x_shape, axis_arg)
            available_dims = sorted(set(range(max_dim)) - used_list_axes)
            if not available_dims:
                raise ValueError("Not enough available dimensions for axis TensorConfig items")
            dim = values.backend.choice(available_dims, replace=False)
            dim = int(dim)
            used_list_axes.add(dim)
            if values.backend.random() > 0.5:
                dim -= max_dim
            return values.backend.asarray(dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 0:
            dim = values.backend.randint(0, max_dim)
            if values.backend.random() > 0.5:
                dim -= max_dim
            return values.backend.asarray(dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1:
            dims = values.backend.choice(max_dim, shape=binding.spec.shape[0], replace=False)
            mask = values.backend.random(binding.spec.shape[0]) > 0.5
            dims = values.backend.where(mask, dims - max_dim, dims)
            return values.backend.asarray(dims, dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register("paddle.unsqueeze")
def unsqueeze_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_shape = config.arg(0, "x").shape
        max_dim = len(x_shape) + 1
        if len(binding.spec.shape) == 0:
            dim = values.backend.randint(0, max_dim)
            if values.backend.random() > 0.5:
                dim -= max_dim
            return values.backend.asarray(dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1:
            dims = values.backend.choice(max_dim, shape=binding.spec.shape[0], replace=False)
            mask = values.backend.random(binding.spec.shape[0]) > 0.5
            dims = values.backend.where(mask, dims - max_dim, dims)
            return values.backend.asarray(dims, dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register("paddle.unflatten", aliases=("paddle.Tensor.unflatten",))
def unflatten_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_shape = config.arg(0, "x").shape
        return values.backend.cast(
            values.backend.randint(0, len(x_shape), shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register("paddle.topk", aliases=("paddle.Tensor.topk",))
def topk_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        dtype = binding.spec.dtype
        if dtype == "bfloat16" or dtype in {"float8_e4m3fn", "float8_e5m2"}:
            dtype = "float32" if dtype == "bfloat16" else "float16"
        if dtype in {"float32", "float64"}:
            return values.backend.cast(
                (values.backend.random(binding.spec.shape) - 0.5) * 1.2,
                dtype,
            )
        if dtype == "float16":
            return values.backend.cast(
                values.backend.cast(values.backend.randn(*binding.spec.shape), dtype) * 1e-3,
                dtype,
            )
        if dtype in {"int32", "int64"}:
            return values.backend.cast(
                values.backend.randint(-10, 10, shape=binding.spec.shape),
                dtype,
            )
        raise ValueError(
            f"Unsupported dtype {binding.spec.dtype} for paddle.topk / paddle.Tensor.topk"
        )

    def k_value(binding):
        x_config = config.arg(0, "x")
        axis = config.arg(2, "axis", -1)
        max_k_value = 1
        if x_config is not None and x_config.shape:
            max_k_value = x_config.shape[axis] if len(x_config.shape) > 0 else 1
        if not binding.spec.shape:
            return values.backend.asarray(
                values.backend.randint(1, max_k_value + 1), dtype=binding.spec.dtype
            )
        return values.backend.cast(
            values.backend.randint(1, max_k_value + 1, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", x_value),
            ("k", k_value),
        ),
        default="default",
    )


@rules.register("paddle.index_sample")
def index_sample_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        x_dim = config.arg(0, "x").shape[1]
        return values.backend.randint(0, x_dim, shape=binding.spec.shape)

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.Tensor.__getitem__")
def tensor_getitem_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def source_binding():
        binding = config.binding("arr") or config.binding("x") or config.binding("self")
        if binding is None:
            raise ValueError("Tensor.__getitem__ rule could not find source tensor")
        return binding

    def item_value(binding):
        min_dim = min(source_binding().spec.shape)
        numel = math.prod(binding.spec.shape)
        if binding.spec.dtype == "bool":
            indices = values.backend.choice([0, 1], shape=numel)
        else:
            indices = values.backend.randint(0, min_dim, shape=numel)
        return values.backend.cast(
            values.backend.reshape(indices, binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("item", item_value),), default="default")


@rules.register("paddle.Tensor.__setitem__")
def tensor_setitem_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def source_binding():
        binding = config.binding("arr") or config.binding("x") or config.binding("self")
        if binding is None:
            raise ValueError("Tensor.__setitem__ rule could not find source tensor")
        return binding

    def item_value(binding):
        min_dim = min(source_binding().spec.shape)
        numel = math.prod(binding.spec.shape)
        if binding.spec.dtype == "bool":
            value = config.arg(2, "value")
            if value is not None and hasattr(value, "shape"):
                indices = values.backend.zeros(numel, dtype="int64")
                num_true = min(value.shape[0], numel)
                true_indices = values.backend.choice(numel, shape=num_true, replace=False)
                indices[true_indices] = 1
            else:
                indices = values.backend.choice([0, 1], shape=numel)
        else:
            indices = values.backend.randint(0, min_dim, shape=numel)
        return values.backend.cast(
            values.backend.reshape(indices, binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("item", item_value),), default="default")


@rules.register("paddle.index_add", "paddle.index_fill")
def index_update_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        axis = config.arg(2, "axis")
        if axis is None:
            raise ValueError("Axis is None")
        x_shape = config.arg(0, "x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
        if len(binding.spec.shape) >= 1:
            return values.backend.cast(
                values.backend.randint(0, x_shape[axis], shape=binding.spec.shape),
                binding.spec.dtype,
            )
        raise ValueError(
            f"Invalid shape for 'index' Tensor in {config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.take")
def take_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        x = config.arg(0, "x")
        dim_size = math.prod(x.shape)
        return values.backend.cast(
            values.backend.randint(0, dim_size, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.gather", aliases=("paddle.Tensor.gather",))
def gather_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        x = config.arg(0, "x")
        if config.has_kwarg("axis"):
            axis = config.arg(2, "axis")
            if hasattr(axis, "shape"):
                axis = axis.shape[0]
        else:
            axis = 0
        return values.backend.cast(
            values.backend.randint(0, x.shape[axis], shape=binding.spec.shape),
            binding.spec.dtype,
        )

    def axis_value(binding):
        return values.backend.cast(
            values.backend.randint(0, 2, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("index", index_value),
            ("axis", axis_value),
        ),
        default="default",
    )


@rules.register("paddle.gather_nd", aliases=("paddle.Tensor.gather_nd",))
def gather_nd_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        x_shape = config.arg(0, "x").shape
        index_shape = config.arg(1, "index").shape
        result = values.backend.zeros(index_shape, dtype=binding.spec.dtype)
        for index in range(index_shape[-1]):
            result[..., index] = values.backend.randint(
                0, x_shape[index], shape=result[..., index].shape
            )
        return result

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.index_select", aliases=("paddle.Tensor.index_select",))
def index_select_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        axis = config.arg(2, "axis")
        if axis is None:
            axis = 0
        x = config.arg(0, "x")
        if x.shape[axis] == 0:
            return values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return values.backend.cast(
            values.backend.randint(0, x.shape[axis], shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.take_along_axis", aliases=("paddle.Tensor.take_along_axis",))
def take_along_axis_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def indices_value(binding):
        arr_shape = config.arg(0, "arr").shape
        axis = config.arg(2, "axis")
        axis_value = axis if axis >= 0 else axis + len(arr_shape)
        dim_size = arr_shape[axis_value]
        dtype = binding.spec.dtype if binding.spec.dtype in {"int32", "int64"} else "int64"
        num_elements = math.prod(binding.spec.shape)
        if num_elements == 0:
            indices = values.backend.asarray([], dtype=dtype)
        elif dim_size == 1:
            indices = values.backend.zeros(num_elements, dtype=dtype)
        elif num_elements == 1:
            indices = values.backend.asarray([0], dtype=dtype)
        else:
            indices = values.backend.cast(
                values.backend.randint(0, dim_size, shape=num_elements), dtype
            )
            positions_to_replace = values.backend.choice(num_elements, shape=2, replace=False)
            flat_indices = values.backend.flatten(indices)
            flat_indices[positions_to_replace[0]] = 0
            flat_indices[positions_to_replace[1]] = dim_size - 1
            indices = flat_indices
        return values.backend.reshape(indices, binding.spec.shape)

    generate_by_parameter(config, values, writer, (("indices", indices_value),), default="default")


@rules.register("paddle.index_put", aliases=("paddle.Tensor.index_put",))
def index_put_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    state = {}

    def prepare_indices_state():
        x = config.arg(0, "x")
        value = config.arg(2, "value")
        indices = config.arg(1, "indices")
        if not isinstance(indices, (list, tuple)):
            return None

        x_shape = x.shape
        value_shape = value.shape
        int_index_shapes = []
        has_bool_index = False
        dims_consumed = 0
        for item in indices:
            if not config.is_tensor_config(item):
                continue
            if item.dtype == "bool":
                has_bool_index = True
                dims_consumed += len(item.shape)
            else:
                int_index_shapes.append(tuple(item.shape))
                dims_consumed += 1

        if dims_consumed > len(x_shape):
            raise ValueError(
                f"Too many indices: consume {dims_consumed} dims but x has {len(x_shape)} dims"
            )

        num_true_needed = -1
        num_remaining_dims = len(x_shape) - dims_consumed
        advanced_shape = ()
        if int_index_shapes:
            try:
                advanced_shape = numpy.broadcast_shapes(*int_index_shapes)
                if (
                    has_bool_index
                    and len(value_shape) > num_remaining_dims
                    and advanced_shape[-1] == 1
                    and value_shape[-num_remaining_dims - 1] != 1
                ):
                    advanced_shape = (
                        *advanced_shape[:-1],
                        value_shape[-num_remaining_dims - 1],
                    )
                num_true_needed = advanced_shape[-1]
            except Exception as err:
                raise ValueError(
                    f"Incompatible integer index shapes for broadcasting: {int_index_shapes}"
                ) from err
        elif has_bool_index:
            if len(value_shape) > num_remaining_dims:
                advanced_shape = (value_shape[0],)
                num_true_needed = value_shape[0]
            else:
                advanced_shape = (1,)
                num_true_needed = 1

        result_shape = advanced_shape + tuple(x_shape[dims_consumed:])
        try:
            numpy.broadcast_shapes(tuple(value_shape), result_shape)
        except ValueError as err:
            raise ValueError(
                f"Value shape {value_shape} cannot be broadcast to the indexed shape "
                f"{result_shape}."
            ) from err

        return {
            "x_shape": x_shape,
            "x_dim_cursor": 0,
            "num_true_needed": num_true_needed,
        }

    def int_indices(shape, dim_size):
        num_elements = values.backend.prod(shape)
        if num_elements > dim_size:
            indices_flat = values.backend.randint(-dim_size, dim_size, shape=num_elements)
        else:
            indices_flat = values.backend.choice(dim_size, shape=num_elements, replace=False)
        return values.backend.reshape(indices_flat, shape)

    def bool_mask(shape, num_true):
        mask_size = values.backend.prod(shape)
        if mask_size < num_true:
            raise ValueError(
                f"Cannot generate a mask with {num_true} true values in a {mask_size} element mask"
            )
        mask_flat = values.backend.zeros(mask_size, dtype="bool")
        true_indices = values.backend.choice(mask_size, shape=num_true, replace=False)
        mask_flat[true_indices] = True
        return values.backend.reshape(mask_flat, shape)

    def index_value(binding):
        if not state:
            prepared = prepare_indices_state()
            if prepared is None:
                return values.domain("default", binding)
            state.update(prepared)

        if binding.spec.dtype == "bool":
            if state["num_true_needed"] < 0:
                raise ValueError(
                    "Cannot determine the number of True elements for the boolean mask."
                )
            state["x_dim_cursor"] += len(binding.spec.shape)
            return bool_mask(binding.spec.shape, state["num_true_needed"])

        x_dim_to_index = state["x_shape"][state["x_dim_cursor"]]
        state["x_dim_cursor"] += 1
        return values.backend.cast(
            int_indices(binding.spec.shape, x_dim_to_index),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("indices", index_value),), default="default")


@rules.register("paddle.multiplex")
def multiplex_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        axis_values = config.arg(0, "inputs")
        return values.backend.cast(
            values.backend.randint(0, len(axis_values), shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


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
def segment_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def segment_ids_value(binding):
        batch_size = config.arg(0, "x").shape[0]
        max_segments = values.backend.randint(1, batch_size + 1)
        segment_ids = values.backend.cast(
            values.backend.randint(0, max_segments, shape=binding.spec.shape),
            binding.spec.dtype,
        )
        return values.backend.sort(segment_ids)

    generate_by_parameter(
        config, values, writer, (("segment_ids", segment_ids_value),), default="default"
    )


@rules.register(
    "paddle.geometric.send_u_recv",
    "paddle.geometric.send_uv",
    "paddle.geometric.send_ue_recv",
)
def geometric_send_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        num_nodes = config.arg(0, "x").shape[0]
        return values.backend.cast(
            values.backend.randint(0, num_nodes, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        ((("src_index", "dst_index"), index_value),),
        default="default",
    )


@rules.register("paddle.geometric.sample_neighbors")
def sample_neighbors_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def row_value(binding):
        colptr_shape = config.arg(1, "colptr").shape
        num_nodes = colptr_shape[0] - 1
        return values.backend.randint(
            0, num_nodes, shape=binding.spec.shape, dtype=binding.spec.dtype
        )

    def colptr_value(binding):
        row = config.arg(0, "row")
        num_edges = row.shape[0]
        num_nodes = binding.spec.shape[0] - 1
        colptr = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        if num_nodes > 0 and num_edges > 0:
            splits = values.backend.choice(
                values.backend.arange(num_edges + 1),
                shape=num_nodes - 1,
                replace=True,
            )
            splits = values.backend.sort(splits)
            colptr[1:num_nodes] = splits
            colptr[num_nodes] = num_edges
        return colptr

    def input_nodes_value(binding):
        num_nodes = binding.spec.shape[0] - 1
        return values.backend.randint(
            0, num_nodes, shape=binding.spec.shape, dtype=binding.spec.dtype
        )

    def edge_order_value(binding):
        num_edges = config.arg(0, "row").shape[0]
        return values.backend.reshape(
            values.backend.arange(num_edges, dtype=binding.spec.dtype),
            binding.spec.shape,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("row", row_value),
            ("colptr", colptr_value),
            ("input_nodes", input_nodes_value),
            (("eids", "perm_buffer"), edge_order_value),
        ),
        default="default",
    )


@rules.register("paddle.reshape", aliases=("paddle.Tensor.reshape",))
def reshape_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
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
            for candidate in config.argument_values():
                if isinstance(candidate, (list, tuple)):
                    for index, item in enumerate(candidate):
                        if isinstance(item, numbers.Integral):
                            if item == 0:
                                state["maxvalue"] //= shape[index]
                            elif item != -1:
                                state["maxvalue"] //= int(item)
                        elif config.is_tensor_config(item):
                            state["tensornum"] += 1
        return values.domain("default", binding)

    def shape_value(binding):
        if state["tensornum"] == 0:
            state["tensornum"] = 1
        dtype = "int32"
        shape = binding.spec.shape
        maxvalue = state["maxvalue"]
        if shape not in ((), (1,)):
            result = values.backend.zeros(shape, dtype=dtype)
            for index in range(shape[0]):
                if index < shape[0] - 1:
                    result[index] = values.backend.randint(1, maxvalue + 1)
                    while maxvalue % result[index]:
                        result[index] = values.backend.randint(1, maxvalue + 1)
                    maxvalue //= result[index]
                else:
                    result[index] = maxvalue
            state["maxvalue"] = maxvalue
            return result
        if state["tensornum"] == 1:
            return values.backend.cast(
                values.backend.randint(maxvalue, maxvalue + 1, shape=shape),
                dtype,
            )
        state["tensornum"] -= 1
        result = values.backend.cast(values.backend.randint(1, maxvalue + 1, shape=shape), dtype)
        while maxvalue % result:
            result = values.backend.cast(
                values.backend.randint(1, maxvalue + 1, shape=shape),
                dtype,
            )
        state["maxvalue"] = maxvalue // result
        return result

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", initialize_from_x),
            ("shape", shape_value),
        ),
        default="default",
    )


@rules.register("paddle.slice")
def slice_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    state = {
        "shape": None,
        "indice": 0,
        "start": [],
        "index": 0,
    }

    def axes():
        return config.arg(1, "axes")

    def input_binding(binding):
        if state["shape"] is None:
            state["shape"] = binding.spec.shape
        return values.domain("default", binding)

    def starts_value(binding):
        dim_sizes = [state["shape"][axis] for axis in axes()]
        if binding.spec.shape == ():
            coin = values.backend.randint(0, 2)
            if coin == 0:
                value = values.backend.randint(
                    0, dim_sizes[state["indice"]] - 1, binding.spec.shape
                )
            else:
                value = values.backend.randint(-65535, -1, binding.spec.shape)
            state["start"].append(value)
            state["indice"] += 1
            return values.backend.asarray(value, dtype=binding.spec.dtype)
        result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(math.prod(binding.spec.shape)):
            coin = values.backend.randint(0, 2)
            if coin == 0:
                result[index] = values.backend.randint(0, dim_sizes[state["indice"]] - 1)
            else:
                result[index] = values.backend.randint(-65535, -1)
            state["start"].append(result[index])
            state["indice"] += 1
        return result

    def ends_value(binding):
        if not state["start"]:
            start_arg = config.arg(2, "starts")
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
            coin = values.backend.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                value = values.backend.randint(current + 1, 65535, binding.spec.shape)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                value = values.backend.randint(
                    min(current - dim_sizes[index] + 1, -1), 0, binding.spec.shape
                )
            state["index"] += 1
            return values.backend.asarray(value, dtype=binding.spec.dtype)
        result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for index in range(math.prod(binding.spec.shape)):
            coin = values.backend.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                result[index] = values.backend.randint(current + 1, 65535)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                result[index] = values.backend.randint(current - dim_sizes[state["index"]] + 1, 0)
            state["index"] += 1
        return result

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("input", input_binding),
            ("starts", starts_value),
            ("ends", ends_value),
        ),
        default="default",
    )


@rules.register("paddle.scatter")
def scatter_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        x = config.arg(0, "x")
        first_dim = x.shape[0]
        overwrite = config.arg(3, "overwrite")
        if (overwrite is None or overwrite is True) and (
            binding.spec.shape == () or binding.spec.shape[0]
        ) <= first_dim:
            return values.backend.cast(
                values.backend.choice(first_dim, shape=binding.spec.shape, replace=False),
                binding.spec.dtype,
            )
        return values.backend.cast(
            values.backend.randint(0, first_dim, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.scatter_nd")
def scatter_nd_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        output_shape = config.arg(2, "shape")
        if output_shape and len(output_shape):
            result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
            for axis in range(len(output_shape)):
                if axis >= binding.spec.shape[-1]:
                    break
                result[..., axis] = values.backend.randint(
                    -output_shape[axis],
                    output_shape[axis],
                    shape=result[..., axis].shape,
                )
                result[..., axis] = values.backend.cast(result[..., axis], binding.spec.dtype)
            return result
        return values.domain("default", binding)

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.scatter_nd_add")
def scatter_nd_add_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def index_value(binding):
        x_shape = config.arg(0, "x").shape
        result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for axis in range(binding.spec.shape[-1]):
            result[..., axis] = values.backend.randint(
                -x_shape[axis],
                x_shape[axis],
                shape=result[..., axis].shape,
            )
            result[..., axis] = values.backend.cast(result[..., axis], binding.spec.dtype)
        return result

    generate_by_parameter(config, values, writer, (("index", index_value),), default="default")


@rules.register("paddle.strided_slice")
def strided_slice_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axes_value(binding):
        x = config.arg(0, "x")
        return values.backend.cast(
            values.backend.randint(0, len(x.shape), shape=binding.spec.shape),
            binding.spec.dtype,
        )

    def list_value(binding):
        x = config.arg(0, "x")
        axes_arg = config.arg(1, "axes")
        axes = axes_arg
        if not isinstance(axes, list):
            axes = writer.value(config.binding("axes"))
        if not binding.path.indices:
            return values.domain("default", binding)
        list_index = binding.path.indices[0]
        parameter = binding.parameter_name
        if parameter == "starts":
            return values.backend.cast(
                values.backend.randint(0, x.shape[axes[list_index]] - 1, shape=binding.spec.shape),
                binding.spec.dtype,
            )
        if parameter == "ends":
            return values.backend.cast(
                values.backend.randint(
                    writer.value(config.binding("starts"))[list_index] + 1,
                    x.shape[axes[list_index]],
                    shape=binding.spec.shape,
                ),
                binding.spec.dtype,
            )
        if parameter == "strides":
            return values.backend.cast(
                values.backend.randint(1, x.shape[axes[list_index]], shape=binding.spec.shape),
                binding.spec.dtype,
            )
        return values.domain("default", binding)

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("axes", axes_value),
            (("starts", "ends", "strides"), list_value),
        ),
        default="default",
    )


@rules.register("paddle.tensordot")
def tensordot_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    state = {"shape1": None, "shape2": None, "tensor1": None}

    def x_value(binding):
        if state["shape1"] is None:
            state["shape1"] = binding.spec.shape
        return values.domain("default", binding)

    def y_value(binding):
        if state["shape2"] is None:
            state["shape2"] = binding.spec.shape
        return values.domain("default", binding)

    def axes_value(binding):
        axes_arg = config.arg(2, "axes")
        rank = len(state["shape1"])
        if isinstance(axes_arg, (list, tuple)):
            if state["tensor1"] is None:
                result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
                used = []
                for index in range(math.prod(binding.spec.shape)):
                    result[index] = values.backend.randint(0, rank)
                    while (
                        state["shape1"][result[index]] not in state["shape2"]
                        or result[index] in used
                    ):
                        result[index] = values.backend.randint(0, rank)
                    used.append(result[index])
                state["tensor1"] = result
                return result
            result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
            used = []
            for index in range(math.prod(binding.spec.shape)):
                result[index] = values.backend.randint(0, rank)
                while (
                    state["shape2"][result[index]] != state["shape1"][state["tensor1"][index]]
                    or result[index] in used
                ):
                    result[index] = values.backend.randint(0, rank)
                used.append(result[index])
            return result
        if binding.spec.shape == () or math.prod(binding.spec.shape) == 1:
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
            return values.backend.asarray(
                [values.backend.choice(candidates)], dtype=binding.spec.dtype
            )
        result = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        used1 = []
        used2 = []
        for index in range(binding.spec.shape[0]):
            result[0][index] = values.backend.randint(0, rank)
            result[1][index] = values.backend.randint(0, rank)
            while (
                state["shape1"][result[0][index]] != state["shape2"][result[1][index]]
                or result[0][index] in used1
                or result[1][index] in used2
            ):
                result[0][index] = values.backend.randint(0, rank)
                result[1][index] = values.backend.randint(0, rank)
            used1.append(result[0][index])
            used2.append(result[1][index])
        return result

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", x_value),
            ("y", y_value),
            ("axes", axes_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.embedding")
def embedding_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def ids_value(binding):
        weight_config = config.arg(1, "weight")
        vocab_size = values.backend.randint(10, 1000)
        if weight_config is not None and weight_config.shape:
            vocab_size = weight_config.shape[0]
        if vocab_size == 0:
            return values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return values.backend.cast(
            values.backend.randint(0, vocab_size, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", ids_value),
            ("ids", ids_value),
            ("weight", "multiply"),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.affine_grid")
def affine_grid_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def out_shape_value(binding):
        theta_shape = config.arg(0, "theta").shape
        out_shape = values.backend.cast(
            values.backend.randint(1, 128, shape=binding.spec.shape),
            binding.spec.dtype,
        )
        out_shape[0] = theta_shape[0]
        return out_shape

    generate_by_parameter(
        config, values, writer, (("out_shape", out_shape_value),), default="default"
    )


@rules.register("paddle.nn.functional.hsigmoid_loss")
def hsigmoid_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def label_value(binding):
        num_classes = config.arg(2, "num_classes")
        return values.backend.cast(
            values.backend.randint(0, num_classes, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    def path_table_value(binding):
        weight = config.arg(3, "weight")
        return values.backend.cast(
            values.backend.randint(0, weight.shape[0], shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("label", label_value),
            ("path_table", path_table_value),
            ("path_code", "binary_0_1"),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.margin_cross_entropy")
def margin_cross_entropy_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def label_value(binding):
        logits = config.arg(0, "logits")
        return values.backend.cast(
            values.backend.randint(0, logits.shape[1], shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.nn.functional.multi_margin_loss")
def multi_margin_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def label_value(binding):
        logits = config.arg(0, "input")
        return values.backend.cast(
            values.backend.randint(0, logits.shape[1], shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.nn.functional.dice_loss")
def dice_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def label_value(binding):
        tensor = config.arg(0, "input")
        return values.backend.cast(
            values.backend.randint(0, tensor.shape[-1], shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.nn.functional.nll_loss")
def nll_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def label_value(binding):
        input_config = config.arg(0, "input")
        n_classes = values.backend.randint(5, 50) if input_config is None else input_config.shape[1]
        return values.backend.cast(
            values.backend.randint(0, n_classes, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.nn.functional.adaptive_log_softmax_with_loss")
def adaptive_log_softmax_with_loss_values(
    config: ConfigView, values: ValueFactory, writer: InputWriter
):
    def label_value(binding):
        cutoffs = config.arg(4, "cutoffs")
        n_classes = cutoffs[-1]
        generation_size = binding.spec.shape
        if len(binding.spec.shape) == 0:
            generation_size = 1
        if n_classes == 1:
            return values.backend.zeros(generation_size, dtype=binding.spec.dtype)
        return values.backend.randint(0, n_classes, shape=generation_size, dtype=binding.spec.dtype)

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.nn.functional.cross_entropy")
def cross_entropy_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def input_value(binding):
        use_softmax = config.arg(8, "use_softmax", True)
        if use_softmax:
            return values.domain("default", binding)
        axis = config.arg(7, "axis", -1)
        logits = values.backend.random(binding.spec.shape)
        probabilities = logits / values.backend.sum(logits, axis=axis, keepdims=True)
        return values.backend.cast(probabilities, binding.spec.dtype)

    def label_value(binding):
        input_shape = config.arg(0, "input").shape
        axis = config.arg(7, "axis", -1)
        num_classes = input_shape[axis]
        soft_label = config.arg(5, "soft_label", False)
        label_smoothing = config.arg(6, "label_smoothing", 0.0)
        if (label_smoothing > 0 and list(binding.spec.shape) == list(input_shape)) or (
            label_smoothing == 0 and soft_label
        ):
            soft_labels = values.backend.random(binding.spec.shape)
            soft_labels = soft_labels / values.backend.sum(soft_labels, axis=axis, keepdims=True)
            return values.backend.cast(soft_labels, binding.spec.dtype)
        if num_classes == 0:
            return values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return values.backend.cast(
            values.backend.randint(0, num_classes, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    def weight_value(binding):
        weights = values.backend.random(binding.spec.shape)
        return weights / values.backend.sum(weights)

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("input", input_value),
            ("label", label_value),
            ("weight", weight_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.ctc_loss")
def ctc_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def labels_value(binding):
        num_classes = config.arg(0, "log_probs").shape[2] - 1
        blank = config.arg(4, "blank", 0)
        valid_label_indices = [index for index in range(num_classes + 1) if index != blank]
        if not valid_label_indices:
            return values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        return values.backend.cast(
            values.backend.choice(valid_label_indices, shape=binding.spec.shape, replace=True),
            binding.spec.dtype,
        )

    def input_lengths_value(binding):
        max_logit_length = config.arg(0, "log_probs").shape[0]
        return values.backend.randint(
            1,
            max_logit_length + 1,
            shape=binding.spec.shape,
            dtype=binding.spec.dtype,
        )

    def label_lengths_value(binding):
        max_label_length = config.arg(1, "labels").shape[1]
        max_logit_length = config.arg(0, "log_probs").shape[0]
        cand_label_lengths = values.backend.randint(
            1,
            max_label_length + 1,
            shape=binding.spec.shape,
            dtype=binding.spec.dtype,
        )
        compatible_input_lengths = values.backend.randint(
            1,
            max_logit_length + 1,
            shape=binding.spec.shape,
            dtype=binding.spec.dtype,
        )
        final_label_lengths = values.backend.minimum(cand_label_lengths, compatible_input_lengths)
        return values.backend.maximum(final_label_lengths, 1)

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("labels", labels_value),
            ("input_lengths", input_lengths_value),
            ("label_lengths", label_lengths_value),
        ),
        default="default",
    )


@rules.register("paddle.nn.functional.sequence_mask")
def sequence_mask_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        maxlen_config = config.arg(1, "maxlen")
        provided_maxlen = None
        if isinstance(maxlen_config, int):
            provided_maxlen = max(1, maxlen_config)
        if provided_maxlen is not None:
            return values.backend.cast(
                values.backend.randint(0, provided_maxlen + 1, shape=binding.spec.shape),
                binding.spec.dtype,
            )
        high_value = values.backend.randint(1, 2048)
        lengths = values.backend.cast(
            values.backend.randint(0, high_value, shape=binding.spec.shape),
            binding.spec.dtype,
        )
        if values.backend.prod(lengths.shape) > 0 and values.backend.count_nonzero(lengths) == 0:
            fix_value = values.backend.randint(1, max(2, high_value))
            values.backend.flatten(lengths)[0] = fix_value
        return lengths

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.nn.functional.softmax_with_cross_entropy")
def softmax_with_cross_entropy_values(
    config: ConfigView, values: ValueFactory, writer: InputWriter
):
    def label_value(binding):
        logits = config.arg(0, "logits")
        if not hasattr(logits, "shape"):
            logits = config.kwarg("logits")
        num_classes = 10
        if logits is not None:
            axis = config.kwarg("axis", -1)
            axis = axis if axis >= 0 else len(logits.shape) + axis
            if 0 <= axis < len(logits.shape):
                num_classes = logits.shape[axis]
        else:
            num_classes = values.backend.randint(5, 20)
        return values.backend.cast(
            values.backend.randint(0, num_classes, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(config, values, writer, (("label", label_value),), default="default")


@rules.register("paddle.linalg.cholesky")
def cholesky_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        if len(binding.spec.shape) < 2 or binding.spec.shape[-1] != binding.spec.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = binding.spec.shape[:-2]
        matrix_dim = binding.spec.shape[-1]
        matrix = values.backend.random(
            [*batch_dims, matrix_dim, matrix_dim], dtype=binding.spec.dtype
        )
        if len(batch_dims) > 0:
            tensor = values.backend.einsum("...ij,...kj->...ik", matrix, matrix)
        else:
            tensor = values.backend.dot(matrix, values.backend.swapaxes(matrix, -1, -2))
        tensor += values.backend.eye(matrix_dim, dtype=binding.spec.dtype) * 10000
        print("cholesky tensor", tensor)
        return tensor

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.linalg.cov")
def covariance_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def observation_count():
        x_shape = config.arg(0, "x").shape
        rowvar = config.arg(1, "rowvar")
        if rowvar is None:
            rowvar = True
        return (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]

    def x_value(binding):
        if len(binding.spec.shape) < 1 or len(binding.spec.shape) > 2:
            raise ValueError("Shape must have 1 or 2 dimensions for covariance input")
        tensor = values.backend.random(binding.spec.shape, dtype=binding.spec.dtype)
        tensor += values.backend.random(binding.spec.shape, dtype=binding.spec.dtype) * 1e-6
        return tensor

    def fweights_value(binding):
        return values.backend.cast(
            values.backend.randint(1, 11, shape=(observation_count(),)),
            binding.spec.dtype,
        )

    def aweights_value(binding):
        if binding.spec.dtype in ["float32", "float64"]:
            return values.backend.uniform(
                0.1, 1.0, shape=(observation_count(),), dtype=binding.spec.dtype
            )
        return values.backend.cast(
            values.backend.randint(1, 11, shape=(observation_count(),)),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", x_value),
            ("fweights", fweights_value),
            ("aweights", aweights_value),
        ),
        default="default",
    )


@rules.register("paddle.linalg.eigh", "paddle.linalg.eigvalsh")
def eigen_symmetric_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        if len(binding.spec.shape) < 2 or binding.spec.shape[-1] != binding.spec.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = binding.spec.shape[:-2]
        matrix_dim = binding.spec.shape[-1]
        matrix = values.backend.random(
            [*batch_dims, matrix_dim, matrix_dim], dtype=binding.spec.dtype
        )
        if binding.spec.dtype in ["complex64", "complex128"]:
            matrix = matrix + 1j * values.backend.random(
                [*batch_dims, matrix_dim, matrix_dim],
                dtype=binding.spec.dtype,
            )
            tensor = matrix + values.backend.conj(values.backend.swapaxes(matrix, -1, -2))
        elif len(batch_dims) > 0:
            tensor = values.backend.einsum("...ij,...kj->...ik", matrix, matrix)
        else:
            tensor = values.backend.dot(matrix, values.backend.swapaxes(matrix, -1, -2))
        tensor += values.backend.eye(matrix_dim, dtype=binding.spec.dtype) * 1e-6
        return tensor

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.linalg.lstsq")
def lstsq_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def matrix_value(binding):
        if len(binding.spec.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for lstsq x")
        batch_dims = binding.spec.shape[:-2]
        rows, cols = binding.spec.shape[-2], binding.spec.shape[-1]
        return values.backend.random([*batch_dims, rows, cols], dtype=binding.spec.dtype)

    generate_by_parameter(config, values, writer, ((("x", "y"), matrix_value),), default="default")


@rules.register("paddle.linalg.lu_unpack")
def lu_unpack_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        if len(binding.spec.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for LU matrix")
        tensor = values.backend.random(binding.spec.shape, dtype=binding.spec.dtype)
        diagonal_size = min(binding.spec.shape[-2], binding.spec.shape[-1])
        tensor[..., range(diagonal_size), range(diagonal_size)] += 1e-6
        return tensor

    def pivot_value(binding):
        row_count = config.arg(0, "x").shape[-2]
        return values.backend.cast(
            values.backend.randint(1, row_count + 1, shape=binding.spec.shape),
            binding.spec.dtype,
        )

    generate_by_parameter(
        config,
        values,
        writer,
        (
            ("x", x_value),
            (("pivot", "y"), pivot_value),
        ),
        default="default",
    )


@rules.register("paddle.linalg.cond")
def condition_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        matrix_size = binding.spec.shape[-1]
        tensor = values.backend.random(binding.spec.shape, dtype=binding.spec.dtype)
        tensor += matrix_size * values.backend.eye(matrix_size, dtype=binding.spec.dtype)
        return tensor

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.linalg.det", "paddle.linalg.slogdet")
def determinant_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        if len(binding.spec.shape) < 2:
            raise AssertionError("Input must be at least 2D.")
        if binding.spec.shape[-1] != binding.spec.shape[-2]:
            raise AssertionError("Input must be square matrices.")
        matrix_size = binding.spec.shape[-1]
        is_complex = binding.spec.dtype.startswith("complex")
        if is_complex:
            real_dtype = "float32" if binding.spec.dtype == "complex64" else "float64"
            real = values.backend.uniform(0.5, 1.0, shape=binding.spec.shape, dtype=real_dtype)
            imag = values.backend.uniform(0.5, 1.0, shape=binding.spec.shape, dtype=real_dtype)
            matrix = values.backend.cast(real + 1j * imag, binding.spec.dtype)
            matrix_h = values.backend.swapaxes(values.backend.conj(matrix), -1, -2)
        else:
            matrix = values.backend.uniform(
                0.5, 1.0, shape=binding.spec.shape, dtype=binding.spec.dtype
            )
            matrix_h = values.backend.swapaxes(matrix, -1, -2)
        return values.backend.matmul(matrix, matrix_h) + values.backend.eye(
            matrix_size, dtype=binding.spec.dtype
        )

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.linalg.pca_lowrank")
def pca_lowrank_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        return values.backend.cast(values.backend.randn(*binding.spec.shape), binding.spec.dtype)

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.linalg.corrcoef")
def corrcoef_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        if binding.spec.dtype == "float16":
            return (
                values.backend.cast(
                    values.backend.randn(*binding.spec.shape),
                    binding.spec.dtype,
                )
                * 1e-3
            )
        return values.domain("default", binding)

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


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
def linalg_default_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    generate_all(config, values, writer, "default")


@rules.register("paddle.linalg.pinv")
def pinv_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    hermitian = bool(config.arg(2, " hermitian", False))
    if not hermitian:
        generate_all(config, values, writer, "default")
        return

    def x_value(binding):
        if len(binding.spec.shape) not in [2, 3]:
            raise ValueError("pinv only supports 2D or 3D tensors")
        if binding.spec.dtype.startswith("complex"):
            real_dtype = "float32" if binding.spec.dtype == "complex64" else "float64"
            real = values.backend.cast(values.backend.randn(*binding.spec.shape), real_dtype)
            imag = values.backend.cast(values.backend.randn(*binding.spec.shape), real_dtype)
            matrix = values.backend.cast(real + 1j * imag, binding.spec.dtype)
        else:
            matrix = values.backend.cast(
                values.backend.randn(*binding.spec.shape),
                binding.spec.dtype,
            )
        if len(binding.spec.shape) == 2:
            matrix_t = (
                values.backend.swapaxes(values.backend.conj(matrix), -1, -2)
                if binding.spec.dtype.startswith("complex")
                else values.backend.swapaxes(matrix, -1, -2)
            )
        else:
            matrix_t = (
                values.backend.swapaxes(values.backend.conj(matrix), -2, -1)
                if binding.spec.dtype.startswith("complex")
                else values.backend.swapaxes(matrix, -2, -1)
            )
        return (matrix + matrix_t) / 2

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register("paddle.linalg.cholesky_solve", aliases=("paddle.Tensor.cholesky_solve",))
def cholesky_solve_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    if config.api_name == "paddle.linalg.cholesky_solve":
        generate_all(config, values, writer, "default")
        return

    def y_value(binding):
        value = values.domain("random_range", binding)
        if config.arg(2, "upper"):
            return values.backend.triu(value)
        return values.backend.tril(value)

    generate_by_parameter(config, values, writer, (("y", y_value),), default="default")


@rules.register("paddle.view", aliases=("paddle.Tensor.view",))
def view_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def x_value(binding):
        if binding.spec.dtype == "uint8":
            target = str(config.arg(1, "shape_or_dtype", ""))
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
                    finite_f32 = values.backend.cast(
                        (values.backend.random(numel) - 0.5) * 1.2,
                        "float32",
                    )
                    uint32_value = values.backend.view_dtype(finite_f32, "uint32")
                    return values.backend.view_dtype(
                        values.backend.cast(
                            values.backend.cast(uint32_value, "int64") >> 16,
                            "uint16",
                        ),
                        "uint8",
                    )
                finite = values.backend.cast(
                    (values.backend.random(numel) - 0.5) * 1.2,
                    target.replace("paddle.", ""),
                )
                return values.backend.view_dtype(values.backend.ascontiguousarray(finite), "uint8")
        return values.domain("default", binding)

    generate_by_parameter(config, values, writer, (("x", x_value),), default="default")


@rules.register(
    "paddle.pow",
    aliases=("paddle.Tensor.pow", "paddle.Tensor.__rpow__", "paddle.Tensor.__pow__"),
)
def pow_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
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
        api_name = config.api_name
        dtype = binding.spec.dtype
        if api_name == "paddle.Tensor.__rpow__":
            is_base_arg = binding.parameter_name in {"other", "y"} or str(binding.path) == "args[1]"
            if is_base_arg:
                const = config.arg(0, "self")
                get_max = get_base_max
                default_max = 10
            else:
                const = config.arg(1, "other")
                get_max = get_exponent_max
                default_max = 5
        else:
            is_base_arg = binding.parameter_name in {"self", "x"}
            if is_base_arg:
                const = config.arg(1, "other", config.arg(1, "y"))
                get_max = get_base_max
                default_max = 10
            else:
                const = config.arg(0, "self", config.arg(0, "x"))
                get_max = get_exponent_max
                default_max = 5
        if isinstance(const, numbers.Number):
            value_max = get_max(const, values.dtype_max(dtype), default_max)
            if is_base_arg and int(const) != const:
                return values.domain("random_range", binding, low=0, high=value_max)
            return values.domain("random_range", binding, low=-value_max, high=value_max)
        if is_base_arg:
            return values.domain("random_range", binding, low=0, high=default_max)
        return values.domain("random_range", binding, low=-default_max, high=default_max)

    generate_all(config, values, writer, value)


@rules.register("paddle.nn.functional.rnnt_loss")
def rnnt_loss_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def logits(binding):
        shape = binding.spec.shape if len(binding.spec.shape) == 4 else (3, 4, 3, 5)
        return values.backend.random(shape, dtype=binding.spec.dtype)

    def labels(binding):
        shape = binding.spec.shape if len(binding.spec.shape) == 2 else (3, 2)
        return values.backend.cast(values.backend.randint(1, 4, shape=shape), binding.spec.dtype)

    def lengths(max_possible_length):
        def generate(binding):
            shape = binding.spec.shape if len(binding.spec.shape) == 1 else (3,)
            return values.backend.ones(shape, dtype=binding.spec.dtype) * max_possible_length

        return generate

    generate_by_parameter(
        config,
        values,
        writer,
        (
            (("input", "logits"), logits),
            (("label", "labels"), labels),
            ("input_lengths", lengths(4)),
            ("label_lengths", lengths(2)),
        ),
        default="default",
    )


@rules.register("paddle.chunk")
def chunk_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_tensor = config.arg(0, "x")
        chunks = config.arg(1, "chunks")
        valid_axes = [
            index for index, dim_size in enumerate(x_tensor.shape) if dim_size % chunks == 0
        ]
        if not valid_axes:
            raise ValueError(
                f"No valid axis found in x.shape = {x_tensor.shape} for chunks = {chunks}. "
                f"Each dim must be divisible by chunks."
            )
        chosen_axis = values.backend.choice(valid_axes)
        if len(binding.spec.shape) == 0:
            return values.backend.asarray(chosen_axis, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1 and binding.spec.shape[0] == 1:
            return values.backend.asarray([chosen_axis], dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register("paddle.split")
def split_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def axis_value(binding):
        x_shape = config.arg(0, "x").shape
        num_or_sections = config.arg(1, "num_or_sections")
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
            target_dim = values.backend.randint(-1, 0)
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
            return values.backend.asarray(target_dim, dtype=binding.spec.dtype)
        if len(binding.spec.shape) == 1 and binding.spec.shape[0] == 1:
            return values.backend.asarray([target_dim], dtype=binding.spec.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.split. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.spec.shape}."
        )

    generate_by_parameter(config, values, writer, (("axis", axis_value),), default="default")


@rules.register("paddle.expand", aliases=("paddle.Tensor.expand",))
def expand_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def shape_value(binding):
        x_shape = config.arg(0, "x").shape
        shape_index = binding.path.indices[0] if binding.path.indices else 0
        if len(x_shape) == 0 or shape_index > len(x_shape) - 1 or x_shape[shape_index] == 1:
            return values.backend.cast(
                values.backend.randint(1, 127, shape=binding.spec.shape),
                binding.spec.dtype,
            )
        if len(binding.spec.shape) == 0 or binding.spec.shape[0] == 1:
            return values.backend.asarray(x_shape[shape_index])
        shape_values = values.backend.cast(
            values.backend.randint(1, 127, shape=binding.spec.shape),
            binding.spec.dtype,
        )
        offset = binding.spec.shape[0] - len(x_shape)
        for index in range(binding.spec.shape[0]):
            if index >= offset and x_shape[index - offset] != 1:
                shape_values[index] = x_shape[index - offset]
        return shape_values

    generate_by_parameter(config, values, writer, (("shape", shape_value),), default="default")


@rules.register("paddle.nn.functional.gather_tree")
def gather_tree_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    def parents_value(binding):
        sequences = config.arg(0, "sequences")
        if hasattr(sequences, "shape") and len(sequences.shape) >= 3:
            beam_size = sequences.shape[2]
        else:
            beam_size = binding.spec.shape[2] if len(binding.spec.shape) >= 3 else 4
        beam_size = 1 if beam_size < 1 else beam_size
        parents = values.backend.zeros(binding.spec.shape, dtype=binding.spec.dtype)
        for time_index in range(binding.spec.shape[0]):
            for batch_index in range(binding.spec.shape[1]):
                for beam_index in range(binding.spec.shape[2]):
                    parents[time_index, batch_index, beam_index] = values.backend.randint(
                        0, beam_size
                    )
        return parents

    generate_by_parameter(config, values, writer, (("parents", parents_value),), default="default")


@rules.register("paddle.multinomial")
def multinomial_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    x_binding = config.binding("x")
    num_samples_binding = config.binding("num_samples")
    if x_binding is not None:
        x_values = values.backend.cast(
            values.backend.abs(values.backend.random(x_binding.spec.shape)),
            x_binding.spec.dtype,
        )
        writer.set_value(x_binding, x_values)
    if num_samples_binding is not None:
        replacement = config.arg(2, "replacement")
        if config.has_kwarg("replacement") and replacement is True:
            max_allow = 1024
        else:
            x_values = writer.value(x_binding)
            max_allow = values.backend.count_nonzero(x_values > 0)
        writer.set_value(
            num_samples_binding,
            values.backend.cast(
                values.backend.randint(
                    1,
                    max_allow + 1,
                    shape=num_samples_binding.spec.shape,
                ),
                num_samples_binding.spec.dtype,
            ),
        )
    generate_remaining(config, values, writer, "default")


@rules.register("paddle.nn.functional.one_hot")
def one_hot_values(config: ConfigView, values: ValueFactory, writer: InputWriter):
    x_binding = config.binding("x")
    num_classes_binding = config.binding("num_classes")
    num_classes_config = config.arg(1, "num_classes")
    default_random_num_classes = values.backend.randint(1, 65535)
    if isinstance(num_classes_config, int):
        determined_num_classes = num_classes_config
    elif config.is_tensor_config(num_classes_config):
        if num_classes_binding is not None and num_classes_config.numel() in {0, 1}:
            writer.set_value(
                num_classes_binding,
                values.backend.asarray([default_random_num_classes], dtype="int64"),
            )
        determined_num_classes = writer.value(num_classes_binding).item()
    else:
        determined_num_classes = default_random_num_classes
    if x_binding is not None:
        writer.set_value(
            x_binding,
            values.backend.randint(
                0,
                determined_num_classes,
                shape=x_binding.spec.shape,
                dtype=x_binding.spec.dtype,
            ),
        )
    generate_remaining(config, values, writer, "default")


# Published registry snapshots and low-level config access.
INPUT_GENERATION_RULES = rules.rules
API_RULE_REGISTRY = rules.by_api
DEFAULT_INPUT_GENERATION_RULE = API_RULE_REGISTRY["paddle.add"]


def _tensor_config_at(api_config, path):
    value = api_config.args[path.key] if path.root == "args" else api_config.kwargs[path.key]
    for index in path.indices:
        value = value[index]
    return value


def _api_arg(api_config, position, name, default=None):
    if 0 <= position < len(api_config.args):
        return api_config.args[position]
    return api_config.kwargs.get(name, default)


def _apply_input_data(api_config, data: InputData, update_config):
    config = _tensor_config_at(api_config, data.path)
    config.input_value = data.value
    config.input_value_backend = data.backend
    if update_config:
        dtype_name = str(getattr(data.value, "dtype", ""))
        dtype_name = dtype_name.split(".")[-1] if dtype_name else dtype_name
        if config.dtype not in CAST_THROUGH_INTERMEDIATE_DTYPES:
            config.dtype = dtype_name
        config.shape = list(data.value.shape)
