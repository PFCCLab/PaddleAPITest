"""输入生成规则的装饰器注册中心。"""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy

from .input_backend import create_input_backend
from .input_binding import InputBinding, InputContext
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
RuleFunction = Callable[["RuleContext"], None]


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


# 规则执行分为“生成”和“提交”两个阶段。
# 规则函数只向暂存 writer 写值，不能直接修改原始 APIConfig。
# 完整性校验通过后才挂载 InputData 并推进全局 NumPy RNG。
# 这条边界保证规则抛错时不会留下半生成输入或消耗后续随机序列。
# backend 的 seed 与配置指纹则由 InputContext 传入，供原生 generator 使用。
# RuleContext 是规则作者唯一需要接触的接口，底层对象保持私有。
@dataclass(frozen=True)
class RegisteredRule:
    """一条通过装饰器注册的输入生成规则。

    这里保存的是规则的元信息和执行入口，不保存 API 约束逻辑本身。
    规则函数负责描述参数关系，`RegisteredRule` 负责完整性检查和提交。
    """

    api_names: tuple[str, ...]
    function: RuleFunction

    def generate(self, context: InputContext, api_config: object) -> bool:
        # 每个配置持有独立 RNG 副本；只有整条规则成功后才提交状态。
        rng = create_config_rng(context)
        backend = create_input_backend(rng)
        rule = RuleContext(context.call, api_config, backend)
        self.function(rule)
        # finish 先检查遗漏，再一次性同步 TensorConfig 元数据和逻辑值。
        attach_values(api_config, rule._finish())
        backend.commit()
        return True


class _InputWriter:
    """规则侧输入写入与完整性状态。"""

    def __init__(self, raw_config: object, backend):
        self._raw_config = raw_config
        self._backend = backend
        self._generated_paths = set()
        self._input_data_by_path: dict[object, InputData] = {}
        self._update_config_by_path: dict[object, bool] = {}

    def finish(self, rule):
        # 注册规则必须覆盖本次调用中的所有 Tensor，避免静默使用旧缓存。
        missing = [
            str(binding.path)
            for binding in rule.all_tensors
            if binding.path not in self._generated_paths
        ]
        if missing:
            raise ValueError(f"rule {rule.api_name} left tensors ungenerated: {missing}")
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
        # 暂存值始终由 backend 拷贝持有，隔离规则内部后续的原地修改。
        storage_value = self._backend.asarray(value, copy=True)
        self._input_data_by_path[binding.path] = InputData(
            binding.path,
            storage_value,
            self._backend.name,
        )
        self._update_config_by_path[binding.path] = update_config
        self._generated_paths.add(binding.path)


class RuleContext:
    """面向规则作者的单一输入生成接口。"""

    def __init__(self, call: InputBinding, raw_config: object, backend):
        self._call = call
        self._raw_config = raw_config
        self._backend = backend
        self._writer = _InputWriter(raw_config, backend)

    @property
    def api_name(self):
        return self._call.api_name

    @property
    def all_tensors(self):
        return self._call.tensors

    @property
    def ops(self):
        return self._backend

    def arg(self, name, default=None):
        # 优先读取签名绑定结果，使位置参数和关键字参数具有同一名称入口。
        for parameter_name, value in self._call.arguments:
            if parameter_name == name:
                return value
        return self._raw_config.kwargs.get(name, default)

    def tensor(self, parameter_name):
        # 单 Tensor 查询对多重匹配直接报错，防止规则只处理嵌套列表的首项。
        matches = self.tensors(parameter_name)
        if len(matches) > 1:
            raise ValueError(
                f"rule {self.api_name} found multiple tensors for parameter "
                f"{parameter_name!r}: {[str(tensor.path) for tensor in matches]}"
            )
        return matches[0] if matches else None

    def tensors(self, parameter_name):
        return tuple(
            tensor for tensor in self._call.tensors if tensor.parameter_name == parameter_name
        )

    def binding_for_value(self, value):
        # identity 查询用于参数值嵌套或参数名不足以区分目标的少数规则。
        if not self.is_tensor_config(value):
            return None
        for tensor in self._call.tensors:
            if _tensor_config_at(self._raw_config, tensor.path) is value:
                return tensor
        return None

    def has_kwarg(self, name):
        return name in self._raw_config.kwargs

    def kwarg(self, name, default=None):
        return self._raw_config.kwargs.get(name, default)

    def argument_values(self):
        return (*self._raw_config.args, *self._raw_config.kwargs.values())

    def is_tensor_config(self, value):
        return isinstance(value, TensorConfig)

    def domain(self, generator, tensor, low=None, high=None):
        # 值域名称在写入前集中校验，拼写错误不会产生部分输入。
        generate_value = _VALUE_GENERATORS.get(generator)
        if generate_value is None:
            raise ValueError(f"unknown input value generator {generator!r} for {tensor.path}")
        return generate_value(tensor.spec, low, high, self._backend)

    def default(self, tensor):
        return self.domain("default", tensor)

    def set(self, tensor, value):
        self._writer.set_value(tensor, value)

    def set_preserving_spec(self, tensor, value):
        # shape 参数等特殊值可改变逻辑数组形状，但调用侧仍需保留原始规格。
        self._writer.set_value_preserving_spec(tensor, value)

    def value(self, tensor):
        return self._writer.value(tensor)

    def is_generated(self, tensor):
        return self._writer.is_generated(tensor)

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

    def generate_all(self, generator="default", low=None, high=None):
        # 全量生成用于所有 Tensor 共享同一值域且不存在参数关系的规则。
        self._validate_generator(generator)
        for tensor in self.all_tensors:
            _generate_binding(self, tensor, generator, low, high)

    def generate_remaining(self, generator="default"):
        # 关系规则先写关键 Tensor，再由该入口补齐未处理的普通参数。
        self._validate_generator(generator)
        for tensor in self.all_tensors:
            if not self.is_generated(tensor):
                _generate_binding(self, tensor, generator)

    def generate(self, parameter_generators=None, *, default="default"):
        # mapping 的 key 可以是名称组，用于兼容同一语义在不同 API 中的命名。
        if parameter_generators is None:
            parameter_generators = {}
        items = (
            parameter_generators.items()
            if isinstance(parameter_generators, Mapping)
            else parameter_generators
        )

        known_names = set(self._call.parameter_names)
        normalized = []
        for parameter_names, generator in items:
            names = (
                (parameter_names,) if isinstance(parameter_names, str) else tuple(parameter_names)
            )
            if not names or any(not isinstance(name, str) or not name for name in names):
                raise ValueError("rule.generate parameter names must be non-empty strings")
            # 名称组只要求一个候选命中；单名称仍能捕获规则拼写错误。
            if known_names.isdisjoint(names) and self._call.binding_source != "unresolved":
                raise ValueError(
                    f"rule {self.api_name} declares parameters absent from its signature: "
                    f"{sorted(names)}"
                )
            self._validate_generator(generator)
            normalized.append((names, generator))

        self._validate_generator(default)

        for binding in self.all_tensors:
            # 映射按声明顺序匹配，首个命中的策略拥有该 Tensor。
            generator = None
            for names, candidate in normalized:
                if binding.parameter_name in names:
                    generator = candidate
                    break
            if generator is None:
                generator = default
            if generator is not None:
                _generate_binding(self, binding, generator)

    def _validate_generator(self, generator):
        if isinstance(generator, str) and generator not in _VALUE_GENERATORS:
            raise ValueError(f"unknown input value generator {generator!r} for {self.api_name}")

    def _numeric_dtype(self, dtype):
        return self._backend.generation_dtype(str(dtype).replace("paddle.", ""))

    @staticmethod
    def _real_dtype(dtype):
        if dtype == "complex64":
            return "float32"
        if dtype == "complex128":
            return "float64"
        return dtype

    def _finish(self):
        return self._writer.finish(self)


def _generate_binding(
    rule: RuleContext,
    binding,
    generator,
    low=None,
    high=None,
):
    if callable(generator):
        value = generator(binding)
    else:
        value = rule.domain(generator, binding, low, high)
    rule.set(binding, value)


# Registration infrastructure.
def _validate_rule_function(function: RuleFunction) -> None:
    # 导入阶段即拒绝旧三参数协议，避免执行到特定 API 时才暴露迁移遗漏。
    parameters = tuple(inspect.signature(function).parameters.values())
    if any(
        parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        for parameter in parameters
    ):
        raise TypeError(f"input-generation rule {function.__name__} cannot use variadic parameters")
    if len(parameters) != 1:
        raise TypeError(f"input-generation rule {function.__name__} must accept one RuleContext")


class RuleRegistry:
    """失败即止的装饰器注册表。"""

    def __init__(self, default_rule: RegisteredRule):
        self._default_rule = default_rule
        self._by_api: dict[str, RegisteredRule] = {}

    def register(
        self,
        *api_names: str,
        aliases: tuple[str, ...] = (),
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
            # 注册表在模块导入时完成构建，API 重叠必须立即失败。
            _validate_rule_function(function)
            for api_name in names:
                if api_name in self._by_api:
                    raise ValueError(f"input-generation API overlap: {api_name}")
            rule = RegisteredRule(
                api_names=names,
                function=function,
            )
            for api_name in names:
                self._by_api[api_name] = rule
            return function

        return decorator

    @property
    def by_api(self) -> dict[str, RegisteredRule]:
        return dict(self._by_api)

    def resolve(self, api_name: str) -> RegisteredRule:
        # 未注册 API 直接返回独立默认规则，不伪造任意 API 的注册关系。
        return self._by_api.get(api_name, self._default_rule)


# Registered API rules remain together in this file for centralized lookup.
def _default_rule(rule: RuleContext):
    """为未注册 API 使用默认值域生成全部 Tensor。"""
    rule.generate()


DEFAULT_INPUT_GENERATION_RULE = RegisteredRule(
    api_names=(),
    function=_default_rule,
)
rules = RuleRegistry(DEFAULT_INPUT_GENERATION_RULE)


@rules.register("paddle.incubate.nn.functional.fused_act_dequant")
def fused_act_dequant_values(rule: RuleContext):
    """为 int32 x_scale 生成满足重复字节编码的指数值。"""

    def x_scale_value(tensor):
        if tensor.dtype == "int32":
            exponent = rule.ops.randint(120, 128, shape=tensor.shape, dtype="int32")
            return exponent * rule.ops.asarray(0x01010101, dtype="int32")
        return rule.default(tensor)

    rule.generate({"x_scale": x_scale_value})


@rules.register("paddle.incubate.nn.functional.variable_length_memory_efficient_attention")
def variable_length_memory_efficient_attention_values(rule: RuleContext):
    """约束序列长度、KV 长度和注意力 mask 的有效范围。"""

    def seq_lens_value(binding):
        query = rule.arg("query")
        q_seq_len = query.shape[2]
        return rule.domain("random_range", binding, 1, q_seq_len)

    def kv_seq_lens_value(binding):
        key = rule.arg("key")
        value = rule.arg("value")
        max_seq_len = min(key.shape[2], value.shape[2])
        return rule.domain("random_range", binding, 1, max_seq_len)

    def mask_value(binding):
        return rule.ops.cast(
            rule.ops.randint(0, 2, shape=binding.shape),
            binding.dtype,
        ) * rule.dtype_min(binding.dtype)

    rule.generate(
        (
            ("seq_lens", seq_lens_value),
            ("kv_seq_lens", kv_seq_lens_value),
            ("mask", mask_value),
        ),
    )


@rules.register(
    "paddle.incubate.nn.functional.block_multihead_attention",
)
def block_multihead_attention_values(rule: RuleContext):
    """联动生成缓存、序列长度、padding offset 和量化参数。"""
    qkv = rule.arg("qkv")
    seq_lens_encoder = rule.arg("seq_lens_encoder")
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
        return rule.ops.asarray([seq_len] * batch_size, dtype=binding.dtype)

    def set_padding_offsets(binding):
        seq_lens_this_time = rule.value(rule.tensor("seq_lens_this_time"))
        cum_offsets_now = rule.ops.cumsum(seq_len - seq_lens_this_time)
        cum_offsets_binding = rule.tensor("cum_offsets")
        cu_seqlens_q_binding = rule.tensor("cu_seqlens_q")
        cu_seqlens_k_binding = rule.tensor("cu_seqlens_k")
        cum_offsets = rule.ops.zeros((batch_size + 1,), dtype=cum_offsets_binding.dtype)
        cum_offsets[1:] = cum_offsets_now
        token_num = rule.ops.sum(seq_lens_this_time)
        padding_offsets = rule.ops.zeros((token_num,), dtype=binding.dtype)
        cu_seqlens_q = rule.ops.zeros((batch_size + 1,), dtype=cu_seqlens_q_binding.dtype)
        cu_seqlens_k = rule.ops.zeros((batch_size + 1,), dtype=cu_seqlens_k_binding.dtype)
        for batch_index in range(batch_size):
            seq_len_now = int(seq_lens_this_time[batch_index])
            cum_offset = int(cum_offsets[batch_index])
            for token_index in range(seq_len_now):
                padding_offsets[batch_index * seq_len - cum_offset + token_index] = cum_offset
            cum_seq_len = (batch_index + 1) * seq_len - cum_offsets[batch_index + 1]
            cu_seqlens_q[batch_index + 1] = cum_seq_len
            cu_seqlens_k[batch_index + 1] = cum_seq_len
        rule.set(cum_offsets_binding, cum_offsets[:-1])
        rule.set(cu_seqlens_q_binding, cu_seqlens_q)
        rule.set(cu_seqlens_k_binding, cu_seqlens_k)
        rule.set(binding, padding_offsets)

    for binding in rule.all_tensors:
        if rule.is_generated(binding):
            continue
        if binding.parameter_name in zero_parameters:
            rule.set(binding, rule.ops.zeros(binding.shape, dtype=binding.dtype))
        elif binding.parameter_name == "seq_lens_encoder":
            rule.set(binding, seq_len_array(binding))
        elif binding.parameter_name == "seq_lens_this_time":
            rule.set(binding, rule.value(rule.tensor("seq_lens_encoder")))
        elif binding.parameter_name == "padding_offsets":
            set_padding_offsets(binding)
        elif binding.parameter_name in positive_range_parameters:
            rule.set(binding, rule.domain("random_range", binding, low=0))
        elif binding.parameter_name == "max_enc_len_this_time":
            rule.set(binding, seq_len_array(binding))
        elif binding.parameter_name in {"mask", "tgt_mask"}:
            rule.set(
                binding,
                rule.domain("random_range", binding, high=rule.dtype_eps(binding.dtype)),
            )
        else:
            rule.set(binding, rule.default(binding))


@rules.register(
    "paddle._C_ops.adam_",
    "paddle._C_ops.adamw_",
    "paddle._C_ops.merged_adam_",
)
def optimizer_values(rule: RuleContext):
    """初始化优化器状态，并按 beta 与随机步数计算幂累计值。"""
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
            beta_pow = rule.ops.power(
                numpy.float32(beta),
                numpy.float32(int(step)),
            )
            beta_pow = beta_pow.item() if hasattr(beta_pow, "item") else beta_pow
        return rule.ops.full(
            binding.shape,
            beta_pow,
            dtype=binding.dtype,
        )

    def generate_value(binding):
        nonlocal optimizer_step
        if binding.parameter_name in zero_parameters:
            return rule.ops.zeros(binding.shape, dtype=binding.dtype)
        if rule.api_name == "paddle._C_ops.adamw_" and binding.parameter_name in {
            "beta1_pow",
            "beta2_pow",
        }:
            if optimizer_step is None:
                optimizer_step = rule.ops.randint(1, 101)
            beta = rule.arg("beta1")
            if binding.parameter_name == "beta2_pow":
                beta = rule.arg("beta2")
            return beta_pow_value(binding, beta, optimizer_step)
        return rule.default(binding)

    rule.generate_all(generate_value)


@rules.register(
    "paddle.nn.functional.max_unpool1d",
    "paddle.nn.functional.max_unpool2d",
    "paddle.nn.functional.max_unpool3d",
)
def max_unpool_values(rule: RuleContext):
    """根据池化参数同步构造输入值和合法的反池化索引。"""

    def resolve_parameters(x_shape, output_size, kernel_size, stride, padding):
        dimensions = {
            "paddle.nn.functional.max_unpool1d": 1,
            "paddle.nn.functional.max_unpool2d": 2,
            "paddle.nn.functional.max_unpool3d": 3,
        }[rule.api_name]
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
                f"invalid output_size for {rule.api_name}, len(output_size) should be "
                f"{dimensions} or {len(x_shape)} or output_size == None, got "
                f"len(output_size)={len(output_size)} and output_size={output_size}"
            )
        return kernel_size, stride, padding, pool_input_size

    x_binding = rule.tensor("x")
    indices_binding = rule.tensor("indices")
    if x_binding is None or indices_binding is None:
        raise ValueError(f"rule {rule.api_name} requires x and indices tensors")

    kernel_size = rule.arg("kernel_size")
    stride = rule.arg("stride")
    padding = rule.arg("padding")
    output_size = rule.arg("output_size")
    kernel_size, stride, padding, pool_input_size = resolve_parameters(
        x_binding.shape,
        output_size,
        kernel_size,
        stride,
        padding,
    )
    data_type = "float64" if x_binding.dtype == "int64" else x_binding.dtype
    pool_input_spec = TensorSpec(
        shape=tuple(pool_input_size),
        dtype=data_type,
        place=x_binding.place,
        is_contiguous=x_binding.is_contiguous,
        strides=x_binding.strides,
    )
    pool_input = generate_random_range(pool_input_spec, low=-5, high=5, rng=rule.ops)
    pool_name = rule.api_name.rsplit(".", 1)[-1].replace("max_unpool", "max_pool")
    if rule.ops.name == "torch":
        import torch.nn.functional as torch_functional

        max_poolxd_func = getattr(torch_functional, pool_name)
        x, indices = max_poolxd_func(
            pool_input,
            kernel_size,
            stride,
            padding,
            return_indices=True,
        )
        rule.set(x_binding, x)
        rule.set(indices_binding, indices)
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
    if rule.ops.name == "paddle":
        rule.set(x_binding, x)
        rule.set(indices_binding, indices)
        return
    rule.set(x_binding, x.numpy())
    rule.set(indices_binding, indices.numpy())


@rules.register("paddle.arange")
def arange_values(rule: RuleContext):
    """联合约束 start、end、step，避免空区间和零步长。"""

    def tensor_binding(value):
        return rule.binding_for_value(value)

    def set_tensor(value, tensor_value):
        binding = tensor_binding(value)
        if binding is not None:
            rule.set(binding, tensor_value)

    def generate_step_tensor(step_config, is_positive):
        if "int" in step_config.dtype:
            if is_positive:
                return rule.ops.cast(
                    rule.ops.randint(1, 10, shape=step_config.shape),
                    step_config.dtype,
                )
            return rule.ops.cast(
                rule.ops.randint(-10, -1, shape=step_config.shape),
                step_config.dtype,
            )
        if is_positive:
            return rule.ops.cast(
                rule.ops.uniform(0.1, 5.0, shape=step_config.shape),
                step_config.dtype,
            )
        return rule.ops.cast(
            rule.ops.uniform(-5.0, -0.1, shape=step_config.shape),
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
            return rule.ops.cast(
                rule.ops.randint(low, high, shape=tensor_config.shape),
                tensor_config.dtype,
            )
        return rule.ops.cast(
            rule.ops.uniform(low, high, shape=tensor_config.shape),
            tensor_config.dtype,
        )

    def handle_arange_relation():
        start_val = rule.arg("start", 0)
        end_val = rule.arg("end", None)
        step_val = rule.arg("step", 1)

        if rule.is_tensor_config(start_val):
            if rule.is_tensor_config(end_val):
                if rule.is_tensor_config(step_val):
                    flag = rule.ops.choice([True, False])
                    set_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                set_tensor(start_val, random_range(start_val, -50, 50))
                start = rule.value(rule.tensor("start")).item()
                if flag:
                    low, high = safe_range(start + 1, start + 50)
                else:
                    low, high = safe_range(start - 50, start - 1)
                set_tensor(end_val, random_range(end_val, low, high))
            elif end_val is None:
                if rule.is_tensor_config(step_val):
                    flag = rule.ops.choice([True, False])
                    set_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                if flag:
                    if "int" in start_val.dtype:
                        value = rule.ops.cast(
                            rule.ops.randint(1, 50, shape=start_val.shape),
                            start_val.dtype,
                        )
                    else:
                        value = rule.ops.cast(
                            rule.ops.uniform(0.1, 50.0, shape=start_val.shape),
                            start_val.dtype,
                        )
                elif "int" in start_val.dtype:
                    value = rule.ops.cast(
                        rule.ops.randint(-50, -1, shape=start_val.shape),
                        start_val.dtype,
                    )
                else:
                    value = rule.ops.cast(
                        rule.ops.uniform(-50.0, -0.1, shape=start_val.shape),
                        start_val.dtype,
                    )
                set_tensor(start_val, value)
            else:
                if rule.is_tensor_config(step_val):
                    flag = rule.ops.choice([True, False])
                    set_tensor(step_val, generate_step_tensor(step_val, flag))
                else:
                    flag = step_val > 0
                if flag:
                    low, high = safe_range(end_val - 50, end_val - 1)
                else:
                    low, high = safe_range(end_val + 1, end_val + 50)
                set_tensor(start_val, random_range(start_val, low, high))
        elif rule.is_tensor_config(end_val):
            if rule.is_tensor_config(step_val):
                flag = rule.ops.choice([True, False])
                set_tensor(step_val, generate_step_tensor(step_val, flag))
            else:
                flag = step_val > 0
            if flag:
                low, high = safe_range(start_val + 1, start_val + 50)
            else:
                low, high = safe_range(start_val - 50, start_val - 1)
            set_tensor(end_val, random_range(end_val, low, high))
        elif end_val is None:
            if rule.is_tensor_config(step_val):
                flag = start_val > 0
                set_tensor(step_val, generate_step_tensor(step_val, flag))
        elif rule.is_tensor_config(step_val):
            flag = start_val < end_val
            set_tensor(step_val, generate_step_tensor(step_val, flag))

    for binding in rule.all_tensors:
        if rule.value(binding) is None:
            handle_arange_relation()


@rules.register("paddle.nn.functional.moe_permute")
def moe_permute_values(rule: RuleContext):
    """按专家路由关系生成 MoE 排列所需的映射和概率。"""

    def expert_routemap_value(binding):
        num_experts = rule.arg("num_experts", 32)
        hidden_states = rule.arg("hidden_states")
        scale = rule.arg("scale")
        expert_prob = rule.arg("expert_prob_topk")
        tokens_per_expert = rule.arg("tokens_per_expert")
        padding_alignment = rule.arg("padding_alignment")
        using_ue8m0_scale = rule.arg("using_ue8m0_scale", False)
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
        if not rule.is_tensor_config(hidden_states) or (
            len(hidden_states.shape) != 2
            or hidden_states.dtype not in {"bfloat16", "float32", "float8_e4m3fn"}
        ):
            raise ValueError("hidden_states must be a rank-2 bfloat16 or float8_e4m3fn tensor")
        if binding.dtype != "int32":
            raise ValueError("expert_routemap_topk dtype must be int32")
        if not rule.is_tensor_config(expert_prob) or (
            len(expert_prob.shape) != 2 or expert_prob.dtype != "float32"
        ):
            raise ValueError("expert_prob_topk must be a rank-2 float32 tensor")
        seqlen, topk = binding.shape[0], binding.shape[1]
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
                rule.is_tensor_config(scale)
                and tuple(scale.shape) == (seqlen, expected_scale_width)
                and scale.dtype == expected_scale_dtype
            ):
                raise ValueError(
                    "float8 hidden_states requires scale with shape "
                    f"[{seqlen}, {expected_scale_width}] and dtype {expected_scale_dtype}"
                )
        elif scale is not None:
            raise ValueError("scale must be None when hidden_states dtype is bfloat16")
        routemap = rule.ops.full((seqlen, topk), -1, dtype="int32")
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
            positions = rule.ops.arange(cursor, cursor + count, dtype="int64")
            rows = positions % seqlen
            columns = (positions // seqlen) % topk
            routemap[rows, columns] = rule.ops.cast(
                rule.ops.full(rows.shape, expert, dtype="int32"), "int32"
            )
            cursor += count
        return routemap

    def expert_prob_value(binding):
        routemap_binding = rule.tensor("expert_routemap_topk")
        probs = rule.ops.zeros(binding.shape, dtype="float32")
        if routemap_binding is not None and rule.value(routemap_binding) is not None:
            mask = rule.value(routemap_binding) >= 0
            raw = rule.ops.cast(rule.ops.random(binding.shape), "float32") * mask
            row_sums = rule.ops.sum(raw, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = raw / row_sums
        else:
            probs = rule.ops.cast(rule.ops.random(binding.shape), "float32")
            row_sums = rule.ops.sum(probs, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums
        return probs

    rule.generate(
        (
            ("expert_routemap_topk", expert_routemap_value),
            ("expert_prob_topk", expert_prob_value),
        ),
    )


@rules.register("paddle.nn.functional.moe_unpermute")
def moe_unpermute_values(rule: RuleContext):
    """按压缩行映射恢复 MoE 反排列输入之间的形状关系。"""

    def expert_routemap_value(binding):
        num_experts = rule.arg("num_experts", 32)
        total_zipped_tokens = rule.arg("total_zipped_tokens")
        hidden_config = rule.arg("hidden_states_unzipped")
        rowmap_config = rule.arg("zipped_expertwise_rowmap")
        prob_config = rule.arg("token_prob_unzipped")
        if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if (
            not isinstance(total_zipped_tokens, int)
            or isinstance(total_zipped_tokens, bool)
            or total_zipped_tokens < 0
        ):
            raise ValueError("total_zipped_tokens must be a non-negative integer")
        if not (
            rule.is_tensor_config(hidden_config)
            and len(hidden_config.shape) == 2
            and hidden_config.dtype in {"bfloat16", "float32"}
        ):
            raise ValueError("hidden_states_unzipped must be a rank-2 bfloat16 tensor")
        if not (
            rule.is_tensor_config(rowmap_config)
            and len(rowmap_config.shape) == 2
            and rowmap_config.dtype == "int32"
            and tuple(rowmap_config.shape) == (total_zipped_tokens, num_experts)
        ):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        if not (
            rule.is_tensor_config(prob_config)
            and len(prob_config.shape) in (1, 2)
            and prob_config.shape[0] == hidden_config.shape[0]
            and (len(prob_config.shape) == 1 or prob_config.shape[1] == 1)
            and prob_config.dtype == "float32"
        ):
            raise ValueError(
                "token_prob_unzipped must have shape "
                "[seqlen_broadcasted] or [seqlen_broadcasted, 1] and dtype float32"
            )
        if binding.dtype != "int32" or len(binding.shape) != 2:
            raise ValueError("expert_routemap_topk must be a rank-2 int32 tensor")
        seqlen, topk = binding.shape[0], binding.shape[1]
        if seqlen != total_zipped_tokens:
            raise ValueError("expert_routemap_topk sequence length must equal total_zipped_tokens")
        if topk <= 0:
            raise ValueError("topk should be greater than 0")
        routemap = rule.ops.full(binding.shape, -1, dtype="int32")
        max_assign = min(topk, num_experts)
        route_count = min(hidden_config.shape[0], seqlen * max_assign)
        positions = rule.ops.arange(route_count, dtype="int64")
        rows = positions % seqlen
        columns = positions // seqlen
        routemap[rows, columns] = rule.ops.cast((rows + columns) % num_experts, "int32")
        return routemap

    def rowmap_value(binding):
        routemap_binding = rule.tensor("expert_routemap_topk")
        if routemap_binding is not None and not rule.is_generated(routemap_binding):
            rule.set(routemap_binding, expert_routemap_value(routemap_binding))
        routemap_config = rule.arg("expert_routemap_topk")
        num_experts = rule.arg("num_experts", 32)
        total_zipped_tokens = rule.arg("total_zipped_tokens")
        hidden_config = rule.arg("hidden_states_unzipped")
        seqlen = total_zipped_tokens
        unzipped_seqlen = hidden_config.shape[0] if rule.is_tensor_config(hidden_config) else seqlen
        if binding.dtype != "int32" or tuple(binding.shape) != (seqlen, num_experts):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        rowmap = rule.ops.full(binding.shape, -1, dtype="int32")
        if rule.is_tensor_config(routemap_config) and routemap_binding is not None:
            routemap = rule.value(routemap_binding)
            expert_counts = rule.ops.asarray(
                [rule.ops.count_nonzero(routemap == expert) for expert in range(num_experts)],
                dtype="int64",
            )
            if int(rule.ops.sum(expert_counts)) > unzipped_seqlen:
                raise ValueError("routemap assignments exceed hidden_states_unzipped capacity")
            expert_offsets = rule.ops.zeros(num_experts, dtype="int64")
            expert_offsets[1:] = rule.ops.cumsum(expert_counts[:-1])
            expert_counters = rule.ops.zeros(num_experts, dtype="int64")
            for row_index in range(seqlen):
                for expert in range(num_experts):
                    positions = rule.ops.nonzero(routemap[row_index] == expert)[0]
                    if rule.ops.prod(positions.shape) == 0:
                        continue
                    rowmap[row_index, expert] = rule.ops.cast(
                        expert_offsets[expert] + expert_counters[expert], "int32"
                    )
                    expert_counters[expert] += 1
        return rowmap

    def token_prob_value(binding):
        hidden_config = rule.arg("hidden_states_unzipped")
        if not (
            binding.dtype == "float32"
            and len(binding.shape) in (1, 2)
            and rule.is_tensor_config(hidden_config)
            and binding.shape[0] == hidden_config.shape[0]
            and (len(binding.shape) == 1 or binding.shape[1] == 1)
        ):
            raise ValueError(
                "token_prob_unzipped must match the broadcasted sequence "
                "length and have dtype float32"
            )
        return rule.ops.cast(rule.ops.random(binding.shape), "float32")

    for binding in rule.all_tensors:
        if rule.is_generated(binding):
            continue
        if binding.parameter_name == "expert_routemap_topk":
            rule.set(binding, expert_routemap_value(binding))
        elif binding.parameter_name == "zipped_expertwise_rowmap":
            rule.set(binding, rowmap_value(binding))
        elif binding.parameter_name == "token_prob_unzipped":
            rule.set(binding, token_prob_value(binding))
        else:
            rule.set(binding, rule.default(binding))


@rules.register(*tuple(sorted(not_zero_apis)))
def nonzero_values(rule: RuleContext):
    """为除法及对数类算子生成非零输入。"""
    rule.generate_all("nonzero")


@rules.register("paddle.bernoulli")
def bernoulli_probability(rule: RuleContext):
    """将伯努利分布概率限制在单位区间。"""
    rule.generate_all("unit_interval")


@rules.register("paddle.standard_gamma")
def standard_gamma_unit_interval(rule: RuleContext):
    """为 Gamma 分布生成单位区间内的正参数。"""
    rule.generate_all("unit_interval")


@rules.register("paddle.poisson")
def poisson_unit_interval(rule: RuleContext):
    """为 Poisson 分布生成非负强度参数。"""
    rule.generate_all("unit_interval")


@rules.register(
    "paddle.sqrt",
    aliases=("paddle.Tensor.sqrt",),
)
def sqrt_nonnegative(rule: RuleContext):
    """为平方根生成非负输入。"""
    rule.generate_all("uniform", low=0, high=1000)


@rules.register(
    "paddle.rsqrt",
    aliases=("paddle.Tensor.rsqrt",),
)
def rsqrt_positive(rule: RuleContext):
    """为倒数平方根生成严格为正的输入。"""
    rule.generate_all("uniform", low=1e-7, high=1000)


@rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def clip_values(rule: RuleContext):
    """联动 min 与 max，保证裁剪上下界有序。"""
    x_binding = rule.tensor("x")
    min_binding = rule.tensor("min")
    max_binding = rule.tensor("max")
    min_config = rule.arg("min")
    max_config = rule.arg("max")

    if rule.is_tensor_config(min_config) and rule.is_tensor_config(max_config):
        min_value = rule.domain("random_range", min_binding)
        max_value = rule.domain("random_range", max_binding, low=min_value)
        rule.set(min_binding, min_value)
        rule.set(max_binding, max_value)
    elif min_config is not None and max_config is not None:
        if rule.is_tensor_config(min_config) and isinstance(max_config, (int, float)):
            min_value = rule.domain("random_range", min_binding, high=max_config)
            rule.set(min_binding, min_value)
        elif rule.is_tensor_config(max_config) and isinstance(min_config, (int, float)):
            max_value = rule.domain("random_range", max_binding, low=min_config)
            rule.set(max_binding, max_value)

    if x_binding is not None:
        rule.set(
            x_binding,
            rule.domain("random_range", x_binding),
        )
    rule.generate_remaining()


@rules.register(
    "paddle.multiply",
    aliases=("paddle.Tensor.__mul__", "paddle.Tensor.multiply", "paddle.Tensor.__rmul__"),
)
def multiply_values(rule: RuleContext):
    """收紧乘法输入值域以降低溢出风险。"""
    rule.generate_all("multiply")


@rules.register(
    "paddle.nn.functional.binary_cross_entropy",
)
def binary_cross_entropy_values(rule: RuleContext):
    """将二元交叉熵输入限制在概率区间。"""
    rule.generate_all("unit_interval")


@rules.register("paddle.nn.functional.alpha_dropout")
def alpha_dropout_values(rule: RuleContext):
    """为 alpha dropout 生成符合概率语义的输入。"""
    rule.generate((("x", "unit_interval"),))


@rules.register("paddle.nn.functional.conv2d_transpose")
def conv2d_transpose_values(rule: RuleContext):
    """为转置卷积生成受控范围内的输入、权重和偏置。"""

    def tensor_value(binding):
        if "int" in binding.dtype:
            return rule.ops.cast(
                rule.ops.randint(-65535, 65535, shape=binding.shape),
                binding.dtype,
            )
        return rule.ops.cast(
            rule.ops.random(binding.shape) - 0.5,
            binding.dtype,
        )

    rule.generate(
        ((("x", "weight", "bias"), tensor_value),),
    )


@rules.register("paddle.vision.ops.distribute_fpn_proposals")
def distribute_fpn_proposals_values(rule: RuleContext):
    """生成有效 ROI 坐标并保持各层 rois_num 总数一致。"""
    state = {"num": None}

    def fpn_rois_value(binding):
        num = binding.shape[0]
        state["num"] = num
        rois = rule.ops.randint(1, 1024, shape=[num, 4])
        rois[:, 0] = rois[:, 0] + rule.ops.random([num])
        rois[:, 1] = rois[:, 1] + rule.ops.random([num])
        rois[:, 2] = rois[:, 0] + rule.ops.randint(1, 1024, shape=[num]) + rule.ops.random([num])
        rois[:, 3] = rois[:, 1] + rule.ops.randint(1, 1024, shape=[num]) + rule.ops.random([num])
        return rois

    def rois_num_value(binding):
        if state["num"] is None:
            fpn_rois = rule.arg("fpn_rois")
            state["num"] = fpn_rois.shape[0]
        num = state["num"]
        remaining = binding.shape[0]
        result = rule.ops.zeros(binding.shape)
        if num > 4096 or remaining > 4096:
            if num < remaining:
                result[:num] = 1
            else:
                result += num // remaining
                result[: num % remaining] += 1
        elif num < remaining:
            indices = rule.ops.choice(remaining, num, replace=False)
            result[indices] = 1
        else:
            for index in range(binding.shape[0] - 1):
                result[index] = rule.ops.randint(1, num - remaining + 2)
                num -= result[index]
                remaining -= 1
            result[binding.shape[0] - 1] = num
        return result

    rule.generate(
        (
            ("fpn_rois", fpn_rois_value),
            ("rois_num", rois_num_value),
        ),
    )


@rules.register("paddle.vision.ops.generate_proposals")
def generate_proposals_values(rule: RuleContext):
    """生成合法图像尺寸、anchor 和 proposal 分数输入。"""

    def random_value(binding):
        return rule.ops.random(binding.shape, dtype=binding.dtype)

    def img_size_value(binding):
        return rule.ops.cast(
            rule.ops.randint(0, 1024, shape=binding.shape),
            binding.dtype,
        )

    def anchors_value(binding):
        anchors = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        width = binding.shape[0]
        height = binding.shape[1]
        for index in range(binding.shape[0]):
            anchors[index][0] = rule.ops.random() * width
            anchors[index][1] = rule.ops.random() * height
            anchors[index][2] = (
                rule.ops.random() * (width - anchors[index][0] + 1) + anchors[index][0] + 1
            )
            anchors[index][3] = (
                rule.ops.random() * (height - anchors[index][1] + 1) + anchors[index][1] + 1
            )
        return anchors

    for binding in rule.all_tensors:
        if binding.parameter_name in {"scores", "bbox_deltas"}:
            rule.set(binding, random_value(binding))
        elif binding.parameter_name == "img_size":
            rule.set(binding, img_size_value(binding))
        elif binding.parameter_name == "anchors":
            rule.set(binding, anchors_value(binding))
        else:
            rule.set(binding, rule.default(binding))


@rules.register("paddle.vision.ops.nms")
def nms_values(rule: RuleContext):
    """生成坐标有序的候选框和对应分数。"""

    def boxes_value(binding):
        boxes = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        for index in range(binding.shape[0]):
            boxes[index][0] = rule.ops.random() * 1023
            boxes[index][1] = rule.ops.random() * 1023
            boxes[index][2] = rule.ops.random() * (1024 - boxes[index][0] + 1) + boxes[index][0] + 1
            boxes[index][3] = rule.ops.random() * (1024 - boxes[index][1] + 1) + boxes[index][1] + 1
        return boxes

    def scores_value(binding):
        return rule.ops.random(binding.shape, dtype=binding.dtype)

    def default_vision_value(binding):
        return rule.ops.cast(
            rule.ops.randint(0, 1024, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
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
def roi_pool_values(rule: RuleContext):
    """按特征图和 ROI 数量约束池化坐标及批次索引。"""
    state = {"x_shape": None, "boxes_shape": None}

    def x_value(binding):
        state["x_shape"] = binding.shape
        return rule.ops.cast(
            rule.ops.random(binding.shape) * 255,
            binding.dtype,
        )

    def boxes_value(binding):
        if state["x_shape"] is None:
            x = rule.arg("x")
            state["x_shape"] = x.shape
        state["boxes_shape"] = binding.shape
        boxes = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        for index in range(binding.shape[0]):
            boxes[index][0] = rule.ops.random() * (state["x_shape"][2] - 2)
            boxes[index][1] = rule.ops.random() * (state["x_shape"][3] - 2)
            boxes[index][2] = (
                rule.ops.random() * (state["x_shape"][2] - 1 - boxes[index][0] + 1)
                + boxes[index][0]
                + 1
            )
            boxes[index][3] = (
                rule.ops.random() * (state["x_shape"][3] - 1 - boxes[index][1] + 1)
                + boxes[index][1]
                + 1
            )
        return boxes

    def boxes_num_value(binding):
        if state["boxes_shape"] is None:
            boxes = rule.arg("boxes")
            state["boxes_shape"] = boxes.shape
        boxes_remaining = state["boxes_shape"][0]
        result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        numel = math.prod(binding.shape)
        for index in range(numel - 1):
            if boxes_remaining < numel:
                result[index] = 0
            else:
                result[index] = rule.ops.randint(1, boxes_remaining - (numel - 1 - index) + 1)
                boxes_remaining -= result[index]
        result[numel - 1] = boxes_remaining
        return result

    rule.generate(
        (
            ("x", x_value),
            ("boxes", boxes_value),
            ("boxes_num", boxes_num_value),
        ),
    )


@rules.register(
    "paddle.gammainc",
    "paddle.gammaincc",
    "paddle.linspace",
)
def zero_65535_or_unit_values(rule: RuleContext):
    """按整数或浮点 dtype 选择计数域与单位区间。"""
    rule.generate_all("int_zero_65535_else_unit")


@rules.register("paddle.dot")
def dot_values(rule: RuleContext):
    """限制点积输入幅度以减少累加溢出。"""
    rule.generate_all("int_minus127_127_else_default")


@rules.register("paddle.normal")
def normal_values(rule: RuleContext):
    """分别约束正态分布的 mean、std 和 shape 输入。"""
    rule.generate(
        (
            ("mean", "signed_half_interval"),
            ("std", "normal_std"),
        ),
        default="int_zero_1024",
    )


@rules.register("paddle.ones")
def ones_shape(rule: RuleContext):
    """为 ones 的 Tensor shape 参数生成正整数。"""
    rule.generate_all("ones_shape")


@rules.register("paddle.zeros")
def zeros_shape(rule: RuleContext):
    """为 zeros 的 Tensor shape 参数生成非负整数。"""
    rule.generate_all("int_zero_2048_no_cast")


@rules.register("paddle.eye")
def eye_shape(rule: RuleContext):
    """为 eye 的行列 Tensor 参数生成非负整数。"""
    rule.generate_all("int_zero_2048_no_cast")


@rules.register(
    "paddle.nn.functional.interpolate",
    "paddle.Tensor.tile",
    "paddle.tile",
)
def shape_parameter_values(rule: RuleContext):
    """为插值和 tile 的动态尺寸参数生成正整数。"""
    rule.generate(
        ((("size", "scale_factor", "repeat_times"), "int_one_128"),),
    )


@rules.register("paddle.nn.functional.upsample")
def upsample_values(rule: RuleContext):
    """分别约束上采样 size 与 scale_factor。"""
    rule.generate(
        (
            ("size", "int_one_128"),
            ("scale_factor", "abs_unit_plus_one"),
        ),
    )


@rules.register(
    "paddle.nn.functional.gaussian_nll_loss",
)
def gaussian_nll_loss_values(rule: RuleContext):
    """保证高斯 NLL 的方差输入严格为正。"""
    rule.generate(
        ((("var", "variance"), "unit_interval_plus_one"),),
    )


@rules.register(
    "paddle.nn.functional.hinge_embedding_loss",
)
def hinge_embedding_loss_values(rule: RuleContext):
    """为 hinge embedding loss 生成合法标签。"""
    rule.generate((("label", "hinge_labels"),))


@rules.register(
    "paddle.nn.functional.sigmoid_focal_loss",
)
def sigmoid_focal_loss_values(rule: RuleContext):
    """为 sigmoid focal loss 生成二值标签。"""
    rule.generate((("label", "binary_0_1"),))


@rules.register("paddle.full")
def full_values(rule: RuleContext):
    """按 shape 与 fill_value 的语义选择整数值域。"""
    rule.generate(
        (
            ("shape", "int_zero_64"),
            ("fill_value", "full_fill_value"),
        ),
        default="int_zero_64",
    )


@rules.register("paddle.standard_normal")
def standard_normal_shape(rule: RuleContext):
    """为标准正态分布的动态 shape 生成正整数。"""
    rule.generate((("shape", "int_one_128"),))


@rules.register("paddle.logspace")
def logspace_values(rule: RuleContext):
    """限制 logspace 的采样数量为正整数。"""
    rule.generate((("num", "int_one_65535_no_cast"),))


@rules.register("paddle.quantile")
def quantile_values(rule: RuleContext):
    """将分位点 q 限制在合法区间。"""
    rule.generate((("q", "quantile_q"),))


@rules.register(
    "paddle.remainder",
    aliases=("paddle.Tensor.remainder",),
)
def remainder_values(rule: RuleContext):
    """为余数运算生成非零右操作数。"""
    rule.generate((("y", "remainder_rhs"),))


@rules.register(
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
)
def dropout_values(rule: RuleContext):
    """将 dropout 概率限制在合法区间。"""
    rule.generate((("p", "dropout_probability"),))


@rules.register("paddle.atan2")
def atan2_values(rule: RuleContext):
    """避开 atan2 在原点处的不确定输入。"""
    rule.generate_all("unit_interval_plus_one")


@rules.register("paddle.bincount")
def bincount_values(rule: RuleContext):
    """生成非负计数索引和相容权重。"""

    def integer_value(binding):
        return rule.ops.cast(
            rule.ops.randint(0, 65535, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            ("x", integer_value),
            ("minlength", integer_value),
        ),
    )


@rules.register(
    "paddle.nn.functional.adaptive_avg_pool2d", "paddle.nn.functional.adaptive_avg_pool3d"
)
def adaptive_avg_pool_values(rule: RuleContext):
    """根据输入 rank 生成合法自适应池化输出尺寸。"""

    def output_size(binding):
        x_shape = rule.arg("x").shape
        return rule.ops.cast(
            rule.ops.randint(1, 2 * max(x_shape), shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (("output_size", output_size),),
    )


@rules.register("paddle.empty")
def empty_values(rule: RuleContext):
    """为 empty 的 Tensor shape 参数生成空尺寸边界值。"""
    rule.generate((("shape", "empty_shape"),))


@rules.register(
    "paddle.repeat_interleave",
    aliases=("paddle.Tensor.repeat_interleave",),
)
def repeat_interleave_values(rule: RuleContext):
    """约束 repeats 为正整数并生成有效 axis。"""

    def axis_value(binding):
        x = rule.arg("x")
        input_dims = len(x.shape)
        if len(binding.shape) == 0:
            return rule.ops.asarray(rule.ops.randint(-input_dims, input_dims), dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(-input_dims, input_dims, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            ("repeats", "int_one_2048"),
            ("axis", axis_value),
        ),
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
def put_along_axis_values(rule: RuleContext):
    """按输入 shape 与 axis 生成合法 indices 和相容 values。"""

    def random_tensor_value(binding, shape):
        scalar_spec = TensorSpec(
            shape=tuple(shape),
            dtype=binding.dtype,
            place=binding.place,
            is_contiguous=binding.is_contiguous,
            strides=binding.strides,
        )
        return generate_random_range(scalar_spec, rng=rule.ops)

    def indices_value(binding):
        x_tensor = rule.arg("arr", rule.arg("x"))
        x_shape = tuple(x_tensor.shape) if x_tensor is not None else ()
        x_dims = len(x_shape)
        current_shape = tuple(binding.shape)
        if len(current_shape) != x_dims:
            new_shape = [current_shape[i] if i < len(current_shape) else 1 for i in range(x_dims)]
            indices = rule.ops.zeros(new_shape, dtype="int64")
            for axis in range(x_dims):
                if axis < len(current_shape):
                    dim_size = x_shape[axis]
                    if dim_size > 0:
                        axis_indices = rule.ops.choice(
                            dim_size, shape=new_shape[axis], replace=False
                        )
                        axis_indices = rule.ops.cast(axis_indices, "int64")
                        idx_tuple = tuple(
                            [slice(None)] * axis
                            + [slice(None, new_shape[axis])]
                            + [slice(None)] * (x_dims - axis - 1)
                        )
                        indices[idx_tuple] = rule.ops.reshape(
                            axis_indices,
                            [-1] + [1] * (x_dims - axis - 1),
                        )
            return indices
        axis = rule.arg("axis", 0)
        axis = axis if isinstance(axis, int) else 0
        axis = axis if axis >= 0 else axis + x_dims
        indices = rule.ops.zeros(current_shape, dtype="int64")
        if 0 <= axis < x_dims:
            dim_size = x_shape[axis]
            for idx in rule.ops.ndindex(tuple(current_shape[:-1])):
                indices[idx] = rule.ops.choice(dim_size, shape=current_shape[-1], replace=False)
        return indices

    def write_values(binding):
        indices_binding = rule.tensor("indices")
        if indices_binding is not None:
            indices = rule.value(indices_binding)
            if tuple(indices.shape) != tuple(binding.shape):
                if rule.ops.prod(binding.shape) == 1:
                    rule.set_preserving_spec(
                        binding,
                        rule.ops.full(
                            indices.shape,
                            random_tensor_value(binding, ())[()],
                            dtype=binding.dtype,
                        ),
                    )
                else:
                    rule.set_preserving_spec(
                        binding,
                        random_tensor_value(binding, indices.shape),
                    )
                return
            rule.set(binding, random_tensor_value(binding, binding.shape))
            return
        rule.set(binding, rule.default(binding))

    for binding in rule.all_tensors:
        if binding.parameter_name == "indices":
            rule.set(binding, indices_value(binding))
        elif binding.parameter_name == "values":
            write_values(binding)
        else:
            rule.set(binding, rule.default(binding))


@rules.register("paddle.matrix_transpose")
def matrix_transpose_values(rule: RuleContext):
    """保证矩阵转置输入至少具有二维结构。"""

    def x_value(binding):
        shape = binding.shape if len(binding.shape) >= 2 else (2, 2)
        dtype = binding.dtype
        if "int" in dtype:
            return rule.ops.cast(rule.ops.randint(-65535, 65535, shape=shape), dtype)
        return rule.ops.cast(rule.ops.random(shape) - 0.5, dtype)

    rule.generate((("x", x_value),))


@rules.register("paddle.nn.functional.softmax")
def softmax_values(rule: RuleContext):
    """根据输入 rank 生成有效 softmax axis。"""

    def axis_value(binding):
        x_shape = rule.arg("x").shape
        return rule.domain("uniform", binding, low=-len(x_shape), high=len(x_shape))

    rule.generate(
        (
            ("x", "random_range"),
            ("axis", axis_value),
        ),
    )


@rules.register("paddle.nn.functional.zeropad2d")
def zeropad2d_values(rule: RuleContext):
    """按输入空间维度约束二维 padding。"""

    def padding_value(binding):
        return rule.domain("uniform", binding, low=0, high=10)

    rule.generate(
        (
            ("x", "random_range"),
            ("padding", padding_value),
        ),
    )


@rules.register("paddle.nn.functional.pad")
def pad_values(rule: RuleContext):
    """按输入 rank 生成不会越界的 padding。"""

    def pad_value(binding):
        x_shape = rule.arg("x").shape
        return rule.domain("uniform", binding, low=0, high=min(x_shape))

    rule.generate((("pad", pad_value),))


@rules.register("paddle.nn.functional.class_center_sample")
def class_center_sample_values(rule: RuleContext):
    """根据类别数生成合法标签索引。"""

    def label_value(binding):
        num_classes = rule.arg("num_classes")
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("label", label_value),))


@rules.register("paddle.shard_index")
def shard_index_values(rule: RuleContext):
    """根据 index_num 限制分片输入索引范围。"""

    def input_binding(binding):
        index_num = rule.arg("index_num")
        if index_num is None:
            index_num = rule.ops.randint(1, 1000)
        return rule.ops.cast(
            rule.ops.randint(0, index_num, shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("input", input_binding),))


@rules.register("paddle.incubate.nn.functional.masked_multihead_attention")
def masked_multihead_attention_values(rule: RuleContext):
    """联动序列长度和 mask 值域生成注意力输入。"""

    def sequence_lengths(binding):
        return rule.domain("random_range", binding, low=1)

    def rotary_tensor(binding):
        return rule.domain("uniform", binding, low=0, high=1000)

    rule.generate(
        (
            ("sequence_lengths", sequence_lengths),
            ("rotary_tensor", rotary_tensor),
        ),
    )


@rules.register(
    "paddle.argmax",
    "paddle.argmin",
    aliases=("paddle.Tensor.argmax", "paddle.Tensor.argmin"),
)
def argminmax_values(rule: RuleContext):
    """根据输入 rank 生成 argmin/argmax 的合法 axis。"""

    def axis_value(binding):
        x_shape = rule.arg("x").shape
        min_dim = len(x_shape)
        return rule.ops.cast(
            rule.ops.randint(-min_dim, min_dim - 1, shape=binding.shape),
            "int64",
        )

    rule.generate((("axis", axis_value),))


@rules.register("paddle.cumsum", aliases=("paddle.Tensor.cumsum",))
def cumsum_values(rule: RuleContext):
    """根据输入 rank 生成 cumsum 的合法 axis。"""

    def axis_value(binding):
        x_shape = rule.arg("x").shape
        return rule.ops.randint(-len(x_shape), len(x_shape), shape=binding.shape)

    rule.generate((("axis", axis_value),))


@rules.register(
    "paddle.mean", "paddle.max", "paddle.min", "paddle.prod", "paddle.sum", "paddle.squeeze"
)
def reduction_axis_values(rule: RuleContext):
    """为归约 API 生成去重且不越界的 axis。"""
    used_list_axes = None

    def init_used_list_axes(x_shape, axis_arg):
        used_axes = set()
        max_dim = max(len(x_shape), 1)
        if isinstance(axis_arg, (list, tuple)):
            for item in axis_arg:
                if rule.is_tensor_config(item):
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
        x_shape = rule.arg("x").shape
        max_dim = max(len(x_shape), 1)
        axis_arg = rule.arg("axis", None)
        if isinstance(axis_arg, (list, tuple)) and binding.path.indices:
            if binding.shape not in [(), (1,)]:
                raise ValueError(
                    f"Invalid TensorConfig for axis: shape {binding.shape} or dtype {binding.dtype}"
                )
            if binding.dtype not in {"int32", "int64"}:
                raise ValueError(
                    f"Invalid TensorConfig for axis: shape {binding.shape} or dtype {binding.dtype}"
                )
            if used_list_axes is None:
                used_list_axes = init_used_list_axes(x_shape, axis_arg)
            available_dims = sorted(set(range(max_dim)) - used_list_axes)
            if not available_dims:
                raise ValueError("Not enough available dimensions for axis TensorConfig items")
            dim = rule.ops.choice(available_dims, replace=False)
            dim = int(dim)
            used_list_axes.add(dim)
            if rule.ops.random() > 0.5:
                dim -= max_dim
            return rule.ops.asarray(dim, dtype=binding.dtype)
        if len(binding.shape) == 0:
            dim = rule.ops.randint(0, max_dim)
            if rule.ops.random() > 0.5:
                dim -= max_dim
            return rule.ops.asarray(dim, dtype=binding.dtype)
        if len(binding.shape) == 1:
            dims = rule.ops.choice(max_dim, shape=binding.shape[0], replace=False)
            mask = rule.ops.random(binding.shape[0]) > 0.5
            dims = rule.ops.where(mask, dims - max_dim, dims)
            return rule.ops.asarray(dims, dtype=binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {rule.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.shape}."
        )

    rule.generate((("axis", axis_value),))


@rules.register("paddle.unsqueeze")
def unsqueeze_values(rule: RuleContext):
    """按扩维后的 rank 生成合法 unsqueeze axis。"""

    def axis_value(binding):
        x_shape = rule.arg("x").shape
        max_dim = len(x_shape) + 1
        if len(binding.shape) == 0:
            dim = rule.ops.randint(0, max_dim)
            if rule.ops.random() > 0.5:
                dim -= max_dim
            return rule.ops.asarray(dim, dtype=binding.dtype)
        if len(binding.shape) == 1:
            dims = rule.ops.choice(max_dim, shape=binding.shape[0], replace=False)
            mask = rule.ops.random(binding.shape[0]) > 0.5
            dims = rule.ops.where(mask, dims - max_dim, dims)
            return rule.ops.asarray(dims, dtype=binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.shape}."
        )

    rule.generate((("axis", axis_value),))


@rules.register("paddle.unflatten", aliases=("paddle.Tensor.unflatten",))
def unflatten_values(rule: RuleContext):
    """根据源维度生成乘积匹配的 unflatten shape。"""

    def axis_value(binding):
        x_shape = rule.arg("x").shape
        return rule.ops.cast(
            rule.ops.randint(0, len(x_shape), shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("axis", axis_value),))


@rules.register("paddle.topk", aliases=("paddle.Tensor.topk",))
def topk_values(rule: RuleContext):
    """联动 axis 维度限制 topk 的 k 值。"""

    def x_value(binding):
        dtype = binding.dtype
        if dtype == "bfloat16" or dtype in {"float8_e4m3fn", "float8_e5m2"}:
            dtype = "float32" if dtype == "bfloat16" else "float16"
        if dtype in {"float32", "float64"}:
            return rule.ops.cast(
                (rule.ops.random(binding.shape) - 0.5) * 1.2,
                dtype,
            )
        if dtype == "float16":
            return rule.ops.cast(
                rule.ops.cast(rule.ops.randn(*binding.shape), dtype) * 1e-3,
                dtype,
            )
        if dtype in {"int32", "int64"}:
            return rule.ops.cast(
                rule.ops.randint(-10, 10, shape=binding.shape),
                dtype,
            )
        raise ValueError(f"Unsupported dtype {binding.dtype} for paddle.topk / paddle.Tensor.topk")

    def k_value(binding):
        x_config = rule.arg("x")
        axis = rule.arg("axis", -1)
        max_k_value = 1
        if x_config is not None and x_config.shape:
            max_k_value = x_config.shape[axis] if len(x_config.shape) > 0 else 1
        if not binding.shape:
            return rule.ops.asarray(rule.ops.randint(1, max_k_value + 1), dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(1, max_k_value + 1, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            ("x", x_value),
            ("k", k_value),
        ),
    )


@rules.register("paddle.index_sample")
def index_sample_values(rule: RuleContext):
    """按输入第二维限制 index_sample 索引。"""

    def index_value(binding):
        x_dim = rule.arg("x").shape[1]
        return rule.ops.randint(0, x_dim, shape=binding.shape)

    rule.generate((("index", index_value),))


@rules.register("paddle.Tensor.__getitem__")
def tensor_getitem_values(rule: RuleContext):
    """根据源 Tensor 最小维度生成合法 getitem 索引。"""

    def source_binding():
        binding = rule.tensor("arr") or rule.tensor("x") or rule.tensor("self")
        if binding is None:
            raise ValueError("Tensor.__getitem__ rule could not find source tensor")
        return binding

    def item_value(binding):
        min_dim = min(source_binding().shape)
        numel = math.prod(binding.shape)
        if binding.dtype == "bool":
            indices = rule.ops.choice([0, 1], shape=numel)
        else:
            indices = rule.ops.randint(0, min_dim, shape=numel)
        return rule.ops.cast(
            rule.ops.reshape(indices, binding.shape),
            binding.dtype,
        )

    rule.generate((("item", item_value),))


@rules.register("paddle.Tensor.__setitem__")
def tensor_setitem_values(rule: RuleContext):
    """根据源和值的 shape 生成合法 setitem 索引。"""

    def source_binding():
        binding = rule.tensor("arr") or rule.tensor("x") or rule.tensor("self")
        if binding is None:
            raise ValueError("Tensor.__setitem__ rule could not find source tensor")
        return binding

    def item_value(binding):
        min_dim = min(source_binding().shape)
        numel = math.prod(binding.shape)
        if binding.dtype == "bool":
            value = rule.arg("value")
            if value is not None and hasattr(value, "shape"):
                indices = rule.ops.zeros(numel, dtype="int64")
                num_true = min(value.shape[0], numel)
                true_indices = rule.ops.choice(numel, shape=num_true, replace=False)
                indices[true_indices] = 1
            else:
                indices = rule.ops.choice([0, 1], shape=numel)
        else:
            indices = rule.ops.randint(0, min_dim, shape=numel)
        return rule.ops.cast(
            rule.ops.reshape(indices, binding.shape),
            binding.dtype,
        )

    rule.generate((("item", item_value),))


@rules.register("paddle.index_add", "paddle.index_fill")
def index_update_values(rule: RuleContext):
    """按目标 axis 尺寸生成 index_add/index_fill 索引。"""

    def index_value(binding):
        axis = rule.arg("axis")
        if axis is None:
            raise ValueError("Axis is None")
        x_shape = rule.arg("x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
        if len(binding.shape) >= 1:
            return rule.ops.cast(
                rule.ops.randint(0, x_shape[axis], shape=binding.shape),
                binding.dtype,
            )
        raise ValueError(
            f"Invalid shape for 'index' Tensor in {rule.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.shape}."
        )

    rule.generate((("index", index_value),))


@rules.register("paddle.take")
def take_values(rule: RuleContext):
    """按源 Tensor 元素数限制 take 索引。"""

    def index_value(binding):
        x = rule.arg("x")
        dim_size = math.prod(x.shape)
        return rule.ops.cast(
            rule.ops.randint(0, dim_size, shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("index", index_value),))


@rules.register("paddle.gather", aliases=("paddle.Tensor.gather",))
def gather_values(rule: RuleContext):
    """按 gather axis 对各维索引进行边界约束。"""

    def index_value(binding):
        x = rule.arg("x")
        if rule.has_kwarg("axis"):
            axis = rule.arg("axis")
            if hasattr(axis, "shape"):
                axis = axis.shape[0]
        else:
            axis = 0
        return rule.ops.cast(
            rule.ops.randint(0, x.shape[axis], shape=binding.shape),
            binding.dtype,
        )

    def axis_value(binding):
        return rule.ops.cast(
            rule.ops.randint(0, 2, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            ("index", index_value),
            ("axis", axis_value),
        ),
    )


@rules.register("paddle.gather_nd", aliases=("paddle.Tensor.gather_nd",))
def gather_nd_values(rule: RuleContext):
    """根据源和索引 rank 生成 gather_nd 索引。"""

    def index_value(binding):
        x_shape = rule.arg("x").shape
        index_shape = rule.arg("index").shape
        result = rule.ops.zeros(index_shape, dtype=binding.dtype)
        for index in range(index_shape[-1]):
            result[..., index] = rule.ops.randint(0, x_shape[index], shape=result[..., index].shape)
        return result

    rule.generate((("index", index_value),))


@rules.register("paddle.index_select", aliases=("paddle.Tensor.index_select",))
def index_select_values(rule: RuleContext):
    """按选取 axis 的尺寸限制 index_select 索引。"""

    def index_value(binding):
        axis = rule.arg("axis")
        if axis is None:
            axis = 0
        x = rule.arg("x")
        if x.shape[axis] == 0:
            return rule.ops.zeros(binding.shape, dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(0, x.shape[axis], shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("index", index_value),))


@rules.register("paddle.take_along_axis", aliases=("paddle.Tensor.take_along_axis",))
def take_along_axis_values(rule: RuleContext):
    """按目标 axis 尺寸限制 take_along_axis 索引。"""

    def indices_value(binding):
        arr_shape = rule.arg("arr").shape
        axis = rule.arg("axis")
        axis_value = axis if axis >= 0 else axis + len(arr_shape)
        dim_size = arr_shape[axis_value]
        dtype = binding.dtype if binding.dtype in {"int32", "int64"} else "int64"
        num_elements = math.prod(binding.shape)
        if num_elements == 0:
            indices = rule.ops.asarray([], dtype=dtype)
        elif dim_size == 1:
            indices = rule.ops.zeros(num_elements, dtype=dtype)
        elif num_elements == 1:
            indices = rule.ops.asarray([0], dtype=dtype)
        else:
            indices = rule.ops.cast(rule.ops.randint(0, dim_size, shape=num_elements), dtype)
            positions_to_replace = rule.ops.choice(num_elements, shape=2, replace=False)
            flat_indices = rule.ops.flatten(indices)
            flat_indices[positions_to_replace[0]] = 0
            flat_indices[positions_to_replace[1]] = dim_size - 1
            indices = flat_indices
        return rule.ops.reshape(indices, binding.shape)

    rule.generate((("indices", indices_value),))


@rules.register("paddle.index_put", aliases=("paddle.Tensor.index_put",))
def index_put_values(rule: RuleContext):
    """联动多个索引 Tensor 的 shape 与各维取值范围。"""
    state = {}

    def prepare_indices_state():
        x = rule.arg("x")
        value = rule.arg("value")
        indices = rule.arg("indices")
        if not isinstance(indices, (list, tuple)):
            return None

        x_shape = x.shape
        value_shape = value.shape
        int_index_shapes = []
        has_bool_index = False
        dims_consumed = 0
        for item in indices:
            if not rule.is_tensor_config(item):
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
        num_elements = rule.ops.prod(shape)
        if num_elements > dim_size:
            indices_flat = rule.ops.randint(-dim_size, dim_size, shape=num_elements)
        else:
            indices_flat = rule.ops.choice(dim_size, shape=num_elements, replace=False)
        return rule.ops.reshape(indices_flat, shape)

    def bool_mask(shape, num_true):
        mask_size = rule.ops.prod(shape)
        if mask_size < num_true:
            raise ValueError(
                f"Cannot generate a mask with {num_true} true values in a {mask_size} element mask"
            )
        mask_flat = rule.ops.zeros(mask_size, dtype="bool")
        true_indices = rule.ops.choice(mask_size, shape=num_true, replace=False)
        mask_flat[true_indices] = True
        return rule.ops.reshape(mask_flat, shape)

    def index_value(binding):
        if not state:
            prepared = prepare_indices_state()
            if prepared is None:
                return rule.default(binding)
            state.update(prepared)

        if binding.dtype == "bool":
            if state["num_true_needed"] < 0:
                raise ValueError(
                    "Cannot determine the number of True elements for the boolean mask."
                )
            state["x_dim_cursor"] += len(binding.shape)
            return bool_mask(binding.shape, state["num_true_needed"])

        x_dim_to_index = state["x_shape"][state["x_dim_cursor"]]
        state["x_dim_cursor"] += 1
        return rule.ops.cast(
            int_indices(binding.shape, x_dim_to_index),
            binding.dtype,
        )

    rule.generate((("indices", index_value),))


@rules.register("paddle.multiplex")
def multiplex_values(rule: RuleContext):
    """按候选输入数量生成 multiplex 选择索引。"""

    def index_value(binding):
        axis_values = rule.arg("inputs")
        return rule.ops.cast(
            rule.ops.randint(0, len(axis_values), shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("index", index_value),))


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
def segment_values(rule: RuleContext):
    """根据 data 批次大小生成有序 segment_ids。"""

    def segment_ids_value(binding):
        batch_size = rule.arg("data").shape[0]
        max_segments = rule.ops.randint(1, batch_size + 1)
        segment_ids = rule.ops.cast(
            rule.ops.randint(0, max_segments, shape=binding.shape),
            binding.dtype,
        )
        return rule.ops.sort(segment_ids)

    rule.generate((("segment_ids", segment_ids_value),))


@rules.register(
    "paddle.geometric.send_u_recv",
    "paddle.geometric.send_uv",
    "paddle.geometric.send_ue_recv",
)
def geometric_send_values(rule: RuleContext):
    """根据节点数生成合法的图消息收发索引。"""

    def index_value(binding):
        num_nodes = rule.arg("x").shape[0]
        return rule.ops.cast(
            rule.ops.randint(0, num_nodes, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        ((("src_index", "dst_index"), index_value),),
    )


@rules.register("paddle.geometric.sample_neighbors")
def sample_neighbors_values(rule: RuleContext):
    """联动 CSR 边界和节点数生成邻居采样输入。"""

    def row_value(binding):
        colptr_shape = rule.arg("colptr").shape
        num_nodes = colptr_shape[0] - 1
        return rule.ops.randint(0, num_nodes, shape=binding.shape, dtype=binding.dtype)

    def colptr_value(binding):
        row = rule.arg("row")
        num_edges = row.shape[0]
        num_nodes = binding.shape[0] - 1
        colptr = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        if num_nodes > 0 and num_edges > 0:
            splits = rule.ops.choice(
                rule.ops.arange(num_edges + 1),
                shape=num_nodes - 1,
                replace=True,
            )
            splits = rule.ops.sort(splits)
            colptr[1:num_nodes] = splits
            colptr[num_nodes] = num_edges
        return colptr

    def input_nodes_value(binding):
        num_nodes = binding.shape[0] - 1
        return rule.ops.randint(0, num_nodes, shape=binding.shape, dtype=binding.dtype)

    def edge_order_value(binding):
        num_edges = rule.arg("row").shape[0]
        return rule.ops.reshape(
            rule.ops.arange(num_edges, dtype=binding.dtype),
            binding.shape,
        )

    rule.generate(
        (
            ("row", row_value),
            ("colptr", colptr_value),
            ("input_nodes", input_nodes_value),
            (("eids", "perm_buffer"), edge_order_value),
        ),
    )


@rules.register("paddle.reshape", aliases=("paddle.Tensor.reshape",))
def reshape_values(rule: RuleContext):
    """生成元素总数与源 Tensor 一致的 reshape shape。"""
    state = {
        "shape": None,
        "maxvalue": None,
        "tensornum": None,
    }

    def initialize_from_x(binding):
        shape = binding.shape
        if 0 not in shape and state["shape"] is None:
            state["shape"] = shape
            state["maxvalue"] = math.prod(shape)
            state["tensornum"] = 0
            for candidate in rule.argument_values():
                if isinstance(candidate, (list, tuple)):
                    for index, item in enumerate(candidate):
                        if isinstance(item, numbers.Integral):
                            if item == 0:
                                state["maxvalue"] //= shape[index]
                            elif item != -1:
                                state["maxvalue"] //= int(item)
                        elif rule.is_tensor_config(item):
                            state["tensornum"] += 1
        return rule.default(binding)

    def shape_value(binding):
        if state["tensornum"] == 0:
            state["tensornum"] = 1
        dtype = "int32"
        shape = binding.shape
        maxvalue = state["maxvalue"]
        if shape not in ((), (1,)):
            result = rule.ops.zeros(shape, dtype=dtype)
            for index in range(shape[0]):
                if index < shape[0] - 1:
                    result[index] = rule.ops.randint(1, maxvalue + 1)
                    while maxvalue % result[index]:
                        result[index] = rule.ops.randint(1, maxvalue + 1)
                    maxvalue //= result[index]
                else:
                    result[index] = maxvalue
            state["maxvalue"] = maxvalue
            return result
        if state["tensornum"] == 1:
            return rule.ops.cast(
                rule.ops.randint(maxvalue, maxvalue + 1, shape=shape),
                dtype,
            )
        state["tensornum"] -= 1
        result = rule.ops.cast(rule.ops.randint(1, maxvalue + 1, shape=shape), dtype)
        while maxvalue % result:
            result = rule.ops.cast(
                rule.ops.randint(1, maxvalue + 1, shape=shape),
                dtype,
            )
        state["maxvalue"] = maxvalue // result
        return result

    rule.generate(
        (
            ("x", initialize_from_x),
            ("shape", shape_value),
        ),
    )


@rules.register("paddle.slice")
def slice_values(rule: RuleContext):
    """联动 axes、starts、ends 和 steps 生成有效切片。"""
    state = {
        "shape": None,
        "indice": 0,
        "start": [],
        "index": 0,
    }

    def axes():
        return rule.arg("axes")

    def input_binding(binding):
        if state["shape"] is None:
            state["shape"] = binding.shape
        return rule.default(binding)

    def starts_value(binding):
        dim_sizes = [state["shape"][axis] for axis in axes()]
        if binding.shape == ():
            coin = rule.ops.randint(0, 2)
            if coin == 0:
                value = rule.ops.randint(0, dim_sizes[state["indice"]] - 1, binding.shape)
            else:
                value = rule.ops.randint(-65535, -1, binding.shape)
            state["start"].append(value)
            state["indice"] += 1
            return rule.ops.asarray(value, dtype=binding.dtype)
        result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        for index in range(math.prod(binding.shape)):
            coin = rule.ops.randint(0, 2)
            if coin == 0:
                result[index] = rule.ops.randint(0, dim_sizes[state["indice"]] - 1)
            else:
                result[index] = rule.ops.randint(-65535, -1)
            state["start"].append(result[index])
            state["indice"] += 1
        return result

    def ends_value(binding):
        if not state["start"]:
            start_arg = rule.arg("starts")
            state["start"] = list(
                start_arg if isinstance(start_arg, (list, tuple)) else [start_arg]
            )
        dim_sizes = [state["shape"][axis] for axis in axes()]
        start = state["start"]
        for index, item in enumerate(start):
            if item < 0:
                item = item if item > -dim_sizes[index] else -dim_sizes[index]
                start[index] = item + dim_sizes[index]
        if binding.shape == ():
            coin = rule.ops.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                value = rule.ops.randint(current + 1, 65535, binding.shape)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                value = rule.ops.randint(min(current - dim_sizes[index] + 1, -1), 0, binding.shape)
            state["index"] += 1
            return rule.ops.asarray(value, dtype=binding.dtype)
        result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        for index in range(math.prod(binding.shape)):
            coin = rule.ops.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                result[index] = rule.ops.randint(current + 1, 65535)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                result[index] = rule.ops.randint(current - dim_sizes[state["index"]] + 1, 0)
            state["index"] += 1
        return result

    rule.generate(
        (
            ("input", input_binding),
            ("starts", starts_value),
            ("ends", ends_value),
        ),
    )


@rules.register("paddle.scatter")
def scatter_values(rule: RuleContext):
    """按源 Tensor 首维生成 scatter 索引。"""

    def index_value(binding):
        x = rule.arg("x")
        first_dim = x.shape[0]
        overwrite = rule.arg("overwrite")
        if (overwrite is None or overwrite is True) and (
            binding.shape == () or binding.shape[0]
        ) <= first_dim:
            return rule.ops.cast(
                rule.ops.choice(first_dim, shape=binding.shape, replace=False),
                binding.dtype,
            )
        return rule.ops.cast(
            rule.ops.randint(0, first_dim, shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("index", index_value),))


@rules.register("paddle.scatter_nd")
def scatter_nd_values(rule: RuleContext):
    """根据输出 shape 生成 scatter_nd 多维索引。"""

    def index_value(binding):
        output_shape = rule.arg("shape")
        if output_shape and len(output_shape):
            result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
            for axis in range(len(output_shape)):
                if axis >= binding.shape[-1]:
                    break
                result[..., axis] = rule.ops.randint(
                    -output_shape[axis],
                    output_shape[axis],
                    shape=result[..., axis].shape,
                )
                result[..., axis] = rule.ops.cast(result[..., axis], binding.dtype)
            return result
        return rule.default(binding)

    rule.generate((("index", index_value),))


@rules.register("paddle.scatter_nd_add")
def scatter_nd_add_values(rule: RuleContext):
    """根据目标 Tensor shape 生成 scatter_nd_add 索引。"""

    def index_value(binding):
        x_shape = rule.arg("x").shape
        result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        for axis in range(binding.shape[-1]):
            result[..., axis] = rule.ops.randint(
                -x_shape[axis],
                x_shape[axis],
                shape=result[..., axis].shape,
            )
            result[..., axis] = rule.ops.cast(result[..., axis], binding.dtype)
        return result

    rule.generate((("index", index_value),))


@rules.register("paddle.strided_slice")
def strided_slice_values(rule: RuleContext):
    """联动 axes 与步长方向生成有效分片边界。"""

    def axes_value(binding):
        x = rule.arg("x")
        return rule.ops.cast(
            rule.ops.randint(0, len(x.shape), shape=binding.shape),
            binding.dtype,
        )

    def list_value(binding):
        x = rule.arg("x")
        axes_arg = rule.arg("axes")
        axes = axes_arg
        if not isinstance(axes, list):
            axes = rule.value(rule.tensor("axes"))
        if not binding.path.indices:
            return rule.default(binding)
        list_index = binding.path.indices[0]
        parameter = binding.parameter_name
        if parameter == "starts":
            return rule.ops.cast(
                rule.ops.randint(0, x.shape[axes[list_index]] - 1, shape=binding.shape),
                binding.dtype,
            )
        if parameter == "ends":
            return rule.ops.cast(
                rule.ops.randint(
                    rule.value(rule.tensor("starts"))[list_index] + 1,
                    x.shape[axes[list_index]],
                    shape=binding.shape,
                ),
                binding.dtype,
            )
        if parameter == "strides":
            return rule.ops.cast(
                rule.ops.randint(1, x.shape[axes[list_index]], shape=binding.shape),
                binding.dtype,
            )
        return rule.default(binding)

    rule.generate(
        (
            ("axes", axes_value),
            (("starts", "ends", "strides"), list_value),
        ),
    )


@rules.register("paddle.tensordot")
def tensordot_values(rule: RuleContext):
    """为两个输入生成不重复且维度相容的收缩 axes。"""
    state = {"shape1": None, "shape2": None, "tensor1": None}

    def x_value(binding):
        if state["shape1"] is None:
            state["shape1"] = binding.shape
        return rule.default(binding)

    def y_value(binding):
        if state["shape2"] is None:
            state["shape2"] = binding.shape
        return rule.default(binding)

    def axes_value(binding):
        axes_arg = rule.arg("axes")
        rank = len(state["shape1"])
        if isinstance(axes_arg, (list, tuple)):
            if state["tensor1"] is None:
                result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
                used = []
                for index in range(math.prod(binding.shape)):
                    result[index] = rule.ops.randint(0, rank)
                    while (
                        state["shape1"][result[index]] not in state["shape2"]
                        or result[index] in used
                    ):
                        result[index] = rule.ops.randint(0, rank)
                    used.append(result[index])
                state["tensor1"] = result
                return result
            result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
            used = []
            for index in range(math.prod(binding.shape)):
                result[index] = rule.ops.randint(0, rank)
                while (
                    state["shape2"][result[index]] != state["shape1"][state["tensor1"][index]]
                    or result[index] in used
                ):
                    result[index] = rule.ops.randint(0, rank)
                used.append(result[index])
            return result
        if binding.shape == () or math.prod(binding.shape) == 1:
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
            return rule.ops.asarray([rule.ops.choice(candidates)], dtype=binding.dtype)
        result = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        used1 = []
        used2 = []
        for index in range(binding.shape[0]):
            result[0][index] = rule.ops.randint(0, rank)
            result[1][index] = rule.ops.randint(0, rank)
            while (
                state["shape1"][result[0][index]] != state["shape2"][result[1][index]]
                or result[0][index] in used1
                or result[1][index] in used2
            ):
                result[0][index] = rule.ops.randint(0, rank)
                result[1][index] = rule.ops.randint(0, rank)
            used1.append(result[0][index])
            used2.append(result[1][index])
        return result

    rule.generate(
        (
            ("x", x_value),
            ("y", y_value),
            ("axes", axes_value),
        ),
    )


@rules.register("paddle.nn.functional.embedding")
def embedding_values(rule: RuleContext):
    """按词表大小生成 embedding ids，并收紧权重值域。"""

    def ids_value(binding):
        weight_config = rule.arg("weight")
        vocab_size = rule.ops.randint(10, 1000)
        if weight_config is not None and weight_config.shape:
            vocab_size = weight_config.shape[0]
        if vocab_size == 0:
            return rule.ops.zeros(binding.shape, dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(0, vocab_size, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            (("x", "ids"), ids_value),
            ("weight", "multiply"),
        ),
    )


@rules.register("paddle.nn.functional.affine_grid")
def affine_grid_values(rule: RuleContext):
    """根据 theta 批次维生成相容的输出 shape。"""

    def out_shape_value(binding):
        theta_shape = rule.arg("theta").shape
        out_shape = rule.ops.cast(
            rule.ops.randint(1, 128, shape=binding.shape),
            binding.dtype,
        )
        out_shape[0] = theta_shape[0]
        return out_shape

    rule.generate((("out_shape", out_shape_value),))


@rules.register("paddle.nn.functional.hsigmoid_loss")
def hsigmoid_loss_values(rule: RuleContext):
    """按类别和权重规模约束标签及路径编码。"""

    def label_value(binding):
        num_classes = rule.arg("num_classes")
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=binding.shape),
            binding.dtype,
        )

    def path_table_value(binding):
        weight = rule.arg("weight")
        return rule.ops.cast(
            rule.ops.randint(0, weight.shape[0], shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            ("label", label_value),
            ("path_table", path_table_value),
            ("path_code", "binary_0_1"),
        ),
    )


@rules.register("paddle.nn.functional.margin_cross_entropy")
def margin_cross_entropy_values(rule: RuleContext):
    """根据 logits 类别维生成合法标签。"""

    def label_value(binding):
        logits = rule.arg("logits")
        return rule.ops.cast(
            rule.ops.randint(0, logits.shape[1], shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("label", label_value),))


@rules.register("paddle.nn.functional.multi_margin_loss")
def multi_margin_loss_values(rule: RuleContext):
    """根据输入类别维生成合法标签。"""

    def label_value(binding):
        logits = rule.arg("input")
        return rule.ops.cast(
            rule.ops.randint(0, logits.shape[1], shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("label", label_value),))


@rules.register("paddle.nn.functional.dice_loss")
def dice_loss_values(rule: RuleContext):
    """根据输入末维生成 dice loss 标签。"""

    def label_value(binding):
        tensor = rule.arg("input")
        return rule.ops.cast(
            rule.ops.randint(0, tensor.shape[-1], shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("label", label_value),))


@rules.register("paddle.nn.functional.nll_loss")
def nll_loss_values(rule: RuleContext):
    """根据输入类别维生成 NLL 标签。"""

    def label_value(binding):
        input_config = rule.arg("input")
        n_classes = rule.ops.randint(5, 50) if input_config is None else input_config.shape[1]
        return rule.ops.cast(
            rule.ops.randint(0, n_classes, shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("label", label_value),))


@rules.register("paddle.nn.functional.adaptive_log_softmax_with_loss")
def adaptive_log_softmax_with_loss_values(rule: RuleContext):
    """根据 cutoffs 推导类别数并生成合法标签。"""

    def label_value(binding):
        cutoffs = rule.arg("cutoffs")
        n_classes = cutoffs[-1]
        generation_size = binding.shape
        if len(binding.shape) == 0:
            generation_size = 1
        if n_classes == 1:
            return rule.ops.zeros(generation_size, dtype=binding.dtype)
        return rule.ops.randint(0, n_classes, shape=generation_size, dtype=binding.dtype)

    rule.generate((("label", label_value),))


@rules.register("paddle.nn.functional.cross_entropy")
def cross_entropy_values(rule: RuleContext):
    """联动 axis、soft_label 和平滑系数生成标签与权重。"""

    def input_value(binding):
        use_softmax = rule.arg("use_softmax", True)
        if use_softmax:
            return rule.default(binding)
        axis = rule.arg("axis", -1)
        logits = rule.ops.random(binding.shape)
        probabilities = logits / rule.ops.sum(logits, axis=axis, keepdims=True)
        return rule.ops.cast(probabilities, binding.dtype)

    def label_value(binding):
        input_shape = rule.arg("input").shape
        axis = rule.arg("axis", -1)
        num_classes = input_shape[axis]
        soft_label = rule.arg("soft_label", False)
        label_smoothing = rule.arg("label_smoothing", 0.0)
        if (label_smoothing > 0 and list(binding.shape) == list(input_shape)) or (
            label_smoothing == 0 and soft_label
        ):
            soft_labels = rule.ops.random(binding.shape)
            soft_labels = soft_labels / rule.ops.sum(soft_labels, axis=axis, keepdims=True)
            return rule.ops.cast(soft_labels, binding.dtype)
        if num_classes == 0:
            return rule.ops.zeros(binding.shape, dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=binding.shape),
            binding.dtype,
        )

    def weight_value(binding):
        weights = rule.ops.random(binding.shape)
        return weights / rule.ops.sum(weights)

    rule.generate(
        (
            ("input", input_value),
            ("label", label_value),
            ("weight", weight_value),
        ),
    )


@rules.register("paddle.nn.functional.ctc_loss")
def ctc_loss_values(rule: RuleContext):
    """联动 blank、标签范围和序列长度生成 CTC 输入。"""

    def labels_value(binding):
        num_classes = rule.arg("log_probs").shape[2] - 1
        blank = rule.arg("blank", 0)
        valid_label_indices = [index for index in range(num_classes + 1) if index != blank]
        if not valid_label_indices:
            return rule.ops.zeros(binding.shape, dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.choice(valid_label_indices, shape=binding.shape, replace=True),
            binding.dtype,
        )

    def input_lengths_value(binding):
        max_logit_length = rule.arg("log_probs").shape[0]
        return rule.ops.randint(
            1,
            max_logit_length + 1,
            shape=binding.shape,
            dtype=binding.dtype,
        )

    def label_lengths_value(binding):
        max_label_length = rule.arg("labels").shape[1]
        max_logit_length = rule.arg("log_probs").shape[0]
        cand_label_lengths = rule.ops.randint(
            1,
            max_label_length + 1,
            shape=binding.shape,
            dtype=binding.dtype,
        )
        compatible_input_lengths = rule.ops.randint(
            1,
            max_logit_length + 1,
            shape=binding.shape,
            dtype=binding.dtype,
        )
        final_label_lengths = rule.ops.minimum(cand_label_lengths, compatible_input_lengths)
        return rule.ops.maximum(final_label_lengths, 1)

    rule.generate(
        (
            ("labels", labels_value),
            ("input_lengths", input_lengths_value),
            ("label_lengths", label_lengths_value),
        ),
    )


@rules.register("paddle.nn.functional.sequence_mask")
def sequence_mask_values(rule: RuleContext):
    """根据 maxlen 约束 sequence_mask 的长度输入。"""

    def x_value(binding):
        maxlen_config = rule.arg("maxlen")
        provided_maxlen = None
        if isinstance(maxlen_config, int):
            provided_maxlen = max(1, maxlen_config)
        if provided_maxlen is not None:
            return rule.ops.cast(
                rule.ops.randint(0, provided_maxlen + 1, shape=binding.shape),
                binding.dtype,
            )
        high_value = rule.ops.randint(1, 2048)
        lengths = rule.ops.cast(
            rule.ops.randint(0, high_value, shape=binding.shape),
            binding.dtype,
        )
        if rule.ops.prod(lengths.shape) > 0 and rule.ops.count_nonzero(lengths) == 0:
            fix_value = rule.ops.randint(1, max(2, high_value))
            rule.ops.flatten(lengths)[0] = fix_value
        return lengths

    rule.generate((("x", x_value),))


@rules.register("paddle.nn.functional.softmax_with_cross_entropy")
def softmax_with_cross_entropy_values(rule: RuleContext):
    """根据 logits 类别维生成交叉熵标签。"""

    def label_value(binding):
        logits = rule.arg("logits")
        if not hasattr(logits, "shape"):
            logits = rule.kwarg("logits")
        num_classes = 10
        if logits is not None:
            axis = rule.kwarg("axis", -1)
            axis = axis if axis >= 0 else len(logits.shape) + axis
            if 0 <= axis < len(logits.shape):
                num_classes = logits.shape[axis]
        else:
            num_classes = rule.ops.randint(5, 20)
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=binding.shape),
            binding.dtype,
        )

    rule.generate((("label", label_value),))


@rules.register("paddle.linalg.cholesky")
def cholesky_values(rule: RuleContext):
    """构造对称正定矩阵供 Cholesky 分解使用。"""

    def x_value(binding):
        if len(binding.shape) < 2 or binding.shape[-1] != binding.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = binding.shape[:-2]
        matrix_dim = binding.shape[-1]
        matrix = rule.ops.random([*batch_dims, matrix_dim, matrix_dim], dtype=binding.dtype)
        if len(batch_dims) > 0:
            tensor = rule.ops.einsum("...ij,...kj->...ik", matrix, matrix)
        else:
            tensor = rule.ops.dot(matrix, rule.ops.swapaxes(matrix, -1, -2))
        tensor += rule.ops.eye(matrix_dim, dtype=binding.dtype) * 10000
        print("cholesky tensor", tensor)
        return tensor

    rule.generate((("x", x_value),))


@rules.register("paddle.linalg.cov")
def covariance_values(rule: RuleContext):
    """根据 rowvar 语义生成非退化协方差输入和权重。"""

    def observation_count():
        x_shape = rule.arg("x").shape
        rowvar = rule.arg("rowvar")
        if rowvar is None:
            rowvar = True
        return (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]

    def x_value(binding):
        if len(binding.shape) < 1 or len(binding.shape) > 2:
            raise ValueError("Shape must have 1 or 2 dimensions for covariance input")
        tensor = rule.ops.random(binding.shape, dtype=binding.dtype)
        tensor += rule.ops.random(binding.shape, dtype=binding.dtype) * 1e-6
        return tensor

    def fweights_value(binding):
        return rule.ops.cast(
            rule.ops.randint(1, 11, shape=(observation_count(),)),
            binding.dtype,
        )

    def aweights_value(binding):
        if binding.dtype in ["float32", "float64"]:
            return rule.ops.uniform(0.1, 1.0, shape=(observation_count(),), dtype=binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(1, 11, shape=(observation_count(),)),
            binding.dtype,
        )

    rule.generate(
        (
            ("x", x_value),
            ("fweights", fweights_value),
            ("aweights", aweights_value),
        ),
    )


@rules.register("paddle.linalg.eigh", "paddle.linalg.eigvalsh")
def eigen_symmetric_values(rule: RuleContext):
    """构造实对称或复 Hermitian 特征分解输入。"""

    def x_value(binding):
        if len(binding.shape) < 2 or binding.shape[-1] != binding.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = binding.shape[:-2]
        matrix_dim = binding.shape[-1]
        matrix = rule.ops.random([*batch_dims, matrix_dim, matrix_dim], dtype=binding.dtype)
        if binding.dtype in ["complex64", "complex128"]:
            matrix = matrix + 1j * rule.ops.random(
                [*batch_dims, matrix_dim, matrix_dim],
                dtype=binding.dtype,
            )
            tensor = matrix + rule.ops.conj(rule.ops.swapaxes(matrix, -1, -2))
        elif len(batch_dims) > 0:
            tensor = rule.ops.einsum("...ij,...kj->...ik", matrix, matrix)
        else:
            tensor = rule.ops.dot(matrix, rule.ops.swapaxes(matrix, -1, -2))
        tensor += rule.ops.eye(matrix_dim, dtype=binding.dtype) * 1e-6
        return tensor

    rule.generate((("x", x_value),))


@rules.register("paddle.linalg.lstsq")
def lstsq_values(rule: RuleContext):
    """为最小二乘生成至少二维且批次相容的矩阵。"""

    def matrix_value(binding):
        if len(binding.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for lstsq x")
        batch_dims = binding.shape[:-2]
        rows, cols = binding.shape[-2], binding.shape[-1]
        return rule.ops.random([*batch_dims, rows, cols], dtype=binding.dtype)

    rule.generate(((("x", "y"), matrix_value),))


@rules.register("paddle.linalg.lu_unpack")
def lu_unpack_values(rule: RuleContext):
    """生成非奇异 LU 数据和范围合法的 pivot。"""

    def x_value(binding):
        if len(binding.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for LU matrix")
        tensor = rule.ops.random(binding.shape, dtype=binding.dtype)
        diagonal_size = min(binding.shape[-2], binding.shape[-1])
        tensor[..., range(diagonal_size), range(diagonal_size)] += 1e-6
        return tensor

    def pivot_value(binding):
        row_count = rule.arg("x").shape[-2]
        return rule.ops.cast(
            rule.ops.randint(1, row_count + 1, shape=binding.shape),
            binding.dtype,
        )

    rule.generate(
        (
            ("x", x_value),
            (("pivot", "y"), pivot_value),
        ),
    )


@rules.register("paddle.linalg.cond")
def condition_values(rule: RuleContext):
    """构造数值稳定的方阵用于条件数计算。"""

    def x_value(binding):
        matrix_size = binding.shape[-1]
        tensor = rule.ops.random(binding.shape, dtype=binding.dtype)
        tensor += matrix_size * rule.ops.eye(matrix_size, dtype=binding.dtype)
        return tensor

    rule.generate((("x", x_value),))


@rules.register("paddle.linalg.det", "paddle.linalg.slogdet")
def determinant_values(rule: RuleContext):
    """构造可逆方阵用于 det 和 slogdet。"""

    def x_value(binding):
        if len(binding.shape) < 2:
            raise AssertionError("Input must be at least 2D.")
        if binding.shape[-1] != binding.shape[-2]:
            raise AssertionError("Input must be square matrices.")
        matrix_size = binding.shape[-1]
        is_complex = binding.dtype.startswith("complex")
        if is_complex:
            real_dtype = "float32" if binding.dtype == "complex64" else "float64"
            real = rule.ops.uniform(0.5, 1.0, shape=binding.shape, dtype=real_dtype)
            imag = rule.ops.uniform(0.5, 1.0, shape=binding.shape, dtype=real_dtype)
            matrix = rule.ops.cast(real + 1j * imag, binding.dtype)
            matrix_h = rule.ops.swapaxes(rule.ops.conj(matrix), -1, -2)
        else:
            matrix = rule.ops.uniform(0.5, 1.0, shape=binding.shape, dtype=binding.dtype)
            matrix_h = rule.ops.swapaxes(matrix, -1, -2)
        return rule.ops.matmul(matrix, matrix_h) + rule.ops.eye(matrix_size, dtype=binding.dtype)

    rule.generate((("x", x_value),))


@rules.register("paddle.linalg.pca_lowrank")
def pca_lowrank_values(rule: RuleContext):
    """为低秩 PCA 生成受控随机矩阵。"""

    def x_value(binding):
        return rule.ops.cast(rule.ops.randn(*binding.shape), binding.dtype)

    rule.generate((("x", x_value),))


@rules.register("paddle.linalg.corrcoef")
def corrcoef_values(rule: RuleContext):
    """为相关系数计算生成带微小扰动的非退化输入。"""

    def x_value(binding):
        if binding.dtype == "float16":
            return (
                rule.ops.cast(
                    rule.ops.randn(*binding.shape),
                    binding.dtype,
                )
                * 1e-3
            )
        return rule.default(binding)

    rule.generate((("x", x_value),))


@rules.register("paddle.linalg.pinv")
def pinv_values(rule: RuleContext):
    """按 hermitian 参数构造一般矩阵或 Hermitian 矩阵。"""
    hermitian = bool(rule.arg("hermitian", False))
    if not hermitian:
        rule.generate()
        return

    def x_value(tensor):
        if len(tensor.shape) not in [2, 3]:
            raise ValueError("pinv only supports 2D or 3D tensors")
        if tensor.dtype.startswith("complex"):
            real_dtype = "float32" if tensor.dtype == "complex64" else "float64"
            real = rule.ops.cast(rule.ops.randn(*tensor.shape), real_dtype)
            imag = rule.ops.cast(rule.ops.randn(*tensor.shape), real_dtype)
            matrix = rule.ops.cast(real + 1j * imag, tensor.dtype)
        else:
            matrix = rule.ops.cast(
                rule.ops.randn(*tensor.shape),
                tensor.dtype,
            )
        if len(tensor.shape) == 2:
            matrix_t = (
                rule.ops.swapaxes(rule.ops.conj(matrix), -1, -2)
                if tensor.dtype.startswith("complex")
                else rule.ops.swapaxes(matrix, -1, -2)
            )
        else:
            matrix_t = (
                rule.ops.swapaxes(rule.ops.conj(matrix), -2, -1)
                if tensor.dtype.startswith("complex")
                else rule.ops.swapaxes(matrix, -2, -1)
            )
        return (matrix + matrix_t) / 2

    rule.generate({"x": x_value})


@rules.register("paddle.linalg.cholesky_solve", aliases=("paddle.Tensor.cholesky_solve",))
def cholesky_solve_values(rule: RuleContext):
    """按 upper 参数生成与三角因子方向一致的输入。"""
    if rule.api_name == "paddle.linalg.cholesky_solve":
        rule.generate_all()
        return

    def y_value(binding):
        value = rule.domain("random_range", binding)
        if rule.arg("upper"):
            return rule.ops.triu(value)
        return rule.ops.tril(value)

    rule.generate((("y", y_value),))


@rules.register("paddle.view", aliases=("paddle.Tensor.view",))
def view_values(rule: RuleContext):
    """按目标 dtype 或 shape 约束 view 的底层字节布局。"""

    def x_value(binding):
        if binding.dtype == "uint8":
            target = str(rule.arg("shape_or_dtype", ""))
            nbytes = math.prod(binding.shape)
            itemsize = {
                "paddle.bfloat16": 2,
                "paddle.float16": 2,
                "paddle.float32": 4,
                "paddle.float64": 8,
            }.get(target)
            if itemsize is not None and nbytes % itemsize == 0:
                numel = nbytes // itemsize
                if target == "paddle.bfloat16":
                    finite_f32 = rule.ops.cast(
                        (rule.ops.random(numel) - 0.5) * 1.2,
                        "float32",
                    )
                    uint32_value = rule.ops.view_dtype(finite_f32, "uint32")
                    return rule.ops.view_dtype(
                        rule.ops.cast(
                            rule.ops.cast(uint32_value, "int64") >> 16,
                            "uint16",
                        ),
                        "uint8",
                    )
                finite = rule.ops.cast(
                    (rule.ops.random(numel) - 0.5) * 1.2,
                    target.replace("paddle.", ""),
                )
                return rule.ops.view_dtype(rule.ops.ascontiguousarray(finite), "uint8")
        return rule.default(binding)

    rule.generate((("x", x_value),))


@rules.register(
    "paddle.pow",
    aliases=("paddle.Tensor.pow", "paddle.Tensor.__rpow__", "paddle.Tensor.__pow__"),
)
def pow_values(rule: RuleContext):
    """根据底数、指数和正反向幂语义限制数值范围。"""

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
        api_name = rule.api_name
        dtype = binding.dtype
        if api_name == "paddle.Tensor.__rpow__":
            base_name, exponent_name = "y", "self"
        elif api_name == "paddle.Tensor.__pow__":
            base_name, exponent_name = "self", "y"
        else:
            base_name, exponent_name = "x", "y"
        is_base_arg = binding.parameter_name == base_name
        if is_base_arg:
            const = rule.arg(exponent_name)
            get_max = get_base_max
            default_max = 10
        else:
            const = rule.arg(base_name)
            get_max = get_exponent_max
            default_max = 5
        if isinstance(const, numbers.Number):
            value_max = get_max(const, rule.dtype_max(dtype), default_max)
            if is_base_arg and int(const) != const:
                return rule.domain("random_range", binding, low=0, high=value_max)
            return rule.domain("random_range", binding, low=-value_max, high=value_max)
        if is_base_arg:
            return rule.domain("random_range", binding, low=0, high=default_max)
        return rule.domain("random_range", binding, low=-default_max, high=default_max)

    rule.generate_all(value)


@rules.register("paddle.nn.functional.rnnt_loss")
def rnnt_loss_values(rule: RuleContext):
    """联动 logits、labels 和长度 Tensor 的默认形状。"""

    def logits(binding):
        shape = binding.shape if len(binding.shape) == 4 else (3, 4, 3, 5)
        return rule.ops.random(shape, dtype=binding.dtype)

    def labels(binding):
        shape = binding.shape if len(binding.shape) == 2 else (3, 2)
        return rule.ops.cast(rule.ops.randint(1, 4, shape=shape), binding.dtype)

    def lengths(max_possible_length):
        def generate(binding):
            shape = binding.shape if len(binding.shape) == 1 else (3,)
            return rule.ops.ones(shape, dtype=binding.dtype) * max_possible_length

        return generate

    rule.generate(
        (
            (("input", "logits"), logits),
            (("label", "labels"), labels),
            ("input_lengths", lengths(4)),
            ("label_lengths", lengths(2)),
        ),
    )


@rules.register("paddle.chunk")
def chunk_values(rule: RuleContext):
    """选择能被 chunks 整除的输入维度作为 axis。"""

    def axis_value(binding):
        x_tensor = rule.arg("x")
        chunks = rule.arg("chunks")
        valid_axes = [
            index for index, dim_size in enumerate(x_tensor.shape) if dim_size % chunks == 0
        ]
        if not valid_axes:
            raise ValueError(
                f"No valid axis found in x.shape = {x_tensor.shape} for chunks = {chunks}. "
                f"Each dim must be divisible by chunks."
            )
        chosen_axis = rule.ops.choice(valid_axes)
        if len(binding.shape) == 0:
            return rule.ops.asarray(chosen_axis, dtype=binding.dtype)
        if len(binding.shape) == 1 and binding.shape[0] == 1:
            return rule.ops.asarray([chosen_axis], dtype=binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.shape}."
        )

    rule.generate((("axis", axis_value),))


@rules.register("paddle.split")
def split_values(rule: RuleContext):
    """根据分段数量或 section 总和选择合法 axis。"""

    def axis_value(binding):
        x_shape = rule.arg("x").shape
        num_or_sections = rule.arg("num_or_sections")
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
            target_dim = rule.ops.randint(-1, 0)
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
        if len(binding.shape) == 0:
            return rule.ops.asarray(target_dim, dtype=binding.dtype)
        if len(binding.shape) == 1 and binding.shape[0] == 1:
            return rule.ops.asarray([target_dim], dtype=binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.split. "
            f"Expected a 0-D or 1-D Tensor, but got shape {binding.shape}."
        )

    rule.generate((("axis", axis_value),))


@rules.register("paddle.expand", aliases=("paddle.Tensor.expand",))
def expand_values(rule: RuleContext):
    """按源 shape 生成满足广播规则的目标 shape。"""

    def shape_value(binding):
        x_shape = rule.arg("x").shape
        shape_index = binding.path.indices[0] if binding.path.indices else 0
        if len(x_shape) == 0 or shape_index > len(x_shape) - 1 or x_shape[shape_index] == 1:
            return rule.ops.cast(
                rule.ops.randint(1, 127, shape=binding.shape),
                binding.dtype,
            )
        if len(binding.shape) == 0 or binding.shape[0] == 1:
            return rule.ops.asarray(x_shape[shape_index])
        shape_values = rule.ops.cast(
            rule.ops.randint(1, 127, shape=binding.shape),
            binding.dtype,
        )
        offset = binding.shape[0] - len(x_shape)
        for index in range(binding.shape[0]):
            if index >= offset and x_shape[index - offset] != 1:
                shape_values[index] = x_shape[index - offset]
        return shape_values

    rule.generate((("shape", shape_value),))


@rules.register("paddle.nn.functional.gather_tree")
def gather_tree_values(rule: RuleContext):
    """根据 beam size 生成合法父节点索引。"""

    def parents_value(binding):
        ids = rule.arg("ids")
        if hasattr(ids, "shape") and len(ids.shape) >= 3:
            beam_size = ids.shape[2]
        else:
            beam_size = binding.shape[2] if len(binding.shape) >= 3 else 4
        beam_size = 1 if beam_size < 1 else beam_size
        parents = rule.ops.zeros(binding.shape, dtype=binding.dtype)
        for time_index in range(binding.shape[0]):
            for batch_index in range(binding.shape[1]):
                for beam_index in range(binding.shape[2]):
                    parents[time_index, batch_index, beam_index] = rule.ops.randint(0, beam_size)
        return parents

    rule.generate((("parents", parents_value),))


@rules.register("paddle.multinomial")
def multinomial_values(rule: RuleContext):
    """生成非负权重并按 replacement 限制采样数量。"""
    x_binding = rule.tensor("x")
    num_samples_binding = rule.tensor("num_samples")
    if x_binding is not None:
        x_values = rule.ops.cast(
            rule.ops.abs(rule.ops.random(x_binding.shape)),
            x_binding.dtype,
        )
        rule.set(x_binding, x_values)
    if num_samples_binding is not None:
        replacement = rule.arg("replacement")
        if rule.has_kwarg("replacement") and replacement is True:
            max_allow = 1024
        else:
            x_values = rule.value(x_binding)
            max_allow = rule.ops.count_nonzero(x_values > 0)
        rule.set(
            num_samples_binding,
            rule.ops.cast(
                rule.ops.randint(
                    1,
                    max_allow + 1,
                    shape=num_samples_binding.shape,
                ),
                num_samples_binding.dtype,
            ),
        )
    rule.generate_remaining()


@rules.register("paddle.nn.functional.one_hot")
def one_hot_values(rule: RuleContext):
    """联动 num_classes 与输入索引的取值范围。"""
    x_binding = rule.tensor("x")
    num_classes_binding = rule.tensor("num_classes")
    num_classes_config = rule.arg("num_classes")
    default_random_num_classes = rule.ops.randint(1, 65535)
    if isinstance(num_classes_config, int):
        determined_num_classes = num_classes_config
    elif rule.is_tensor_config(num_classes_config):
        if num_classes_binding is not None and num_classes_config.numel() in {0, 1}:
            rule.set(
                num_classes_binding,
                rule.ops.asarray([default_random_num_classes], dtype="int64"),
            )
        determined_num_classes = rule.value(num_classes_binding).item()
    else:
        determined_num_classes = default_random_num_classes
    if x_binding is not None:
        rule.set(
            x_binding,
            rule.ops.randint(
                0,
                determined_num_classes,
                shape=x_binding.shape,
                dtype=x_binding.dtype,
            ),
        )
    rule.generate_remaining()


# Low-level config access.
def _tensor_config_at(api_config, path):
    value = api_config.args[path.key] if path.root == "args" else api_config.kwargs[path.key]
    for index in path.indices:
        value = value[index]
    return value


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
