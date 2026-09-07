"""输入生成规则的装饰器注册中心。"""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy

from .backend_runtime import create_input_backend
from .binding import InputApiBinding, InputGenerationContext, build_input_generation_context
from .tensor_config import (
    CAST_THROUGH_INTERMEDIATE_DTYPES,
    TensorConfig,
    shape_numel,
)
from .value_generators import (
    create_input_config_random_state,
    generate_abs_plus_one_input_value,
    generate_binary_input_value,
    generate_default_input_value,
    generate_dropout_probability_input_value,
    generate_empty_shape_input_value,
    generate_hinge_label_input_value,
    generate_int_64_input_value,
    generate_int_128_input_value,
    generate_int_1024_input_value,
    generate_int_2048_input_value,
    generate_int_2048_raw_input_value,
    generate_int_65535_raw_input_value,
    generate_int_or_unit_input_value,
    generate_multiply_input_value,
    generate_nonzero_input_value,
    generate_nonzero_symmetric_input_value,
    generate_normal_input_value,
    generate_normal_std_input_value,
    generate_ones_shape_input_value,
    generate_quantile_input_value,
    generate_random_range_input_value,
    generate_symmetric_input_value,
    generate_uniform_input_value,
    generate_unit_interval_input_value,
    generate_unit_plus_one_input_value,
)

# 规则层只通过 value 模块读取路径对象，避免再次实现 APIConfig 遍历。
from .values import (
    InputTensorSpec,
    InputValue,
    attach_input_values,
    input_tensor_config_at,
    read_input_value,
)

# API 值域策略属于规则层，不由 TensorConfig 物化模块持有。
not_zero_apis = frozenset(
    {
        "paddle.Tensor.__div__",
        "paddle.Tensor.__floordiv__",
        "paddle.Tensor.__mod__",
        "paddle.Tensor.__rdiv__",
        "paddle.Tensor.__rfloordiv__",
        "paddle.Tensor.__rmod__",
        "paddle.Tensor.__rtruediv__",
        "paddle.Tensor.__truediv__",
        "paddle.Tensor.divide",
        "paddle.Tensor.floor_divide",
        "paddle.Tensor.floor_mod",
        "paddle.Tensor.mod",
        "paddle.divide",
        "paddle.floor_divide",
        "paddle.floor_mod",
        "paddle.mod",
        "paddle.nn.functional.kl_div",
        "paddle.sparse.divide",
    }
)

# Core protocols and value-domain descriptors.
InputValueGenerator = Callable[..., object]
InputRuleFunction = Callable[["InputRuleContext"], None]

# `generate_*_inputs` 只描述 API 级参数关系，不能直接写回 APIConfig。
# `generate_*_input_value` 只生成单个绑定值，不负责完整性检查或 RNG 提交。

# 这些描述符字符串要保持稳定，因为规则体直接依赖它们。
_INPUT_VALUE_GENERATORS: dict[str, InputValueGenerator] = {
    "nonzero": lambda spec, low, high, rng: generate_nonzero_input_value(spec, rng),
    "unit_interval": lambda spec, low, high, rng: generate_unit_interval_input_value(spec, rng),
    "multiply": lambda spec, low, high, rng: generate_multiply_input_value(spec, rng),
    "unit_interval_plus_one": lambda spec, low, high, rng: generate_unit_plus_one_input_value(
        spec, rng
    ),
    "normal_std": lambda spec, low, high, rng: generate_normal_std_input_value(spec, rng),
    "dropout_probability": lambda spec, low, high, rng: generate_dropout_probability_input_value(
        spec, rng
    ),
    "quantile_q": lambda spec, low, high, rng: generate_quantile_input_value(spec, rng),
    "int_zero_1024": lambda spec, low, high, rng: generate_int_1024_input_value(spec, rng),
    "int_zero_64": lambda spec, low, high, rng: generate_int_64_input_value(spec, rng),
    "int_zero_2048_no_cast": lambda spec, low, high, rng: generate_int_2048_raw_input_value(
        spec, rng
    ),
    "empty_shape": lambda spec, low, high, rng: generate_empty_shape_input_value(spec, rng),
    "int_one_128": lambda spec, low, high, rng: generate_int_128_input_value(spec, rng),
    "int_one_2048": lambda spec, low, high, rng: generate_int_2048_input_value(spec, rng),
    "int_one_65535_no_cast": lambda spec, low, high, rng: generate_int_65535_raw_input_value(
        spec, rng
    ),
    "ones_shape": lambda spec, low, high, rng: generate_ones_shape_input_value(spec, rng),
    "int_zero_65535_else_unit": lambda spec, low, high, rng: generate_int_or_unit_input_value(
        spec, rng
    ),
    "binary_0_1": lambda spec, low, high, rng: generate_binary_input_value(spec, rng),
    "hinge_labels": lambda spec, low, high, rng: generate_hinge_label_input_value(spec, rng),
    "abs_unit_plus_one": lambda spec, low, high, rng: generate_abs_plus_one_input_value(spec, rng),
    "uniform": lambda spec, low, high, rng: generate_uniform_input_value(spec, low, high, rng),
    "random_range": lambda spec, low, high, rng: generate_random_range_input_value(
        spec, low, high, rng
    ),
}


# 规则执行分为“生成”和“提交”两个阶段。
# 规则函数只向暂存 writer 写值，不能直接修改原始 APIConfig。
# 完整性校验通过后才挂载 InputValue；配置级 RNG 始终独立于全局状态。
# 这条边界保证规则抛错时不会留下半生成输入或消耗后续配置随机流。
# backend 的 seed 与配置指纹则由 InputGenerationContext 传入，供原生 generator 使用。
# InputRuleContext 是规则作者唯一需要接触的接口，底层对象保持私有。
# InputRule 负责一次规则执行的生命周期，InputRuleContext 负责作者侧查询与生成。
# _InputValueWriter 只暂存路径级结果，最终提交集中发生在完整性检查之后。
@dataclass(frozen=True)
class InputRule:
    """一条通过装饰器注册的输入生成规则。

    这里仅保存执行入口，不保存 API 约束逻辑本身。
    规则函数负责描述参数关系，`InputRule` 负责完整性检查和提交。
    """

    function: InputRuleFunction

    def generate(
        self, input_generation_context: InputGenerationContext, api_config: object
    ) -> bool:
        # NumPy backend 需要局部 RandomState；原生 backend 直接消费 context 中的 seed 元数据。
        backend_policy = input_generation_context.backend_policy
        if backend_policy is None:
            raise ValueError("input backend policy is required for input generation")
        if backend_policy.resolved == "numpy":
            input_random_state = create_input_config_random_state(input_generation_context)
        else:
            input_random_state = input_generation_context
        input_backend = create_input_backend(
            input_random_state,
            policy=backend_policy,
        )
        input_rule_context = InputRuleContext(
            input_generation_context.input_binding,
            api_config,
            input_backend,
            input_generation_context.input_max_abs,
        )
        self.function(input_rule_context)
        # finish 先检查遗漏，再一次性同步 TensorConfig 元数据和逻辑值。
        attach_input_values(api_config, input_rule_context._finish())
        return True


class _InputValueWriter:
    """规则侧输入写入与完整性状态。"""

    def __init__(self, api_config: object, input_backend):
        self._api_config = api_config
        self._input_backend = input_backend
        self._input_value_by_path: dict[object, InputValue] = {}
        self._update_config_by_path: dict[object, bool] = {}

    def finish(self, rule):
        # 注册规则必须覆盖本次调用中的所有 Tensor，避免静默使用旧缓存。
        missing = [
            str(input_binding.path)
            for input_binding in rule.all_tensors
            if input_binding.path not in self._input_value_by_path
        ]
        if missing:
            raise ValueError(f"input rule {rule.api_name} left tensors ungenerated: {missing}")
        for path, input_value in self._input_value_by_path.items():
            _apply_input_value(
                self._api_config,
                input_value,
                update_config=self._update_config_by_path[path],
            )
        return tuple(self._input_value_by_path.values())

    def is_generated(self, input_binding):
        return input_binding.path in self._input_value_by_path

    def set_value(self, input_binding, input_value):
        if input_binding.path in self._input_value_by_path:
            raise ValueError(f"input rule generated tensor twice: {input_binding.path}")
        self._write_value(input_binding, input_value, update_config=True)

    def set_value_preserving_spec(self, input_binding, input_value):
        if input_binding.path in self._input_value_by_path:
            raise ValueError(f"input rule generated tensor twice: {input_binding.path}")
        self._write_value(input_binding, input_value, update_config=False)

    def value(self, input_binding):
        input_value = self._input_value_by_path.get(input_binding.path)
        if input_value is not None:
            return input_value.generated_value
        tensor_config = input_tensor_config_at(self._api_config, input_binding.path)
        return read_input_value(self._api_config, tensor_config)

    def _write_value(self, input_binding, input_value, update_config):
        # Torch 原生随机结果通常是新分配的；转移其所有权可省掉一次 GPU clone。
        # Paddle 无可靠视图标记，仍保持复制以隔离规则内的原地修改。
        copy_value = not self._can_transfer_ownership(input_value)
        stored_input_value = self._input_backend.asarray(input_value, copy=copy_value)
        self._input_value_by_path[input_binding.path] = InputValue(
            input_binding.path,
            stored_input_value,
            self._input_backend.name,
        )
        self._update_config_by_path[input_binding.path] = update_config

    def _can_transfer_ownership(self, input_value):
        """判断值是否可由 writer 独占，避免破坏规则的别名隔离。"""
        backend_name = self._input_backend.name
        if backend_name != "torch":
            return False
        # 同一对象已被其他路径保存时，后续写入必须建立独立 storage。
        if any(item.generated_value is input_value for item in self._input_value_by_path.values()):
            return False
        # 视图共享底层 storage，转移视图会让调用方仍可原地改变已保存输入。
        if getattr(input_value, "_base", None) is not None:
            return False
        return True


class InputRuleContext:
    """面向规则作者的单一输入生成接口。"""

    def __init__(
        self,
        input_binding: InputApiBinding,
        api_config: object,
        input_backend,
        input_max_abs,
    ):
        self._input_binding = input_binding
        self._api_config = api_config
        self._input_backend = input_backend
        self._input_max_abs = input_max_abs
        self._input_value_writer = _InputValueWriter(api_config, input_backend)

    @property
    def api_name(self):
        return self._input_binding.api_name

    @property
    def all_tensors(self):
        return self._input_binding.tensor_bindings

    @property
    def ops(self):
        return self._input_backend

    def arg(self, name, default=None):
        # 优先读取签名绑定结果，使位置参数和关键字参数具有同一名称入口。
        for parameter_name, value in self._input_binding.arguments:
            if parameter_name == name:
                return value
        return self._api_config.kwargs.get(name, default)

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
            tensor
            for tensor in self._input_binding.tensor_bindings
            if tensor.parameter_name == parameter_name
        )

    def binding_for_value(self, value):
        # identity 查询用于参数值嵌套或参数名不足以区分目标的少数规则。
        if not self.is_tensor_config(value):
            return None
        for tensor in self._input_binding.tensor_bindings:
            if input_tensor_config_at(self._api_config, tensor.path) is value:
                return tensor
        return None

    def has_kwarg(self, name):
        return name in self._api_config.kwargs

    def kwarg(self, name, default=None):
        return self._api_config.kwargs.get(name, default)

    def argument_values(self):
        return (*self._api_config.args, *self._api_config.kwargs.values())

    def is_tensor_config(self, value):
        return isinstance(value, TensorConfig)

    def domain(self, generator, tensor, low=None, high=None):
        # 值域名称在写入前集中校验，拼写错误不会产生部分输入。
        if generator == "default":
            if low is not None or high is not None:
                raise ValueError("default input generator does not accept explicit bounds")
            return self.default(tensor)
        generate_value = _INPUT_VALUE_GENERATORS.get(generator)
        if generate_value is None:
            raise ValueError(f"unknown input value generator {generator!r} for {tensor.path}")
        return generate_value(tensor.input_spec, low, high, self._input_backend)

    def default(self, tensor, *, shape=None):
        # shape-only 规则复用同一默认值域，不能在规则体内复制随机公式。
        spec = (
            tensor.input_spec if shape is None else replace(tensor.input_spec, shape=tuple(shape))
        )
        return generate_default_input_value(
            spec,
            self._input_backend,
            max_abs=self._input_max_abs,
        )

    def uniform(self, tensor, low, high, *, shape=None):
        # uniform 统一负责复数实部和虚部，规则层不得通过实数 cast 构造复数。
        spec = (
            tensor.input_spec if shape is None else replace(tensor.input_spec, shape=tuple(shape))
        )
        return generate_uniform_input_value(spec, low, high, self._input_backend)

    def normal(self, tensor, *, shape=None, scale=1.0):
        spec = (
            tensor.input_spec if shape is None else replace(tensor.input_spec, shape=tuple(shape))
        )
        return generate_normal_input_value(spec, self._input_backend, scale=scale)

    def default_nonzero(self, tensor):
        # 除数继承 default 范围，但量化为零时必须遵守非零协议。
        return generate_nonzero_symmetric_input_value(
            tensor.input_spec,
            self._input_max_abs,
            self._input_backend,
        )

    def set(self, tensor, value):
        self._input_value_writer.set_value(tensor, value)

    def set_preserving_spec(self, tensor, value):
        # shape 参数等特殊值可改变逻辑数组形状，但调用侧仍需保留原始规格。
        self._input_value_writer.set_value_preserving_spec(tensor, value)

    def value(self, tensor):
        return self._input_value_writer.value(tensor)

    def is_generated(self, tensor):
        return self._input_value_writer.is_generated(tensor)

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
            _generate_input_binding_value(self, tensor, generator, low, high)

    def generate_remaining(self, generator="default"):
        # 关系规则先写关键 Tensor，再由该入口补齐未处理的普通参数。
        self._validate_generator(generator)
        for tensor in self.all_tensors:
            if not self.is_generated(tensor):
                _generate_input_binding_value(self, tensor, generator)

    def generate(self, parameter_generators=None, *, default="default"):
        # mapping 的 key 可以是名称组，用于表达同一语义在不同 API 中的命名。
        if parameter_generators is None:
            parameter_generators = {}
        items = (
            parameter_generators.items()
            if isinstance(parameter_generators, Mapping)
            else parameter_generators
        )

        known_names = set(self._input_binding.parameter_names)
        normalized = []
        for parameter_names, generator in items:
            names = (
                (parameter_names,) if isinstance(parameter_names, str) else tuple(parameter_names)
            )
            if not names or any(not isinstance(name, str) or not name for name in names):
                raise ValueError("rule.generate parameter names must be non-empty strings")
            # 名称组只要求一个候选命中；单名称仍能捕获规则拼写错误。
            if known_names.isdisjoint(names) and self._input_binding.binding_source != "unresolved":
                raise ValueError(
                    f"rule {self.api_name} declares parameters absent from its signature: "
                    f"{sorted(names)}"
                )
            self._validate_generator(generator)
            normalized.append((names, generator))

        self._validate_generator(default)

        for input_binding in self.all_tensors:
            # 映射按声明顺序匹配，首个命中的策略拥有该 Tensor。
            generator = None
            for names, candidate in normalized:
                if input_binding.parameter_name in names:
                    generator = candidate
                    break
            if generator is None:
                generator = default
            if generator is not None:
                _generate_input_binding_value(self, input_binding, generator)

    def _validate_generator(self, generator):
        if (
            isinstance(generator, str)
            and generator != "default"
            and generator not in _INPUT_VALUE_GENERATORS
        ):
            raise ValueError(f"unknown input value generator {generator!r} for {self.api_name}")

    def _numeric_dtype(self, dtype):
        return self._input_backend.resolve_input_dtype(str(dtype).replace("paddle.", ""))

    @staticmethod
    def _real_dtype(dtype):
        if dtype == "complex64":
            return "float32"
        if dtype == "complex128":
            return "float64"
        return dtype

    def _finish(self):
        return self._input_value_writer.finish(self)


def _generate_input_binding_value(
    rule: InputRuleContext,
    input_binding,
    input_value_generator,
    low=None,
    high=None,
):
    if callable(input_value_generator):
        input_value = input_value_generator(input_binding)
    else:
        input_value = rule.domain(input_value_generator, input_binding, low, high)
    rule.set(input_binding, input_value)


# Registration infrastructure.
def _validate_input_rule_function(function: InputRuleFunction) -> None:
    # 导入阶段校验三参数协议，尽早暴露注册错误。
    parameters = tuple(inspect.signature(function).parameters.values())
    if any(
        parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        for parameter in parameters
    ):
        raise TypeError(f"input-generation rule {function.__name__} cannot use variadic parameters")
    if len(parameters) != 1:
        raise TypeError(
            f"input-generation rule {function.__name__} must accept one InputRuleContext"
        )


class InputRuleRegistry:
    """失败即止的装饰器注册表。"""

    # 注册表在模块导入期间完成构建，任何 API 所有权冲突都会阻止进程启动。
    def __init__(self, default_rule: InputRule):
        self._default_input_rule = default_rule
        self._by_api: dict[str, InputRule] = {}

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
            raise ValueError("registered input rule must declare at least one API")

        def decorator(function: InputRuleFunction):
            # 注册表在模块导入时完成构建，API 重叠必须立即失败。
            _validate_input_rule_function(function)
            for api_name in names:
                if api_name in self._by_api:
                    raise ValueError(f"input-generation API overlap: {api_name}")
            rule = InputRule(
                function=function,
            )
            for api_name in names:
                self._by_api[api_name] = rule
            return function

        return decorator

    def resolve(self, api_name: str) -> InputRule:
        # 未注册 API 直接返回独立默认规则，不伪造任意 API 的注册关系。
        return self._by_api.get(api_name, self._default_input_rule)

    def generate(self, api_test_case) -> bool:
        """为测试用例创建上下文并执行一次规则事务。"""
        api_config = api_test_case.api_config
        input_rule = self.resolve(api_config.api_name)
        # 上下文在 registry owner 内构造，确保规则选择与提交共享同一入口。
        input_generation_context = build_input_generation_context(
            api_config,
            seed=api_test_case.runtime_config.random_seed,
            backend_policy=api_test_case.runtime_config.input_backend_policy,
            input_max_abs=api_test_case.runtime_config.input_max_abs,
        )
        return input_rule.generate(input_generation_context, api_config)


# API 输入规则集中定义，确保注册顺序和查找结果保持稳定。
def generate_default_inputs(rule: InputRuleContext):
    """为未注册 API 使用默认值域生成全部 Tensor。"""
    rule.generate()


DEFAULT_INPUT_RULE = InputRule(function=generate_default_inputs)
input_rules = InputRuleRegistry(DEFAULT_INPUT_RULE)

# 以下顶层函数均属于 API 输入规则方法族，名称固定使用 `generate_*_inputs`。


# 融合与注意力规则联动序列长度、缓存布局和量化参数。
@input_rules.register("paddle.incubate.nn.functional.fused_act_dequant")
def generate_fused_act_dequant_inputs(rule: InputRuleContext):
    """输入规则：为 int32 x_scale 生成满足重复字节编码的指数值。"""

    def generate_x_scale_input_value(tensor):
        if tensor.dtype == "int32":
            exponent = rule.ops.randint(120, 128, shape=tensor.shape, dtype="int32")
            return exponent * rule.ops.asarray(0x01010101, dtype="int32")
        return rule.default(tensor)

    rule.generate({"x_scale": generate_x_scale_input_value})


@input_rules.register("paddle.incubate.nn.functional.variable_length_memory_efficient_attention")
def generate_variable_length_memory_efficient_attention_inputs(rule: InputRuleContext):
    """输入规则：约束序列长度、KV 长度和注意力 mask 的有效范围。"""

    def generate_seq_lens_input_value(input_binding):
        query = rule.arg("query")
        q_seq_len = query.shape[2]
        return rule.domain("random_range", input_binding, 1, q_seq_len)

    def generate_kv_seq_lens_input_value(input_binding):
        key = rule.arg("key")
        value = rule.arg("value")
        max_seq_len = min(key.shape[2], value.shape[2])
        return rule.domain("random_range", input_binding, 1, max_seq_len)

    def generate_mask_input_value(input_binding):
        return rule.ops.cast(
            rule.ops.randint(0, 2, shape=input_binding.shape),
            input_binding.dtype,
        ) * rule.dtype_min(input_binding.dtype)

    rule.generate(
        (
            ("seq_lens", generate_seq_lens_input_value),
            ("kv_seq_lens", generate_kv_seq_lens_input_value),
            ("mask", generate_mask_input_value),
        ),
    )


@input_rules.register(
    "paddle.incubate.nn.functional.block_multihead_attention",
)
def generate_block_multihead_attention_inputs(rule: InputRuleContext):
    """输入规则：联动生成缓存、序列长度、padding offset 和量化参数。"""
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

    def generate_sequence_length_input_value(input_binding):
        return rule.ops.asarray([seq_len] * batch_size, dtype=input_binding.dtype)

    def write_padding_offset_inputs(input_binding):
        seq_lens_this_time = rule.value(rule.tensor("seq_lens_this_time"))
        cum_offsets_now = rule.ops.cumsum(seq_len - seq_lens_this_time)
        cum_offsets_binding = rule.tensor("cum_offsets")
        cu_seqlens_q_binding = rule.tensor("cu_seqlens_q")
        cu_seqlens_k_binding = rule.tensor("cu_seqlens_k")
        cum_offsets = rule.ops.zeros((batch_size + 1,), dtype=cum_offsets_binding.dtype)
        cum_offsets[1:] = cum_offsets_now
        token_num = rule.ops.sum(seq_lens_this_time)
        padding_offsets = rule.ops.zeros((token_num,), dtype=input_binding.dtype)
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
        rule.set(input_binding, padding_offsets)

    for input_binding in rule.all_tensors:
        if rule.is_generated(input_binding):
            continue
        if input_binding.parameter_name in zero_parameters:
            rule.set(input_binding, rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype))
        elif input_binding.parameter_name == "seq_lens_encoder":
            rule.set(
                input_binding,
                generate_sequence_length_input_value(input_binding),
            )
        elif input_binding.parameter_name == "seq_lens_this_time":
            rule.set(input_binding, rule.value(rule.tensor("seq_lens_encoder")))
        elif input_binding.parameter_name == "padding_offsets":
            write_padding_offset_inputs(input_binding)
        elif input_binding.parameter_name in positive_range_parameters:
            rule.set(input_binding, rule.domain("random_range", input_binding, low=0))
        elif input_binding.parameter_name == "max_enc_len_this_time":
            rule.set(
                input_binding,
                generate_sequence_length_input_value(input_binding),
            )
        elif input_binding.parameter_name in {"mask", "tgt_mask"}:
            rule.set(
                input_binding,
                rule.domain(
                    "random_range", input_binding, high=rule.dtype_eps(input_binding.dtype)
                ),
            )
        else:
            rule.set(input_binding, rule.default(input_binding))


@input_rules.register(
    "paddle._C_ops.adam_",
    "paddle._C_ops.adamw_",
    "paddle._C_ops.merged_adam_",
)
def generate_optimizer_inputs(rule: InputRuleContext):
    """输入规则：初始化优化器状态，并按 beta 与随机步数计算幂累计值。"""
    zero_parameters = {"moment1", "moment2", "moment2_max"}
    optimizer_step = None

    def generate_beta_pow_input_value(input_binding, beta, step):
        import paddle

        use_accuracy_compatible = paddle.get_flags("FLAGS_use_accuracy_compatible_kernel")[
            "FLAGS_use_accuracy_compatible_kernel"
        ]
        if use_accuracy_compatible:
            beta_pow = beta**step
        else:
            # backend-native 标量保留 float32 幂语义，不把 Paddle/Torch 值交给 NumPy。
            step = step.item() if hasattr(step, "item") else step
            beta_pow = rule.ops.power(
                rule.ops.full((), beta, dtype="float32"),
                rule.ops.full((), int(step), dtype="float32"),
            )
            beta_pow = beta_pow.item() if hasattr(beta_pow, "item") else beta_pow
        return rule.ops.full(
            input_binding.shape,
            beta_pow,
            dtype=input_binding.dtype,
        )

    def generate_optimizer_input_value(input_binding):
        nonlocal optimizer_step
        if input_binding.parameter_name in zero_parameters:
            return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        if rule.api_name == "paddle._C_ops.adamw_" and input_binding.parameter_name in {
            "beta1_pow",
            "beta2_pow",
        }:
            if optimizer_step is None:
                optimizer_step = rule.ops.randint(1, 101)
            beta = rule.arg("beta1")
            if input_binding.parameter_name == "beta2_pow":
                beta = rule.arg("beta2")
            return generate_beta_pow_input_value(input_binding, beta, optimizer_step)
        return rule.default(input_binding)

    rule.generate_all(generate_optimizer_input_value)


@input_rules.register(
    "paddle.nn.functional.max_unpool1d",
    "paddle.nn.functional.max_unpool2d",
    "paddle.nn.functional.max_unpool3d",
)
def generate_max_unpool_inputs(rule: InputRuleContext):
    """输入规则：根据池化参数同步构造输入值和合法的反池化索引。"""

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
        elif padding is None:
            # Paddle 默认 padding 为各空间维度的零值，不能对 None 做 list 转换。
            padding = [0] * dimensions
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
    pool_input_spec = InputTensorSpec(
        shape=tuple(pool_input_size),
        dtype=data_type,
        place=x_binding.place,
        is_contiguous=x_binding.is_contiguous,
        strides=x_binding.strides,
    )
    pool_input = generate_random_range_input_value(pool_input_spec, low=-5, high=5, rng=rule.ops)
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


@input_rules.register("paddle.arange")
def generate_arange_inputs(rule: InputRuleContext):
    """输入规则：联合约束 start、end、step，避免空区间和零步长。"""

    def tensor_binding(value):
        return rule.binding_for_value(value)

    def set_tensor(value, tensor_value):
        input_binding = tensor_binding(value)
        if input_binding is not None:
            rule.set(input_binding, tensor_value)

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

    for input_binding in rule.all_tensors:
        if rule.value(input_binding) is None:
            handle_arange_relation()


# MoE 规则共同约束 token 路由、专家概率和压缩前后的行映射。
@input_rules.register("paddle.nn.functional.moe_permute")
def generate_moe_permute_inputs(rule: InputRuleContext):
    """输入规则：按专家路由关系生成 MoE 排列所需的映射和概率。"""

    def generate_expert_routemap_input_value(input_binding):
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
        if input_binding.dtype != "int32":
            raise ValueError("expert_routemap_topk dtype must be int32")
        if not rule.is_tensor_config(expert_prob) or (
            len(expert_prob.shape) != 2 or expert_prob.dtype != "float32"
        ):
            raise ValueError("expert_prob_topk must be a rank-2 float32 tensor")
        seqlen, topk = input_binding.shape[0], input_binding.shape[1]
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

    def generate_expert_prob_input_value(input_binding):
        routemap_binding = rule.tensor("expert_routemap_topk")
        probs = rule.ops.zeros(input_binding.shape, dtype="float32")
        if routemap_binding is not None and rule.value(routemap_binding) is not None:
            # Paddle 不对 float 与 bool 做隐式类型提升，路由掩码在规则层明确转为概率 dtype。
            mask = rule.ops.cast(rule.value(routemap_binding) >= 0, "float32")
            raw = rule.ops.cast(rule.ops.random(input_binding.shape), "float32") * mask
            row_sums = rule.ops.sum(raw, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = raw / row_sums
        else:
            probs = rule.ops.cast(rule.ops.random(input_binding.shape), "float32")
            row_sums = rule.ops.sum(probs, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums
        return probs

    rule.generate(
        (
            ("expert_routemap_topk", generate_expert_routemap_input_value),
            ("expert_prob_topk", generate_expert_prob_input_value),
        ),
    )


@input_rules.register("paddle.nn.functional.moe_unpermute")
def generate_moe_unpermute_inputs(rule: InputRuleContext):
    """输入规则：按压缩行映射恢复 MoE 反排列输入之间的形状关系。"""

    def generate_expert_routemap_input_value(input_binding):
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
        if input_binding.dtype != "int32" or len(input_binding.shape) != 2:
            raise ValueError("expert_routemap_topk must be a rank-2 int32 tensor")
        seqlen, topk = input_binding.shape[0], input_binding.shape[1]
        if seqlen != total_zipped_tokens:
            raise ValueError("expert_routemap_topk sequence length must equal total_zipped_tokens")
        if topk <= 0:
            raise ValueError("topk should be greater than 0")
        routemap = rule.ops.full(input_binding.shape, -1, dtype="int32")
        max_assign = min(topk, num_experts)
        route_count = min(hidden_config.shape[0], seqlen * max_assign)
        positions = rule.ops.arange(route_count, dtype="int64")
        rows = positions % seqlen
        columns = positions // seqlen
        routemap[rows, columns] = rule.ops.cast((rows + columns) % num_experts, "int32")
        return routemap

    def generate_rowmap_input_value(input_binding):
        routemap_binding = rule.tensor("expert_routemap_topk")
        if routemap_binding is not None and not rule.is_generated(routemap_binding):
            rule.set(routemap_binding, generate_expert_routemap_input_value(routemap_binding))
        routemap_config = rule.arg("expert_routemap_topk")
        num_experts = rule.arg("num_experts", 32)
        total_zipped_tokens = rule.arg("total_zipped_tokens")
        hidden_config = rule.arg("hidden_states_unzipped")
        seqlen = total_zipped_tokens
        unzipped_seqlen = hidden_config.shape[0] if rule.is_tensor_config(hidden_config) else seqlen
        if input_binding.dtype != "int32" or tuple(input_binding.shape) != (seqlen, num_experts):
            raise ValueError(
                "zipped_expertwise_rowmap must have shape "
                "[total_zipped_tokens, num_experts] and dtype int32"
            )
        rowmap = rule.ops.full(input_binding.shape, -1, dtype="int32")
        if rule.is_tensor_config(routemap_config) and routemap_binding is not None:
            routemap = rule.value(routemap_binding)
            # present[row, expert] 标记该 token 是否被路由到该 expert；只按 topk 迭代，
            # 避免 seqlen x num_experts 级别的 python 双循环。
            present = rule.ops.zeros(input_binding.shape, dtype="int64")
            for topk_index in range(routemap.shape[1]):
                column = rule.ops.cast(routemap[:, topk_index], "int64")
                selected = rule.ops.nonzero(column >= 0)[0]
                # Paddle 不接受空高级索引，空列保持 present 全 0。
                if selected.shape[0] == 0:
                    continue
                present[selected, column[selected]] = 1
            expert_counts = rule.ops.cast(rule.ops.sum(present, axis=0), "int64")
            if int(rule.ops.sum(expert_counts)) > unzipped_seqlen:
                raise ValueError("routemap assignments exceed hidden_states_unzipped capacity")
            expert_offsets = rule.ops.zeros(num_experts, dtype="int64")
            expert_offsets[1:] = rule.ops.cumsum(expert_counts[:-1])
            # 每个 expert 内按 row 升序自增编号，等价于列向 present 的 exclusive cumsum。
            ranks = rule.ops.cumsum(present, axis=0) - 1
            rowmap = rule.ops.cast(rule.ops.where(present > 0, expert_offsets + ranks, -1), "int32")
        return rowmap

    def generate_token_prob_input_value(input_binding):
        hidden_config = rule.arg("hidden_states_unzipped")
        if not (
            input_binding.dtype == "float32"
            and len(input_binding.shape) in (1, 2)
            and rule.is_tensor_config(hidden_config)
            and input_binding.shape[0] == hidden_config.shape[0]
            and (len(input_binding.shape) == 1 or input_binding.shape[1] == 1)
        ):
            raise ValueError(
                "token_prob_unzipped must match the broadcasted sequence "
                "length and have dtype float32"
            )
        return rule.ops.cast(rule.ops.random(input_binding.shape), "float32")

    for input_binding in rule.all_tensors:
        if rule.is_generated(input_binding):
            continue
        if input_binding.parameter_name == "expert_routemap_topk":
            rule.set(input_binding, generate_expert_routemap_input_value(input_binding))
        elif input_binding.parameter_name == "zipped_expertwise_rowmap":
            rule.set(input_binding, generate_rowmap_input_value(input_binding))
        elif input_binding.parameter_name == "token_prob_unzipped":
            rule.set(input_binding, generate_token_prob_input_value(input_binding))
        else:
            rule.set(input_binding, rule.default(input_binding))


# 基础值域规则覆盖非零、概率、开方和逐元素算子的通用约束。
@input_rules.register(*tuple(sorted(not_zero_apis)))
def generate_nonzero_inputs(rule: InputRuleContext):
    """输入规则：为除法及对数类算子生成非零输入。"""
    rule.generate_all("nonzero")


@input_rules.register("paddle.bernoulli")
def generate_bernoulli_inputs(rule: InputRuleContext):
    """输入规则：将伯努利分布概率限制在单位区间。"""
    rule.generate_all("unit_interval")


@input_rules.register("paddle.standard_gamma")
def generate_standard_gamma_inputs(rule: InputRuleContext):
    """输入规则：为 Gamma 分布生成单位区间内的正参数。"""
    rule.generate_all("unit_interval")


@input_rules.register("paddle.poisson")
def generate_poisson_inputs(rule: InputRuleContext):
    """输入规则：为 Poisson 分布生成非负强度参数。"""
    rule.generate_all("unit_interval")


@input_rules.register(
    "paddle.sqrt",
    aliases=("paddle.Tensor.sqrt",),
)
def generate_sqrt_inputs(rule: InputRuleContext):
    """输入规则：为平方根生成严格正输入。"""
    # sqrt 在零点前向有定义，但反向导数为 Inf，普通随机用例需避开该奇异点。
    rule.generate_all("uniform", low=0.1, high=1000)


@input_rules.register(
    "paddle.acosh",
    aliases=("paddle.Tensor.acosh",),
)
def generate_acosh_inputs(rule: InputRuleContext):
    """输入规则：为反双曲余弦生成严格大于一的输入。"""
    # acosh(1) 的前向有限但反向导数为 Inf，因此为定义域边界保留余量。
    rule.generate_all("uniform", low=1.1, high=1000)


@input_rules.register(
    "paddle.reciprocal",
    aliases=("paddle.Tensor.reciprocal",),
)
def generate_reciprocal_inputs(rule: InputRuleContext):
    """输入规则：为倒数运算生成远离零点的输入。"""
    rule.generate_all("uniform", low=0.5, high=2.0)


@input_rules.register(
    "paddle.digamma",
    "paddle.lgamma",
    "paddle.polygamma",
    aliases=("paddle.Tensor.digamma", "paddle.Tensor.lgamma"),
)
def generate_gamma_function_inputs(rule: InputRuleContext):
    """输入规则：为 Gamma 派生函数生成正区间输入。"""
    # polygamma 的 n 是普通整数；全量生成只会处理 Tensor 自变量 x，不会改写阶数。
    # 正区间同时避开 0 与全部非正整数极点，并与极点保留足够数值距离。
    rule.generate_all("uniform", low=0.5, high=10.0)


@input_rules.register(
    "paddle.rsqrt",
    aliases=("paddle.Tensor.rsqrt",),
)
def generate_rsqrt_inputs(rule: InputRuleContext):
    """输入规则：为倒数平方根生成严格为正的输入。"""
    rule.generate_all("uniform", low=1e-7, high=1000)


# 有界算术规则维护上下界、除数和广播参数的有效范围。
@input_rules.register("paddle.clip", aliases=("paddle.Tensor.clip",))
def generate_clip_inputs(rule: InputRuleContext):
    """输入规则：联动 min 与 max，保证裁剪上下界有序。"""
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


@input_rules.register(
    "paddle.multiply",
    aliases=("paddle.Tensor.__mul__", "paddle.Tensor.multiply", "paddle.Tensor.__rmul__"),
)
def generate_multiply_inputs(rule: InputRuleContext):
    """输入规则：收紧乘法输入值域以降低溢出风险。"""
    rule.generate_all("multiply")


@input_rules.register(
    "paddle.nn.functional.binary_cross_entropy",
)
def generate_binary_cross_entropy_inputs(rule: InputRuleContext):
    """输入规则：按概率、标签和权重语义生成二元交叉熵输入。"""

    def generate_probability_input_value(input_binding):
        # input 远离 0/1，避免合法端点在反向公式中形成 0/0。
        return rule.domain("uniform", input_binding, low=0.05, high=0.95)

    def generate_label_input_value(input_binding):
        # label 允许概率软标签，包括数学上合法的端点。
        return rule.domain("unit_interval", input_binding)

    def generate_weight_input_value(input_binding):
        # BCE 权重只缩放 loss，使用正有限范围避免随机负权重改变损失语义。
        return rule.domain("uniform", input_binding, low=0.1, high=1.0)

    rule.generate(
        (
            (("input", "x"), generate_probability_input_value),
            (("label", "target"), generate_label_input_value),
            ("weight", generate_weight_input_value),
        )
    )


@input_rules.register("paddle.nn.functional.binary_cross_entropy_with_logits")
def generate_binary_cross_entropy_with_logits_inputs(rule: InputRuleContext):
    """输入规则：保持 logits、软标签和权重处于有限数值域。"""

    def generate_logit_input_value(input_binding):
        # 有界 logits 既覆盖正负区域，也避免低精度 exp 路径出现不必要的溢出。
        return rule.domain("uniform", input_binding, low=-10.0, high=10.0)

    def generate_positive_weight_input_value(input_binding):
        # weight 与 pos_weight 都是 loss 的非负缩放因子，使用严格正值保留梯度。
        return rule.domain("uniform", input_binding, low=0.1, high=2.0)

    rule.generate(
        (
            (("logit", "input"), generate_logit_input_value),
            (("label", "target"), "unit_interval"),
            (("weight", "pos_weight"), generate_positive_weight_input_value),
        )
    )


@input_rules.register("paddle.nn.functional.margin_ranking_loss")
def generate_margin_ranking_loss_inputs(rule: InputRuleContext):
    """输入规则：为 margin ranking loss 生成二元排序标签。"""
    rule.generate((("label", "hinge_labels"),))


@input_rules.register("paddle.nn.functional.poisson_nll_loss")
def generate_poisson_nll_loss_inputs(rule: InputRuleContext):
    """输入规则：按 log_input 语义约束 Poisson 预测值与计数标签。"""

    def generate_input_value(input_binding):
        if rule.arg("log_input", True):
            # log-rate 保持有界，避免 exp(input) 在 float16 等低精度类型中溢出。
            return rule.domain("uniform", input_binding, low=-4.0, high=4.0)
        # rate 路径会计算 log(input + epsilon)，严格正值避免定义域边界。
        return rule.domain("uniform", input_binding, low=0.1, high=10.0)

    def generate_label_input_value(input_binding):
        # Poisson target 表示非负计数；连续值也属于该 API 接受的目标范围。
        return rule.domain("uniform", input_binding, low=0.0, high=10.0)

    rule.generate(
        (
            ("input", generate_input_value),
            (("label", "target"), generate_label_input_value),
        )
    )


@input_rules.register("paddle.nn.functional.soft_margin_loss")
def generate_soft_margin_loss_inputs(rule: InputRuleContext):
    """输入规则：为 soft margin loss 生成有界得分和二元标签。"""

    def generate_input_value(input_binding):
        # softplus(-label * input) 对大幅值低精度输入敏感，普通用例限制到稳定区间。
        return rule.domain("uniform", input_binding, low=-10.0, high=10.0)

    rule.generate(
        (
            ("input", generate_input_value),
            ("label", "hinge_labels"),
        )
    )


@input_rules.register("paddle.nn.functional.batch_norm")
def generate_batch_norm_inputs(rule: InputRuleContext):
    """输入规则：保持 batch norm 统计量与仿射参数的数值语义。"""

    def generate_running_var_input_value(input_binding):
        # 推理和全局统计路径会直接开方 running_var，因此必须严格为正。
        return rule.domain("uniform", input_binding, low=0.5, high=1.5)

    def generate_weight_input_value(input_binding):
        # 仿射 scale 保持正值可避免额外符号翻转，同时继续覆盖非单位缩放。
        return rule.domain("uniform", input_binding, low=0.5, high=1.5)

    rule.generate(
        (
            ("running_var", generate_running_var_input_value),
            ("weight", generate_weight_input_value),
        )
    )


@input_rules.register("paddle.nn.functional.alpha_dropout")
def generate_alpha_dropout_inputs(rule: InputRuleContext):
    """输入规则：为 alpha dropout 生成符合概率语义的输入。"""
    rule.generate((("x", "unit_interval"),))


@input_rules.register("paddle.nn.functional.conv2d_transpose")
def generate_conv2d_transpose_inputs(rule: InputRuleContext):
    """输入规则：为转置卷积生成受控范围内的输入、权重和偏置。"""

    def generate_tensor_input_value(input_binding):
        if "int" in input_binding.dtype:
            return rule.ops.cast(
                rule.ops.randint(-65535, 65535, shape=input_binding.shape),
                input_binding.dtype,
            )
        return rule.ops.cast(
            rule.ops.random(input_binding.shape) - 0.5,
            input_binding.dtype,
        )

    rule.generate(
        ((("x", "weight", "bias"), generate_tensor_input_value),),
    )


# 视觉候选框规则联动图像尺寸、框坐标、分数和每批框数量。
@input_rules.register("paddle.vision.ops.distribute_fpn_proposals")
def generate_distribute_fpn_proposals_inputs(rule: InputRuleContext):
    """输入规则：生成有效 ROI 坐标并保持各层 rois_num 总数一致。"""
    state = {"num": None}

    def generate_fpn_rois_input_value(input_binding):
        num = input_binding.shape[0]
        state["num"] = num
        # ROI 坐标需要保留随机小数，先转换到配置浮点 dtype 再做坐标联动。
        rois = rule.ops.cast(
            rule.ops.randint(1, 1024, shape=[num, 4]),
            input_binding.dtype,
        )
        rois[:, 0] = rois[:, 0] + rule.ops.random([num], dtype=input_binding.dtype)
        rois[:, 1] = rois[:, 1] + rule.ops.random([num], dtype=input_binding.dtype)
        widths = rule.ops.cast(
            rule.ops.randint(1, 1024, shape=[num]),
            input_binding.dtype,
        )
        heights = rule.ops.cast(
            rule.ops.randint(1, 1024, shape=[num]),
            input_binding.dtype,
        )
        rois[:, 2] = rois[:, 0] + widths + rule.ops.random([num], dtype=input_binding.dtype)
        rois[:, 3] = rois[:, 1] + heights + rule.ops.random([num], dtype=input_binding.dtype)
        return rois

    def generate_rois_num_input_value(input_binding):
        if state["num"] is None:
            fpn_rois = rule.arg("fpn_rois")
            state["num"] = fpn_rois.shape[0]
        num = state["num"]
        remaining = input_binding.shape[0]
        # rois_num 的整型协议必须沿用配置 dtype，避免默认 float32 触发算子类型提升。
        result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
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
            for index in range(input_binding.shape[0] - 1):
                result[index] = rule.ops.randint(1, num - remaining + 2)
                num -= result[index]
                remaining -= 1
            result[input_binding.shape[0] - 1] = num
        return result

    rule.generate(
        (
            ("fpn_rois", generate_fpn_rois_input_value),
            ("rois_num", generate_rois_num_input_value),
        ),
    )


@input_rules.register("paddle.vision.ops.generate_proposals")
def generate_proposals_inputs(rule: InputRuleContext):
    """输入规则：生成合法图像尺寸、anchor 和 proposal 分数输入。"""

    def generate_img_size_input_value(input_binding):
        return rule.ops.cast(
            rule.ops.randint(0, 1024, shape=input_binding.shape),
            input_binding.dtype,
        )

    def generate_anchors_input_value(input_binding):
        anchors = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        width = input_binding.shape[0]
        height = input_binding.shape[1]
        for index in range(input_binding.shape[0]):
            anchors[index][0] = rule.ops.random() * width
            anchors[index][1] = rule.ops.random() * height
            anchors[index][2] = (
                rule.ops.random() * (width - anchors[index][0] + 1) + anchors[index][0] + 1
            )
            anchors[index][3] = (
                rule.ops.random() * (height - anchors[index][1] + 1) + anchors[index][1] + 1
            )
        return anchors

    for input_binding in rule.all_tensors:
        if input_binding.parameter_name == "scores":
            # score 是概率，不能与无界的 bbox delta 共用值域。
            rule.set(input_binding, rule.uniform(input_binding, 0, 1))
        elif input_binding.parameter_name == "bbox_deltas":
            rule.set(input_binding, rule.default(input_binding))
        elif input_binding.parameter_name == "img_size":
            rule.set(input_binding, generate_img_size_input_value(input_binding))
        elif input_binding.parameter_name == "anchors":
            rule.set(input_binding, generate_anchors_input_value(input_binding))
        else:
            rule.set(input_binding, rule.default(input_binding))


@input_rules.register("paddle.vision.ops.nms")
def generate_nms_inputs(rule: InputRuleContext):
    """输入规则：生成坐标有序的候选框和对应分数。"""

    def generate_boxes_input_value(input_binding):
        boxes = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        for index in range(input_binding.shape[0]):
            boxes[index][0] = rule.ops.random() * 1023
            boxes[index][1] = rule.ops.random() * 1023
            boxes[index][2] = rule.ops.random() * (1024 - boxes[index][0] + 1) + boxes[index][0] + 1
            boxes[index][3] = rule.ops.random() * (1024 - boxes[index][1] + 1) + boxes[index][1] + 1
        return boxes

    def generate_scores_input_value(input_binding):
        return rule.ops.random(input_binding.shape, dtype=input_binding.dtype)

    def generate_default_vision_input_value(input_binding):
        return rule.ops.cast(
            rule.ops.randint(0, 1024, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("boxes", generate_boxes_input_value),
            ("scores", generate_scores_input_value),
        ),
        default=generate_default_vision_input_value,
    )


@input_rules.register(
    "paddle.vision.ops.roi_align",
    "paddle.vision.ops.roi_pool",
    "paddle.vision.ops.psroi_pool",
)
def generate_roi_pool_inputs(rule: InputRuleContext):
    """输入规则：按特征图和 ROI 数量约束池化坐标及批次索引。"""
    state = {"x_shape": None, "boxes_shape": None}

    def generate_x_input_value(input_binding):
        state["x_shape"] = input_binding.shape
        return rule.ops.cast(
            rule.ops.random(input_binding.shape) * 255,
            input_binding.dtype,
        )

    def generate_boxes_input_value(input_binding):
        if state["x_shape"] is None:
            x = rule.arg("x")
            state["x_shape"] = x.shape
        state["boxes_shape"] = input_binding.shape
        boxes = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        for index in range(input_binding.shape[0]):
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

    def generate_boxes_num_input_value(input_binding):
        if state["boxes_shape"] is None:
            boxes = rule.arg("boxes")
            state["boxes_shape"] = boxes.shape
        boxes_remaining = state["boxes_shape"][0]
        result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        numel = shape_numel(input_binding.shape)
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
            ("x", generate_x_input_value),
            ("boxes", generate_boxes_input_value),
            ("boxes_num", generate_boxes_num_input_value),
        ),
    )


@input_rules.register(
    "paddle.gammainc",
    "paddle.gammaincc",
    "paddle.linspace",
)
def generate_zero_65535_or_unit_inputs(rule: InputRuleContext):
    """输入规则：按整数或浮点 dtype 选择计数域与单位区间。"""
    rule.generate_all("int_zero_65535_else_unit")


@input_rules.register("paddle.dot")
def generate_dot_inputs(rule: InputRuleContext):
    """输入规则：整数保守取值，浮点值复用可配置 default。"""

    def generate_dot_input_value(input_binding):
        if "int" in input_binding.dtype:
            # 整数点积限制累加幅度，浮点和复数才跟随全局范围。
            return rule.domain("uniform", input_binding, low=-127, high=127)
        return rule.default(input_binding)

    rule.generate_all(generate_dot_input_value)


@input_rules.register("paddle.normal")
def generate_normal_inputs(rule: InputRuleContext):
    """输入规则：分别约束正态分布的 mean、std 和 shape 输入。"""
    rule.generate(
        (
            ("mean", "default"),
            ("std", "normal_std"),
        ),
        default="int_zero_1024",
    )


# Tensor 创建规则生成 shape、fill value 和采样分布等构造参数。
@input_rules.register("paddle.ones")
def generate_ones_inputs(rule: InputRuleContext):
    """输入规则：为 ones 的 Tensor shape 参数生成正整数。"""
    rule.generate_all("ones_shape")


@input_rules.register("paddle.zeros")
def generate_zeros_inputs(rule: InputRuleContext):
    """输入规则：为 zeros 的 Tensor shape 参数生成非负整数。"""
    rule.generate_all("int_zero_2048_no_cast")


@input_rules.register("paddle.eye")
def generate_eye_inputs(rule: InputRuleContext):
    """输入规则：为 eye 的行列 Tensor 参数生成非负整数。"""
    rule.generate_all("int_zero_2048_no_cast")


@input_rules.register(
    "paddle.nn.functional.interpolate",
    "paddle.Tensor.tile",
    "paddle.tile",
)
def generate_shape_parameter_inputs(rule: InputRuleContext):
    """输入规则：为插值和 tile 的动态尺寸参数生成正整数。"""
    rule.generate(
        ((("size", "scale_factor", "repeat_times"), "int_one_128"),),
    )


# shape 与缩放规则根据输入 rank 生成合法的尺寸、轴和缩放倍数。
@input_rules.register("paddle.nn.functional.upsample")
def generate_upsample_inputs(rule: InputRuleContext):
    """输入规则：分别约束上采样 size 与 scale_factor。"""
    rule.generate(
        (
            ("size", "int_one_128"),
            ("scale_factor", "abs_unit_plus_one"),
        ),
    )


@input_rules.register(
    "paddle.nn.functional.gaussian_nll_loss",
)
def generate_gaussian_nll_loss_inputs(rule: InputRuleContext):
    """输入规则：保证高斯 NLL 的方差输入严格为正。"""
    rule.generate(
        ((("var", "variance"), "unit_interval_plus_one"),),
    )


@input_rules.register(
    "paddle.nn.functional.hinge_embedding_loss",
)
def generate_hinge_embedding_loss_inputs(rule: InputRuleContext):
    """输入规则：为 hinge embedding loss 生成合法标签。"""
    rule.generate((("label", "hinge_labels"),))


@input_rules.register(
    "paddle.nn.functional.sigmoid_focal_loss",
)
def generate_sigmoid_focal_loss_inputs(rule: InputRuleContext):
    """输入规则：为 sigmoid focal loss 生成二值标签。"""
    rule.generate((("label", "binary_0_1"),))


@input_rules.register("paddle.full")
def generate_full_inputs(rule: InputRuleContext):
    """输入规则：仅约束 shape，fill_value 使用普通输入策略。"""
    rule.generate(
        (("shape", "int_zero_64"),),
    )


@input_rules.register("paddle.standard_normal")
def generate_standard_normal_inputs(rule: InputRuleContext):
    """输入规则：为标准正态分布的动态 shape 生成正整数。"""
    rule.generate((("shape", "int_one_128"),))


@input_rules.register("paddle.logspace")
def generate_logspace_inputs(rule: InputRuleContext):
    """输入规则：限制 logspace 的采样数量为正整数。"""
    rule.generate((("num", "int_one_65535_no_cast"),))


@input_rules.register("paddle.quantile")
def generate_quantile_inputs(rule: InputRuleContext):
    """输入规则：将分位点 q 限制在合法区间。"""
    rule.generate((("q", "quantile_q"),))


@input_rules.register(
    "paddle.remainder",
    aliases=("paddle.Tensor.remainder",),
)
def generate_remainder_inputs(rule: InputRuleContext):
    """输入规则：为余数运算生成非零右操作数。"""

    def generate_remainder_rhs(input_binding):
        if "int" in input_binding.dtype:
            # 整数直接避开零，浮点还需处理 cast 后的量化零。
            return rule.domain("uniform", input_binding, low=1, high=65535)
        return rule.default_nonzero(input_binding)

    rule.generate((("y", generate_remainder_rhs),))


@input_rules.register(
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
)
def generate_dropout_inputs(rule: InputRuleContext):
    """输入规则：将 dropout 概率限制在合法区间。"""
    rule.generate((("p", "dropout_probability"),))


@input_rules.register("paddle.atan2")
def generate_atan2_inputs(rule: InputRuleContext):
    """输入规则：避开 atan2 在原点处的不确定输入。"""
    rule.generate_all("unit_interval_plus_one")


@input_rules.register("paddle.ldexp")
def generate_ldexp_inputs(rule: InputRuleContext):
    """输入规则：联动底数精度限制二进制指数范围。"""
    x_binding = rule.tensor("x")
    # 这里保留逻辑 dtype；backend 的 bfloat16 中间表示不能改变溢出分档。
    x_dtype = str(x_binding.dtype).replace("paddle.", "")
    exponent_limit = {
        "float16": 8,
        "bfloat16": 16,
        "float32": 32,
        "float64": 256,
        "complex64": 32,
        "complex128": 256,
    }.get(x_dtype, 8)

    def generate_x_input_value(input_binding):
        dtype = rule._numeric_dtype(input_binding.dtype)
        if "int" in dtype or dtype == "bool":
            return rule.domain("random_range", input_binding, low=1, high=2)
        return rule.domain("uniform", input_binding, low=0.5, high=1.0)

    def generate_exponent_input_value(input_binding):
        # 指数必须保持整数语义；按 x 精度留出充足溢出余量，也避免负端直接下溢。
        value = rule.ops.randint(
            -exponent_limit,
            exponent_limit + 1,
            shape=input_binding.shape,
        )
        return rule.ops.cast(value, input_binding.dtype)

    rule.generate(
        (
            ("x", generate_x_input_value),
            (("y", "exponent"), generate_exponent_input_value),
        )
    )


@input_rules.register(
    "paddle.log",
    "paddle.log10",
    "paddle.log2",
    "paddle.log1p",
    "paddle.logit",
    aliases=(
        "paddle.Tensor.log",
        "paddle.Tensor.log10",
        "paddle.Tensor.log2",
        "paddle.Tensor.log1p",
        "paddle.Tensor.logit",
    ),
)
def generate_log_domain_inputs(rule: InputRuleContext):
    """输入规则：限制 log 类 API 的数学定义域，避免输入诱发 NaN/INF。"""

    def generate_log_input_value(input_binding):
        dtype = rule._numeric_dtype(input_binding.dtype)
        # 整数 Tensor 不能用 uniform 后再 cast，否则小数会量化为零。
        if "int" in dtype or dtype == "bool":
            low, high = (0, 10) if rule.api_name.endswith("log1p") else (1, 10)
            return rule.domain("random_range", input_binding, low=low, high=high)
        if rule.api_name.endswith("logit"):
            # 留出边界余量，保证 logit 的输入严格位于 (0, 1)。
            return rule.domain("uniform", input_binding, low=0.01, high=0.99)
        if rule.api_name.endswith("log1p"):
            return rule.domain("uniform", input_binding, low=-0.9, high=10.0)
        return rule.domain("uniform", input_binding, low=0.01, high=10.0)

    # Tensor 方法的 receiver 名称为 self，函数式 API 通常使用 x/input。
    rule.generate(((("x", "input", "self"), generate_log_input_value),))


@input_rules.register(
    "paddle.logsumexp",
    aliases=("paddle.Tensor.logsumexp",),
)
def generate_logsumexp_inputs(rule: InputRuleContext):
    """输入规则：限制 logsumexp 数据输入，保留 axis 等控制参数的默认规则。"""

    def generate_logsumexp_input_value(input_binding):
        dtype = rule._numeric_dtype(input_binding.dtype)
        # 有限且较小的输入可避免指数溢出；超大 0size 配置仍由上层分类处理。
        if "int" in dtype or dtype == "bool":
            return rule.domain("random_range", input_binding, low=-10, high=10)
        return rule.domain("uniform", input_binding, low=-10.0, high=10.0)

    rule.generate(((("x", "input", "self"), generate_logsumexp_input_value),))


@input_rules.register("paddle.bincount")
def generate_bincount_inputs(rule: InputRuleContext):
    """输入规则：生成非负计数索引和相容权重。"""

    def generate_integer_input_value(input_binding):
        return rule.ops.cast(
            rule.ops.randint(0, 65535, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("x", generate_integer_input_value),
            ("minlength", generate_integer_input_value),
        ),
    )


@input_rules.register(
    "paddle.nn.functional.adaptive_avg_pool2d", "paddle.nn.functional.adaptive_avg_pool3d"
)
def generate_adaptive_avg_pool_inputs(rule: InputRuleContext):
    """输入规则：根据输入 rank 生成合法自适应池化输出尺寸。"""

    def generate_output_size_input_value(input_binding):
        x_shape = rule.arg("x").shape
        return rule.ops.cast(
            rule.ops.randint(1, 2 * max(x_shape), shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (("output_size", generate_output_size_input_value),),
    )


@input_rules.register("paddle.empty")
def generate_empty_inputs(rule: InputRuleContext):
    """输入规则：为 empty 的 Tensor shape 参数生成空尺寸边界值。"""
    rule.generate((("shape", "empty_shape"),))


@input_rules.register(
    "paddle.repeat_interleave",
    aliases=("paddle.Tensor.repeat_interleave",),
)
def generate_repeat_interleave_inputs(rule: InputRuleContext):
    """输入规则：约束 repeats 为正整数并生成有效 axis。"""

    def generate_axis_input_value(input_binding):
        x = rule.arg("x")
        input_dims = len(x.shape)
        if len(input_binding.shape) == 0:
            return rule.ops.asarray(
                rule.ops.randint(-input_dims, input_dims), dtype=input_binding.dtype
            )
        return rule.ops.cast(
            rule.ops.randint(-input_dims, input_dims, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("repeats", "int_one_2048"),
            ("axis", generate_axis_input_value),
        ),
    )


@input_rules.register(
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
def generate_put_along_axis_inputs(rule: InputRuleContext):
    """输入规则：按输入 shape 与 axis 生成合法 indices 和相容 values。"""

    def generate_unique_indices(dim_size, width, prefix_shape=()):
        """用随机循环偏移批量生成每个切片内无重复的 index。"""
        if width == 0:
            return rule.ops.zeros(prefix_shape + (0,), dtype="int64")
        offsets = rule.ops.randint(0, dim_size, shape=prefix_shape + (1,))
        base = rule.ops.arange(width, dtype="int64")
        return rule.ops.remainder(offsets + base, dim_size)

    def generate_random_tensor_input_value(input_binding, shape):
        scalar_spec = InputTensorSpec(
            shape=tuple(shape),
            dtype=input_binding.dtype,
            place=input_binding.place,
            is_contiguous=input_binding.is_contiguous,
            strides=input_binding.strides,
        )
        return generate_random_range_input_value(scalar_spec, rng=rule.ops)

    def generate_indices_input_value(input_binding):
        x_tensor = rule.arg("arr", rule.arg("x"))
        x_shape = tuple(x_tensor.shape) if x_tensor is not None else ()
        x_dims = len(x_shape)
        current_shape = tuple(input_binding.shape)
        if len(current_shape) != x_dims:
            new_shape = [current_shape[i] if i < len(current_shape) else 1 for i in range(x_dims)]
            indices = rule.ops.zeros(new_shape, dtype="int64")
            for axis in range(x_dims):
                if axis < len(current_shape):
                    dim_size = x_shape[axis]
                    if dim_size > 0:
                        width = new_shape[axis]
                        if width <= dim_size:
                            axis_indices = generate_unique_indices(dim_size, width)
                        else:
                            axis_indices = rule.ops.choice(dim_size, shape=width, replace=False)
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
        # 最后一维为空时每个切片都无索引，直接返回可避免遍历巨大的前缀维度。
        if current_shape and current_shape[-1] == 0:
            return indices
        if 0 <= axis < x_dims:
            dim_size = x_shape[axis]
            width = current_shape[-1]
            if width <= dim_size:
                return generate_unique_indices(dim_size, width, current_shape[:-1])
            for idx in rule.ops.ndindex(tuple(current_shape[:-1])):
                indices[idx] = rule.ops.choice(dim_size, shape=width, replace=False)
        return indices

    def write_related_input_values(input_binding):
        indices_binding = rule.tensor("indices")
        if indices_binding is not None:
            indices = rule.value(indices_binding)
            if tuple(indices.shape) != tuple(input_binding.shape):
                if rule.ops.prod(input_binding.shape) == 1:
                    rule.set_preserving_spec(
                        input_binding,
                        rule.ops.full(
                            indices.shape,
                            generate_random_tensor_input_value(input_binding, ())[()],
                            dtype=input_binding.dtype,
                        ),
                    )
                else:
                    rule.set_preserving_spec(
                        input_binding,
                        generate_random_tensor_input_value(input_binding, indices.shape),
                    )
                return
            rule.set(
                input_binding,
                generate_random_tensor_input_value(input_binding, input_binding.shape),
            )
            return
        rule.set(input_binding, rule.default(input_binding))

    for input_binding in rule.all_tensors:
        if input_binding.parameter_name == "indices":
            rule.set(input_binding, generate_indices_input_value(input_binding))
        elif input_binding.parameter_name == "values":
            write_related_input_values(input_binding)
        else:
            rule.set(input_binding, rule.default(input_binding))


# 转置、softmax 与 padding 规则维护轴和边界列表的合法取值。
@input_rules.register("paddle.matrix_transpose")
def generate_matrix_transpose_inputs(rule: InputRuleContext):
    """输入规则：保证矩阵转置输入至少具有二维结构。"""

    def generate_x_input_value(input_binding):
        # 该规则只修复 rank，不应另行拥有随机数值分布。
        shape = input_binding.shape if len(input_binding.shape) >= 2 else (2, 2)
        return rule.default(input_binding, shape=shape)

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.nn.functional.softmax")
def generate_softmax_inputs(rule: InputRuleContext):
    """输入规则：根据输入 rank 生成有效 softmax axis。"""

    def generate_axis_input_value(input_binding):
        x_shape = rule.arg("x").shape
        return rule.domain("uniform", input_binding, low=-len(x_shape), high=len(x_shape))

    rule.generate(
        (
            ("x", "random_range"),
            ("axis", generate_axis_input_value),
        ),
    )


@input_rules.register("paddle.nn.functional.zeropad2d")
def generate_zeropad2d_inputs(rule: InputRuleContext):
    """输入规则：按输入空间维度约束二维 padding。"""

    def generate_padding_input_value(input_binding):
        return rule.domain("uniform", input_binding, low=0, high=10)

    rule.generate(
        (
            ("x", "random_range"),
            ("padding", generate_padding_input_value),
        ),
    )


@input_rules.register("paddle.nn.functional.pad")
def generate_pad_inputs(rule: InputRuleContext):
    """输入规则：按输入 rank 生成不会越界的 padding。"""

    def generate_pad_input_value(input_binding):
        x_shape = rule.arg("x").shape
        return rule.domain("uniform", input_binding, low=0, high=min(x_shape))

    rule.generate((("pad", generate_pad_input_value),))


@input_rules.register("paddle.nn.functional.class_center_sample")
def generate_class_center_sample_inputs(rule: InputRuleContext):
    """输入规则：根据类别数生成合法标签索引。"""

    def generate_label_input_value(input_binding):
        num_classes = rule.arg("num_classes")
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("label", generate_label_input_value),))


@input_rules.register("paddle.shard_index")
def generate_shard_index_inputs(rule: InputRuleContext):
    """输入规则：根据 index_num 限制分片输入索引范围。"""

    def generate_shard_index_input_value(input_binding):
        index_num = rule.arg("index_num")
        if index_num is None:
            index_num = rule.ops.randint(1, 1000)
        return rule.ops.cast(
            rule.ops.randint(0, index_num, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("input", generate_shard_index_input_value),))


# mask 注意力规则需要同步生成序列长度和旋转位置编码 Tensor。
@input_rules.register("paddle.incubate.nn.functional.masked_multihead_attention")
def generate_masked_multihead_attention_inputs(rule: InputRuleContext):
    """输入规则：联动序列长度和 mask 值域生成注意力输入。"""

    def generate_sequence_lengths_input_value(input_binding):
        return rule.domain("random_range", input_binding, low=1)

    def generate_rotary_tensor_input_value(input_binding):
        return rule.domain("uniform", input_binding, low=0, high=1000)

    rule.generate(
        (
            ("sequence_lengths", generate_sequence_lengths_input_value),
            ("rotary_tensor", generate_rotary_tensor_input_value),
        ),
    )


@input_rules.register(
    "paddle.argmax",
    "paddle.argmin",
    aliases=("paddle.Tensor.argmax", "paddle.Tensor.argmin"),
)
def generate_argminmax_inputs(rule: InputRuleContext):
    """输入规则：根据输入 rank 生成 argmin/argmax 的合法 axis。"""

    def generate_axis_input_value(input_binding):
        x_shape = rule.arg("x").shape
        min_dim = len(x_shape)
        return rule.ops.cast(
            rule.ops.randint(-min_dim, min_dim - 1, shape=input_binding.shape),
            "int64",
        )

    rule.generate((("axis", generate_axis_input_value),))


# reduction 规则集中处理负轴、轴列表、重复轴和空轴语义。
@input_rules.register("paddle.cumsum", aliases=("paddle.Tensor.cumsum",))
def generate_cumsum_inputs(rule: InputRuleContext):
    """输入规则：根据输入 rank 生成 cumsum 的合法 axis。"""

    def generate_axis_input_value(input_binding):
        x_shape = rule.arg("x").shape
        return rule.ops.randint(-len(x_shape), len(x_shape), shape=input_binding.shape)

    rule.generate((("axis", generate_axis_input_value),))


@input_rules.register(
    "paddle.mean", "paddle.max", "paddle.min", "paddle.prod", "paddle.sum", "paddle.squeeze"
)
def generate_reduction_axis_inputs(rule: InputRuleContext):
    """输入规则：为归约 API 生成去重且不越界的 axis。"""
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

    def generate_axis_input_value(input_binding):
        nonlocal used_list_axes
        x_shape = rule.arg("x").shape
        max_dim = max(len(x_shape), 1)
        axis_arg = rule.arg("axis", None)
        if isinstance(axis_arg, (list, tuple)) and input_binding.path.item_indices:
            if input_binding.shape not in [(), (1,)]:
                raise ValueError(
                    f"Invalid TensorConfig for axis: shape {input_binding.shape} or dtype {input_binding.dtype}"
                )
            if input_binding.dtype not in {"int32", "int64"}:
                raise ValueError(
                    f"Invalid TensorConfig for axis: shape {input_binding.shape} or dtype {input_binding.dtype}"
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
            return rule.ops.asarray(dim, dtype=input_binding.dtype)
        if len(input_binding.shape) == 0:
            dim = rule.ops.randint(0, max_dim)
            if rule.ops.random() > 0.5:
                dim -= max_dim
            return rule.ops.asarray(dim, dtype=input_binding.dtype)
        if len(input_binding.shape) == 1:
            dims = rule.ops.choice(max_dim, shape=input_binding.shape[0], replace=False)
            mask = rule.ops.random(input_binding.shape[0]) > 0.5
            dims = rule.ops.where(mask, dims - max_dim, dims)
            return rule.ops.asarray(dims, dtype=input_binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {rule.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {input_binding.shape}."
        )

    rule.generate((("axis", generate_axis_input_value),))


# 维度变换规则确保 axis 与输入 rank 及目标 shape 保持一致。
@input_rules.register("paddle.unsqueeze")
def generate_unsqueeze_inputs(rule: InputRuleContext):
    """输入规则：按扩维后的 rank 生成合法 unsqueeze axis。"""

    def generate_axis_input_value(input_binding):
        x_shape = rule.arg("x").shape
        max_dim = len(x_shape) + 1
        if len(input_binding.shape) == 0:
            dim = rule.ops.randint(0, max_dim)
            if rule.ops.random() > 0.5:
                dim -= max_dim
            return rule.ops.asarray(dim, dtype=input_binding.dtype)
        if len(input_binding.shape) == 1:
            dims = rule.ops.choice(max_dim, shape=input_binding.shape[0], replace=False)
            mask = rule.ops.random(input_binding.shape[0]) > 0.5
            dims = rule.ops.where(mask, dims - max_dim, dims)
            return rule.ops.asarray(dims, dtype=input_binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
            f"Expected a 0-D or 1-D Tensor, but got shape {input_binding.shape}."
        )

    rule.generate((("axis", generate_axis_input_value),))


@input_rules.register("paddle.unflatten", aliases=("paddle.Tensor.unflatten",))
def generate_unflatten_inputs(rule: InputRuleContext):
    """输入规则：根据源维度生成乘积匹配的 unflatten shape。"""

    def generate_axis_input_value(input_binding):
        x_shape = rule.arg("x").shape
        return rule.ops.cast(
            rule.ops.randint(0, len(x_shape), shape=input_binding.shape),
            input_binding.dtype,
        )

    # shape 中的 0-D Tensor 是动态维度，必须和被展开维度联动，不能走通用随机值域。
    def generate_shape_tensor_values():
        x_config = rule.arg("x")
        shape_config = rule.arg("shape")
        if x_config is None or shape_config is None:
            return
        x_shape = tuple(int(dim) for dim in x_config.shape)
        axis = rule.arg("axis", 0)
        if rule.is_tensor_config(axis):
            axis_binding = rule.binding_for_value(axis)
            axis_value = rule.value(axis_binding) if axis_binding is not None else None
            axis = int(axis_value.item()) if axis_value is not None else 0
        axis = int(axis)
        # 将负轴转换为规范下标，保证乘积校验访问同一目标维度。
        if axis < 0:
            axis += len(x_shape)
        if axis < 0 or axis >= len(x_shape):
            return
        target = x_shape[axis]

        dynamic = []
        fixed_product = 1
        inferred_dims = 0

        # 递归拆解 shape 容器，分别记录静态维度、推导维度和动态 Tensor。
        def collect(value):
            nonlocal fixed_product, inferred_dims
            if rule.is_tensor_config(value):
                binding = rule.binding_for_value(value)
                if binding is None:
                    return
                if len(binding.shape) == 0:
                    dynamic.append((binding, 1))
                    return
                if len(binding.shape) != 1:
                    raise ValueError(f"Invalid TensorConfig for unflatten shape: {binding.shape!r}")
                count = int(binding.shape[0])
                if count <= 0:
                    raise ValueError(
                        f"unflatten shape TensorConfig must contain at least one value: {binding.shape!r}"
                    )
                dynamic.append((binding, count))
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
                return
            value = int(value)
            if value == -1:
                inferred_dims += 1
                return
            if value < 0:
                raise ValueError(f"unflatten shape contains unsupported dimension {value}")
            fixed_product *= value

        collect(shape_config)
        # 没有动态 Tensor 时保留静态配置原样，让 Paddle 决定其错误语义。
        if not dynamic:
            return
        # 多个 -1 无法确定唯一结果，必须在输入生成阶段拒绝。
        if inferred_dims > 1:
            raise ValueError("unflatten shape can contain at most one inferred dimension")
        if fixed_product == 0:
            if target != 0:
                raise ValueError(
                    f"unflatten shape fixed dimensions have product 0 but target is {target}"
                )
            quotient = 1
        elif target % fixed_product:
            raise ValueError(
                f"unflatten shape dimensions must divide target {target}: fixed product {fixed_product}"
            )
        else:
            quotient = target // fixed_product
        if dynamic and not inferred_dims and quotient <= 0:
            raise ValueError(
                f"unflatten dynamic shape dimensions must be positive, got target {target}"
            )
        remaining = quotient if dynamic and not inferred_dims else 1
        # 剩余乘积写入第一个动态维度，其余动态槽位用 1 保持乘积稳定。
        for binding, count in dynamic:
            values = [remaining] + [1] * (count - 1)
            if len(binding.shape) == 0:
                rule.set(binding, rule.ops.asarray(values[0], dtype=binding.dtype))
            else:
                value = rule.ops.asarray(values, dtype=binding.dtype)
                if tuple(value.shape) != tuple(binding.shape):
                    value = rule.ops.reshape(value, binding.shape)
                rule.set(binding, value)

    # 先生成 axis，再生成 shape 动态值；其余普通 Tensor 继续使用默认值域。
    rule.generate((("axis", generate_axis_input_value),), default=None)
    generate_shape_tensor_values()
    rule.generate_remaining()


@input_rules.register("paddle.topk", aliases=("paddle.Tensor.topk",))
def generate_topk_inputs(rule: InputRuleContext):
    """输入规则：联动 axis 维度限制 topk 的 k 值。"""

    def generate_x_input_value(input_binding):
        dtype = input_binding.dtype
        if dtype in {"float32", "float64", "bfloat16"}:
            # 普通浮点 topk 没有特殊定义域，继承 configurable default。
            return rule.default(input_binding)
        if dtype in {"float16", "float8_e4m3fn", "float8_e5m2"}:
            # 低精度保留小尺度正态分布，控制排序比较的舍入误差。
            return rule.normal(input_binding, scale=1e-3)
        if dtype in {"int32", "int64"}:
            return rule.ops.cast(
                rule.ops.randint(-10, 10, shape=input_binding.shape),
                dtype,
            )
        raise ValueError(
            f"Unsupported dtype {input_binding.dtype} for paddle.topk / paddle.Tensor.topk"
        )

    def generate_k_input_value(input_binding):
        x_config = rule.arg("x")
        axis = rule.arg("axis", -1)
        max_k_value = 1
        if x_config is not None and x_config.shape:
            max_k_value = x_config.shape[axis] if len(x_config.shape) > 0 else 1
        if not input_binding.shape:
            return rule.ops.asarray(rule.ops.randint(1, max_k_value + 1), dtype=input_binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(1, max_k_value + 1, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("x", generate_x_input_value),
            ("k", generate_k_input_value),
        ),
    )


# 采样与 gather 规则根据源 Tensor 尺寸限制索引的有效范围。
@input_rules.register("paddle.index_sample")
def generate_index_sample_inputs(rule: InputRuleContext):
    """输入规则：按输入第二维限制 index_sample 索引。"""

    def generate_index_input_value(input_binding):
        x_dim = rule.arg("x").shape[1]
        # 空的采样维度没有合法随机上界，只能生成空语义的零索引。
        if x_dim == 0:
            return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        return rule.ops.randint(0, x_dim, shape=input_binding.shape)

    rule.generate((("index", generate_index_input_value),))


# Tensor 下标读写规则同时约束切片对象、布尔 mask 和写入值形状。
@input_rules.register("paddle.Tensor.__getitem__")
def generate_tensor_getitem_inputs(rule: InputRuleContext):
    """输入规则：根据源 Tensor 最小维度生成合法 getitem 索引。"""

    def source_binding():
        input_binding = rule.tensor("arr") or rule.tensor("x") or rule.tensor("self")
        if input_binding is None:
            raise ValueError("Tensor.__getitem__ rule could not find source tensor")
        return input_binding

    def generate_item_input_value(input_binding):
        min_dim = min(source_binding().shape)
        numel = shape_numel(input_binding.shape)
        if input_binding.dtype == "bool":
            indices = rule.ops.choice([0, 1], shape=numel)
        else:
            indices = rule.ops.randint(0, min_dim, shape=numel)
        return rule.ops.cast(
            rule.ops.reshape(indices, input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("item", generate_item_input_value),))


@input_rules.register("paddle.Tensor.__setitem__")
def generate_tensor_setitem_inputs(rule: InputRuleContext):
    """输入规则：根据源和值的 shape 生成合法 setitem 索引。"""

    def source_binding():
        input_binding = rule.tensor("arr") or rule.tensor("x") or rule.tensor("self")
        if input_binding is None:
            raise ValueError("Tensor.__setitem__ rule could not find source tensor")
        return input_binding

    def generate_item_input_value(input_binding):
        min_dim = min(source_binding().shape)
        numel = shape_numel(input_binding.shape)
        if input_binding.dtype == "bool":
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
            rule.ops.reshape(indices, input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("item", generate_item_input_value),))


@input_rules.register("paddle.index_add", "paddle.index_fill")
def generate_index_update_inputs(rule: InputRuleContext):
    """输入规则：按目标 axis 尺寸生成 index_add/index_fill 索引。"""

    def generate_index_input_value(input_binding):
        axis = rule.arg("axis")
        if axis is None:
            raise ValueError("Axis is None")
        x_shape = rule.arg("x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
        if len(input_binding.shape) >= 1:
            # 目标轴为空时 index 必须全为零，避免 randint(0, 0) 抛错。
            if x_shape[axis] == 0:
                return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
            return rule.ops.cast(
                rule.ops.randint(0, x_shape[axis], shape=input_binding.shape),
                input_binding.dtype,
            )
        raise ValueError(
            f"Invalid shape for 'index' Tensor in {rule.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {input_binding.shape}."
        )

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.take")
def generate_take_inputs(rule: InputRuleContext):
    """输入规则：按源 Tensor 元素数限制 take 索引。"""

    def generate_index_input_value(input_binding):
        x = rule.arg("x")
        dim_size = shape_numel(x.shape)
        return rule.ops.cast(
            rule.ops.randint(0, dim_size, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.gather", aliases=("paddle.Tensor.gather",))
def generate_gather_inputs(rule: InputRuleContext):
    """输入规则：按 gather axis 对各维索引进行边界约束。"""

    def generate_index_input_value(input_binding):
        x = rule.arg("x")
        if rule.has_kwarg("axis"):
            axis = rule.arg("axis")
            if hasattr(axis, "shape"):
                axis = axis.shape[0]
        else:
            axis = 0
        return rule.ops.cast(
            rule.ops.randint(0, x.shape[axis], shape=input_binding.shape),
            input_binding.dtype,
        )

    def generate_axis_input_value(input_binding):
        return rule.ops.cast(
            rule.ops.randint(0, 2, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("index", generate_index_input_value),
            ("axis", generate_axis_input_value),
        ),
    )


@input_rules.register("paddle.gather_nd", aliases=("paddle.Tensor.gather_nd",))
def generate_gather_nd_inputs(rule: InputRuleContext):
    """输入规则：根据源和索引 rank 生成 gather_nd 索引。"""

    def generate_index_input_value(input_binding):
        x_shape = rule.arg("x").shape
        index_shape = rule.arg("index").shape
        result = rule.ops.zeros(index_shape, dtype=input_binding.dtype)
        for index in range(index_shape[-1]):
            result[..., index] = rule.ops.randint(0, x_shape[index], shape=result[..., index].shape)
        return result

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.index_select", aliases=("paddle.Tensor.index_select",))
def generate_index_select_inputs(rule: InputRuleContext):
    """输入规则：按选取 axis 的尺寸限制 index_select 索引。"""

    def generate_index_input_value(input_binding):
        axis = rule.arg("axis")
        if axis is None:
            axis = 0
        x = rule.arg("x")
        if x.shape[axis] == 0:
            return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(0, x.shape[axis], shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.take_along_axis", aliases=("paddle.Tensor.take_along_axis",))
def generate_take_along_axis_inputs(rule: InputRuleContext):
    """输入规则：按目标 axis 尺寸限制 take_along_axis 索引。"""

    def generate_indices_input_value(input_binding):
        # 函数参数名为 arr，Tensor method 的接收者绑定后统一为 x。
        arr_shape = rule.arg("arr", rule.arg("x")).shape
        axis = rule.arg("axis")
        generate_axis_input_value = axis if axis >= 0 else axis + len(arr_shape)
        dim_size = arr_shape[generate_axis_input_value]
        dtype = input_binding.dtype if input_binding.dtype in {"int32", "int64"} else "int64"
        num_elements = shape_numel(input_binding.shape)
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
        return rule.ops.reshape(indices, input_binding.shape)

    rule.generate((("indices", generate_indices_input_value),))


# 高级索引规则为整型坐标和布尔 mask 维护统一的形状状态。
@input_rules.register("paddle.index_put", aliases=("paddle.Tensor.index_put",))
def generate_index_put_inputs(rule: InputRuleContext):
    """输入规则：联动多个索引 Tensor 的 shape 与各维取值范围。"""
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

    def generate_index_input_value(input_binding):
        if not state:
            prepared = prepare_indices_state()
            if prepared is None:
                return rule.default(input_binding)
            state.update(prepared)

        if input_binding.dtype == "bool":
            if state["num_true_needed"] < 0:
                raise ValueError(
                    "Cannot determine the number of True elements for the boolean mask."
                )
            state["x_dim_cursor"] += len(input_binding.shape)
            return bool_mask(input_binding.shape, state["num_true_needed"])

        x_dim_to_index = state["x_shape"][state["x_dim_cursor"]]
        state["x_dim_cursor"] += 1
        return rule.ops.cast(
            int_indices(input_binding.shape, x_dim_to_index),
            input_binding.dtype,
        )

    rule.generate((("indices", generate_index_input_value),))


# 分段规则要求索引有序，并与数据批次大小保持一致。
@input_rules.register("paddle.multiplex")
def generate_multiplex_inputs(rule: InputRuleContext):
    """输入规则：按候选输入数量生成 multiplex 选择索引。"""

    def generate_index_input_value(input_binding):
        axis_values = rule.arg("inputs")
        return rule.ops.cast(
            rule.ops.randint(0, len(axis_values), shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("index", generate_index_input_value),))


@input_rules.register(
    "paddle.geometric.segment_sum",
    "paddle.geometric.segment_max",
    "paddle.geometric.segment_mean",
    "paddle.geometric.segment_min",
    "paddle.incubate.segment_sum",
    "paddle.incubate.segment_max",
    "paddle.incubate.segment_mean",
    "paddle.incubate.segment_min",
)
def generate_segment_inputs(rule: InputRuleContext):
    """输入规则：根据 data 批次大小生成有序 segment_ids。"""

    def generate_segment_ids_input_value(input_binding):
        batch_size = rule.arg("data").shape[0]
        max_segments = rule.ops.randint(1, batch_size + 1)
        segment_ids = rule.ops.cast(
            rule.ops.randint(0, max_segments, shape=input_binding.shape),
            input_binding.dtype,
        )
        return rule.ops.sort(segment_ids)

    rule.generate((("segment_ids", generate_segment_ids_input_value),))


@input_rules.register(
    "paddle.geometric.send_u_recv",
    "paddle.geometric.send_uv",
    "paddle.geometric.send_ue_recv",
)
def generate_geometric_send_inputs(rule: InputRuleContext):
    """输入规则：根据节点数生成合法的图消息收发索引。"""

    def generate_index_input_value(input_binding):
        num_nodes = rule.arg("x").shape[0]
        return rule.ops.cast(
            rule.ops.randint(0, num_nodes, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        ((("src_index", "dst_index"), generate_index_input_value),),
    )


# 图采样规则联动节点表、边表、列指针和采样节点范围。
@input_rules.register("paddle.geometric.sample_neighbors")
def generate_sample_neighbors_inputs(rule: InputRuleContext):
    """输入规则：联动 CSR 边界和节点数生成邻居采样输入。"""

    def generate_row_input_value(input_binding):
        colptr_shape = rule.arg("colptr").shape
        num_nodes = colptr_shape[0] - 1
        return rule.ops.randint(0, num_nodes, shape=input_binding.shape, dtype=input_binding.dtype)

    def generate_colptr_input_value(input_binding):
        row = rule.arg("row")
        num_edges = row.shape[0]
        num_nodes = input_binding.shape[0] - 1
        colptr = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
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

    def generate_input_nodes_input_value(input_binding):
        num_nodes = input_binding.shape[0] - 1
        return rule.ops.randint(0, num_nodes, shape=input_binding.shape, dtype=input_binding.dtype)

    def generate_edge_order_input_value(input_binding):
        num_edges = rule.arg("row").shape[0]
        return rule.ops.reshape(
            rule.ops.arange(num_edges, dtype=input_binding.dtype),
            input_binding.shape,
        )

    rule.generate(
        (
            ("row", generate_row_input_value),
            ("colptr", generate_colptr_input_value),
            ("input_nodes", generate_input_nodes_input_value),
            (("eids", "perm_buffer"), generate_edge_order_input_value),
        ),
    )


# 变形规则通过共享状态联动源 Tensor 与目标 shape。
@input_rules.register(
    "paddle.reshape",
    aliases=(
        "paddle.Tensor.reshape",
        "paddle.reshape_",
        "paddle.Tensor.reshape_",
        "paddle._C_ops.reshape_",
    ),
)
def generate_reshape_inputs(rule: InputRuleContext):
    """输入规则：生成元素总数与源 Tensor 一致的 reshape shape。"""
    state = {
        "shape": None,
        "maxvalue": None,
        "tensornum": None,
    }

    def initialize_from_x(input_binding):
        shape = input_binding.shape
        if state["shape"] is None:
            state["shape"] = shape
            state["maxvalue"] = shape_numel(shape)
            state["tensornum"] = 0
            for candidate in rule.argument_values():
                if isinstance(candidate, (list, tuple)):
                    for index, item in enumerate(candidate):
                        if isinstance(item, numbers.Integral):
                            if item == 0:
                                # 0 表示抄 x 同位置维度；index 越界属于无效配置，
                                # 这里不能崩，要让 Paddle 去报 InvalidArgument。
                                if index < len(shape) and shape[index] != 0:
                                    state["maxvalue"] //= shape[index]
                            elif item != -1:
                                state["maxvalue"] //= int(item)
                        elif rule.is_tensor_config(item):
                            state["tensornum"] += 1
        return rule.default(input_binding)

    def generate_shape_input_value(input_binding):
        if state["tensornum"] == 0:
            state["tensornum"] = 1
        dtype = "int32"
        shape = input_binding.shape
        maxvalue = state["maxvalue"]
        if maxvalue == 0:
            # zero-size 输入的目标 shape 至少要保留一个零维，避免构造非零容量。
            return rule.ops.zeros(shape, dtype=dtype)
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
            ("shape", generate_shape_input_value),
        ),
    )


# 切片规则共同维护 axes、starts、ends 和 steps 的关系。
@input_rules.register("paddle.slice")
def generate_slice_inputs(rule: InputRuleContext):
    """输入规则：联动 axes、starts、ends 和 steps 生成有效切片。"""
    state = {
        "shape": None,
        "indice": 0,
        "start": [],
        "index": 0,
    }

    def axes():
        return rule.arg("axes")

    def generate_source_input_value(input_binding):
        if state["shape"] is None:
            state["shape"] = input_binding.shape
        return rule.default(input_binding)

    def generate_starts_input_value(input_binding):
        dim_sizes = [state["shape"][axis] for axis in axes()]
        if input_binding.shape == ():
            coin = rule.ops.randint(0, 2)
            if coin == 0:
                value = rule.ops.randint(0, dim_sizes[state["indice"]] - 1, input_binding.shape)
            else:
                value = rule.ops.randint(-65535, -1, input_binding.shape)
            state["start"].append(value)
            state["indice"] += 1
            return rule.ops.asarray(value, dtype=input_binding.dtype)
        result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        for index in range(shape_numel(input_binding.shape)):
            coin = rule.ops.randint(0, 2)
            if coin == 0:
                result[index] = rule.ops.randint(0, dim_sizes[state["indice"]] - 1)
            else:
                result[index] = rule.ops.randint(-65535, -1)
            state["start"].append(result[index])
            state["indice"] += 1
        return result

    def generate_ends_input_value(input_binding):
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
        if input_binding.shape == ():
            coin = rule.ops.randint(0, 2)
            current = start[state["index"]]
            if coin == 0:
                value = rule.ops.randint(current + 1, 65535, input_binding.shape)
            else:
                if current - dim_sizes[index] == 0:
                    current -= 1
                    start[state["index"]] = current
                value = rule.ops.randint(
                    min(current - dim_sizes[index] + 1, -1), 0, input_binding.shape
                )
            state["index"] += 1
            return rule.ops.asarray(value, dtype=input_binding.dtype)
        result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        for index in range(shape_numel(input_binding.shape)):
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
            ("input", generate_source_input_value),
            ("starts", generate_starts_input_value),
            ("ends", generate_ends_input_value),
        ),
    )


# scatter 规则依据目标维度生成坐标并保持 updates 的广播关系。
@input_rules.register("paddle.scatter")
def generate_scatter_inputs(rule: InputRuleContext):
    """输入规则：按源 Tensor 首维生成 scatter 索引。"""

    def generate_index_input_value(input_binding):
        x = rule.arg("x")
        first_dim = x.shape[0]
        overwrite = rule.arg("overwrite")
        if (overwrite is None or overwrite is True) and (
            input_binding.shape == () or input_binding.shape[0]
        ) <= first_dim:
            return rule.ops.cast(
                rule.ops.choice(first_dim, shape=input_binding.shape, replace=False),
                input_binding.dtype,
            )
        return rule.ops.cast(
            rule.ops.randint(0, first_dim, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.scatter_nd")
def generate_scatter_nd_inputs(rule: InputRuleContext):
    """输入规则：根据输出 shape 生成 scatter_nd 多维索引。"""

    def generate_index_input_value(input_binding):
        output_shape = rule.arg("shape")
        if output_shape and len(output_shape):
            result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
            for axis in range(len(output_shape)):
                if axis >= input_binding.shape[-1]:
                    break
                result[..., axis] = rule.ops.randint(
                    -output_shape[axis],
                    output_shape[axis],
                    shape=result[..., axis].shape,
                )
                result[..., axis] = rule.ops.cast(result[..., axis], input_binding.dtype)
            return result
        return rule.default(input_binding)

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.scatter_nd_add")
def generate_scatter_nd_add_inputs(rule: InputRuleContext):
    """输入规则：根据目标 Tensor shape 生成 scatter_nd_add 索引。"""

    def generate_index_input_value(input_binding):
        x_shape = rule.arg("x").shape
        result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        for axis in range(input_binding.shape[-1]):
            result[..., axis] = rule.ops.randint(
                -x_shape[axis],
                x_shape[axis],
                shape=result[..., axis].shape,
            )
            result[..., axis] = rule.ops.cast(result[..., axis], input_binding.dtype)
        return result

    rule.generate((("index", generate_index_input_value),))


@input_rules.register("paddle.strided_slice")
def generate_strided_slice_inputs(rule: InputRuleContext):
    """输入规则：联动 axes 与步长方向生成有效分片边界。"""

    def generate_axes_input_value(input_binding):
        x = rule.arg("x")
        return rule.ops.cast(
            rule.ops.randint(0, len(x.shape), shape=input_binding.shape),
            input_binding.dtype,
        )

    def generate_list_input_value(input_binding):
        x = rule.arg("x")
        axes_arg = rule.arg("axes")
        axes = axes_arg
        if not isinstance(axes, list):
            axes = rule.value(rule.tensor("axes"))
        if not input_binding.path.item_indices:
            return rule.default(input_binding)
        item_index = input_binding.path.item_indices[0]
        parameter = input_binding.parameter_name
        if parameter == "starts":
            return rule.ops.cast(
                rule.ops.randint(0, x.shape[axes[item_index]] - 1, shape=input_binding.shape),
                input_binding.dtype,
            )
        if parameter == "ends":
            starts_arg = rule.arg("starts")
            starts_value = None
            if isinstance(starts_arg, (list, tuple)):
                starts_config = starts_arg[item_index]
                starts_binding = rule.binding_for_value(starts_config)
                if starts_binding is not None:
                    starts_value = rule.value(starts_binding)
                else:
                    starts_value = starts_config
            else:
                starts_binding = rule.binding_for_value(starts_arg)
                if starts_binding is not None:
                    starts_value = rule.value(starts_binding)
                else:
                    starts_value = starts_arg
            if starts_value is None:
                starts_value = 0
            elif hasattr(starts_value, "item"):
                starts_value = starts_value.item()
            starts_value = int(starts_value)
            return rule.ops.cast(
                rule.ops.randint(
                    starts_value + 1,
                    x.shape[axes[item_index]],
                    shape=input_binding.shape,
                ),
                input_binding.dtype,
            )
        if parameter == "strides":
            return rule.ops.cast(
                rule.ops.randint(1, x.shape[axes[item_index]], shape=input_binding.shape),
                input_binding.dtype,
            )
        return rule.default(input_binding)

    rule.generate(
        (
            ("axes", generate_axes_input_value),
            (("starts", "ends", "strides"), generate_list_input_value),
        ),
    )


# 张量收缩规则联动两侧收缩轴及对应维度。
@input_rules.register("paddle.tensordot")
def generate_tensordot_inputs(rule: InputRuleContext):
    """输入规则：为两个输入生成不重复且维度相容的收缩 axes。"""
    state = {"shape1": None, "shape2": None, "tensor1": None}

    def generate_x_input_value(input_binding):
        if state["shape1"] is None:
            state["shape1"] = input_binding.shape
        return rule.default(input_binding)

    def generate_y_input_value(input_binding):
        if state["shape2"] is None:
            state["shape2"] = input_binding.shape
        return rule.default(input_binding)

    def generate_axes_input_value(input_binding):
        axes_arg = rule.arg("axes")
        rank = len(state["shape1"])
        if isinstance(axes_arg, (list, tuple)):
            if state["tensor1"] is None:
                result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
                used = []
                for index in range(shape_numel(input_binding.shape)):
                    result[index] = rule.ops.randint(0, rank)
                    while (
                        state["shape1"][result[index]] not in state["shape2"]
                        or result[index] in used
                    ):
                        result[index] = rule.ops.randint(0, rank)
                    used.append(result[index])
                state["tensor1"] = result
                return result
            result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
            used = []
            for index in range(shape_numel(input_binding.shape)):
                result[index] = rule.ops.randint(0, rank)
                while (
                    state["shape2"][result[index]] != state["shape1"][state["tensor1"][index]]
                    or result[index] in used
                ):
                    result[index] = rule.ops.randint(0, rank)
                used.append(result[index])
            return result
        if input_binding.shape == () or shape_numel(input_binding.shape) == 1:
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
            return rule.ops.asarray([rule.ops.choice(candidates)], dtype=input_binding.dtype)
        result = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        used1 = []
        used2 = []
        for index in range(input_binding.shape[0]):
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
            ("x", generate_x_input_value),
            ("y", generate_y_input_value),
            ("axes", generate_axes_input_value),
        ),
    )


# embedding 规则根据词表大小和 padding index 限制输入索引。
@input_rules.register("paddle.nn.functional.embedding")
def generate_embedding_inputs(rule: InputRuleContext):
    """输入规则：按词表大小生成 embedding ids，并收紧权重值域。"""

    def generate_ids_input_value(input_binding):
        weight_config = rule.arg("weight")
        vocab_size = rule.ops.randint(10, 1000)
        if weight_config is not None and weight_config.shape:
            vocab_size = weight_config.shape[0]
        if vocab_size == 0:
            return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(0, vocab_size, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            (("x", "ids"), generate_ids_input_value),
            ("weight", "multiply"),
        ),
    )


@input_rules.register("paddle.nn.functional.affine_grid")
def generate_affine_grid_inputs(rule: InputRuleContext):
    """输入规则：根据 theta 批次维生成相容的输出 shape。"""

    def generate_out_shape_input_value(input_binding):
        theta_shape = rule.arg("theta").shape
        out_shape = rule.ops.cast(
            rule.ops.randint(1, 128, shape=input_binding.shape),
            input_binding.dtype,
        )
        out_shape[0] = theta_shape[0]
        return out_shape

    rule.generate((("out_shape", generate_out_shape_input_value),))


# 检测与 margin 损失规则根据类别数生成标签和路径表。
@input_rules.register("paddle.nn.functional.hsigmoid_loss")
def generate_hsigmoid_loss_inputs(rule: InputRuleContext):
    """输入规则：按类别和权重规模约束标签及路径编码。"""

    def generate_label_input_value(input_binding):
        num_classes = rule.arg("num_classes")
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=input_binding.shape),
            input_binding.dtype,
        )

    def generate_path_table_input_value(input_binding):
        weight = rule.arg("weight")
        return rule.ops.cast(
            rule.ops.randint(0, weight.shape[0], shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("label", generate_label_input_value),
            ("path_table", generate_path_table_input_value),
            ("path_code", "binary_0_1"),
        ),
    )


@input_rules.register("paddle.nn.functional.margin_cross_entropy")
def generate_margin_cross_entropy_inputs(rule: InputRuleContext):
    """输入规则：根据 logits 类别维生成合法标签。"""

    def generate_label_input_value(input_binding):
        logits = rule.arg("logits")
        return rule.ops.cast(
            rule.ops.randint(0, logits.shape[1], shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("label", generate_label_input_value),))


@input_rules.register("paddle.nn.functional.multi_margin_loss")
def generate_multi_margin_loss_inputs(rule: InputRuleContext):
    """输入规则：根据输入类别维生成合法标签。"""

    def generate_label_input_value(input_binding):
        logits = rule.arg("input")
        return rule.ops.cast(
            rule.ops.randint(0, logits.shape[1], shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("label", generate_label_input_value),))


# 分类损失规则统一处理类别轴、soft label、权重和 ignore index。
@input_rules.register("paddle.nn.functional.dice_loss")
def generate_dice_loss_inputs(rule: InputRuleContext):
    """输入规则：生成概率输入并根据末维生成 dice loss 标签。"""

    def generate_probability_input_value(input_binding):
        # dice loss 接收类别概率；远离全零输入可避免空交集分母退化。
        return rule.domain("uniform", input_binding, low=0.05, high=0.95)

    def generate_label_input_value(input_binding):
        tensor = rule.arg("input")
        return rule.ops.cast(
            rule.ops.randint(0, tensor.shape[-1], shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("input", generate_probability_input_value),
            ("label", generate_label_input_value),
        )
    )


@input_rules.register("paddle.nn.functional.nll_loss")
def generate_nll_loss_inputs(rule: InputRuleContext):
    """输入规则：根据输入类别维生成 NLL 标签。"""

    def generate_label_input_value(input_binding):
        input_config = rule.arg("input")
        n_classes = rule.ops.randint(5, 50) if input_config is None else input_config.shape[1]
        return rule.ops.cast(
            rule.ops.randint(0, n_classes, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("label", generate_label_input_value),))


@input_rules.register("paddle.nn.functional.adaptive_log_softmax_with_loss")
def generate_adaptive_log_softmax_with_loss_inputs(rule: InputRuleContext):
    """输入规则：根据 cutoffs 推导类别数并生成合法标签。"""

    def generate_label_input_value(input_binding):
        cutoffs = rule.arg("cutoffs")
        n_classes = cutoffs[-1]
        generation_size = input_binding.shape
        if len(input_binding.shape) == 0:
            generation_size = 1
        if n_classes == 1:
            return rule.ops.zeros(generation_size, dtype=input_binding.dtype)
        return rule.ops.randint(0, n_classes, shape=generation_size, dtype=input_binding.dtype)

    rule.generate((("label", generate_label_input_value),))


@input_rules.register("paddle.nn.functional.cross_entropy")
def generate_cross_entropy_inputs(rule: InputRuleContext):
    """输入规则：联动 axis、soft_label 和平滑系数生成标签与权重。"""

    def generate_cross_entropy_input_value(input_binding):
        use_softmax = rule.arg("use_softmax", True)
        if use_softmax:
            return rule.default(input_binding)
        axis = rule.arg("axis", -1)
        logits = rule.ops.random(input_binding.shape)
        probabilities = logits / rule.ops.sum(logits, axis=axis, keepdims=True)
        return rule.ops.cast(probabilities, input_binding.dtype)

    def generate_label_input_value(input_binding):
        input_shape = rule.arg("input").shape
        axis = rule.arg("axis", -1)
        num_classes = input_shape[axis]
        soft_label = rule.arg("soft_label", False)
        label_smoothing = rule.arg("label_smoothing", 0.0)
        if (label_smoothing > 0 and list(input_binding.shape) == list(input_shape)) or (
            label_smoothing == 0 and soft_label
        ):
            soft_labels = rule.ops.random(input_binding.shape)
            soft_labels = soft_labels / rule.ops.sum(soft_labels, axis=axis, keepdims=True)
            return rule.ops.cast(soft_labels, input_binding.dtype)
        if num_classes == 0:
            return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        return rule.ops.cast(
            rule.ops.randint(0, num_classes, shape=input_binding.shape),
            input_binding.dtype,
        )

    def generate_weight_input_value(input_binding):
        weights = rule.ops.random(input_binding.shape)
        return weights / rule.ops.sum(weights)

    rule.generate(
        (
            ("input", generate_cross_entropy_input_value),
            ("label", generate_label_input_value),
            ("weight", generate_weight_input_value),
        ),
    )


# 序列损失规则联动标签长度、输入长度和时间维度上限。
@input_rules.register("paddle.nn.functional.ctc_loss")
def generate_ctc_loss_inputs(rule: InputRuleContext):
    """输入规则：联动 blank、标签范围和序列长度生成 CTC 输入。"""

    def generate_labels_input_value(input_binding):
        num_classes = rule.arg("log_probs").shape[2] - 1
        blank = rule.arg("blank", 0)
        valid_label_indices = [index for index in range(num_classes + 1) if index != blank]
        if not valid_label_indices:
            return rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        return rule.ops.cast(
            rule.ops.choice(valid_label_indices, shape=input_binding.shape, replace=True),
            input_binding.dtype,
        )

    def generate_input_lengths_input_value(input_binding):
        max_logit_length = rule.arg("log_probs").shape[0]
        return rule.ops.randint(
            1,
            max_logit_length + 1,
            shape=input_binding.shape,
            dtype=input_binding.dtype,
        )

    def generate_label_lengths_input_value(input_binding):
        max_label_length = rule.arg("labels").shape[1]
        max_logit_length = rule.arg("log_probs").shape[0]
        cand_label_lengths = rule.ops.randint(
            1,
            max_label_length + 1,
            shape=input_binding.shape,
            dtype=input_binding.dtype,
        )
        compatible_input_lengths = rule.ops.randint(
            1,
            max_logit_length + 1,
            shape=input_binding.shape,
            dtype=input_binding.dtype,
        )
        final_label_lengths = rule.ops.minimum(cand_label_lengths, compatible_input_lengths)
        return rule.ops.maximum(final_label_lengths, 1)

    rule.generate(
        (
            ("labels", generate_labels_input_value),
            ("input_lengths", generate_input_lengths_input_value),
            ("label_lengths", generate_label_lengths_input_value),
        ),
    )


@input_rules.register("paddle.nn.functional.sequence_mask")
def generate_sequence_mask_inputs(rule: InputRuleContext):
    """输入规则：根据 maxlen 约束 sequence_mask 的长度输入。"""

    def generate_x_input_value(input_binding):
        maxlen_config = rule.arg("maxlen")
        provided_maxlen = None
        if isinstance(maxlen_config, int):
            provided_maxlen = max(1, maxlen_config)
        if provided_maxlen is not None:
            return rule.ops.cast(
                rule.ops.randint(0, provided_maxlen + 1, shape=input_binding.shape),
                input_binding.dtype,
            )
        high_value = rule.ops.randint(1, 2048)
        lengths = rule.ops.cast(
            rule.ops.randint(0, high_value, shape=input_binding.shape),
            input_binding.dtype,
        )
        if rule.ops.prod(lengths.shape) > 0 and rule.ops.count_nonzero(lengths) == 0:
            fix_value = rule.ops.randint(1, max(2, high_value))
            rule.ops.flatten(lengths)[0] = fix_value
        return lengths

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.nn.functional.softmax_with_cross_entropy")
def generate_softmax_with_cross_entropy_inputs(rule: InputRuleContext):
    """输入规则：根据 logits 类别维生成交叉熵标签。"""

    def generate_label_input_value(input_binding):
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
            rule.ops.randint(0, num_classes, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate((("label", generate_label_input_value),))


# 线性代数分解规则构造对称、正定或满足分解前提的矩阵。
@input_rules.register("paddle.linalg.cholesky")
def generate_cholesky_inputs(rule: InputRuleContext):
    """输入规则：构造对称正定矩阵供 Cholesky 分解使用。"""

    def generate_x_input_value(input_binding):
        if len(input_binding.shape) < 2 or input_binding.shape[-1] != input_binding.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = input_binding.shape[:-2]
        matrix_dim = input_binding.shape[-1]
        if input_binding.dtype.startswith("complex"):
            # 复数正定矩阵必须使用共轭转置，普通转置不保证 Hermitian。
            matrix = rule.uniform(input_binding, 0, 1)
            matrix_h = rule.ops.conj(rule.ops.swapaxes(matrix, -1, -2))
            tensor = rule.ops.matmul(matrix, matrix_h)
        else:
            matrix = rule.ops.random(input_binding.shape, dtype=input_binding.dtype)
            if batch_dims:
                tensor = rule.ops.einsum("...ij,...kj->...ik", matrix, matrix)
            else:
                tensor = rule.ops.dot(matrix, rule.ops.swapaxes(matrix, -1, -2))
        tensor += rule.ops.eye(matrix_dim, dtype=input_binding.dtype) * 10000
        return tensor

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.cov")
def generate_covariance_inputs(rule: InputRuleContext):
    """输入规则：根据 rowvar 语义生成非退化协方差输入和权重。"""

    def observation_count():
        x_shape = rule.arg("x").shape
        rowvar = rule.arg("rowvar")
        if rowvar is None:
            rowvar = True
        return (x_shape[1] if rowvar else x_shape[0]) if len(x_shape) > 1 else x_shape[0]

    def generate_x_input_value(input_binding):
        if len(input_binding.shape) < 1 or len(input_binding.shape) > 2:
            raise ValueError("Shape must have 1 or 2 dimensions for covariance input")
        if input_binding.dtype.startswith("complex"):
            # complex 样本独立生成实部和虚部，权重仍保持实数协议。
            tensor = rule.uniform(input_binding, 0, 1)
            tensor += rule.uniform(input_binding, 0, 1) * 1e-6
        else:
            tensor = rule.ops.random(input_binding.shape, dtype=input_binding.dtype)
            tensor += rule.ops.random(input_binding.shape, dtype=input_binding.dtype) * 1e-6
        return tensor

    def generate_fweights_input_value(input_binding):
        return rule.ops.cast(
            rule.ops.randint(1, 11, shape=(observation_count(),)),
            input_binding.dtype,
        )

    def generate_aweights_input_value(input_binding):
        if input_binding.dtype in ["float32", "float64"]:
            return rule.ops.uniform(
                0.1, 1.0, shape=(observation_count(),), dtype=input_binding.dtype
            )
        return rule.ops.cast(
            rule.ops.randint(1, 11, shape=(observation_count(),)),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("x", generate_x_input_value),
            ("fweights", generate_fweights_input_value),
            ("aweights", generate_aweights_input_value),
        ),
    )


# 谱分解与最小二乘规则根据矩阵维度联动输入、权重和 pivot。
@input_rules.register("paddle.linalg.eigh", "paddle.linalg.eigvalsh")
def generate_eigen_symmetric_inputs(rule: InputRuleContext):
    """输入规则：构造实对称或复 Hermitian 特征分解输入。"""

    def generate_x_input_value(input_binding):
        if len(input_binding.shape) < 2 or input_binding.shape[-1] != input_binding.shape[-2]:
            raise ValueError(
                "Shape must have at least 2 dimensions and last two dimensions must be equal"
            )
        batch_dims = input_binding.shape[:-2]
        matrix_dim = input_binding.shape[-1]
        if input_binding.dtype.startswith("complex"):
            # Hermitian 输入由 complex 矩阵及其共轭转置构造。
            matrix = rule.uniform(input_binding, 0, 1)
            matrix_h = rule.ops.conj(rule.ops.swapaxes(matrix, -1, -2))
            tensor = matrix + matrix_h
        else:
            matrix = rule.ops.random(input_binding.shape, dtype=input_binding.dtype)
            if batch_dims:
                tensor = rule.ops.einsum("...ij,...kj->...ik", matrix, matrix)
            else:
                tensor = rule.ops.dot(matrix, rule.ops.swapaxes(matrix, -1, -2))
        tensor += rule.ops.eye(matrix_dim, dtype=input_binding.dtype) * 1e-6
        return tensor

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.lstsq")
def generate_lstsq_inputs(rule: InputRuleContext):
    """输入规则：为最小二乘生成至少二维且批次相容的矩阵。"""

    def generate_matrix_input_value(input_binding):
        if len(input_binding.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for lstsq x")
        # lstsq 只要求二维矩阵，x 和 y 没有额外数值定义域。
        return rule.default(input_binding)

    rule.generate(((("x", "y"), generate_matrix_input_value),))


@input_rules.register("paddle.linalg.lu_unpack")
def generate_lu_unpack_inputs(rule: InputRuleContext):
    """输入规则：生成非奇异 LU 数据和范围合法的 pivot。"""

    def generate_x_input_value(input_binding):
        if len(input_binding.shape) < 2:
            raise ValueError("Shape must have at least 2 dimensions for LU matrix")
        tensor = (
            rule.uniform(input_binding, 0, 1)
            if input_binding.dtype.startswith("complex")
            else rule.ops.random(input_binding.shape, dtype=input_binding.dtype)
        )
        diagonal_size = min(input_binding.shape[-2], input_binding.shape[-1])
        tensor[..., range(diagonal_size), range(diagonal_size)] += 1e-6
        return tensor

    def generate_pivot_input_value(input_binding):
        row_count = rule.arg("x").shape[-2]
        return rule.ops.cast(
            rule.ops.randint(1, row_count + 1, shape=input_binding.shape),
            input_binding.dtype,
        )

    rule.generate(
        (
            ("x", generate_x_input_value),
            (("pivot", "y"), generate_pivot_input_value),
        ),
    )


@input_rules.register("paddle.linalg.cond")
def generate_condition_inputs(rule: InputRuleContext):
    """输入规则：构造数值稳定的方阵用于条件数计算。"""

    def generate_x_input_value(input_binding):
        matrix_size = input_binding.shape[-1]
        tensor = (
            rule.uniform(input_binding, 0, 1)
            if input_binding.dtype.startswith("complex")
            else rule.ops.random(input_binding.shape, dtype=input_binding.dtype)
        )
        tensor += matrix_size * rule.ops.eye(matrix_size, dtype=input_binding.dtype)
        return tensor

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.det", "paddle.linalg.slogdet")
def generate_determinant_inputs(rule: InputRuleContext):
    """输入规则：构造可逆方阵用于 det 和 slogdet。"""

    def generate_x_input_value(input_binding):
        if len(input_binding.shape) < 2:
            raise AssertionError("Input must be at least 2D.")
        if input_binding.shape[-1] != input_binding.shape[-2]:
            raise AssertionError("Input must be square matrices.")
        matrix_size = input_binding.shape[-1]
        matrix = rule.uniform(input_binding, 0.5, 1.0)
        matrix_h = rule.ops.swapaxes(rule.ops.conj(matrix), -1, -2)
        return rule.ops.matmul(matrix, matrix_h) + rule.ops.eye(
            matrix_size, dtype=input_binding.dtype
        )

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.pca_lowrank")
def generate_pca_lowrank_inputs(rule: InputRuleContext):
    """输入规则：为低秩 PCA 生成受控随机矩阵。"""

    def generate_x_input_value(input_binding):
        return rule.normal(input_binding)

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.corrcoef")
def generate_corrcoef_inputs(rule: InputRuleContext):
    """输入规则：为相关系数计算生成带微小扰动的非退化输入。"""

    def generate_x_input_value(input_binding):
        if input_binding.dtype == "float16":
            return rule.normal(input_binding, scale=1e-3)
        return rule.default(input_binding)

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.pinv")
def generate_pinv_inputs(rule: InputRuleContext):
    """输入规则：构造满秩一般矩阵或正定 Hermitian 矩阵。"""
    hermitian = bool(rule.arg("hermitian", False))

    def generate_x_input_value(tensor):
        if len(tensor.shape) < 2:
            raise ValueError("pinv input must have at least two dimensions")
        matrix = rule.normal(tensor)
        if not hermitian:
            # 连续随机的一般矩阵以概率 1 满秩，并保留矩形矩阵覆盖。
            return matrix
        if tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("hermitian pinv input must be square")
        # M M^H + I 的最小特征值有正下界，避免随机对称矩阵落在奇异点。
        matrix_h = rule.ops.swapaxes(rule.ops.conj(matrix), -1, -2)
        return rule.ops.matmul(matrix, matrix_h) + rule.ops.eye(
            tensor.shape[-1], dtype=tensor.dtype
        )

    def generate_rcond_input_value(tensor):
        # Tensor rcond 遵守非负相对阈值语义，并避免阈值大于最大奇异值。
        return rule.domain("uniform", tensor, low=1e-6, high=0.1)

    rule.generate(
        (
            ("x", generate_x_input_value),
            ("rcond", generate_rcond_input_value),
        )
    )


@input_rules.register("paddle.linalg.triangular_solve")
def generate_triangular_solve_inputs(rule: InputRuleContext):
    """输入规则：生成方向匹配且对角稳定的三角系数矩阵。"""

    def generate_x_input_value(tensor):
        if len(tensor.shape) < 2 or tensor.shape[-1] != tensor.shape[-2]:
            raise ValueError("triangular_solve x must contain square matrices")
        matrix = rule.default(tensor)
        matrix = rule.ops.triu(matrix) if rule.arg("upper", True) else rule.ops.tril(matrix)
        if rule.arg("unitriangular", False):
            # unitriangular 协议忽略输入对角，无需人为覆盖该区域。
            return matrix
        # 固定对角偏移使系数矩阵远离奇异点，同时保留非对角随机覆盖。
        return matrix + rule.ops.eye(tensor.shape[-1], dtype=tensor.dtype) * 2

    rule.generate((("x", generate_x_input_value),))


@input_rules.register("paddle.linalg.cholesky_solve", aliases=("paddle.Tensor.cholesky_solve",))
def generate_cholesky_solve_inputs(rule: InputRuleContext):
    """输入规则：按 upper 参数生成与三角因子方向一致的输入。"""
    if rule.api_name == "paddle.linalg.cholesky_solve":
        rule.generate_all()
        return

    def generate_y_input_value(input_binding):
        value = rule.domain("random_range", input_binding)
        if rule.arg("upper"):
            return rule.ops.triu(value)
        return rule.ops.tril(value)

    rule.generate((("y", generate_y_input_value),))


@input_rules.register("paddle.view", aliases=("paddle.Tensor.view",))
def generate_view_inputs(rule: InputRuleContext):
    """输入规则：按目标 dtype 或 shape 约束 view 的底层字节布局。"""

    def generate_x_input_value(input_binding):
        if input_binding.dtype == "uint8":
            target = str(rule.arg("shape_or_dtype", ""))
            nbytes = shape_numel(input_binding.shape)
            # view 的随机数只用于构造有限字节，不能跟随全局压力测试范围。
            finite_max_abs = 0.6
            itemsize = {
                "paddle.bfloat16": 2,
                "paddle.float16": 2,
                "paddle.float32": 4,
                "paddle.float64": 8,
            }.get(target)
            if itemsize is not None and nbytes % itemsize == 0:
                numel = nbytes // itemsize
                if target == "paddle.bfloat16":
                    finite_f32 = generate_symmetric_input_value(
                        replace(input_binding.input_spec, shape=(numel,), dtype="float32"),
                        finite_max_abs,
                        rule.ops,
                    )
                    uint32_value = rule.ops.view_dtype(finite_f32, "uint32")
                    return rule.ops.view_dtype(
                        rule.ops.cast(
                            rule.ops.cast(uint32_value, "int64") >> 16,
                            "uint16",
                        ),
                        "uint8",
                    )
                finite = generate_symmetric_input_value(
                    replace(
                        input_binding.input_spec,
                        shape=(numel,),
                        dtype=target.replace("paddle.", ""),
                    ),
                    finite_max_abs,
                    rule.ops,
                )
                return rule.ops.view_dtype(rule.ops.ascontiguousarray(finite), "uint8")
        return rule.default(input_binding)

    rule.generate((("x", generate_x_input_value),))


@input_rules.register(
    "paddle.pow",
    aliases=("paddle.Tensor.pow", "paddle.Tensor.__rpow__", "paddle.Tensor.__pow__"),
)
def generate_pow_inputs(rule: InputRuleContext):
    """输入规则：根据底数、指数和正反向幂语义限制数值范围。"""

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

    def generate_power_input_value(input_binding):
        api_name = rule.api_name
        dtype = input_binding.dtype
        if api_name == "paddle.Tensor.__rpow__":
            base_name, exponent_name = "y", "self"
        elif api_name == "paddle.Tensor.__pow__":
            base_name, exponent_name = "self", "y"
        else:
            base_name, exponent_name = "x", "y"
        is_base_arg = input_binding.parameter_name == base_name
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
                return rule.domain("random_range", input_binding, low=0, high=value_max)
            return rule.domain("random_range", input_binding, low=-value_max, high=value_max)
        if is_base_arg:
            return rule.domain("random_range", input_binding, low=0, high=default_max)
        return rule.domain("random_range", input_binding, low=-default_max, high=default_max)

    rule.generate_all(generate_power_input_value)


@input_rules.register("paddle.nn.functional.rnnt_loss")
def generate_rnnt_loss_inputs(rule: InputRuleContext):
    """输入规则：联动 logits、labels 和长度 Tensor 的默认形状。"""

    def generate_logits_input_value(input_binding):
        # RNNT 规则只补合法四维 shape，logits 数值仍属于 default。
        shape = input_binding.shape if len(input_binding.shape) == 4 else (3, 4, 3, 5)
        return rule.default(input_binding, shape=shape)

    def generate_labels_input_value(input_binding):
        shape = input_binding.shape if len(input_binding.shape) == 2 else (3, 2)
        return rule.ops.cast(rule.ops.randint(1, 4, shape=shape), input_binding.dtype)

    def create_length_input_value_generator(max_possible_length):
        def generate_length_input_value(input_binding):
            shape = input_binding.shape if len(input_binding.shape) == 1 else (3,)
            return rule.ops.ones(shape, dtype=input_binding.dtype) * max_possible_length

        return generate_length_input_value

    rule.generate(
        (
            (("input", "logits"), generate_logits_input_value),
            (("label", "labels"), generate_labels_input_value),
            ("input_lengths", create_length_input_value_generator(4)),
            ("label_lengths", create_length_input_value_generator(2)),
        ),
    )


# 分块、拆分和扩展规则校验目标 shape 与源维度的可实现性。
@input_rules.register("paddle.chunk")
def generate_chunk_inputs(rule: InputRuleContext):
    """输入规则：选择能被 chunks 整除的输入维度作为 axis。"""

    def generate_axis_input_value(input_binding):
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
        if len(input_binding.shape) == 0:
            return rule.ops.asarray(chosen_axis, dtype=input_binding.dtype)
        if len(input_binding.shape) == 1 and input_binding.shape[0] == 1:
            return rule.ops.asarray([chosen_axis], dtype=input_binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
            f"Expected a 0-D or 1-D Tensor, but got shape {input_binding.shape}."
        )

    rule.generate((("axis", generate_axis_input_value),))


@input_rules.register("paddle.split")
def generate_split_inputs(rule: InputRuleContext):
    """输入规则：根据分段数量或 section 总和选择合法 axis。"""

    def generate_axis_input_value(input_binding):
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
        if len(input_binding.shape) == 0:
            return rule.ops.asarray(target_dim, dtype=input_binding.dtype)
        if len(input_binding.shape) == 1 and input_binding.shape[0] == 1:
            return rule.ops.asarray([target_dim], dtype=input_binding.dtype)
        raise ValueError(
            f"Invalid shape for 'axis' Tensor in paddle.split. "
            f"Expected a 0-D or 1-D Tensor, but got shape {input_binding.shape}."
        )

    rule.generate((("axis", generate_axis_input_value),))


@input_rules.register("paddle.expand", aliases=("paddle.Tensor.expand",))
def generate_expand_inputs(rule: InputRuleContext):
    """输入规则：按源 shape 生成满足广播规则的目标 shape。"""

    def generate_shape_input_value(input_binding):
        x_shape = rule.arg("x").shape
        shape_index = input_binding.path.item_indices[0] if input_binding.path.item_indices else 0
        if len(x_shape) == 0 or shape_index > len(x_shape) - 1 or x_shape[shape_index] == 1:
            return rule.ops.cast(
                rule.ops.randint(1, 127, shape=input_binding.shape),
                input_binding.dtype,
            )
        if len(input_binding.shape) == 0 or input_binding.shape[0] == 1:
            return rule.ops.asarray(x_shape[shape_index])
        shape_values = rule.ops.cast(
            rule.ops.randint(1, 127, shape=input_binding.shape),
            input_binding.dtype,
        )
        offset = input_binding.shape[0] - len(x_shape)
        for index in range(input_binding.shape[0]):
            if index >= offset and x_shape[index - offset] != 1:
                shape_values[index] = x_shape[index - offset]
        return shape_values

    rule.generate((("shape", generate_shape_input_value),))


@input_rules.register("paddle.nn.functional.gather_tree")
def generate_gather_tree_inputs(rule: InputRuleContext):
    """输入规则：根据 beam size 生成合法父节点索引。"""

    def generate_parents_input_value(input_binding):
        ids = rule.arg("ids")
        if hasattr(ids, "shape") and len(ids.shape) >= 3:
            beam_size = ids.shape[2]
        else:
            beam_size = input_binding.shape[2] if len(input_binding.shape) >= 3 else 4
        beam_size = 1 if beam_size < 1 else beam_size
        parents = rule.ops.zeros(input_binding.shape, dtype=input_binding.dtype)
        for time_index in range(input_binding.shape[0]):
            for batch_index in range(input_binding.shape[1]):
                for beam_index in range(input_binding.shape[2]):
                    parents[time_index, batch_index, beam_index] = rule.ops.randint(0, beam_size)
        return parents

    rule.generate((("parents", generate_parents_input_value),))


@input_rules.register("paddle.multinomial")
def generate_multinomial_inputs(rule: InputRuleContext):
    """输入规则：生成非负权重并按 replacement 限制采样数量。"""
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


@input_rules.register("paddle.nn.functional.one_hot")
def generate_one_hot_inputs(rule: InputRuleContext):
    """输入规则：联动 num_classes 与输入索引的取值范围。"""
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


def _apply_input_value(api_config, input_value: InputValue, update_config):
    tensor_config = input_tensor_config_at(api_config, input_value.path)
    if update_config:
        dtype_name = str(getattr(input_value.generated_value, "dtype", ""))
        dtype_name = dtype_name.split(".")[-1] if dtype_name else dtype_name
        if tensor_config.dtype not in CAST_THROUGH_INTERMEDIATE_DTYPES:
            tensor_config.dtype = dtype_name
        tensor_config.shape = list(input_value.generated_value.shape)
