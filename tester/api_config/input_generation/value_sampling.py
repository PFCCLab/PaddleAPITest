"""供装饰器规则复用的纯 NumPy 值生成器。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy

from .case_model import TensorSpec

# 这些中间 dtype 转换要与历史路径一致，才能保持输出字节稳定。
_INTERMEDIATE_DTYPES = {
    "bfloat16": "float32",
    "float8_e4m3fn": "float16",
    "float8_e5m2": "float16",
}


class NumPyRNG:
    """全局 NumPy RNG 的兼容适配器。"""

    def random(self, *args, **kwargs):
        return numpy.random.random(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return numpy.random.randint(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return numpy.random.uniform(*args, **kwargs)

    def randn(self, *args, **kwargs):
        return numpy.random.randn(*args, **kwargs)

    def choice(self, *args, **kwargs):
        return numpy.random.choice(*args, **kwargs)


NUMPY_RNG = NumPyRNG()


@dataclass(frozen=True)
class CaseNumpyRNG(NumPyRNG):
    """基于独立 RandomState 副本的单 case RNG 外观。"""

    seed: int
    config_fingerprint: str
    runtime_mode: str
    backend: str = "case-owned-numpy-state"
    _state: numpy.random.RandomState = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        state = numpy.random.RandomState()
        state.set_state(numpy.random.get_state())
        object.__setattr__(self, "_state", state)

    def commit(self):
        numpy.random.set_state(self._state.get_state())

    def random(self, *args, **kwargs):
        return self._state.random(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return self._state.randint(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self._state.uniform(*args, **kwargs)

    def randn(self, *args, **kwargs):
        return self._state.randn(*args, **kwargs)

    def choice(self, *args, **kwargs):
        return self._state.choice(*args, **kwargs)


def create_case_rng(context) -> CaseNumpyRNG:
    # 每个 case 都拿到自己的 RNG 快照，只在成功后提交回全局。
    return CaseNumpyRNG(
        seed=context.seed,
        config_fingerprint=context.config_fingerprint,
        runtime_mode=context.runtime_mode,
    )


def generation_dtype(dtype: str) -> str:
    return _INTERMEDIATE_DTYPES.get(dtype, dtype)


def generate_default(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """匹配迁移后的默认生成分支。"""
    dtype = generation_dtype(spec.dtype)
    shape = spec.shape
    if not shape:
        if "int" in dtype:
            return numpy.asarray(rng.randint(-65535, 65535)).astype(dtype)
        if dtype.startswith("complex"):
            real = (rng.random() - 0.5) * 1.2
            imag = (rng.random() - 0.5) * 1.2
            return numpy.array(real + 1j * imag, dtype=dtype)
        return numpy.array((rng.random() - 0.5) * 1.2, dtype=dtype)

    if "int" in dtype:
        return rng.randint(-65535, 65535, size=shape).astype(dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = ((rng.random(shape) - 0.5) * 1.2).astype(real_dtype)
        imag = ((rng.random(shape) - 0.5) * 1.2).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return ((rng.random(shape) - 0.5) * 1.2).astype(dtype)


def generate_nonzero(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """匹配迁移后的非零生成分支。"""
    dtype = generation_dtype(spec.dtype)
    shape = spec.shape
    if "int" in dtype:
        if dtype == "int8":
            values = rng.randint(1, 256, size=shape, dtype=numpy.int32)
            values[values > 127] -= 256
            return values.astype(dtype)
        if dtype == "uint8":
            return rng.randint(1, 256, size=shape).astype(dtype)
        return rng.randint(1, 65535, size=shape).astype(dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = (rng.random(shape) + 0.5).astype(real_dtype)
        imag = (rng.random(shape) + 0.5).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return (rng.random(shape) + 0.5).astype(dtype)


def generate_fill_value(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 `paddle.full` fill_value。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(1, 65535, size=spec.shape).astype(dtype)
    return (rng.random(spec.shape) + 0.5).astype(dtype)


def generate_unit_interval(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 [0, 1) 随机值。"""
    return rng.random(spec.shape).astype(generation_dtype(spec.dtype))


def generate_multiply(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 `paddle.multiply` 值。"""
    dtype = generation_dtype(spec.dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = rng.random(spec.shape).astype(real_dtype)
        imag = rng.random(spec.shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return rng.random(spec.shape).astype(dtype)


def generate_unit_plus_one(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 [1, 2) 随机值。"""
    return (rng.random(spec.shape) + 1.0).astype(generation_dtype(spec.dtype))


def generate_signed_half(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 [-0.5, 0.5) 随机值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(-65535, 65535, size=spec.shape).astype(dtype)
    return (rng.random(spec.shape) - 0.5).astype(dtype)


def generate_normal_std(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 `paddle.normal` std 参数。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(0, 65535, size=spec.shape).astype(dtype)
    return rng.random(spec.shape).astype(dtype)


def generate_int_1024(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [0, 1024)。"""
    return rng.randint(0, 1024, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_64(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [0, 64)。"""
    return rng.randint(0, 64, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_2048_raw(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [0, 2048)，但不做 dtype 强转。"""
    return rng.randint(0, 2048, size=spec.shape)


def generate_int_128(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [1, 128)。"""
    return rng.randint(1, 128, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_one_10(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [1, 10)。"""
    return rng.randint(1, 10, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_empty_shape(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 `paddle.empty` shape Tensor 值。"""
    dtype = generation_dtype(spec.dtype) if "int" in spec.dtype else "int32"
    return rng.randint(1, 10, size=spec.shape).astype(dtype)


def generate_int_2048(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [1, 2048)。"""
    return rng.randint(1, 2048, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_65535(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [1, 65535)。"""
    return rng.randint(1, 65535, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_65535_raw(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间 [1, 65535)，但不做 dtype 强转。"""
    return rng.randint(1, 65535, size=spec.shape)


def generate_ones_shape(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 `paddle.ones` shape Tensor 值。"""
    dtype = generation_dtype(spec.dtype)
    if len(spec.shape) == 0:
        return numpy.array(rng.randint(1, 2048), dtype=dtype)
    return rng.randint(1, 65535, size=spec.shape).astype(dtype)


def generate_int_or_unit(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的整数区间或单位区间值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(0, 65535, size=spec.shape).astype(dtype)
    return rng.random(spec.shape).astype(dtype)


def generate_int_or_default(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 `paddle.dot` 值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(-127, 127, size=spec.shape).astype(dtype)
    return generate_default(spec, rng)


def generate_binary01(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的二值标签 {0, 1}。"""
    return rng.randint(0, 2, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_hinge_label(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 hinge 标签 {-1, 1}。"""
    values = generate_binary01(spec, rng)
    values[values == 0] = -1
    return values


def generate_abs_plus_one(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的上采样 scale_factor 值。"""
    dtype = generation_dtype(spec.dtype)
    return numpy.ones(spec.shape).astype(dtype) + numpy.abs(rng.random(spec.shape)).astype(dtype)


def generate_dropout_prob(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 dropout 概率 Tensor 值。"""
    value = generate_uniform(spec, 0, 1.1, rng)
    return numpy.where(value > 1, 1, value)


def generate_quantile(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 quantile q 值。"""
    return rng.random(1).astype(generation_dtype(spec.dtype))


def generate_remainder(spec: TensorSpec, rng=NUMPY_RNG) -> numpy.ndarray:
    """生成迁移后的 remainder 除数值。"""
    dtype = generation_dtype(spec.dtype)
    if dtype in {"int32", "int64"}:
        return generate_uniform(spec, 1, 65535, rng)
    return generate_default(spec, rng)


def generate_random_range(
    spec: TensorSpec,
    low=None,
    high=None,
    rng=NUMPY_RNG,
) -> numpy.ndarray:
    """匹配迁移后规则调用点的 `TensorConfig.get_random_numpy_tensor` 行为。"""
    # 这里要保留 TensorConfig 按 dtype 区分的大范围取值语义。
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        low = low if low is not None else -65535
        high = high if high is not None else 65535
        return rng.randint(low, high, size=spec.shape).astype(dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real_low = low if low is not None else numpy.finfo(real_dtype).min / 2
        real_high = high if high is not None else numpy.finfo(real_dtype).max / 2
        real = rng.uniform(real_low, real_high, size=spec.shape).astype(real_dtype)
        imag = rng.uniform(real_low, real_high, size=spec.shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    low = low if low is not None else numpy.finfo(dtype).min / 2
    high = high if high is not None else numpy.finfo(dtype).max / 2
    return rng.uniform(low, high, size=spec.shape).astype(dtype)


def generate_uniform(
    spec: TensorSpec,
    low,
    high,
    rng=NUMPY_RNG,
) -> numpy.ndarray:
    """匹配迁移后的固定区间随机生成器。"""
    # `uniform` 是这些区间类 helper 的共同底层原语。
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(low, high, size=spec.shape).astype(dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = rng.uniform(low, high, size=spec.shape).astype(real_dtype)
        imag = rng.uniform(low, high, size=spec.shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return rng.uniform(low, high, size=spec.shape).astype(dtype)


# 兼容别名用于平滑迁移；新代码应优先使用短命名。
generate_full_fill_value = generate_fill_value
generate_unit_interval_plus_one = generate_unit_plus_one
generate_signed_half_interval = generate_signed_half
generate_int_zero_1024 = generate_int_1024
generate_int_zero_64 = generate_int_64
generate_int_zero_2048_no_cast = generate_int_2048_raw
generate_int_one_128 = generate_int_128
generate_int_one_2048 = generate_int_2048
generate_int_one_65535 = generate_int_65535
generate_int_one_65535_no_cast = generate_int_65535_raw
generate_int_zero_65535_else_unit = generate_int_or_unit
generate_int_minus127_127_else_default = generate_int_or_default
generate_binary_0_1 = generate_binary01
generate_hinge_labels = generate_hinge_label
generate_abs_unit_plus_one = generate_abs_plus_one
generate_dropout_probability = generate_dropout_prob
generate_quantile_q = generate_quantile
generate_remainder_rhs = generate_remainder
