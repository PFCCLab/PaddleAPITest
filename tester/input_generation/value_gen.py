"""供规则复用的 backend-native 值生成器。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy

from .tensor_spec import TensorSpec

# 这些中间 dtype 转换要保持固定，才能保证输出字节稳定。
_INTERMEDIATE_DTYPES = {
    "bfloat16": "float32",
    "float8_e4m3fn": "float16",
    "float8_e5m2": "float16",
}


class NumPyRNG:
    """全局 NumPy backend RNG 适配器。"""

    def random(self, shape=None):
        return numpy.random.random(shape)

    def randint(self, low, high=None, shape=None, dtype=None):
        if dtype is None:
            return numpy.random.randint(low, high, size=shape)
        return numpy.random.randint(low, high, size=shape, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, shape=None):
        return numpy.random.uniform(low, high, size=shape)

    def randn(self, *args, **kwargs):
        return numpy.random.randn(*args, **kwargs)

    def choice(self, values, shape=None, replace=True, p=None):
        return numpy.random.choice(values, size=shape, replace=replace, p=p)

    def asarray(self, value, dtype=None, copy=True):
        return numpy.array(value, dtype=dtype, copy=copy)

    def cast(self, value, dtype):
        return numpy.asarray(value).astype(dtype)

    def ones(self, shape, dtype=None):
        return numpy.ones(shape, dtype=dtype)

    def where(self, condition, x, y):
        return numpy.where(condition, x, y)

    def abs(self, value):
        return numpy.abs(value)


NUMPY_RNG = NumPyRNG()


@dataclass(frozen=True)
class ConfigNumpyRNG(NumPyRNG):
    """基于独立 RandomState 副本的单 config RNG。"""

    seed: int
    config_fingerprint: str
    _state: numpy.random.RandomState = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        state = numpy.random.RandomState()
        state.set_state(numpy.random.get_state())
        object.__setattr__(self, "_state", state)

    def commit(self):
        numpy.random.set_state(self._state.get_state())

    def random(self, shape=None):
        return self._state.random(shape)

    def randint(self, low, high=None, shape=None, dtype=None):
        if dtype is None:
            return self._state.randint(low, high, size=shape)
        return self._state.randint(low, high, size=shape, dtype=dtype)

    def uniform(self, low=0.0, high=1.0, shape=None):
        return self._state.uniform(low, high, size=shape)

    def randn(self, *args, **kwargs):
        return self._state.randn(*args, **kwargs)

    def choice(self, values, shape=None, replace=True, p=None):
        return self._state.choice(values, size=shape, replace=replace, p=p)


def create_config_rng(context) -> ConfigNumpyRNG:
    return ConfigNumpyRNG(seed=context.seed, config_fingerprint=context.config_fingerprint)


def generation_dtype(dtype: str) -> str:
    return _INTERMEDIATE_DTYPES.get(dtype, dtype)


def _complex_parts(dtype, shape, rng, *, low=None, high=None, offset=0.0, scale=1.0):
    real_dtype = "float32" if dtype == "complex64" else "float64"
    if low is None and high is None:
        real = rng.random(shape) * scale + offset
        imag = rng.random(shape) * scale + offset
    else:
        real = rng.uniform(low, high, shape=shape)
        imag = rng.uniform(low, high, shape=shape)
    return rng.cast(real, real_dtype), rng.cast(imag, real_dtype)


def _complex_value(dtype, shape, rng, **kwargs):
    real, imag = _complex_parts(dtype, shape, rng, **kwargs)
    return rng.cast(real + 1j * imag, dtype)


def generate_default(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成默认值。"""
    dtype = generation_dtype(spec.dtype)
    shape = spec.shape
    if not shape:
        if "int" in dtype:
            return rng.asarray(rng.randint(-65535, 65535), dtype=dtype)
        if dtype.startswith("complex"):
            real = (rng.random() - 0.5) * 1.2
            imag = (rng.random() - 0.5) * 1.2
            return rng.asarray(real + 1j * imag, dtype=dtype)
        return rng.asarray((rng.random() - 0.5) * 1.2, dtype=dtype)

    if "int" in dtype:
        return rng.cast(rng.randint(-65535, 65535, shape=shape), dtype)
    if dtype.startswith("complex"):
        return _complex_value(dtype, shape, rng, offset=-0.6, scale=1.2)
    return rng.cast((rng.random(shape) - 0.5) * 1.2, dtype)


def generate_nonzero(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成非零值。"""
    dtype = generation_dtype(spec.dtype)
    shape = spec.shape
    if "int" in dtype:
        if dtype == "int8":
            values = rng.randint(1, 256, shape=shape, dtype=numpy.int32)
            values[values > 127] -= 256
            return rng.cast(values, dtype)
        if dtype == "uint8":
            return rng.cast(rng.randint(1, 256, shape=shape), dtype)
        return rng.cast(rng.randint(1, 65535, shape=shape), dtype)
    if dtype.startswith("complex"):
        return _complex_value(dtype, shape, rng, offset=0.5)
    return rng.cast(rng.random(shape) + 0.5, dtype)


def generate_fill_value(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 `paddle.full` 的填充值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.cast(rng.randint(1, 65535, shape=spec.shape), dtype)
    return rng.cast(rng.random(spec.shape) + 0.5, dtype)


def generate_unit_interval(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 [0, 1) 随机值。"""
    return rng.cast(rng.random(spec.shape), generation_dtype(spec.dtype))


def generate_multiply(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 `paddle.multiply` 值。"""
    dtype = generation_dtype(spec.dtype)
    if dtype.startswith("complex"):
        return _complex_value(dtype, spec.shape, rng)
    return rng.cast(rng.random(spec.shape), dtype)


def generate_unit_plus_one(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 [1, 2) 随机值。"""
    return rng.cast(rng.random(spec.shape) + 1.0, generation_dtype(spec.dtype))


def generate_signed_half(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 [-0.5, 0.5) 随机值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.cast(rng.randint(-65535, 65535, shape=spec.shape), dtype)
    return rng.cast(rng.random(spec.shape) - 0.5, dtype)


def generate_normal_std(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 `paddle.normal` 的 std 参数。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.cast(rng.randint(0, 65535, shape=spec.shape), dtype)
    return rng.cast(rng.random(spec.shape), dtype)


def generate_int_1024(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间 [0, 1024)。"""
    return rng.cast(rng.randint(0, 1024, shape=spec.shape), generation_dtype(spec.dtype))


def generate_int_64(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间 [0, 64)。"""
    return rng.cast(rng.randint(0, 64, shape=spec.shape), generation_dtype(spec.dtype))


def generate_int_2048_raw(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间 [0, 2048)，但不做 dtype 强转。"""
    return rng.randint(0, 2048, shape=spec.shape)


def generate_int_128(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间 [1, 128)。"""
    return rng.cast(rng.randint(1, 128, shape=spec.shape), generation_dtype(spec.dtype))


def generate_empty_shape(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 `paddle.empty` 的 shape Tensor 值。"""
    dtype = generation_dtype(spec.dtype) if "int" in spec.dtype else "int32"
    return rng.cast(rng.randint(1, 10, shape=spec.shape), dtype)


def generate_int_2048(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间 [1, 2048)。"""
    return rng.cast(rng.randint(1, 2048, shape=spec.shape), generation_dtype(spec.dtype))


def generate_int_65535_raw(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间 [1, 65535)，但不做 dtype 强转。"""
    return rng.randint(1, 65535, shape=spec.shape)


def generate_ones_shape(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 `paddle.ones` 的 shape Tensor 值。"""
    dtype = generation_dtype(spec.dtype)
    if len(spec.shape) == 0:
        return rng.asarray(rng.randint(1, 2048), dtype=dtype)
    return rng.cast(rng.randint(1, 65535, shape=spec.shape), dtype)


def generate_int_or_unit(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成整数区间或单位区间值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.cast(rng.randint(0, 65535, shape=spec.shape), dtype)
    return rng.cast(rng.random(spec.shape), dtype)


def generate_int_or_default(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 `paddle.dot` 值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.cast(rng.randint(-127, 127, shape=spec.shape), dtype)
    return generate_default(spec, rng)


def generate_binary01(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成二值标签 {0, 1}。"""
    return rng.cast(rng.randint(0, 2, shape=spec.shape), generation_dtype(spec.dtype))


def generate_hinge_label(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 hinge 标签 {-1, 1}。"""
    values = generate_binary01(spec, rng)
    values[values == 0] = -1
    return values


def generate_abs_plus_one(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成上采样 scale_factor 值。"""
    dtype = generation_dtype(spec.dtype)
    return rng.ones(spec.shape, dtype=dtype) + rng.cast(rng.abs(rng.random(spec.shape)), dtype)


def generate_dropout_prob(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 dropout 概率 Tensor 值。"""
    value = generate_uniform(spec, 0, 1.1, rng)
    return rng.where(value > 1, rng.asarray(1, dtype=generation_dtype(spec.dtype)), value)


def generate_quantile(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 quantile q 值。"""
    return rng.cast(rng.random(1), generation_dtype(spec.dtype))


def generate_remainder(spec: TensorSpec, rng=NUMPY_RNG) -> object:
    """生成 remainder 除数值。"""
    dtype = generation_dtype(spec.dtype)
    if dtype in {"int32", "int64"}:
        return generate_uniform(spec, 1, 65535, rng)
    return generate_default(spec, rng)


def generate_random_range(
    spec: TensorSpec,
    low=None,
    high=None,
    rng=NUMPY_RNG,
) -> object:
    """生成指定区间内的随机值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        low = low if low is not None else -65535
        high = high if high is not None else 65535
        return rng.cast(rng.randint(low, high, shape=spec.shape), dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real_low = low if low is not None else numpy.finfo(real_dtype).min / 2
        real_high = high if high is not None else numpy.finfo(real_dtype).max / 2
        return _complex_value(dtype, spec.shape, rng, low=real_low, high=real_high)
    low = low if low is not None else numpy.finfo(dtype).min / 2
    high = high if high is not None else numpy.finfo(dtype).max / 2
    return rng.cast(rng.uniform(low, high, shape=spec.shape), dtype)


def generate_uniform(
    spec: TensorSpec,
    low,
    high,
    rng=NUMPY_RNG,
) -> object:
    """生成固定区间随机值。"""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.cast(rng.randint(low, high, shape=spec.shape), dtype)
    if dtype.startswith("complex"):
        return _complex_value(dtype, spec.shape, rng, low=low, high=high)
    return rng.cast(rng.uniform(low, high, shape=spec.shape), dtype)
