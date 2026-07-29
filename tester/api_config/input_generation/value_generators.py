"""Pure NumPy value generators used by decorator-registered input rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from .model import TensorSpec

_INTERMEDIATE_DTYPES = {
    "bfloat16": "float32",
    "float8_e4m3fn": "float16",
    "float8_e5m2": "float16",
}


class LegacyNumpyRNG:
    """Compatibility adapter over the legacy global NumPy RNG."""

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


LEGACY_NUMPY_RNG = LegacyNumpyRNG()


@dataclass(frozen=True)
class CaseNumpyRNG(LegacyNumpyRNG):
    """Per-case RNG facade that currently preserves legacy global RNG behavior."""

    seed: int
    config_fingerprint: str
    runtime_mode: str
    backend: str = "legacy-global"


def create_case_rng(context) -> CaseNumpyRNG:
    return CaseNumpyRNG(
        seed=context.seed,
        config_fingerprint=context.config_fingerprint,
        runtime_mode=context.runtime_mode,
    )


def generation_dtype(dtype: str) -> str:
    return _INTERMEDIATE_DTYPES.get(dtype, dtype)


def generate_default(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Match the legacy fallback generation branch."""
    dtype = generation_dtype(spec.dtype)
    shape = spec.shape
    if not shape:
        if "int" in dtype:
            return numpy.array(rng.randint(-65535, 65535), dtype=dtype)
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


def generate_nonzero(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Match the legacy non-zero generation branch."""
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


def generate_full_fill_value(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy paddle.full fill_value values."""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(1, 65535, size=spec.shape).astype(dtype)
    return (rng.random(spec.shape) + 0.5).astype(dtype)


def generate_unit_interval(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy [0, 1) random value."""
    return rng.random(spec.shape).astype(generation_dtype(spec.dtype))


def generate_multiply(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy paddle.multiply value."""
    dtype = generation_dtype(spec.dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = rng.random(spec.shape).astype(real_dtype)
        imag = rng.random(spec.shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return rng.random(spec.shape).astype(dtype)


def generate_unit_interval_plus_one(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy random value in [1, 2)."""
    return (rng.random(spec.shape) + 1.0).astype(generation_dtype(spec.dtype))


def generate_signed_half_interval(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy random value in [-0.5, 0.5)."""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(-65535, 65535, size=spec.shape).astype(dtype)
    return (rng.random(spec.shape) - 0.5).astype(dtype)


def generate_normal_std(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy paddle.normal std argument."""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(0, 65535, size=spec.shape).astype(dtype)
    return rng.random(spec.shape).astype(dtype)


def generate_int_zero_1024(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [0, 1024)."""
    return rng.randint(0, 1024, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_zero_64(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [0, 64)."""
    return rng.randint(0, 64, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_zero_2048_no_cast(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [0, 2048) without dtype cast."""
    return rng.randint(0, 2048, size=spec.shape)


def generate_int_one_128(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [1, 128)."""
    return rng.randint(1, 128, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_one_10(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [1, 10)."""
    return rng.randint(1, 10, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_empty_shape(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy paddle.empty shape tensor values."""
    dtype = generation_dtype(spec.dtype) if "int" in spec.dtype else "int32"
    return rng.randint(1, 10, size=spec.shape).astype(dtype)


def generate_int_one_2048(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [1, 2048)."""
    return rng.randint(1, 2048, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_one_65535(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [1, 65535)."""
    return rng.randint(1, 65535, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_int_one_65535_no_cast(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy integer range [1, 65535) without dtype cast."""
    return rng.randint(1, 65535, size=spec.shape)


def generate_ones_shape(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate the legacy paddle.ones shape tensor values."""
    dtype = generation_dtype(spec.dtype)
    if len(spec.shape) == 0:
        return numpy.array(rng.randint(1, 2048), dtype=dtype)
    return rng.randint(1, 65535, size=spec.shape).astype(dtype)


def generate_int_zero_65535_else_unit(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy linspace/gammainc-style values."""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(0, 65535, size=spec.shape).astype(dtype)
    return rng.random(spec.shape).astype(dtype)


def generate_int_minus127_127_else_default(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy paddle.dot values."""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(-127, 127, size=spec.shape).astype(dtype)
    return generate_default(spec, rng)


def generate_binary_0_1(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy binary labels in {0, 1}."""
    return rng.randint(0, 2, size=spec.shape).astype(generation_dtype(spec.dtype))


def generate_hinge_labels(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy hinge labels in {-1, 1}."""
    values = generate_binary_0_1(spec, rng)
    values[values == 0] = -1
    return values


def generate_abs_unit_plus_one(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy upsample scale_factor values."""
    dtype = generation_dtype(spec.dtype)
    return numpy.ones(spec.shape).astype(dtype) + numpy.abs(rng.random(spec.shape)).astype(dtype)


def generate_dropout_probability(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy dropout probability tensor values."""
    value = generate_uniform(spec, 0, 1.1, rng)
    return numpy.where(value > 1, 1, value)


def generate_quantile_q(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy quantile q values."""
    return rng.random(1).astype(generation_dtype(spec.dtype))


def generate_remainder_rhs(spec: TensorSpec, rng=LEGACY_NUMPY_RNG) -> numpy.ndarray:
    """Generate legacy remainder divisor values."""
    dtype = generation_dtype(spec.dtype)
    if dtype in {"int32", "int64"}:
        return generate_uniform(spec, 1, 65535, rng)
    return generate_default(spec, rng)


def generate_legacy_random_range(
    spec: TensorSpec,
    low=None,
    high=None,
    rng=LEGACY_NUMPY_RNG,
) -> numpy.ndarray:
    """Match TensorConfig.get_random_numpy_tensor for migrated rule call sites."""
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
    rng=LEGACY_NUMPY_RNG,
) -> numpy.ndarray:
    """Match the legacy fixed-range random helper."""
    dtype = generation_dtype(spec.dtype)
    if "int" in dtype:
        return rng.randint(low, high, size=spec.shape).astype(dtype)
    if dtype.startswith("complex"):
        real_dtype = "float32" if dtype == "complex64" else "float64"
        real = rng.uniform(low, high, size=spec.shape).astype(real_dtype)
        imag = rng.uniform(low, high, size=spec.shape).astype(real_dtype)
        return (real + 1j * imag).astype(dtype)
    return rng.uniform(low, high, size=spec.shape).astype(dtype)
