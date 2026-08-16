from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

import numpy
from tester.api_config import APIConfig
from tester.input_generation.backend_runtime import (
    clear_input_backend_runtime,
    generate_output_grad,
    resolve_input_backend_policy,
)
from tester.input_generation.dispatcher import dispatch_input_generation
from tester.input_generation.value_generators import (
    generate_default_input_value,
    generate_normal_input_value,
    generate_random_range_input_value,
)
from tester.input_generation.values import InputTensorSpec
from tester.runtime_config import (
    DEFAULT_INPUT_MAX_ABS,
    DEFAULT_OUTPUT_GRAD_MAX_ABS,
    resolve_input_max_abs,
    resolve_output_grad_max_abs,
)


def _spec(dtype):
    return InputTensorSpec((4096,), dtype, None, True, None)


def _generated_values(config, max_abs=10):
    # dispatcher 级检查覆盖参数绑定和规则选择，而不只验证底层函数。
    api_config = APIConfig(config)
    runtime_config = SimpleNamespace(
        random_seed=1,
        input_backend_policy=resolve_input_backend_policy(
            requested="numpy",
            use_gpu_mode=False,
            use_cached_numpy=False,
            mode="paddle_only",
        ),
        input_max_abs=max_abs,
    )
    dispatch_input_generation(SimpleNamespace(api_config=api_config, runtime_config=runtime_config))
    return tuple(value.generated_value for value in api_config._input_generation_values)


class InputRangeTest(TestCase):
    def test_environment_value_is_validated(self):
        self.assertEqual(resolve_input_max_abs({}), DEFAULT_INPUT_MAX_ABS)
        self.assertEqual(resolve_input_max_abs({"PADDLEAPITEST_INPUT_MAX_ABS": "10"}), 10)
        for invalid in ("0", "-1", "nan", "inf", "bad"):
            with self.assertRaises(ValueError):
                resolve_input_max_abs({"PADDLEAPITEST_INPUT_MAX_ABS": invalid})
        self.assertEqual(resolve_output_grad_max_abs({}), DEFAULT_OUTPUT_GRAD_MAX_ABS)
        self.assertEqual(
            resolve_output_grad_max_abs({"PADDLEAPITEST_INPUT_MAX_ABS": "10"}),
            10,
        )

    def test_default_bound_preserves_historical_values(self):
        # 默认配置必须保持相同 seed 下的历史输入，避免普通回归基线漂移。
        numpy.random.seed(7)
        expected = ((numpy.random.random(4096) - 0.5) * 1.2).astype("float32")
        numpy.random.seed(7)
        actual = generate_default_input_value(_spec("float32"))
        numpy.testing.assert_array_equal(actual, expected)

    def test_default_float_range_can_expand_without_changing_integers(self):
        # 扩大后的样本既要落在新范围内，也必须实际越过历史默认上界。
        numpy.random.seed(0)
        values = generate_default_input_value(_spec("float32"), max_abs=10)
        self.assertTrue(numpy.all(values >= -10))
        self.assertTrue(numpy.all(values < 10))
        self.assertTrue(numpy.any(numpy.abs(values) > DEFAULT_INPUT_MAX_ABS))

        # 固定 RNG 输出用于证明 max_abs 不参与整数上下界计算。
        integer_rng = SimpleNamespace(
            randint=lambda low, high, shape=None: numpy.full(shape, high - 1),
            cast=lambda value, dtype: numpy.asarray(value).astype(dtype),
        )
        integers = generate_default_input_value(_spec("int64"), integer_rng, max_abs=10)
        self.assertTrue(numpy.all(integers == 65534))

    def test_bool_and_complex_paths_cover_their_full_value_domains(self):
        numpy.random.seed(3)
        bool_values = generate_default_input_value(_spec("bool"))
        self.assertTrue(numpy.any(bool_values))
        self.assertTrue(numpy.any(~bool_values))

        complex_values = generate_normal_input_value(_spec("complex64"))
        self.assertTrue(numpy.any(complex_values.real != 0))
        self.assertTrue(numpy.any(complex_values.imag != 0))

    def test_float64_random_range_stays_finite(self):
        numpy.random.seed(5)
        values = generate_random_range_input_value(_spec("float64"))
        self.assertTrue(numpy.all(numpy.isfinite(values)))
        scalar = generate_random_range_input_value(InputTensorSpec((), "float64", None, True, None))
        self.assertTrue(numpy.isfinite(scalar))

    def test_output_grad_uses_configured_range_and_separate_cache_entries(self):
        # 相同 stream 的不同范围必须得到各自的 cached output-grad。
        def generate(max_abs):
            return generate_output_grad(
                dtype="complex64",
                shape=(4096,),
                backend_name="numpy",
                device="cpu",
                seed=11,
                config_fingerprint="output-grad-range",
                max_abs=max_abs,
                range_configured=max_abs != DEFAULT_OUTPUT_GRAD_MAX_ABS,
                cache_enabled=True,
            )

        clear_input_backend_runtime()
        narrow = generate(DEFAULT_OUTPUT_GRAD_MAX_ABS)
        wide = generate(10)
        self.assertTrue(numpy.all(numpy.abs(narrow.real) < DEFAULT_OUTPUT_GRAD_MAX_ABS))
        self.assertTrue(numpy.all(numpy.abs(narrow.imag) < DEFAULT_OUTPUT_GRAD_MAX_ABS))
        self.assertTrue(numpy.any(numpy.abs(wide.real) > DEFAULT_OUTPUT_GRAD_MAX_ABS))
        self.assertTrue(numpy.any(numpy.abs(wide.imag) > DEFAULT_OUTPUT_GRAD_MAX_ABS))

    def test_default_equivalent_rules_use_configured_complex_values(self):
        # shape-only 和参数规则都必须回到 complex-aware default。
        configs = (
            'paddle.full(shape=Tensor([2],"int64"), '
            'fill_value=Tensor([1],"complex64"), dtype="complex64", )',
            'paddle.normal(Tensor([128],"complex64"), Tensor([128],"float32"), )',
            'paddle.matrix_transpose(Tensor([4],"complex64"), )',
            'paddle.dot(Tensor([128],"complex64"), Tensor([128],"complex64"), )',
        )
        for config in configs:
            values = _generated_values(config)
            complex_values = [value for value in values if numpy.iscomplexobj(value)]
            self.assertTrue(complex_values, config)
            self.assertTrue(any(numpy.any(value.imag != 0) for value in complex_values), config)
            self.assertTrue(
                any(
                    numpy.any(numpy.abs(value.real) > DEFAULT_INPUT_MAX_ABS)
                    for value in complex_values
                ),
                config,
            )

    def test_complex_cholesky_input_is_hermitian(self):
        # 验证生成协议，不依赖 Paddle 是否注册 complex Cholesky kernel。
        (value,) = _generated_values('paddle.linalg.cholesky(Tensor([2,3,3],"complex64"), )')
        self.assertTrue(numpy.any(value.imag != 0))
        numpy.testing.assert_allclose(value, value.swapaxes(-1, -2).conj(), rtol=1e-6, atol=1e-6)

    def test_remainder_rhs_is_nonzero_and_uses_configured_range(self):
        # 扩大范围不能重新引入 remainder 除零输入。
        _, rhs = _generated_values(
            'paddle.remainder(Tensor([4096],"float32"), Tensor([4096],"float32"), )'
        )
        self.assertTrue(numpy.all(rhs != 0))
        self.assertTrue(numpy.any(numpy.abs(rhs) > DEFAULT_INPUT_MAX_ABS))
