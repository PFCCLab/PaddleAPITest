from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

import numpy
from tester.input_generation.value_generators import generate_default_input_value
from tester.input_generation.values import InputTensorSpec
from tester.runtime_config import DEFAULT_INPUT_MAX_ABS, resolve_input_max_abs


def _spec(dtype):
    return InputTensorSpec((4096,), dtype, None, True, None)


class InputRangeTest(TestCase):
    def test_environment_value_is_validated(self):
        self.assertEqual(resolve_input_max_abs({}), DEFAULT_INPUT_MAX_ABS)
        self.assertEqual(resolve_input_max_abs({"PADDLEAPITEST_INPUT_MAX_ABS": "10"}), 10)
        for invalid in ("0", "-1", "nan", "inf", "bad"):
            with self.assertRaises(ValueError):
                resolve_input_max_abs({"PADDLEAPITEST_INPUT_MAX_ABS": invalid})

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
