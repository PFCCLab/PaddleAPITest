from __future__ import annotations

import sys
import types
import unittest


def install_torch_stub_if_needed() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})
        torch_stub.float16 = "torch.float16"
        torch_stub.float32 = "torch.float32"
        torch_stub.float64 = "torch.float64"
        torch_stub.bfloat16 = "torch.bfloat16"
        torch_stub.int8 = "torch.int8"
        torch_stub.int16 = "torch.int16"
        torch_stub.int32 = "torch.int32"
        torch_stub.int64 = "torch.int64"
        torch_stub.uint8 = "torch.uint8"
        torch_stub.bool = "torch.bool"
        torch_stub.complex64 = "torch.complex64"
        torch_stub.complex128 = "torch.complex128"
        torch_stub.float8_e4m3fn = "torch.float8_e4m3fn"
        torch_stub.float8_e5m2 = "torch.float8_e5m2"
        sys.modules["torch"] = torch_stub


install_torch_stub_if_needed()

from tester.api_config.config_analyzer import APIConfig, TensorConfig
from tester.operator_compare.config_loader import case_from_config_line
from tester.operator_compare.implementations import build_compare_suite, expand_implementations
from tester.operator_compare.runner import run_compare_suite


class FlatOperatorCompareTest(unittest.TestCase):
    def test_case_from_config_line_reuses_api_config_parser(self):
        case = case_from_config_line('paddle.Tensor.__abs__(Tensor([], "float32"), )')

        self.assertEqual(case.id, "case_0")
        self.assertEqual(case.metadata["api_name"], "paddle.Tensor.__abs__")
        self.assertEqual(
            case.metadata["raw_config"], 'paddle.Tensor.__abs__(Tensor([], "float32"), )'
        )
        self.assertIsInstance(case.tensors["api_config"], APIConfig)
        self.assertIsInstance(case.tensors["api_config"].args[0], TensorConfig)
        self.assertEqual(case.tensors["api_config"].args[0].shape, [])
        self.assertEqual(case.tensors["api_config"].args[0].dtype, "float32")

    def test_expand_implementations_creates_default_paddle_and_torch_specs(self):
        specs = expand_implementations(
            op_name="paddle.Tensor.__abs__",
            implementation_names=["paddle", "torch"],
            dtypes=["float32", "float64"],
        )

        self.assertEqual(
            [spec.id for spec in specs],
            [
                "paddle|float32|default",
                "torch|float32|default",
                "paddle|float64|default",
                "torch|float64|default",
            ],
        )
        self.assertEqual(
            [spec.group for spec in specs], ["target", "reference", "target", "reference"]
        )
        self.assertTrue(all(spec.runner is not None for spec in specs))

    def test_build_compare_suite_uses_config_lines_and_standard(self):
        suite = build_compare_suite(
            config_lines=['paddle.Tensor.__abs__(Tensor([2], "float32"), )'],
            implementation_names=["paddle", "torch"],
            standard="torch|float32|default",
            dtypes=["float32"],
            enable_fingerprint=False,
        )

        self.assertEqual(suite.op_name, "paddle.Tensor.__abs__")
        self.assertEqual(suite.standard_id, "torch|float32|default")
        self.assertEqual(suite.metrics_dtype, "fp64")
        self.assertFalse(suite.enable_fingerprint)
        self.assertEqual([case.id for case in suite.cases], ["case_0"])
        self.assertEqual(
            [spec.id for spec in suite.implementations],
            ["paddle|float32|default", "torch|float32|default"],
        )

    def test_abs_case_runs_against_torch_reference(self):
        self.assert_case_runs_against_torch_reference(
            'paddle.Tensor.__abs__(Tensor([2], "float32"), )'
        )

    def test_add_case_runs_against_torch_reference(self):
        self.assert_case_runs_against_torch_reference(
            'paddle.add(Tensor([2], "float32"), Tensor([2], "float32"), )'
        )

    def test_fused_linear_param_grad_add_c_ops_runs_against_torch_reference(self):
        self.assert_case_runs_against_torch_reference(
            'paddle._C_ops.fused_linear_param_grad_add(Tensor([8, 4], "float32"), Tensor([8, 4], "float32"), Tensor([4, 4], "float32"), None, False, False, )',
            max_abs=1e-5,
        )

    def assert_case_runs_against_torch_reference(
        self, config_line: str, max_abs: float = 0
    ) -> None:
        suite = build_compare_suite(
            config_lines=[config_line],
            implementation_names=["paddle", "torch"],
            standard="torch|config|default",
            enable_fingerprint=False,
        )

        run_data = run_compare_suite(suite)

        self.assertEqual([result.status for result in run_data["results"]], ["ok", "ok"])
        paddle_result = next(
            result for result in run_data["results"] if result.spec.id == "paddle|config|default"
        )
        self.assertIsNotNone(paddle_result.metrics_vs_standard)
        self.assertLessEqual(paddle_result.metrics_vs_standard.max_abs, max_abs)


if __name__ == "__main__":
    unittest.main()
