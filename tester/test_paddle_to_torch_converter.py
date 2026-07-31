from __future__ import annotations

import concurrent.futures
import os
import threading
import time
import unittest
from collections import OrderedDict
from dataclasses import FrozenInstanceError
from unittest import mock

import tester.paddle_to_torch.converter as converter_module
import tester.paddle_to_torch.rules as rules_module
import torch
from tester.paddle_to_torch.config import (
    ConversionEnvironment,
    read_conversion_environment,
    select_implementation,
)
from tester.paddle_to_torch.converter import Paddle2TorchConverter
from tester.paddle_to_torch.rules import (
    BaseRule,
    Code,
    ConversionKind,
    ConvertResult,
    GenericRule,
    adaptive_workspace_bytes,
)


class PaddleToTorchConverterTest(unittest.TestCase):
    def test_implementation_environment_is_validated_and_selected_per_rule(self):
        default_environment = read_conversion_environment({})
        self.assertIsNone(default_environment.implementation)
        self.assertEqual(
            select_implementation(
                default_environment,
                supported={"apex", "te", "torch"},
                default="te",
            ),
            "te",
        )
        environment = read_conversion_environment({"PADDLEAPITEST_IMPL": "apex"})
        self.assertEqual(
            select_implementation(
                environment,
                supported={"te", "torch"},
                default="torch",
            ),
            "torch",
        )
        with self.assertRaisesRegex(ValueError, "PADDLEAPITEST_IMPL.*invalid"):
            read_conversion_environment({"PADDLEAPITEST_IMPL": "invalid"})

    def test_all_configured_mappings_generate_valid_code(self):
        converter = Paddle2TorchConverter()

        failures = []
        for paddle_api in converter.mapping:
            result = converter.convert(paddle_api)
            if result.kind is ConversionKind.UNSUPPORTED:
                failures.append(f"{paddle_api}: {result.error_message}")

        self.assertEqual(failures, [])

    def test_unknown_rule_is_rejected_during_initialization(self):
        mapping = OrderedDict(
            {
                "paddle.invalid": OrderedDict(
                    {
                        "Rule": "MissingRule",
                    }
                )
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "paddle.invalid.*MissingRule|MissingRule.*paddle.invalid",
        ):
            Paddle2TorchConverter(mapping_data=mapping)

    def test_log_normal_conversion_executes(self):
        converter = Paddle2TorchConverter()
        convert_result = converter.convert("paddle.log_normal")

        self.assertIs(convert_result.kind, ConversionKind.COMPOSITE)
        result = converter.execute(
            convert_result,
            [],
            OrderedDict(mean=0.0, std=0.0, shape=(2, 3)),
        )

        torch.testing.assert_close(result, torch.ones((2, 3)))

    def test_execute_accepts_extra_locals_and_core_executor(self):
        convert_result = ConvertResult.success(
            "paddle.test",
            Code(
                preprocess=["prepared = seed + 1"],
                core=["core_value = prepared + 1"],
                postprocess=["answer = core_value + 1"],
            ),
            output_var="answer",
        )
        calls = []

        def execute_core(compiled, exec_globals, exec_locals):
            calls.append(compiled)
            exec(compiled, exec_globals, exec_locals)

        result = Paddle2TorchConverter.execute(
            convert_result,
            [],
            OrderedDict(),
            execution_locals={"seed": 1},
            core_executor=execute_core,
        )

        self.assertEqual(result, 4)
        self.assertEqual(len(calls), 1)

    def test_conversion_cache_tracks_implementation_environment(self):
        converter = Paddle2TorchConverter()
        api = "paddle.incubate.nn.functional.fp8_quant_blockwise"

        with mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "torch"}):
            torch_code = converter.convert(api).code.core
        with mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "te"}):
            te_code = converter.convert(api).code.core

        self.assertNotEqual(torch_code, te_code)

    def test_conversion_uses_the_environment_snapshot_for_rule_generation(self):
        converter = Paddle2TorchConverter()
        api = "paddle.incubate.nn.functional.fp8_quant_blockwise"

        with (
            mock.patch.object(
                converter_module,
                "read_conversion_environment",
                return_value=ConversionEnvironment("torch"),
            ),
            mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "te"}),
        ):
            snapshot_code = converter.convert(api).code.core
        with mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "torch"}):
            torch_code = Paddle2TorchConverter().convert(api).code.core

        self.assertEqual(snapshot_code, torch_code)

    def test_fused_linear_uses_shared_implementation_environment(self):
        converter = Paddle2TorchConverter()
        api = "paddle._C_ops.fused_linear_param_grad_add"

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PADDLEAPITEST_IMPL", None)
            default_code = converter.convert(api).code.core
        with mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "torch"}):
            torch_code = converter.convert(api).code.core
        with mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "te"}):
            te_code = converter.convert(api).code.core
        with mock.patch.dict(os.environ, {"PADDLEAPITEST_IMPL": "apex"}):
            apex_code = converter.convert(api).code.core

        self.assertEqual(default_code, te_code)
        self.assertNotEqual(torch_code, apex_code)

    def test_workspace_worker_count_must_be_positive_integer(self):
        class FakeCuda:
            @staticmethod
            def mem_get_info():
                return 8 << 30, 16 << 30

        class FakeTorch:
            cuda = FakeCuda()

        for worker_count in ("0", "-1", "invalid"):
            with self.subTest(worker_count=worker_count):
                with mock.patch.dict(
                    os.environ,
                    {"PADDLEAPITEST_WORKERS_ON_GPU": worker_count},
                ):
                    with self.assertRaisesRegex(ValueError, "PADDLEAPITEST_WORKERS_ON_GPU"):
                        adaptive_workspace_bytes(FakeTorch)

    def test_workspace_cache_tracks_worker_count(self):
        class FakeCuda:
            @staticmethod
            def mem_get_info():
                return 20 << 30, 32 << 30

        class FakeTorch:
            cuda = FakeCuda()

        rules_module._WORKSPACE_PROBE_CACHE.clear()
        self.addCleanup(rules_module._WORKSPACE_PROBE_CACHE.clear)

        with mock.patch.dict(os.environ, {"PADDLEAPITEST_WORKERS_ON_GPU": "1"}):
            single_worker = adaptive_workspace_bytes(FakeTorch)
        with mock.patch.dict(os.environ, {"PADDLEAPITEST_WORKERS_ON_GPU": "2"}):
            two_workers = adaptive_workspace_bytes(FakeTorch)

        self.assertEqual(single_worker, 4 << 30)
        self.assertEqual(two_workers, 2 << 30)

    def test_workspace_cache_isolated_by_device(self):
        class FakeCuda:
            device = 0

            @classmethod
            def current_device(cls):
                return cls.device

            @classmethod
            def mem_get_info(cls):
                free_gib = 20 if cls.device == 0 else 10
                return free_gib << 30, 32 << 30

        class FakeTorch:
            cuda = FakeCuda()

        rules_module._WORKSPACE_PROBE_CACHE.clear()
        self.addCleanup(rules_module._WORKSPACE_PROBE_CACHE.clear)

        FakeCuda.device = 0
        device_zero = adaptive_workspace_bytes(FakeTorch)
        FakeCuda.device = 1
        device_one = adaptive_workspace_bytes(FakeTorch)

        self.assertEqual(device_zero, 4 << 30)
        self.assertEqual(device_one, 2 << 30)

    def test_mapping_schema_reports_api_and_field(self):
        rule_map = {"GenericRule": GenericRule}

        with self.assertRaisesRegex(ValueError, "paddle.invalid.*unknown fields.*unexpected"):
            Paddle2TorchConverter._validate_mapping(
                {"paddle.invalid": {"torch_api": "torch.add", "unexpected": True}},
                rule_map,
            )
        with self.assertRaisesRegex(ValueError, "paddle.invalid.torch_api.*str"):
            Paddle2TorchConverter._validate_mapping(
                {"paddle.invalid": {"torch_api": 1}},
                rule_map,
            )
        with self.assertRaisesRegex(ValueError, "paddle.invalid.torch_api.*required"):
            Paddle2TorchConverter._validate_mapping(
                {"paddle.invalid": {}},
                rule_map,
            )
        with self.assertRaisesRegex(ValueError, "paddle.invalid.*string values"):
            Paddle2TorchConverter._validate_mapping(
                {
                    "paddle.invalid": {
                        "torch_api": "torch.add",
                        "paddle_torch_args_map": {"x": 1},
                    }
                },
                rule_map,
            )
        with self.assertRaisesRegex(ValueError, "paddle.invalid.torch_kwargs.*scalar values"):
            Paddle2TorchConverter._validate_mapping(
                {
                    "paddle.invalid": {
                        "torch_api": "torch.add",
                        "torch_kwargs": {"alpha": [1]},
                    }
                },
                rule_map,
            )
        with self.assertRaisesRegex(ValueError, "Duplicate.*torch_api"):
            Paddle2TorchConverter._reject_duplicate_keys(
                [("torch_api", "torch.add"), ("torch_api", "torch.sub")]
            )

    def test_explicit_generic_rule_initializes_generic_mapping_fields(self):
        mapping = OrderedDict(
            {
                "paddle.test": OrderedDict(
                    {
                        "Rule": "GenericRule",
                        "torch_api": "torch.add",
                        "set_defaults": {},
                        "paddle_torch_args_map": {"x": "input", "y": "other"},
                    }
                )
            }
        )

        converter = Paddle2TorchConverter(mapping_data=mapping)
        result = converter.convert("paddle.test")
        cloned_converter = Paddle2TorchConverter(mapping_data=converter.mapping)

        self.assertIs(result.kind, ConversionKind.DIRECT)
        self.assertEqual(cloned_converter.mapping, converter.mapping)
        with self.assertRaises(TypeError):
            converter.mapping["paddle.other"] = {}
        with self.assertRaises(TypeError):
            converter.mapping["paddle.test"]["torch_api"] = "torch.sub"

    def test_convert_result_enforces_supported_and_error_invariants(self):
        with self.assertRaisesRegex(ValueError, "requires code"):
            ConvertResult("paddle.invalid", kind=ConversionKind.DIRECT)
        with self.assertRaisesRegex(ValueError, "requires an error message"):
            ConvertResult(
                "paddle.invalid",
                kind=ConversionKind.UNSUPPORTED,
            )
        for invalid_kind in ("direct", None, 1):
            with self.subTest(invalid_kind=invalid_kind):
                with self.assertRaisesRegex(TypeError, "Conversion kind"):
                    ConvertResult(
                        "paddle.invalid",
                        kind=invalid_kind,
                        code=Code(core=["result = 1"]),
                    )

        error = ConvertResult.error("paddle.invalid", "unsupported")
        self.assertIs(error.kind, ConversionKind.UNSUPPORTED)
        self.assertIsNone(error.code)
        with self.assertRaises(FrozenInstanceError):
            error.error_message = "changed"

        code = Code(core=["result = 1"])
        self.assertEqual(code.core, ("result = 1",))
        with self.assertRaises(FrozenInstanceError):
            code.core = ("result = 2",)
        with self.assertRaisesRegex(ValueError, "Cannot execute unsupported"):
            Paddle2TorchConverter.execute(error, [], {})

        with self.assertRaisesRegex(RuntimeError, "built-in Rule registry is frozen"):

            class LateRule(BaseRule):
                def apply(self, paddle_api):
                    return ConvertResult.success(paddle_api, Code(core=["result = 1"]))

    def test_conversion_cache_serializes_generation_per_environment(self):
        generation_lock = threading.Lock()

        class CountingRule(BaseRule, register=False):
            generation_count = 0

            def apply(self, paddle_api):
                with generation_lock:
                    type(self).generation_count += 1
                time.sleep(0.01)
                return ConvertResult.success(paddle_api, Code(core=["result = 1"]))

        converter = Paddle2TorchConverter(
            mapping_data={"paddle.concurrent": {"Rule": "CountingRule"}},
            extra_rules={"CountingRule": CountingRule},
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: converter.convert("paddle.concurrent"), range(8)))

        self.assertEqual(CountingRule.generation_count, 1)
        self.assertTrue(all(result is results[0] for result in results))

    def test_conversion_and_execution_errors_include_context(self):
        class ExplodingRule(BaseRule, register=False):
            def apply(self, paddle_api):
                raise LookupError("broken rule")

        class InvalidCodeRule(BaseRule, register=False):
            def apply(self, paddle_api):
                return ConvertResult.success(
                    paddle_api,
                    Code(core=["not valid python !"]),
                )

        converter = Paddle2TorchConverter(
            mapping_data={
                "paddle.invalid": {"Rule": "ExplodingRule"},
                "paddle.invalid_code": {"Rule": "InvalidCodeRule"},
            },
            extra_rules={
                "ExplodingRule": ExplodingRule,
                "InvalidCodeRule": InvalidCodeRule,
            },
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "paddle.invalid.*ExplodingRule.*broken rule",
        ):
            converter.convert("paddle.invalid")

        with self.assertRaisesRegex(
            RuntimeError,
            "paddle.invalid_code.*InvalidCodeRule.*invalid syntax",
        ):
            converter.convert("paddle.invalid_code")

        for stage, code in (
            ("preprocess", Code(preprocess=["raise LookupError('failed')"])),
            ("core", Code(core=["raise LookupError('failed')"])),
            ("postprocess", Code(postprocess=["raise LookupError('failed')"])),
        ):
            with self.subTest(stage=stage):
                result = ConvertResult.success("paddle.invalid", code)
                with self.assertRaisesRegex(RuntimeError, f"paddle.invalid.*{stage}"):
                    Paddle2TorchConverter.execute(result, [], OrderedDict())

        missing_output = ConvertResult.success(
            "paddle.invalid",
            Code(core=["value = 1"]),
            output_var="missing",
        )
        with self.assertRaisesRegex(ValueError, "missing.*paddle.invalid"):
            Paddle2TorchConverter.execute(missing_output, [], OrderedDict())


if __name__ == "__main__":
    unittest.main()
