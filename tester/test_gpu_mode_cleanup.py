from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest import mock

from tester import base, runtime_config
from tester.api_config.log_writer import print_run_header
from tester.base import APITestBase, gpu_mode_maybe_empty_cache
from tester.runtime_config import (
    GPU_MEMORY_POLICY_ENV,
    GpuModeConfig,
    TestRuntimeConfig,
    resolve_gpu_memory_policy,
)


class GpuModeCleanupTest(unittest.TestCase):
    def test_gpu_memory_policy_defaults_to_conservative(self):
        with mock.patch.dict(runtime_config.os.environ, {}, clear=True):
            self.assertEqual(resolve_gpu_memory_policy(), "conservative")

    def test_gpu_memory_policy_normalizes_environment_value(self):
        with mock.patch.dict(
            runtime_config.os.environ,
            {GPU_MEMORY_POLICY_ENV: " AGGRESSIVE "},
            clear=True,
        ):
            self.assertEqual(resolve_gpu_memory_policy(), "aggressive")

    def test_gpu_memory_policy_rejects_invalid_value(self):
        with self.assertRaisesRegex(ValueError, GPU_MEMORY_POLICY_ENV):
            resolve_gpu_memory_policy("balanced")

    def test_runtime_config_captures_resolved_policy(self):
        options = SimpleNamespace(
            use_gpu_mode=True,
            gpu_memory_policy="aggressive",
            random_seed=0,
            bitwise_alignment=False,
            exit_on_error=False,
        )

        config = TestRuntimeConfig.from_options(options)

        self.assertEqual(config.gpu_mode.memory_policy, "aggressive")

    def test_run_header_reports_gpu_memory_policy(self):
        options = SimpleNamespace(
            accuracy=False,
            paddle_only=False,
            paddle_cinn=False,
            paddle_gpu_performance=False,
            torch_gpu_performance=False,
            paddle_torch_gpu_performance=False,
            accuracy_stable=True,
            paddle_custom_device=False,
            custom_device_vs_gpu=False,
            api_config_file="configs.txt",
            api_config_file_pattern="",
            log_dir="logs",
            test_cpu=False,
            gpu_ids="0",
            use_gpu_mode=True,
            use_cached_numpy=False,
            gpu_memory_policy="aggressive",
            atol=0.0,
            rtol=0.0,
            timeout=60,
            show_runtime_status=False,
        )
        output = io.StringIO()

        with mock.patch("sys.stdout", output):
            print_run_header(options, "test-version")

        self.assertIn("GPU memory aggressive", output.getvalue())

    def test_run_header_defaults_policy_for_legacy_engine_options(self):
        options = SimpleNamespace(
            accuracy_stable=True,
            api_config_file="configs.txt",
            api_config_file_pattern="",
            log_dir="logs",
            test_cpu=False,
            gpu_ids="0",
            use_gpu_mode=True,
            use_cached_numpy=False,
            atol=0.0,
            rtol=0.0,
            timeout=60,
            show_runtime_status=False,
        )
        output = io.StringIO()

        with mock.patch("sys.stdout", output):
            print_run_header(options, "test-version")

        self.assertIn("GPU memory conservative", output.getvalue())

    def test_gen_torch_input_does_not_run_gpu_mode_cleanup(self):
        case = object.__new__(APITestBase)
        case.gpu_mode_config = GpuModeConfig(enabled=True)
        case.torch_args_config = []
        case.torch_kwargs_config = {}
        case.api_config = SimpleNamespace(api_name="paddle.max")

        with (
            mock.patch.object(base, "gpu_mode_maybe_empty_cache") as cleanup,
            mock.patch.object(base.torch.cuda, "empty_cache") as torch_empty_cache,
        ):
            generated = case.gen_torch_input()

        self.assertTrue(generated)
        cleanup.assert_not_called()
        torch_empty_cache.assert_not_called()

    def test_clear_runtime_inputs_releases_gpu_mode_references(self):
        case = object.__new__(APITestBase)
        case.gpu_mode_config = GpuModeConfig(enabled=True)
        case.torch_args = [object()]
        case.torch_kwargs = {"x": object()}

        with (
            mock.patch.object(base.gc, "collect") as collect,
            mock.patch.object(base.torch.cuda, "empty_cache") as torch_empty_cache,
            mock.patch.object(base.paddle.device.cuda, "empty_cache") as paddle_empty_cache,
        ):
            case.clear_runtime_inputs("torch")

        self.assertFalse(hasattr(case, "torch_args"))
        self.assertFalse(hasattr(case, "torch_kwargs"))
        collect.assert_not_called()
        torch_empty_cache.assert_not_called()
        paddle_empty_cache.assert_not_called()

    def test_gpu_mode_cleanup_skips_cache_release_without_pressure(self):
        with (
            mock.patch.object(base.paddle.device, "get_device", return_value="gpu:0"),
            mock.patch.object(base.torch.cuda, "memory_reserved", return_value=2 * 1024**3),
            mock.patch.object(base.torch.cuda, "memory_allocated", return_value=int(1.5 * 1024**3)),
            mock.patch.object(base.gc, "collect") as collect,
            mock.patch.object(base.torch.cuda, "empty_cache") as torch_empty_cache,
            mock.patch.object(base.paddle.device.cuda, "empty_cache") as paddle_empty_cache,
        ):
            cleaned = gpu_mode_maybe_empty_cache(
                GpuModeConfig(enabled=True, memory_budget=100.0),
                "accuracy_stable_after_first_compare_spill",
            )

        self.assertFalse(cleaned)
        collect.assert_not_called()
        torch_empty_cache.assert_not_called()
        paddle_empty_cache.assert_not_called()

    def test_gpu_mode_cleanup_releases_caches_under_pressure(self):
        calls = []
        with (
            mock.patch.object(base.paddle.device, "get_device", return_value="gpu:0"),
            mock.patch.object(base.torch.cuda, "memory_reserved", return_value=95 * 1024**3),
            mock.patch.object(base.torch.cuda, "memory_allocated", return_value=90 * 1024**3),
            mock.patch.object(base.gc, "collect", side_effect=lambda: calls.append("gc")),
            mock.patch.object(
                base.torch.cuda, "empty_cache", side_effect=lambda: calls.append("torch")
            ),
            mock.patch.object(
                base.paddle.device.cuda,
                "empty_cache",
                side_effect=lambda: calls.append("paddle"),
            ),
        ):
            cleaned = gpu_mode_maybe_empty_cache(
                GpuModeConfig(enabled=True, memory_budget=100.0),
                "accuracy_stable_after_first_compare_spill",
            )

        self.assertTrue(cleaned)
        self.assertEqual(calls, ["gc", "torch", "paddle"])

    def test_snapshot_helpers_do_not_collect_or_clear_caches(self):
        case = object.__new__(APITestBase)
        case.api_config = SimpleNamespace(config="paddle.max(...)")
        events = []
        configs = [
            SimpleNamespace(
                save_original_tensor_to_cpu=lambda api_config: events.append("save"),
                clear_original_cpu_tensor=lambda: events.append("clear"),
            )
        ]
        case._for_each_tensor_config = lambda callback: [callback(config) for config in configs]

        with (
            mock.patch.object(base.gc, "collect") as collect,
            mock.patch.object(base, "gpu_mode_maybe_empty_cache") as cleanup,
        ):
            case.save_original_inputs_to_cpu()
            case.clear_original_cpu_inputs()

        self.assertEqual(events, ["save", "clear"])
        collect.assert_not_called()
        cleanup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
