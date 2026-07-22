from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from tester.accuracy_stable import APITestAccuracyStable
from tester.runtime_config import GpuModeConfig, TestRuntimeConfig


class AccuracyStableMemoryPolicyTest(unittest.TestCase):
    def _make_case(self, policy):
        api_config = SimpleNamespace(config="paddle.max(...)")
        runtime_config = TestRuntimeConfig(
            gpu_mode=GpuModeConfig(enabled=True, memory_policy=policy)
        )

        def init_base(instance, _api_config, **_kwargs):
            instance.gpu_mode_config = runtime_config.gpu_mode

        with (
            mock.patch("tester.accuracy_stable.APITestBase.__init__", new=init_base),
            mock.patch("tester.accuracy_stable.get_converter"),
            mock.patch("tester.accuracy_stable.torch.set_default_device"),
        ):
            case = APITestAccuracyStable(api_config, runtime_config=runtime_config)
        return case

    def test_aggressive_policy_keeps_first_results_on_device(self):
        case = self._make_case("aggressive")
        self.assertFalse(case.should_spill_first_results())

    def test_conservative_policy_spills_first_results(self):
        case = self._make_case("conservative")
        self.assertTrue(case.should_spill_first_results())


if __name__ == "__main__":
    unittest.main()
