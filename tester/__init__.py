# tester/__init__.py

from typing import TYPE_CHECKING, Any

__all__ = [
    "USE_CACHED_NUMPY",
    "APIConfig",
    "APITestAccuracy",
    "APITestAccuracyStable",
    "APITestBase",
    "APITestCINNVSDygraph",
    "APITestCustomDeviceVSCPU",
    "APITestPaddleDeviceVSGPU",
    "APITestPaddleGPUPerformance",
    "APITestPaddleOnly",
    "APITestPaddleTorchGPUPerformance",
    "APITestTorchGPUPerformance",
    "TensorConfig",
    "analyse_configs",
    "cached_numpy",
    "paddle_to_torch",
]

if TYPE_CHECKING:
    USE_CACHED_NUMPY: bool

    from . import paddle_to_torch
    from .accuracy import APITestAccuracy
    from .accuracy_stable import APITestAccuracyStable
    from .api_config import (
        APIConfig,
        TensorConfig,
        analyse_configs,
        cached_numpy,
    )
    from .base import APITestBase
    from .paddle_cinn_vs_dygraph import APITestCINNVSDygraph
    from .paddle_device_vs_cpu import APITestCustomDeviceVSCPU
    from .paddle_device_vs_gpu import APITestPaddleDeviceVSGPU
    from .paddle_gpu_performance import APITestPaddleGPUPerformance
    from .paddle_only import APITestPaddleOnly
    from .paddle_torch_gpu_performance import APITestPaddleTorchGPUPerformance
    from .torch_gpu_performance import APITestTorchGPUPerformance


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "APITestBase":
        from .base import APITestBase

        return APITestBase
    elif name == "APITestAccuracy":
        from .accuracy import APITestAccuracy

        return APITestAccuracy
    elif name == "APITestPaddleOnly":
        from .paddle_only import APITestPaddleOnly

        return APITestPaddleOnly
    elif name == "APITestCINNVSDygraph":
        from .paddle_cinn_vs_dygraph import APITestCINNVSDygraph

        return APITestCINNVSDygraph
    elif name == "APITestPaddleGPUPerformance":
        from .paddle_gpu_performance import APITestPaddleGPUPerformance

        return APITestPaddleGPUPerformance
    elif name == "APITestTorchGPUPerformance":
        from .torch_gpu_performance import APITestTorchGPUPerformance

        return APITestTorchGPUPerformance
    elif name == "APITestPaddleTorchGPUPerformance":
        from .paddle_torch_gpu_performance import APITestPaddleTorchGPUPerformance

        return APITestPaddleTorchGPUPerformance
    elif name == "APITestAccuracyStable":
        from .accuracy_stable import APITestAccuracyStable

        return APITestAccuracyStable
    elif name == "APITestCustomDeviceVSCPU":
        from .paddle_device_vs_cpu import APITestCustomDeviceVSCPU

        return APITestCustomDeviceVSCPU
    elif name == "APITestPaddleDeviceVSGPU":
        from .paddle_device_vs_gpu import APITestPaddleDeviceVSGPU

        return APITestPaddleDeviceVSGPU
    elif name == "paddle_to_torch":
        from . import paddle_to_torch

        return paddle_to_torch
    elif name == "TensorConfig":
        from .api_config import TensorConfig

        return TensorConfig
    elif name == "APIConfig":
        from .api_config import APIConfig

        return APIConfig
    elif name == "analyse_configs":
        from .api_config import analyse_configs

        return analyse_configs
    elif name == "USE_CACHED_NUMPY":
        from .runtime_config import numpy_cache_enabled

        return numpy_cache_enabled()
    elif name == "cached_numpy":
        from .api_config import cached_numpy

        return cached_numpy

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
