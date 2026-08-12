# tester/__init__.py

from typing import TYPE_CHECKING, Any

__all__ = [
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
    "paddle_to_torch",
    "prepare_process_runtime",
]

if TYPE_CHECKING:
    from . import paddle_to_torch
    from .accuracy import APITestAccuracy
    from .accuracy_stable import APITestAccuracyStable
    from .api_config import APIConfig, TensorConfig, analyse_configs
    from .base import APITestBase
    from .paddle_cinn_vs_dygraph import APITestCINNVSDygraph
    from .paddle_device_vs_cpu import APITestCustomDeviceVSCPU
    from .paddle_device_vs_gpu import APITestPaddleDeviceVSGPU
    from .paddle_gpu_performance import APITestPaddleGPUPerformance
    from .paddle_only import APITestPaddleOnly
    from .paddle_torch_gpu_performance import APITestPaddleTorchGPUPerformance
    from .torch_gpu_performance import APITestTorchGPUPerformance


def prepare_process_runtime(options):
    """按 engine 已冻结的配置准备当前进程使用的输入 backend。"""
    runtime_config = getattr(options, "runtime_config", None)
    if runtime_config is None:
        raise ValueError("runtime_config must be frozen before process runtime preparation")

    from .input_generation.backend_runtime import prepare_input_backend

    return prepare_input_backend(runtime_config.input_backend_policy)


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
