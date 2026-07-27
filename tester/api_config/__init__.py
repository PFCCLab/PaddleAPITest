from typing import TYPE_CHECKING, Any

__all__ = [
    "USE_CACHED_NUMPY",
    "APIConfig",
    "TensorConfig",
    "analyse_configs",
    "cached_numpy",
]

if TYPE_CHECKING:
    from .input_generation.tensor_config import (
        USE_CACHED_NUMPY,
        TensorConfig,
        cached_numpy,
    )
    from .parser import APIConfig, analyse_configs


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "TensorConfig":
        from .input_generation.tensor_config import TensorConfig

        return TensorConfig
    elif name == "APIConfig":
        from .parser import APIConfig

        return APIConfig
    elif name == "analyse_configs":
        from .parser import analyse_configs

        return analyse_configs
    elif name == "USE_CACHED_NUMPY":
        from .input_generation.tensor_config import USE_CACHED_NUMPY

        return USE_CACHED_NUMPY
    elif name == "cached_numpy":
        from .input_generation.tensor_config import cached_numpy

        return cached_numpy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
