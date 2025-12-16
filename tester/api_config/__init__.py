from typing import TYPE_CHECKING, Any

__all__ = [
    "USE_CACHED_NUMPY",
    "APIConfig",
    "TensorConfig",
    "analyse_configs",
    "cached_numpy",
    "get_cfg",
    "set_cfg",
]

if TYPE_CHECKING:
    from .config_analyzer import (
        USE_CACHED_NUMPY,
        APIConfig,
        TensorConfig,
        analyse_configs,
        cached_numpy,
    )
    from .log_writer import get_cfg, set_cfg


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "TensorConfig":
        from .config_analyzer import TensorConfig

        return TensorConfig
    elif name == "APIConfig":
        from .config_analyzer import APIConfig

        return APIConfig
    elif name == "analyse_configs":
        from .config_analyzer import analyse_configs

        return analyse_configs
    elif name == "USE_CACHED_NUMPY":
        from .config_analyzer import USE_CACHED_NUMPY

        return USE_CACHED_NUMPY
    elif name == "cached_numpy":
        from .config_analyzer import cached_numpy

        return cached_numpy
    elif name == "get_cfg":
        from .log_writer import get_cfg

        return get_cfg
    elif name == "set_cfg":
        from .log_writer import set_cfg

        return set_cfg

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
