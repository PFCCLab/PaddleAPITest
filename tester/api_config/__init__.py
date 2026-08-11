from typing import TYPE_CHECKING, Any

__all__ = [
    "APIConfig",
    "TensorConfig",
    "analyse_configs",
]

if TYPE_CHECKING:
    from ..input_generation.tensor_config import TensorConfig
    from .parser import APIConfig, analyse_configs


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "TensorConfig":
        from ..input_generation.tensor_config import TensorConfig

        return TensorConfig
    elif name == "APIConfig":
        from .parser import APIConfig

        return APIConfig
    elif name == "analyse_configs":
        from .parser import analyse_configs

        return analyse_configs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
