from __future__ import annotations

import os
from collections.abc import Mapping, Set
from dataclasses import dataclass

IMPLEMENTATION_ENV_VAR = "PADDLEAPITEST_IMPL"
WORKERS_ON_GPU_ENV_VAR = "PADDLEAPITEST_WORKERS_ON_GPU"
VALID_IMPLEMENTATIONS = frozenset({"apex", "te", "torch"})


@dataclass(frozen=True)
class ConversionEnvironment:
    implementation: str | None


def read_conversion_environment(
    environ: Mapping[str, str] | None = None,
) -> ConversionEnvironment:
    source = os.environ if environ is None else environ
    implementation = source.get(IMPLEMENTATION_ENV_VAR)
    if implementation is not None and implementation not in VALID_IMPLEMENTATIONS:
        expected = ", ".join(sorted(VALID_IMPLEMENTATIONS))
        raise ValueError(
            f"{IMPLEMENTATION_ENV_VAR} must be one of {expected}, got {implementation!r}"
        )
    return ConversionEnvironment(implementation=implementation)


def select_implementation(
    environment: ConversionEnvironment,
    *,
    supported: Set[str],
    default: str,
) -> str:
    if default not in supported:
        raise ValueError(f"default implementation {default!r} is not supported")
    if not supported <= VALID_IMPLEMENTATIONS:
        invalid = ", ".join(sorted(supported - VALID_IMPLEMENTATIONS))
        raise ValueError(f"unknown supported implementations: {invalid}")
    if environment.implementation is not None and environment.implementation in supported:
        return environment.implementation
    return default


def read_workers_on_gpu(environ: Mapping[str, str] | None = None) -> int:
    source = os.environ if environ is None else environ
    raw_workers_on_gpu = source.get(WORKERS_ON_GPU_ENV_VAR, "1")
    try:
        workers_on_gpu = int(raw_workers_on_gpu)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{WORKERS_ON_GPU_ENV_VAR} must be a positive integer, got {raw_workers_on_gpu!r}"
        ) from exc
    if workers_on_gpu < 1:
        raise ValueError(
            f"{WORKERS_ON_GPU_ENV_VAR} must be a positive integer, got {raw_workers_on_gpu!r}"
        )
    return workers_on_gpu
