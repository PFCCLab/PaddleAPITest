from __future__ import annotations

import copy
import inspect
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any

import paddle
import torch
from tester.api_config.config_analyzer import APIConfig, TensorConfig
from tester.paddle_to_torch.converter import Paddle2TorchConverter, get_converter

from .config_loader import cases_from_config_lines
from .spec import CompareCase, CompareSuite, ImplementationSpec

CustomRunner = Callable[[CompareCase, ImplementationSpec], torch.Tensor]

CUSTOM_IMPLEMENTATIONS: dict[str, CustomRunner] = {}
MANUAL_ARGUMENT_NAMES: dict[str, tuple[str, ...]] = {
    "paddle._C_ops.fused_linear_param_grad_add": (
        "x",
        "dout",
        "dweight",
        "dbias",
        "multi_precision",
        "has_bias",
    ),
}
SIGNATURE_ARGUMENT_NAMES: dict[str, tuple[str, ...]] = {}


def register_custom_implementation(name: str, runner: CustomRunner) -> None:
    if name in CUSTOM_IMPLEMENTATIONS:
        raise ValueError(f"custom implementation already registered: {name}")
    CUSTOM_IMPLEMENTATIONS[name] = runner


def implementation_id(name: str, dtype: str | None, precision: str) -> str:
    dtype_part = dtype or "config"
    return f"{name}|{dtype_part}|{precision}"


def expand_implementations(
    *,
    op_name: str,
    implementation_names: Iterable[str],
    dtypes: Iterable[str | None] | None = None,
    precisions: Iterable[str] | None = None,
) -> list[ImplementationSpec]:
    dtype_values = list(dtypes or [None])
    precision_values = list(precisions or ["default"])
    specs: list[ImplementationSpec] = []
    for dtype in dtype_values:
        for precision in precision_values:
            for name in implementation_names:
                kind = implementation_kind(name)
                specs.append(
                    ImplementationSpec(
                        id=implementation_id(name, dtype, precision),
                        display_name=name,
                        group="reference" if kind == "torch" else "target",
                        runner=runner_for_kind(kind, name),
                        dtype=dtype,
                        multi_precision=precision != "default",
                        metadata={
                            "kind": kind,
                            "implementation": name,
                            "api_name": op_name,
                            "precision": precision,
                        },
                    )
                )
    return specs


def implementation_kind(name: str) -> str:
    if name in {"paddle", "torch"}:
        return name
    return "custom"


def runner_for_kind(kind: str, name: str):
    if kind == "paddle":
        return run_paddle_case
    if kind == "torch":
        return run_torch_case
    try:
        custom_runner = CUSTOM_IMPLEMENTATIONS[name]
    except KeyError as err:
        raise ValueError(f"unknown custom implementation: {name}") from err

    def runner(case: CompareCase) -> torch.Tensor:
        return custom_runner(case, current_spec(case))

    return runner


def build_compare_suite(
    *,
    config_lines: Iterable[str],
    implementation_names: Iterable[str],
    standard: str,
    dtypes: Iterable[str | None] | None = None,
    precisions: Iterable[str] | None = None,
    metrics_dtype: str = "fp64",
    enable_fingerprint: bool = True,
) -> CompareSuite:
    cases = cases_from_config_lines(config_lines)
    if not cases:
        raise ValueError("no operator compare cases loaded")
    op_name = cases[0].metadata["api_name"]
    if any(case.metadata["api_name"] != op_name for case in cases):
        raise ValueError("all cases in one compare suite must use the same api")
    implementations = expand_implementations(
        op_name=op_name,
        implementation_names=implementation_names,
        dtypes=dtypes,
        precisions=precisions,
    )
    return CompareSuite(
        op_name=op_name,
        cases=cases,
        implementations=implementations,
        standard_id=standard,
        target_groups={"target"},
        reference_groups={"reference"},
        metrics_dtype=metrics_dtype,
        enable_fingerprint=enable_fingerprint,
        metadata={"source": "config"},
        report_config={
            "title": f"{op_name} 多实现对比报告",
            "method_intro": "测试用例来自 PaddleAPITest config。",
            "shape_metadata_keys": ["api_name"],
            "shape_label_prefix": "API",
            "summary_case_metadata_columns": ["api_name", "raw_config"],
            "summary_implementation_metadata_columns": [
                "kind",
                "implementation",
                "precision",
                "output_fingerprint",
            ],
        },
    )


def clone_api_config(api_config: APIConfig, dtype: str | None) -> APIConfig:
    seeded = seeded_api_config(api_config, dtype)
    cloned = copy.deepcopy(seeded)
    prepare_tensor_configs(cloned.args, dtype)
    prepare_tensor_configs(cloned.kwargs.values(), dtype)
    copy_numpy_tensors(seeded.args, cloned.args)
    copy_numpy_tensors(seeded.kwargs.values(), cloned.kwargs.values())
    return cloned


def seeded_api_config(api_config: APIConfig, dtype: str | None) -> APIConfig:
    cache = getattr(api_config, "_operator_compare_seeded_configs", None)
    if cache is None:
        cache = {}
        setattr(api_config, "_operator_compare_seeded_configs", cache)
    cache_key = dtype or "config"
    if cache_key not in cache:
        seeded = copy.deepcopy(api_config)
        prepare_tensor_configs(seeded.args, dtype)
        prepare_tensor_configs(seeded.kwargs.values(), dtype)
        materialize_numpy_tensors(seeded.args, seeded)
        materialize_numpy_tensors(seeded.kwargs.values(), seeded)
        cache[cache_key] = seeded
    return cache[cache_key]


def prepare_tensor_configs(values: Iterable[Any], dtype: str | None) -> None:
    for value in values:
        if isinstance(value, TensorConfig):
            if dtype is not None:
                value.dtype = dtype
            value.numpy_tensor = None
            value.paddle_tensor = None
            value.torch_tensor = None
            if not hasattr(value, "shuffle_dims"):
                value.shuffle_dims = None
        elif isinstance(value, (list, tuple)):
            prepare_tensor_configs(value, dtype)


def materialize_numpy_tensors(values: Iterable[Any], api_config: APIConfig) -> None:
    for value in values:
        if isinstance(value, TensorConfig):
            value.get_numpy_tensor(api_config)
        elif isinstance(value, (list, tuple)):
            materialize_numpy_tensors(value, api_config)


def copy_numpy_tensors(source_values: Iterable[Any], target_values: Iterable[Any]) -> None:
    for source, target in zip(source_values, target_values):
        if isinstance(source, TensorConfig) and isinstance(target, TensorConfig):
            target.numpy_tensor = copy_tensor_value(source.numpy_tensor)
        elif isinstance(source, (list, tuple)) and isinstance(target, (list, tuple)):
            copy_numpy_tensors(source, target)


def copy_tensor_value(value: Any) -> Any:
    if hasattr(value, "copy"):
        return value.copy()
    return copy.deepcopy(value)


def run_paddle_case(case: CompareCase) -> torch.Tensor:
    spec = current_spec(case)
    api_config = clone_api_config(case.tensors["api_config"], spec.dtype)
    args = materialize_paddle_args(api_config.args, api_config)
    kwargs = OrderedDict(
        (key, materialize_paddle_value(value, api_config))
        for key, value in api_config.kwargs.items()
    )
    output = eval(api_config.api_name)(*args, **kwargs)
    return to_torch_tensor(output)


def run_torch_case(case: CompareCase) -> torch.Tensor:
    spec = current_spec(case)
    api_config = clone_api_config(case.tensors["api_config"], spec.dtype)
    convert_result = get_converter().convert(api_config.api_name)
    if not convert_result.is_supported:
        raise RuntimeError(
            convert_result.error_message or f"unsupported torch mapping: {api_config.api_name}"
        )
    torch_args = materialize_torch_args(api_config.args, api_config)
    torch_kwargs = bind_torch_kwargs(api_config, torch_args)
    output = Paddle2TorchConverter.execute(convert_result, torch_args, torch_kwargs)
    return to_torch_tensor(output)


def current_spec(case: CompareCase) -> ImplementationSpec:
    try:
        return case.tensors["_current_spec"]
    except KeyError as err:
        raise RuntimeError("operator compare runner missing current implementation spec") from err


def bind_torch_kwargs(api_config: APIConfig, torch_args: list[Any]) -> OrderedDict[str, Any]:
    named_values = bind_api_arguments(api_config, torch_args)
    named_values.update(
        (key, materialize_torch_value(value, api_config))
        for key, value in api_config.kwargs.items()
    )
    return OrderedDict((key, value) for key, value in named_values.items() if value is not None)


def bind_api_arguments(api_config: APIConfig, torch_args: list[Any]) -> OrderedDict[str, Any]:
    argument_names = argument_names_for_api(api_config.api_name)
    return OrderedDict(
        (name, torch_args[index])
        for index, name in enumerate(argument_names)
        if index < len(torch_args)
    )


def argument_names_for_api(api_name: str) -> tuple[str, ...]:
    if api_name in MANUAL_ARGUMENT_NAMES:
        return MANUAL_ARGUMENT_NAMES[api_name]
    if api_name not in SIGNATURE_ARGUMENT_NAMES:
        SIGNATURE_ARGUMENT_NAMES[api_name] = signature_argument_names(api_name)
    return SIGNATURE_ARGUMENT_NAMES[api_name]


def signature_argument_names(api_name: str) -> tuple[str, ...]:
    try:
        signature = inspect.signature(eval(api_name))
    except (TypeError, ValueError):
        return ()
    return tuple(
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    )


def materialize_paddle_args(values: Iterable[Any], api_config: APIConfig) -> list[Any]:
    return [materialize_paddle_value(value, api_config) for value in values]


def materialize_paddle_value(value: Any, api_config: APIConfig) -> Any:
    if isinstance(value, TensorConfig):
        tensor = value.get_paddle_tensor(api_config)
        value.clear_paddle_tensor()
        return tensor
    if isinstance(value, list):
        return [materialize_paddle_value(item, api_config) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_paddle_value(item, api_config) for item in value)
    return value


def materialize_torch_args(values: Iterable[Any], api_config: APIConfig) -> list[Any]:
    return [materialize_torch_value(value, api_config) for value in values]


def materialize_torch_value(value: Any, api_config: APIConfig) -> Any:
    if isinstance(value, TensorConfig):
        tensor = value.get_torch_tensor(api_config)
        value.clear_torch_tensor()
        return tensor
    if isinstance(value, list):
        return [materialize_torch_value(item, api_config) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize_torch_value(item, api_config) for item in value)
    return value


def to_torch_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, paddle.Tensor):
        return torch.utils.dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(output))
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, (paddle.Tensor, torch.Tensor)):
                return to_torch_tensor(item)
        raise TypeError(f"output {type(output).__name__} contains no tensor")
    raise TypeError(f"output type {type(output).__name__} is not supported")
