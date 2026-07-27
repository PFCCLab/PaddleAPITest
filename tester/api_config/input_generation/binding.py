"""Shadow argument binding for the future case-level input generator."""

from __future__ import annotations

import collections
import inspect
from dataclasses import dataclass
from pathlib import Path

import yaml

from .model import (
    ArgPath,
    BoundCall,
    GenerationContext,
    ParameterBinding,
    TensorBinding,
    TensorSpec,
)
from .tensor_config import TensorConfig

BASE_CONFIG = Path(__file__).resolve().parents[2] / "base_config.yaml"

COPS_PUBLIC_ALIASES = {
    "paddle._C_ops.add_": "paddle.add",
    "paddle._C_ops.bitwise_not": "paddle.bitwise_not",
    "paddle._C_ops.clip": "paddle.clip",
    "paddle._C_ops.concat": "paddle.concat",
    "paddle._C_ops.flatten_": "paddle.flatten",
    "paddle._C_ops.matmul": "paddle.matmul",
    "paddle._C_ops.multiply_": "paddle.multiply",
    "paddle._C_ops.numel": "paddle.numel",
    "paddle._C_ops.put_along_axis_": "paddle.put_along_axis",
    "paddle._C_ops.reshape_": "paddle.reshape",
    "paddle._C_ops.scale_": "paddle.scale",
    "paddle._C_ops.subtract_": "paddle.subtract",
    "paddle._C_ops.transpose": "paddle.transpose",
    "paddle._C_ops.uniform": "paddle.uniform",
}

MANUAL_PARAMETER_NAMES = {
    "paddle.Tensor.__getitem__": ("self", "item"),
    "paddle.Tensor.__setitem__": ("self", "item", "value"),
    "paddle._C_ops.adamw_": (
        "param",
        "grad",
        "learning_rate",
        "moment1",
        "moment2",
        "moment2_max",
        "beta1_pow",
        "beta2_pow",
        "master_param",
        "skip_update",
        "beta1",
        "beta2",
        "epsilon",
        "lr_ratio",
        "coeff",
        "with_decay",
        "lazy_mode",
        "min_row_size_to_use_multithread",
        "multi_precision",
        "use_global_beta_pow",
        "amsgrad",
    ),
    "paddle._C_ops.full_": ("x", "shape", "value", "dtype"),
    "paddle._C_ops.fused_linear_param_grad_add": (
        "x",
        "dout",
        "dweight",
        "dbias",
        "multi_precision",
        "has_bias",
    ),
    "paddle._C_ops.gaussian": ("shape", "mean", "std", "seed", "dtype"),
    "paddle._C_ops.matmul_grad": ("x", "y", "dout", "transpose_x", "transpose_y"),
    "paddle._C_ops.squared_l2_norm": ("x",),
    "paddle._C_ops.swiglu_grad": ("x", "y", "dout"),
    "paddle._C_ops._run_custom_op": ("op_name", "arg1", "arg2", "arg3", "arg4"),
}


def _load_single_op_parameter_names():
    data = yaml.safe_load(BASE_CONFIG.read_text())
    for method in data.get("single_op_no_signature_apis", []):
        MANUAL_PARAMETER_NAMES[f"paddle.Tensor.{method}"] = ("self", "y")


_load_single_op_parameter_names()


@dataclass(frozen=True)
class BindingResolution:
    arguments: collections.OrderedDict
    source: str
    path_parameters: tuple[ParameterBinding, ...]
    unresolved_reason: str | None = None


def _resolve_dotted_api(api_name):
    import paddle

    value = paddle
    parts = api_name.split(".")
    if not parts or parts[0] != "paddle":
        raise ValueError(f"unsupported API root: {api_name}")
    for part in parts[1:]:
        value = getattr(value, part)
    return value


def _get_arg(api_config, position, name, default=None):
    if 0 <= position < len(api_config.args):
        return api_config.args[position]
    if name in api_config.kwargs:
        return api_config.kwargs[name]
    return default


def _manual_arguments(api_config, parameter_names):
    return collections.OrderedDict(
        (name, _get_arg(api_config, position, name))
        for position, name in enumerate(parameter_names)
    )


def _contains_identity(value, target):
    if value is target:
        return True
    if isinstance(value, (list, tuple)):
        return any(_contains_identity(item, target) for item in value)
    return False


def _path_parameters(api_config, arguments):
    bindings = []
    for index, value in enumerate(api_config.args):
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        bindings.append(
            ParameterBinding(
                path=ArgPath.positional(index),
                parameter_name=names[0] if len(names) == 1 else None,
            )
        )
    for key, value in api_config.kwargs.items():
        names = [name for name, bound in arguments.items() if _contains_identity(bound, value)]
        bindings.append(
            ParameterBinding(
                path=ArgPath.keyword(key),
                parameter_name=names[0] if len(names) == 1 else key,
            )
        )
    return tuple(bindings)


class SignatureResolver:
    def __init__(self):
        self._signature_cache = {}

    def _signature(self, cache_key, api):
        if cache_key not in self._signature_cache:
            try:
                self._signature_cache[cache_key] = inspect.signature(api)
            except (TypeError, ValueError):
                self._signature_cache[cache_key] = None
        return self._signature_cache[cache_key]

    def _resolved(self, api_config, arguments, source, keep_name=False):
        arguments = collections.OrderedDict(arguments)
        if not keep_name:
            arguments.pop("name", None)
        return BindingResolution(
            arguments=arguments,
            source=source,
            path_parameters=_path_parameters(api_config, arguments),
        )

    def resolve(self, api_config, api=None):
        api_name = api_config.api_name
        if api_name in MANUAL_PARAMETER_NAMES:
            return self._resolved(
                api_config,
                _manual_arguments(api_config, MANUAL_PARAMETER_NAMES[api_name]),
                "manual",
            )

        api = api or _resolve_dotted_api(api_name)
        args = api_config.args
        if api_name in ("paddle.Tensor.view", "paddle.view"):
            rest = args[1:]
            if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
                return self._resolved(
                    api_config,
                    {"x": args[0], "shape_or_dtype": list(rest)},
                    "variadic-view",
                )
        if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
            rest = args[1:]
            if rest and all(isinstance(arg, int) for arg in rest):
                return self._resolved(
                    api_config,
                    {"x": args[0], "shape": list(rest)},
                    "variadic-reshape",
                )

        signature = self._signature(api_name, api)
        source = "signature"
        if signature is None:
            public_api_name = COPS_PUBLIC_ALIASES.get(api_name)
            if public_api_name is None:
                return BindingResolution(
                    arguments=collections.OrderedDict(),
                    source="unresolved",
                    path_parameters=_path_parameters(api_config, {}),
                    unresolved_reason="API has no inspectable signature or manual mapping",
                )
            public_api = _resolve_dotted_api(public_api_name)
            signature = self._signature(public_api_name, public_api)
            if signature is None:
                return BindingResolution(
                    arguments=collections.OrderedDict(),
                    source="unresolved",
                    path_parameters=_path_parameters(api_config, {}),
                    unresolved_reason=f"public alias has no signature: {public_api_name}",
                )
            source = f"public-alias:{public_api_name}"
            positional_count = sum(
                parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                for parameter in signature.parameters.values()
            )
            valid_names = set(signature.parameters)
            kwargs = {key: value for key, value in api_config.kwargs.items() if key in valid_names}
            bound = signature.bind(*args[:positional_count], **kwargs)
        else:
            bound = signature.bind(*args, **api_config.kwargs)

        arguments = collections.OrderedDict(bound.arguments)
        if api_name == "paddle.arange" and "end" not in arguments:
            arguments["end"] = arguments["start"]
            arguments["start"] = 0
        if api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"}:
            arguments.setdefault("name", None)
            return self._resolved(api_config, arguments, source, keep_name=True)
        return self._resolved(api_config, arguments, source)


def _walk_tensors(value, path, parameter_name, output):
    if isinstance(value, TensorConfig):
        output.append(
            TensorBinding(
                path=path,
                parameter_name=parameter_name,
                spec=TensorSpec.from_tensor_config(value),
            )
        )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_tensors(child, path.child(index), parameter_name, output)


def bind_call(api_config, resolver=None):
    resolver = resolver or SignatureResolver()
    resolution = resolver.resolve(api_config)
    parameter_by_path = {
        binding.path: binding.parameter_name for binding in resolution.path_parameters
    }
    tensors = []
    for index, value in enumerate(api_config.args):
        path = ArgPath.positional(index)
        _walk_tensors(value, path, parameter_by_path.get(path), tensors)
    for key, value in api_config.kwargs.items():
        path = ArgPath.keyword(key)
        _walk_tensors(value, path, parameter_by_path.get(path), tensors)
    return BoundCall(
        api_name=api_config.api_name,
        binding_source=resolution.source,
        parameter_bindings=resolution.path_parameters,
        tensors=tuple(tensors),
        unresolved_reason=resolution.unresolved_reason,
    )


def build_generation_context(
    api_config,
    resolver=None,
    seed=0,
    runtime_mode="legacy",
    use_torch=True,
    gpu_enabled=False,
):
    call = bind_call(api_config, resolver=resolver)
    return GenerationContext.create(
        call=call,
        config_text=api_config.config,
        seed=seed,
        runtime_mode=runtime_mode,
        use_torch=use_torch,
        gpu_enabled=gpu_enabled,
    )
