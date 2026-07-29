"""Shadow argument binding for the future case-level input generator."""

from __future__ import annotations

import collections
from dataclasses import dataclass

from .model import (
    ArgPath,
    BoundCall,
    GenerationContext,
    ParameterBinding,
    TensorBinding,
    TensorSpec,
)
from .signature_mappings import bind_api_arguments
from .tensor_config import TensorConfig


@dataclass(frozen=True)
class BindingResolution:
    arguments: collections.OrderedDict
    source: str
    path_parameters: tuple[ParameterBinding, ...]
    unresolved_reason: str | None = None


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
        resolution = bind_api_arguments(
            api_name,
            api_config.args,
            api_config.kwargs,
            api=api or None,
            signature_cache=self._signature_cache,
            keep_name=api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"},
        )
        if resolution.source == "unresolved":
            return BindingResolution(
                arguments=collections.OrderedDict(),
                source="unresolved",
                path_parameters=_path_parameters(api_config, {}),
                unresolved_reason=resolution.unresolved_reason
                or "API has no inspectable signature or public alias",
            )
        return self._resolved(
            api_config,
            resolution.arguments,
            resolution.source,
            keep_name=api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"},
        )


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
