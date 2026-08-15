from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from . import rules
from .rules import (
    BaseRule,
    ConversionEnvironment,
    ConversionKind,
    ConvertResult,
    GenericRule,
    adaptive_workspace_bytes,
    read_conversion_environment,
)

_MAPPING_FIELD_TYPES = {
    "Rule": str,
    "description": str,
    "is_attribute": bool,
    "paddle_torch_args_map": Mapping,
    "set_defaults": Mapping,
    "torch_api": str,
    "torch_args": (list, tuple),
    "torch_kwargs": Mapping,
}


@dataclass
class ExecutionContext:
    convert_result: ConvertResult
    namespace: dict[str, Any]


class Paddle2TorchConverter:
    __slots__ = ("_cache_lock", "_cached_results", "_mapping", "_rules")

    def __init__(
        self,
        *,
        mapping_data: Mapping[str, Any] | None = None,
        extra_rules: Mapping[str, type[BaseRule]] | None = None,
    ):
        api_rules = dict(rules.get_rule_registry())
        if extra_rules:
            for paddle_api, rule_class in extra_rules.items():
                if not isinstance(rule_class, type) or not issubclass(rule_class, BaseRule):
                    raise TypeError(f"Rule for {paddle_api!r} must inherit from BaseRule")
                if paddle_api not in rule_class.PADDLE_APIS:
                    raise ValueError(
                        f"Extra Rule {rule_class.__name__} does not declare {paddle_api!r}"
                    )
                if paddle_api in api_rules and api_rules[paddle_api] is not rule_class:
                    raise ValueError(f"Rule for {paddle_api!r} is already registered")
                api_rules[paddle_api] = rule_class
        require_complete_registry = mapping_data is None
        if require_complete_registry:
            mapping_data = self._load_mapping_file()
        self._validate_mapping(
            mapping_data,
            api_rules,
            require_complete_registry=require_complete_registry,
        )
        self._mapping = self._freeze_mapping(mapping_data)
        self._rules = MappingProxyType(
            {paddle_api: api_rules.get(paddle_api, GenericRule) for paddle_api in self._mapping}
        )
        self._cached_results: dict[tuple[str, ConversionEnvironment], ConvertResult] = {}
        self._cache_lock = threading.Lock()

    @property
    def mapping(self) -> Mapping[str, Mapping[str, Any]]:
        return self._mapping

    @staticmethod
    def _reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate Paddle-to-Torch mapping key {key!r}")
            result[key] = value
        return result

    @classmethod
    def _load_mapping_file(cls) -> Mapping[str, Any]:
        mapping_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapping.json")
        with open(mapping_file) as mapping_stream:
            return json.load(mapping_stream, object_pairs_hook=cls._reject_duplicate_keys)

    @classmethod
    def _freeze_mapping(cls, value):
        if isinstance(value, Mapping):
            return MappingProxyType(
                {key: cls._freeze_mapping(nested) for key, nested in value.items()}
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_mapping(item) for item in value)
        return value

    @staticmethod
    def _validate_mapping(
        paddle2torch_mapping: Any,
        api_rules: Mapping[str, type[BaseRule]],
        *,
        require_complete_registry: bool = False,
    ) -> None:
        if not isinstance(paddle2torch_mapping, Mapping):
            raise ValueError("Paddle-to-Torch mapping root must be an object")
        for paddle_api, mapping in paddle2torch_mapping.items():
            if not isinstance(paddle_api, str) or not paddle_api.startswith("paddle."):
                raise ValueError(f"Invalid Paddle API name {paddle_api!r}")
            if not isinstance(mapping, Mapping):
                raise ValueError(f"Mapping for {paddle_api} must be an object")
            unknown_fields = set(mapping) - _MAPPING_FIELD_TYPES.keys()
            if unknown_fields:
                fields = ", ".join(sorted(unknown_fields))
                raise ValueError(f"Mapping for {paddle_api} has unknown fields: {fields}")
            for field_name, value in mapping.items():
                expected_type = _MAPPING_FIELD_TYPES[field_name]
                if not isinstance(value, expected_type):
                    expected_types = (
                        expected_type if isinstance(expected_type, tuple) else (expected_type,)
                    )
                    expected_name = " or ".join(item.__name__ for item in expected_types)
                    raise ValueError(
                        f"Mapping field {paddle_api}.{field_name} must be "
                        f"{expected_name}, got {type(value).__name__}"
                    )
            configured_rule = mapping.get("Rule")
            registered_rule = api_rules.get(paddle_api)
            if registered_rule is None:
                if configured_rule is not None:
                    raise ValueError(
                        f"Mapping for {paddle_api} configures unknown Rule {configured_rule!r}"
                    )
            elif configured_rule != registered_rule.__name__:
                raise ValueError(
                    f"Mapping for {paddle_api} must configure Rule "
                    f"{registered_rule.__name__!r}, got {configured_rule!r}"
                )
            if registered_rule is None and not mapping.get("torch_api"):
                raise ValueError(
                    f"Mapping field {paddle_api}.torch_api is required for GenericRule"
                )
            for field_name in ("set_defaults", "torch_kwargs"):
                for key in mapping.get(field_name, {}):
                    if not isinstance(key, str) or not key:
                        raise ValueError(
                            f"Mapping field {paddle_api}.{field_name} requires non-empty string keys"
                        )
            for paddle_name, torch_name in mapping.get("paddle_torch_args_map", {}).items():
                if not isinstance(paddle_name, str) or not paddle_name:
                    raise ValueError(
                        f"Mapping field {paddle_api}.paddle_torch_args_map "
                        "requires non-empty string keys"
                    )
                if not isinstance(torch_name, str) or not torch_name:
                    raise ValueError(
                        f"Mapping field {paddle_api}.paddle_torch_args_map "
                        "requires non-empty string values"
                    )
            for arg in mapping.get("torch_args", []):
                if not isinstance(arg, str):
                    raise ValueError(
                        f"Mapping field {paddle_api}.torch_args requires string values"
                    )
        if require_complete_registry:
            missing_apis = set(api_rules) - set(paddle2torch_mapping)
            if missing_apis:
                raise ValueError(
                    "Registered Rules are missing mapping entries: "
                    + ", ".join(sorted(missing_apis))
                )

    def convert(self, paddle_api: str) -> ConvertResult:
        """将 Paddle API 转换为 Torch API

        Args:
            paddle_api (str): 需要转换的 Paddle API 名称

        Returns:
            ConvertResult: 转换结果，包括转换后的 Torch API 代码、输出变量或错误信息

        """
        try:
            environment = read_conversion_environment()
        except ValueError as exc:
            raise ValueError(f"Cannot convert {paddle_api}: {exc}") from exc
        cache_key = (paddle_api, environment)

        with self._cache_lock:
            try:
                return self._cached_results[cache_key]
            except KeyError:
                pass

            try:
                rule_cls = self._rules[paddle_api]
            except KeyError:
                result = ConvertResult.error(
                    paddle_api,
                    f"Rule for {paddle_api} is not implemented",
                )
                self._cached_results[cache_key] = result
                return result

            rule = rule_cls(environment)
            try:
                rule.read_mapping(self._mapping[paddle_api])
                result = rule.apply(paddle_api)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to convert {paddle_api} with {rule_cls.__name__}: {exc}"
                ) from exc
            if not isinstance(result, ConvertResult):
                raise TypeError(
                    f"Rule {rule_cls.__name__} for {paddle_api} returned "
                    f"{type(result).__name__}, expected ConvertResult"
                )
            self._cached_results[cache_key] = result
            return result

    @staticmethod
    def prepare_execution(
        convert_result: ConvertResult,
        torch_args: list,
        bound_arguments: Mapping[str, Any],
        *,
        execution_locals: Mapping[str, Any] | None = None,
    ) -> ExecutionContext:
        if convert_result.kind is ConversionKind.UNSUPPORTED:
            raise ValueError(
                f"Cannot execute unsupported conversion for {convert_result.paddle_api}: "
                f"{convert_result.error_message}"
            )
        namespace = {
            "torch": torch,
            "_adaptive_workspace_bytes": adaptive_workspace_bytes,
            "positional_arguments": torch_args,
            "bound_arguments": bound_arguments,
            "result": None,
            **bound_arguments,
            **(execution_locals or {}),
        }
        return ExecutionContext(convert_result=convert_result, namespace=namespace)

    @staticmethod
    def _execute_stage(
        context: ExecutionContext,
        stage: str,
        *,
        core_executor: Callable[[Any, dict[str, Any], dict[str, Any]], None] | None = None,
        repeat: int = 1,
    ) -> None:
        compiled = getattr(context.convert_result.code, f"{stage}_compiled")
        if compiled is None:
            return
        namespace = context.namespace
        executor = core_executor or exec
        try:
            for _ in range(repeat):
                executor(compiled, namespace, namespace)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to execute {context.convert_result.paddle_api} during {stage}: {exc!s}"
            ) from exc

    @classmethod
    def run_preprocess(cls, context: ExecutionContext) -> None:
        cls._execute_stage(context, "preprocess")

    @classmethod
    def run_core(
        cls,
        context: ExecutionContext,
        *,
        core_executor: Callable[[Any, dict[str, Any], dict[str, Any]], None] | None = None,
        repeat: int = 1,
    ) -> None:
        cls._execute_stage(context, "core", core_executor=core_executor, repeat=repeat)

    @classmethod
    def run_postprocess(cls, context: ExecutionContext) -> None:
        cls._execute_stage(context, "postprocess")

    @staticmethod
    def get_output(context: ExecutionContext) -> Any:
        output_var = context.convert_result.output_var or "result"
        try:
            return context.namespace[output_var]
        except KeyError:
            raise ValueError(
                f"Output variable {output_var!r} for {context.convert_result.paddle_api} "
                "was not found in the execution context"
            )

    @classmethod
    def execute(
        cls,
        convert_result: ConvertResult,
        torch_args: list,
        bound_arguments: Mapping[str, Any],
        *,
        execution_locals: Mapping[str, Any] | None = None,
        core_executor: Callable[[Any, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> Any:
        """Prepare and execute all stages of one converted Paddle invocation."""
        context = cls.prepare_execution(
            convert_result,
            torch_args,
            bound_arguments,
            execution_locals=execution_locals,
        )
        cls.run_preprocess(context)
        cls.run_core(context, core_executor=core_executor)
        cls.run_postprocess(context)
        return cls.get_output(context)


# 模块级变量与实例管理
_converter_instance = None
_converter_lock = threading.Lock()


def get_converter() -> Paddle2TorchConverter:
    global _converter_instance
    if _converter_instance is None:
        with _converter_lock:
            if _converter_instance is None:
                _converter_instance = Paddle2TorchConverter()
    return _converter_instance
