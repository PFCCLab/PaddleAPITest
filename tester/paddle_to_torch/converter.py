from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch

from . import rules
from .config import ConversionEnvironment, read_conversion_environment
from .rules import BaseRule, ConversionKind, ConvertResult, adaptive_workspace_bytes

_MAPPING_FIELDS = frozenset(
    {
        "Rule",
        "description",
        "is_attribute",
        "paddle_torch_args_map",
        "set_defaults",
        "torch_api",
        "torch_args",
        "torch_kwargs",
    }
)
_MAPPING_FIELD_TYPES = {
    "Rule": str,
    "description": str,
    "is_attribute": bool,
    "paddle_torch_args_map": dict,
    "set_defaults": dict,
    "torch_api": str,
    "torch_args": list,
    "torch_kwargs": dict,
}


class Paddle2TorchConverter:
    __slots__ = ("_cache_lock", "_cached_results", "_mapping", "_rules")

    def __init__(
        self,
        *,
        mapping_data: Mapping[str, Any] | None = None,
        extra_rules: Mapping[str, type[BaseRule]] | None = None,
    ):
        rule_classes = dict(rules.get_rule_registry())
        if extra_rules:
            for rule_name, rule_class in extra_rules.items():
                if rule_name != rule_class.__name__:
                    raise ValueError(
                        f"Rule registry key {rule_name!r} does not match {rule_class.__name__!r}"
                    )
                if not issubclass(rule_class, BaseRule):
                    raise TypeError(f"Rule {rule_name!r} must inherit from BaseRule")
                if rule_name in rule_classes and rule_classes[rule_name] is not rule_class:
                    raise ValueError(f"Rule {rule_name!r} is already registered")
                rule_classes[rule_name] = rule_class
        if mapping_data is None:
            mapping_data = self._load_mapping_file()
        self._validate_mapping(mapping_data, rule_classes)
        self._mapping = self._freeze_mapping(mapping_data)
        self._rules = MappingProxyType(
            {
                paddle_api: rule_classes[mapping.get("Rule", "GenericRule")]
                for paddle_api, mapping in self._mapping.items()
            }
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
        if isinstance(value, list):
            return tuple(cls._freeze_mapping(item) for item in value)
        return value

    @staticmethod
    def _cache_key(
        paddle_api: str,
        environment: ConversionEnvironment,
    ) -> tuple[str, ConversionEnvironment]:
        return paddle_api, environment

    @staticmethod
    def _validate_mapping(
        paddle2torch_mapping: Any,
        rule_cls_map: Mapping[str, type[BaseRule]],
    ) -> None:
        if not isinstance(paddle2torch_mapping, Mapping):
            raise ValueError("Paddle-to-Torch mapping root must be an object")
        for paddle_api, mapping in paddle2torch_mapping.items():
            if not isinstance(paddle_api, str) or not paddle_api.startswith("paddle."):
                raise ValueError(f"Invalid Paddle API name {paddle_api!r}")
            if not isinstance(mapping, dict):
                raise ValueError(f"Mapping for {paddle_api} must be an object")
            unknown_fields = set(mapping) - _MAPPING_FIELDS
            if unknown_fields:
                fields = ", ".join(sorted(unknown_fields))
                raise ValueError(f"Mapping for {paddle_api} has unknown fields: {fields}")
            for field_name, value in mapping.items():
                expected_type = _MAPPING_FIELD_TYPES[field_name]
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Mapping field {paddle_api}.{field_name} must be "
                        f"{expected_type.__name__}, got {type(value).__name__}"
                    )
            rule_name = mapping.get("Rule", "GenericRule")
            if not rule_name:
                raise ValueError(f"Mapping field {paddle_api}.Rule must not be empty")
            if rule_name not in rule_cls_map:
                raise ValueError(f"Unknown rule {rule_name!r} configured for {paddle_api}")
            if rule_name == "GenericRule" and not mapping.get("torch_api"):
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
            if any(not isinstance(arg, str) for arg in mapping.get("torch_args", [])):
                raise ValueError(f"Mapping field {paddle_api}.torch_args requires string values")

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
        cache_key = self._cache_key(paddle_api, environment)

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

            rule = rule_cls()
            rule.set_conversion_environment(environment)
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
    def execute(
        convert_result: ConvertResult,
        torch_args: list,
        torch_kwargs: Mapping[str, Any],
        *,
        execution_locals: Mapping[str, Any] | None = None,
        core_executor: Callable[[Any, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> Any:
        """执行转换后的代码。

        Args:
            convert_result (ConvertResult): 转换结果对象
            torch_args (List): 传递给 Paddle API 的含有 Torch Tensors 的位置参数列表
            torch_kwargs (OrderedDict): 传递给 Paddle API 的含有 Torch Tensors 的关键字参数字典
            execution_locals (Mapping): 额外注入生成代码执行环境的局部变量
            core_executor (Callable): 可选的 core 阶段执行器，用于包裹 AMP 等调用方上下文

        Returns:
            Any: 执行结果

        Raises:
            RuntimeError: 执行转换后的代码时发生异常
            ValueError: 转换结果中指定的输出变量在执行上下文中不存在
        """
        if convert_result.kind is ConversionKind.UNSUPPORTED:
            raise ValueError(
                f"Cannot execute unsupported conversion for {convert_result.paddle_api}: "
                f"{convert_result.error_message}"
            )

        # 准备执行环境，将参数(torch tensors)直接映射至locals
        exec_globals = {"torch": torch, "_adaptive_workspace_bytes": adaptive_workspace_bytes}
        exec_locals = {
            "args": torch_args,
            "kwargs": torch_kwargs,
            "result": None,
            **torch_kwargs,
        }
        if execution_locals:
            exec_locals.update(execution_locals)

        code = convert_result.code
        stage = "preprocess"
        try:
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)
            stage = "core"
            if code.core_compiled:
                if core_executor is None:
                    exec(code.core_compiled, exec_globals, exec_locals)
                else:
                    core_executor(code.core_compiled, exec_globals, exec_locals)
            stage = "postprocess"
            if code.postprocess_compiled:
                exec(code.postprocess_compiled, exec_globals, exec_locals)
        except Exception as e:
            raise RuntimeError(
                f"Failed to execute {convert_result.paddle_api} during {stage}: {e!s}"
            ) from e

        output_var = convert_result.output_var or "result"
        try:
            return exec_locals[output_var]
        except KeyError:
            raise ValueError(
                f"Output variable {output_var!r} for {convert_result.paddle_api} "
                "was not found in the execution context"
            )


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
