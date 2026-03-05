from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Any

import torch

from . import rules
from .rules import BaseRule, ConvertResult, ErrorRule, GenericRule

PADDLE2TORCH_WRONG_CONFIG = frozenset([])


class Paddle2TorchConverter:
    __slots__ = ("cached_results", "mapping", "rules")

    def __init__(self):
        self.rules: dict[str, Any] = {}
        self.mapping: dict[str, Any] = {}
        self.cached_results: dict[str, ConvertResult] = {}
        self._load_mapping_and_rules()

    def _load_mapping_and_rules(self):
        rule_cls_map: dict[str, type[BaseRule]] = {
            "GenericRule": GenericRule,
            "ErrorRule": ErrorRule,
        }
        for rule_name in rules.ALL_RULES:
            rule_cls_map[rule_name] = getattr(rules, rule_name)

        mapping_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapping.json")
        with open(mapping_file) as f:
            paddle2torch_mapping = json.load(f, object_pairs_hook=OrderedDict)
            for key, value in paddle2torch_mapping.items():
                if not key.startswith("paddle.") or key in self.rules:
                    continue
                self.mapping[key] = value
                rule_name = value.get("Rule")
                if rule_name is None:
                    self.rules[key] = GenericRule
                elif rule_name in rule_cls_map:
                    self.rules[key] = rule_cls_map[rule_name]
                else:
                    self.rules[key] = ErrorRule

    def convert(self, paddle_api: str) -> ConvertResult:
        """将 Paddle API 转换为 Torch API

        Args:
            paddle_api (str): 需要转换的 Paddle API 名称

        Returns:
            ConvertResult: 转换结果，包括转换后的 Torch API 代码、输出变量或错误信息

        """
        try:
            return self.cached_results[paddle_api]
        except KeyError:
            pass

        if paddle_api in PADDLE2TORCH_WRONG_CONFIG:
            result = ConvertResult.error(paddle_api, f"{paddle_api} is in the wrong config")
            self.cached_results[paddle_api] = result
            return result

        try:
            rule = self.rules[paddle_api]()
        except KeyError:
            result = ConvertResult.error(paddle_api, f"Rule for {paddle_api} is not implemented")
            self.cached_results[paddle_api] = result
            return result

        rule.read_mapping(self.mapping[paddle_api])
        result = rule.apply(paddle_api)
        self.cached_results[paddle_api] = result
        return result

    @staticmethod
    def execute(
        convert_result: ConvertResult,
        torch_args: list,
        torch_kwargs: OrderedDict,
    ) -> Any:
        """执行转换后的代码。

        Args:
            convert_result (ConvertResult): 转换结果对象
            torch_args (List): 传递给 Paddle API 的含有 Torch Tensors 的位置参数列表
            torch_kwargs (OrderedDict): 传递给 Paddle API 的含有 Torch Tensors 的关键字参数字典

        Returns:
            Any: 执行结果

        Raises:
            RuntimeError: 执行转换后的代码时发生异常
            ValueError: 转换结果中指定的输出变量在执行上下文中不存在
        """
        if not convert_result.is_supported or not convert_result.code:
            return None

        # 准备执行环境，将参数(torch tensors)直接映射至locals
        exec_globals = {"torch": torch}
        exec_locals = {
            "args": torch_args,
            "kwargs": torch_kwargs,
            "result": None,
            **torch_kwargs,
        }

        try:
            code = convert_result.code
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)
            if code.core_compiled:
                exec(code.core_compiled, exec_globals, exec_locals)
            if code.postprocess_compiled:
                exec(code.postprocess_compiled, exec_globals, exec_locals)
        except Exception as e:
            raise RuntimeError(f"Error during execution of converted code: {e!s}") from e

        output_var = convert_result.output_var or "result"
        try:
            return exec_locals[output_var]
        except KeyError:
            raise ValueError(f"Variable {output_var} not found in execution context")


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


def clear_converter():
    global _converter_instance
    with _converter_lock:
        _converter_instance = None
