"""输入路径。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputTensorPath:
    """一次 API 调用中 Tensor 的稳定路径。"""

    # 这个路径不受 TensorConfig 可变字段影响，规则和输入数据都依赖它。
    argument_kind: str
    argument_key: int | str
    item_indices: tuple[int, ...] = ()

    def resolve(self, api_config):
        """按稳定路径读取 APIConfig 中的当前值。"""
        # 路径对象统一处理 args、kwargs 和嵌套索引，调用方不重复解析。
        value = (
            api_config.args[self.argument_key]
            if self.argument_kind == "args"
            else api_config.kwargs[self.argument_key]
        )
        for index in self.item_indices:
            value = value[index]
        return value

    def __post_init__(self):
        if self.argument_kind == "args":
            if not isinstance(self.argument_key, int) or self.argument_key < 0:
                raise ValueError("args path key must be a non-negative integer")
        elif self.argument_kind == "kwargs":
            if not isinstance(self.argument_key, str) or not self.argument_key:
                raise ValueError("kwargs path key must be a non-empty string")
        else:
            raise ValueError(f"unsupported argument kind: {self.argument_kind!r}")
        if any(not isinstance(index, int) or index < 0 for index in self.item_indices):
            raise ValueError("nested argument indices must be non-negative integers")

    @classmethod
    def positional(cls, index, indices=()):
        return cls("args", index, tuple(indices))

    @classmethod
    def keyword(cls, name, indices=()):
        return cls("kwargs", name, tuple(indices))

    def child(self, index):
        return InputTensorPath(
            self.argument_kind,
            self.argument_key,
            (*self.item_indices, index),
        )

    def top_level(self):
        return InputTensorPath(self.argument_kind, self.argument_key)

    def __str__(self):
        value = (
            f"args[{self.argument_key}]"
            if self.argument_kind == "args"
            else f"kwargs.{self.argument_key}"
        )
        for index in self.item_indices:
            value += f"[{index}]"
        return value
