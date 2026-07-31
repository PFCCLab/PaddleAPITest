"""输入路径。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputPath:
    """一次 API 调用中 Tensor 的稳定路径。"""

    # 这个路径不受 TensorConfig 可变字段影响，规则和 payload 都依赖它。
    root: str
    key: int | str
    indices: tuple[int, ...] = ()

    def __post_init__(self):
        if self.root == "args":
            if not isinstance(self.key, int) or self.key < 0:
                raise ValueError("args path key must be a non-negative integer")
        elif self.root == "kwargs":
            if not isinstance(self.key, str) or not self.key:
                raise ValueError("kwargs path key must be a non-empty string")
        else:
            raise ValueError(f"unsupported argument root: {self.root!r}")
        if any(not isinstance(index, int) or index < 0 for index in self.indices):
            raise ValueError("nested argument indices must be non-negative integers")

    @classmethod
    def positional(cls, index, indices=()):
        return cls("args", index, tuple(indices))

    @classmethod
    def keyword(cls, name, indices=()):
        return cls("kwargs", name, tuple(indices))

    def child(self, index):
        return InputPath(self.root, self.key, (*self.indices, index))

    def top_level(self):
        return InputPath(self.root, self.key)

    def __str__(self):
        value = f"args[{self.key}]" if self.root == "args" else f"kwargs.{self.key}"
        for index in self.indices:
            value += f"[{index}]"
        return value
