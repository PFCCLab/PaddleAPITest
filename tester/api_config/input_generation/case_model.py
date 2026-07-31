"""输入生成管线使用的不可变数据模型。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ArgPath:
    """一次 API 调用中 Tensor 的稳定路径标识。"""

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
        return ArgPath(self.root, self.key, (*self.indices, index))

    def top_level(self):
        return ArgPath(self.root, self.key)

    def __str__(self):
        value = f"args[{self.key}]" if self.root == "args" else f"kwargs.{self.key}"
        for index in self.indices:
            value += f"[{index}]"
        return value


@dataclass(frozen=True)
class TensorSpec:
    """供值生成器消费的 TensorConfig 只读快照。"""

    # 这里只保留值域生成需要的字段，避免规则直接触碰可变 TensorConfig。
    shape: tuple[int, ...]
    dtype: str
    place: str | None
    is_contiguous: bool
    strides: tuple[int, ...] | None

    @classmethod
    def from_tensor_config(cls, config):
        return cls(
            shape=tuple(int(dim) for dim in config.shape),
            dtype=str(config.dtype),
            place=str(config.place) if config.place is not None else None,
            is_contiguous=bool(config.is_contiguous),
            strides=(
                tuple(int(stride) for stride in config.strides)
                if config.strides is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ParameterBinding:
    path: ArgPath
    parameter_name: str | None


@dataclass(frozen=True)
class TensorBinding:
    path: ArgPath
    parameter_name: str | None
    spec: TensorSpec


@dataclass(frozen=True)
class BoundCall:
    """规则侧看到的一次 APIConfig 绑定结果。"""

    # 规则只读取这个对象，不再自己做签名解析或参数重排。
    api_name: str
    binding_source: str
    parameter_bindings: tuple[ParameterBinding, ...]
    tensors: tuple[TensorBinding, ...]
    unresolved_reason: str | None = None

    def parameter_for(self, path):
        top_level = path.top_level()
        for binding in self.parameter_bindings:
            if binding.path == top_level:
                return binding.parameter_name
        return None

    def tensor_at(self, path):
        for binding in self.tensors:
            if binding.path == path:
                return binding
        raise KeyError(str(path))


@dataclass(frozen=True)
class GenerationContext:
    """一次 case 的运行时上下文。"""

    # 运行时标志在这里一次性快照，规则不应直接读全局状态。
    call: BoundCall
    config_fingerprint: str
    seed: int
    runtime_mode: str
    use_torch: bool
    gpu_enabled: bool

    @classmethod
    def create(
        cls,
        call,
        config_text,
        seed=0,
        runtime_mode="v2",
        use_torch=True,
        gpu_enabled=False,
    ):
        fingerprint = hashlib.sha256(config_text.encode()).hexdigest()
        return cls(
            call=call,
            config_fingerprint=fingerprint,
            seed=int(seed),
            runtime_mode=str(runtime_mode),
            use_torch=bool(use_torch),
            gpu_enabled=bool(gpu_enabled),
        )
