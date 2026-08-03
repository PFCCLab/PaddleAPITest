"""张量只读视图。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSpec:
    """供值生成器消费的 TensorConfig 只读视图。"""

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
