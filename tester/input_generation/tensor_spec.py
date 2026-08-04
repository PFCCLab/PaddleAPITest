"""张量只读视图。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputTensorSpec:
    """供值生成器消费的 TensorConfig 只读视图。"""

    # 这里只保留值域生成需要的字段，避免规则直接触碰可变 TensorConfig。
    shape: tuple[int, ...]
    dtype: str
    place: str | None
    is_contiguous: bool
    strides: tuple[int, ...] | None

    @classmethod
    def from_tensor_config(cls, tensor_config):
        return cls(
            shape=tuple(int(dim) for dim in tensor_config.shape),
            dtype=str(tensor_config.dtype),
            place=str(tensor_config.place) if tensor_config.place is not None else None,
            is_contiguous=bool(tensor_config.is_contiguous),
            strides=(
                tuple(int(stride) for stride in tensor_config.strides)
                if tensor_config.strides is not None
                else None
            ),
        )
