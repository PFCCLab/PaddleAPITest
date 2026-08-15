"""输入生成的路径、规格和逻辑值对象。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputTensorPath:
    """一次 API 调用中 Tensor 的稳定路径。"""

    argument_kind: str
    argument_key: int | str
    item_indices: tuple[int, ...] = ()

    # path 只允许 args/kwargs 两种顶层容器，保证字符串格式可逆。

    def __post_init__(self):
        # 路径是规则和输入值的稳定身份，构造时拒绝不可寻址的位置。
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

    def resolve(self, api_config):
        """按稳定路径读取 APIConfig 中的当前值。"""
        # 路径解析集中在值对象内，规则不需要知道 args/kwargs 的容器细节。
        value = (
            api_config.args[self.argument_key]
            if self.argument_kind == "args"
            else api_config.kwargs[self.argument_key]
        )
        for index in self.item_indices:
            value = value[index]
        return value

    def child(self, index):
        return InputTensorPath(self.argument_kind, self.argument_key, (*self.item_indices, index))

    def __str__(self):
        value = (
            f"args[{self.argument_key}]"
            if self.argument_kind == "args"
            else f"kwargs.{self.argument_key}"
        )
        for index in self.item_indices:
            value += f"[{index}]"
        return value


@dataclass(frozen=True)
class InputTensorSpec:
    """供值生成器消费的 TensorConfig 只读视图。"""

    shape: tuple[int, ...]
    dtype: str
    place: str | None
    is_contiguous: bool
    strides: tuple[int, ...] | None

    # spec 不携带可变 Tensor 或随机状态，适合传入纯值域生成器。

    @classmethod
    def from_tensor_config(cls, tensor_config):
        # spec 是只读快照，避免生成器意外修改共享 TensorConfig。
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


@dataclass(frozen=True)
class InputValue:
    """保存一次输入路径对应的逻辑值及其来源 backend。"""

    path: InputTensorPath
    generated_value: object
    backend_name: str

    # generated_value 可能是 NumPy 数组或原生 Tensor，取决于 backend。


_INPUT_VALUES_ATTR = "_input_generation_values"
_INPUT_VALUE_BY_TENSOR_ID_ATTR = "_input_generation_value_by_tensor_id"


def input_tensor_config_at(api_config, path: InputTensorPath):
    """读取路径对应的 TensorConfig，供规则提交和生命周期管理共用。"""
    # 统一入口确保规则提交和物化读取使用同一寻址语义。
    return path.resolve(api_config)


def attach_input_values(api_config, input_values):
    # 同时保留有序 value 和对象 id 索引，避免物化阶段重复遍历嵌套参数。
    input_values = tuple(input_values)
    input_value_by_tensor_id = {
        id(input_tensor_config_at(api_config, input_value.path)): input_value
        for input_value in input_values
    }
    setattr(api_config, _INPUT_VALUES_ATTR, input_values)
    setattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, input_value_by_tensor_id)
    return input_values


def find_input_value(api_config, tensor_config):
    # object id 索引避免每次物化都递归扫描参数树。
    input_values = getattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, None)
    return None if input_values is None else input_values.get(id(tensor_config))


def read_input_value(api_config, tensor_config):
    input_value = find_input_value(api_config, tensor_config)
    return input_value.generated_value if input_value is not None else None


def read_input_value_backend(api_config, tensor_config):
    input_value = find_input_value(api_config, tensor_config)
    return input_value.backend_name if input_value is not None else None


def clear_input_value(api_config, tensor_config):
    # 清理必须同步移除路径序列和对象索引，防止留下半失效缓存。
    input_values = getattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, None)
    if input_values is not None:
        input_values.pop(id(tensor_config), None)
    ordered_values = getattr(api_config, _INPUT_VALUES_ATTR, None)
    if ordered_values is not None:
        setattr(
            api_config,
            _INPUT_VALUES_ATTR,
            tuple(
                item
                for item in ordered_values
                if input_tensor_config_at(api_config, item.path) is not tensor_config
            ),
        )


__all__ = [
    "InputTensorPath",
    "InputTensorSpec",
    "InputValue",
    "attach_input_values",
    "clear_input_value",
    "find_input_value",
    "input_tensor_config_at",
    "read_input_value",
    "read_input_value_backend",
]
