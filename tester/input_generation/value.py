from __future__ import annotations

from dataclasses import dataclass

from .tensor_path import InputTensorPath

# InputValue 是规则事务提交后的记录，不代表框架已经完成 Tensor 物化。


@dataclass(frozen=True)
class InputValue:
    """保存一次输入路径对应的逻辑值及其来源 backend。"""

    path: InputTensorPath
    generated_value: object
    backend_name: str


_INPUT_VALUES_ATTR = "_input_generation_values"
_INPUT_VALUE_BY_TENSOR_ID_ATTR = "_input_generation_value_by_tensor_id"


def _tensor_value_at_path(api_config, path: InputTensorPath):
    if path.argument_kind == "args":
        value = api_config.args[path.argument_key]
    else:
        value = api_config.kwargs[path.argument_key]
    for index in path.item_indices:
        value = value[index]
    return value


def attach_input_values(api_config, input_values):
    # 同时保留有序 value 和对象 id 索引，便于顺序遍历和快速查找。
    # 路径用于稳定提交，对象 id 用于 TensorConfig 在物化阶段进行常数时间查询。
    input_values = tuple(input_values)
    input_value_by_tensor_id = {
        id(_tensor_value_at_path(api_config, input_value.path)): input_value
        for input_value in input_values
    }
    setattr(api_config, _INPUT_VALUES_ATTR, input_values)
    setattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, input_value_by_tensor_id)
    return input_values


def find_input_value(api_config, tensor_config):
    input_value_by_tensor_id = getattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, None)
    if input_value_by_tensor_id is None:
        return None
    return input_value_by_tensor_id.get(id(tensor_config))


def read_input_value(api_config, tensor_config):
    # 未挂载事务结果时回退到 TensorConfig 缓存，兼容物化后的局部更新。
    input_value = find_input_value(api_config, tensor_config)
    if input_value is not None:
        return input_value.generated_value
    return tensor_config.input_value


def read_input_value_backend(api_config, tensor_config):
    input_value = find_input_value(api_config, tensor_config)
    if input_value is not None:
        return input_value.backend_name
    return tensor_config.input_value_backend or "numpy"


def _detect_input_value_backend(value):
    # 这里只识别可能跨框架零拷贝的原生 Tensor，其余值统一按 NumPy 语义处理。
    module = value.__class__.__module__.split(".", 1)[0]
    if module in {"torch", "paddle"}:
        return module
    return "numpy"


def write_input_value(api_config, tensor_config, new_value):
    # 已提交记录和 TensorConfig 缓存必须同步更新，避免后续框架读取到不同逻辑值。
    current_input_value = find_input_value(api_config, tensor_config)
    input_backend = (
        current_input_value.backend_name
        if current_input_value is not None
        else _detect_input_value_backend(new_value)
    )
    if current_input_value is not None:
        updated_input_value = InputValue(
            current_input_value.path,
            new_value,
            backend_name=input_backend,
        )
        input_value_by_tensor_id = getattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, None)
        if input_value_by_tensor_id is not None:
            input_value_by_tensor_id[id(tensor_config)] = updated_input_value
        input_values = getattr(api_config, _INPUT_VALUES_ATTR, None)
        if input_values is not None:
            updated_input_values = (
                updated_input_value if item.path == current_input_value.path else item
                for item in input_values
            )
            setattr(api_config, _INPUT_VALUES_ATTR, tuple(updated_input_values))
    tensor_config.input_value = new_value
    tensor_config.input_value_backend = input_backend
    return new_value


def clear_input_value(api_config, tensor_config):
    # 清理同时覆盖路径序列、对象索引和 TensorConfig 缓存三个存储位置。
    input_value = find_input_value(api_config, tensor_config)
    input_value_by_tensor_id = getattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, None)
    if input_value_by_tensor_id is not None:
        input_value_by_tensor_id.pop(id(tensor_config), None)
    if input_value is not None:
        input_values = getattr(api_config, _INPUT_VALUES_ATTR, None)
        if input_values is not None:
            setattr(
                api_config,
                _INPUT_VALUES_ATTR,
                tuple(item for item in input_values if item.path != input_value.path),
            )
    tensor_config.input_value = None
    tensor_config.input_value_backend = None
