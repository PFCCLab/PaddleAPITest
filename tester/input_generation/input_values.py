from __future__ import annotations

from dataclasses import dataclass

from .input_path import InputPath


@dataclass(frozen=True)
class InputValue:
    path: InputPath
    value: object
    update_metadata: bool = True
    backend: str = "numpy"
    declared_dtype: str | None = None
    storage_dtype: str | None = None


_VALUES_ATTR = "_input_generation_values"
_VALUE_BY_TENSOR_ID_ATTR = "_input_generation_value_by_tensor_id"


def _tensor_value_at_path(api_config, path: InputPath):
    if path.root == "args":
        value = api_config.args[path.key]
    else:
        value = api_config.kwargs[path.key]
    for index in path.indices:
        value = value[index]
    return value


def attach_values(api_config, values):
    # 同时保留有序 value 和对象 id 索引，便于顺序遍历和快速查找。
    values = tuple(values)
    value_by_tensor_id = {}
    for value in values:
        value_by_tensor_id[id(_tensor_value_at_path(api_config, value.path))] = value
    setattr(api_config, _VALUES_ATTR, values)
    setattr(api_config, _VALUE_BY_TENSOR_ID_ATTR, value_by_tensor_id)
    return values


def values_for(api_config):
    return getattr(api_config, _VALUES_ATTR, None)


def value_for_tensor(api_config, tensor_config):
    value_by_tensor_id = getattr(api_config, _VALUE_BY_TENSOR_ID_ATTR, None)
    if value_by_tensor_id is None:
        return None
    return value_by_tensor_id.get(id(tensor_config))


def input_value(api_config, tensor_config):
    value = value_for_tensor(api_config, tensor_config)
    if value is not None:
        return value.value
    return tensor_config.numpy_tensor


def input_value_backend(api_config, tensor_config):
    value = value_for_tensor(api_config, tensor_config)
    if value is not None:
        return value.backend
    return "numpy"


def write_input_value(api_config, tensor_config, new_value, update_metadata=True):
    existing_value = value_for_tensor(api_config, tensor_config)
    if existing_value is not None:
        updated_value = InputValue(
            existing_value.path,
            new_value,
            update_metadata=update_metadata,
            backend=existing_value.backend,
            declared_dtype=existing_value.declared_dtype,
            storage_dtype=existing_value.storage_dtype,
        )
        value_by_tensor_id = getattr(api_config, _VALUE_BY_TENSOR_ID_ATTR, None)
        if value_by_tensor_id is not None:
            value_by_tensor_id[id(tensor_config)] = updated_value
        values = getattr(api_config, _VALUES_ATTR, None)
        if values is not None:
            setattr(
                api_config,
                _VALUES_ATTR,
                tuple(
                    updated_value if item.path == existing_value.path else item for item in values
                ),
            )
    tensor_config.numpy_tensor = new_value
    return new_value


def clear_input_value(api_config, tensor_config):
    value = value_for_tensor(api_config, tensor_config)
    value_by_tensor_id = getattr(api_config, _VALUE_BY_TENSOR_ID_ATTR, None)
    if value_by_tensor_id is not None:
        value_by_tensor_id.pop(id(tensor_config), None)
    if value is not None:
        values = getattr(api_config, _VALUES_ATTR, None)
        if values is not None:
            setattr(
                api_config,
                _VALUES_ATTR,
                tuple(item for item in values if item.path != value.path),
            )
