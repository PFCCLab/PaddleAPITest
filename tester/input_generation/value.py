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
    return path.resolve(api_config)


def input_tensor_config_at(api_config, path: InputTensorPath):
    """读取路径对应的 TensorConfig，供规则提交和生命周期管理共用。"""
    # TensorConfig 的查找只依赖稳定路径，不读取或修改物化缓存。
    return path.resolve(api_config)


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
    input_value = find_input_value(api_config, tensor_config)
    return input_value.generated_value if input_value is not None else None


def read_input_value_backend(api_config, tensor_config):
    input_value = find_input_value(api_config, tensor_config)
    return input_value.backend_name if input_value is not None else None


def clear_input_value(api_config, tensor_config):
    # 清理同时覆盖路径序列和对象索引，不能留下只可按其中一种方式访问的值。
    input_value_by_tensor_id = getattr(api_config, _INPUT_VALUE_BY_TENSOR_ID_ATTR, None)
    if input_value_by_tensor_id is not None:
        input_value_by_tensor_id.pop(id(tensor_config), None)
    input_values = getattr(api_config, _INPUT_VALUES_ATTR, None)
    if input_values is not None:
        # 同一个 TensorConfig 可能出现在多个参数路径，释放时必须清掉全部别名记录。
        setattr(
            api_config,
            _INPUT_VALUES_ATTR,
            tuple(
                item
                for item in input_values
                if _tensor_value_at_path(api_config, item.path) is not tensor_config
            ),
        )
