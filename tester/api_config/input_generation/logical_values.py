from __future__ import annotations

from dataclasses import dataclass

from .case_model import ArgPath


@dataclass(frozen=True)
class TensorPayload:
    path: ArgPath
    value: object
    update_metadata: bool = True


_PAYLOADS_ATTR = "_input_generation_payloads"
_PAYLOAD_BY_TENSOR_ID_ATTR = "_input_generation_payload_by_tensor_id"


def _tensor_value_at_path(api_config, path: ArgPath):
    if path.root == "args":
        value = api_config.args[path.key]
    else:
        value = api_config.kwargs[path.key]
    for index in path.indices:
        value = value[index]
    return value


def attach_payloads(api_config, payloads):
    # 同时保留有序 payload 和对象 id 索引：规则要顺序，物化要查找速度。
    payloads = tuple(payloads)
    payload_by_tensor_id = {}
    for payload in payloads:
        payload_by_tensor_id[id(_tensor_value_at_path(api_config, payload.path))] = payload
    setattr(api_config, _PAYLOADS_ATTR, payloads)
    setattr(api_config, _PAYLOAD_BY_TENSOR_ID_ATTR, payload_by_tensor_id)
    return payloads


def payloads_for(api_config):
    return getattr(api_config, _PAYLOADS_ATTR, None)


def payload_for_tensor(api_config, tensor_config):
    payload_by_tensor_id = getattr(api_config, _PAYLOAD_BY_TENSOR_ID_ATTR, None)
    if payload_by_tensor_id is None:
        return None
    return payload_by_tensor_id.get(id(tensor_config))


def logical_value(api_config, tensor_config):
    # v2 生成后以 payload 为准；numpy_tensor 只作为历史回退。
    payload = payload_for_tensor(api_config, tensor_config)
    if payload is not None:
        return payload.value
    return tensor_config.numpy_tensor


def write_logical_value(api_config, tensor_config, value, update_metadata=True):
    # 同步 payload 与 TensorConfig，兼容混合的 legacy/v2 调用方。
    payload = payload_for_tensor(api_config, tensor_config)
    if payload is not None:
        old_payload = payload
        payload = TensorPayload(old_payload.path, value, update_metadata=update_metadata)
        payload_by_tensor_id = getattr(api_config, _PAYLOAD_BY_TENSOR_ID_ATTR, None)
        if payload_by_tensor_id is not None:
            payload_by_tensor_id[id(tensor_config)] = payload
        payloads = getattr(api_config, _PAYLOADS_ATTR, None)
        if payloads is not None:
            setattr(
                api_config,
                _PAYLOADS_ATTR,
                tuple(payload if item.path == old_payload.path else item for item in payloads),
            )
    tensor_config.numpy_tensor = value
    return payload


def clear_logical_value(api_config, tensor_config):
    # 旧调用方可能不带 payload 清理，这里按最佳努力处理。
    payload = payload_for_tensor(api_config, tensor_config)
    payload_by_tensor_id = getattr(api_config, _PAYLOAD_BY_TENSOR_ID_ATTR, None)
    if payload_by_tensor_id is not None:
        payload_by_tensor_id.pop(id(tensor_config), None)
    if payload is not None:
        payloads = getattr(api_config, _PAYLOADS_ATTR, None)
        if payloads is not None:
            setattr(
                api_config,
                _PAYLOADS_ATTR,
                tuple(item for item in payloads if item.path != payload.path),
            )
