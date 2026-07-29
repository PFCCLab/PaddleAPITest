from __future__ import annotations

from dataclasses import dataclass

from .model import ArgPath


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
    payload = payload_for_tensor(api_config, tensor_config)
    if payload is not None:
        return payload.value
    return tensor_config.numpy_tensor


def write_logical_value(api_config, tensor_config, value, update_metadata=True):
    payload = payload_for_tensor(api_config, tensor_config)
    if payload is not None:
        payload = TensorPayload(payload.path, value, update_metadata=update_metadata)
        payload_by_tensor_id = getattr(api_config, _PAYLOAD_BY_TENSOR_ID_ATTR, None)
        if payload_by_tensor_id is not None:
            payload_by_tensor_id[id(tensor_config)] = payload
    tensor_config.numpy_tensor = value
    return payload


def clear_logical_value(api_config, tensor_config):
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
