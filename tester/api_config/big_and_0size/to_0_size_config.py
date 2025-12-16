from __future__ import annotations

import copy
import math

import numpy
import paddle
from config_analyzer import TensorConfig, analyse_configs
from tqdm import tqdm


def is_0_size_tensor(tensor_config):
    return any(i == 0 for i in tensor_config.shape)


def is_0D_tensor(tensor_config):
    return len(tensor_config.shape) == 0


def tensor_numel(tensor_config):
    numel = 1
    for i in tensor_config.shape:
        numel = numel * i
    return numel


def get_tensor_configs(api_config):
    tensor_configs = []
    for arg_config in api_config.args:
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])

    for _key, arg_config in api_config.kwargs.items():
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
    return tensor_configs


def to_0_size_config(api_config):
    if api_config.api_name in [
        "paddle.Tensor.__getitem__",
        "paddle.Tensor.__setitem__",
    ]:
        return []
    if api_config.api_name not in apis_map:
        apis_map[api_config.api_name] = {}

    key = config_key(api_config)

    if key not in apis_map[api_config.api_name]:
        apis_map[api_config.api_name][key] = 1
    else:
        apis_map[api_config.api_name][key] += 1

    if apis_map[api_config.api_name][key] > 2:
        return []

    result = []
    tensor_configs = get_tensor_configs(api_config)

    if len(tensor_configs) == 0:
        return []

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config) or is_0D_tensor(tensor_config):
            return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    for i in range(len(tensor_configs)):
        for j in range(len(tensor_configs[i].shape)):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            tmp_tensor_configs[i].shape[j] = 0
            result.append(str(tmp_api_config))

    if shape_equal:
        for j in range(shape_len):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            for i in range(len(tensor_configs)):
                tmp_tensor_configs[i].shape[j] = 0
            result.append(str(tmp_api_config))
    return result


apis_map = {}


def dump_item_str(item):
    type_mapping = {
        numpy.int16: int,
        numpy.int32: int,
        numpy.int64: int,
        numpy.float16: float,
        numpy.float32: float,
        numpy.float64: float,
        numpy.integer: int,
        numpy.floating: float,
        numpy.bool_: bool,
        numpy.complexfloating: complex,
        numpy.str_: str,
        numpy.bytes_: bytes,
        # numpy.unicode_: str,
    }
    for numpy_type, builtin_type in type_mapping.items():
        if isinstance(item, numpy_type):
            item = builtin_type(item)
            break

    if isinstance(item, TensorConfig):
        return "Tensor(" + str(len(item.shape)) + ")"
    elif isinstance(item, paddle.base.core.DataType):
        return "Dtype(" + str(item)[7:] + ")"
    elif isinstance(item, paddle.base.core.VarDesc.VarType):
        return "VarType(" + str(item)[7:] + ")"
    elif isinstance(item, list):
        result = "list["
        for sub_item in item:
            tmp = dump_item_str(sub_item)
            if tmp == "":
                return ""
            result = result + tmp + ","
        result = result + "]"
        return result
    elif isinstance(item, tuple):
        result = "tuple("
        for sub_item in item:
            tmp = dump_item_str(sub_item)
            if tmp == "":
                return ""
            result = result + tmp + ","
        result = result + ")"
        return result
    elif isinstance(item, slice):
        return "slice(" + str(item.start) + "," + str(item.stop) + "," + str(item.step) + ")"
    elif isinstance(item, complex):
        return "complex(" + dump_item_str(item.real) + "," + dump_item_str(item.imag) + ")"
    elif item is None:
        return "None"
    elif isinstance(item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)):
        return ""
    elif item == math.inf:
        return "math.inf"
    elif item == -math.inf:
        return "-math.inf"
    elif item == math.nan:
        return "math.nan"
    elif item == -math.nan:
        return "-math.nan"
    elif isinstance(item, (bool, int, float)):
        return str(item)
    elif isinstance(item, str):
        return '"' + item + '"'
    elif isinstance(item, type):
        return "type(" + str(item)[str(item).index("'") + 1 : str(item).rindex("'")] + ")"
    else:
        return str(item)


def config_key(api_config):
    result = ""
    for arg in api_config.args:
        result = result + dump_item_str(arg) + ", "

    for key, value in api_config.kwargs.items():
        result = result + key + "=" + dump_item_str(value) + ", "

    return result


def to_big_tensor_config(api_config):
    if api_config.api_name not in apis_map:
        apis_map[api_config.api_name] = {}

    key = config_key(api_config)

    if key not in apis_map[api_config.api_name]:
        apis_map[api_config.api_name][key] = 1
    else:
        apis_map[api_config.api_name][key] += 1

    if apis_map[api_config.api_name][key] > 5:
        return []

    tensor_configs = get_tensor_configs(api_config)

    result = []

    if len(tensor_configs) == 0:
        return []

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config) or is_0D_tensor(tensor_config):
            return []
        if tensor_config.dtype in ["complex64", "complex128"]:
            return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    for i in range(len(tensor_configs)):
        for j in range(len(tensor_configs[i].shape)):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            if tmp_tensor_configs[i].dtype in [
                "float8",
                "float16",
                "bfloat16",
                "int16",
                "uint16",
                "int8",
                "uint8",
            ]:
                base_size = 4294967296
            elif tmp_tensor_configs[i].dtype in ["float64"]:
                base_size = 4294967296
                for k in range(len(tmp_tensor_configs)):
                    if tmp_tensor_configs[k].dtype in ["float64"]:
                        tmp_tensor_configs[k].dtype = "float16"
            else:
                base_size = 2281701378
            tmp_tensor_configs[i].shape[j] = (
                int(
                    base_size
                    / (tensor_numel(tmp_tensor_configs[i]) / tmp_tensor_configs[i].shape[j])
                )
                + 1
            )
            config_str = str(tmp_api_config)
            if len(config_str) < 1000:
                result.append(config_str)

    if shape_equal:
        for j in range(shape_len):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            for i in range(len(tensor_configs)):
                if tmp_tensor_configs[i].dtype in [
                    "float8",
                    "float16",
                    "bfloat16",
                    "int16",
                    "uint16",
                    "int8",
                    "uint8",
                ]:
                    base_size = 4294967296
                elif tmp_tensor_configs[i].dtype in ["float64"]:
                    base_size = 4294967296
                    tmp_tensor_configs[i].dtype = "float16"
                else:
                    base_size = 2281701378
                tmp_tensor_configs[i].shape[j] = (
                    int(
                        base_size
                        / (tensor_numel(tmp_tensor_configs[0]) / tmp_tensor_configs[0].shape[j])
                    )
                    + 1
                )
            config_str = str(tmp_api_config)
            if len(config_str) < 1000:
                result.append(config_str)
    return result


# if __name__ == '__main__':
#     config_0_size = set()
#     api_configs = analyse_configs("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log_acc/api_config_pass.txt")

#     with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/0sizetensor2.txt", "w") as f:
#         for api_config in tqdm(api_configs):
#             # print(api_config.config)
#             # config_0_size = config_0_size.union(set(to_0_size_config(api_config)))


#             for api_config in to_0_size_config(api_config):
#                 f.write(str(api_config)+"\n")
#         f.close()

if __name__ == "__main__":
    config_big_tensor = set()
    api_configs = analyse_configs(
        "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log_accuracy_all/api_config_pass2.txt"
    )
    for api_config in tqdm(api_configs):
        # print(api_config.config)
        config_big_tensor = config_big_tensor.union(set(to_big_tensor_config(api_config)))
    config_big_tensor = set(config_big_tensor)
    with open(
        "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log_accuracy_all/bigtensor_accuracy3.txt",
        "w",
    ) as f:
        for api_config in config_big_tensor:
            f.write(str(api_config) + "\n")
        f.close()
