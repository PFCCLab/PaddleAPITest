from __future__ import annotations

import copy
import math
from pathlib import Path

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


apis_map = {}
base_size_map = {
    "paddle.Tensor.__nonzero__": 0,
    "paddle.zeros": 1,
    "paddle.ones": 1,
    "paddle.full": 1,
}


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

    # if api_config.api_name in count_map:
    #     if apis_map[api_config.api_name][key] > count_map[api_config.api_name]:
    #         return []
    # else:
    #     if apis_map[api_config.api_name][key] > 5:
    #         return []

    tensor_configs = get_tensor_configs(api_config)
    if len(tensor_configs) == 0:
        return []

    result = []  # 结果列表

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config):
            return []
        # if tensor_config.dtype in ["complex64", "complex128"]:
        #     return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    # 改变每一个tensor的shape，使得每个tensor的大小都接近2^32
    for i in range(len(tensor_configs)):
        for j in range(len(tensor_configs[i].shape)):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            if api_config.api_name in base_size_map:
                base_size = base_size_map[api_config.api_name]
            elif tmp_tensor_configs[i].dtype in [
                "float8",
                "float16",
                "bfloat16",
                "int16",
                "uint16",
                "int8",
                "uint8",
                "bool",
            ]:
                base_size = 101606400
            elif tmp_tensor_configs[i].dtype in ["int32", "uint32", "float32"]:
                base_size = 50803200
            elif tmp_tensor_configs[i].dtype in [
                "float64",
                "double",
                "int64",
                "complex64",
            ]:
                base_size = 20401600
            else:
                base_size = 10200800

            old_dim = tmp_tensor_configs[i].shape[j]
            new_dim = (
                int(
                    base_size
                    / (tensor_numel(tmp_tensor_configs[i]) / tmp_tensor_configs[i].shape[j])
                )
                + 1
            )
            tmp_tensor_configs[i].shape[j] = new_dim
            config_str = str(tmp_api_config)
            if len(config_str) < 1000:
                result.append(config_str)

            valid = True
            tmp_tensor_configs[i].shape[j] = old_dim  # 恢复原来的shape
            for m in range(len(tensor_configs)):
                is_first = True
                for n in range(len(tensor_configs[m].shape)):
                    if tmp_tensor_configs[m].shape[n] == old_dim:
                        tmp_tensor_configs[m].shape[n] = new_dim
                        if is_first:
                            is_first = False
                        else:
                            valid = False
            if valid == False:
                continue
            config_str2 = str(tmp_api_config)
            if len(config_str2) < 1000:
                result.append(config_str2)

    # api_configs中所有的tensor的shape都相同
    if shape_equal:
        for j in range(shape_len):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            for i in range(len(tensor_configs)):
                if api_config.api_name in base_size_map:
                    base_size = base_size_map[api_config.api_name]
                elif tmp_tensor_configs[i].dtype in [
                    "float8",
                    "float16",
                    "bfloat16",
                    "int16",
                    "uint16",
                    "int8",
                    "uint8",
                    "bool",
                ]:
                    base_size = 101606400
                elif tmp_tensor_configs[i].dtype in ["int32", "uint32", "float32"]:
                    base_size = 50803200
                elif tmp_tensor_configs[i].dtype in [
                    "float64",
                    "double",
                    "int64",
                    "complex64",
                ]:
                    base_size = 20401600
                else:
                    base_size = 10200800
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


if __name__ == "__main__":
    INPUT_PATH = Path(
        "/root/paddlejob/workspace/env_run/ningzs/PaddleAPITest/tester/api_config/10_performance/not_pass_top3.txt"
    )
    OUTPUT_PATH = Path(
        "/root/paddlejob/workspace/env_run/ningzs/PaddleAPITest/tester/api_config/10_performance/not_pass_top3_gen.txt"
    )

    api_configs = analyse_configs(INPUT_PATH)
    # count = 0
    with open(OUTPUT_PATH, "w") as f:
        for api_config in tqdm(api_configs):
            # print(api_config)
            try:
                configs = to_big_tensor_config(api_config)
                # print(configs, "\n\n")
                for a in configs:
                    f.write(str(a) + "\n")
            except Exception:
                continue
            # count += 1
            # if (count >= 2):
            #     break

    # 去重
    api_configs = set()
    try:
        with open(OUTPUT_PATH) as f:
            lines = [line.strip() for line in f if line.strip()]
            print(f"Read {len(lines)} api configs from {INPUT_PATH}", flush=True)
            api_configs = set(lines)
    except Exception as err:
        print(f"Error reading {OUTPUT_PATH}: {err}", flush=True)
        exit(0)

    try:
        with open(OUTPUT_PATH, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(api_configs))
    except Exception as err:
        print(f"Error writing {OUTPUT_PATH}: {err}", flush=True)
        exit(0)
    print(f"Write {len(api_configs)} api configs to {OUTPUT_PATH}", flush=True)
