"""Parser and serializer for the textual API configuration format."""

from __future__ import annotations

import collections
import copy
import math
import re

import numpy
import paddle

if __package__:
    from .input_generation.tensor_config import TensorConfig
else:
    from input_generation.tensor_config import TensorConfig


class APIConfig:
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.args = copy.deepcopy(self.args)
        result.kwargs = copy.deepcopy(self.kwargs)
        result.api_name = self.api_name
        return result

    def __init__(self, config):
        config = config.replace("\n", "")
        self.config = config
        self.args = []
        self.kwargs = collections.OrderedDict()

        # 兼容 paddle.Size([...]) 格式：将其替换为 [...]
        def replace_paddle_size(match):
            shape_list = match.group(1)  # 提取 [...] 部分
            return shape_list

        config = re.sub(r"paddle\.Size\(\s*(\[[^\]]*\])\s*\)", replace_paddle_size, config)
        config = config.replace("Tensor(", "TensorConfig(")

        self.api_name, offset = self.get_api(config)

        if self.api_name == "paddle.einsum":
            tmp = config[config.index('"') + 1 :]
            value = tmp[: tmp.index('"')]
            offset = config.index('"') + 1 + tmp.index('"')
            if "equation" in config:
                self.append_kwargs("equation", value)
            else:
                self.append_args(value)

        while True:
            prev_offset = offset
            token, offset = self.get_token(config, offset)
            if offset is None:
                # Check for empty string "" that get_token cannot match
                remaining = config[prev_offset:]
                idx = remaining.find('""')
                if idx >= 0:
                    offset = prev_offset + idx + 2
                    self.append_args("")
                    continue
                return

            is_kwarg = config[offset] == "="
            if is_kwarg:
                key = token
                prev_offset2 = offset + 1
                token, offset = self.get_token(config, prev_offset2)
                # Handle kwarg with empty string value: key=""
                if token is None:
                    remaining = config[prev_offset2:]
                    idx = remaining.find('""')
                    if idx >= 0:
                        offset = prev_offset2 + idx + 2
                        self.append_kwargs(key, "")
                        continue
                    else:
                        return

            value, offset = self.get_one_arg(token, config, offset)

            if offset is None:
                return

            if is_kwarg:
                self.append_kwargs(key, value)
            else:
                self.append_args(value)

    def append_args(self, arg):
        self.args.append(arg)

    def append_kwargs(self, name, arg):
        self.kwargs[name] = arg

    def dump_item_str(self, item):
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
            return str(item)
        elif isinstance(item, paddle.base.core.DataType):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.core.VarDesc.VarType):
            return "VarType(" + str(item)[7:] + ")"
        elif isinstance(item, list):
            result = "list["
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + "]"
            return result
        elif isinstance(item, tuple):
            result = "tuple("
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + ")"
            return result
        elif isinstance(item, slice):
            return "slice(" + str(item.start) + "," + str(item.stop) + "," + str(item.step) + ")"
        elif isinstance(item, complex):
            return (
                "complex("
                + self.dump_item_str(item.real)
                + ","
                + self.dump_item_str(item.imag)
                + ")"
            )
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

    def __str__(self):
        result = self.api_name + "("
        for arg in self.args:
            result += self.dump_item_str(arg) + ", "
        for key, value in self.kwargs.items():
            result += key + "=" + self.dump_item_str(value) + ", "
        result += ")"
        return result

    def __repr__(self):
        return self.__str__()

    # def get_token(self, config, offset):
    #     def is_int(token):
    #         try:
    #             int(token)
    #             return True
    #         except Exception as err:
    #             return False
    #     pattern = r'\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b'
    #     match = re.search(pattern, config[offset:])
    #     if match:
    #         if is_int(match.group()) and config[offset + match.start() + len(match.group())] == ".":
    #             return match.group()+".", offset + match.start() + len(match.group()) + 1
    #         return match.group(), offset + match.start() + len(match.group())
    #     return None, None

    def get_token(self, config, offset):
        def is_int(token):
            try:
                int(token)
                return True
            except Exception:
                return False

        # Modified pattern to handle decimal numbers starting with dot
        pattern = r"\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b|\.[0-9]+"
        match = re.search(pattern, config[offset:])
        if match:
            token = match.group()
            # Handle the case where token starts with dot followed by digits
            if token.startswith(".") and token[1:].isdigit():
                return token, offset + match.start() + len(token)

            if (
                is_int(token)
                and offset + match.start() + len(token) < len(config)
                and config[offset + match.start() + len(token)] == "."
            ):
                return token + ".", offset + match.start() + len(token) + 1
            return token, offset + match.start() + len(token)
        return None, None

    def get_api(self, config):
        return config[0 : config.index("(")], len(config[0 : config.index("(")])

    def _match_parens(self, config, start):
        depth = 0
        for i in range(start, len(config)):
            if config[i] == "(":
                depth += 1
            elif config[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        raise ValueError(f"Unclosed parentheses starting at offset {start}")

    def get_tensor(self, config, offset):
        """Parse TensorConfig(...), including nested kwargs like place=Place(cpu)."""
        start = config.index("(", offset)
        end = self._match_parens(config, start)
        # Slice to the matched span so get_token cannot look past the closing ')'.
        tensor_str = config[start : end + 1]
        args = []
        kwargs = collections.OrderedDict()
        pos = 1

        def skip(p):
            while p < len(tensor_str) and tensor_str[p] in " ,\t\n":
                p += 1
            return p

        while True:
            pos = skip(pos)
            if pos >= len(tensor_str) or tensor_str[pos] == ")":
                break

            # Tensor shapes use bare list syntax, not list[...].
            if tensor_str[pos] == "[":
                value, pos = self.get_list(tensor_str, pos)
                args.append(value)
                continue

            key = None
            token, pos = self.get_token(tensor_str, pos)
            if pos is None:
                break
            if pos < len(tensor_str) and tensor_str[pos] == "=":
                key = token
                pos = skip(pos + 1)
                if pos < len(tensor_str) and tensor_str[pos] == "[":
                    value, pos = self.get_list(tensor_str, pos)
                    kwargs[key] = value
                    continue
                token, pos = self.get_token(tensor_str, pos)
                if pos is None:
                    break

            value, pos = self.get_one_arg(token, tensor_str, pos)
            if pos is None:
                break
            if key is not None:
                kwargs[key] = value
            else:
                args.append(value)

        return TensorConfig(*args, **kwargs), end + 1

    def get_dtype(self, config, offset):
        token, offset = self.get_token(config, offset)
        if hasattr(paddle.framework, "convert_nptype_to_datatype_or_vartype"):
            return paddle.framework.convert_nptype_to_datatype_or_vartype(token), offset
        # fallback for older Paddle versions
        return paddle.pir.core.convert_np_dtype_to_dtype_(token), offset

    def get_place(self, config, offset):
        """Parse Place(gpu:0), Place(cpu), etc."""
        config_slice = config[offset:]
        place_str = config_slice[config_slice.index("(") + 1 : config_slice.index(")")]
        end_offset = offset + config_slice.index(")") + 1
        if place_str == "cpu":
            return paddle.CPUPlace(), end_offset
        elif place_str.startswith("gpu"):
            if ":" in place_str:
                device_id = int(place_str.split(":")[1])
            else:
                device_id = 0
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count > 0:
                device_id = device_id % gpu_count
            return paddle.CUDAPlace(device_id), end_offset
        else:
            return paddle.CPUPlace(), end_offset

    def get_vartype(self, config, offset):
        token, offset = self.get_token(config, offset)
        return paddle.base.framework.convert_np_dtype_to_proto_type(token), offset

    def get_list(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "[":
                tmp = tmp + 1
            if config[i] == "]":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break

        list_str = config[offset : last_index + 1]
        if "TensorConfig" not in list_str:
            list_str = list_str.replace(",", " ")

        offset = 1
        while True:
            token, offset = self.get_token(list_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(token, list_str, offset)

            if offset is None:
                break

            result.append(value)

        return result, last_index + 1

    def get_tuple(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "(":
                tmp = tmp + 1
            if config[i] == ")":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break

        tuple_str = config[offset : last_index + 1]

        tuple_str = tuple_str.replace(",", " , ")

        offset = 1
        while True:
            token, offset = self.get_token(tuple_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(token, tuple_str, offset)

            if offset is None:
                break

            result.append(value)

        return tuple(result), last_index + 1

    def get_slice(self, config, offset):
        config = config[offset:]
        slice_str = config[config.index("(") : config.index(")") + 1]
        return eval("slice" + slice_str), offset + len(slice_str)

    def get_complex(self, config, offset):
        config = config[offset:]
        complex_str = config[config.index("(") : config.index(")") + 1]
        if "nan" in complex_str and complex_str[complex_str.index("nan") - 1] != ".":
            complex_str = complex_str.replace("nan", "float('nan')")
        return eval("complex" + complex_str), offset + len(complex_str)

    def get_numpy_type(self, config, offset):
        config = config[offset:]
        numpy_type_str = config[config.index("(") + 1 : config.index(")")]
        if numpy_type_str == "numpy.bool":
            return numpy.bool_, offset + len(numpy_type_str) + 2
        return eval(numpy_type_str), offset + len(numpy_type_str) + 2

    def get_one_arg(self, token, config, offset):
        if token == "TensorConfig":
            value, offset = self.get_tensor(config, offset - len(token))
        elif token == "Dtype":
            value, offset = self.get_dtype(config, offset)
        elif token == "Place":
            value, offset = self.get_place(config, offset)
        elif token == "VarType":
            value, offset = self.get_vartype(config, offset)
        elif token == "list":
            value, offset = self.get_list(config, offset)
        elif token == "tuple":
            value, offset = self.get_tuple(config, offset)
        elif token == "slice":
            value, offset = self.get_slice(config, offset)
        elif token == "complex":
            value, offset = self.get_complex(config, offset)
        elif token == "type":
            value, offset = self.get_numpy_type(config, offset)
        elif token == "nan":
            value = float("nan")
        elif token is not None and config[offset - len(token) - 1] == '"':
            # fix token is not correct in str with spaces
            next_quote_idx = config.index('"', offset)
            value = config[offset - len(token) : next_quote_idx]
            offset = next_quote_idx
        elif token is None:
            return None, None
        else:
            if token[0] == ".":
                token = "0" + token
            value = eval(token)
        return value, offset


def analyse_configs(config_path):
    with open(config_path) as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        # print(config)
        api_config = APIConfig(config)
        api_configs.append(api_config)
    return api_configs
