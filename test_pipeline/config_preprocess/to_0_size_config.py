from __future__ import annotations

import argparse
import copy
import heapq
import math
import os
import tempfile

import numpy
import paddle
from test_pipeline.config_preprocess.config_lines import (
    matching_close as _matching_close,
)
from test_pipeline.config_preprocess.config_lines import (
    split_top_level_calls as _split_top_level_calls,
)
from tester.api_config.parser import APIConfig
from tester.input_generation.tensor_config import TensorConfig
from tqdm import tqdm


def is_0_size_tensor(tensor_config):
    return any(i == 0 for i in tensor_config.shape)


def is_0D_tensor(tensor_config):
    return len(tensor_config.shape) == 0


def get_tensor_configs(api_config):
    # 只展开参数容器，不改变 parser 对嵌套值的既有边界。
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


def iter_api_configs(config_path, on_reject=None):
    """解析配置，支持一行中连续的多个顶层 API 调用。"""
    # 未提供回调时保持严格失败语义，并附加输入文件与行号。
    # 提供回调时记录坏配置后继续，yield 始终位于异常处理范围之外。
    with open(config_path, encoding="utf-8") as config_file:
        for line_number, raw_line in enumerate(config_file, start=1):
            config = raw_line.strip()
            if not config:
                continue
            if not config.startswith("paddle."):
                error = ValueError("配置不是 paddle.* 顶层调用")
                if on_reject is None:
                    raise ValueError(
                        f"{config_path}:{line_number}: {error}"
                    ) from error
                on_reject(config_path, line_number, config, error)
                continue
            try:
                calls = _split_top_level_calls(config)
            except ValueError as error:
                if on_reject is None:
                    raise ValueError(f"{config_path}:{line_number}: {error}") from error
                on_reject(config_path, line_number, config, error)
                continue
            for call in calls:
                # APIConfig 失败也必须保留文件和行号，便于定位截断配置。
                try:
                    api_config = APIConfig(call)
                except Exception as error:
                    parse_error = ValueError(
                        f"无法解析拆分后的调用 {call[:80]!r}: "
                        f"{type(error).__name__}: {error}"
                    )
                    if on_reject is None:
                        raise ValueError(
                            f"{config_path}:{line_number}: {parse_error}"
                        ) from error
                    on_reject(config_path, line_number, call, parse_error)
                    continue
                yield api_config


def _find_tensor_shape_spans(config):
    """记录规范化配置中每个 Tensor shape 列表的字符区间。"""
    # 偏移区间只覆盖 shape 列表，保留 Tensor 外层语法和 dtype 文本。
    spans = []
    search_from = 0
    marker = "Tensor("
    while True:
        tensor_start = config.find(marker, search_from)
        if tensor_start < 0:
            return spans
        shape_start = tensor_start + len(marker)
        size_prefix = "paddle.Size("
        if config.startswith(size_prefix, shape_start):
            shape_start += len(size_prefix)
        while shape_start < len(config) and config[shape_start].isspace():
            shape_start += 1
        if shape_start >= len(config) or config[shape_start] != "[":
            search_from = tensor_start + len(marker)
            continue
        shape_end = _matching_close(config, shape_start, "[", "]")
        if shape_end is None:
            return []
        spans.append((shape_start, shape_end + 1))
        search_from = shape_end + 1


def _selected_positions(tensor_configs):
    """返回全部 Tensor 维度位置，保持 0-size 覆盖范围不变。"""
    return [
        (index, axis)
        for index, tensor in enumerate(tensor_configs)
        for axis in range(len(tensor.shape))
    ]


def _replace_shape_spans(config, spans, replacements):
    """按位置替换 shape，避免为每个候选重新解析 APIConfig。"""
    # 按原始区间拼接，避免正则替换同形状 Tensor 时误伤其他位置。
    chunks = []
    cursor = 0
    for index, (start, end) in enumerate(spans):
        chunks.append(config[cursor:start])
        chunks.append(str(replacements.get(index, config[start:end])))
        cursor = end
    chunks.append(config[cursor:])
    return "".join(chunks)


def _object_variants(api_config):
    """字符串定位失败时的兼容路径；同一配置只深拷贝一次。"""
    # 仅定位失败时使用对象回退，且整个配置生命周期只复制一次。
    work_api_config = copy.deepcopy(api_config)
    work_tensors = get_tensor_configs(work_api_config)
    for tensor_index, axis in _selected_positions(work_tensors):
        tensor = work_tensors[tensor_index]
        old_value = tensor.shape[axis]
        tensor.shape[axis] = 0
        try:
            yield str(work_api_config)
        finally:
            tensor.shape[axis] = old_value
    if all(len(tensor.shape) == len(work_tensors[0].shape) for tensor in work_tensors):
        for axis in range(len(work_tensors[0].shape)):
            old_values = [tensor.shape[axis] for tensor in work_tensors]
            try:
                for tensor in work_tensors:
                    tensor.shape[axis] = 0
                yield str(work_api_config)
            finally:
                for tensor, old_value in zip(work_tensors, old_values, strict=True):
                    tensor.shape[axis] = old_value


def to_0_size_config(api_config):
    # 同一 API/结构最多保留少量重复样本，这是原有稳定性探测协议。
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

    if len(tensor_configs) == 0:
        return []

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config) or is_0D_tensor(tensor_config):
            return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    # 正常路径复用 parser 的规范格式，保证与历史输出兼容。
    canonical_config = str(api_config)
    # 运行时 callable 的 repr 不能被 parser 重新执行，此时保留原始可解析文本。
    runtime_repr_markers = ("<function ", "<method ", "<built-in ", "<slot wrapper ", " at 0x")
    if any(marker in canonical_config for marker in runtime_repr_markers):
        canonical_config = api_config.config
    spans = _find_tensor_shape_spans(canonical_config)
    if len(spans) != len(tensor_configs):
        yield from _object_variants(api_config)
        return

    # 每个变体只复制字符串，不复制整棵 APIConfig 对象树。
    for tensor_index, axis in _selected_positions(tensor_configs):
        shape = list(tensor_configs[tensor_index].shape)
        shape[axis] = 0
        yield _replace_shape_spans(
            canonical_config,
            spans,
            {tensor_index: shape},
        )

    if shape_equal:
        for axis in range(shape_len):
            replacements = {}
            for index, tensor in enumerate(tensor_configs):
                shape = list(tensor.shape)
                shape[axis] = 0
                replacements[index] = shape
            yield _replace_shape_spans(canonical_config, spans, replacements)


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
    # elif callable(item):
    #     name = getattr(item, "__name__", None) or getattr(item, "__qualname__", None)
    #     if name:
    #         return "callable(" + name + ")"
    #     return "callable(unknown)"
    else:
        return str(item)


def config_key(api_config):
    result = ""
    for arg in api_config.args:
        result = result + dump_item_str(arg) + ", "

    for key, value in api_config.kwargs.items():
        result = result + key + "=" + dump_item_str(value) + ", "

    return result


CHUNK_BYTES = 64 * 1024 * 1024


def _flush_chunk(lines, chunk_dir, chunk_index):
    """将有限大小的去重块排序落盘，避免长配置常驻内存。"""
    # 块边界按字节计算，避免单条超长配置突破内存预算。
    if not lines:
        return None
    path = os.path.join(chunk_dir, f"chunk_{chunk_index:06d}.txt")
    with open(path, "w", encoding="utf-8") as output_file:
        for line in sorted(lines):
            output_file.write(line + "\n")
    return path


def _merge_chunks(chunk_paths, output_path):
    """归并已排序分块并再次去重，输出顺序与 dedup_config 保持一致。"""
    # 归并输入均已排序，因此只需比较相邻行即可完成全局去重。
    streams = [open(path, encoding="utf-8") for path in chunk_paths]
    count = 0
    previous = None
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            for line in heapq.merge(*(stream for stream in streams)):
                if line == previous:
                    continue
                output_file.write(line)
                previous = line
                count += 1
    finally:
        for stream in streams:
            stream.close()
    return count


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
                int(base_size / (tmp_tensor_configs[i].numel() / tmp_tensor_configs[i].shape[j]))
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
                        base_size / (tmp_tensor_configs[0].numel() / tmp_tensor_configs[0].shape[j])
                    )
                    + 1
                )
            config_str = str(tmp_api_config)
            if len(config_str) < 1000:
                result.append(config_str)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 API 配置转换为 0-size tensor 变体，用于边界测试。",
    )
    parser.add_argument(
        "-i", "--inputs", nargs="+", required=True, help="输入配置文件路径（可指定多个）"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="api_config_0_size.txt",
        help="输出文件路径（默认：当前目录下 api_config_0_size.txt）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="扫描完成后如有无法解析的配置则返回非零状态",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.dirname(args.output) or "."
    reject_path = f"{args.output}.unparsed.txt"
    rejected_count = [0]
    with open(reject_path, "w", encoding="utf-8") as reject_file:

        def record_reject(config_path, line_number, config, error):
            rejected_count[0] += 1
            reject_file.write(
                f"{config_path}:{line_number}\t{error}\t{config}\n"
            )

        with tempfile.TemporaryDirectory(prefix=".0size_chunks.", dir=output_dir) as chunk_dir:
            # 临时块位于输出目录，保证大文件处理不依赖系统 /tmp 空间。
            chunk_paths = []
            chunk_lines = set()
            chunk_bytes = 0
            chunk_index = 0
            for input_file in args.inputs:
                print(f"处理: {input_file}")
                for api_config in tqdm(
                    iter_api_configs(input_file, on_reject=record_reject)
                ):
                    # 逐个变体进入有限大小的块集合，避免累计完整输出。
                    for variant in to_0_size_config(api_config):
                        if variant in chunk_lines:
                            continue
                        chunk_lines.add(variant)
                        chunk_bytes += len(variant) + 1
                        if chunk_bytes >= CHUNK_BYTES:
                            path = _flush_chunk(chunk_lines, chunk_dir, chunk_index)
                            chunk_paths.append(path)
                            chunk_index += 1
                            chunk_lines.clear()
                            chunk_bytes = 0
            path = _flush_chunk(chunk_lines, chunk_dir, chunk_index)
            if path is not None:
                chunk_paths.append(path)
            unique_count = _merge_chunks(chunk_paths, args.output)

    print(f"输出: {args.output}，共 {unique_count} 行")
    if rejected_count[0]:
        print(f"警告: {rejected_count[0]} 条配置无法解析，详见 {reject_path}")
        if args.strict:
            raise SystemExit(1)
    else:
        os.remove(reject_path)
