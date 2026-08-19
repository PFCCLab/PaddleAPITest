from __future__ import annotations

import argparse
import copy
import heapq
import math
import os
import tempfile
from typing import NamedTuple

import numpy
import paddle
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


def _matching_close(text, start, opening, closing):
    """返回嵌套括号的结束位置，字符串内容中的括号不参与配对。"""
    # 需要跳过字符串中的括号，否则 callable 或字符串参数会破坏定位。
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top_level_calls(text):
    """将一行中连续的顶层 paddle.* 调用无损拆分。"""
    calls = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if not text.startswith("paddle.", cursor):
            raise ValueError("顶层调用结束后存在无法识别的内容")

        opening = text.find("(", cursor + len("paddle."))
        next_call = text.find("paddle.", cursor + len("paddle."))
        if opening < 0 or (next_call >= 0 and next_call < opening):
            raise ValueError("顶层 paddle.* 调用缺少左括号")
        closing = _matching_close(text, opening, "(", ")")
        if closing is None:
            raise ValueError("顶层 paddle.* 调用括号不匹配")
        calls.append(text[cursor : closing + 1])
        cursor = closing + 1
    return calls


def iter_api_configs(config_path):
    """逐行解析配置，并拆分粘连的顶层 API 调用。"""
    with open(config_path, encoding="utf-8") as config_file:
        for raw_line in config_file:
            config = raw_line.strip()
            if config and config.startswith("paddle."):
                for call in _split_top_level_calls(config):
                    yield APIConfig(call)


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


def _selected_positions(tensor_configs, collapsed):
    """返回逐位置枚举的 Tensor 维度位置，collapsed 内的下标交由分档取样覆盖。"""
    return [
        (index, axis)
        for index, tensor in enumerate(tensor_configs)
        if index not in collapsed
        for axis in range(len(tensor.shape))
    ]


# 同质容器的逐位置枚举产出 O(N²) 文本，而这 N 个变体走的 kernel 分档完全相同。
# 阈值取 256 使历史基线（同质列表最大 216 元素）的输出保持不变。
HOMOGENEOUS_CONTAINER_LIMIT = 256


class FoldSpec(NamedTuple):
    """单个 API 的折叠规格。取样表只在推导它的那个 kernel 上成立，故按 API 隔离。"""

    # (容器长度, 置 0 元素个数)，用于命中 kernel 的输入个数分档边界。
    lengths: tuple[tuple[int, int], ...]


# concat 前向按有效输入数 M 分 4/8/16/32/64/128/default 七档，反向按原长度 N+1
# 分 <=4/8/16/32/64/else 六档；长度 10 还是 axis==0 直接拷贝与 functor 的边界。
# 反向前缀和可观察 0 宽分段的位置，因此每组样本都覆盖容器首尾。
_CONCAT_FOLD_SPEC = FoldSpec(
    lengths=(
        (2, 1),  # 反向 limit=3，kFixed4 档内最小可用长度（N=1 时输出全 0 会提前返回）
        (3, 1),  # 反向 limit=4，kFixed4 上边界
        (4, 1),  # 反向 limit=5，kFixed8 下边界
        (7, 1),  # 反向 limit=8，kFixed8 上边界
        (8, 1),  # 反向 limit=9，kFixed16 下边界
        (9, 1),  # 前后向直接拷贝旁路的上边界（axis==0 时）
        (10, 9),  # M=1，前向档 4 下边界；旁路下边界，functor 内只剩单个有效输入
        (10, 6),  # M=4，前向档 4 上边界
        (10, 5),  # M=5，前向档 8 下边界
        (10, 2),  # M=8，前向档 8 上边界
        (10, 1),  # M=9，前向档 16 下边界
        (15, 1),  # 反向 limit=16，kFixed16 上边界
        (16, 1),  # 反向 limit=17，kFixed32 下边界
        (17, 1),  # M=16，前向档 16 上边界
        (18, 1),  # M=17，前向档 32 下边界
        (31, 1),  # 反向 limit=32，kFixed32 上边界
        (32, 1),  # 反向 limit=33，kFixed64 下边界
        (33, 1),  # M=32，前向档 32 上边界
        (34, 1),  # M=33，前向档 64 下边界
        (63, 1),  # 反向 limit=64，kFixed64 上边界
        (64, 1),  # 反向 limit=65，kVariableLength 下边界
        (65, 1),  # M=64，前向档 64 上边界
        (66, 1),  # M=65，前向档 128 下边界
        (129, 1),  # M=128，前向档 128 上边界
        (130, 1),  # M=129，前向 default 档下边界；档内与任意更大的 M 等价
    )
)

# add_n 只有 in_num==2 的 Eigen 特化；一般路径会用最后一个 Tensor 覆盖 lod_length，
# 所以首尾位置都要覆盖。长度 10 是一般多输入路径的代表值，不对应新模板分档。
_ADD_N_FOLD_SPEC = FoldSpec(
    lengths=(
        (2, 1),  # in_num==2 特化，首尾分别命中 length_1==0 与 length_0==0 两个分支
        (2, 2),  # in_num==2 特化，两侧皆 0，三个分支都不进入
        (3, 1),  # in_num==2 分界的另一侧，转入一般路径
        (10, 1),  # 一般路径的多输入代表值，覆盖 in_data 剔除后个数大于 2 的情形
    )
)

# 折叠规格逐 API 注册。取样表是从具体 kernel 的分支反推的，跨 API 复用既会漏掉本算子
# 独有的分支，也会让本算子背上别人的边界，因此未注册的 API 一律不折叠、只告警。
_API_FOLD_SPECS = {
    "paddle.add_n": _ADD_N_FOLD_SPEC,
    "paddle.concat": _CONCAT_FOLD_SPEC,
}

# 记录被折叠的 API 及配置条数，供收尾汇总；语义与 apis_map 一致，为模块级状态。
folded_apis = {}

# 记录容器超限但未注册折叠规格的 API，收尾告警提示其输出仍是 O(N²)。
unfolded_apis = {}


def _zero_placements(length, zero_count):
    """返回容器首尾的置 0 下标；全置 0 时去重。"""
    head = frozenset(range(zero_count))
    tail = frozenset(range(length - zero_count, length))
    return (head,) if head == tail else (head, tail)


class TensorContainer(NamedTuple):
    """纯 Tensor 容器在扁平 Tensor 列表中的位置及其参数所有权。"""

    is_kwargs: bool
    key: int | str
    start: int
    count: int


def _oversized_containers(api_config, tensor_configs):
    """定位元素超限且 shape/dtype 相同的纯 Tensor 容器。"""
    oversized = []
    offset = 0
    for is_kwargs, items in (
        (False, enumerate(api_config.args)),
        (True, api_config.kwargs.items()),
    ):
        for key, arg_config in items:
            if isinstance(arg_config, TensorConfig):
                offset += 1
                continue
            if not isinstance(arg_config, (list, tuple)):
                continue

            tensor_count = sum(isinstance(item, TensorConfig) for item in arg_config)
            start = offset
            offset += tensor_count
            if tensor_count != len(arg_config) or tensor_count <= HOMOGENEOUS_CONTAINER_LIMIT:
                continue
            tensors = tensor_configs[start : start + tensor_count]
            first = tensors[0]
            if all(
                tensor.shape == first.shape and tensor.dtype == first.dtype for tensor in tensors
            ):
                oversized.append(TensorContainer(is_kwargs, key, start, tensor_count))
    return oversized


def _length_sampled_variants(api_config, container, spec):
    """按 kernel 输入个数分档的边界取样，替代超长同质容器的逐位置枚举。"""
    work_api_config = copy.deepcopy(api_config)
    owner = work_api_config.kwargs if container.is_kwargs else work_api_config.args
    values = owner[container.key]
    element = values[0]
    for axis in range(len(element.shape)):
        zero_element = copy.deepcopy(element)
        zero_element.shape[axis] = 0
        for length, zero_count in spec.lengths:
            for zeros in _zero_placements(length, zero_count):
                # 容器内元素签名一致，序列化只读取 shape 与 dtype，可共享同一对象。
                owner[container.key] = type(values)(
                    [zero_element if position in zeros else element for position in range(length)]
                )
                yield str(work_api_config)


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


def _object_variants(api_config, collapsed):
    """字符串定位失败时的兼容路径；同一配置只深拷贝一次。"""
    # 仅定位失败时使用对象回退，且整个配置生命周期只复制一次。
    work_api_config = copy.deepcopy(api_config)
    work_tensors = get_tensor_configs(work_api_config)
    for tensor_index, axis in _selected_positions(work_tensors, collapsed):
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


def _position_variants(api_config, tensor_configs, canonical_config, spans, shape_equal, collapsed):
    """按位置生成样本，共用字符串快速路径和对象回退路径的覆盖规则。"""
    if len(spans) != len(tensor_configs):
        yield from _object_variants(api_config, collapsed)
        return

    for tensor_index, axis in _selected_positions(tensor_configs, collapsed):
        shape = list(tensor_configs[tensor_index].shape)
        shape[axis] = 0
        yield _replace_shape_spans(canonical_config, spans, {tensor_index: shape})

    if shape_equal:
        for axis in range(len(tensor_configs[0].shape)):
            replacements = {
                index: [*tensor.shape[:axis], 0, *tensor.shape[axis + 1 :]]
                for index, tensor in enumerate(tensor_configs)
            }
            yield _replace_shape_spans(canonical_config, spans, replacements)


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
    canonical_reusable = not any(marker in canonical_config for marker in runtime_repr_markers)
    if not canonical_reusable:
        canonical_config = api_config.config

    # 重新序列化不可用时无法做容器截断，此时保持逐位置枚举，不牺牲覆盖。
    oversized = _oversized_containers(api_config, tensor_configs) if canonical_reusable else []
    spec = _API_FOLD_SPECS.get(api_config.api_name)
    if oversized and spec is None:
        # 取样表是从具体 kernel 的分支反推的，套用到未核对的 API 上可能漏分支，
        # 因此未注册规格时不折叠，只记账告警，输出规模仍是 O(N²)。
        unfolded_apis[api_config.api_name] = unfolded_apis.get(api_config.api_name, 0) + 1
        oversized = []
    # 折叠时保留容器的首、末两个元素：全规模下各留一条位置样本，作为不依赖 kernel
    # 分支推导的兜底，覆盖「位置在大规模下仍有影响」这种取样表可能读漏的情况。
    collapsed = frozenset(
        index
        for container in oversized
        for index in range(container.start + 1, container.start + container.count - 1)
    )
    if oversized:
        folded_apis[api_config.api_name] = folded_apis.get(api_config.api_name, 0) + 1

    spans = _find_tensor_shape_spans(canonical_config)
    yield from _position_variants(
        api_config,
        tensor_configs,
        canonical_config,
        spans,
        shape_equal,
        collapsed,
    )

    for container in oversized:
        yield from _length_sampled_variants(api_config, container, spec)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_dir = os.path.dirname(args.output) or "."
    with tempfile.TemporaryDirectory(prefix=".0size_chunks.", dir=output_dir) as chunk_dir:
        # 临时块位于输出目录，保证大文件处理不依赖系统 /tmp 空间。
        chunk_paths = []
        chunk_lines = set()
        chunk_bytes = 0
        chunk_index = 0
        for input_file in args.inputs:
            print(f"处理: {input_file}")
            for api_config in tqdm(iter_api_configs(input_file)):
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
    if folded_apis:
        print(f"超长同质容器折叠为分档取样（阈值 {HOMOGENEOUS_CONTAINER_LIMIT} 元素）:")
        for api_name, count in sorted(folded_apis.items()):
            print(f"  {api_name}: {count} 条")
    if unfolded_apis:
        print("[告警] 下列 API 的同质容器超限但未注册折叠规格，输出规模仍是 O(N²)；")
        print("       需先核对其 kernel 按输入个数的分支，再补进 _API_FOLD_SPECS:")
        for api_name, count in sorted(unfolded_apis.items()):
            print(f"  {api_name}: {count} 条")
