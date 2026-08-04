#!/usr/bin/env python3
"""Self-contained PaddleAPITest config model, parser, and generation helpers."""

from __future__ import annotations

import ast
import collections
import copy
import json
import math
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

# 这些规格名同时出现在输出文件名、索引和校验逻辑中，不能任意改写。
# 统一保留字符串形式，避免把 1M 在不同阶段解释成不同的整数。
SPECS = ("4096", "1M", "0size")
# custom op 的第一个参数是 op_name，API 名称需要据此单独归类。
RUN_CUSTOM_OP = "paddle._C_ops._run_custom_op"

# 4096 规格优先覆盖常见边界，尾部再使用稳定递增的扩展值。
# 边界表不代表某个 API 的契约，只用于扩大静态样本的形状覆盖面。
ROW_BOUNDARIES_4096 = (
    1,
    2,
    3,
    7,
    15,
    31,
    63,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16384,
)
# 1M 规格靠近上限和对齐点取样，避免静态生成真的分配百万级数据。
# 这里的值只写入配置文本，实际内存行为由后续运行时决定。
ROW_BOUNDARIES_1M = (
    999_872,
    999_936,
    999_999,
    1_000_000,
    1_000_001,
    1_000_064,
    1_000_128,
    1_048_447,
    1_048_448,
    1_048_575,
    1_048_576,
    1_048_577,
    1_048_704,
)
# 宽度边界用于 API-aware 生成器选择次级维度。
# 保留 1、2、4 等小值可以覆盖标量和向量化分支。
WIDTH_BOUNDARIES = (
    1,
    2,
    3,
    4,
    7,
    8,
    15,
    16,
    31,
    32,
    63,
    64,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2048,
    3072,
    4096,
    5120,
    7168,
    8192,
    16384,
)


@dataclass(frozen=True)
class RawExpression:
    # 无法安全求值的表达式原样保留，保证配置解析不会破坏未知协议。
    # 重新序列化时仍使用原文，因此官方 analyzer 可以继续接管语义校验。
    text: str

    def __str__(self) -> str:
        return self.text


class TensorConfig:
    # 这是一个最小 Tensor 模型，只承担形状关联和静态物化职责。
    # 它不模拟 Paddle kernel，避免生成工具隐式依赖运行时环境。
    def __init__(
        self,
        shape: Sequence[int],
        dtype: str,
        place: Any = None,
        is_contiguous: bool = True,
        strides: Sequence[int] | None = None,
    ) -> None:
        # 复制输入形状，防止后续变异反向修改 seed 配置。
        self.shape = list(shape)
        # dtype、place 和布局属性必须原样保留，生成器只调整选定形状。
        self.dtype = dtype
        self.place = place
        self.is_contiguous = is_contiguous
        self.strides = list(strides) if strides is not None else None

    def __str__(self) -> str:
        # 输出格式与 APIConfig 的 round-trip 约束一致，便于逐行比较。
        result = f"Tensor({self.shape},{dump_item(self.dtype)}"
        if self.place is not None:
            result += f",place={dump_item(self.place)}"
        if not self.is_contiguous:
            result += ",is_contiguous=False"
        if self.strides is not None:
            result += f",strides={dump_plain_list(self.strides)}"
        return result + ")"

    __repr__ = __str__

    def get_numpy_tensor(self, *_args, **_kwargs):
        # 只在 validator 请求时物化；1M 的非零 Tensor 不会走到这里。
        dtype = {
            "bfloat16": "float32",
            "float8_e4m3fn": "float16",
            "float8_e5m2": "float16",
        }.get(self.dtype, self.dtype)
        return numpy.empty(tuple(self.shape), dtype=dtype)


class APIConfig:
    def __init__(self, config: str) -> None:
        normalized = config.replace("\n", "").strip()
        open_index = find_top_level_open_paren(normalized)
        if open_index < 1 or not normalized.endswith(")"):
            raise ValueError(f"invalid API config: {config}")
        self.api_name = normalized[:open_index].strip()
        self.args: list[Any] = []
        self.kwargs: collections.OrderedDict[str, Any] = collections.OrderedDict()
        body = normalized[open_index + 1 : -1]
        for part in split_top_level(body):
            if not part:
                continue
            key_value = split_top_level_keyword(part)
            if key_value is None:
                self.args.append(parse_value(part))
            else:
                key, value = key_value
                self.kwargs[key] = parse_value(value)

    def __str__(self) -> str:
        result = self.api_name + "("
        for arg in self.args:
            result += dump_item(arg) + ", "
        for key, value in self.kwargs.items():
            result += key + "=" + dump_item(value) + ", "
        return result + ")"

    __repr__ = __str__


@dataclass(frozen=True)
class CaseRecord:
    # manifest 记录携带生成原因，便于区分有效、边界和故意非法样本。
    # config 保留结构化对象，最终写文件时再统一做 round-trip 检查。
    spec: str
    api: str
    index: int
    category: str
    violations: tuple[str, ...]
    config: APIConfig
    source: str | None = None


def find_top_level_open_paren(text: str) -> int:
    # 不能用 text.find("(")，因为字符串属性中也可能出现括号。
    # 当前状态机只处理引号和转义，足以定位 API 调用的最外层入口。
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "(":
            return index
    return -1


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    # 配置参数允许嵌套 list、tuple、dict 和 Tensor，分隔符只能在顶层生效。
    # 发现不平衡结构立即失败，避免生成一份表面可写、实际不可解析的配置。
    parts = []
    start = 0
    stack: list[str] = []
    # pairs 用于在遇到闭括号时确认嵌套类型没有交叉。
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != pairs[char]:
                raise ValueError(f"unbalanced expression: {text}")
        elif char == delimiter and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if quote is not None or stack:
        raise ValueError(f"unbalanced expression: {text}")
    parts.append(text[start:].strip())
    return parts


def split_top_level_keyword(text: str) -> tuple[str, str] | None:
    # 关键字等号与容器内部的等号含义不同，只接受顶层等号。
    # 关键字名按 Python 标识符校验，保证 dump 后仍能被同一解析器识别。
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack.pop() != pairs[char]:
                raise ValueError(f"unbalanced expression: {text}")
        elif char == "=" and not stack:
            key = text[:index].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid keyword name: {key}")
            return key, text[index + 1 :].strip()
    return None


def call_body(text: str, names: Sequence[str]) -> str | None:
    # 仅匹配完整调用名，避免把普通字符串或带后缀的表达式误判成 Tensor。
    for name in names:
        prefix = name + "("
        if text.startswith(prefix) and text.endswith(")"):
            return text[len(prefix) : -1]
    return None


def parse_sequence(body: str) -> list[Any]:
    # 空参数列表是合法的 tuple/list 表达，不能被当成解析失败。
    if not body.strip():
        return []
    return [parse_value(part) for part in split_top_level(body) if part]


def parse_tensor(body: str) -> TensorConfig:
    # Tensor 的前两个位置参数是形状和 dtype，其余字段按关键字保留。
    # 形状必须是整数列表，否则无法可靠地进行跨 Tensor 关联变换。
    positional = []
    kwargs = {}
    for part in split_top_level(body):
        if not part:
            continue
        key_value = split_top_level_keyword(part)
        if key_value is None:
            positional.append(parse_value(part))
        else:
            key, value = key_value
            kwargs[key] = parse_value(value)
    if len(positional) < 2:
        raise ValueError(f"Tensor requires shape and dtype: Tensor({body})")
    shape, dtype = positional[:2]
    # dtype 允许未知字符串，工具不替 API 契约做额外的白名单裁剪。
    if not isinstance(shape, list) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError(f"invalid Tensor shape: {shape}")
    if not isinstance(dtype, str):
        dtype = str(dtype)
    return TensorConfig(
        shape,
        dtype,
        place=kwargs.get("place"),
        is_contiguous=kwargs.get("is_contiguous", True),
        strides=kwargs.get("strides"),
    )


def parse_value(text: str) -> Any:
    # 解析优先处理 Paddle 配置中的结构化包装，再回退到 literal_eval。
    # 回退到 RawExpression 是兼容未知属性表达式的关键，而不是吞掉错误。
    value = text.strip()
    tensor_body = call_body(value, ("Tensor", "TensorConfig"))
    if tensor_body is not None:
        return parse_tensor(tensor_body)
    size_body = call_body(value, ("paddle.Size",))
    if size_body is not None:
        return parse_value(size_body)
    tuple_body = call_body(value, ("tuple",))
    if tuple_body is not None:
        return tuple(parse_sequence(tuple_body))
    slice_body = call_body(value, ("slice",))
    if slice_body is not None:
        parts = parse_sequence(slice_body)
        return slice(*(parts + [None] * (3 - len(parts))))
    complex_body = call_body(value, ("complex",))
    if complex_body is not None:
        parts = parse_sequence(complex_body)
        return complex(*parts)
    if value.startswith("list[") and value.endswith("]"):
        return parse_sequence(value[5:-1])
    if value.startswith("[") and value.endswith("]"):
        return parse_sequence(value[1:-1])
    # 字典可能包含非 Python literal 的表达式，失败时必须保持原文。
    if value.startswith("{") and value.endswith("}"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return RawExpression(value)
    if value in {"math.inf", "inf"}:
        return math.inf
    if value in {"-math.inf", "-inf"}:
        return -math.inf
    if value in {"math.nan", "nan", "-math.nan", "-nan"}:
        return math.nan
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return RawExpression(value)


def dump_plain_list(values: Sequence[Any]) -> str:
    # 该函数只负责列表外壳，元素格式由 dump_item 统一决定。
    return "[" + ", ".join(dump_item(value) for value in values) + "]"


def dump_item(item: Any) -> str:
    # 序列化顺序按最具体类型到标量类型排列，避免 bool 被当成 int 输出。
    # 字符串使用 JSON 转义，确保换行和引号不会破坏一行一个配置的约定。
    if isinstance(item, TensorConfig):
        return str(item)
    if isinstance(item, RawExpression):
        return item.text
    if isinstance(item, list):
        return "list[" + "".join(dump_item(value) + "," for value in item) + "]"
    if isinstance(item, tuple):
        return "tuple(" + "".join(dump_item(value) + "," for value in item) + ")"
    if isinstance(item, dict):
        return repr(item)
    if isinstance(item, slice):
        return f"slice({dump_item(item.start)},{dump_item(item.stop)},{dump_item(item.step)})"
    if isinstance(item, complex):
        return f"complex({dump_item(item.real)},{dump_item(item.imag)})"
    if item is None:
        return "None"
    if isinstance(item, float):
        if math.isnan(item):
            return "math.nan"
        if item == math.inf:
            return "math.inf"
        if item == -math.inf:
            return "-math.inf"
    if isinstance(item, (bool, int, float)):
        return str(item)
    if isinstance(item, str):
        return json.dumps(item, ensure_ascii=True)
    return str(item)


def find_apitest_root(start: Path | str) -> Path:
    # 官方 analyzer 是可选兼容校验，因此通过文件探测根目录而不硬编码 cwd。
    # 找不到根目录时明确报错，调用方可以选择关闭 official-analyzer。
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "tester/api_config/config_analyzer.py").is_file():
            return candidate
    raise FileNotFoundError(f"cannot find a PaddleAPITest checkout above {path}")


def import_official_config_types(apitest_root: Path | str):
    # 只在用户显式要求时导入仓库实现，普通静态生成保持自包含。
    root = find_apitest_root(apitest_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from tester.api_config.config_analyzer import APIConfig as OfficialAPIConfig
    from tester.api_config.config_analyzer import TensorConfig as OfficialTensorConfig

    return OfficialAPIConfig, OfficialTensorConfig


def api_key(config: Any) -> str:
    # 普通 API 以 api_name 分组，custom op 以真实 op_name 分组。
    if config.api_name == RUN_CUSTOM_OP and config.args and isinstance(config.args[0], str):
        return config.args[0]
    return config.api_name


def api_slug(name: str) -> str:
    # 目录名只使用可移植字符；完整 API 名仍保存在 manifest 的 api 字段中。
    short_name = name.rsplit(".", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", short_name).strip("_").lower()
    if not slug:
        raise ValueError(f"cannot create output slug for API {name!r}")
    return slug


def iter_tensor_configs(
    value: Any,
    tensor_type: type = TensorConfig,
) -> Iterator[Any]:
    # 递归遍历参数容器，覆盖 Tensor、Tensor 列表和嵌套字典。
    # 只识别指定 tensor_type，方便官方 analyzer 类型与本地模型切换。
    if isinstance(value, tensor_type):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensor_configs(item, tensor_type)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensor_configs(item, tensor_type)


def config_tensors(
    config: Any,
    tensor_type: type = TensorConfig,
) -> list[Any]:
    # args 与 kwargs 都可能包含 Tensor，统一视图避免漏掉可选关键字输入。
    values = (*config.args, *config.kwargs.values())
    return [tensor for value in values for tensor in iter_tensor_configs(value, tensor_type)]


def clone_config(config: Any) -> Any:
    # 每个 case 从独立副本开始，禁止一次变异污染同一 seed 的后续样本。
    return copy.deepcopy(config)


def parse_config_lines(
    paths: Iterable[Path | str],
    api_config_type: type = APIConfig,
) -> list[tuple[Path, int, Any]]:
    # 输入文件按非空、非注释、完整调用逐行读取，保留来源行号用于 manifest。
    # 这里不执行 kernel，也不主动过滤契约非法配置，职责只限于结构化解析。
    parsed = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "(" not in line or not line.endswith(")"):
                continue
            config = api_config_type(line)
            # 解析后立即重新打印，尽早发现 parser 丢失字段或改变转义的缺陷。
            normalized = str(config)
            if str(api_config_type(normalized)) != normalized:
                raise ValueError(f"non-round-trip seed at {path}:{line_number}")
            parsed.append((path, line_number, config))
    return parsed


def serialize_config(
    config: Any,
    api_config_type: type = APIConfig,
) -> str:
    # 写盘前再次解析序列化结果，保证 validator 可以使用相同的稳定表示。
    line = str(config)
    reparsed = api_config_type(line)
    if str(reparsed) != line:
        raise ValueError(f"configuration is not round-trip stable: {line}")
    return line


def anchor_value(spec: str, index: int) -> int:
    # 规格决定主维度的边界策略，index 负责在固定 case 数下持续提供唯一值。
    if spec == "0size":
        return 0
    if spec == "4096":
        if index < len(ROW_BOUNDARIES_4096):
            return ROW_BOUNDARIES_4096[index]
        return 20_000 + index * 128
    if spec == "1M":
        if index < len(ROW_BOUNDARIES_1M):
            return ROW_BOUNDARIES_1M[index]
        return 1_100_000 + index * 128
    raise ValueError(f"unknown spec: {spec}")


def case_category(index: int) -> str:
    # 四格轮换只提供默认分类；API-aware 生成器可以覆盖更精确的违规原因。
    slot = index % 4
    if slot == 2:
        return "edge"
    if slot == 3:
        return "intentionally_invalid"
    return "contract_valid"


def ensure_zero_dimension(
    config: Any,
    tensor_type: type = TensorConfig,
) -> bool:
    # 0size 只要求至少一个维度为零，不改变已有零维 Tensor 的布局信息。
    tensors = config_tensors(config, tensor_type)
    if not tensors or any(0 in tensor.shape for tensor in tensors):
        return False
    tensors[0].shape.append(0)
    return True


def write_atomic(path: Path, content: str) -> None:
    # 临时文件与目标同目录，replace 在同一文件系统内提供原子替换语义。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_case_tree(
    output_dir: Path | str,
    records: Sequence[CaseRecord],
    api_config_type: type = APIConfig,
) -> None:
    # 先按 API/spec 聚合，随后同时写配置文件、manifest 和根 index。
    # 所有重复、API 不匹配和 round-trip 错误都在写入前失败，避免半成品目录。
    output_root = Path(output_dir).resolve()
    grouped: dict[str, dict[str, list[CaseRecord]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    # grouped 的嵌套 default dict 保持输出目录结构与规格顺序相互独立。
    for record in records:
        if api_key(record.config) != record.api:
            raise ValueError(f"record API does not match config: {record.api}")
        grouped[record.api][record.spec].append(record)

    # index 只记录每个规格文件的摘要，详细 case 信息留在 manifest.jsonl。
    index_records = []
    for name, by_spec in grouped.items():
        api_dir = output_root / api_slug(name)
        manifest_lines = []
        # 按 SPECS 固定顺序输出，避免文件系统遍历顺序导致结果不稳定。
        for spec in SPECS:
            spec_records = by_spec.get(spec, [])
            if not spec_records:
                continue
            # 先完成整组序列化和去重，再创建目标文件。
            lines = [serialize_config(record.config, api_config_type) for record in spec_records]
            if len(lines) != len(set(lines)):
                raise ValueError(f"duplicate configurations for {name}/{spec}")
            write_atomic(api_dir / f"{spec}.txt", "\n".join(lines) + "\n")
            # manifest 与配置行一一对应，validator 会按这个顺序反向核对。
            for record, line in zip(spec_records, lines):
                payload = {
                    "api": record.api,
                    "category": record.category,
                    "config": line,
                    "index": record.index,
                    "source": record.source,
                    "spec": record.spec,
                    "violations": list(record.violations),
                }
                manifest_lines.append(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            index_records.append(
                {
                    "api": name,
                    "cases": len(lines),
                    "categories": dict(
                        collections.Counter(record.category for record in spec_records)
                    ),
                    "path": f"{api_slug(name)}/{spec}.txt",
                    "spec": spec,
                }
            )
        write_atomic(api_dir / "manifest.jsonl", "\n".join(manifest_lines) + "\n")
    write_atomic(
        output_root / "index.json",
        json.dumps(index_records, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
