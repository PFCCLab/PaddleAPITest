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

SPECS = ("4096", "1M", "0size")
RUN_CUSTOM_OP = "paddle._C_ops._run_custom_op"

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
    text: str

    def __str__(self) -> str:
        return self.text


class TensorConfig:
    def __init__(
        self,
        shape: Sequence[int],
        dtype: str,
        place: Any = None,
        is_contiguous: bool = True,
        strides: Sequence[int] | None = None,
    ) -> None:
        self.shape = list(shape)
        self.dtype = dtype
        self.place = place
        self.is_contiguous = is_contiguous
        self.strides = list(strides) if strides is not None else None

    def __str__(self) -> str:
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
    spec: str
    api: str
    index: int
    category: str
    violations: tuple[str, ...]
    config: APIConfig
    source: str | None = None


def find_top_level_open_paren(text: str) -> int:
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
    parts = []
    start = 0
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
        elif char == delimiter and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    if quote is not None or stack:
        raise ValueError(f"unbalanced expression: {text}")
    parts.append(text[start:].strip())
    return parts


def split_top_level_keyword(text: str) -> tuple[str, str] | None:
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
    for name in names:
        prefix = name + "("
        if text.startswith(prefix) and text.endswith(")"):
            return text[len(prefix) : -1]
    return None


def parse_sequence(body: str) -> list[Any]:
    if not body.strip():
        return []
    return [parse_value(part) for part in split_top_level(body) if part]


def parse_tensor(body: str) -> TensorConfig:
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
    return "[" + ", ".join(dump_item(value) for value in values) + "]"


def dump_item(item: Any) -> str:
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
    path = Path(start).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "tester/api_config/config_analyzer.py").is_file():
            return candidate
    raise FileNotFoundError(f"cannot find a PaddleAPITest checkout above {path}")


def import_official_config_types(apitest_root: Path | str):
    root = find_apitest_root(apitest_root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from tester.api_config.config_analyzer import APIConfig as OfficialAPIConfig
    from tester.api_config.config_analyzer import TensorConfig as OfficialTensorConfig

    return OfficialAPIConfig, OfficialTensorConfig


def api_key(config: Any) -> str:
    if config.api_name == RUN_CUSTOM_OP and config.args and isinstance(config.args[0], str):
        return config.args[0]
    return config.api_name


def api_slug(name: str) -> str:
    short_name = name.rsplit(".", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", short_name).strip("_").lower()
    if not slug:
        raise ValueError(f"cannot create output slug for API {name!r}")
    return slug


def iter_tensor_configs(
    value: Any,
    tensor_type: type = TensorConfig,
) -> Iterator[Any]:
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
    values = (*config.args, *config.kwargs.values())
    return [tensor for value in values for tensor in iter_tensor_configs(value, tensor_type)]


def clone_config(config: Any) -> Any:
    return copy.deepcopy(config)


def parse_config_lines(
    paths: Iterable[Path | str],
    api_config_type: type = APIConfig,
) -> list[tuple[Path, int, Any]]:
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
            normalized = str(config)
            if str(api_config_type(normalized)) != normalized:
                raise ValueError(f"non-round-trip seed at {path}:{line_number}")
            parsed.append((path, line_number, config))
    return parsed


def serialize_config(
    config: Any,
    api_config_type: type = APIConfig,
) -> str:
    line = str(config)
    reparsed = api_config_type(line)
    if str(reparsed) != line:
        raise ValueError(f"configuration is not round-trip stable: {line}")
    return line


def anchor_value(spec: str, index: int) -> int:
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
    tensors = config_tensors(config, tensor_type)
    if not tensors or any(0 in tensor.shape for tensor in tensors):
        return False
    tensors[0].shape.append(0)
    return True


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_case_tree(
    output_dir: Path | str,
    records: Sequence[CaseRecord],
    api_config_type: type = APIConfig,
) -> None:
    output_root = Path(output_dir).resolve()
    grouped: dict[str, dict[str, list[CaseRecord]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for record in records:
        if api_key(record.config) != record.api:
            raise ValueError(f"record API does not match config: {record.api}")
        grouped[record.api][record.spec].append(record)

    index_records = []
    for name, by_spec in grouped.items():
        api_dir = output_root / api_slug(name)
        manifest_lines = []
        for spec in SPECS:
            spec_records = by_spec.get(spec, [])
            if not spec_records:
                continue
            lines = [serialize_config(record.config, api_config_type) for record in spec_records]
            if len(lines) != len(set(lines)):
                raise ValueError(f"duplicate configurations for {name}/{spec}")
            write_atomic(api_dir / f"{spec}.txt", "\n".join(lines) + "\n")
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
