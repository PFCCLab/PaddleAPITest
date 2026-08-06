#!/usr/bin/env python3
"""Shrink line-oriented API configuration sets while preserving modeled coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_API_RE = re.compile(r"^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TENSOR_RE = re.compile(
    r"Tensor\s*\(\s*paddle\.Size\s*\(\s*\[([^\]]*)\]\s*\)\s*,\s*"
    r"(['\"])([^'\"]+)\2"
)
_KWARG_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=")
_BOOL_RE = re.compile(r"(?:(?<![\w.])([A-Za-z_]\w*)\s*=\s*)?\b(True|False|None)\b")
_SEQUENCE_RE = re.compile(r"(?:list\s*)?\[([^\[\]]*)\]|tuple\s*\(([^()]*)\)")
_STRING_RE = re.compile(r"(['\"])(.*?)(?<!\\)\1")

_HIGH_NAME_RE = re.compile(
    r"(?:^|[._])(?:custom|fused|fusion|moe|flash|quant|dequant|scatter|gather|"
    r"index_put|put_along|conv|attention|grid_sample|solve|svd)(?:$|[._])",
    re.IGNORECASE,
)
_MEDIUM_NAME_RE = re.compile(
    r"(?:^|[._])(?:matmul|bmm|baddbmm|addmm|einsum|norm|normalize|rms_norm|softmax|"
    r"embedding|concat|stack|split|unbind|pad|where|lerp|reduce|mean|median|var)"
    r"(?:$|[._])",
    re.IGNORECASE,
)
_SIMPLE_NAME_RE = re.compile(
    r"(?:transpose|reshape|flatten|squeeze|detach|clone|item|tolist|numel|\.dim|"
    r"zeros|full|assign|arange|cast|__add__|__mul__|__sub__|__truediv__)$",
    re.IGNORECASE,
)

PRIORITIES = ("custom", "high", "medium", "low")


@dataclass
class TensorInfo:
    dims: tuple[int | None, ...]
    dtype: str


@dataclass
class ConfigCase:
    case_id: int
    source: Path
    source_order: int
    line_no: int
    text: str
    api_name: str | None
    custom_name: str | None
    signature: str
    tensors: tuple[TensorInfo, ...]
    numbers: tuple[float, ...]
    base_features: frozenset[str]
    priority: str
    duplicate_occurrences: int = 0
    selected: bool = False
    reason: str = "sampled_out"
    representative_id: int | None = None
    features: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SlimOptions:
    rates: dict[str, float]
    minimums: dict[str, int]
    seed: str
    slim_strength: float = 1.0
    high_patterns: tuple[re.Pattern[str], ...] = ()
    custom_patterns: tuple[re.Pattern[str], ...] = ()
    medium_patterns: tuple[re.Pattern[str], ...] = ()
    low_patterns: tuple[re.Pattern[str], ...] = ()
    preserve_patterns: tuple[re.Pattern[str], ...] = ()
    pinned_lines: frozenset[str] = frozenset()


def _outside_strings(text: str, replacement: str = " ") -> str:
    result: list[str] = []
    quote: str | None = None
    escaped = False
    for char in text:
        if quote is not None:
            result.append(replacement)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            result.append(replacement)
        else:
            result.append(char)
    return "".join(result)


def _mask_numbers(text: str) -> str:
    masked = list(text)
    searchable = _outside_strings(text, replacement=" ")
    for match in _NUMBER_RE.finditer(searchable):
        masked[match.start() : match.end()] = "#" * (match.end() - match.start())
    compact = re.sub(r"#+", "#", "".join(masked))
    return re.sub(r"\s+", "", compact)


def _extract_numbers(text: str) -> tuple[float, ...]:
    searchable = _outside_strings(text, replacement=" ")
    values = []
    for match in _NUMBER_RE.finditer(searchable):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return tuple(values)


def _parse_dims(content: str) -> tuple[int | None, ...]:
    dims: list[int | None] = []
    for token in content.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            dims.append(int(token))
        except ValueError:
            dims.append(None)
    return tuple(dims)


def _extract_tensors(text: str) -> tuple[TensorInfo, ...]:
    return tuple(
        TensorInfo(dims=_parse_dims(match.group(1)), dtype=match.group(3))
        for match in _TENSOR_RE.finditer(text)
    )


def _canonical_equality_pattern(values: Sequence[int | None]) -> str:
    labels: dict[int | None, int] = {}
    pattern = []
    for value in values:
        if value not in labels:
            labels[value] = len(labels)
        pattern.append(str(labels[value]))
    return ",".join(pattern)


def _dimension_features(prefix: str, value: int | None) -> set[str]:
    if value is None:
        return {f"{prefix}:dynamic"}
    if value in (-1, 0, 1):
        return {f"{prefix}:special={value}"}
    absolute = abs(value)
    bucket = int(math.floor(math.log2(absolute))) if absolute else 0
    features = {f"{prefix}:log2={bucket}"}
    for alignment in (2, 8, 16, 64, 128):
        if value % alignment == 0:
            features.add(f"{prefix}:aligned={alignment}")
    if value % 8:
        features.add(f"{prefix}:unaligned8")
    nearest_power = 2 ** round(math.log2(absolute))
    delta = absolute - nearest_power
    if abs(delta) <= 2:
        features.add(f"{prefix}:power2_delta={delta}")
    return features


def _broadcastable(left: Sequence[int | None], right: Sequence[int | None]) -> bool:
    for lhs, rhs in zip(reversed(left), reversed(right), strict=False):
        if lhs is None or rhs is None:
            continue
        if lhs != rhs and lhs != 1 and rhs != 1:
            return False
    return True


def _sequence_features(text: str) -> set[str]:
    features: set[str] = set()
    for sequence_index, match in enumerate(_SEQUENCE_RE.finditer(text)):
        content = match.group(1) if match.group(1) is not None else match.group(2)
        tokens = [token.strip() for token in content.split(",") if token.strip()]
        features.add(f"seq:{sequence_index}:len={len(tokens)}")
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            continue
        if not values:
            continue
        zero_count = sum(value == 0 for value in values)
        mean_abs = sum(abs(value) for value in values) / len(values)
        max_ratio = max(abs(value) for value in values) / mean_abs if mean_abs else 0.0
        ratio_bucket = min(8, int(max_ratio))
        features.add(f"seq:{sequence_index}:zeros={zero_count}")
        features.add(f"seq:{sequence_index}:skew={ratio_bucket}")
        features.add(
            f"seq:{sequence_index}:order={'sorted' if values == sorted(values) else 'unsorted'}"
        )
    return features


def _base_features(
    text: str,
    api_name: str | None,
    custom_name: str | None,
    tensors: Sequence[TensorInfo],
) -> frozenset[str]:
    features = {f"api={api_name or '<unparsed>'}", f"tensor_count={len(tensors)}"}
    if custom_name:
        features.add(f"custom={custom_name}")
    for tensor_index, tensor in enumerate(tensors):
        features.add(f"tensor:{tensor_index}:dtype={tensor.dtype}")
        features.add(f"tensor:{tensor_index}:rank={len(tensor.dims)}")
        features.add(f"tensor:{tensor_index}:equality={_canonical_equality_pattern(tensor.dims)}")
        for dim_index, value in enumerate(tensor.dims):
            features.update(_dimension_features(f"tensor:{tensor_index}:dim:{dim_index}", value))
    for left_index, left in enumerate(tensors):
        for right_index in range(left_index + 1, len(tensors)):
            right = tensors[right_index]
            relation = "equal" if left.dims == right.dims else "different"
            features.add(f"tensor_pair:{left_index}:{right_index}:shape={relation}")
            broadcastable = _broadcastable(left.dims, right.dims)
            features.add(f"tensor_pair:{left_index}:{right_index}:broadcast={broadcastable}")
    for keyword in _KWARG_RE.findall(_outside_strings(text)):
        features.add(f"kwarg={keyword}")
    for match in _BOOL_RE.finditer(_outside_strings(text)):
        name = match.group(1) or "<positional>"
        features.add(f"literal:{name}={match.group(2)}")
    for _, value in _STRING_RE.findall(text):
        features.add(f"string={value}")
    features.update(_sequence_features(text))
    return frozenset(features)


def _matches_any(value: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def _complexity_priority(
    api_name: str,
    text: str,
    tensors: Sequence[TensorInfo],
    options: SlimOptions,
) -> str:
    if "_run_custom_op" in api_name or _matches_any(api_name, options.custom_patterns):
        return "custom"
    if _matches_any(api_name, options.high_patterns):
        return "high"
    if _matches_any(api_name, options.medium_patterns):
        return "medium"
    if _matches_any(api_name, options.low_patterns):
        return "low"
    if _HIGH_NAME_RE.search(api_name):
        return "high"

    keyword_count = len(set(_KWARG_RE.findall(_outside_strings(text))))
    dtypes = {tensor.dtype for tensor in tensors}
    max_rank = max((len(tensor.dims) for tensor in tensors), default=0)
    score = 0
    score += 2 if len(tensors) >= 3 else 0
    score += 1 if len(dtypes) >= 2 else 0
    score += 1 if max_rank >= 3 else 0
    score += 1 if keyword_count >= 3 else 0
    score += 1 if len(_SEQUENCE_RE.findall(text)) >= 2 else 0
    if score >= 4:
        return "high"
    if score >= 2 or _MEDIUM_NAME_RE.search(api_name):
        return "medium"
    if _SIMPLE_NAME_RE.search(api_name):
        return "low"
    return "low"


def analyze_line(
    case_id: int,
    source: Path,
    source_order: int,
    line_no: int,
    text: str,
    options: SlimOptions,
) -> ConfigCase:
    api_match = _API_RE.match(text)
    api_name = api_match.group(1) if api_match else None
    strings = [match.group(2) for match in _STRING_RE.finditer(text)]
    custom_name = strings[0] if api_name and "_run_custom_op" in api_name and strings else None
    tensors = _extract_tensors(text)
    signature = _mask_numbers(text)
    priority = (
        _complexity_priority(api_name, text, tensors, options) if api_name is not None else "high"
    )
    return ConfigCase(
        case_id=case_id,
        source=source,
        source_order=source_order,
        line_no=line_no,
        text=text,
        api_name=api_name,
        custom_name=custom_name,
        signature=signature,
        tensors=tensors,
        numbers=_extract_numbers(text[text.find("(") + 1 :]),
        base_features=_base_features(text, api_name, custom_name, tensors),
        priority=priority,
    )


def load_cases(
    inputs: Sequence[Path], options: SlimOptions
) -> tuple[list[ConfigCase], dict[Path, dict[int, str]], dict[str, object]]:
    cases: list[ConfigCase] = []
    passthrough: dict[Path, dict[int, str]] = defaultdict(dict)
    first_by_text: dict[str, ConfigCase] = {}
    by_source: dict[str, dict[str, int]] = {}
    for source_order, path in enumerate(inputs):
        source_stats = {
            "total_lines": 0,
            "config_occurrences": 0,
            "unique_configs": 0,
            "exact_duplicates_removed": 0,
            "passthrough_lines": 0,
        }
        by_source[str(path)] = source_stats
        with path.open(encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                source_stats["total_lines"] += 1
                text = raw_line.rstrip("\r\n")
                if not text.strip() or text.lstrip().startswith("#"):
                    passthrough[path][line_no] = text
                    source_stats["passthrough_lines"] += 1
                    continue
                source_stats["config_occurrences"] += 1
                representative = first_by_text.get(text)
                if representative is not None:
                    representative.duplicate_occurrences += 1
                    source_stats["exact_duplicates_removed"] += 1
                    continue
                case = analyze_line(len(cases), path, source_order, line_no, text, options)
                first_by_text[text] = case
                cases.append(case)
                source_stats["unique_configs"] += 1
    preprocessing = {
        "total_lines": sum(item["total_lines"] for item in by_source.values()),
        "config_occurrences": sum(item["config_occurrences"] for item in by_source.values()),
        "unique_configs": len(cases),
        "exact_duplicates_removed": sum(
            item["exact_duplicates_removed"] for item in by_source.values()
        ),
        "passthrough_lines": sum(item["passthrough_lines"] for item in by_source.values()),
        "by_source": by_source,
    }
    return cases, passthrough, preprocessing


def _scan_source(path: Path) -> dict[str, object]:
    stats: dict[str, object] = {
        "total_lines": 0,
        "config_occurrences": 0,
        "passthrough_lines": 0,
        "api_distribution": Counter(),
    }
    api_distribution = stats["api_distribution"]
    assert isinstance(api_distribution, Counter)
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stats["total_lines"] = int(stats["total_lines"]) + 1
            text = raw_line.rstrip("\r\n")
            if not text.strip() or text.lstrip().startswith("#"):
                stats["passthrough_lines"] = int(stats["passthrough_lines"]) + 1
                continue
            stats["config_occurrences"] = int(stats["config_occurrences"]) + 1
            api_match = _API_RE.match(text)
            api_name = api_match.group(1) if api_match else "<unparsed>"
            api_distribution[api_name] += 1
    return stats


def _new_report_state() -> dict[str, object]:
    return {
        "input_cases": 0,
        "unique_cases": 0,
        "exact_duplicates_removed": 0,
        "kept_cases": 0,
        "excluded_cases": 0,
        "total_removed_occurrences": 0,
        "retention_rate_from_original": 1.0,
        "post_dedup_retention_rate": 1.0,
        "api_count": 0,
        "structural_group_count": 0,
        "api_occurrences": Counter(),
        "api_unique": Counter(),
        "api_kept": Counter(),
        "api_priorities": defaultdict(Counter),
        "source_counts": {},
        "feature_totals": {priority: set() for priority in PRIORITIES},
        "feature_kept": {priority: set() for priority in PRIORITIES},
    }


def _record_group_report(report_state: dict[str, object], group: Sequence[ConfigCase]) -> None:
    report_state["structural_group_count"] = int(report_state["structural_group_count"]) + 1
    api_occurrences = report_state["api_occurrences"]
    api_unique = report_state["api_unique"]
    api_kept = report_state["api_kept"]
    api_priorities = report_state["api_priorities"]
    feature_totals = report_state["feature_totals"]
    feature_kept = report_state["feature_kept"]
    assert isinstance(api_occurrences, Counter)
    assert isinstance(api_unique, Counter)
    assert isinstance(api_kept, Counter)
    assert isinstance(api_priorities, defaultdict)
    assert isinstance(feature_totals, dict)
    assert isinstance(feature_kept, dict)
    group_id = "\0".join(
        [
            group[0].api_name or "<unparsed>",
            group[0].custom_name or "",
            group[0].signature,
        ]
    )
    for case in group:
        api_name = case.api_name or "<unparsed>"
        api_unique[api_name] += 1
        api_priorities[api_name][case.priority] += 1
        feature_totals[case.priority].update((group_id, feature) for feature in case.features)
        if case.selected:
            api_kept[api_name] += 1
            feature_kept[case.priority].update((group_id, feature) for feature in case.features)


def _finalize_report(
    report_state: dict[str, object], preprocessing: dict[str, object]
) -> dict[str, object]:
    input_cases = int(report_state["input_cases"])
    unique_cases = int(report_state["unique_cases"])
    kept_cases = int(report_state["kept_cases"])
    report_state["exact_duplicates_removed"] = input_cases - unique_cases
    report_state["excluded_cases"] = unique_cases - kept_cases
    report_state["total_removed_occurrences"] = input_cases - kept_cases
    report_state["retention_rate_from_original"] = kept_cases / input_cases if input_cases else 1.0
    report_state["post_dedup_retention_rate"] = kept_cases / unique_cases if unique_cases else 1.0
    report_state["api_count"] = len(report_state["api_unique"])

    by_api: dict[str, dict[str, object]] = {}
    api_occurrences = report_state["api_occurrences"]
    api_unique = report_state["api_unique"]
    api_kept = report_state["api_kept"]
    api_priorities = report_state["api_priorities"]
    assert isinstance(api_occurrences, Counter)
    assert isinstance(api_unique, Counter)
    assert isinstance(api_kept, Counter)
    assert isinstance(api_priorities, defaultdict)
    for api_name in sorted(api_unique):
        priority_counts = api_priorities[api_name]
        by_api[api_name] = {
            "input_occurrences": int(api_occurrences[api_name]),
            "unique": int(api_unique[api_name]),
            "kept": int(api_kept[api_name]),
            "post_dedup_retention_rate": (
                api_kept[api_name] / api_unique[api_name] if api_unique[api_name] else 1.0
            ),
            "priority": priority_counts.most_common(1)[0][0] if priority_counts else "low",
        }

    feature_coverage = {}
    all_total: set[tuple[str, str]] = set()
    all_kept: set[tuple[str, str]] = set()
    feature_totals = report_state["feature_totals"]
    feature_kept = report_state["feature_kept"]
    assert isinstance(feature_totals, dict)
    assert isinstance(feature_kept, dict)
    for priority in PRIORITIES:
        total = feature_totals[priority]
        kept = feature_kept[priority]
        all_total.update(total)
        all_kept.update(kept)
        feature_coverage[priority] = {
            "modeled_features": len(total),
            "covered_features": len(kept),
            "coverage_rate": len(kept) / len(total) if total else 1.0,
        }
    feature_coverage["overall"] = {
        "modeled_features": len(all_total),
        "covered_features": len(all_kept),
        "coverage_rate": len(all_kept) / len(all_total) if all_total else 1.0,
    }

    source_counts = {}
    preprocessing_by_source = preprocessing["by_source"]
    assert isinstance(preprocessing_by_source, dict)
    for source, stats in preprocessing_by_source.items():
        unique_configs = int(stats["unique_configs"])
        kept = int(report_state["source_counts"].get(source, {}).get("kept", 0))
        source_counts[source] = {
            **stats,
            "kept": kept,
            "excluded": unique_configs - kept,
            "post_dedup_retention_rate": kept / unique_configs if unique_configs else 1.0,
        }

    return {
        "input_cases": input_cases,
        "unique_cases": unique_cases,
        "exact_duplicates_removed": int(report_state["exact_duplicates_removed"]),
        "kept_cases": kept_cases,
        "excluded_cases": int(report_state["excluded_cases"]),
        "total_removed_occurrences": int(report_state["total_removed_occurrences"]),
        "retention_rate_from_original": float(report_state["retention_rate_from_original"]),
        "post_dedup_retention_rate": float(report_state["post_dedup_retention_rate"]),
        "api_count": int(report_state["api_count"]),
        "structural_group_count": int(report_state["structural_group_count"]),
        "modeled_feature_coverage": feature_coverage,
        "by_source": source_counts,
        "by_api": by_api,
        "preprocessing": preprocessing,
    }


def _process_source_file(
    source: Path,
    source_order: int,
    options: SlimOptions,
    seen_texts: set[str],
    case_id_start: int,
    report_state: dict[str, object],
    preprocessing: dict[str, object],
    output_dir: Path,
    sort_output: bool,
    force: bool,
    audit_handle,
    write_files: bool,
    progress: bool,
) -> int:
    passthrough: dict[int, str] = {}
    groups: dict[tuple[str | None, str | None, str], list[ConfigCase]] = defaultdict(list)
    source_unique = 0
    source_removed = 0

    with source.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if progress and line_no % 50_000 == 0:
                _progress(progress, f"read {source.name}: {line_no} lines")
            text = raw_line.rstrip("\r\n")
            if not text.strip() or text.lstrip().startswith("#"):
                passthrough[line_no] = text
                continue
            if text in seen_texts:
                source_removed += 1
                continue
            seen_texts.add(text)
            case = analyze_line(
                case_id_start + source_unique, source, source_order, line_no, text, options
            )
            groups[(case.api_name, case.custom_name, case.signature)].append(case)
            source_unique += 1

    large_groups = sorted(
        ((key, len(group)) for key, group in groups.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    _progress(
        progress,
        f"{source.name}: unique={source_unique}, groups={len(groups)}, "
        f"top_groups={[(key[0] or '<unparsed>', size) for key, size in large_groups]}",
    )

    selected_cases: list[ConfigCase] = []
    all_cases: list[ConfigCase] = []
    for group_index, group in enumerate(groups.values(), start=1):
        if progress and (len(group) >= 1000 or group_index % 250 == 0):
            _progress(
                progress,
                f"{source.name}: selecting group {group_index}/{len(groups)} "
                f"api={group[0].api_name or '<unparsed>'} size={len(group)}",
            )
        select_cases(group, options)
        if progress and len(group) >= 1000:
            kept = sum(case.selected for case in group)
            _progress(
                progress,
                f"{source.name}: selected api={group[0].api_name or '<unparsed>'} "
                f"size={len(group)} kept={kept}",
            )
        _record_group_report(report_state, group)
        selected_cases.extend(case for case in group if case.selected)
        all_cases.extend(group)

    source_key = str(source)
    report_state["unique_cases"] = int(report_state["unique_cases"]) + source_unique
    report_state["kept_cases"] = int(report_state["kept_cases"]) + len(selected_cases)
    source_counts = report_state["source_counts"]
    assert isinstance(source_counts, dict)
    source_counts[source_key] = {
        "unique": source_unique,
        "kept": len(selected_cases),
        "removed": source_removed,
    }

    slim_lines = dict(passthrough)
    slim_lines.update({case.line_no: case.text for case in selected_cases})
    dedup_lines = dict(passthrough)
    dedup_lines.update({case.line_no: case.text for case in all_cases})
    excluded_lines = {case.line_no: case.text for case in all_cases if not case.selected}

    slim_path = output_dir / f"{source.stem}_slim{source.suffix}"
    dedup_path = output_dir / f"{source.stem}_deduplicated{source.suffix}"
    excluded_path = output_dir / f"{source.stem}_excluded{source.suffix}"

    if write_files:
        existing = [path for path in [slim_path, dedup_path, excluded_path] if path.exists()]
        if existing and not force:
            raise FileExistsError(
                "refusing to overwrite existing output: " + ", ".join(map(str, existing))
            )

        _atomic_write(slim_path, _render_lines(slim_lines, sort_output))
        _atomic_write(dedup_path, _render_lines(dedup_lines, sort_output))
        _atomic_write(excluded_path, _render_lines(excluded_lines, sort_output))

        for case in all_cases:
            audit_handle.write(
                "\t".join(
                    [
                        str(case.case_id),
                        str(case.source),
                        str(case.line_no),
                        case.api_name or "<unparsed>",
                        case.priority,
                        "keep" if case.selected else "remove",
                        case.reason,
                        "" if case.representative_id is None else str(case.representative_id),
                        str(case.duplicate_occurrences),
                        case.text,
                    ]
                )
                + "\n"
            )

    preprocessing_by_source = preprocessing["by_source"]
    assert isinstance(preprocessing_by_source, dict)
    preprocessing_by_source[source_key]["unique_configs"] = source_unique
    preprocessing_by_source[source_key]["exact_duplicates_removed"] = source_removed
    return case_id_start + source_unique


def run_partitioned(args: argparse.Namespace) -> dict[str, object]:
    inputs = args.inputs
    output_dir = args.output_dir
    outputs = _output_paths(inputs, output_dir, "slim")
    deduplicated_outputs = _output_paths(inputs, output_dir, "deduplicated")
    excluded_outputs = _output_paths(inputs, output_dir, "excluded")
    report_path = output_dir / "coverage_report.json"
    audit_path = output_dir / "decisions.tsv"
    targets = [
        *outputs.values(),
        *deduplicated_outputs.values(),
        *excluded_outputs.values(),
        report_path,
        audit_path,
    ]
    if not args.dry_run:
        existing = [path for path in targets if path.exists()]
        if existing and not args.force:
            raise FileExistsError(
                "refusing to overwrite existing output: " + ", ".join(map(str, existing))
            )

    options = _options_from_args(args)
    report_state = _new_report_state()
    preprocessing = {
        "total_lines": 0,
        "config_occurrences": 0,
        "unique_configs": 0,
        "exact_duplicates_removed": 0,
        "passthrough_lines": 0,
        "api_distribution": Counter(),
        "by_source": {},
    }
    seen_texts: set[str] = set()
    case_id_start = 0
    for source_order, source in enumerate(inputs):
        _progress(args.progress, f"scan {source}")
        source_stats = _scan_source(source)
        _progress(
            args.progress,
            f"scan done {source.name}: lines={source_stats['total_lines']} "
            f"configs={source_stats['config_occurrences']} apis={len(source_stats['api_distribution'])}",
        )
        preprocessing["total_lines"] += int(source_stats["total_lines"])
        preprocessing["config_occurrences"] += int(source_stats["config_occurrences"])
        preprocessing["passthrough_lines"] += int(source_stats["passthrough_lines"])
        api_distribution = preprocessing["api_distribution"]
        assert isinstance(api_distribution, Counter)
        api_distribution.update(source_stats["api_distribution"])
        api_occurrences = report_state["api_occurrences"]
        assert isinstance(api_occurrences, Counter)
        api_occurrences.update(source_stats["api_distribution"])
        preprocessing["by_source"][str(source)] = {
            "total_lines": int(source_stats["total_lines"]),
            "config_occurrences": int(source_stats["config_occurrences"]),
            "unique_configs": 0,
            "exact_duplicates_removed": 0,
            "passthrough_lines": int(source_stats["passthrough_lines"]),
            "api_distribution": dict(source_stats["api_distribution"]),
        }

    preprocessing["api_distribution"] = dict(preprocessing["api_distribution"])
    report_state["input_cases"] = preprocessing["config_occurrences"]

    audit_handle = None
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_handle = audit_path.open("w", encoding="utf-8")
        audit_handle.write(
            "\t".join(
                [
                    "case_id",
                    "source",
                    "line_no",
                    "api",
                    "priority",
                    "decision",
                    "reason",
                    "representative_id",
                    "duplicate_occurrences_removed",
                    "config",
                ]
            )
            + "\n"
        )
    try:
        for source_order, source in enumerate(inputs):
            _progress(args.progress, f"process {source}")
            case_id_start = _process_source_file(
                source,
                source_order,
                options,
                seen_texts,
                case_id_start,
                report_state,
                preprocessing,
                output_dir,
                sort_output=not args.preserve_input_order,
                force=args.force,
                audit_handle=audit_handle,
                write_files=not args.dry_run,
                progress=args.progress,
            )
            _progress(args.progress, f"done {source.name}: cumulative_unique={case_id_start}")
    finally:
        if audit_handle is not None:
            audit_handle.close()

    preprocessing["unique_configs"] = int(report_state["unique_cases"])
    preprocessing["exact_duplicates_removed"] = int(preprocessing["config_occurrences"]) - int(
        report_state["unique_cases"]
    )
    report = _finalize_report(report_state, preprocessing)
    if not args.dry_run:
        _atomic_write(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _quantile_boundaries(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(values)
    if len(ordered) < 2:
        return ()
    boundaries = []
    for numerator in (1, 2, 3):
        index = min(len(ordered) - 1, math.floor((len(ordered) - 1) * numerator / 4))
        boundaries.append(ordered[index])
    return tuple(sorted(set(boundaries)))


def _add_group_features(group: Sequence[ConfigCase]) -> None:
    max_numbers = max((len(case.numbers) for case in group), default=0)
    columns = [
        [case.numbers[index] for case in group if index < len(case.numbers)]
        for index in range(max_numbers)
    ]
    boundaries = [_quantile_boundaries(column) for column in columns]
    minimums = [min(column) for column in columns]
    maximums = [max(column) for column in columns]
    for case in group:
        features = set(case.base_features)
        for index, value in enumerate(case.numbers):
            features.add(f"number:{index}:quartile={bisect_right(boundaries[index], value)}")
            if value == minimums[index]:
                features.add(f"number:{index}:minimum")
            if value == maximums[index]:
                features.add(f"number:{index}:maximum")
            if value in (-1.0, 0.0, 1.0):
                features.add(f"number:{index}:special={value:g}")
        case.features = frozenset(features)


def _group_target_size(group_size: int, priority: str, options: SlimOptions) -> int:
    effective_rate = options.rates[priority] * options.slim_strength
    return min(
        group_size,
        max(options.minimums[priority], math.ceil(group_size * effective_rate)),
    )


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[config_slimmer] {message}", file=sys.stderr, flush=True)


def _stable_digest(case: ConfigCase, seed: str) -> str:
    payload = f"{seed}\0{case.text}".encode()
    return hashlib.sha256(payload).hexdigest()


def _coverage_select(group: Sequence[ConfigCase], seed: str) -> list[ConfigCase]:
    uncovered = set().union(*(case.features for case in group))
    remaining = set(range(len(group)))
    selected: list[ConfigCase] = []
    while uncovered:
        best_index = max(
            remaining,
            key=lambda index: (
                len(group[index].features & uncovered),
                -len(group[index].features),
                _stable_digest(group[index], seed),
            ),
        )
        best = group[best_index]
        gain = best.features & uncovered
        if not gain:
            break
        selected.append(best)
        uncovered.difference_update(gain)
        remaining.remove(best_index)
    return selected


def _fill_diverse(
    group: Sequence[ConfigCase], selected: list[ConfigCase], target: int, seed: str
) -> list[ConfigCase]:
    selected_ids = {case.case_id for case in selected}
    buckets: dict[tuple[str, ...], list[ConfigCase]] = defaultdict(list)
    for case in group:
        if case.case_id not in selected_ids:
            buckets[tuple(sorted(case.features))].append(case)
    for bucket in buckets.values():
        bucket.sort(key=lambda case: _stable_digest(case, seed))
    bucket_keys = sorted(
        buckets,
        key=lambda key: hashlib.sha256((seed + "\0" + repr(key)).encode()).hexdigest(),
    )
    while len(selected) < target and bucket_keys:
        next_keys = []
        for key in bucket_keys:
            if len(selected) >= target:
                break
            bucket = buckets[key]
            selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        bucket_keys = next_keys
    return selected


def _assign_representatives(group: Sequence[ConfigCase], selected: Sequence[ConfigCase]) -> None:
    if not selected:
        return
    if len(group) * len(selected) > 2_000_000:
        for case in group:
            if case.selected:
                continue
            index = int(_stable_digest(case, "representative")[:16], 16) % len(selected)
            case.reason = "covered_by_representative"
            case.representative_id = selected[index].case_id
        return

    selected_ids = {case.case_id for case in selected}
    selected_by_features: dict[frozenset[str], ConfigCase] = {}
    feature_index: dict[str, list[ConfigCase]] = defaultdict(list)
    feature_counts = Counter(feature for case in group for feature in case.features)
    varying_features = {feature for feature, count in feature_counts.items() if count < len(group)}
    for case in selected:
        selected_by_features.setdefault(case.features, case)
        for feature in case.features & varying_features:
            feature_index[feature].append(case)

    for case in group:
        if case.case_id in selected_ids:
            continue
        representative = selected_by_features.get(case.features)
        if representative is None:
            shared_counts: Counter[int] = Counter()
            representatives: dict[int, ConfigCase] = {}
            for feature in case.features & varying_features:
                for candidate in feature_index.get(feature, ()):
                    shared_counts[candidate.case_id] += 1
                    representatives[candidate.case_id] = candidate
            if shared_counts:
                best_id = max(
                    shared_counts,
                    key=lambda case_id: (
                        shared_counts[case_id],
                        _stable_digest(representatives[case_id], "representative"),
                    ),
                )
                representative = representatives[best_id]
            else:
                index = int(_stable_digest(case, "representative")[:16], 16) % len(selected)
                representative = selected[index]
        case.reason = "covered_by_representative"
        case.representative_id = representative.case_id


def select_cases(cases: Sequence[ConfigCase], options: SlimOptions) -> None:
    groups: dict[tuple[str | None, str | None, str], list[ConfigCase]] = defaultdict(list)
    for case in cases:
        groups[(case.api_name, case.custom_name, case.signature)].append(case)

    for group in groups.values():
        _add_group_features(group)
        priority = group[0].priority
        target = _group_target_size(len(group), priority, options)
        pinned = [
            case
            for case in group
            if case.text in options.pinned_lines
            or case.api_name is None
            or (
                case.api_name is not None and _matches_any(case.api_name, options.preserve_patterns)
            )
        ]
        pinned_ids = {case.case_id for case in pinned}
        candidates = [case for case in group if case.case_id not in pinned_ids]
        selected = list(pinned)
        if candidates:
            covered = set().union(*(case.features for case in selected)) if selected else set()
            if covered:
                original_features = {case.case_id: case.features for case in candidates}
                for case in candidates:
                    case.features = frozenset(case.features - covered)
                coverage = _coverage_select(candidates, options.seed)
                for case in candidates:
                    case.features = original_features[case.case_id]
            else:
                coverage = _coverage_select(candidates, options.seed)
            selected.extend(coverage)
        selected = _fill_diverse(group, selected, max(target, len(selected)), options.seed)
        coverage_ids = {case.case_id for case in selected[len(pinned) :]}
        for case in selected:
            case.selected = True
            if case.case_id in pinned_ids:
                case.reason = "pinned"
            elif case.case_id in coverage_ids:
                case.reason = "coverage_or_quota"
        _assign_representatives(group, selected)


def _case_report(
    cases: Sequence[ConfigCase], preprocessing: dict[str, object]
) -> dict[str, object]:
    input_count = int(preprocessing["config_occurrences"])
    unique_count = len(cases)
    kept = [case for case in cases if case.selected]
    by_api: dict[str, dict[str, object]] = {}
    for api_name in sorted({case.api_name or "<unparsed>" for case in cases}):
        api_cases = [case for case in cases if (case.api_name or "<unparsed>") == api_name]
        kept_count = sum(case.selected for case in api_cases)
        occurrence_count = sum(1 + case.duplicate_occurrences for case in api_cases)
        priorities = Counter(case.priority for case in api_cases)
        by_api[api_name] = {
            "input_occurrences": occurrence_count,
            "unique": len(api_cases),
            "kept": kept_count,
            "post_dedup_retention_rate": kept_count / len(api_cases),
            "priority": priorities.most_common(1)[0][0],
        }
    by_source: dict[str, dict[str, int | float]] = {}
    source_preprocessing = preprocessing["by_source"]
    assert isinstance(source_preprocessing, dict)
    for source in sorted({*source_preprocessing, *(str(case.source) for case in cases)}):
        source_cases = [case for case in cases if str(case.source) == source]
        kept_count = sum(case.selected for case in source_cases)
        stats = source_preprocessing[str(source)]
        unique_configs = int(stats["unique_configs"])
        by_source[str(source)] = {
            **stats,
            "kept": kept_count,
            "excluded": unique_configs - kept_count,
            "post_dedup_retention_rate": (kept_count / unique_configs if unique_configs else 1.0),
        }

    feature_totals: dict[str, set[tuple[str, str]]] = defaultdict(set)
    feature_kept: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for case in cases:
        group_id = f"{case.api_name}\0{case.custom_name}\0{case.signature}"
        for feature in case.features:
            item = (group_id, feature)
            feature_totals[case.priority].add(item)
            if case.selected:
                feature_kept[case.priority].add(item)
    feature_coverage = {}
    all_total: set[tuple[str, str]] = set()
    all_kept: set[tuple[str, str]] = set()
    for priority in PRIORITIES:
        total = feature_totals[priority]
        covered = feature_kept[priority]
        all_total.update(total)
        all_kept.update(covered)
        feature_coverage[priority] = {
            "modeled_features": len(total),
            "covered_features": len(covered),
            "coverage_rate": len(covered) / len(total) if total else 1.0,
        }
    feature_coverage["overall"] = {
        "modeled_features": len(all_total),
        "covered_features": len(all_kept),
        "coverage_rate": len(all_kept) / len(all_total) if all_total else 1.0,
    }
    return {
        "input_cases": input_count,
        "unique_cases": unique_count,
        "exact_duplicates_removed": input_count - unique_count,
        "kept_cases": len(kept),
        "excluded_cases": unique_count - len(kept),
        "total_removed_occurrences": input_count - len(kept),
        "retention_rate_from_original": len(kept) / input_count if input_count else 1.0,
        "post_dedup_retention_rate": len(kept) / unique_count if unique_count else 1.0,
        "api_count": len(by_api),
        "structural_group_count": len({case.signature for case in cases}),
        "modeled_feature_coverage": feature_coverage,
        "by_source": by_source,
        "by_api": by_api,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(path)


def _output_paths(inputs: Sequence[Path], output_dir: Path, label: str) -> dict[Path, Path]:
    names = [f"{path.stem}_{label}{path.suffix}" for path in inputs]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"input basenames collide in output directory: {', '.join(duplicates)}")
    return {path: output_dir / name for path, name in zip(inputs, names, strict=True)}


def _render_lines(lines: dict[int, str], sort_output: bool) -> str:
    ordered = sorted(lines.values()) if sort_output else [lines[key] for key in sorted(lines)]
    return "".join(f"{line}\n" for line in ordered)


def write_outputs(
    cases: Sequence[ConfigCase],
    inputs: Sequence[Path],
    passthrough: dict[Path, dict[int, str]],
    output_dir: Path,
    report: dict[str, object],
    force: bool,
    sort_output: bool,
) -> dict[Path, Path]:
    outputs = _output_paths(inputs, output_dir, "slim")
    deduplicated_outputs = _output_paths(inputs, output_dir, "deduplicated")
    excluded_outputs = _output_paths(inputs, output_dir, "excluded")
    report_path = output_dir / "coverage_report.json"
    audit_path = output_dir / "decisions.tsv"
    targets = [
        *outputs.values(),
        *deduplicated_outputs.values(),
        *excluded_outputs.values(),
        report_path,
        audit_path,
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing output: " + ", ".join(map(str, existing))
        )

    for source, destination in outputs.items():
        retained_lines = dict(passthrough.get(source, {}))
        retained_lines.update(
            {case.line_no: case.text for case in cases if case.source == source and case.selected}
        )
        _atomic_write(destination, _render_lines(retained_lines, sort_output))
    for source, destination in deduplicated_outputs.items():
        deduplicated_lines = dict(passthrough.get(source, {}))
        deduplicated_lines.update(
            {case.line_no: case.text for case in cases if case.source == source}
        )
        _atomic_write(destination, _render_lines(deduplicated_lines, sort_output))
    for source, destination in excluded_outputs.items():
        excluded = {
            case.line_no: case.text for case in cases if case.source == source and not case.selected
        }
        _atomic_write(destination, _render_lines(excluded, sort_output))
    _atomic_write(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_dir, delete=False
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "case_id",
                "source",
                "line_no",
                "api",
                "priority",
                "decision",
                "reason",
                "representative_id",
                "duplicate_occurrences_removed",
                "config",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.case_id,
                    case.source,
                    case.line_no,
                    case.api_name or "<unparsed>",
                    case.priority,
                    "keep" if case.selected else "remove",
                    case.reason,
                    "" if case.representative_id is None else case.representative_id,
                    case.duplicate_occurrences,
                    case.text,
                ]
            )
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(audit_path)
    return outputs


def _compile_patterns(values: Iterable[str], option: str) -> tuple[re.Pattern[str], ...]:
    patterns = []
    for value in values:
        try:
            patterns.append(re.compile(value))
        except re.error as error:
            raise ValueError(f"invalid {option} regex {value!r}: {error}") from error
    return tuple(patterns)


def _load_pinned_lines(paths: Sequence[Path]) -> frozenset[str]:
    lines = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            lines.update(line.rstrip("\r\n") for line in handle if line.strip())
    return frozenset(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coverage-aware slimming for line-oriented API configuration sets."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input configuration files")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("config_slimmer_output"))
    parser.add_argument("--high-rate", type=float, default=0.35)
    parser.add_argument("--custom-rate", type=float, default=0.50)
    parser.add_argument("--medium-rate", type=float, default=0.20)
    parser.add_argument("--low-rate", type=float, default=0.08)
    parser.add_argument("--min-high", type=int, default=32)
    parser.add_argument("--min-custom", type=int, default=64)
    parser.add_argument("--min-medium", type=int, default=12)
    parser.add_argument("--min-low", type=int, default=4)
    parser.add_argument("--high-api", action="append", default=[], metavar="REGEX")
    parser.add_argument("--custom-api", action="append", default=[], metavar="REGEX")
    parser.add_argument("--medium-api", action="append", default=[], metavar="REGEX")
    parser.add_argument("--low-api", action="append", default=[], metavar="REGEX")
    parser.add_argument(
        "--preserve-api",
        action="append",
        default=[],
        metavar="REGEX",
        help="keep every configuration for matching APIs",
    )
    parser.add_argument(
        "--pin-file",
        action="append",
        default=[],
        type=Path,
        help="files containing exact configuration lines that must be kept",
    )
    parser.add_argument("--seed", default="config-slimmer-v1")
    parser.add_argument(
        "--slim-strength",
        type=float,
        default=1.0,
        help="0 means minimum retention only, 1 means default retention rates",
    )
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help="keep first-occurrence order instead of sorting output lines",
    )
    parser.add_argument("--progress", action="store_true", help="print progress to stderr")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _options_from_args(args: argparse.Namespace) -> SlimOptions:
    rates = {
        "custom": args.custom_rate,
        "high": args.high_rate,
        "medium": args.medium_rate,
        "low": args.low_rate,
    }
    minimums = {
        "custom": args.min_custom,
        "high": args.min_high,
        "medium": args.min_medium,
        "low": args.min_low,
    }
    for name, value in rates.items():
        if not 0 <= value <= 1:
            raise ValueError(f"--{name}-rate must be between 0 and 1")
    for name, value in minimums.items():
        if value < 1:
            raise ValueError(f"--min-{name} must be at least 1")
    if not 0 <= args.slim_strength <= 1:
        raise ValueError("--slim-strength must be between 0 and 1")
    return SlimOptions(
        rates=rates,
        minimums=minimums,
        seed=args.seed,
        slim_strength=args.slim_strength,
        custom_patterns=_compile_patterns(args.custom_api, "--custom-api"),
        high_patterns=_compile_patterns(args.high_api, "--high-api"),
        medium_patterns=_compile_patterns(args.medium_api, "--medium-api"),
        low_patterns=_compile_patterns(args.low_api, "--low-api"),
        preserve_patterns=_compile_patterns(args.preserve_api, "--preserve-api"),
        pinned_lines=_load_pinned_lines(args.pin_file),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    for path in [*args.inputs, *args.pin_file]:
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
    report = run_partitioned(args)
    options = _options_from_args(args)
    report["settings"] = {
        "rates": options.rates,
        "minimums": options.minimums,
        "seed": options.seed,
        "slim_strength": options.slim_strength,
        "sorted_outputs": not args.preserve_input_order,
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("dry-run: no files were written", file=sys.stderr)
    else:
        print(f"outputs written to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
