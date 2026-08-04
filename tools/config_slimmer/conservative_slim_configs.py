#!/usr/bin/env python3
"""Conservatively remove numeric near-duplicates from API configuration sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tools.config_slimmer.api_models import (
    ApiModel,
    modeled_api_names,
    resolve_api_model,
)
from tools.config_slimmer.slim_configs import (
    _API_RE,
    _NUMBER_RE,
    _SEQUENCE_RE,
    _STRING_RE,
    SlimOptions,
    _outside_strings,
)
from tools.config_slimmer.slim_configs import (
    analyze_line as analyze_coverage_line,
)
from tools.config_slimmer.slim_configs import (
    select_cases as select_coverage_cases,
)

_MOE_PERMUTE_API = "paddle.nn.functional.moe_permute"
_MOE_UNPERMUTE_API = "paddle.nn.functional.moe_unpermute"


@dataclass(frozen=True)
class NumericToken:
    value: int | float
    kind: str
    raw: str


@dataclass
class ConservativeCase:
    case_id: int
    source: Path
    source_order: int
    line_no: int
    text: str
    api_name: str | None
    custom_name: str | None
    signature: str
    numbers: tuple[NumericToken, ...]
    numeric_sequences: tuple[tuple[int, ...], ...]
    duplicate_occurrences: int = 0
    selected: bool = True
    reason: str = "unique"
    representative_id: int | None = None
    differing_positions: tuple[int, ...] = ()
    max_absolute_delta: float = 0.0
    max_relative_delta: float = 0.0
    sequence_sum_relative_delta: float = 0.0
    boundary_features: frozenset[str] = field(default_factory=frozenset)
    model_name: str = "unassigned"


@dataclass(frozen=True)
class ConservativeOptions:
    relative_tolerance: float = 0.02
    absolute_tolerance: float = 0.0
    max_absolute_delta: float = 256.0
    exact_small_integer: int = 16
    exact_integer_above: int = 1_000_000_000
    max_removal_rate: float = 0.20
    min_group_size: int = 20
    max_candidate_checks: int = 256
    seed: str = "conservative-config-slimmer-v1"
    protect_boundaries: bool = True
    complex_api_profiles: bool = True
    moe_sequence_relative_tolerance: float = 0.20
    moe_sequence_max_absolute_delta: float = 1024.0
    moe_sequence_sum_relative_tolerance: float = 0.05
    preserve_patterns: tuple[re.Pattern[str], ...] = ()
    pinned_lines: frozenset[str] = frozenset()


def _numeric_kind(raw: str) -> str:
    return "int" if re.fullmatch(r"[-+]?\d+", raw) else "float"


def _numeric_signature(text: str) -> tuple[str, tuple[NumericToken, ...]]:
    """Replace every non-string number with a typed placeholder."""
    searchable = _outside_strings(text, replacement=" ")
    parts: list[str] = []
    tokens: list[NumericToken] = []
    cursor = 0
    for match in _NUMBER_RE.finditer(searchable):
        raw = match.group(0)
        kind = _numeric_kind(raw)
        try:
            value = int(raw) if kind == "int" else float(raw)
        except ValueError:
            continue
        parts.append(text[cursor : match.start()])
        parts.append(f"<number:{kind}>")
        tokens.append(NumericToken(value=value, kind=kind, raw=raw))
        cursor = match.end()
    parts.append(text[cursor:])
    # Keeping whitespace makes every non-numeric character, including string
    # contents, part of the strict structural identity.
    return "".join(parts), tuple(tokens)


def _numeric_sequences(text: str) -> tuple[tuple[int, ...], ...]:
    """Return numeric-token indexes for non-shape list and tuple literals."""
    searchable = _outside_strings(text, replacement=" ")
    number_matches = list(_NUMBER_RE.finditer(searchable))
    sequences: list[tuple[int, ...]] = []
    for sequence in _SEQUENCE_RE.finditer(searchable):
        prefix = searchable[max(0, sequence.start() - 40) : sequence.start()]
        if re.search(r"paddle\.Size\s*\(\s*$", prefix):
            continue
        indexes = tuple(
            index
            for index, number in enumerate(number_matches)
            if sequence.start() <= number.start() and number.end() <= sequence.end()
        )
        if indexes:
            sequences.append(indexes)
    return tuple(sequences)


def analyze_line(
    case_id: int,
    source: Path,
    source_order: int,
    line_no: int,
    text: str,
) -> ConservativeCase:
    api_match = _API_RE.match(text)
    api_name = api_match.group(1) if api_match else None
    strings = [match.group(2) for match in _STRING_RE.finditer(text)]
    custom_name = strings[0] if api_name and "_run_custom_op" in api_name and strings else None
    signature, numbers = _numeric_signature(text)
    return ConservativeCase(
        case_id=case_id,
        source=source,
        source_order=source_order,
        line_no=line_no,
        text=text,
        api_name=api_name,
        custom_name=custom_name,
        signature=signature,
        numbers=numbers,
        numeric_sequences=_numeric_sequences(text),
    )


def load_cases(
    inputs: Sequence[Path],
) -> tuple[list[ConservativeCase], dict[Path, dict[int, str]], dict[str, object]]:
    cases: list[ConservativeCase] = []
    passthrough: dict[Path, dict[int, str]] = defaultdict(dict)
    first_by_text: dict[str, ConservativeCase] = {}
    by_source: dict[str, dict[str, int]] = {}

    for source_order, source in enumerate(inputs):
        stats = {
            "total_lines": 0,
            "config_occurrences": 0,
            "unique_configs": 0,
            "exact_duplicates_removed": 0,
            "passthrough_lines": 0,
        }
        by_source[str(source)] = stats
        with source.open(encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                stats["total_lines"] += 1
                text = raw_line.rstrip("\r\n")
                if not text.strip() or text.lstrip().startswith("#"):
                    passthrough[source][line_no] = text
                    stats["passthrough_lines"] += 1
                    continue
                stats["config_occurrences"] += 1
                representative = first_by_text.get(text)
                if representative is not None:
                    representative.duplicate_occurrences += 1
                    stats["exact_duplicates_removed"] += 1
                    continue
                case = analyze_line(len(cases), source, source_order, line_no, text)
                first_by_text[text] = case
                cases.append(case)
                stats["unique_configs"] += 1

    return (
        cases,
        passthrough,
        {
            "total_lines": sum(item["total_lines"] for item in by_source.values()),
            "config_occurrences": sum(item["config_occurrences"] for item in by_source.values()),
            "unique_configs": len(cases),
            "exact_duplicates_removed": sum(
                item["exact_duplicates_removed"] for item in by_source.values()
            ),
            "passthrough_lines": sum(item["passthrough_lines"] for item in by_source.values()),
            "by_source": by_source,
        },
    )


def _stable_digest(case: ConservativeCase, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{case.text}".encode()).hexdigest()


def _magnitude_bucket(value: float) -> int | None:
    if value == 0 or not math.isfinite(value):
        return None
    return math.floor(math.log2(abs(value)))


def _is_integral(token: NumericToken) -> bool:
    return token.kind == "int" and math.isfinite(token.value)


def _number_close(left: NumericToken, right: NumericToken, options: ConservativeOptions) -> bool:
    return _number_close_with_limits(
        left,
        right,
        options,
        relative_tolerance=options.relative_tolerance,
        max_absolute_delta=options.max_absolute_delta,
    )


def _number_close_with_limits(
    left: NumericToken,
    right: NumericToken,
    options: ConservativeOptions,
    relative_tolerance: float,
    max_absolute_delta: float,
) -> bool:
    if left.value == right.value:
        return math.isfinite(left.value) or left.raw == right.raw
    if not math.isfinite(left.value) or not math.isfinite(right.value):
        return False
    if left.value * right.value <= 0:
        return False
    if _magnitude_bucket(left.value) != _magnitude_bucket(right.value):
        return False
    if _is_integral(left) and _is_integral(right):
        if min(abs(left.value), abs(right.value)) <= options.exact_small_integer:
            return False
        if max(abs(left.value), abs(right.value)) >= options.exact_integer_above:
            return False

    delta = abs(left.value - right.value)
    if delta > max_absolute_delta:
        return False
    scale = max(abs(left.value), abs(right.value))
    allowed = max(options.absolute_tolerance, relative_tolerance * scale)
    return delta <= allowed


def _case_close(
    left: ConservativeCase,
    right: ConservativeCase,
    options: ConservativeOptions,
) -> bool:
    if options.complex_api_profiles:
        if left.api_name == _MOE_PERMUTE_API:
            return _moe_permute_close(left, right, options)
        if left.api_name == _MOE_UNPERMUTE_API:
            return _moe_unpermute_close(left, right, options)
    return all(
        _number_close(lhs, rhs, options)
        for lhs, rhs in zip(left.numbers, right.numbers, strict=True)
    )


def _equality_pattern(case: ConservativeCase, positions: Sequence[int]) -> tuple[int, ...]:
    labels: dict[int | float, int] = {}
    pattern = []
    for position in positions:
        value = case.numbers[position].value
        if value not in labels:
            labels[value] = len(labels)
        pattern.append(labels[value])
    return tuple(pattern)


def _moe_permute_close(
    left: ConservativeCase,
    right: ConservativeCase,
    options: ConservativeOptions,
) -> bool:
    if len(left.numeric_sequences) != 1 or len(right.numeric_sequences) != 1:
        return False
    left_sequence = left.numeric_sequences[0]
    right_sequence = right.numeric_sequences[0]
    if len(left_sequence) != len(right_sequence):
        return False

    sequence_positions = set(left_sequence)
    regular_positions = [
        index for index in range(len(left.numbers)) if index not in sequence_positions
    ]
    if _equality_pattern(left, regular_positions) != _equality_pattern(right, regular_positions):
        return False
    if not all(
        _number_close(left.numbers[index], right.numbers[index], options)
        for index in regular_positions
    ):
        return False

    left_values = sorted(
        (left.numbers[index] for index in left_sequence),
        key=lambda token: token.value,
    )
    right_values = sorted(
        (right.numbers[index] for index in right_sequence),
        key=lambda token: token.value,
    )
    if not all(
        _number_close_with_limits(
            lhs,
            rhs,
            options,
            relative_tolerance=options.moe_sequence_relative_tolerance,
            max_absolute_delta=options.moe_sequence_max_absolute_delta,
        )
        for lhs, rhs in zip(left_values, right_values, strict=True)
    ):
        return False

    left_sum = sum(token.value for token in left_values)
    right_sum = sum(token.value for token in right_values)
    maximum_sum = max(abs(left_sum), abs(right_sum))
    if maximum_sum == 0:
        return True
    return abs(left_sum - right_sum) <= options.moe_sequence_sum_relative_tolerance * maximum_sum


def _moe_unpermute_close(
    left: ConservativeCase,
    right: ConservativeCase,
    options: ConservativeOptions,
) -> bool:
    positions = list(range(len(left.numbers)))
    if _equality_pattern(left, positions) != _equality_pattern(right, positions):
        return False
    return all(
        _number_close(lhs, rhs, options)
        for lhs, rhs in zip(left.numbers, right.numbers, strict=True)
    )


def _case_distance(
    case: ConservativeCase,
    representative: ConservativeCase,
    options: ConservativeOptions,
) -> tuple[tuple[int, ...], float, float]:
    differing = [
        index
        for index, (left, right) in enumerate(
            zip(case.numbers, representative.numbers, strict=True)
        )
        if left.value != right.value or (not math.isfinite(left.value) and left.raw != right.raw)
    ]
    max_absolute = 0.0
    max_relative = 0.0
    for left, right in _comparison_pairs(case, representative, options):
        if left.value == right.value and (math.isfinite(left.value) or left.raw == right.raw):
            continue
        delta = abs(left.value - right.value)
        scale = max(abs(left.value), abs(right.value))
        max_absolute = max(max_absolute, delta)
        max_relative = max(max_relative, delta / scale if scale else 0.0)
    return tuple(differing), max_absolute, max_relative


def _comparison_pairs(
    case: ConservativeCase,
    representative: ConservativeCase,
    options: ConservativeOptions,
) -> list[tuple[NumericToken, NumericToken]]:
    if (
        options.complex_api_profiles
        and case.api_name == _MOE_PERMUTE_API
        and len(case.numeric_sequences) == 1
        and len(representative.numeric_sequences) == 1
    ):
        case_sequence = case.numeric_sequences[0]
        representative_sequence = representative.numeric_sequences[0]
        sequence_positions = set(case_sequence)
        pairs = [
            (case.numbers[index], representative.numbers[index])
            for index in range(len(case.numbers))
            if index not in sequence_positions
        ]
        case_values = sorted(
            (case.numbers[index] for index in case_sequence),
            key=lambda token: token.value,
        )
        representative_values = sorted(
            (representative.numbers[index] for index in representative_sequence),
            key=lambda token: token.value,
        )
        pairs.extend(zip(case_values, representative_values, strict=True))
        return pairs
    return list(zip(case.numbers, representative.numbers, strict=True))


def _sequence_sum_relative_delta(
    case: ConservativeCase,
    representative: ConservativeCase,
    options: ConservativeOptions,
) -> float:
    if (
        not options.complex_api_profiles
        or case.api_name != _MOE_PERMUTE_API
        or len(case.numeric_sequences) != 1
        or len(representative.numeric_sequences) != 1
    ):
        return 0.0
    left_sum = sum(case.numbers[index].value for index in case.numeric_sequences[0])
    right_sum = sum(
        representative.numbers[index].value for index in representative.numeric_sequences[0]
    )
    scale = max(abs(left_sum), abs(right_sum))
    return abs(left_sum - right_sum) / scale if scale else 0.0


def _alignment_class(token: NumericToken) -> str:
    if not _is_integral(token):
        return "not-integer"
    value = int(token.value)
    for alignment in (128, 64, 16, 8, 2):
        if value % alignment == 0:
            return str(alignment)
    return "unaligned"


def _power_delta(token: NumericToken) -> int | None:
    if not _is_integral(token) or token.value == 0:
        return None
    absolute = abs(int(token.value))
    nearest = 2 ** round(math.log2(absolute))
    delta = absolute - nearest
    return delta if abs(delta) <= 2 else None


def _add_boundary_features(group: Sequence[ConservativeCase]) -> None:
    if not group or not group[0].numbers:
        return
    columns = [
        [case.numbers[index].value for case in group] for index in range(len(group[0].numbers))
    ]
    varying = {index for index, column in enumerate(columns) if len(set(column)) > 1}
    minimums = {index: min(columns[index]) for index in varying}
    maximums = {index: max(columns[index]) for index in varying}

    for case in group:
        features: set[str] = set()
        for index in varying:
            token = case.numbers[index]
            value = token.value
            if value == minimums[index]:
                features.add(f"number:{index}:minimum")
            if value == maximums[index]:
                features.add(f"number:{index}:maximum")
            sign = "negative" if value < 0 else "positive" if value > 0 else "zero"
            features.add(f"number:{index}:sign={sign}")
            features.add(f"number:{index}:magnitude={_magnitude_bucket(value)}")
            features.add(f"number:{index}:alignment={_alignment_class(token)}")
            power_delta = _power_delta(token)
            if power_delta is not None:
                features.add(f"number:{index}:power2_delta={power_delta}")
        case.boundary_features = frozenset(features)


def _boundary_representatives(group: Sequence[ConservativeCase], seed: str) -> set[int]:
    uncovered = set().union(*(case.boundary_features for case in group))
    remaining = set(range(len(group)))
    selected: set[int] = set()
    while uncovered and remaining:
        best_index = max(
            remaining,
            key=lambda index: (
                len(group[index].boundary_features & uncovered),
                -len(group[index].boundary_features),
                _stable_digest(group[index], seed),
            ),
        )
        gain = group[best_index].boundary_features & uncovered
        if not gain:
            break
        selected.add(group[best_index].case_id)
        uncovered.difference_update(gain)
        remaining.remove(best_index)
    return selected


def _matches_any(value: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(value) for pattern in patterns)


def _specialized_model_features(case: ConservativeCase, model: ApiModel) -> frozenset[str]:
    features: set[str] = set()
    values = [token.value for token in case.numbers]
    if model.feature_profile == "moe_permute" and len(case.numeric_sequences) == 1:
        sequence = [case.numbers[index].value for index in case.numeric_sequences[0]]
        if sequence:
            ordered = sorted(sequence)
            total = sum(ordered)
            mean = total / len(ordered)
            features.add(f"moe:load:zeros={sum(value == 0 for value in ordered)}")
            features.add(f"moe:load:sum_log2={_magnitude_bucket(total)}")
            imbalance = max(ordered) / mean if mean else 0.0
            features.add(f"moe:load:imbalance_quarter={round(imbalance * 4)}")
            for index, value in enumerate(ordered):
                features.add(f"moe:load:{index}:log2={_magnitude_bucket(value)}")
                padded_blocks = math.ceil(value / 128) if value > 0 else 0
                features.add(f"moe:load:{index}:blocks_log2={_magnitude_bucket(padded_blocks)}")
            token_count = values[0] if values else 0
            if token_count:
                features.add(f"moe:load_to_token:tenth={round(total / token_count * 10)}")
    elif model.feature_profile == "moe_unpermute" and values:
        counts = Counter(values)
        linked = [value for value, count in counts.items() if count >= 2 and abs(value) > 16]
        for index, value in enumerate(sorted(linked)):
            features.add(f"moe:linked:{index}:log2={_magnitude_bucket(value)}")
            if values[0]:
                features.add(f"moe:linked:{index}:ratio_tenth={round(value / values[0] * 10)}")
    elif model.feature_profile == "fp8_blockwise" and len(values) >= 2:
        rows, columns = values[:2]
        row_blocks = math.ceil(rows / 128) if rows > 0 else 0
        column_blocks = math.ceil(columns / 128) if columns > 0 else 0
        features.update(
            {
                f"fp8:row_blocks_log2={_magnitude_bucket(row_blocks)}",
                f"fp8:column_blocks_log2={_magnitude_bucket(column_blocks)}",
                f"fp8:row_remainder={int(rows) % 128 if float(rows).is_integer() else 'float'}",
            }
        )
    elif model.feature_profile in {"fused_act_dequant", "custom"}:
        for index, tensor in enumerate(case.numbers[:8]):
            features.add(f"fused:number:{index}:magnitude={_magnitude_bucket(tensor.value)}")
    return frozenset(features)


def _select_coverage_model_group(
    group: Sequence[ConservativeCase],
    options: ConservativeOptions,
    model: ApiModel,
) -> None:
    rates = dict.fromkeys(("custom", "high", "medium", "low"), model.retention_rate)
    minimums = dict.fromkeys(("custom", "high", "medium", "low"), model.minimum_cases)
    coverage_options = SlimOptions(
        rates=rates,
        minimums=minimums,
        seed=options.seed + "\0simple-kernel",
        pinned_lines=options.pinned_lines,
    )
    coverage_cases = [
        analyze_coverage_line(
            case.case_id,
            case.source,
            case.source_order,
            case.line_no,
            case.text,
            coverage_options,
        )
        for case in group
    ]
    for case, coverage_case in zip(group, coverage_cases, strict=True):
        coverage_case.base_features = frozenset(
            set(coverage_case.base_features) | set(_specialized_model_features(case, model))
        )
    select_coverage_cases(coverage_cases, coverage_options)
    coverage_by_id = {case.case_id: case for case in coverage_cases}
    conservative_by_id = {case.case_id: case for case in group}

    for case in group:
        coverage_case = coverage_by_id[case.case_id]
        if coverage_case.selected:
            case.reason = f"{model.name}_coverage"
            continue
        case.selected = False
        case.reason = f"{model.name}_sampled_out"
        case.representative_id = coverage_case.representative_id
        if case.representative_id is not None:
            representative = conservative_by_id[case.representative_id]
            (
                case.differing_positions,
                case.max_absolute_delta,
                case.max_relative_delta,
            ) = _case_distance(case, representative, options)


def _index_position(group: Sequence[ConservativeCase], options: ConservativeOptions) -> int:
    positions = list(range(len(group[0].numbers)))
    if (
        options.complex_api_profiles
        and group[0].api_name == _MOE_PERMUTE_API
        and len(group[0].numeric_sequences) == 1
    ):
        sequence_positions = set(group[0].numeric_sequences[0])
        regular_positions = [
            position for position in positions if position not in sequence_positions
        ]
        if regular_positions:
            positions = regular_positions
    return max(
        positions,
        key=lambda index: (
            len({case.numbers[index].value for case in group}),
            max(case.numbers[index].value for case in group)
            - min(case.numbers[index].value for case in group),
            -index,
        ),
    )


def _normalized_distance(
    case: ConservativeCase,
    representative: ConservativeCase,
    options: ConservativeOptions,
) -> float:
    distances = []
    for left, right in _comparison_pairs(case, representative, options):
        delta = abs(left.value - right.value)
        if not delta:
            continue
        scale = max(abs(left.value), abs(right.value))
        allowed = max(options.absolute_tolerance, options.relative_tolerance * scale)
        distances.append(delta / allowed if allowed else math.inf)
    return max(distances, default=0.0)


def select_group(group: Sequence[ConservativeCase], options: ConservativeOptions) -> None:
    if not group:
        return
    model = resolve_api_model(group[0].api_name, group[0].custom_name)
    for case in group:
        case.model_name = model.name
    if group[0].api_name is not None and _matches_any(group[0].api_name, options.preserve_patterns):
        for case in group:
            case.reason = "preserved_api"
        return
    if model.strategy == "preserve":
        for case in group:
            case.reason = f"{model.name}_model"
        return
    if len(group) < options.min_group_size:
        for case in group:
            case.reason = "small_group"
        return
    if group[0].api_name is None or not group[0].numbers:
        for case in group:
            case.reason = "unparsed_or_non_numeric"
        return
    if model.strategy == "coverage":
        _select_coverage_model_group(group, options, model)
        return

    _add_boundary_features(group)
    protected_ids = (
        _boundary_representatives(group, options.seed) if options.protect_boundaries else set()
    )
    protected_ids.update(case.case_id for case in group if case.text in options.pinned_lines)
    index_position = _index_position(group, options)
    ordered = sorted(
        group,
        key=lambda case: (
            case.numbers[index_position].value,
            tuple(token.value for token in case.numbers),
            _stable_digest(case, options.seed),
        ),
    )
    max_removed = math.floor(len(group) * options.max_removal_rate)
    active: deque[ConservativeCase] = deque()
    near_duplicates: list[ConservativeCase] = []

    for case in ordered:
        while active and not _number_close(
            active[0].numbers[index_position],
            case.numbers[index_position],
            options,
        ):
            active.popleft()

        if case.case_id in protected_ids:
            case.reason = "boundary_or_pinned"
            active.append(case)
            continue
        candidates: list[ConservativeCase] = []
        checked = 0
        for representative in reversed(active):
            if checked >= options.max_candidate_checks:
                break
            checked += 1
            if _case_close(case, representative, options):
                candidates.append(representative)
        if not candidates:
            case.reason = "no_near_neighbor"
            active.append(case)
            continue

        representative = min(
            candidates,
            key=lambda item: (
                _normalized_distance(case, item, options),
                _stable_digest(item, options.seed),
            ),
        )
        case.selected = False
        case.reason = "numeric_near_duplicate"
        case.representative_id = representative.case_id
        (
            case.differing_positions,
            case.max_absolute_delta,
            case.max_relative_delta,
        ) = _case_distance(case, representative, options)
        case.sequence_sum_relative_delta = _sequence_sum_relative_delta(
            case, representative, options
        )
        near_duplicates.append(case)

    if len(near_duplicates) > max_removed:
        removed_ids = {
            case.case_id
            for case in sorted(
                near_duplicates,
                key=lambda item: _stable_digest(item, options.seed + "\0removal-cap"),
            )[:max_removed]
        }
        for case in near_duplicates:
            if case.case_id in removed_ids:
                continue
            case.selected = True
            case.reason = "removal_cap"
            case.representative_id = None
            case.differing_positions = ()
            case.max_absolute_delta = 0.0
            case.max_relative_delta = 0.0
            case.sequence_sum_relative_delta = 0.0


def select_cases(cases: Sequence[ConservativeCase], options: ConservativeOptions) -> None:
    groups: dict[str, list[ConservativeCase]] = defaultdict(list)
    for case in cases:
        groups[case.signature].append(case)
    for group in groups.values():
        select_group(group, options)


def _build_report(
    cases: Sequence[ConservativeCase],
    preprocessing: dict[str, object],
    options: ConservativeOptions,
) -> dict[str, object]:
    kept = [case for case in cases if case.selected]
    input_count = int(preprocessing["config_occurrences"])
    by_api: dict[str, dict[str, int | float]] = {}
    api_names = sorted({case.api_name or "<unparsed>" for case in cases})
    for api_name in api_names:
        api_cases = [case for case in cases if (case.api_name or "<unparsed>") == api_name]
        api_kept = sum(case.selected for case in api_cases)
        models = sorted({case.model_name for case in api_cases})
        by_api[api_name] = {
            "unique": len(api_cases),
            "kept": api_kept,
            "excluded": len(api_cases) - api_kept,
            "post_dedup_retention_rate": api_kept / len(api_cases),
            "models": models,
        }

    source_preprocessing = preprocessing["by_source"]
    assert isinstance(source_preprocessing, dict)
    by_source = {}
    for source, stats in source_preprocessing.items():
        source_cases = [case for case in cases if str(case.source) == source]
        source_kept = sum(case.selected for case in source_cases)
        by_source[source] = {
            **stats,
            "kept": source_kept,
            "excluded": len(source_cases) - source_kept,
            "post_dedup_retention_rate": (source_kept / len(source_cases) if source_cases else 1.0),
        }

    unique_count = len(cases)
    reason_counts = Counter(case.reason for case in cases)
    excluded_count = unique_count - len(kept)
    excluded_coverage = sum(
        count for reason, count in reason_counts.items() if reason.endswith("_sampled_out")
    )
    unmodeled_apis = sorted(
        {case.api_name or "<unparsed>" for case in cases if case.model_name == "unmodeled_preserve"}
    )
    by_model: dict[str, dict[str, int | float]] = {}
    for model_name in sorted({case.model_name for case in cases}):
        model_cases = [case for case in cases if case.model_name == model_name]
        model_kept = sum(case.selected for case in model_cases)
        by_model[model_name] = {
            "unique": len(model_cases),
            "kept": model_kept,
            "excluded": len(model_cases) - model_kept,
            "retention_rate": model_kept / len(model_cases),
        }
    return {
        "input_cases": input_count,
        "unique_cases": unique_count,
        "exact_duplicates_removed": int(preprocessing["exact_duplicates_removed"]),
        "kept_cases": len(kept),
        "excluded_cases": excluded_count,
        "excluded_near_duplicates": reason_counts["numeric_near_duplicate"],
        "excluded_coverage_cases": excluded_coverage,
        "excluded_simple_kernel_cases": reason_counts["simple_coverage_sampled_out"],
        "total_removed_occurrences": input_count - len(kept),
        "retention_rate_from_original": len(kept) / input_count if input_count else 1.0,
        "post_dedup_retention_rate": len(kept) / unique_count if unique_count else 1.0,
        "api_count": len(api_names),
        "structural_group_count": len({case.signature for case in cases}),
        "decision_reasons": dict(sorted(reason_counts.items())),
        "unmodeled_apis": unmodeled_apis,
        "by_model": by_model,
        "by_source": by_source,
        "by_api": by_api,
        "preprocessing": preprocessing,
        "settings": {
            "relative_tolerance": options.relative_tolerance,
            "absolute_tolerance": options.absolute_tolerance,
            "max_absolute_delta": options.max_absolute_delta,
            "exact_small_integer": options.exact_small_integer,
            "exact_integer_above": options.exact_integer_above,
            "max_removal_rate": options.max_removal_rate,
            "min_group_size": options.min_group_size,
            "max_candidate_checks": options.max_candidate_checks,
            "seed": options.seed,
            "protect_boundaries": options.protect_boundaries,
            "complex_api_profiles": options.complex_api_profiles,
            "moe_sequence_relative_tolerance": (options.moe_sequence_relative_tolerance),
            "moe_sequence_max_absolute_delta": (options.moe_sequence_max_absolute_delta),
            "moe_sequence_sum_relative_tolerance": (options.moe_sequence_sum_relative_tolerance),
            "api_model_registry_size": len(modeled_api_names()),
        },
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


def _render_lines(lines: dict[int, str], sort_output: bool) -> str:
    ordered = (
        sorted(lines.values()) if sort_output else [lines[line_no] for line_no in sorted(lines)]
    )
    return "".join(f"{line}\n" for line in ordered)


def _output_paths(inputs: Sequence[Path], output_dir: Path, label: str) -> dict[Path, Path]:
    names = [f"{path.stem}_{label}{path.suffix}" for path in inputs]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ValueError(f"input basenames collide in output directory: {', '.join(duplicates)}")
    return {path: output_dir / name for path, name in zip(inputs, names, strict=True)}


def write_outputs(
    cases: Sequence[ConservativeCase],
    inputs: Sequence[Path],
    passthrough: dict[Path, dict[int, str]],
    output_dir: Path,
    report: dict[str, object],
    force: bool,
    sort_output: bool,
) -> dict[Path, Path]:
    slim_paths = _output_paths(inputs, output_dir, "conservative_slim")
    dedup_paths = _output_paths(inputs, output_dir, "conservative_deduplicated")
    excluded_paths = _output_paths(inputs, output_dir, "conservative_excluded")
    report_path = output_dir / "conservative_report.json"
    audit_path = output_dir / "conservative_decisions.tsv"
    targets = [
        *slim_paths.values(),
        *dedup_paths.values(),
        *excluded_paths.values(),
        report_path,
        audit_path,
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing output: " + ", ".join(map(str, existing))
        )

    for source in inputs:
        retained = dict(passthrough.get(source, {}))
        retained.update(
            {case.line_no: case.text for case in cases if case.source == source and case.selected}
        )
        deduplicated = dict(passthrough.get(source, {}))
        deduplicated.update({case.line_no: case.text for case in cases if case.source == source})
        excluded = {
            case.line_no: case.text for case in cases if case.source == source and not case.selected
        }
        _atomic_write(slim_paths[source], _render_lines(retained, sort_output))
        _atomic_write(dedup_paths[source], _render_lines(deduplicated, sort_output))
        _atomic_write(excluded_paths[source], _render_lines(excluded, sort_output))

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
                "model",
                "decision",
                "reason",
                "representative_id",
                "differing_number_positions",
                "max_absolute_delta",
                "max_relative_delta",
                "sequence_sum_relative_delta",
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
                    case.model_name,
                    "keep" if case.selected else "remove",
                    case.reason,
                    "" if case.representative_id is None else case.representative_id,
                    ",".join(map(str, case.differing_positions)),
                    f"{case.max_absolute_delta:g}",
                    f"{case.max_relative_delta:.12g}",
                    f"{case.sequence_sum_relative_delta:.12g}",
                    case.duplicate_occurrences,
                    case.text,
                ]
            )
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(audit_path)
    return slim_paths


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
        description="Conservative numeric near-duplicate slimming for API configs."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("conservative_config_slimmer_output"),
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--max-absolute-delta", type=float, default=256.0)
    parser.add_argument("--exact-small-integer", type=int, default=16)
    parser.add_argument("--exact-integer-above", type=int, default=1_000_000_000)
    parser.add_argument("--max-removal-rate", type=float, default=0.20)
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--max-candidate-checks", type=int, default=256)
    parser.add_argument("--seed", default="conservative-config-slimmer-v1")
    parser.add_argument("--no-boundary-protection", action="store_true")
    parser.add_argument("--no-complex-api-profiles", action="store_true")
    parser.add_argument("--moe-sequence-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--moe-sequence-max-absolute-delta", type=float, default=1024.0)
    parser.add_argument("--moe-sequence-sum-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--preserve-api", action="append", default=[], metavar="REGEX")
    parser.add_argument("--pin-file", action="append", default=[], type=Path)
    parser.add_argument("--preserve-input-order", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _options_from_args(args: argparse.Namespace) -> ConservativeOptions:
    if args.relative_tolerance < 0:
        raise ValueError("--relative-tolerance must be non-negative")
    if args.absolute_tolerance < 0:
        raise ValueError("--absolute-tolerance must be non-negative")
    if args.max_absolute_delta < 0:
        raise ValueError("--max-absolute-delta must be non-negative")
    if args.exact_small_integer < 0:
        raise ValueError("--exact-small-integer must be non-negative")
    if args.exact_integer_above < 1:
        raise ValueError("--exact-integer-above must be at least 1")
    if not 0 <= args.max_removal_rate <= 1:
        raise ValueError("--max-removal-rate must be between 0 and 1")
    if args.min_group_size < 2:
        raise ValueError("--min-group-size must be at least 2")
    if args.max_candidate_checks < 1:
        raise ValueError("--max-candidate-checks must be at least 1")
    if args.moe_sequence_relative_tolerance < 0:
        raise ValueError("--moe-sequence-relative-tolerance must be non-negative")
    if args.moe_sequence_max_absolute_delta < 0:
        raise ValueError("--moe-sequence-max-absolute-delta must be non-negative")
    if not 0 <= args.moe_sequence_sum_relative_tolerance <= 1:
        raise ValueError("--moe-sequence-sum-relative-tolerance must be between 0 and 1")
    return ConservativeOptions(
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
        max_absolute_delta=args.max_absolute_delta,
        exact_small_integer=args.exact_small_integer,
        exact_integer_above=args.exact_integer_above,
        max_removal_rate=args.max_removal_rate,
        min_group_size=args.min_group_size,
        max_candidate_checks=args.max_candidate_checks,
        seed=args.seed,
        protect_boundaries=not args.no_boundary_protection,
        complex_api_profiles=not args.no_complex_api_profiles,
        moe_sequence_relative_tolerance=args.moe_sequence_relative_tolerance,
        moe_sequence_max_absolute_delta=args.moe_sequence_max_absolute_delta,
        moe_sequence_sum_relative_tolerance=(args.moe_sequence_sum_relative_tolerance),
        preserve_patterns=_compile_patterns(args.preserve_api, "--preserve-api"),
        pinned_lines=_load_pinned_lines(args.pin_file),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    for path in [*args.inputs, *args.pin_file]:
        if not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")
    options = _options_from_args(args)
    cases, passthrough, preprocessing = load_cases(args.inputs)
    select_cases(cases, options)
    report = _build_report(cases, preprocessing, options)
    report["settings"]["sorted_outputs"] = not args.preserve_input_order
    if not args.dry_run:
        write_outputs(
            cases,
            args.inputs,
            passthrough,
            args.output_dir,
            report,
            force=args.force,
            sort_output=not args.preserve_input_order,
        )
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
        print("dry-run: no files were written")
    else:
        print(f"outputs written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
