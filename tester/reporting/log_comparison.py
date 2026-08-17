"""精度与稳定性比较的详细日志和 CSV 记录。"""

from __future__ import annotations

import re

from . import log_runtime as runtime
from .log_schema import MAX_CSV_CONFIG_LENGTH, STABLE_HEADER, TOL_HEADER
from .log_worker import _record_case_comparison, write_to_comp_log

_DIFF_VALUE = r"(\d+\.?\d*(?:[eE][+-]?\d+)?|nan|inf)\b"
_TOL_ABS_PATTERN = rf"(?:Absolute|Greatest absolute) difference: {_DIFF_VALUE}"
_TOL_REL_PATTERN = rf"(?:Relative|Greatest relative) difference: {_DIFF_VALUE}"
_STABLE_ABS_PATTERN = (
    rf"(?:Absolute|Greatest absolute|Max absolute) "
    rf"difference(?: among violations)?: {_DIFF_VALUE}"
)
_STABLE_REL_PATTERN = (
    rf"(?:Relative|Greatest relative|Max relative) "
    rf"difference(?: among violations)?: {_DIFF_VALUE}"
)
_FRAMEWORK_NAMES = {"P": "Paddle", "T": "Torch"}


def _get_diff(error_msg, abs_pattern, rel_pattern):
    """从断言消息中提取绝对误差和相对误差。"""
    if error_msg == "Identical":
        return 0.0, 0.0

    abs_match = re.search(abs_pattern, error_msg)
    rel_match = re.search(rel_pattern, error_msg)
    if not abs_match or not rel_match:
        return None, None
    try:
        return float(abs_match.group(1)), float(rel_match.group(1))
    except ValueError:
        return None, None


def log_accuracy_tolerance(
    error_msg,
    api,
    config,
    dtype,
    is_backward,
    *,
    tensor_index,
    tensor_count,
):
    """记录从 assert-close 失败消息中解析出的容差。"""
    mode = "backward" if is_backward else "forward"
    print(
        f"[tolerance] {mode} | tensor {tensor_index + 1}/{tensor_count} | {config}\n{error_msg}",
        flush=True,
    )
    max_abs_diff, max_rel_diff = _get_diff(error_msg, _TOL_ABS_PATTERN, _TOL_REL_PATTERN)
    row = [api, config, dtype, mode, str(max_abs_diff), str(max_rel_diff)]
    runtime.append_csv_row(
        runtime.RESULT_LOG_PATH / f"tol{runtime.RESULT_LOG_SUFFIX}.csv",
        TOL_HEADER,
        row,
    )


def _format_comp_line(
    comp,
    result,
    *,
    tensor_index,
    tensor_count,
    **details,
):
    """构造稳定、单行且便于 grep 的 accuracy_stable 比较标记。"""
    phase = "backward" if comp.endswith("B") else "forward"
    base_comp = comp.removesuffix("B")
    actual_kind, actual_run, expected_kind, expected_run = base_comp
    actual_source = f"{_FRAMEWORK_NAMES[actual_kind]}#{actual_run}"
    expected_source = f"{_FRAMEWORK_NAMES[expected_kind]}#{expected_run}"
    fields = [f"> COMP {comp}", result, phase]
    if tensor_index is not None and tensor_count is not None:
        fields.append(f"tensor {tensor_index + 1}/{tensor_count}")
    fields.append(f"{actual_source} vs {expected_source}")
    for key, value in details.items():
        display_value = str(value).replace("\r", "\\r").replace("\n", "\\n")
        if key == "reason":
            fields.append(display_value.replace("_", " "))
        else:
            fields.append(f"{key.replace('_', ' ')} {display_value}")
    return " | ".join(fields)


def log_comp_issue(comp, result, config, **kwargs):
    """打印非 bitwise 比较问题并写入对应维度分类。"""
    print("\n" + _format_comp_line(comp, result, **kwargs), flush=True)
    tensor_count = kwargs.get("tensor_count")
    if tensor_count is None:
        tensor_count = max(kwargs.get("actual_count", 1), kwargs.get("expected_count", 1))
    _record_case_comparison(comp, result, 0, tensor_count)
    write_to_comp_log(comp, result, config)


def log_accuracy_stable(
    error_msg,
    api,
    config,
    dtype,
    comp,
    *,
    tensor_index,
    tensor_count,
):
    """记录一个比较模式下的稳定性误差。"""
    if "\n" in error_msg:
        header = _format_comp_line(
            comp,
            "paddle_bitwise",
            tensor_index=tensor_index,
            tensor_count=tensor_count,
        )
        print(f"\n{header}\n{error_msg}", flush=True)
        _record_case_comparison(comp, "paddle_bitwise", 0, tensor_count)
        write_to_comp_log(comp, "paddle_bitwise", config)
    else:
        _record_case_comparison(comp, error_msg, 1, tensor_count)
    max_abs_diff, max_rel_diff = _get_diff(error_msg, _STABLE_ABS_PATTERN, _STABLE_REL_PATTERN)
    row = [
        api,
        config[:MAX_CSV_CONFIG_LENGTH],
        dtype,
        comp,
        str(max_abs_diff),
        str(max_rel_diff),
    ]
    runtime.append_csv_row(
        runtime.RESULT_LOG_PATH / f"stable{runtime.RESULT_LOG_SUFFIX}.csv",
        STABLE_HEADER,
        row,
    )
