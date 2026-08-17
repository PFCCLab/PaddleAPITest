"""日志格式、分类和比较维度协议。"""

from __future__ import annotations

from typing import Literal

LogType = Literal[
    "checkpoint",
    "pass",
    "skip",
    "paddle_error",
    "paddle_accuracy",
    "paddle_bitwise_knows",
    "paddle_bitwise",
    "paddle_cuda",
    "paddle_crash",
    "oom",
    "timeout",
    "torch_error",
    "config_input",
    "config_parse",
    "config_convert",
]

LOG_PREFIXES: dict[LogType, str] = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "skip": "api_config_skip",
    "paddle_error": "api_config_paddle_error",
    "paddle_accuracy": "api_config_paddle_accuracy",
    "paddle_bitwise": "api_config_paddle_bitwise",
    "paddle_bitwise_knows": "api_config_paddle_bitwise_knows",
    "paddle_cuda": "api_config_paddle_cuda",
    "paddle_crash": "api_config_paddle_crash",
    "oom": "api_config_oom",
    "timeout": "api_config_timeout",
    "torch_error": "api_config_torch_error",
    "config_input": "api_config_config_input",
    "config_parse": "api_config_config_parse",
    "config_convert": "api_config_config_convert",
}

TERMINAL_LOG_TYPES = frozenset(LOG_PREFIXES) - {"checkpoint"}
RESULT_LOG_PREFIXES = {
    log_type: prefix for log_type, prefix in LOG_PREFIXES.items() if log_type in TERMINAL_LOG_TYPES
}
COMP_TO_DIMENSION = {
    "P1T1": "accuracy",
    "P2T2": "accuracy",
    "P2T1": "accuracy",
    "P1T2": "accuracy",
    "P1T1B": "accuracy_backward",
    "P2T2B": "accuracy_backward",
    "P2T1B": "accuracy_backward",
    "P1T2B": "accuracy_backward",
    "T1T2": "torch_stable",
    "T1T2B": "torch_stable_backward",
    "P1P2": "paddle_stable",
    "P1P2B": "paddle_stable_backward",
}
COMP_SUMMARY_PAIRS = (
    ("P1T1", "P1T1B"),
    ("P2T2", "P2T2B"),
    ("P2T1", "P2T1B"),
    ("P1T2", "P1T2B"),
    ("T1T2", "T1T2B"),
    ("P1P2", "P1P2B"),
)
ALL_DIMENSIONS = sorted(set(COMP_TO_DIMENSION.values()))
FINAL_RESULT_PRIORITY = (
    "paddle_cuda",
    "paddle_crash",
    "oom",
    "timeout",
    "paddle_error",
    "paddle_accuracy",
    "paddle_bitwise_knows",
    "paddle_bitwise",
    "torch_error",
    "config_input",
    "config_parse",
    "config_convert",
    "skip",
    "pass",
)
TOL_HEADER = ["API", "config", "dtype", "mode", "max_abs_diff", "max_rel_diff"]
STABLE_HEADER = ["API", "config", "dtype", "comp", "max_abs_diff", "max_rel_diff"]
CASE_BEGIN_TAG = ">>> CASE"
CASE_END_TAG = "<<< CASE"
MAX_CSV_CONFIG_LENGTH = 120000
