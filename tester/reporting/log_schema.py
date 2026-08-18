"""日志格式、分类和比较维度协议。"""

from __future__ import annotations

from typing import Literal

Stage = Literal[
    "Input",
    "Paddle forward",
    "Paddle backward",
    "Paddle forward sync",
    "Paddle backward sync",
    "Torch forward",
    "Torch backward",
    "Torch forward sync",
    "Torch backward sync",
    "Compare forward",
    "Compare backward",
    "Memory preflight",
]

# 输入阶段覆盖参数解析和框架输入物化。
INPUT_STAGE: Stage = "Input"
# Paddle 前向阶段只表示算子执行本身。
PADDLE_FORWARD_STAGE: Stage = "Paddle forward"
# Paddle 反向阶段覆盖梯度计算与梯度收集。
PADDLE_BACKWARD_STAGE: Stage = "Paddle backward"
# 前向同步单独标记 CUDA 异步错误的落点。
PADDLE_FORWARD_SYNC_STAGE: Stage = "Paddle forward sync"
# 反向同步单独标记 CUDA 异步错误的落点。
PADDLE_BACKWARD_SYNC_STAGE: Stage = "Paddle backward sync"
# Torch 前向阶段与 Paddle 阶段使用同一命名规则。
TORCH_FORWARD_STAGE: Stage = "Torch forward"
# Torch 反向阶段与 Paddle 阶段使用同一命名规则。
TORCH_BACKWARD_STAGE: Stage = "Torch backward"
# Torch 前向同步保持框架信息，避免只显示 sync。
TORCH_FORWARD_SYNC_STAGE: Stage = "Torch forward sync"
# Torch 反向同步保持框架信息，避免只显示 sync。
TORCH_BACKWARD_SYNC_STAGE: Stage = "Torch backward sync"
# 前向比较错误属于比较阶段，不归入任一执行框架。
COMPARE_FORWARD_STAGE: Stage = "Compare forward"
# 反向比较错误属于比较阶段，不归入任一执行框架。
COMPARE_BACKWARD_STAGE: Stage = "Compare backward"
# 资源预检查失败发生在算子执行之前。
MEMORY_PREFLIGHT_STAGE: Stage = "Memory preflight"

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
