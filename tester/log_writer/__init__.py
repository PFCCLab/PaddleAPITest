"""Paddle API 测试日志包。"""

from __future__ import annotations

from . import log_aggregation as _aggregation
from . import log_runtime as _runtime
from . import log_worker as _worker


def init_log(log_dir):
    """初始化日志路径及进程内状态。"""
    _runtime._configure_log(log_dir)
    _worker._reset_worker_state()
    _aggregation._reset_aggregation_state()


__all__ = ("init_log",)
