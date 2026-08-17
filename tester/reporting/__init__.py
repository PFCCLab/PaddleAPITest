"""Paddle API 测试结果、日志和 dump 输出包。"""

# 各输出模块按用途拆分，调用方从本包获取稳定的日志接口。
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
