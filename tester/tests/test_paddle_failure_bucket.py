"""Torch 跑通、Paddle 失败时的分桶回归测试。

Paddle 自报的 (InvalidArgument) 不能算「语料非法」：同一份配置 Torch 已经跑出结果，
这种单侧拒绝必须落到 paddle_error，否则会被当成脏语料过滤掉。

不初始化完整 tester，只把被测方法绑到 ``object.__new__(APITestAccuracy)``
的裸实例上，其余协作者用 stub 替换。

运行：``python -m pytest tester/tests/test_paddle_failure_bucket.py -q``
"""

from __future__ import annotations

import pytest
from tester.accuracy import APITestAccuracy


class _Recorder:
    def __init__(self):
        self.results = []
        self.finalized = []

    def report_case_result(self, log_type, message=None, *, stage=None, error=None, **_):
        self.results.append({"log_type": log_type, "stage": stage})
        return log_type

    def dump_finalize(self, log_type, **_):
        self.finalized.append(log_type)


def _raiser(err_msg):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(err_msg)

    return _raise


def _make_case():
    """构造「Torch 已跑通」的裸实例：_torch_failure 为 None 即非合并路径。"""
    case = object.__new__(APITestAccuracy)
    case._torch_failure = None
    case._paddle_failure = None
    case._merge_terminal_written = False
    case.clear_output_grad_cache = lambda: None
    case.clear_runtime_inputs = lambda *_a, **_k: None
    case.dump_error = lambda *_a, **_k: None
    case.dump_event = lambda *_a, **_k: None
    recorder = _Recorder()
    case.report_case_result = recorder.report_case_result
    case.dump_finalize = recorder.dump_finalize
    return case, recorder


# Paddle 用来表达「输入不合法」的三种 tag，在 Torch 已跑通的前提下都属于单侧拒绝。
PADDLE_REFUSALS = [
    "(InvalidArgument) The rows of X should be positive, but received 0.",
    "(PreconditionNotMet) Tensor holds no memory.",
    "(OutOfRange) The index is out of range.",
]


@pytest.mark.parametrize("err_msg", PADDLE_REFUSALS)
def test_paddle_forward_refusal_is_paddle_error(err_msg):
    case, recorder = _make_case()
    case.build_paddle_input = _raiser(err_msg)
    success, _ = case.get_paddle_output()
    assert success is False
    assert [r["log_type"] for r in recorder.results] == ["paddle_error"]
    assert recorder.finalized == ["paddle_error"]


@pytest.mark.parametrize("err_msg", PADDLE_REFUSALS)
def test_paddle_backward_refusal_is_paddle_error(err_msg):
    case, recorder = _make_case()
    case.get_paddle_input_list = _raiser(err_msg)
    success, _ = case.get_paddle_grad(None)
    assert success is False
    assert [r["log_type"] for r in recorder.results] == ["paddle_error"]
    assert recorder.finalized == ["paddle_error"]


@pytest.mark.parametrize(
    ("err_msg", "expected", "expect_raise"),
    [
        # (Unimplemented) 仍表示无法有效验证，不是 Paddle 拒绝输入。
        ("(Unimplemented) not supported on this place", "skip", False),
        ("CUDA out of memory. Tried to allocate 2.00 GiB", "oom", True),
        ("CUDA error: an illegal memory access was encountered", "paddle_cuda", True),
        ("there are NAN or INF in output", "paddle_error", False),
        ("plain runtime boom", "paddle_error", False),
    ],
)
def test_other_paddle_failures_unchanged(err_msg, expected, expect_raise):
    case, recorder = _make_case()
    case.build_paddle_input = _raiser(err_msg)
    if expect_raise:
        with pytest.raises(RuntimeError):
            case.get_paddle_output()
    else:
        success, _ = case.get_paddle_output()
        assert success is False
    assert [r["log_type"] for r in recorder.results] == [expected]


def test_input_stage_still_config_input():
    """造数阶段的 InvalidArgument 仍是语料问题：两框架都还没跑，不受本次改动影响。"""
    case, recorder = _make_case()
    case.generate_input_values = _raiser("(InvalidArgument) shape is illegal")
    assert case._generate_input_values() is False
    assert [r["log_type"] for r in recorder.results] == ["config_input"]


def test_merge_path_still_config_input():
    """Torch 也挂时双侧 config_input 仍合并为 config_input，改动不得污染合并路径。"""
    case, _ = _make_case()
    case._torch_failure = ("config_input", "torch boom", "Torch forward")
    case._paddle_failure = ("config_input", "paddle boom", "Paddle forward")
    assert case._merge_torch_paddle_failure(False) == "config_input"
