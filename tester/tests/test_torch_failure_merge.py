"""Torch 失败后继续执行 Paddle 的合并终态回归测试。

不初始化完整 tester（依赖真实 paddle/torch 图），只把被测方法绑到
``object.__new__(APITestAccuracy)`` 的裸实例上，其余协作者用 stub 替换。

运行：``python -m pytest tester/tests/test_torch_failure_merge.py -q``
"""

from __future__ import annotations

import types

import pytest
import torch
from tester.accuracy import APITestAccuracy


class _Recorder:
    def __init__(self):
        self.results = []
        self.finalized = []

    def report_case_result(self, log_type, message=None, *, stage=None, error=None, **_):
        self.results.append({"log_type": log_type, "stage": stage, "error": error})
        return log_type

    def dump_finalize(self, log_type, **_):
        self.finalized.append(log_type)


def _make_case(torch_failure, paddle_failure=None, merge_terminal_written=False):
    case = object.__new__(APITestAccuracy)
    case._torch_failure = torch_failure
    case._paddle_failure = paddle_failure
    case._merge_terminal_written = merge_terminal_written
    case.use_gpu_mode = False
    case.is_backward = False
    case.clear_runtime_inputs = lambda *_args, **_kwargs: None
    case.clear_output_grad_cache = lambda: None
    recorder = _Recorder()
    case.report_case_result = recorder.report_case_result
    case.dump_finalize = recorder.dump_finalize
    return case, recorder


@pytest.mark.parametrize(
    ("paddle_success", "torch_type", "paddle_type", "expected"),
    [
        (True, "torch_error", None, "torch_only_error"),
        (False, "config_input", "config_input", "config_input"),
        (False, "config_input", "paddle_error", "both_error"),
        (False, "torch_error", "paddle_error", "both_error"),
        # 既定语义：Paddle 的 skip/未实现也参与合并，归 both_error。
        (False, "torch_error", "skip", "both_error"),
    ],
)
def test_merge_matrix(paddle_success, torch_type, paddle_type, expected):
    case, _ = _make_case(("torch_error", "boom", "Torch forward"))
    case._torch_failure = (torch_type, "torch boom", "Torch forward")
    if not paddle_success:
        case._paddle_failure = (paddle_type, "paddle boom", "Paddle forward")
    assert case._merge_torch_paddle_failure(paddle_success) == expected


def test_merged_terminal_carries_torch_stage():
    case, recorder = _make_case(("torch_error", "build failed", "Input"))
    case.get_paddle_output = lambda: (True, None)
    case.need_check_grad = lambda: False

    case._test_paddle_after_torch_failure(probe_bytes=0)

    assert len(recorder.results) == 1
    result = recorder.results[0]
    assert result["log_type"] == "torch_only_error"
    # 回归点：合并终态必须带上 torch 失败阶段，否则按 stage 归因的下游读不到。
    assert result["stage"] == "Input"
    assert "[paddle] forward/backward succeeded" in result["error"]


def test_terminal_written_blocks_second_merge():
    case, recorder = _make_case(
        ("torch_error", "boom", "Torch forward"), merge_terminal_written=True
    )
    case.get_paddle_output = lambda: (True, None)
    case.need_check_grad = lambda: False

    case._test_paddle_after_torch_failure(probe_bytes=0)

    # fatal/skip 已写过终态，收尾不得再落一条分类。
    assert recorder.results == []
    assert case._torch_failure is None


def test_torch_backward_failure_defers_to_paddle(monkeypatch):
    case, recorder = _real_case(None)
    case.need_check_grad = lambda: True
    case.get_torch_input_list = lambda: [torch.zeros(1, requires_grad=True)]
    case.gen_torch_output_and_output_grad = lambda _out: ([_out], [torch.ones_like(_out)])
    case.torch_args = ()
    case.torch_kwargs = {}
    case.check_torch_operator_cuda_error = lambda: None
    case.torch_operator_device = lambda: torch.device("cpu")
    case.build_torch_input = lambda: True
    case.converter = types.SimpleNamespace(execute=lambda *_a, **_k: torch.zeros(1))
    case.paddle_api = lambda *_a, **_k: "out"

    def _backward_fails(**_kwargs):
        raise RuntimeError("backward is not implemented")

    monkeypatch.setattr(torch.autograd, "grad", _backward_fails)
    monkeypatch.setattr(
        "tester.accuracy.bind_paddle_arguments", lambda *_a, **_k: types.SimpleNamespace()
    )

    success, _output, _grads, grad_success = case.get_torch_output(None)

    # 反向失败先不写终态，只留下可合并的 torch 失败。
    assert recorder.results == [], case._torch_failure
    assert case._torch_failure[0] == "torch_error"
    assert case._torch_failure[2] == APITestAccuracy.STAGE_TORCH_BACKWARD, case._torch_failure

    case.need_check_grad = lambda: False
    case._test_paddle_after_torch_failure(0)

    assert [result["log_type"] for result in recorder.results] == ["torch_only_error"], (
        recorder.results
    )
    assert recorder.results[0]["stage"] == APITestAccuracy.STAGE_TORCH_BACKWARD
    case, _ = _make_case(("torch_error", "boom", "Torch forward"))
    case.paddle_args = ()
    case.paddle_kwargs = {}
    case.get_paddle_input_list = lambda: []
    case.gen_paddle_output_and_output_grad = lambda _out: ([], [])
    case.enforce_paddle_backward_capacity = lambda *_args: None

    def _raise_cuda_error():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    case.check_paddle_kernel_cuda_error = _raise_cuda_error

    calls = []
    case._record_paddle_failure_or_raise = lambda err, default, stage, **kw: (
        calls.append((default, stage)) or "raise"
    )
    fallback = []
    case._report_runtime_error_and_finalize = lambda *args, **kwargs: fallback.append(args)

    with pytest.raises(RuntimeError, match="CUDA error"):
        case.get_paddle_grad(None)

    # 反向同步与前向同步保持一致：合并模式下走统一 handler。
    assert calls == [("paddle_cuda", APITestAccuracy.STAGE_PADDLE_BACKWARD_SYNC)]
    assert fallback == []


# ---- 真实链路：不桩分类与合并 handler，只桩环境副作用 ----


def _real_case(torch_failure):
    """构造能跑真实 get_paddle_output / describe_runtime_error 的裸实例。"""
    case, recorder = _make_case(torch_failure)
    case.api_config = types.SimpleNamespace(api_name="paddle.example")
    case.test_amp = False
    case.paddle_args = ()
    case.paddle_kwargs = {}
    case.dump_error = lambda *_a, **_k: None
    case.dump_event = lambda *_a, **_k: None
    case.dump_save = lambda *_a, **_k: None
    case.reset_random_state = lambda: None
    case.clear_generated_input_values = lambda: None
    case.build_paddle_input = lambda: True
    case.need_check_grad = lambda: False
    case.check_paddle_kernel_cuda_error = lambda: None
    return case, recorder


def test_real_chain_paddle_success_is_torch_only_error():
    case, recorder = _real_case(("torch_error", "torch boom", "Torch forward"))
    case.paddle_api = lambda *_a, **_k: "out"

    case._test_paddle_after_torch_failure(probe_bytes=0)

    # 真实分类 + 真实合并出口：恰好一条终态，且带 torch 失败阶段。
    assert [result["log_type"] for result in recorder.results] == ["torch_only_error"]
    assert recorder.results[0]["stage"] == "Torch forward"
    assert recorder.finalized == ["torch_only_error"]


def test_real_chain_both_invalid_config():
    case, recorder = _real_case(("config_input", "torch boom", "Torch forward"))

    def _invalid(*_a, **_k):
        raise RuntimeError("(InvalidArgument) The shape of input is invalid")

    case.paddle_api = _invalid

    case._test_paddle_after_torch_failure(probe_bytes=0)

    # 两侧都被真实分类器识别为非法 config 时合并为 config_input，只落一条。
    assert [result["log_type"] for result in recorder.results] == ["config_input"]
    assert "[torch]" in recorder.results[0]["error"]
    assert "[paddle]" in recorder.results[0]["error"]


def test_real_chain_cuda_failure_writes_single_terminal():
    case, recorder = _real_case(("torch_error", "torch boom", "Torch forward"))

    def _cuda(*_a, **_k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    case.paddle_api = _cuda

    with pytest.raises(RuntimeError, match="CUDA error"):
        case._test_paddle_after_torch_failure(probe_bytes=0)

    # fatal 由真实 classify_runtime_error 判定；异常穿出后不得再合并出第二条。
    assert [result["log_type"] for result in recorder.results] == ["paddle_cuda"]
    assert recorder.finalized == ["paddle_cuda"]
    assert case._merge_terminal_written is True
