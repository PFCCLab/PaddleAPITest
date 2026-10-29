"""Paddle 全局 grad 开关泄漏的回归测试。

Paddle 组合反向（如 matmul_double_grad）抛异常时不会恢复 grad 开关，常驻 worker
后续 case 的输出都变成 stop_gradient。这里覆盖两处防御：case 构造时恢复开关，
以及 accuracy 反向阶段不再把"无可求导输出"静默当作反向成功。

运行：``python -m pytest tester/tests/test_paddle_grad_mode_leak.py -q``
"""

from __future__ import annotations

import types

import paddle
import pytest
from tester.accuracy import APITestAccuracy
from tester.base import APITestBase


@pytest.fixture(autouse=True)
def _restore_grad_mode():
    yield
    # 测试会主动关闭开关，结束后必须恢复，避免污染同进程的其他测试。
    paddle.set_grad_enabled(True)


def _leak_grad_mode_via_composite_backward():
    """用真实的 matmul_double_grad 失败制造泄漏，而不是直接调用 set_grad_enabled。"""

    def make(shape):
        tensor = paddle.randn(shape)
        tensor.stop_gradient = False
        return tensor

    x, y, grad_out = make([2, 3]), make([3, 4]), make([2, 4])
    dx, dy = paddle._C_ops.matmul_grad(x, y, grad_out, False, False)
    with pytest.raises(ValueError):
        # 故意传入不匹配的 grad_outputs，使组合反向在恢复开关前抛异常。
        paddle.grad(
            [dx, dy],
            [x, y, grad_out],
            grad_outputs=[paddle.ones([5, 7]), paddle.ones([9, 1])],
            allow_unused=True,
        )


def test_composite_backward_failure_leaks_grad_mode():
    _leak_grad_mode_via_composite_backward()

    # 前提：该 Paddle 行为存在时，下面两处防御才有意义。
    assert not paddle.is_grad_enabled()


def test_case_init_restores_grad_mode(capsys):
    _leak_grad_mode_via_composite_backward()
    api_config = types.SimpleNamespace(config='paddle.exp(Tensor([2],"float32"), )')
    runtime_config = types.SimpleNamespace(gpu_mode=None, test_cpu=True)

    APITestBase(api_config, use_torch=False, runtime_config=runtime_config)

    assert paddle.is_grad_enabled()
    assert "Paddle grad mode was disabled before case" in capsys.readouterr().out


def _grad_case(torch_failure):
    case = object.__new__(APITestAccuracy)
    case._torch_failure = torch_failure
    return case


def test_disabled_grad_mode_rejects_skipped_backward():
    paddle.set_grad_enabled(False)
    # Torch 已失败的合并路径同样要拒绝，否则 Paddle 会被当成成功而归为 torch_only_error。
    case = _grad_case(("torch_error", "boom", "Torch forward"))

    with pytest.raises(RuntimeError, match="grad mode is disabled"):
        case._check_paddle_backward_outputs([object()], [])


def test_missing_paddle_outputs_rejected_when_torch_has_grads():
    case = _grad_case(None)

    with pytest.raises(RuntimeError, match="no differentiable outputs"):
        case._check_paddle_backward_outputs([object()], [])


@pytest.mark.parametrize(
    ("inputs_list", "result_outputs", "torch_failure"),
    [
        # 没有需要梯度的输入：本来就不测反向。
        ([], [], None),
        # 有可求导输出：正常进入 paddle.grad。
        ([object()], [object()], None),
        # Torch 失败且开关正常：无参考值，无法判断 Paddle 无输出是否合理，保持原行为。
        ([object()], [], ("torch_error", "boom", "Torch forward")),
    ],
)
def test_check_passes_for_legitimate_cases(inputs_list, result_outputs, torch_failure):
    case = _grad_case(torch_failure)

    case._check_paddle_backward_outputs(inputs_list, result_outputs)
