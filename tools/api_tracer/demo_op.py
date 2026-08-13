from __future__ import annotations

import torch
from api_tracer import APITracer


@torch.library.custom_op("custom_ops::custom_leaky_relu", mutates_args=())
def custom_leaky_relu(input: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return torch.where(input > 0, input, input * negative_slope)


@custom_leaky_relu.register_fake
def custom_leaky_relu_fake(input: torch.Tensor, negative_slope: float) -> torch.Tensor:
    return torch.empty_like(input)


def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
    (input,) = ctx.saved_tensors
    negative_slope = ctx.negative_slope
    grad_input = torch.where(input > 0, grad_output, grad_output * negative_slope)
    return grad_input, None


def setup_context(ctx, inputs, output):
    input, negative_slope = inputs
    ctx.save_for_backward(input)
    ctx.negative_slope = negative_slope


custom_leaky_relu.register_autograd(backward, setup_context=setup_context)


class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        return grad_input


def run_pytorch_code():
    print("--- [Demo] Running PyTorch code to be traced ---")
    # 1. 常规API调用
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cpu")
    b = b.to(device="cuda")
    c = a + b
    d = torch.relu(c)
    e = d + 10
    torch.nn.functional.softmax(e, dim=1)
    g = torch.cat((a, b), dim=0)
    g.argmax(dim=0)

    # # 2. 自定义算子调用
    print("--- [Demo] Calling custom operator ---")
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = CustomReLUFunction.apply(x)
    y.backward(torch.ones_like(y))
    y = custom_leaky_relu(x, 0.2)
    (_grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
    print("--- [Demo] PyTorch code finished ---")


def main():
    output_path = "tools/api_tracer/trace_output_test_op"
    with APITracer(
        "torch",
        output_path=output_path,
        levels=[0, 1],
        merge_output=True,
        record_stack=True,
    ):
        run_pytorch_code()


if __name__ == "__main__":
    main()
