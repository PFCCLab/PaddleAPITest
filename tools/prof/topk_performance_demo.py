from __future__ import annotations

import time

import numpy
import paddle
import torch

device = torch.device("cuda:0")
torch.set_default_device(device)
paddle.device.set_device("gpu:0")


def init_input(numpy_tensor):
    paddle_x = paddle.to_tensor(numpy_tensor)
    torch_x = torch.tensor(numpy_tensor, requires_grad=True)
    paddle_x.stop_gradient = False

    numpy.testing.assert_allclose(
        paddle_x.numpy(),
        torch_x.cpu().detach().numpy(),
        1e-10,
        1e-10,
        err_msg="intput diff",
    )
    return paddle_x, torch_x


# paddle.topk(Tensor([4, 205923],"float32"), k=1, axis=0, )

m = 4
n = 205923
k = 1
test_loop = 240662
numpy_tensor = (numpy.random.random([m, n]) - 0.5).astype("float32")
paddle_x, torch_x = init_input(numpy_tensor)
numel = numpy_tensor.size
test_loop = 2147483647 * 20 // numel
print("numel=", numel, "test_loop=", test_loop)

print(torch_x.device)

paddle_out = paddle.topk(paddle_x, k, axis=0)

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.topk(paddle_x, k, axis=0)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle forward", timeused)

numpy_tensor1 = (numpy.random.random(paddle_out[0].shape) - 0.5).astype("float32")
numpy_tensor2 = (numpy.random.random(paddle_out[1].shape) - 0.5).astype("float32")
paddle_grad1, torch_grad1 = init_input(numpy_tensor1)
paddle_grad2, torch_grad2 = init_input(numpy_tensor2)

try:
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.grad(
            [paddle_out],
            [paddle_x],
            grad_outputs=[paddle_grad1, paddle_grad2],
            allow_unused=True,
        )
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle backward", timeused)
except Exception:
    print("paddle 反向失败")


torch_out = torch.topk(torch_x, k, axis=0)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _i in range(test_loop):
        torch.topk(torch_x, k, axis=0)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)
try:
    torch.cuda.synchronize()
    start = time.time()
    # for i in range(test_loop):
    #     torch.autograd.grad(
    #         [torch_out],
    #         [torch_x1, torch_x2],
    #         grad_outputs=torch_grad,
    #         retain_graph=True,
    #     )
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch backward", timeused)
except Exception:
    print("torch 反向失败")
