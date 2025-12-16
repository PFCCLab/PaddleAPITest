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
    if numpy_tensor.dtype == "int32":
        torch_x = torch.tensor(numpy_tensor)
    else:
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


# paddle.bincount(Tensor([16000],"int32"), weights=Tensor([16000],"float32"), )

m = 16000
test_loop = 240662
numpy_tensor1 = numpy.random.randint(0, 10, size=(m)).astype("int32")
numpy_tensor2 = (numpy.random.random([m]) - 0.5).astype("float32")
paddle_x1, torch_x1 = init_input(numpy_tensor1)
paddle_x2, torch_x2 = init_input(numpy_tensor2)
numel = numpy_tensor1.size + numpy_tensor2.size
test_loop = 2147483647 * 20 // numel
print("numel=", numel, "test_loop=", test_loop)

print(torch_x1.device)

paddle_out = paddle.bincount(paddle_x1, paddle_x2)

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.bincount(paddle_x1, paddle_x2)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle forward", timeused)

numpy_tensor = (numpy.random.random(paddle_out.size) - 0.5).astype("float32")
paddle_grad, torch_grad = init_input(numpy_tensor)

try:
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.grad(
            [paddle_out],
            [paddle_x1, paddle_x2],
            grad_outputs=paddle_grad,
            allow_unused=True,
        )
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle backward", timeused)
except Exception:
    print("paddle 反向失败")

torch_out = torch.bincount(torch_x1, torch_x2)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _i in range(test_loop):
        torch.bincount(torch_x1, torch_x2)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)

try:
    torch.cuda.synchronize()
    start = time.time()
    for _i in range(test_loop):
        torch.autograd.grad(
            [torch_out],
            [torch_x1, torch_x2],
            grad_outputs=torch_grad,
            retain_graph=True,
        )
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch backward", timeused)
except Exception:
    print("paddle 反向失败")
