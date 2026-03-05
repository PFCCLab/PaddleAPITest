from __future__ import annotations

import time

import numpy
import paddle
import torch

device = torch.device("cuda:0")
torch.set_default_device(device)

# paddle.device.set_device('cpu')


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


test_loop = 240662
# paddle.nn.functional.relu(Tensor([2, 528, 13, 13],"float32"), )
numpy_tensor = (numpy.random.random([2, 528, 13, 13]) - 0.5).astype("float32")
paddle_x, torch_x = init_input(numpy_tensor)

paddle_out = paddle.nn.functional.relu(paddle_x)

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.nn.functional.relu(paddle_x)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle forward", timeused)

numpy_tensor = (numpy.random.random([2, 528, 13, 13]) - 0.5).astype("float32")
paddle_grad, torch_grad = init_input(numpy_tensor)

paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
start = time.time()
for _i in range(test_loop):
    paddle.grad([paddle_out], [paddle_x], grad_outputs=paddle_grad, allow_unused=True)
paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
end = time.time()
timeused = end - start
print("paddle backward", timeused)

torch_out = torch.nn.functional.relu(torch_x)

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _i in range(test_loop):
        torch.nn.functional.relu(torch_x)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)

torch.cuda.synchronize()
start = time.time()
for _i in range(test_loop):
    torch.autograd.grad([torch_out], [torch_x], grad_outputs=torch_grad, retain_graph=True)
torch.cuda.synchronize()
end = time.time()
timeused = end - start
print("torch backward", timeused)
