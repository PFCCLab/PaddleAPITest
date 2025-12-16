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


# paddle.concat(list[Tensor([4, 178176, 1],"float32"),Tensor([4, 44544, 1],"float32"),Tensor([4, 11136, 1],"float32"),Tensor([4, 2784, 1],"float32"),Tensor([4, 720, 1],"float32"),], axis=1, )

m = 4
n1 = 178176
n2 = 44544
n3 = 11136
n4 = 2784
n5 = 720
k = 1

test_loop = 240662
numpy_tensor1 = (numpy.random.random([m, n1, k]) - 0.5).astype("float32")
numpy_tensor2 = (numpy.random.random([m, n2, k]) - 0.5).astype("float32")
numpy_tensor3 = (numpy.random.random([m, n3, k]) - 0.5).astype("float32")
numpy_tensor4 = (numpy.random.random([m, n4, k]) - 0.5).astype("float32")
numpy_tensor5 = (numpy.random.random([m, n5, k]) - 0.5).astype("float32")
paddle_x1, torch_x1 = init_input(numpy_tensor1)
paddle_x2, torch_x2 = init_input(numpy_tensor2)
paddle_x3, torch_x3 = init_input(numpy_tensor3)
paddle_x4, torch_x4 = init_input(numpy_tensor4)
paddle_x5, torch_x5 = init_input(numpy_tensor5)
numel = (
    numpy_tensor1.size
    + numpy_tensor2.size
    + numpy_tensor3.size
    + numpy_tensor4.size
    + numpy_tensor5.size
)
test_loop = 2147483647 * 20 // numel
print("numel=", numel, "test_loop=", test_loop)

print(torch_x1.device)

paddle_out = paddle.concat((paddle_x1, paddle_x2, paddle_x3, paddle_x4, paddle_x5), axis=1)

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.concat((paddle_x1, paddle_x2, paddle_x3, paddle_x4, paddle_x5), axis=1)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle forward", timeused)

numpy_tensor = (numpy.random.random(paddle_out.shape) - 0.5).astype("float32")
paddle_grad, torch_grad = init_input(numpy_tensor)

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

torch_out = torch.concat((torch_x1, torch_x2, torch_x3, torch_x4, torch_x5), axis=1)

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _i in range(test_loop):
        torch.concat((torch_x1, torch_x2, torch_x3, torch_x4, torch_x5), axis=1)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)

torch.cuda.synchronize()
start = time.time()
for _i in range(test_loop):
    torch.autograd.grad(
        [torch_out],
        [torch_x1, torch_x2, torch_x3, torch_x4, torch_x5],
        grad_outputs=torch_grad,
        retain_graph=True,
    )
torch.cuda.synchronize()
end = time.time()
timeused = end - start
print("torch backward", timeused)
