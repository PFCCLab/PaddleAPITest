from __future__ import annotations

import time

import numpy
import paddle
import torch

device = torch.device("cuda:0")
torch.set_default_device(device)
paddle.device.set_device("gpu:0")


def fused_dropout_add(x, y, p=0.5, training=True, mode="upscale_in_train"):
    if mode == "upscale_in_train":
        x = torch.nn.functional.dropout(x, p=p, training=training)
        x = x + y
    else:
        # handle downscale dropout
        mask = torch.bernoulli(torch.full(x.shape, 1 - p)).to(x.device)
        x = x * mask if training else x * (1 - p)
        x = x + y
    return x


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


# paddle.incubate.nn.functional.fused_dropout_add(Tensor([384, 1280],"float32"), Tensor([384, 1280],"float32"), p=0.0, training=True, mode="upscale_in_train", name=None, )

m = 384
n = 1280
test_loop = 240662
numpy_tensor1 = (numpy.random.random([m, n]) - 0.5).astype("float32")
numpy_tensor2 = (numpy.random.random([m, n]) - 0.5).astype("float32")
paddle_x1, torch_x1 = init_input(numpy_tensor1)
paddle_x2, torch_x2 = init_input(numpy_tensor2)
numel = numpy_tensor1.size + numpy_tensor2.size
test_loop = 2147483647 * 20 // numel
print("numel=", numel, "test_loop=", test_loop)

print(torch_x1.device)

paddle_out = paddle.incubate.nn.functional.fused_dropout_add(
    paddle_x1, paddle_x2, p=0.0, training=True, mode="upscale_in_train"
)

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for _i in range(test_loop):
        paddle.incubate.nn.functional.fused_dropout_add(
            paddle_x1, paddle_x2, p=0.0, training=True, mode="upscale_in_train"
        )
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

torch_out = fused_dropout_add(torch_x1, torch_x2, p=0.0, training=True, mode="upscale_in_train")

with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for _i in range(test_loop):
        fused_dropout_add(torch_x1, torch_x2, p=0.0, training=True, mode="upscale_in_train")
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)

torch.cuda.synchronize()
start = time.time()
for _i in range(test_loop):
    torch.autograd.grad(
        [torch_out], [torch_x1, torch_x2], grad_outputs=torch_grad, retain_graph=True
    )
torch.cuda.synchronize()
end = time.time()
timeused = end - start
print("torch backward", timeused)
