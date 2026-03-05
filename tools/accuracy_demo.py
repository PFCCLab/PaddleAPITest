from __future__ import annotations

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


# paddle.maximum(Tensor([0, 1, 2],"float32"), Tensor([1, 8, 2],"float32"), )
numpy_tensor = (numpy.random.random([0, 1, 2]) - 0.5).astype("float32")
paddle_x, torch_x = init_input(numpy_tensor)
numpy_tensor = (numpy.random.random([1, 8, 2]) - 0.5).astype("float32")
paddle_y, torch_y = init_input(numpy_tensor)
paddle_out = paddle.maximum(paddle_x, paddle_y)
torch_out = torch.maximum(torch_x, torch_y)
numpy_tensor = (numpy.random.random([0, 8, 2]) - 0.5).astype("float32")
paddle_grad, torch_grad = init_input(numpy_tensor)

torch_x_grad, torch_y_grad = torch.autograd.grad(
    [torch_out], [torch_x, torch_y], grad_outputs=torch_grad
)
paddle_x_grad, paddle_y_grad = paddle.grad(
    [paddle_out], [paddle_x, paddle_y], grad_outputs=paddle_grad, allow_unused=True
)
print(paddle_x_grad.shape)
print(paddle_y_grad.shape)
numpy.testing.assert_allclose(
    paddle_out.numpy(),
    torch_out.cpu().detach().numpy(),
    1e-2,
    1e-2,
    err_msg="output diff",
)
numpy.testing.assert_allclose(
    paddle_x_grad.numpy(),
    torch_x_grad.cpu().detach().numpy(),
    1e-2,
    1e-2,
    err_msg="output diff",
)
numpy.testing.assert_allclose(
    paddle_y_grad.numpy(),
    torch_y_grad.cpu().detach().numpy(),
    1e-2,
    1e-2,
    err_msg="output diff",
)
