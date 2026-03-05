from __future__ import annotations

import numpy
import paddle
import torch

device = torch.device("cuda:0")
torch.set_default_device(device)


def init_input(numpy_tensor):
    paddle_x = paddle.to_tensor(numpy_tensor)
    torch_x = torch.tensor(numpy_tensor)

    numpy.testing.assert_allclose(
        paddle_x.numpy(), torch_x.cpu().numpy(), 1e-10, 1e-10, err_msg="intput diff"
    )
    return paddle_x, torch_x


# # paddle.cumprod(x=Tensor([12],"int32"), dim=0, )
# numpy_tensor = numpy.random.randint(1, 2, size=[1000], dtype="int32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.cumprod(paddle_x, dim=0)
# torch_out = torch.cumprod(torch_x, dim=0)

# # paddle.argsort(Tensor([30000],"float32"), descending=True, )
# # numpy_tensor = (numpy.random.random([30000]) - 0.5).astype("float32")
# numpy_tensor = numpy.random.randint(-65535, 65535, size=[30000], dtype="int32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.argsort(paddle_x, descending=True)
# torch_out = torch.argsort(torch_x, descending=True)

# # paddle.pow(Tensor([216],"int32"), Tensor([216],"int32"), )
# numpy_tensor = numpy.random.randint(1, 65535, size=[216], dtype="int32")
# paddle_x, torch_x = init_input(numpy_tensor)
# numpy_tensor = numpy.random.randint(1, 10, size=[216], dtype="int32")
# paddle_x2, torch_x2 = init_input(numpy_tensor)
# paddle_out = paddle.pow(paddle_x, paddle_x2)
# torch_out = torch.pow(torch_x, torch_x2)

# # paddle.lcm(Tensor([10, 20],"int32"), Tensor([10, 20],"int32"), )
# numpy_tensor = numpy.random.randint(-100, 100, size=[10, 20], dtype="int32")
# paddle_x, torch_x = init_input(numpy_tensor)
# numpy_tensor = numpy.random.randint(-100, 100, size=[10, 20], dtype="int32")
# paddle_x2, torch_x2 = init_input(numpy_tensor)
# paddle_out = paddle.lcm(paddle_x, paddle_x2)
# torch_out = torch.lcm(torch_x, torch_x2)


# # paddle.linalg.norm(Tensor([16, 16],"float32"), 2.0, )
# numpy_tensor = (numpy.random.random([16, 16]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.linalg.norm(paddle_x, 2.0)
# torch_out = torch.linalg.norm(torch_x, 2.0)

# # paddle.linalg.pinv(Tensor([3, 5, 5],"float32"), rcond=1e-15, hermitian=True, )
# numpy_tensor = (numpy.random.random([3, 5, 5]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.linalg.pinv(paddle_x, rcond=1e-15, hermitian=True)
# torch_out = torch.linalg.pinv(torch_x, rcond=1e-15, hermitian=True)

# paddle.linalg.svd_lowrank(Tensor([3, 100, 40],"float64"), q=6, )
# numpy_tensor = (numpy.random.random([3, 100, 40]) - 0.5).astype("float64")
# paddle_x, torch_x = init_input(numpy_tensor)
# U, S, V = paddle.linalg.svd_lowrank(paddle_x, q=6)
# t_U, t_S, t_V = torch.svd_lowrank(torch_x, q=6)

# numpy.testing.assert_allclose(
#     U.numpy(),
#     t_U.cpu().numpy(),
#     1e-2,
#     1e-2,
#     err_msg='output diff'
# )
# numpy.testing.assert_allclose(
#     S.numpy(),
#     t_S.cpu().numpy(),
#     1e-2,
#     1e-2,
#     err_msg='output diff'
# )
# numpy.testing.assert_allclose(
#     V.numpy(),
#     t_V.cpu().numpy(),
#     1e-2,
#     1e-2,
#     err_msg='output diff'
# )

# # paddle.nextafter(Tensor([],"float32"), Tensor([2, 3, 4],"float32"), )
# numpy_tensor = (numpy.random.random([]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# numpy_tensor = (numpy.random.random([2, 3, 4]) - 0.5).astype("float32")
# paddle_x2, torch_x2 = init_input(numpy_tensor)
# paddle_out = paddle.nextafter(paddle_x, paddle_x2)
# torch_out = torch.nextafter(torch_x, torch_x2)

# # paddle.nn.functional.cosine_similarity(Tensor([23, 12, 1],"float32"), Tensor([23, 1, 10],"float32"), axis=2, eps=1e-06, )
# numpy_tensor = (numpy.random.random([23, 12, 1]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# numpy_tensor = (numpy.random.random([23, 1, 10]) - 0.5).astype("float32")
# paddle_x2, torch_x2 = init_input(numpy_tensor)
# paddle_out = paddle.nn.functional.cosine_similarity(paddle_x, paddle_x2, axis=2, eps=1e-06)
# torch_out = torch.nn.functional.cosine_similarity(torch_x, torch_x2, dim=2, eps=1e-06)

# # paddle.nn.functional.grid_sample(Tensor([1, 128, 128, 128],"float32"), Tensor([1, 128, 128, 2],"float32"), )
# numpy_tensor = (numpy.random.random([1, 128, 128, 128]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# numpy_tensor = (numpy.random.random([1, 128, 128, 2]) - 0.5).astype("float32")
# paddle_x2, torch_x2 = init_input(numpy_tensor)
# paddle_out = paddle.nn.functional.grid_sample(paddle_x, paddle_x2)
# torch_out = torch.nn.functional.grid_sample(torch_x, torch_x2)

# # paddle.nn.functional.gumbel_softmax(x=Tensor([4],"float32"), )
# numpy_tensor = (numpy.random.random([4]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.nn.functional.gumbel_softmax(paddle_x)
# torch_out = torch.nn.functional.gumbel_softmax(torch_x)

# # paddle.nn.functional.rrelu(Tensor([1, 2, 3, 4],"float64"), lower=0.05, upper=0.25, training=True, )
# numpy_tensor = (numpy.random.random([1, 2, 3, 4]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.nn.functional.rrelu(paddle_x, lower=0.05, upper=0.25, training=True)
# torch_out = torch.nn.functional.rrelu(torch_x, lower=0.05, upper=0.25, training=True)
# # paddle_out = paddle.nn.functional.rrelu(paddle_x, lower=0.05, upper=0.25, training=False) # right
# # torch_out = torch.nn.functional.rrelu(torch_x, lower=0.05, upper=0.25, training=False) # right

# # paddle.std(Tensor([],"float32"), )
# numpy_tensor = (numpy.random.random([]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.std(paddle_x)
# torch_out = torch.std(torch_x)

# # paddle.var(Tensor([],"float32"), )
# numpy_tensor = (numpy.random.random([]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.var(paddle_x)
# torch_out = torch.var(torch_x)

# # paddle.unique_consecutive(x=Tensor([4, 1],"float32"), return_inverse=True, )
# numpy_tensor = (numpy.random.random([4, 1]) - 0.5).astype("float32")
# paddle_x, torch_x = init_input(numpy_tensor)
# paddle_out = paddle.unique_consecutive(paddle_x, return_inverse=True)
# torch_out = torch.unique_consecutive(torch_x, return_inverse=True)

# numpy.testing.assert_allclose(
#     paddle_out[0].numpy(),
#     torch_out[0].cpu().numpy(),
#     1e-2,
#     1e-2,
#     err_msg='output diff'
# )
# numpy.testing.assert_allclose(
#     paddle_out[1].numpy(),
#     torch_out[1].cpu().numpy(),
#     1e-2,
#     1e-2,
#     err_msg='output diff'
# )
# numpy.testing.assert_allclose(
#     paddle_out[2].numpy(),
#     torch_out[2].cpu().numpy(),
#     1e-2,
#     1e-2,
#     err_msg='output diff'
# )

# # paddle.copysign(Tensor([12, 20, 2],"int8"), Tensor([12, 20, 2],"int8"), )
# numpy_tensor = numpy.random.randint(-100, 100, size=[12, 20, 2], dtype="int8")
# paddle_x, torch_x = init_input(numpy_tensor)
# numpy_tensor = numpy.random.randint(-100, 100, size=[12, 20, 2], dtype="int8")
# paddle_x2, torch_x2 = init_input(numpy_tensor)
# paddle_out = paddle.copysign(paddle_x, paddle_x2)
# torch_out = torch.copysign(torch_x, torch_x2)

# paddle.cumsum(Tensor([1, 32000],"float16"), axis=-1, )
numpy_tensor = (numpy.random.random([1, 32000]) - 0.5).astype("float16")
paddle_x, torch_x = init_input(numpy_tensor)
paddle_out = paddle.cumsum(paddle_x, axis=-1)
torch_out = torch.cumsum(torch_x, dim=-1)


numpy.testing.assert_allclose(
    paddle_out.numpy(), torch_out.cpu().numpy(), 1e-2, 1e-2, err_msg="output diff"
)
