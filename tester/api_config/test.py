import paddle
import numpy


y = paddle.empty([2]).astype("int32")
y[0] = 1
y[1] = 4

out = paddle.reshape(paddle.ones(shape=[2, 2]),[])
print(out)