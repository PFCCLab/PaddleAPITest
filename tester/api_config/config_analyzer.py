from __future__ import annotations

import collections
import copy
import math
import os
import re

import numpy
import paddle
import torch

USE_CACHED_NUMPY = os.getenv("USE_CACHED_NUMPY", "False").lower() == "true"
cached_numpy = {}

not_zero_apis = frozenset(
    [
        "paddle.Tensor.__div__",
        "paddle.Tensor.__floordiv__",
        "paddle.Tensor.__mod__",
        "paddle.Tensor.__rdiv__",
        "paddle.Tensor.__rfloordiv__",
        "paddle.Tensor.__rmod__",
        "paddle.Tensor.__rtruediv__",
        "paddle.Tensor.__truediv__",
        "paddle.Tensor.divide",
        "paddle.Tensor.floor_divide",
        "paddle.Tensor.floor_mod",
        "paddle.Tensor.mod",
        "paddle.divide",
        "paddle.floor_divide",
        "paddle.floor_mod",
        "paddle.mod",
        "paddle.nn.functional.kl_div",
        "paddle.sparse.divide",
    ]
)


def generate_unique_array(num_items, float_dtype):
    def get_integer_dtype(float_dtype):
        float_dtype = numpy.dtype(float_dtype)
        if float_dtype == numpy.float16:
            return numpy.uint16, 16
        elif float_dtype == numpy.float32:
            return numpy.uint32, 32
        elif float_dtype == numpy.float64:
            return numpy.uint64, 64
        else:
            raise ValueError(f"Unsupported float dtype: {float_dtype}")

    integer_dtype, bits = get_integer_dtype(float_dtype)
    max_int = (1 << bits) - 1
    current_start_value = 1
    return_list = []
    attemp_count = 0
    while len(return_list) < num_items and attemp_count < 3:
        nums_to_generate = int(num_items * 1.5)
        if current_start_value >= max_int:
            raise ValueError(
                f"Cannot generate {num_items} unique items of type {float_dtype} within the range."
            )
        end_value = min(current_start_value + nums_to_generate, max_int)
        random_arr = numpy.arange(current_start_value, end_value, dtype=integer_dtype)
        float_arr = random_arr.view(float_dtype)
        if return_list is None:
            return_list = float_arr[numpy.isfinite(float_arr)]
        else:
            return_list = numpy.unique(
                numpy.concatenate([return_list, float_arr[numpy.isfinite(float_arr)]])
            )
        current_start_value = end_value
        attemp_count += 1
    if len(return_list) < num_items:
        raise ValueError(f"Could not generate {num_items} unique items of type {float_dtype}")
    return return_list[:num_items]


class TensorConfig:
    def __init__(self, shape, dtype, place=None):
        self.shape = shape
        self.dtype = dtype
        self.place = place
        self.numpy_tensor = None
        self.paddle_tensor = None
        self.torch_tensor = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.shape = copy.deepcopy(self.shape)
        result.dtype = copy.deepcopy(self.dtype)
        result.place = copy.deepcopy(self.place)
        return result

    def __str__(self):
        return f'Tensor({self.shape},"{self.dtype}")'

    def __repr__(self):
        return f'Tensor({self.shape},"{self.dtype}")'

    def convert_dtype_to_torch_type(self, dtype):
        if dtype in ["float32", numpy.float32]:
            return torch.float32
        elif dtype in ["float16", numpy.float16]:
            return torch.float16
        elif dtype in ["float64", numpy.float64]:
            return torch.float64
        elif dtype in ["int16", numpy.int16]:
            return torch.int16
        elif dtype in ["int8", numpy.int8]:
            return torch.int8
        elif dtype in ["bool", numpy.bool_]:
            return torch.bool
        elif dtype in ["bfloat16", numpy.uint16]:
            return torch.bfloat16
        elif dtype in ["uint8", numpy.uint8]:
            return torch.uint8
        elif dtype in ["int32", numpy.int32]:
            return torch.int32
        elif dtype in ["int64", numpy.int64]:
            return torch.int64
        elif dtype in ["complex64", numpy.complex64]:
            return torch.complex64
        elif dtype in ["complex128", numpy.complex128]:
            return torch.complex128
        else:
            raise ValueError(f"Unsupport dtype: {dtype}")

    def numel(self):
        numel = 1
        for i in self.shape:
            numel = numel * i
        return numel

    def get_cached_numpy(self, dtype, shape):
        numel = 1
        for i in shape:
            numel = numel * i
        if numel > 4300000000:
            raise RuntimeError(f"Too large tensor to get cached numpy: {numel}")

        if dtype in cached_numpy:
            tensor = cached_numpy[dtype][:numel].reshape(shape)
        else:
            if "int" in dtype:
                cached_numpy[dtype] = numpy.random.randint(
                    -65535, 65535, size=4300000000, dtype="int64"
                ).astype(dtype)
                tensor = cached_numpy[dtype][:numel].reshape(shape)
            else:
                cached_numpy[dtype] = (numpy.random.random([4300000000]) - 0.5).astype(dtype)
                tensor = cached_numpy[dtype][:numel].reshape(shape)
        return tensor

    def generate_random_axes(self, api_config):
        x_shape = self.get_arg(api_config, 0, "x").shape
        max_dim = max(len(x_shape), 1)  # scalar

        if len(self.shape) == 0:
            dim = numpy.random.randint(0, max_dim)
            if numpy.random.rand() > 0.5:
                dim -= max_dim
            return numpy.array(dim, dtype=self.dtype)

        if len(self.shape) == 1:
            dims = numpy.random.choice(max_dim, size=self.shape[0], replace=False)
            mask = numpy.random.rand(self.shape[0]) > 0.5
            dims = numpy.where(mask, dims - max_dim, dims)
            return numpy.array(dims, dtype=self.dtype)

        raise ValueError(
            f"Invalid shape for 'axis' Tensor in {api_config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
        )

    def generate_random_index(self, api_config, allow_none=False):
        axis = self.get_arg(api_config, 2, "axis")
        if axis is None and not allow_none:
            raise ValueError("Axis is None")

        x_shape = self.get_arg(api_config, 0, "x").shape
        axis = axis if axis >= 0 else axis + len(x_shape)
        if not (0 <= axis < len(x_shape)):
            raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
        if len(self.shape) >= 1:
            return numpy.random.randint(0, x_shape[axis], size=self.shape, dtype=self.dtype)

        raise ValueError(
            f"Invalid shape for 'index' Tensor in {api_config.api_name}. "
            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
        )

    def get_random_axis_on_tensor(self, api_config, arg_pos, kwargs_name):
        cfg = self.get_arg(api_config, arg_pos, kwargs_name)
        if isinstance(cfg, TensorConfig):
            max_idx = len(cfg.shape)
            return self.get_random_numpy_tensor([], data_type=self.dtype, min=0, max=max_idx)
        else:
            raise ValueError(f"Invalid axis config={cfg} in {api_config.api_name}")

    def get_numpy_tensor(self, api_config, index=None, key=None, **kwargs):
        if index is not None:
            self.index = index
        if key is not None:
            self.key = key

        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        original_dtype = self.dtype
        self.dtype = "float32" if self.dtype == "bfloat16" else self.dtype

        if self.numpy_tensor is None:
            if api_config.api_name in not_zero_apis:
                if "int" in self.dtype:
                    if self.dtype == "int8":
                        arr = numpy.random.randint(1, 256, size=self.shape, dtype=numpy.int32)
                        # 128-255 -> -128~-1
                        arr[arr > 127] -= 256
                        self.numpy_tensor = arr.astype(self.dtype)
                    elif self.dtype == "uint8":
                        self.numpy_tensor = numpy.random.randint(1, 256, size=self.shape).astype(
                            self.dtype
                        )
                    else:
                        self.numpy_tensor = (
                            numpy.random.randint(1, 65535, size=self.shape)
                        ).astype(self.dtype)
                else:
                    self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(self.dtype)
            elif api_config.api_name == "paddle.arange":
                start_val = self.get_arg(api_config, 0, "start", 0)
                end_val = self.get_arg(api_config, 1, "end", None)
                step_val = self.get_arg(api_config, 2, "step", 1)

                def generate_step_tensor(step_config, is_positive):
                    if "int" in step_config.dtype:
                        if is_positive:
                            return numpy.random.randint(1, 10, step_config.shape).astype(
                                step_config.dtype
                            )
                        else:
                            return numpy.random.randint(-10, -1, step_config.shape).astype(
                                step_config.dtype
                            )
                    else:
                        if is_positive:
                            return numpy.random.uniform(0.1, 5.0, step_config.shape).astype(
                                step_config.dtype
                            )
                        else:
                            return numpy.random.uniform(-5.0, -0.1, step_config.shape).astype(
                                step_config.dtype
                            )

                def safe_range(low, high):
                    max_range = 100
                    if high - low > max_range:
                        if low < 0:
                            high = low + max_range
                        else:
                            low = high - max_range
                    if low >= high:
                        low = high - 10
                    return max(low, -1000), min(high, 1000)

                if isinstance(start_val, TensorConfig):
                    if isinstance(end_val, TensorConfig):
                        if isinstance(step_val, TensorConfig):
                            flag = numpy.random.choice([True, False])
                            step_val.numpy_tensor = generate_step_tensor(step_val, flag)
                        else:
                            flag = step_val > 0
                        if "int" in start_val.dtype:
                            start_val.numpy_tensor = numpy.random.randint(
                                -50, 50, start_val.shape
                            ).astype(start_val.dtype)
                        else:
                            start_val.numpy_tensor = numpy.random.uniform(
                                -50.0, 50.0, start_val.shape
                            ).astype(start_val.dtype)
                        start = start_val.numpy_tensor.item()
                        if flag:
                            low, high = safe_range(start + 1, start + 50)
                            if "int" in end_val.dtype:
                                end_val.numpy_tensor = numpy.random.randint(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                            else:
                                end_val.numpy_tensor = numpy.random.uniform(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                        else:
                            low, high = safe_range(start - 50, start - 1)
                            if "int" in end_val.dtype:
                                end_val.numpy_tensor = numpy.random.randint(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                            else:
                                end_val.numpy_tensor = numpy.random.uniform(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                    elif end_val is None:
                        if isinstance(step_val, TensorConfig):
                            flag = numpy.random.choice([True, False])
                            step_val.numpy_tensor = generate_step_tensor(step_val, flag)
                        else:
                            flag = step_val > 0
                        if flag:
                            if "int" in start_val.dtype:
                                start_val.numpy_tensor = numpy.random.randint(
                                    1, 50, start_val.shape
                                ).astype(start_val.dtype)
                            else:
                                start_val.numpy_tensor = numpy.random.uniform(
                                    0.1, 50.0, start_val.shape
                                ).astype(start_val.dtype)
                        else:
                            if "int" in start_val.dtype:
                                start_val.numpy_tensor = numpy.random.randint(
                                    -50, -1, start_val.shape
                                ).astype(start_val.dtype)
                            else:
                                start_val.numpy_tensor = numpy.random.uniform(
                                    -50.0, -0.1, start_val.shape
                                ).astype(start_val.dtype)
                    else:
                        if isinstance(step_val, TensorConfig):
                            flag = numpy.random.choice([True, False])
                            step_val.numpy_tensor = generate_step_tensor(step_val, flag)
                        else:
                            flag = step_val > 0
                        if flag:
                            low, high = safe_range(end_val - 50, end_val - 1)
                            if "int" in start_val.dtype:
                                start_val.numpy_tensor = numpy.random.randint(
                                    low, high, start_val.shape
                                ).astype(start_val.dtype)
                            else:
                                start_val.numpy_tensor = numpy.random.uniform(
                                    low, high, start_val.shape
                                ).astype(start_val.dtype)
                        else:
                            low, high = safe_range(end_val + 1, end_val + 50)
                            if "int" in start_val.dtype:
                                start_val.numpy_tensor = numpy.random.randint(
                                    low, high, start_val.shape
                                ).astype(start_val.dtype)
                            else:
                                start_val.numpy_tensor = numpy.random.uniform(
                                    low, high, start_val.shape
                                ).astype(start_val.dtype)
                else:
                    if isinstance(end_val, TensorConfig):
                        if isinstance(step_val, TensorConfig):
                            flag = numpy.random.choice([True, False])
                            step_val.numpy_tensor = generate_step_tensor(step_val, flag)
                        else:
                            flag = step_val > 0
                        if flag:
                            low, high = safe_range(start_val + 1, start_val + 50)
                            if "int" in end_val.dtype:
                                end_val.numpy_tensor = numpy.random.randint(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                            else:
                                end_val.numpy_tensor = numpy.random.uniform(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                        else:
                            low, high = safe_range(start_val - 50, start_val - 1)
                            if "int" in end_val.dtype:
                                end_val.numpy_tensor = numpy.random.randint(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                            else:
                                end_val.numpy_tensor = numpy.random.uniform(
                                    low, high, end_val.shape
                                ).astype(end_val.dtype)
                    elif end_val is None:
                        if isinstance(step_val, TensorConfig):
                            flag = start_val > 0
                            step_val.numpy_tensor = generate_step_tensor(step_val, flag)
                        else:
                            pass
                    else:
                        if isinstance(step_val, TensorConfig):
                            flag = start_val < end_val
                            step_val.numpy_tensor = generate_step_tensor(step_val, flag)
                        else:
                            pass

                dtype_val = self.get_arg(api_config, 3, "dtype")
                if (
                    dtype_val
                    and "int" in str(dtype_val)
                    and isinstance(step_val, TensorConfig)
                    and "int" not in step_val.dtype
                ):
                    if step_val.numpy_tensor.item() > 0:
                        step_val.numpy_tensor = numpy.random.uniform(
                            1.0, 5.0, step_config.shape
                        ).astype(step_config.dtype)
                    else:
                        step_val.numpy_tensor = numpy.random.uniform(
                            -5.0, -1.0, step_config.shape
                        ).astype(step_config.dtype)

            elif api_config.api_name in {
                "paddle.argmax",
                "paddle.argmin",
                "paddle.Tensor.argmax",
                "paddle.Tensor.argmin",
            }:
                if self.check_arg(api_config, 1, "axis"):
                    arr = self.get_arg(api_config, 0, "x")
                    min_dim = len(arr.shape)
                    self.numpy_tensor = numpy.random.randint(
                        -min_dim, min_dim - 1, size=self.shape
                    ).astype("int64")
                    self.dtype = "int64"

            elif api_config.api_name == "paddle.atan2":
                if self.check_arg(api_config, 0, "x"):
                    s1 = self.get_arg(api_config, 0, "x")
                    s1 = s1.shape
                    self.numpy_tensor = (numpy.random.random(s1) + 1).astype(self.dtype)
                elif self.check_arg(api_config, 1, "y"):
                    s2 = self.get_arg(api_config, 1, "y")
                    s2 = s2.shape
                    self.numpy_tensor = (numpy.random.random(s2) + 1).astype(self.dtype)

            elif api_config.api_name == "paddle.bernoulli":
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)
            elif api_config.api_name == "paddle.bincount":
                if self.check_arg(api_config, 0, "x"):
                    if "int" in self.dtype:
                        self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(
                            self.dtype
                        )
                    else:
                        raise ValueError(
                            f"The input of paddle.bincount must be of integer type, but the current type is {self.dtype}"
                        )
                elif self.check_arg(api_config, 2, "minlength") or self.check_arg(
                    api_config, None, "minlength"
                ):
                    if "int" in self.dtype:
                        self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(
                            self.dtype
                        )
                    else:
                        dtype = "int64"
                        self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(
                            dtype
                        )
                        self.dtype = dtype
            elif api_config.api_name == "paddle.incubate.nn.functional.block_multihead_attention":
                qkv_shape = self.get_arg(
                    api_config, 0, "qkv"
                ).shape  # [token_num, 3 * num_head * head_size].
                bs = self.get_arg(api_config, 3, "seq_lens_encoder").shape[0]
                seq_len = qkv_shape[0] // bs

                if (
                    self.check_arg(api_config, 1, "key_cache")
                    or self.check_arg(api_config, 2, "value_cache")
                    or self.check_arg(api_config, 4, "seq_lens_decoder")
                    or self.check_arg(api_config, 10, "block_tables")
                ):
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                elif self.check_arg(api_config, 3, "seq_lens_encoder"):
                    self.numpy_tensor = numpy.array([seq_len] * bs, dtype=self.dtype)
                elif self.check_arg(api_config, 5, "seq_lens_this_time"):
                    self.numpy_tensor = self.get_initialized_value(
                        api_config, 3, "seq_lens_encoder"
                    )
                elif self.check_arg(api_config, 6, "padding_offsets"):
                    padding_offsets_dtype = self.get_arg(api_config, 6, "padding_offsets").dtype
                    cum_offsets_dtype = self.get_arg(api_config, 7, "cum_offsets").dtype
                    cu_seqlens_q_dtype = self.get_arg(api_config, 8, "cu_seqlens_q").dtype
                    cu_seqlens_k_dtype = self.get_arg(api_config, 9, "cu_seqlens_k").dtype
                    seq_lens_this_time = self.get_initialized_value(
                        api_config, 5, "seq_lens_this_time"
                    )

                    def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
                        cum_offsets_now = numpy.cumsum(max_seq_len - seq_lens_this_time)
                        cum_offsets = numpy.zeros(shape=(bsz + 1), dtype=cum_offsets_dtype)
                        cum_offsets[1:] = cum_offsets_now
                        token_num = numpy.sum(seq_lens_this_time)
                        padding_offsets = numpy.zeros(
                            shape=(token_num), dtype=padding_offsets_dtype
                        )
                        cu_seqlens_q = numpy.zeros(shape=(bsz + 1), dtype=cu_seqlens_q_dtype)
                        cu_seqlens_k = numpy.zeros(shape=(bsz + 1), dtype=cu_seqlens_k_dtype)
                        for i in range(bsz):
                            seq_len_now = seq_lens_this_time[i]
                            cum_offset = cum_offsets[i]
                            for j in range(seq_len_now):
                                padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
                            cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
                            cu_seqlens_q[i + 1] = cum_seq_len
                            cu_seqlens_k[i + 1] = cum_seq_len
                        return (
                            padding_offsets,
                            cum_offsets[:-1],
                            cu_seqlens_q,
                            cu_seqlens_k,
                        )

                    padding_offset, cum_offset, cu_seqlens_q, cu_seqlens_k = get_padding_offset(
                        bs, seq_len, seq_lens_this_time
                    )
                    self.numpy_tensor = padding_offset
                    self.set_tensor_arg_value(api_config, 7, "cum_offsets", cum_offset)
                    self.set_tensor_arg_value(api_config, 8, "cu_seqlens_q", cu_seqlens_q)
                    self.set_tensor_arg_value(api_config, 9, "cu_seqlens_k", cu_seqlens_k)
                elif (
                    self.check_arg(api_config, 13, "cache_k_quant_scales")
                    or self.check_arg(api_config, 14, "cache_v_quant_scales")
                    or self.check_arg(api_config, 15, "cache_k_dequant_scales")
                    or self.check_arg(api_config, 16, "cache_v_dequant_scales")
                    or self.check_arg(api_config, 17, "qkv_out_scale")
                    or self.check_arg(api_config, 20, "out_smooth")
                ):
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype, min=0)
                elif self.check_arg(api_config, 22, "max_dec_len_this_time") or self.check_arg(
                    api_config, 21, "max_enc_len_this_time"
                ):
                    self.place = "cpu"
                    if self.check_arg(api_config, 22, "max_dec_len_this_time"):
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    else:  # 21, "max_enc_len_this_time"
                        self.numpy_tensor = numpy.array([seq_len] * bs, dtype=self.dtype)
                elif self.check_arg(api_config, 23, "rope_emb") and self.place == "cpu":
                    self.place = "gpu"
                elif self.check_arg(api_config, 24, "mask") or self.check_arg(
                    api_config, 25, "tgt_mask"
                ):
                    eps = numpy.finfo(self.dtype).eps
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        self.shape, self.dtype, max=0 + eps
                    )

            # c
            elif api_config.api_name == "paddle.chunk":
                import random

                if self.check_arg(api_config, 2, "axis"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    chunks = self.get_arg(api_config, 1, "chunks")
                    valid_axes = []
                    for i, dim_size in enumerate(x_tensor.shape):
                        if dim_size % chunks == 0:
                            valid_axes.append(i)
                    if not valid_axes:
                        raise ValueError(
                            f"No valid axis found in x.shape = {x_tensor.shape} for chunks = {chunks}. "
                            f"Each dim must be divisible by chunks."
                        )
                    chosen_axis = random.choice(valid_axes)
                    if len(self.shape) == 0:
                        self.numpy_tensor = numpy.array(chosen_axis, dtype=self.dtype)
                    elif len(self.shape) == 1:
                        if self.shape[0] == 1:
                            self.numpy_tensor = numpy.array([chosen_axis], dtype=self.dtype)
                        else:
                            raise ValueError(
                                f"Invalid shape for 'axis' Tensor in paddle.chunk. "
                                f"Expected a 0-D or 1-D Tensor with 1 element, but got shape {self.shape}."
                            )
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.chunk. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )

            elif api_config.api_name == "paddle.nn.functional.conv2d_transpose":
                if (index is not None and index == 0) or key == "x":
                    if not hasattr(api_config, "x"):
                        if "int" in self.dtype:
                            self.numpy_tensor = (
                                numpy.random.randint(-65535, 65535, size=self.shape)
                            ).astype(self.dtype)
                        else:
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(
                                self.dtype
                            )
                        api_config.x = self.numpy_tensor
                elif (index is not None and index == 1) or key == "weight":
                    if not hasattr(api_config, "weight"):
                        if "int" in self.dtype:
                            self.numpy_tensor = (
                                numpy.random.randint(-65535, 65535, size=self.shape)
                            ).astype(self.dtype)
                        else:
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(
                                self.dtype
                            )
                        api_config.weight = self.numpy_tensor
                elif (index is not None and index == 2) or key == "bias":
                    if not hasattr(api_config, "bias"):
                        if "int" in self.dtype:
                            self.numpy_tensor = (
                                numpy.random.randint(-65535, 65535, size=self.shape)
                            ).astype(self.dtype)
                        else:
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(
                                self.dtype
                            )
                        api_config.bias = self.numpy_tensor
                elif key == "output_size":
                    if not hasattr(api_config, "bias"):
                        bias = None
                    else:
                        bias = paddle.to_tensor(api_config.bias)
                    stride = api_config.kwargs.get("stride", 1)
                    padding = api_config.kwargs.get("padding", 0)
                    if "dilation" in api_config.kwargs:
                        dilation = api_config.kwargs["dilation"]
                    else:
                        dilation = 1
                    groups = api_config.kwargs.get("groups", 1)
                    if "output_padding" in api_config.kwargs:
                        output_padding = api_config.kwargs["output_padding"]
                    else:
                        output_padding = 0
                    if "data_format" in api_config.kwargs:
                        data_format = api_config.kwargs["data_format"]
                    else:
                        data_format = "NCHW"

                    output_size = paddle.nn.functional.conv2d_transpose(
                        paddle.to_tensor(api_config.x),
                        paddle.to_tensor(api_config.weight),
                        bias=bias,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        groups=groups,
                        dilation=dilation,
                        data_format=data_format,
                    )

                    last = [0, 0]
                    last[0] = output_size.shape[data_format.find("H")]
                    last[1] = output_size.shape[data_format.find("W")]
                    s = [1, 1]
                    if isinstance(stride, int):
                        s[0] = stride
                        s[1] = stride
                    else:
                        s = stride
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    self.numpy_tensor[0] = numpy.random.randint(last[0], last[0] + s[0])
                    self.numpy_tensor[1] = numpy.random.randint(last[1], last[1] + s[1])

            elif api_config.api_name in {"paddle.cumsum", "paddle.Tensor.cumsum"}:
                if self.check_arg(api_config, 1, "axis"):
                    # special args[1] tensor init, for the rest reuse default initialization logic
                    x_tensor_config = self.get_arg(api_config, 0, "x")
                    len_shape = len(x_tensor_config.shape)
                    self.numpy_tensor = numpy.random.randint(-len_shape, len_shape, size=self.shape)

            elif api_config.api_name in {
                "paddle.clip",
                "paddle.Tensor.clip",
            } and self.check_arg(api_config, 0, "x"):
                # if both min and max need a Tensor instead of None, init min and max at the same TensorConfig numpy tensor init process
                min_config = self.get_arg(api_config, 1, "min")
                max_config = self.get_arg(api_config, 2, "max")
                if isinstance(min_config, TensorConfig) and isinstance(max_config, TensorConfig):
                    min_shape = min_config.shape
                    min_dtype = min_config.dtype
                    min_numpy_tensor = self.get_random_numpy_tensor(
                        shape=min_shape, data_type=min_dtype
                    )

                    max_shape = max_config.shape
                    max_dtype = max_config.dtype
                    max_numpy_tensor = self.get_random_numpy_tensor(
                        shape=max_shape, data_type=max_dtype, min=min_numpy_tensor
                    )

                    self.set_tensor_arg_value(api_config, 1, "min", min_numpy_tensor)
                    self.set_tensor_arg_value(api_config, 2, "max", max_numpy_tensor)
                elif min_config is not None and max_config is not None:
                    # min and max args are specified but at least one of them is scalar (not a TensorConfig)
                    # according to API DOC, min and max is float|int|Tensor
                    if isinstance(min_config, TensorConfig) and (
                        isinstance(max_config, (int, float))
                    ):
                        min_shape = min_config.shape
                        min_dtype = min_config.dtype
                        min_numpy_tensor = self.get_random_numpy_tensor(
                            shape=min_shape, data_type=min_dtype, max=max_config
                        )
                        self.set_tensor_arg_value(api_config, 1, "min", min_numpy_tensor)
                    elif isinstance(max_config, TensorConfig) and (
                        isinstance(min_config, (int, float))
                    ):
                        max_shape = max_config.shape
                        max_dtype = max_config.dtype
                        max_numpy_tensor = self.get_random_numpy_tensor(
                            shape=max_shape, data_type=max_dtype, min=min_config
                        )
                        self.set_tensor_arg_value(api_config, 2, "max", max_numpy_tensor)
                    # for both min and max are scalar, there is no need to init numpy tensor
                # init input tensor x randomly
                self.numpy_tensor = self.get_random_numpy_tensor(
                    shape=self.shape, data_type=self.dtype
                )
            elif api_config.api_name == "paddle.vision.ops.distribute_fpn_proposals":
                if (index is not None and index == 0) or (key is not None and key == "fpn_rois"):
                    num = self.shape[0]
                    self.numpy_tensor = numpy.random.randint(1, 1024, [num, 4])
                    self.numpy_tensor[:, 0] = self.numpy_tensor[:, 0] + numpy.random.random([num])
                    self.numpy_tensor[:, 1] = self.numpy_tensor[:, 1] + numpy.random.random([num])
                    self.numpy_tensor[:, 2] = (
                        self.numpy_tensor[:, 0]
                        + numpy.random.randint(1, 1024, [num])
                        + numpy.random.random([num])
                    )
                    self.numpy_tensor[:, 3] = (
                        self.numpy_tensor[:, 1]
                        + numpy.random.randint(1, 1024, [num])
                        + numpy.random.random([num])
                    )
                    if not hasattr(api_config, "num"):
                        api_config.num = num
                elif (index is not None and index == 6) or (key is not None and key == "rois_num"):
                    num = api_config.num
                    re = self.shape[0]
                    self.numpy_tensor = numpy.zeros(self.shape)
                    if num > 4096 or re > 4096:
                        if num < re:
                            self.numpy_tensor[:num] = 1
                        else:
                            self.numpy_tensor += num // re
                            self.numpy_tensor[: num % re] += 1
                    else:
                        if num < re:
                            indices = numpy.random.choice(re, num, replace=False)
                            self.numpy_tensor[indices] = 1
                        else:
                            for i in range(self.shape[0] - 1):
                                self.numpy_tensor[i] = numpy.random.randint(1, num - re + 2)
                                num = num - self.numpy_tensor[i]
                                re -= 1
                            self.numpy_tensor[self.shape[0] - 1] = num

            elif api_config.api_name == "paddle.dot":
                if "int" in self.dtype:
                    self.numpy_tensor = (numpy.random.randint(-127, 127, size=self.shape)).astype(
                        self.dtype
                    )

            elif api_config.api_name in [
                "paddle.nn.functional.dropout",
                "paddle.nn.functional.dropout2d",
                "paddle.nn.functional.dropout3d",
            ]:
                if self.check_arg(api_config, 1, "p"):
                    eps = 0.1
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        shape=self.shape, data_type=self.dtype, min=0, max=1 + eps
                    )
                    # include 1 in numpy tensor
                    self.numpy_tensor = numpy.where(self.numpy_tensor > 1, 1, self.numpy_tensor)
                elif api_config.api_name == "paddle.nn.functional.dropout" and self.check_arg(
                    api_config, 2, "axis"
                ):
                    self.numpy_tensor = self.get_random_axis_on_tensor(api_config, 0, "x")
            elif api_config.api_name == "paddle.empty":
                is_shape_param = False
                if len(api_config.args) > 0:
                    if self.check_arg(api_config, 0, "shape"):
                        is_shape_param = True
                    elif isinstance(api_config.args[0], list):
                        for item in api_config.args[0]:
                            if str(item) == str(self):
                                is_shape_param = True
                                break
                if "shape" in api_config.kwargs:
                    if str(api_config.kwargs["shape"]) == str(self):
                        is_shape_param = True
                    elif isinstance(api_config.kwargs["shape"], list):
                        for item in api_config.kwargs["shape"]:
                            if str(item) == str(self):
                                is_shape_param = True
                                break
                if is_shape_param:
                    if "int" in self.dtype:
                        self.numpy_tensor = numpy.random.randint(1, 10, size=self.shape).astype(
                            self.dtype
                        )
                    else:
                        dtype = "int32"
                        self.numpy_tensor = numpy.random.randint(1, 10, size=self.shape).astype(
                            dtype
                        )
                        self.dtype = dtype
            elif api_config.api_name == "paddle.eye":
                self.numpy_tensor = numpy.random.randint(0, 2048, size=self.shape)

            elif api_config.api_name in {"paddle.expand", "paddle.Tensor.expand"}:
                if key == "shape" or index == 1:
                    d = self.get_arg(api_config, 0, "x")
                    s = d.shape
                    ind = kwargs["list_index"][0] if "list_index" in kwargs else 0
                    if len(s) == 0 or ind > len(s) - 1 or s[ind] == 1:
                        self.numpy_tensor = (numpy.random.randint(1, 127, size=self.shape)).astype(
                            self.dtype
                        )
                    else:
                        if len(self.shape) == 0 or self.shape[0] == 1:
                            self.numpy_tensor = numpy.array(s[ind])
                        else:
                            self.numpy_tensor = (
                                numpy.random.randint(1, 127, size=self.shape)
                            ).astype(self.dtype)
                            dis = self.shape[0] - len(s)
                            for i in range(self.shape[0]):
                                if i >= dis and s[i - dis] != 1:
                                    self.numpy_tensor[i] = s[i - dis]

            elif api_config.api_name == "paddle.full":
                if self.check_arg(api_config, 1, "fill_value"):
                    if "int" in self.dtype:
                        self.numpy_tensor = (
                            numpy.random.randint(1, 65535, size=self.shape)
                        ).astype(self.dtype)
                    else:
                        self.numpy_tensor = (numpy.random.random(self.shape) + 0.5).astype(
                            self.dtype
                        )
                else:
                    self.numpy_tensor = (numpy.random.randint(0, 64, size=self.shape)).astype(
                        self.dtype
                    )
            elif api_config.api_name in {"paddle.gammainc", "paddle.gammaincc"}:
                if "int" in self.dtype:
                    self.numpy_tensor = numpy.random.randint(0, 65535, size=self.shape).astype(
                        self.dtype
                    )
                else:
                    self.numpy_tensor = numpy.abs(numpy.random.random(self.shape)).astype(
                        self.dtype
                    )
            elif api_config.api_name == "paddle.vision.ops.generate_proposals":
                if (
                    (index is not None and index == 0)
                    or key == "scores"
                    or (index is not None and index == 1)
                    or key == "bbox_deltas"
                ):
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)
                elif (index is not None and index == 2) or key == "img_size":
                    self.numpy_tensor = numpy.random.randint(0, 1024, size=self.shape).astype(
                        self.dtype
                    )
                elif (index is not None and index == 3) or key == "anchors":
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    w = self.shape[0]
                    h = self.shape[1]
                    for i in range(self.shape[0]):
                        self.numpy_tensor[i][0] = numpy.random.random() * w
                        self.numpy_tensor[i][1] = numpy.random.random() * h
                        self.numpy_tensor[i][2] = (
                            numpy.random.random() * (w - self.numpy_tensor[i][0] + 1)
                            + self.numpy_tensor[i][0]
                            + 1
                        )
                        self.numpy_tensor[i][3] = (
                            numpy.random.random() * (h - self.numpy_tensor[i][1] + 1)
                            + self.numpy_tensor[i][1]
                            + 1
                        )

            elif api_config.api_name.startswith("paddle.geometric.segment_"):
                if self.check_arg(api_config, 1, "segment_ids"):
                    batch_size = self.get_arg(api_config, 0, "x").shape[0]
                    max_segments = numpy.random.randint(1, batch_size + 1)
                    self.numpy_tensor = numpy.sort(
                        numpy.random.randint(0, max_segments, size=self.shape).astype(self.dtype)
                    )
            elif api_config.api_name == "paddle.geometric.sample_neighbors":
                if self.check_arg(api_config, 0, "row"):
                    colptr_shape = self.get_arg(api_config, 1, "colptr").shape
                    num_nodes = colptr_shape[0] - 1
                    self.numpy_tensor = numpy.random.randint(
                        0, num_nodes, size=self.shape, dtype=self.dtype
                    )
                elif self.check_arg(api_config, 1, "colptr"):
                    num_edges = self.get_arg(api_config, 0, "row").shape[0]
                    num_nodes = self.shape[0] - 1
                    colptr = numpy.zeros(self.shape, dtype=self.dtype)
                    if num_nodes > 0 and num_edges > 0:
                        splits = numpy.random.choice(
                            numpy.arange(num_edges + 1), num_nodes - 1, replace=True
                        )
                        splits.sort()
                        colptr[1:num_nodes] = splits
                        colptr[num_nodes] = num_edges
                    self.numpy_tensor = colptr
                elif self.check_arg(api_config, 2, "input_nodes"):
                    num_nodes = self.shape[0] - 1
                    self.numpy_tensor = numpy.random.randint(
                        0, num_nodes, size=self.shape, dtype=self.dtype
                    )
                elif self.check_arg(api_config, 4, "eids") or self.check_arg(
                    api_config, 6, "perm_buffer"
                ):
                    num_edges = self.get_arg(api_config, 0, "row").shape[0]
                    self.numpy_tensor = numpy.arange(num_edges, dtype=self.dtype).reshape(
                        self.shape
                    )

            elif api_config.api_name.startswith("paddle.geometric.send_"):
                if api_config.api_name.endswith("u_recv"):
                    if self.check_arg(api_config, 1, "src_index") or self.check_arg(
                        api_config, 2, "dst_index"
                    ):
                        num_nodes = self.get_arg(api_config, 0, "x").shape[0]
                        self.numpy_tensor = numpy.random.randint(
                            0, num_nodes, size=self.shape
                        ).astype(self.dtype)
                elif self.check_arg(api_config, 2, "src_index") or self.check_arg(
                    api_config, 3, "dst_index"
                ):
                    num_nodes = self.get_arg(api_config, 0, "x").shape[0]
                    self.numpy_tensor = numpy.random.randint(0, num_nodes, size=self.shape).astype(
                        self.dtype
                    )
            elif api_config.api_name in {"paddle.index_add", "paddle.index_fill"}:
                if self.check_arg(api_config, 1, "index"):
                    self.numpy_tensor = self.generate_random_index(api_config)
            elif api_config.api_name == "paddle.index_sample":
                if self.check_arg(api_config, 1, "index"):
                    x_dim = self.get_arg(api_config, 0, "x").shape[1]
                    self.numpy_tensor = numpy.random.randint(0, x_dim, size=self.shape)
            elif api_config.api_name.startswith("paddle.incubate.segment_"):
                if self.check_arg(api_config, 1, "segment_ids"):
                    batch_size = self.get_arg(api_config, 0, "x").shape[0]
                    max_segments = numpy.random.randint(1, batch_size + 1)
                    self.numpy_tensor = numpy.sort(
                        numpy.random.randint(0, max_segments, size=self.shape).astype(self.dtype)
                    )
            elif api_config.api_name == "paddle.logspace":
                if self.check_arg(api_config, 2, "num"):
                    self.numpy_tensor = numpy.random.randint(1, 65535, size=self.shape)
            elif api_config.api_name.startswith("paddle.linalg."):
                if api_config.api_name.endswith("cholesky"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 2 or self.shape[-1] != self.shape[-2]:
                            raise ValueError(
                                "Shape must have at least 2 dimensions and last two dimensions must be equal"
                            )
                        batch_dims = self.shape[:-2]
                        matrix_dim = self.shape[-1]
                        A = numpy.random.random([*batch_dims, matrix_dim, matrix_dim]).astype(
                            self.dtype
                        )
                        if len(batch_dims) > 0:
                            tensor = numpy.einsum("...ij,...kj->...ik", A, A)
                        else:
                            tensor = numpy.dot(A, A.T)
                        tensor += numpy.eye(matrix_dim, dtype=self.dtype) * 10000
                        print("cholesky tensor", tensor)
                        self.numpy_tensor = tensor
                elif api_config.api_name.endswith("cov"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 1 or len(self.shape) > 2:
                            raise ValueError(
                                "Shape must have 1 or 2 dimensions for covariance input"
                            )
                        tensor = numpy.random.random(self.shape).astype(self.dtype)
                        tensor += numpy.random.random(self.shape).astype(self.dtype) * 1e-6
                        self.numpy_tensor = tensor
                    elif self.check_arg(api_config, 3, "fweights"):
                        x_shape = self.get_arg(api_config, 0, "x").shape
                        rowvar = self.get_arg(api_config, 1, "rowvar")
                        if rowvar is None:
                            rowvar = True
                        n_observations = (
                            (x_shape[1] if rowvar else x_shape[0])
                            if len(x_shape) > 1
                            else x_shape[0]
                        )
                        self.numpy_tensor = numpy.random.randint(
                            1, 11, size=(n_observations,)
                        ).astype(self.dtype)
                    elif self.check_arg(api_config, 4, "aweights"):
                        x_shape = self.get_arg(api_config, 0, "x").shape
                        rowvar = self.get_arg(api_config, 1, "rowvar")
                        if rowvar is None:
                            rowvar = True
                        n_observations = (
                            (x_shape[1] if rowvar else x_shape[0])
                            if len(x_shape) > 1
                            else x_shape[0]
                        )
                        if self.dtype in ["float32", "float64"]:
                            self.numpy_tensor = numpy.random.uniform(
                                0.1, 1.0, size=(n_observations,)
                            ).astype(self.dtype)
                        else:
                            self.numpy_tensor = numpy.random.randint(
                                1, 11, size=(n_observations,)
                            ).astype(self.dtype)
                elif api_config.api_name.endswith("eigh") or api_config.api_name.endswith(
                    "eigvalsh"
                ):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 2 or self.shape[-1] != self.shape[-2]:
                            raise ValueError(
                                "Shape must have at least 2 dimensions and last two dimensions must be equal"
                            )
                        batch_dims = self.shape[:-2]
                        matrix_dim = self.shape[-1]
                        A = numpy.random.random([*batch_dims, matrix_dim, matrix_dim]).astype(
                            self.dtype
                        )
                        if self.dtype in ["complex64", "complex128"]:
                            A = A + 1j * numpy.random.random(
                                [*batch_dims, matrix_dim, matrix_dim]
                            ).astype(self.dtype)
                            tensor = A + A.swapaxes(-1, -2).conj()  # A + A^H
                        else:
                            if len(batch_dims) > 0:
                                tensor = numpy.einsum("...ij,...kj->...ik", A, A)
                            else:
                                tensor = numpy.dot(A, A.T)
                        tensor += numpy.eye(matrix_dim, dtype=self.dtype) * 1e-6
                        self.numpy_tensor = tensor
                elif api_config.api_name.endswith("lstsq"):
                    if self.check_arg(api_config, 0, "x") or self.check_arg(api_config, 1, "y"):
                        if len(self.shape) < 2:
                            raise ValueError("Shape must have at least 2 dimensions for lstsq x")
                        batch_dims = self.shape[:-2]
                        M, N = self.shape[-2], self.shape[-1]
                        self.numpy_tensor = numpy.random.random([*batch_dims, M, N]).astype(
                            self.dtype
                        )
                elif api_config.api_name.endswith("lu_unpack"):
                    if self.check_arg(api_config, 0, "x"):
                        if len(self.shape) < 2:
                            raise ValueError("Shape must have at least 2 dimensions for LU matrix")
                        batch_dims = self.shape[:-2]
                        LU_tensor = numpy.random.random(self.shape).astype(self.dtype)
                        K = min(self.shape[-2], self.shape[-1])
                        LU_tensor[..., range(K), range(K)] += 1e-6
                        self.numpy_tensor = LU_tensor
                    if self.check_arg(api_config, 1, "pivot"):
                        M = self.get_arg(api_config, 0, "x").shape[-2]
                        self.numpy_tensor = numpy.random.randint(1, M + 1, size=self.shape).astype(
                            self.dtype
                        )
                elif api_config.api_name.endswith("pca_lowrank"):
                    self.numpy_tensor = numpy.random.randn(*self.shape).astype(self.dtype)
                elif api_config.api_name.endswith("cond"):
                    # produce non-singular matrix
                    n = self.shape[-1]
                    # Generate random matrix, value in [0, 1)
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)
                    # Create scaled identity matrix, value is n
                    eye_matrix = n * numpy.eye(n, dtype=self.dtype)
                    # Construct a non-singular matrix: A = random_matrix + n*I
                    # strict diagonal dominant matrix is non-singular. https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
                    self.numpy_tensor += eye_matrix
                elif api_config.api_name.endswith("det"):
                    if self.check_arg(api_config, 0, "x"):
                        assert len(self.shape) >= 2, "Input must be at least 2D."
                        assert self.shape[-1] == self.shape[-2], "Input must be square matrices."
                        n = self.shape[-1]
                        A = numpy.random.uniform(low=0.5, high=1.0, size=self.shape).astype(
                            self.dtype
                        )
                        A_T = numpy.swapaxes(A, -1, -2)
                        self.numpy_tensor = numpy.matmul(A, A_T) + numpy.eye(n, dtype=self.dtype)
                elif api_config.api_name.endswith("pinv"):
                    if self.check_arg(api_config, 0, "x") and self.get_arg(
                        api_config, 2, " hermitian"
                    ):
                        is_complex = self.dtype.startswith("complex")
                        if len(self.shape) not in [2, 3]:
                            raise ValueError("pinv only supports 2D or 3D tensors")
                        if is_complex:
                            if self.dtype == "complex64":
                                real_dtype = numpy.float32
                            elif self.dtype == "complex128":
                                real_dtype = numpy.float64
                            A_real = numpy.random.randn(*self.shape).astype(real_dtype)
                            A_imag = numpy.random.randn(*self.shape).astype(real_dtype)
                            A = A_real + 1j * A_imag
                            A = A.astype(self.dtype)
                        else:
                            A = numpy.random.randn(*self.shape).astype(self.dtype)
                        if len(self.shape) == 2:
                            A_T = A.conj().T if is_complex else A.T
                        else:
                            A_T = (
                                numpy.conj(A).swapaxes(-2, -1) if is_complex else A.swapaxes(-2, -1)
                            )
                        self.numpy_tensor = (A + A_T) / 2
                elif api_config.api_name.endswith("corrcoef"):
                    if self.dtype == "float16":
                        # 1e-3 to avoid inf
                        self.numpy_tensor = (
                            numpy.random.randn(*self.shape).astype(self.dtype) * 1e-3
                        )
            elif api_config.api_name == "paddle.linspace":
                if "int" in self.dtype:
                    self.numpy_tensor = (numpy.random.randint(0, 65535, size=self.shape)).astype(
                        self.dtype
                    )
                else:
                    self.numpy_tensor = (numpy.random.random(self.shape)).astype(self.dtype)
            # m
            elif api_config.api_name == "paddle.incubate.nn.functional.masked_multihead_attention":
                if self.check_arg(api_config, 4, "sequence_lengths"):
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype, min=1)
                elif self.check_arg(api_config, 5, "rotary_tensor"):
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        self.shape, self.dtype, min=0, max=1000
                    )

            elif api_config.api_name == "paddle.matrix_transpose":
                if self.check_arg(api_config, 0, "x"):
                    if len(self.shape) < 2:
                        matrix_shape = [2, 2]
                        if "int" in self.dtype:
                            self.numpy_tensor = numpy.random.randint(
                                -65535, 65535, size=matrix_shape
                            ).astype(self.dtype)
                        else:
                            self.numpy_tensor = (numpy.random.random(matrix_shape) - 0.5).astype(
                                self.dtype
                            )
                    else:
                        if "int" in self.dtype:
                            self.numpy_tensor = (
                                numpy.random.randint(-65535, 65535, size=self.shape)
                            ).astype(self.dtype)
                        else:
                            self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(
                                self.dtype
                            )
            elif api_config.api_name in {"paddle.mean", "paddle.max", "paddle.min"}:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)

            elif api_config.api_name == "paddle.multinomial":
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = numpy.abs(numpy.random.random(self.shape)).astype(
                        self.dtype
                    )

                if key == "num_samples" or index == 1:
                    if (
                        "replacement" in api_config.kwargs
                        and self.get_arg(api_config, 2, "replacement") == True
                    ):
                        max_allow = 1024
                    else:
                        inputs = self.get_arg(api_config, 0, "x")
                        inputs = inputs.numpy_tensor
                        max_allow = (inputs > 0).sum().item()
                    self.numpy_tensor = numpy.random.randint(
                        1, max_allow + 1, size=self.shape
                    ).astype(self.dtype)

            elif api_config.api_name == "paddle.multiplex":
                s = self.get_arg(api_config, 0, "inputs")
                if key == "index" or index == 1:
                    self.numpy_tensor = (numpy.random.randint(0, len(s), size=self.shape)).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.multiply":
                if self.dtype == "bfloat16":
                    self.dtype = "float32"
                self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name in {
                "paddle.nn.functional.max_unpool1d",
                "paddle.nn.functional.max_unpool2d",
                "paddle.nn.functional.max_unpool3d",
            } and self.check_arg(api_config, 0, "x"):
                # use max_pool to generate legal max_unpool input
                kernel_size = self.get_initialized_value(api_config, 2, "kernel_size")
                stride = self.get_initialized_value(api_config, 3, "stride")
                padding = self.get_initialized_value(api_config, 4, "padding")
                padding = 0 if padding is None else padding
                stride = kernel_size if stride is None else stride
                unpool_output_size = self.get_initialized_value(api_config, 5, "output_size")
                pool_input_size = unpool_output_size

                ndim = 1
                if "max_unpool2d" in api_config.api_name:
                    ndim = 2
                elif "max_unpool3d" in api_config.api_name:
                    ndim = 3
                if isinstance(kernel_size, int):
                    kernel_size = [kernel_size] * ndim
                if isinstance(stride, int):
                    stride = [stride] * ndim
                if isinstance(padding, int):
                    padding = [padding] * ndim

                # if max_unpool output_size (max_pool input_size) is not set, calculate manually
                unpool_input_size = self.get_arg(api_config, 0, "x").shape
                pool_output_size = unpool_input_size
                if pool_input_size is None:
                    if ndim == 1:
                        w_in = pool_output_size[-1]
                        w_out = (w_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
                        pool_input_size = [*pool_output_size[:-1], w_out]
                    elif ndim == 2:
                        h_in, w_in = pool_output_size[-2], pool_output_size[-1]
                        h_out = (h_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
                        w_out = (w_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
                        pool_input_size = [*pool_output_size[:-2], h_out, w_out]
                    else:
                        d_in, h_in, w_in = (
                            pool_output_size[-3],
                            pool_output_size[-2],
                            pool_output_size[-1],
                        )
                        d_out = (d_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
                        h_out = (h_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1]
                        w_out = (w_in - 1) * stride[2] - 2 * padding[2] + kernel_size[2]
                        pool_input_size = [*pool_output_size[:-3], d_out, h_out, w_out]
                elif len(pool_input_size) == ndim:
                    # fill the lost dimensions since unpool_output_size has only last ndim dims
                    pool_input_size = [
                        *pool_output_size[:-ndim],
                        *pool_input_size[-ndim:],
                    ]
                elif len(pool_input_size) != len(pool_output_size):
                    raise ValueError(
                        f"invalid argument output_size {pool_input_size} for {api_config.api_name}, len(output_size) should be {ndim} or {len(pool_output_size)} or output_size == None, got len(output_size)={len(pool_input_size)} and output_size={unpool_output_size}"
                    )

                # int64 handle
                data_type = "float64" if self.dtype == "int64" else self.dtype
                x = paddle.to_tensor(
                    self.get_random_numpy_tensor(
                        shape=pool_input_size, data_type=data_type, min=-5, max=5
                    )
                )
                max_poolxd_func = eval(api_config.api_name.replace("max_unpool", "max_pool"))
                x, indices = max_poolxd_func(x, kernel_size, stride, padding, return_mask=True)
                self.numpy_tensor = x.numpy()
                self.set_tensor_arg_value(api_config, 1, "indices", indices)

            elif api_config.api_name == "paddle.vision.ops.nms":
                if index == 0 or key == "boxes":
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    for i in range(self.shape[0]):
                        self.numpy_tensor[i][0] = numpy.random.random() * 1023
                        self.numpy_tensor[i][1] = numpy.random.random() * 1023
                        self.numpy_tensor[i][2] = (
                            numpy.random.random() * (1024 - self.numpy_tensor[i][0] + 1)
                            + self.numpy_tensor[i][0]
                            + 1
                        )
                        self.numpy_tensor[i][3] = (
                            numpy.random.random() * (1024 - self.numpy_tensor[i][1] + 1)
                            + self.numpy_tensor[i][1]
                            + 1
                        )
                elif index == 3 or key == "scores":
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)
                else:
                    self.numpy_tensor = numpy.random.randint(0, 1024, self.shape).astype(self.dtype)

            elif api_config.api_name in {
                "paddle.nn.functional.adaptive_avg_pool2d",
                "paddle.nn.functional.adaptive_avg_pool3d",
            }:
                if key == "output_size" or index == 1:
                    s = self.get_arg(api_config, 0, "x")
                    s = s.shape
                    self.numpy_tensor = numpy.random.randint(
                        1, 2 * numpy.max(s), size=self.shape
                    ).astype(self.dtype)
            elif api_config.api_name == "paddle.nn.functional.adaptive_log_softmax_with_loss":
                if self.check_arg(api_config, 1, "label"):
                    cutoffs = self.get_arg(api_config, 4, "cutoffs")
                    n_classes = cutoffs[-1]
                    generation_size = self.shape
                    if isinstance(self.shape, (list, tuple)) and len(self.shape) == 0:
                        generation_size = 1
                    if n_classes == 1:
                        self.numpy_tensor = numpy.zeros(generation_size, dtype=self.dtype)
                    else:
                        self.numpy_tensor = numpy.random.randint(
                            low=0,
                            high=n_classes,
                            size=generation_size,
                            dtype=self.dtype,
                        )
            elif api_config.api_name == "paddle.nn.functional.affine_grid":
                if key == "out_shape" or index == 1:
                    s = self.get_arg(api_config, 0, "theta")
                    s = s.shape
                    self.numpy_tensor = numpy.random.randint(1, 128, size=self.shape).astype(
                        self.dtype
                    )
                    self.numpy_tensor[0] = s[0]
            elif api_config.api_name == "paddle.nn.functional.alpha_dropout":
                if key == "x" or index == 0:
                    if self.dtype == "bfloat16":
                        self.dtype = "float32"
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name == "paddle.nn.functional.interpolate":
                if key == "size" or index == 1 or key == "scale_factor" or index == 2:
                    self.numpy_tensor = numpy.random.randint(1, 128, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.gather_tree":
                if self.check_arg(api_config, 1, "parents"):
                    sequences = self.get_arg(api_config, 0, "sequences")
                    if hasattr(sequences, "shape") and len(sequences.shape) >= 3:
                        beam_size = sequences.shape[2]
                    else:
                        beam_size = self.shape[2] if len(self.shape) >= 3 else 4
                    beam_size = 1 if beam_size < 1 else beam_size
                    parents = numpy.zeros(self.shape, dtype=self.dtype)
                    for t in range(self.shape[0]):
                        for b in range(self.shape[1]):
                            for i in range(self.shape[2]):
                                parents[t, b, i] = numpy.random.randint(0, beam_size)
                    self.numpy_tensor = parents

            elif api_config.api_name == "paddle.nn.functional.gaussian_nll_loss":
                if self.check_arg(api_config, 2, "var"):
                    self.numpy_tensor = (numpy.random.random(self.shape) + 1.0).astype(self.dtype)

            elif api_config.api_name == "paddle.nn.functional.hinge_embedding_loss":
                if self.check_arg(api_config, 1, "label"):
                    self.numpy_tensor = numpy.random.randint(0, 2, size=self.shape).astype(
                        self.dtype
                    )
                    self.numpy_tensor[self.numpy_tensor == 0] = -1

            elif api_config.api_name == "paddle.nn.functional.hsigmoid_loss":
                nclass = self.get_arg(api_config, 2, "num_classes")
                weight = self.get_arg(api_config, 3, "weight")
                if key == "label" or index == 1:
                    self.numpy_tensor = numpy.random.randint(0, nclass, size=self.shape).astype(
                        self.dtype
                    )
                elif key == "path_table" or index == 5:
                    self.numpy_tensor = numpy.random.randint(
                        0, weight.shape[0], size=self.shape
                    ).astype(self.dtype)
                elif key == "path_code" or index == 6:
                    self.numpy_tensor = numpy.random.randint(0, 2, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.upsample":
                if self.check_arg(api_config, 1, "size"):
                    self.numpy_tensor = numpy.random.randint(1, 128, size=self.shape).astype(
                        self.dtype
                    )
                if self.check_arg(api_config, 2, "scale_factor"):
                    self.numpy_tensor = numpy.ones(self.shape).astype(self.dtype) + numpy.abs(
                        numpy.random.random(self.shape)
                    ).astype(self.dtype)

            elif api_config.api_name == "paddle.nn.functional.binary_cross_entropy":
                self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name == "paddle.nn.functional.embedding":
                if self.check_arg(api_config, 0, "x") or self.check_arg(api_config, 0, "ids"):
                    weight_config = self.get_arg(api_config, 1, "weight")
                    if not weight_config:
                        weight_config = self.get_arg(api_config, None, "weight")
                    vocab_size = numpy.random.randint(10, 1000)
                    if isinstance(weight_config, TensorConfig) and weight_config.shape:
                        vocab_size = weight_config.shape[0]
                    self.numpy_tensor = numpy.random.randint(0, vocab_size, size=self.shape).astype(
                        self.dtype
                    )
                elif self.check_arg(api_config, 1, "weight"):
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name == "paddle.nn.functional.margin_cross_entropy":
                if index == 1 or key == "label":
                    s = self.get_arg(api_config, 0, "logits")
                    self.numpy_tensor = numpy.random.randint(0, s.shape[1], size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.multi_margin_loss":
                if index == 1 or key == "label":
                    s = self.get_arg(api_config, 0, "input")
                    self.numpy_tensor = numpy.random.randint(0, s.shape[1], size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.cross_entropy":
                if self.check_arg(api_config, 1, "label"):
                    input_shape = self.get_arg(api_config, 0, "input").shape
                    axis = self.get_arg(api_config, 7, "axis", -1)
                    num_classes = self.get_arg(api_config, 0, "input").shape[axis]
                    soft_label = self.get_arg(api_config, 5, "soft_label", False)
                    label_smoothing = self.get_arg(api_config, 6, "label_smoothing", 0.0)
                    if (label_smoothing > 0 and self.shape == input_shape) or (
                        label_smoothing == 0 and soft_label
                    ):
                        soft_labels = numpy.random.random(size=self.shape)
                        soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
                        self.numpy_tensor = soft_labels.astype(self.dtype)
                    else:
                        self.numpy_tensor = numpy.random.randint(
                            0, num_classes, size=self.shape
                        ).astype(self.dtype)
                elif self.check_arg(api_config, 3, "weight"):
                    self.numpy_tensor = numpy.random.random(size=self.shape)
                    self.numpy_tensor = self.numpy_tensor / self.numpy_tensor.sum()

            elif api_config.api_name == "paddle.nn.functional.ctc_loss":
                if self.check_arg(api_config, 1, "labels"):
                    num_classes = self.get_arg(api_config, 0, "log_probs").shape[2] - 1
                    blank = self.get_arg(api_config, 4, "blank", 0)
                    valid_label_indices = [i for i in range(num_classes + 1) if i != blank]
                    if not valid_label_indices:
                        self.numpy_tensor = numpy.zeros(self.shape, dtype=self.dtype)
                    else:
                        self.numpy_tensor = numpy.random.choice(
                            valid_label_indices, size=self.shape, replace=True
                        ).astype(self.dtype)
                elif self.check_arg(api_config, 2, "input_lengths"):
                    max_logit_length = self.get_arg(api_config, 0, "log_probs").shape[0]
                    self.numpy_tensor = numpy.random.randint(
                        1, max_logit_length + 1, size=self.shape, dtype=self.dtype
                    )
                elif self.check_arg(api_config, 3, "label_lengths"):
                    max_label_length = self.get_arg(api_config, 1, "labels").shape[1]
                    max_logit_length = self.get_arg(api_config, 0, "log_probs").shape[0]
                    cand_label_lengths = numpy.random.randint(
                        1, max_label_length + 1, size=self.shape, dtype=self.dtype
                    )
                    compatible_input_lengths = numpy.random.randint(
                        1, max_logit_length + 1, size=self.shape, dtype=self.dtype
                    )
                    final_label_lengths = numpy.minimum(
                        cand_label_lengths, compatible_input_lengths
                    )
                    final_label_lengths = numpy.maximum(final_label_lengths, 1)
                    self.numpy_tensor = final_label_lengths

            elif api_config.api_name == "paddle.nn.functional.dice_loss":
                if self.check_arg(api_config, 1, "label"):
                    num_classes = self.get_arg(api_config, 0, "input").shape[-1]
                    self.numpy_tensor = numpy.random.randint(
                        0, num_classes, size=self.shape, dtype=self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.nll_loss":
                if self.check_arg(api_config, 1, "label"):
                    input_config = self.get_arg(api_config, 0, "input")
                    n_classes = (
                        numpy.random.randint(5, 50)
                        if not isinstance(input_config, TensorConfig)
                        else input_config.shape[1]
                    )
                    self.numpy_tensor = numpy.random.randint(0, n_classes, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.one_hot":
                num_classes_config = self.get_arg(api_config, 1, "num_classes")
                determined_num_classes = None
                default_random_num_classes = numpy.random.randint(1, 65535)
                if isinstance(num_classes_config, int):
                    determined_num_classes = num_classes_config
                elif isinstance(num_classes_config, TensorConfig):
                    if num_classes_config.numpy_tensor is None:
                        if num_classes_config.numel() == 0 or num_classes_config.numel() == 1:
                            num_classes_config.numpy_tensor = numpy.array(
                                [default_random_num_classes], dtype="int64"
                            )
                    determined_num_classes = num_classes_config.numpy_tensor.item()
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = numpy.random.randint(
                        0, determined_num_classes, size=self.shape, dtype=self.dtype
                    )

            elif api_config.api_name == "paddle.nn.functional.rnnt_loss":
                if self.check_arg(api_config, 0, "logits"):
                    if len(self.shape) != 4:
                        self.shape = [3, 4, 3, 5]
                    self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)
                elif self.check_arg(api_config, 1, "labels"):
                    batch_size = 3
                    max_label_len = 2
                    if len(self.shape) != 2:
                        self.shape = [batch_size, max_label_len]
                    vocab_size = 5
                    self.numpy_tensor = numpy.random.randint(
                        1, vocab_size - 1, size=self.shape
                    ).astype(self.dtype)
                elif self.check_arg(api_config, 2, "input_lengths") or self.check_arg(
                    api_config, 3, "label_lengths"
                ):
                    batch_size = 3
                    if len(self.shape) != 1:
                        self.shape = [batch_size]
                    if self.check_arg(api_config, 2, "input_lengths"):
                        max_possible_length = 4
                        self.numpy_tensor = (
                            numpy.ones(self.shape, dtype=self.dtype) * max_possible_length
                        )
                    else:
                        max_possible_length = 2
                        self.numpy_tensor = (
                            numpy.ones(self.shape, dtype=self.dtype) * max_possible_length
                        )

            elif api_config.api_name == "paddle.nn.functional.sequence_mask":
                if self.check_arg(api_config, 0, "x"):
                    maxlen_config = self.get_arg(api_config, 1, "maxlen")
                    provided_maxlen = None
                    if isinstance(maxlen_config, int):
                        provided_maxlen = max(1, maxlen_config)
                    if provided_maxlen is not None:
                        self.numpy_tensor = numpy.random.randint(
                            0, provided_maxlen + 1, size=self.shape
                        ).astype(self.dtype)
                    else:
                        high_val = numpy.random.randint(1, 2048)
                        self.numpy_tensor = numpy.random.randint(
                            0, high_val, size=self.shape
                        ).astype(self.dtype)
                        if self.numpy_tensor.size > 0 and numpy.max(self.numpy_tensor) == 0:
                            fix_value = numpy.random.randint(1, max(2, high_val))
                            first_element_index = numpy.unravel_index(0, self.shape)
                            self.numpy_tensor[first_element_index] = fix_value

            elif api_config.api_name == "paddle.nn.functional.softmax_with_cross_entropy":
                if self.check_arg(api_config, 1, "label"):
                    logits = None
                    if len(api_config.args) > 0 and isinstance(api_config.args[0], TensorConfig):
                        logits = api_config.args[0]
                    elif "logits" in api_config.kwargs and isinstance(
                        api_config.kwargs["logits"], TensorConfig
                    ):
                        logits = api_config.kwargs["logits"]
                    num_classes = 10
                    if logits is not None:
                        axis = api_config.kwargs.get("axis", -1)
                        axis = axis if axis >= 0 else len(logits.shape) + axis
                        if 0 <= axis < len(logits.shape):
                            num_classes = logits.shape[axis]
                    else:
                        num_classes = numpy.random.randint(5, 20)
                    self.numpy_tensor = numpy.random.randint(
                        0, num_classes, size=self.shape
                    ).astype(self.dtype)

            elif api_config.api_name == "paddle.normal":
                if self.check_arg(api_config, 0, "mean"):
                    if "int" in self.dtype:
                        self.numpy_tensor = (
                            numpy.random.randint(-65535, 65535, size=self.shape)
                        ).astype(self.dtype)
                    else:
                        self.numpy_tensor = (numpy.random.random(self.shape) - 0.5).astype(
                            self.dtype
                        )
                elif self.check_arg(api_config, 1, "std"):
                    if "int" in self.dtype:
                        self.numpy_tensor = (
                            numpy.random.randint(0, 65535, size=self.shape)
                        ).astype(self.dtype)
                    else:
                        self.numpy_tensor = (numpy.random.random(self.shape)).astype(self.dtype)
                else:
                    self.numpy_tensor = (numpy.random.randint(0, 1024, size=self.shape)).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.ones":
                if len(self.shape) == 0:
                    self.numpy_tensor = numpy.array(numpy.random.randint(1, 2048), dtype=self.dtype)
                else:
                    self.numpy_tensor = numpy.random.randint(1, 65535, size=self.shape).astype(
                        self.dtype
                    )
            elif api_config.api_name == "paddle.nn.functional.pad":
                if self.check_arg(api_config, 1, "pad"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    min_dim_len = min(x_shape)
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        shape=self.shape, data_type=self.dtype, min=0, max=min_dim_len
                    )
            elif api_config.api_name == "paddle.nn.functional.class_center_sample":
                if self.check_arg(api_config, 0, "label"):
                    num_classes = self.get_arg(api_config, 1, "num_classes")
                    self.numpy_tensor = numpy.random.randint(
                        0, num_classes, size=self.shape
                    ).astype(self.dtype)
            elif api_config.api_name == "paddle.prod":
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)

            elif api_config.api_name == "paddle.vision.ops.psroi_pool":
                if (index is not None and index == 0) or key == "x":
                    self.numpy_tensor = ((numpy.random.random(self.shape)) * 255).astype(self.dtype)
                    if not hasattr(api_config, "x"):
                        api_config.x = self.shape
                elif index == 1 or key == "boxes":
                    if not hasattr(api_config, "boxes"):
                        api_config.boxes = self.shape
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    for i in range(self.shape[0]):
                        self.numpy_tensor[i][0] = numpy.random.random() * (api_config.x[2] - 2)
                        self.numpy_tensor[i][1] = numpy.random.random() * (api_config.x[3] - 2)
                        self.numpy_tensor[i][2] = (
                            numpy.random.random()
                            * (api_config.x[2] - 1 - self.numpy_tensor[i][0] + 1)
                            + self.numpy_tensor[i][0]
                            + 1
                        )
                        self.numpy_tensor[i][3] = (
                            numpy.random.random()
                            * (api_config.x[3] - 1 - self.numpy_tensor[i][1] + 1)
                            + self.numpy_tensor[i][1]
                            + 1
                        )

                elif index == 2 or key == "boxes_num":
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    all = api_config.boxes[0]
                    for i in range(self.numel() - 1):
                        if all < self.numel():
                            self.numpy_tensor[i] = 0
                        else:
                            self.numpy_tensor[i] = numpy.random.randint(
                                1, all - (self.numel() - 1 - i) + 1
                            )
                            all = all - self.numpy_tensor[i]
                    self.numpy_tensor[self.numel() - 1] = all
                else:
                    self.numpy_tensor = numpy.random.randint(0, 1024, self.shape).astype(self.dtype)

            elif api_config.api_name in {
                "paddle.put_along_axis",
                "paddle.Tensor.put_along_axis",
            }:
                if self.check_arg(api_config, 1, "indices"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    x_dims = len(x_tensor.shape) if x_tensor.shape else 0
                    if len(self.shape) != x_dims:
                        new_shape = []
                        for i, dim in enumerate(x_tensor.shape):
                            if i < len(self.shape):
                                new_shape.append(self.shape[i])
                            else:
                                new_shape.append(1)
                        indices = numpy.zeros(new_shape, dtype="int64")
                        for axis in range(x_dims):
                            if axis < len(self.shape):
                                dim_size = x_tensor.shape[axis]
                                if dim_size > 0:
                                    axis_indices = numpy.random.choice(
                                        dim_size, size=new_shape[axis], replace=False
                                    ).astype("int64")
                                    idx_tuple = tuple(
                                        [slice(None)] * axis
                                        + [slice(None, new_shape[axis])]
                                        + [slice(None)] * (x_dims - axis - 1)
                                    )
                                    indices[idx_tuple] = axis_indices.reshape(
                                        [-1] + [1] * (x_dims - axis - 1)
                                    )
                        self.numpy_tensor = indices
                        self.shape = new_shape
                    else:
                        axis = self.get_arg(api_config, 3, "axis")
                        axis = axis if isinstance(axis, int) else 0
                        axis = axis if axis >= 0 else axis + x_dims
                        if 0 <= axis < x_dims:
                            dim_size = x_tensor.shape[axis]
                            indices = numpy.zeros(self.shape, dtype="int64")
                            for idx in numpy.ndindex(tuple(self.shape[:-1])):
                                indices[idx] = numpy.random.choice(
                                    dim_size, size=self.shape[-1], replace=False
                                )
                            self.numpy_tensor = indices
                    self.dtype = "int64"
                elif self.check_arg(api_config, 2, "values"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    indices = self.get_arg(api_config, 1, "indices")
                    if hasattr(indices, "shape"):
                        if indices.shape != self.shape:
                            if numpy.prod(self.shape) == 1:
                                self.numpy_tensor = numpy.full(
                                    indices.shape,
                                    self.get_random_numpy_tensor(shape=[], data_type=self.dtype)[
                                        ()
                                    ],
                                    dtype=self.dtype,
                                )
                            else:
                                random_values = self.get_random_numpy_tensor(
                                    shape=numpy.prod(indices.shape),
                                    data_type=self.dtype,
                                )
                                self.numpy_tensor = random_values.reshape(indices.shape)
                        else:
                            self.numpy_tensor = self.get_random_numpy_tensor(
                                shape=self.shape, data_type=self.dtype
                            )
            elif api_config.api_name == "paddle.quantile":
                if not (key == "x" or index == 0):
                    self.numpy_tensor = numpy.random.rand(1).astype(self.dtype)
            elif api_config.api_name in {"paddle.Tensor.reshape", "paddle.reshape"}:
                if index == 0 or key == "x":
                    if 0 not in self.shape:
                        if not hasattr(api_config, "shape"):
                            api_config.shape = self.shape
                        if not hasattr(api_config, "maxvalue"):
                            api_config.maxvalue = self.numel()
                        if not hasattr(api_config, "tensornum"):
                            api_config.tensornum = 0
                        for arg in api_config.args:
                            if isinstance(arg, (list, tuple)):
                                i = 0
                                for item in arg:
                                    if "int" in str(type(item)):
                                        if item == 0:
                                            api_config.maxvalue = (
                                                api_config.maxvalue // self.shape[i]
                                            )
                                        elif item != -1:
                                            api_config.maxvalue = api_config.maxvalue // item
                                    if "Tensor" in str(type(item)):
                                        api_config.tensornum += 1
                                    i += 1
                        for _thekey, thevalue in api_config.kwargs.items():
                            if isinstance(thevalue, (list, tuple)):
                                i = 0
                                for item in thevalue:
                                    if "int" in str(type(item)):
                                        if item == 0:
                                            api_config.maxvalue = (
                                                api_config.maxvalue // self.shape[i]
                                            )
                                        elif item != -1:
                                            api_config.maxvalue = api_config.maxvalue // item
                                    if "Tensor" in str(type(item)):
                                        api_config.tensornum += 1
                                    i += 1
                else:
                    if api_config.tensornum == 0:
                        api_config.tensornum = 1
                    self.dtype = "int32"
                    if self.shape != [] and self.shape != [1]:
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.shape[0]):
                            if i < self.shape[0] - 1:
                                self.numpy_tensor[i] = numpy.random.randint(
                                    1, api_config.maxvalue + 1
                                )
                                while api_config.maxvalue % self.numpy_tensor[i]:
                                    self.numpy_tensor[i] = numpy.random.randint(
                                        1, api_config.maxvalue + 1
                                    )
                                api_config.maxvalue = api_config.maxvalue // self.numpy_tensor[i]
                            else:
                                self.numpy_tensor[i] = api_config.maxvalue
                    else:
                        if api_config.tensornum == 1:
                            self.numpy_tensor = numpy.random.randint(
                                api_config.maxvalue,
                                api_config.maxvalue + 1,
                                size=self.shape,
                            ).astype(self.dtype)
                        else:
                            api_config.tensornum -= 1
                            self.numpy_tensor = numpy.random.randint(
                                1, api_config.maxvalue + 1, size=self.shape
                            ).astype(self.dtype)
                            while api_config.maxvalue % self.numpy_tensor:
                                self.numpy_tensor = numpy.random.randint(
                                    1, api_config.maxvalue + 1, size=self.shape
                                ).astype(self.dtype)
                            api_config.maxvalue = api_config.maxvalue // self.numpy_tensor

            elif api_config.api_name in {
                "paddle.vision.ops.roi_align",
                "paddle.vision.ops.roi_pool",
            }:
                if (index is not None and index == 0) or key == "x":
                    self.numpy_tensor = ((numpy.random.random(self.shape)) * 255).astype(self.dtype)
                    if not hasattr(api_config, "x"):
                        api_config.x = self.shape
                elif (index is not None and index == 1) or key == "boxes":
                    if not hasattr(api_config, "boxes"):
                        api_config.boxes = self.shape
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    for i in range(self.shape[0]):
                        self.numpy_tensor[i][0] = numpy.random.random() * (api_config.x[2] - 2)
                        self.numpy_tensor[i][1] = numpy.random.random() * (api_config.x[3] - 2)
                        self.numpy_tensor[i][2] = (
                            numpy.random.random()
                            * (api_config.x[2] - 1 - self.numpy_tensor[i][0] + 1)
                            + self.numpy_tensor[i][0]
                            + 1
                        )
                        self.numpy_tensor[i][3] = (
                            numpy.random.random()
                            * (api_config.x[3] - 1 - self.numpy_tensor[i][1] + 1)
                            + self.numpy_tensor[i][1]
                            + 1
                        )
                elif index == 2 or key == "boxes_num":
                    self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                    all = api_config.boxes[0]
                    for i in range(self.numel() - 1):
                        if all < self.numel():
                            self.numpy_tensor[i] = 0
                        else:
                            self.numpy_tensor[i] = numpy.random.randint(
                                1, all - (self.numel() - 1 - i) + 1
                            )
                            all = all - self.numpy_tensor[i]
                    self.numpy_tensor[self.numel() - 1] = all

            elif api_config.api_name.find(".repeat_interleave") > 0:
                if self.check_arg(api_config, 0, "x"):
                    if self.dtype == "bfloat16":
                        self.dtype = "float32"
                elif self.check_arg(api_config, 1, "repeats"):
                    self.numpy_tensor = numpy.random.randint(1, 2048, size=self.shape).astype(
                        self.dtype
                    )
                elif self.check_arg(api_config, 2, "axis"):
                    x_tensor = self.get_arg(api_config, 0, "x")
                    input_dims = len(x_tensor.shape)
                    if len(self.shape) == 0:
                        self.numpy_tensor = numpy.array(
                            numpy.random.randint(-input_dims, input_dims),
                            dtype=self.dtype,
                        )
                    else:
                        self.numpy_tensor = numpy.random.randint(
                            -input_dims, input_dims, size=self.shape
                        ).astype(self.dtype)
            elif api_config.api_name == "paddle.slice":
                # if not hasattr(api_config, "element1"):
                #     if "axes" in api_config.kwargs:
                #         lens = len(api_config.kwargs["axes"])
                #     else:
                #         lens = len(api_config.args[1])
                #     api_config.element1 = lens + 1
                # if not hasattr(api_config, "element2"):
                #     if "starts" in api_config.kwargs:
                #         item = api_config.kwargs["starts"]
                #     else:
                #         item = api_config.args[2]
                #     if isinstance(item, list):
                #         api_config.element2 = api_config.element1 + len(item)
                #     else:
                #         api_config.element2 = api_config.element1 + 1
                axis = api_config.args[1] if len(api_config.args) > 1 else api_config.kwargs["axes"]
                if (index is not None and index == 0) or key == "input":
                    if not hasattr(api_config, "shape"):
                        api_config.shape = self.shape
                elif (index is not None and index == 2) or key == "starts":
                    num = []
                    for i in axis:
                        num.append(api_config.shape[i])
                    if not hasattr(api_config, "indice"):
                        api_config.indice = 0
                    if not hasattr(api_config, "start"):
                        api_config.start = []
                    if self.shape == []:
                        x = numpy.random.randint(0, 2)
                        if x == 0:
                            self.numpy_tensor = numpy.random.randint(
                                0, num[api_config.indice] - 1, self.shape
                            )
                        else:
                            self.numpy_tensor = numpy.random.randint(-65535, -1, self.shape)
                        api_config.start.append(self.numpy_tensor)
                        api_config.indice += 1
                    else:
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.numel()):
                            x = numpy.random.randint(0, 2)
                            if x == 0:
                                self.numpy_tensor[i] = numpy.random.randint(
                                    0, num[api_config.indice] - 1
                                )
                            else:
                                self.numpy_tensor[i] = numpy.random.randint(-65535, -1)
                            api_config.start.append(self.numpy_tensor[i])
                            api_config.indice += 1
                else:
                    if not hasattr(api_config, "start"):
                        if len(api_config.args) > 2:
                            api_config.start = api_config.args[2]
                        else:
                            api_config.start = api_config.kwargs["starts"]
                    num = []
                    for i in axis:
                        num.append(api_config.shape[i])
                    start = api_config.start
                    for i in range(len(start)):
                        if start[i] < 0:
                            start[i] = start[i] if start[i] > -1 * num[i] else -1 * num[i]
                            start[i] += num[i]
                    if not hasattr(api_config, "index"):
                        api_config.index = 0
                    if self.shape == []:
                        x = numpy.random.randint(0, 2)
                        if x == 0:
                            self.numpy_tensor = numpy.random.randint(
                                start[api_config.index] + 1, 65535, self.shape
                            )
                        else:
                            if start[api_config.index] - num[i] == 0:
                                start[api_config.index] -= 1
                            self.numpy_tensor = numpy.random.randint(
                                min(start[api_config.index] - num[i] + 1, -1),
                                0,
                                self.shape,
                            )
                        api_config.index += 1
                    else:
                        self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                        for i in range(self.numel()):
                            x = numpy.random.randint(0, 2)
                            if x == 0:
                                self.numpy_tensor[i] = numpy.random.randint(
                                    start[api_config.index] + 1, 65535
                                )
                            else:
                                if start[api_config.index] - num[i] == 0:
                                    start[api_config.index] -= 1
                                self.numpy_tensor[i] = numpy.random.randint(
                                    start[api_config.index] - num[api_config.index] + 1,
                                    0,
                                )
                            api_config.index += 1

            elif api_config.api_name == "paddle.scatter":
                if key == "index" or index == 1:
                    d = self.get_arg(api_config, 0, "x")
                    s = d.shape[0]
                    overwrite = self.get_arg(api_config, 3, "overwrite")
                    if (overwrite == None or overwrite == True) and (
                        self.shape == [] or self.shape[0]
                    ) <= s:
                        self.numpy_tensor = numpy.random.choice(
                            s, size=self.shape, replace=False
                        ).astype(self.dtype)
                    else:
                        self.numpy_tensor = numpy.random.randint(0, s, size=self.shape).astype(
                            self.dtype
                        )

            elif api_config.api_name == "paddle.scatter_nd":
                future_data = self.get_arg(api_config, 2, "shape")
                if (key == "index" or index == 0) and future_data and len(future_data):
                    self.numpy_tensor = numpy.zeros(self.shape)
                    s = self.shape
                    for ii in range(len(future_data)):
                        if ii >= s[-1]:
                            break
                        self.numpy_tensor[..., ii] = numpy.random.randint(
                            -future_data[ii],
                            future_data[ii],
                            size=self.numpy_tensor[..., ii].shape,
                        ).astype(self.dtype)

            elif api_config.api_name == "paddle.scatter_nd_add":
                if key == "index" or index == 1:
                    org = self.get_arg(api_config, 0, "x")
                    org = org.shape
                    self.numpy_tensor = numpy.zeros(self.shape)
                    ind = self.get_arg(api_config, 1, "index")
                    s = ind.shape
                    for ii in range(s[-1]):
                        self.numpy_tensor[..., ii] = numpy.random.randint(
                            -org[ii], org[ii], size=self.numpy_tensor[..., ii].shape
                        ).astype(self.dtype)
            elif api_config.api_name == "paddle.shard_index":
                if self.check_arg(api_config, 0, "input"):
                    index_num = self.get_arg(api_config, 1, "index_num")
                    if index_num is None:
                        index_num = numpy.random.randint(1, 1000)
                    self.numpy_tensor = numpy.random.randint(0, index_num, size=self.shape).astype(
                        self.dtype
                    )
            elif api_config.api_name in {"paddle.sum", "paddle.squeeze"}:
                if self.check_arg(api_config, 1, "axis"):
                    self.numpy_tensor = self.generate_random_axes(api_config)
            elif api_config.api_name == "paddle.split":
                if self.check_arg(api_config, 2, "axis"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    num_or_sections = self.get_arg(api_config, 1, "num_or_sections")
                    if isinstance(num_or_sections, (list, tuple)):
                        neg_one_count = sum(1 for x in num_or_sections if x == -1)
                        if neg_one_count > 1:
                            raise ValueError(
                                f"num_or_sections can contain at most one -1, but got {num_or_sections}"
                            )
                        num_splits = len(num_or_sections)
                        known_size = sum(num_or_sections) + neg_one_count
                    elif isinstance(num_or_sections, int):
                        num_splits = num_or_sections
                        known_size = None
                    else:
                        raise ValueError(
                            f"num_or_sections must be an int, list, or tuple, but got {type(num_or_sections)}"
                        )

                    target_dim = None
                    max_dim = len(x_shape)
                    if max_dim == 0:
                        target_dim = numpy.random.randint(-1, 0)
                    else:
                        for dim in range(max_dim):
                            dim_size = x_shape[dim]
                            if isinstance(num_or_sections, int) and dim_size % num_splits == 0:
                                target_dim = dim
                            elif isinstance(num_or_sections, (list, tuple)):
                                if (neg_one_count == 0 and dim_size == known_size) or (
                                    neg_one_count == 1 and dim_size > known_size
                                ):
                                    target_dim = dim
                    if target_dim is None:
                        raise ValueError(
                            f"No valid axis found for paddle.split with x.shape={x_shape} and num_or_sections={num_or_sections}"
                        )

                    shape_len = len(self.shape)
                    if shape_len == 0:
                        self.numpy_tensor = numpy.array(target_dim, dtype=self.dtype)
                    elif shape_len == 1 and self.shape[0] == 1:
                        self.numpy_tensor = numpy.array([target_dim], dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.split. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )

            elif api_config.api_name == "paddle.nn.functional.softmax":
                # for TensorConfig axis
                x_tensor_config = self.get_arg(api_config, 0, "x")
                axis_config = self.get_arg(api_config, 1, "axis")

                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype)
                elif self.check_arg(api_config, 1, "axis"):
                    len_shape_x = len(x_tensor_config.shape)
                    # specify if axis is a scalar tensor, else is a int according to doc
                    if isinstance(axis_config, TensorConfig):
                        axis = self.get_random_numpy_tensor(
                            axis_config.shape,
                            axis_config.dtype,
                            min=-len_shape_x,
                            max=len_shape_x,
                        )
                        self.numpy_tensor = axis

            elif api_config.api_name == "paddle.standard_gamma":
                self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name == "paddle.standard_normal":
                if index == 0 or key == "shape":
                    self.numpy_tensor = numpy.random.randint(1, 128, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.strided_slice":
                s = self.get_arg(api_config, 0, "x")
                if self.check_arg(api_config, 1, "axes"):
                    self.numpy_tensor = numpy.random.randint(
                        0, len(s.shape), size=self.shape
                    ).astype(self.dtype)
                elif index:
                    axes = self.get_arg(api_config, 1, "axes")
                    for i in range(len(axes)):
                        if isinstance(axes[i], TensorConfig):
                            axes[i] = int(axes[i].numpy_tensor)
                    if self.check_arg(api_config, 2, "starts"):
                        axes = self.get_arg(api_config, 1, "axes")
                        if not isinstance(axes, list):
                            axes = axes.numpy_tensor
                        ind = kwargs["list_index"][0]
                        self.numpy_tensor = numpy.random.randint(
                            0, s.shape[axes[ind]] - 1, size=self.shape
                        ).astype(self.dtype)
                    elif self.check_arg(api_config, 3, "ends"):
                        ind = kwargs["list_index"][0]
                        pre = self.get_arg(api_config, 2, "starts")
                        self.numpy_tensor = numpy.random.randint(
                            pre[ind].numpy_tensor + 1,
                            s.shape[axes[ind]],
                            size=self.shape,
                        ).astype(self.dtype)
                    elif self.check_arg(api_config, 4, "strides"):
                        ind = kwargs["list_index"][0]
                        self.numpy_tensor = numpy.random.randint(
                            1, s.shape[axes[ind]], size=self.shape
                        ).astype(self.dtype)
            elif api_config.api_name == "paddle.tensordot":
                if index == 0 or key == "x":
                    if not hasattr(api_config, "shape1"):
                        api_config.shape1 = self.shape
                elif index == 1 or key == "y":
                    if not hasattr(api_config, "shape2"):
                        api_config.shape2 = self.shape
                else:
                    item = self.get_arg(api_config, 2, "axes")
                    num = len(api_config.shape1)
                    used = []
                    if isinstance(item, (list, tuple)):
                        if not hasattr(api_config, "tensor1"):
                            self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                            for i in range(self.numel()):
                                self.numpy_tensor[i] = numpy.random.randint(0, num)
                                while (
                                    api_config.shape1[self.numpy_tensor[i]] not in api_config.shape2
                                    or self.numpy_tensor[i] in used
                                ):
                                    self.numpy_tensor[i] = numpy.random.randint(0, num)
                                used.append(self.numpy_tensor[i])
                            api_config.tensor1 = self.numpy_tensor
                        else:
                            self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                            for i in range(self.numel()):
                                self.numpy_tensor[i] = numpy.random.randint(0, num)
                                while (
                                    api_config.shape2[self.numpy_tensor[i]]
                                    != api_config.shape1[api_config.tensor1[i]]
                                    or self.numpy_tensor[i] in used
                                ):
                                    self.numpy_tensor[i] = numpy.random.randint(0, num)
                                used.append(self.numpy_tensor[i])

                    elif isinstance(item, TensorConfig):
                        self.tensor = numpy.random.randint(0, 2, size=self.shape).astype(self.dtype)
                        if self.numel() == 1:
                            self.numpy_tensor = numpy.random.randint(0, num, self.shape).astype(
                                self.dtype
                            )
                            dim1 = len(api_config.shape1)
                            dim2 = len(api_config.shape2)
                            min_dim = min(dim1, dim2)
                            candidate_set = set()
                            for i in range(min_dim):
                                if api_config.shape1[i] == api_config.shape2[i]:
                                    candidate_set.add(i)
                            if candidate_set:
                                import random

                                self.numpy_tensor = [random.choice(list(candidate_set))]
                            else:
                                raise ValueError(
                                    f"No valid axis found for tensordot,x shape {api_config.shape1}, y shape {api_config.shape2},axes {item}"
                                )
                        else:
                            used1 = []
                            used2 = []
                            self.numpy_tensor = numpy.zeros(self.shape).astype(self.dtype)
                            for i in range(self.shape[0]):
                                self.numpy_tensor[0][i] = numpy.random.randint(0, num)
                                self.numpy_tensor[1][i] = numpy.random.randint(0, num)
                                while (
                                    api_config.shape1[self.numpy_tensor[0][i]]
                                    != api_config.shape2[self.numpy_tensor[1][i]]
                                    or self.numpy_tensor[0][i] in used1
                                    or self.numpy_tensor[1][i] in used2
                                ):
                                    self.numpy_tensor[0][i] = numpy.random.randint(0, num)
                                    self.numpy_tensor[1][i] = numpy.random.randint(0, num)
                                used1.append(self.numpy_tensor[0][i])
                                used2.append(self.numpy_tensor[1][i])

            elif api_config.api_name in {
                "paddle.Tensor.take_along_axis",
                "paddle.take_along_axis",
            }:
                if self.check_arg(api_config, 1, "indices"):
                    arr_config = self.get_arg(api_config, 0, "arr")
                    axis = self.get_arg(api_config, 2, "axis")
                    arr_shape = arr_config.shape
                    arr_rank = len(arr_shape)
                    axis_val = axis if axis >= 0 else axis + arr_rank
                    dim_size = arr_shape[axis_val]
                    if self.dtype not in ["int32", "int64"]:
                        self.dtype = "int64"
                    num_elements = self.numel()
                    if num_elements == 0:
                        indices = numpy.array([], dtype=self.dtype)
                    elif dim_size == 1:
                        indices = numpy.zeros(num_elements, dtype=self.dtype)
                    elif num_elements == 1:
                        indices = numpy.array([0], dtype=self.dtype)
                    else:
                        indices = numpy.random.randint(0, dim_size, size=num_elements).astype(
                            self.dtype
                        )
                        positions_to_replace = numpy.random.choice(
                            num_elements, size=2, replace=False
                        )
                        flat_indices = indices.flatten()
                        flat_indices[positions_to_replace[0]] = 0
                        flat_indices[positions_to_replace[1]] = dim_size - 1
                        indices = flat_indices
                    self.numpy_tensor = indices.reshape(self.shape)

            elif api_config.api_name == "paddle.take":
                if self.check_arg(api_config, 1, "index"):
                    x = self.get_arg(api_config, 0, "x")
                    dim_size = numpy.prod(x.shape)
                    self.numpy_tensor = numpy.random.randint(0, dim_size, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name in {"paddle.Tensor.gather", "paddle.gather"}:
                if key == "index" or index == 1:
                    s = self.get_arg(api_config, 0, "x")
                    if "axis" in api_config.kwargs:
                        tmp = self.get_arg(api_config, 2, "axis")
                        if isinstance(tmp, TensorConfig):
                            tmp = tmp.shape
                            tmp = tmp[0]
                    else:
                        tmp = 0
                    self.numpy_tensor = (
                        numpy.random.randint(0, s.shape[tmp], size=self.shape)
                    ).astype(self.dtype)
                elif key == "axis" or index == 2:
                    self.numpy_tensor = (numpy.random.randint(0, 2, size=self.shape)).astype(
                        self.dtype
                    )

            elif api_config.api_name in {"paddle.Tensor.gather_nd", "paddle.gather_nd"}:
                if key == "index" or index == 1:
                    org = self.get_arg(api_config, 0, "x")
                    org = org.shape
                    s = self.get_arg(api_config, 1, "index")
                    s = s.shape
                    self.numpy_tensor = numpy.zeros(s)
                    for i in range(s[-1]):
                        self.numpy_tensor[..., i] = (
                            numpy.random.randint(0, org[i], size=self.numpy_tensor[..., i].shape)
                        ).astype(self.dtype)

            elif api_config.api_name in {
                "paddle.Tensor.index_select",
                "paddle.index_select",
            }:
                if self.check_arg(api_config, 1, "index") or self.check_arg(api_config, 2, "index"):
                    axis = self.get_arg(api_config, 2, "axis")
                    if axis is None:
                        axis = 0
                    inputs = self.get_arg(api_config, 0, "x")
                    self.numpy_tensor = numpy.random.randint(
                        0, inputs.shape[axis], size=self.shape
                    ).astype(self.dtype)

            elif api_config.api_name in {"paddle.Tensor.index_put", "paddle.index_put"}:
                if self.check_arg(api_config, 1, "indices") and not self.get_arg(
                    api_config, 3, "accumulate"
                ):
                    # NOTE(zrr1999): If accumulate is False, the behavior is undefined if indices contain duplicate elements in torch.

                    inputs = self.get_arg(api_config, 0, "x")
                    value = self.get_arg(api_config, 2, "value")
                    inputs_numel = inputs.numel()
                    value_numel = value.numel()
                    if inputs_numel < value_numel:
                        raise ValueError(
                            "Invalid input for paddle.index_put: inputs.numel() < value.numel() when accumulate=False. "
                        )
                    inputs_shape = inputs.shape
                    value_shape = value.shape

                    flat_indices = numpy.random.choice(
                        inputs_numel, size=value_numel, replace=False
                    )
                    indices = [
                        index.reshape(value_shape)
                        for index in numpy.unravel_index(flat_indices, inputs_shape)
                    ]
                    self.numpy_tensor = indices.astype(self.dtype)

            elif api_config.api_name == "paddle.Tensor.tile":
                if index == 1 or key == "repeat_times":
                    self.numpy_tensor = numpy.random.randint(1, 128, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name == "paddle.tile":
                if self.check_arg(api_config, 1, "repeat_times"):
                    self.numpy_tensor = numpy.random.randint(1, 128, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name in {"paddle.topk", "paddle.Tensor.topk"}:
                if self.check_arg(api_config, 0, "x"):
                    x_numel = self.numel()
                    if self.dtype in {"bfloat16", "float32", "float64"}:
                        self.numpy_tensor = numpy.linspace(
                            -x_numel, x_numel, x_numel, dtype=self.dtype
                        ).reshape(self.shape)
                        if numpy.unique(self.numpy_tensor).size < x_numel:
                            self.numpy_tensor = generate_unique_array(x_numel, self.dtype).reshape(
                                self.shape
                            )
                    elif self.dtype == "float16":
                        self.numpy_tensor = generate_unique_array(x_numel, self.dtype).reshape(
                            self.shape
                        )
                    elif self.dtype in {"int32", "int64"}:
                        self.numpy_tensor = self.get_random_numpy_tensor(
                            self.shape, self.dtype, min=1
                        )
                    else:
                        raise ValueError(
                            f"Unsupported dtype {self.dtype} for paddle.topk / paddle.Tensor.topk"
                        )
                elif self.check_arg(api_config, 1, "k"):
                    x_config = self.get_arg(api_config, 0, "x")
                    axis = self.get_arg(api_config, 2, "axis", -1)
                    max_k_value = 1
                    if isinstance(x_config, TensorConfig) and x_config.shape:
                        max_k_value = x_config.shape[axis] if len(x_config.shape) > 0 else 1
                    if not self.shape:
                        self.numpy_tensor = numpy.array(
                            numpy.random.randint(1, max_k_value + 1), dtype=self.dtype
                        )
                    else:
                        self.numpy_tensor = numpy.random.randint(
                            1, max_k_value + 1, size=self.shape
                        ).astype(self.dtype)
            elif api_config.api_name in {"paddle.Tensor.unflatten", "paddle.unflatten"}:
                if self.check_arg(api_config, 1, "axis"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    self.numpy_tensor = numpy.random.randint(
                        0, len(x_shape), size=self.shape
                    ).astype(self.dtype)
                elif self.check_arg(api_config, 2, "shape"):
                    axis = self.get_arg(api_config, 1, "axis")
                    x_dim = self.get_arg(api_config, 0, "x").shape[axis]
                    shape = self.get_arg(api_config, 2, "shape")
                    if isinstance(shape, TensorConfig):
                        self.numpy_tensor = numpy.ones(self.shape).astype(self.dtype)
                        remaining = x_dim
                        for i in range(shape.numel() - 1):
                            if remaining <= 1:
                                break
                            divisors = [d for d in range(2, remaining + 1) if remaining % d == 0]
                            if divisors:
                                divisor = numpy.random.choice(divisors)
                                self.numpy_tensor[i] = divisor
                                remaining = remaining // divisor
                        self.numpy_tensor[-1] = remaining
                    elif isinstance(shape, (list, tuple)):
                        tensor_configs = [item for item in shape if isinstance(item, TensorConfig)]
                        tensornum = len(tensor_configs)
                        if tensornum > 0:
                            remaining = x_dim
                            fixed_product = 1
                            for dim in shape:
                                if isinstance(dim, int) and dim != -1:
                                    fixed_product *= dim
                            if fixed_product > 0 and x_dim % fixed_product == 0:
                                remaining = x_dim // fixed_product
                            for i in range(tensornum - 1):
                                if remaining <= 1:
                                    break
                                divisors = [
                                    d for d in range(2, remaining + 1) if remaining % d == 0
                                ]
                                if divisors:
                                    divisor = numpy.random.choice(divisors)
                                    tensor_config = tensor_configs[i]
                                    tensor_config.numpy_tensor = numpy.full(
                                        tensor_config.shape,
                                        divisor,
                                        dtype=tensor_config.dtype,
                                    )
                                    remaining = remaining // divisor
                            tensor_config = tensor_configs[-1]
                            tensor_config.numpy_tensor = numpy.full(
                                tensor_config.shape,
                                remaining,
                                dtype=tensor_config.dtype,
                            )

            elif api_config.api_name == "paddle.unsqueeze":
                if self.check_arg(api_config, 1, "axis"):
                    x_shape = self.get_arg(api_config, 0, "x").shape
                    max_dim = len(x_shape) + 1
                    if len(self.shape) == 0:
                        dim = numpy.random.randint(0, max_dim)
                        if numpy.random.rand() > 0.5:
                            dim -= max_dim
                        self.numpy_tensor = numpy.array(dim, dtype=self.dtype)
                    elif len(self.shape) == 1:
                        dims = numpy.random.choice(max_dim, size=self.shape[0], replace=False)
                        mask = numpy.random.rand(self.shape[0]) > 0.5
                        dims = numpy.where(mask, dims - max_dim, dims)
                        self.numpy_tensor = numpy.array(dims, dtype=self.dtype)
                    else:
                        raise ValueError(
                            f"Invalid shape for 'axis' Tensor in paddle.unsqueeze. "
                            f"Expected a 0-D or 1-D Tensor, but got shape {self.shape}."
                        )
            elif (
                api_config.api_name
                == "paddle.incubate.nn.functional.variable_length_memory_efficient_attention"
            ):
                if self.check_arg(api_config, 3, "seq_lens"):
                    q_seq_len = self.get_arg(api_config, 0, "query").shape[2]
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        shape=self.shape, data_type=self.dtype, min=1, max=q_seq_len
                    )
                elif self.check_arg(api_config, 4, "kv_seq_lens"):
                    k_seq_len = self.get_arg(api_config, 1, "key").shape[2]
                    v_seq_len = self.get_arg(api_config, 2, "value").shape[2]
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        shape=self.shape,
                        data_type=self.dtype,
                        min=1,
                        max=min(k_seq_len, v_seq_len),
                    )
                elif self.check_arg(api_config, 5, "mask"):
                    # mask should between -inf and 0 (0 is included)
                    # eps = numpy.finfo(self.dtype).eps
                    # self.numpy_tensor = self.get_random_numpy_tensor(shape=self.shape, data_type=self.dtype, max=0 + eps)
                    # mask should be -inf(masked) or 0(not masked)
                    self.numpy_tensor = numpy.random.randint(0, 2, size=self.shape).astype(
                        self.dtype
                    ) * (numpy.finfo(self.dtype).min)
            elif api_config.api_name == "paddle.zeros":
                self.numpy_tensor = numpy.random.randint(0, 2048, size=self.shape)

            elif api_config.api_name == "paddle.nn.functional.zeropad2d":
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = self.get_random_numpy_tensor(self.shape, self.dtype)
                elif self.check_arg(api_config, 1, "padding"):
                    # padding value should not be too large
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        self.shape, self.dtype, min=0, max=10
                    )

            elif api_config.api_name == "paddle.Tensor.__getitem__":
                if self.check_arg(api_config, 1, "item"):
                    arr = self.get_arg(api_config, 0, "arr")
                    min_dim = min(arr.shape)
                    if self.dtype == "bool":
                        indices = numpy.random.choice([0, 1], size=self.numel())
                    else:
                        indices = numpy.random.randint(0, min_dim, size=self.numel())
                    self.numpy_tensor = indices.reshape(self.shape).astype(self.dtype)

            elif api_config.api_name == "paddle.Tensor.__setitem__":
                if self.check_arg(api_config, 1, "item"):
                    arr = self.get_arg(api_config, 0, "arr")
                    value = self.get_arg(api_config, 2, "value")
                    min_dim = min(arr.shape)
                    if value is not None and hasattr(value, "shape"):
                        indices = numpy.zeros(self.numel(), dtype="int64")
                        num_true = min(value.shape[0], self.numel())
                        true_indices = numpy.random.choice(
                            self.numel(), size=num_true, replace=False
                        )
                        indices[true_indices] = 1
                    else:
                        indices = numpy.random.choice([0, 1], size=self.numel())
                    self.numpy_tensor = indices.reshape(self.shape).astype(self.dtype)

            elif api_config.api_name == "paddle.poisson":
                self.numpy_tensor = numpy.random.random(self.shape).astype(self.dtype)

            elif api_config.api_name in {
                "paddle.Tensor.__pow__",
                "paddle.Tensor.pow",
                "paddle.pow",
                "paddle.Tensor.__rpow__",
            }:
                dtype = self.dtype

                def get_base_max(value, dtype_max, default_max=5):
                    value_max = default_max
                    if value <= 0:
                        return value_max
                    if value < 1:
                        # value**(-max) < MAX => (1/value)**max < MAX
                        value = 1 / value
                    ln_value = math.log(value)
                    # dy/dx = y*ln(value) < MAX, y < MAX => y*max(ln(value), 1) < MAX
                    output_max = dtype_max / max(1, ln_value)
                    value_max = math.log(output_max) / ln_value
                    if isinstance(value, int):
                        value_max = math.floor(value_max)
                    return value_max

                def get_exponent_max(value, dtype_max, default_max=5):
                    value_max = default_max
                    if isinstance(value, (int, float, bool, numpy.number)):
                        if value <= 2:
                            return value_max
                        value_max = math.pow(dtype_max / value, 1 / value)
                        if isinstance(value, int):
                            value_max = math.floor(value_max)
                    return value_max

                if api_config.api_name == "paddle.Tensor.__rpow__":
                    # paddle.Tensor.__rpow__(a, b) => b ^ a, where a is self and b is other
                    is_base_arg = self.check_arg(api_config, 1, "other")
                    if is_base_arg:
                        const = self.get_arg(api_config, 0, "self")
                        get_max = get_base_max
                        default_max = 10
                    else:
                        const = self.get_arg(api_config, 1, "other")
                        get_max = get_exponent_max
                        default_max = 5
                else:
                    # paddle.Tensor.__pow__(a, b) => a ^ b, where a is self and b is other
                    is_base_arg = self.check_arg(api_config, 0, "self") or self.check_arg(
                        api_config, 0, "x"
                    )
                    if is_base_arg:
                        const = self.get_arg(api_config, 1, "other")
                        get_max = get_base_max
                        default_max = 10
                    else:
                        const = self.get_arg(api_config, 0, "self")
                        get_max = get_exponent_max
                        default_max = 5
                if isinstance(const, (int, float, bool, numpy.number)):
                    value_max = get_max(const, numpy.finfo(self.dtype).max, default_max)
                    if is_base_arg and int(const) != const:
                        # Avoid situations like (-2.3) ^ 0.5
                        self.numpy_tensor = self.get_random_numpy_tensor(
                            self.shape, self.dtype, min=0, max=value_max
                        )
                    else:
                        self.numpy_tensor = self.get_random_numpy_tensor(
                            self.shape, self.dtype, min=-value_max, max=value_max
                        )
                else:
                    if is_base_arg:
                        # Avoid situations like (-2.3) ^ 0.5
                        self.numpy_tensor = self.get_random_numpy_tensor(
                            self.shape, self.dtype, min=0, max=default_max
                        )
                    else:
                        self.numpy_tensor = self.get_random_numpy_tensor(
                            self.shape, self.dtype, min=-default_max, max=default_max
                        )
            elif api_config.api_name == "paddle.nn.functional.sigmoid_focal_loss":
                if self.check_arg(api_config, 1, "label"):
                    self.numpy_tensor = numpy.random.randint(low=0, high=2, size=self.shape).astype(
                        self.dtype
                    )

            elif api_config.api_name.endswith("cholesky_solve"):
                if self.check_arg(api_config, 1, "y"):
                    is_upper = self.get_arg(api_config, 2, "upper")
                    if is_upper:
                        self.numpy_tensor = numpy.triu(
                            self.get_random_numpy_tensor(self.shape, self.dtype)
                        )
                    else:
                        self.numpy_tensor = numpy.tril(
                            self.get_random_numpy_tensor(self.shape, self.dtype)
                        )
            elif api_config.api_name in {"paddle.rsqrt", "paddle.Tensor.rsqrt"}:
                if self.check_arg(api_config, 0, "x"):
                    self.numpy_tensor = self.get_random_numpy_tensor(
                        self.shape, self.dtype, min=1e-7, max=1000
                    )
            elif api_config.api_name in {"paddle.remainder", "paddle.Tensor.remainder"}:
                if self.check_arg(api_config, 1, "y"):
                    if self.dtype in {"int32", "int64"}:
                        self.numpy_tensor = self.get_random_numpy_tensor(
                            self.shape, self.dtype, min=1
                        )

            if self.numpy_tensor is None:
                if self.shape == []:
                    if "int" in self.dtype:
                        scalar_val = numpy.random.randint(-65535, 65535)
                        self.numpy_tensor = numpy.array(scalar_val, dtype=self.dtype)
                    else:
                        scalar_val = (numpy.random.random() - 0.5) * 1.2
                        self.numpy_tensor = numpy.array(scalar_val, dtype=self.dtype)
                elif USE_CACHED_NUMPY and self.dtype not in ["int64", "float64"]:
                    self.numpy_tensor = self.get_cached_numpy(self.dtype, self.shape)
                else:
                    if "int" in self.dtype:
                        self.numpy_tensor = (
                            numpy.random.randint(-65535, 65535, size=self.shape)
                        ).astype(self.dtype)
                    else:
                        self.numpy_tensor = ((numpy.random.random(self.shape) - 0.5) * 1.2).astype(
                            self.dtype
                        )

        self.dtype = original_dtype
        return self.numpy_tensor

    def get_paddle_tensor(self, api_config):
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        if self.paddle_tensor is None:
            self.paddle_tensor = paddle.to_tensor(
                self.get_numpy_tensor(api_config),
                dtype="float32" if self.dtype == "bfloat16" else self.dtype,
                place=self.place,
            )

            self.paddle_tensor.stop_gradient = False
            if self.dtype == "bfloat16":
                self.paddle_tensor = paddle.cast(self.paddle_tensor, dtype="bfloat16")
        return self.paddle_tensor

    def get_torch_tensor(self, api_config):
        if self.dtype in ["float8_e5m2", "float8_e4m3fn"]:
            print("Warning ", self.dtype, "not supported")
            return

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        torch.set_default_device(device)
        if self.torch_tensor is None:
            self.torch_tensor = torch.tensor(
                self.get_numpy_tensor(api_config),
                dtype=self.convert_dtype_to_torch_type(self.dtype)
                if self.dtype != "bfloat16"
                else torch.float32,
                requires_grad=self.dtype
                in [
                    "float32",
                    "float64",
                    "float16",
                    "complex64",
                    "complex128",
                    "bfloat16",
                ],
            )
            if self.dtype == "bfloat16":
                self.torch_tensor = self.torch_tensor.to(dtype=torch.bfloat16)
        return self.torch_tensor

    def clear_tensor(self):
        self.torch_tensor = None
        self.paddle_tensor = None
        self.numpy_tensor = None
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        del self.paddle_tensor
        self.paddle_tensor = None
        paddle.device.cuda.empty_cache()

    def clear_numpy_tensor(self):
        del self.numpy_tensor
        self.numpy_tensor = None

    def clear_torch_tensor(self):
        del self.torch_tensor
        self.torch_tensor = None
        torch.cuda.empty_cache()

    def fill_numpy_tensor(self, full_value):
        self.numpy_tensor = numpy.full(shape=self.shape, fill_value=full_value, dtype=self.dtype)

    def check_arg(self, api_config, arg_pos, arg_name):
        """检查api_config中的参数是否与当前实例匹配。
        必须同时提供参数位置与参数名称, 具体请查看API文档。

        Args:
            api_config (ApiConfig): API配置对象, 包含args和kwargs。
            arg_pos (int): 参数的位置索引。
            arg_name (str): 参数的名称。

        Returns:
            bool: 如果参数匹配当前实例，则返回 True; 否则返回 False。

        """
        return (hasattr(self, "index") and self.index == arg_pos) or (
            hasattr(self, "key") and self.key == arg_name
        )

    def get_arg(self, api_config, arg_pos, arg_name, default=None):
        """从api_config中获取参数值。
        必须同时提供参数位置与参数名称, 具体请查看API文档。

        Args:
            api_config (ApiConfig): API配置对象, 包含args和kwargs。
            arg_pos (int): 参数的位置索引。
            arg_name (str): 参数的名称。
            default (Any, optional): 参数的默认值。默认为None。

        Returns:
            Any: 参数的值。如果参数位置索引有效, 则返回args列表中对应位置的值;
                    如果参数名称在kwargs字典中存在, 则返回对应名称的值;
                    否则返回默认值。

        """
        if 0 <= arg_pos < len(api_config.args):
            return api_config.args[arg_pos]
        if arg_name in api_config.kwargs:
            return api_config.kwargs[arg_name]
        return default

    def get_initialized_value(self, api_config, arg_pos=None, arg_name=None):
        """Get the initialized numpy_tensor value from the api_config instead of the TensorConfig"""
        # for uninitialized numpy_tensor, return None implicitly as numpy_tensor is None
        if arg_pos is not None and 0 <= arg_pos < len(api_config.args):
            if isinstance(api_config.args[arg_pos], TensorConfig):
                return api_config.args[arg_pos].numpy_tensor
            else:
                return api_config.args[arg_pos]
        if arg_name and arg_name in api_config.kwargs:
            if isinstance(api_config.kwargs[arg_name], TensorConfig):
                return api_config.kwargs[arg_name].numpy_tensor
            else:
                return api_config.kwargs[arg_name]
        # for args that does not appear in api_config
        if arg_pos >= len(api_config.args) or arg_name not in api_config.kwargs:
            return None
        # error case
        if arg_pos is None and arg_name is None:
            raise ValueError("either arg_pos or arg_name must be provided.")
        elif arg_pos:
            if arg_pos < 0:
                raise IndexError(
                    f"argument position {arg_pos} is out of range for api_config with {len(api_config.args)} arguments."
                )
            else:
                # case type(api_config.args[arg_pos]) != TensorConfig:
                raise TypeError(f"argument at position {arg_pos} is not of type TensorConfig.")
        else:
            # case type(api_config.kwargs[arg_name]) != TensorConfig:
            raise TypeError(f"argument '{arg_name}' is not of type TensorConfig.")

    def set_tensor_arg_value(self, api_config, arg_pos=None, arg_name=None, value=None):
        if (
            arg_pos is not None
            and 0 <= arg_pos < len(api_config.args)
            and isinstance(api_config.args[arg_pos], TensorConfig)
        ):
            api_config.args[arg_pos].numpy_tensor = value
        elif (
            arg_name
            and arg_name in api_config.kwargs
            and isinstance(api_config.kwargs[arg_name], TensorConfig)
        ):
            api_config.kwargs[arg_name].numpy_tensor = value
        else:
            raise ValueError(
                f"argument at position {arg_pos} or name '{arg_name}' is not of type TensorConfig."
            )

    def get_random_numpy_tensor(self, shape=None, data_type=None, min=None, max=None):
        """Generate a random numpy tensor with data in [min, max) given shape and data_type"""
        if "int" in data_type:
            min = min if min is not None else -65535
            max = max if max is not None else 65535
            numpy_tensor = (numpy.random.randint(min, max, size=shape)).astype(data_type)
        else:
            dtype = "float32" if data_type == "bfloat16" else data_type
            min = min if min is not None else numpy.finfo(dtype).min / 2
            max = max if max is not None else numpy.finfo(dtype).max / 2
            numpy_tensor = (numpy.random.uniform(min, max, size=shape)).astype(dtype)
        return numpy_tensor


class APIConfig:
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.args = copy.deepcopy(self.args)
        result.kwargs = copy.deepcopy(self.kwargs)
        result.api_name = self.api_name
        return result

    def __init__(self, config):
        config = config.replace("\n", "")
        self.config = config
        self.args = []
        self.kwargs = collections.OrderedDict()
        config = config.replace("Tensor(", "TensorConfig(")

        self.api_name, offset = self.get_api(config)

        if self.api_name == "paddle.einsum":
            tmp = config[config.index('"') + 1 :]
            value = tmp[: tmp.index('"')]
            offset = config.index('"') + 1 + tmp.index('"')
            if "equation" in config:
                self.append_kwargs("equation", value)
            else:
                self.append_args(value)

        while True:
            tocken, offset = self.get_tocken(config, offset)
            if offset is None:
                return

            is_kwarg = config[offset] == "="
            if is_kwarg:
                key = tocken
                tocken, offset = self.get_tocken(config, offset + 1)

            value, offset = self.get_one_arg(tocken, config, offset)

            if offset is None:
                return

            if is_kwarg:
                self.append_kwargs(key, value)
            else:
                self.append_args(value)

    def append_args(self, arg):
        self.args.append(arg)

    def append_kwargs(self, name, arg):
        self.kwargs[name] = arg

    def dump_item_str(self, item):
        type_mapping = {
            numpy.int16: int,
            numpy.int32: int,
            numpy.int64: int,
            numpy.float16: float,
            numpy.float32: float,
            numpy.float64: float,
            numpy.integer: int,
            numpy.floating: float,
            numpy.bool_: bool,
            numpy.complexfloating: complex,
            numpy.str_: str,
            numpy.bytes_: bytes,
            # numpy.unicode_: str,
        }
        for numpy_type, builtin_type in type_mapping.items():
            if isinstance(item, numpy_type):
                item = builtin_type(item)
                break

        if isinstance(item, TensorConfig):
            return str(item)
        elif isinstance(item, paddle.base.core.DataType):
            return "Dtype(" + str(item)[7:] + ")"
        elif isinstance(item, paddle.base.core.VarDesc.VarType):
            return "VarType(" + str(item)[7:] + ")"
        elif isinstance(item, list):
            result = "list["
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + "]"
            return result
        elif isinstance(item, tuple):
            result = "tuple("
            for sub_item in item:
                tmp = self.dump_item_str(sub_item)
                if tmp == "":
                    return ""
                result = result + tmp + ","
            result = result + ")"
            return result
        elif isinstance(item, slice):
            return "slice(" + str(item.start) + "," + str(item.stop) + "," + str(item.step) + ")"
        elif isinstance(item, complex):
            return (
                "complex("
                + self.dump_item_str(item.real)
                + ","
                + self.dump_item_str(item.imag)
                + ")"
            )
        elif item is None:
            return "None"
        elif isinstance(item, (paddle.base.Variable, paddle.base.libpaddle.pir.Value)):
            return ""
        elif item == math.inf:
            return "math.inf"
        elif item == -math.inf:
            return "-math.inf"
        elif item == math.nan:
            return "math.nan"
        elif item == -math.nan:
            return "-math.nan"
        elif isinstance(item, (bool, int, float)):
            return str(item)
        elif isinstance(item, str):
            return '"' + item + '"'
        elif isinstance(item, type):
            return "type(" + str(item)[str(item).index("'") + 1 : str(item).rindex("'")] + ")"
        else:
            return str(item)

    def __str__(self):
        result = self.api_name + "("
        for arg in self.args:
            result += self.dump_item_str(arg) + ", "
        for key, value in self.kwargs.items():
            result += key + "=" + self.dump_item_str(value) + ", "
        result += ")"
        return result

    def __repr__(self):
        return self.__str__()

    # def get_tocken(self, config, offset):
    #     def is_int(tocken):
    #         try:
    #             int(tocken)
    #             return True
    #         except Exception as err:
    #             return False
    #     pattern = r'\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b'
    #     match = re.search(pattern, config[offset:])
    #     if match:
    #         if is_int(match.group()) and config[offset + match.start() + len(match.group())] == ".":
    #             return match.group()+".", offset + match.start() + len(match.group()) + 1
    #         return match.group(), offset + match.start() + len(match.group())
    #     return None, None

    def get_tocken(self, config, offset):
        def is_int(token):
            try:
                int(token)
                return True
            except Exception:
                return False

        # Modified pattern to handle decimal numbers starting with dot
        pattern = r"\b[A-Za-z0-9._+-]+\b|-[A-Za-z0-9._+-]+\b|\.[0-9]+"
        match = re.search(pattern, config[offset:])
        if match:
            token = match.group()
            # Handle the case where token starts with dot followed by digits
            if token.startswith(".") and token[1:].isdigit():
                return token, offset + match.start() + len(token)

            if (
                is_int(token)
                and offset + match.start() + len(token) < len(config)
                and config[offset + match.start() + len(token)] == "."
            ):
                return token + ".", offset + match.start() + len(token) + 1
            return token, offset + match.start() + len(token)
        return None, None

    def get_api(self, config):
        return config[0 : config.index("(")], len(config[0 : config.index("(")])

    def get_tensor(self, config, offset):
        config = config[offset:]
        tensor_str = config[config.index("TensorConfig") : config.index(")") + 1]
        return eval(tensor_str), offset + len(tensor_str)

    def get_dtype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.pir.core.convert_np_dtype_to_dtype_(tocken), offset

    def get_vartype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.base.framework.convert_np_dtype_to_proto_type(tocken), offset

    def get_list(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "[":
                tmp = tmp + 1
            if config[i] == "]":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break

        list_str = config[offset : last_index + 1]
        if "TensorConfig" not in list_str:
            list_str = list_str.replace(",", " ")

        offset = 1
        while True:
            tocken, offset = self.get_tocken(list_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, list_str, offset)

            if offset is None:
                break

            result.append(value)

        return result, last_index + 1

    def get_tuple(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "(":
                tmp = tmp + 1
            if config[i] == ")":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break

        tuple_str = config[offset : last_index + 1]

        tuple_str = tuple_str.replace(",", " , ")

        offset = 1
        while True:
            tocken, offset = self.get_tocken(tuple_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, tuple_str, offset)

            if offset is None:
                break

            result.append(value)

        return tuple(result), last_index + 1

    def get_slice(self, config, offset):
        config = config[offset:]
        slice_str = config[config.index("(") : config.index(")") + 1]
        return eval("slice" + slice_str), offset + len(slice_str)

    def get_complex(self, config, offset):
        config = config[offset:]
        complex_str = config[config.index("(") : config.index(")") + 1]
        if "nan" in complex_str and complex_str[complex_str.index("nan") - 1] != ".":
            complex_str = complex_str.replace("nan", "float('nan')")
        return eval("complex" + complex_str), offset + len(complex_str)

    def get_numpy_type(self, config, offset):
        config = config[offset:]
        numpy_type_str = config[config.index("(") + 1 : config.index(")")]
        if numpy_type_str == "numpy.bool":
            return numpy.bool_, offset + len(numpy_type_str) + 2
        return eval(numpy_type_str), offset + len(numpy_type_str) + 2

    def get_one_arg(self, tocken, config, offset):
        if tocken == "TensorConfig":
            value, offset = self.get_tensor(config, offset - len(tocken))
        elif tocken == "Dtype":
            value, offset = self.get_dtype(config, offset)
        elif tocken == "VarType":
            value, offset = self.get_vartype(config, offset)
        elif tocken == "list":
            value, offset = self.get_list(config, offset)
        elif tocken == "tuple":
            value, offset = self.get_tuple(config, offset)
        elif tocken == "slice":
            value, offset = self.get_slice(config, offset)
        elif tocken == "complex":
            value, offset = self.get_complex(config, offset)
        elif tocken == "type":
            value, offset = self.get_numpy_type(config, offset)
        elif tocken == "nan":
            value = float("nan")
        elif tocken is not None and config[offset - len(tocken) - 1] == '"':
            # fix tocken is not correct in str with spaces
            next_quote_idx = config.index('"', offset)
            value = config[offset - len(tocken) : next_quote_idx]
            offset = next_quote_idx
        elif tocken is None:
            return None, None
        else:
            if tocken[0] == ".":
                tocken = "0" + tocken
            value = eval(tocken)
        return value, offset


def analyse_configs(config_path):
    with open(config_path) as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        # print(config)
        api_config = APIConfig(config)
        api_configs.append(api_config)
    return api_configs
