from __future__ import annotations

import copy
import math

import numpy
import paddle
from config_analyzer import TensorConfig, analyse_configs
from tqdm import tqdm


def is_0_size_tensor(tensor_config):
    return any(i == 0 for i in tensor_config.shape)


def is_0D_tensor(tensor_config):
    return len(tensor_config.shape) == 0


def tensor_numel(tensor_config):
    numel = 1
    for i in tensor_config.shape:
        numel = numel * i
    return numel


def get_tensor_configs(api_config):
    tensor_configs = []
    for arg_config in api_config.args:
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])

    for _key, arg_config in api_config.kwargs.items():
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
    return tensor_configs


def to_0_size_config(api_config):
    if api_config.api_name in [
        "paddle.Tensor.__getitem__",
        "paddle.Tensor.__setitem__",
    ]:
        return []
    if api_config.api_name not in apis_map:
        apis_map[api_config.api_name] = {}

    key = config_key(api_config)

    if key not in apis_map[api_config.api_name]:
        apis_map[api_config.api_name][key] = 1
    else:
        apis_map[api_config.api_name][key] += 1

    # if apis_map[api_config.api_name][key] > 2:
    #     return []

    result = []
    tensor_configs = get_tensor_configs(api_config)

    if len(tensor_configs) == 0:
        return []

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config) or is_0D_tensor(tensor_config):
            return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    for i in range(len(tensor_configs)):
        for j in range(len(tensor_configs[i].shape)):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            tmp_tensor_configs[i].shape[j] = 0
            result.append(str(tmp_api_config))

    if shape_equal:
        for j in range(shape_len):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            for i in range(len(tensor_configs)):
                tmp_tensor_configs[i].shape[j] = 0
            result.append(str(tmp_api_config))
    return result


apis_map = {}


def dump_item_str(item):
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
        return "Tensor(" + str(len(item.shape)) + ")"
    elif isinstance(item, paddle.base.core.DataType):
        return "Dtype(" + str(item)[7:] + ")"
    elif isinstance(item, paddle.base.core.VarDesc.VarType):
        return "VarType(" + str(item)[7:] + ")"
    elif isinstance(item, list):
        result = "list["
        for sub_item in item:
            tmp = dump_item_str(sub_item)
            if tmp == "":
                return ""
            result = result + tmp + ","
        result = result + "]"
        return result
    elif isinstance(item, tuple):
        result = "tuple("
        for sub_item in item:
            tmp = dump_item_str(sub_item)
            if tmp == "":
                return ""
            result = result + tmp + ","
        result = result + ")"
        return result
    elif isinstance(item, slice):
        return "slice(" + str(item.start) + "," + str(item.stop) + "," + str(item.step) + ")"
    elif isinstance(item, complex):
        return "complex(" + dump_item_str(item.real) + "," + dump_item_str(item.imag) + ")"
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


def config_key(api_config):
    result = ""
    for arg in api_config.args:
        result = result + dump_item_str(arg) + ", "

    for key, value in api_config.kwargs.items():
        result = result + key + "=" + dump_item_str(value) + ", "

    return result


count_map = {
    "paddle.add": 1,
    "paddle.cast": 1,
    "paddle.Tensor.transpose": 1,
    "paddle.Tensor.tile": 1,
    "paddle.broadcast_to": 1,
    "paddle.Tensor.__mul__": 1,
    "paddle.Tensor.__rmul__": 1,
    "paddle.Tensor.reshape": 1,
    "paddle.Tensor.astype": 1,
    "paddle.nn.functional.conv2d": 1,
    "paddle.vision.ops.deform_conv2d": 1,
    "paddle.nn.functional.pad": 1,
    "paddle.split": 1,
    "paddle.nn.functional.conv2d_transpose": 1,
    "paddle.einsum": 1,
    "paddle.Tensor.__add__": 1,
    "paddle.Tensor.__truediv__": 1,
    "paddle.tensordot": 1,
    "paddle.Tensor.expand": 1,
    "paddle.nn.functional.interpolate": 1,
    "paddle.nn.functional.linear": 1,
    "paddle.multiply": 1,
    "paddle.Tensor.__eq__": 1,
    "paddle.nn.functional.batch_norm": 1,
    "paddle.Tensor.cast": 1,
    "paddle.Tensor.__sub__": 1,
    "paddle.nn.functional.conv3d_transpose": 1,
    "paddle.nn.functional.conv1d": 1,
    "paddle.matmul": 1,
    "paddle.slice": 1,
    "paddle.reshape": 1,
    "paddle.concat": 1,
    "paddle.stack": 1,
    "paddle.broadcast_tensors": 2,
    "paddle.Tensor.sum": 2,
    "paddle.Tensor.unsqueeze": 3,
    "paddle.nn.functional.pairwise_distance": 3,
    "paddle.Tensor.squeeze": 3,
    "paddle.assign": 3,
    "paddle.Tensor.flatten": 3,
    "paddle.allclose": 3,
    "paddle.gather": 3,
    "paddle.Tensor.__gt__": 3,
    "paddle.nn.functional.l1_loss": 3,
    "paddle.nn.functional.max_unpool2d": 3,
    "paddle.nn.functional.mse_loss": 3,
    "paddle.nn.functional.conv3d": 3,
    "paddle.nn.functional.max_pool2d": 3,
    "paddle.nn.functional.softmax": 3,
    "paddle.Tensor.__lt__": 3,
    "paddle.nn.functional.conv1d_transpose": 3,
    "paddle.nn.functional.cross_entropy": 3,
    "paddle.where": 3,
    "paddle.minimum": 4,
    "paddle.add_n": 4,
    "paddle.nn.functional.nll_loss": 4,
    "paddle.nn.functional.softmax_with_cross_entropy": 4,
    "paddle.nanmedian": 4,
    "paddle.nn.functional.hinge_embedding_loss": 4,
    "paddle.Tensor.__mod__": 4,
    "paddle.nn.functional.adaptive_avg_pool3d": 4,
    "paddle.put_along_axis": 4,
    "paddle.nn.functional.binary_cross_entropy": 4,
    "paddle.nn.functional.soft_margin_loss": 4,
    "paddle.nn.functional.margin_ranking_loss": 4,
    "paddle.Tensor.__le__": 4,
    "paddle.not_equal": 4,
    "paddle.Tensor.pow": 4,
    "paddle.pow": 4,
    "paddle.Tensor.__ge__": 4,
    "paddle.scatter": 4,
    "paddle.clip": 5,
    "paddle.vstack": 5,
    "paddle.Tensor.split": 5,
    "paddle.max": 5,
    "paddle.nn.functional.binary_cross_entropy_with_logits": 5,
    "paddle.gather_nd": 5,
    "paddle.nn.functional.max_pool3d": 5,
    "paddle.divide": 5,
    "paddle.topk": 5,
    "paddle.vision.ops.roi_align": 5,
    "paddle.index_select": 5,
    "paddle.subtract": 5,
    "paddle.index_put": 5,
    "paddle.expand": 5,
    "paddle.nn.functional.adaptive_avg_pool2d": 5,
    "paddle.Tensor.__and__": 5,
    "paddle.Tensor.__radd__": 5,
    "paddle.Tensor.__ne__": 5,
    "paddle.nn.functional.avg_pool2d": 5,
    "paddle.median": 5,
    "paddle.take_along_axis": 5,
    "paddle.nn.functional.grid_sample": 5,
    "paddle.sum": 5,
    "paddle.transpose": 5,
    "paddle.nn.functional.cosine_similarity": 6,
    "paddle.lerp": 6,
    "paddle.complex": 6,
    "paddle.Tensor.masked_fill": 6,
    "paddle.bitwise_xor": 6,
    "paddle.bitwise_or": 6,
    "paddle.bitwise_and": 6,
    "paddle.nn.functional.max_unpool3d": 6,
    "paddle.scale": 6,
    "paddle.equal_all": 6,
    "paddle.Tensor.max": 6,
    "paddle.geometric.send_ue_recv": 6,
    "paddle.Tensor.__rsub__": 6,
    "paddle.nn.functional.avg_pool3d": 6,
    "paddle.Tensor.__pow__": 6,
    "paddle.nn.functional.gelu": 6,
    "paddle.Tensor.__matmul__": 6,
    "paddle.remainder": 6,
    "paddle.nn.functional.smooth_l1_loss": 6,
    "paddle.greater_equal": 6,
    "paddle.mean": 6,
    "paddle.linalg.cond": 7,
    "paddle.bitwise_left_shift": 7,
    "paddle.nn.functional.embedding": 7,
    "paddle.mm": 7,
    "paddle.expand_as": 7,
    "paddle.Tensor.__or__": 7,
    "paddle.maximum": 7,
    "paddle.full_like": 7,
    "paddle.Tensor.mean": 7,
    "paddle.masked_fill": 8,
    "paddle.atleast_1d": 8,
    "paddle.atleast_2d": 8,
    "paddle.logical_and": 8,
    "paddle.diff": 8,
    "paddle.atleast_3d": 8,
    "paddle.chunk": 9,
    "paddle.nn.functional.relu": 9,
    "paddle.Tensor.matmul": 9,
    "paddle.Tensor.expand_as": 10,
    "paddle.atan2": 10,
    "paddle.hstack": 10,
    "paddle.column_stack": 10,
    "paddle.row_stack": 10,
    "paddle.addmm": 10,
    "paddle.nn.functional.layer_norm": 10,
    "paddle.dstack": 10,
    "paddle.Tensor.scale": 10,
    "paddle.Tensor.__xor__": 10,
    "paddle.Tensor.__floordiv__": 10,
    "paddle.nn.functional.leaky_relu": 10,
    "paddle.argsort": 11,
    "paddle.argmax": 11,
    "paddle.dist": 11,
    "paddle.masked_select": 11,
    "paddle.Tensor.unbind": 11,
    "paddle.abs": 11,
    "paddle.nn.functional.sigmoid": 11,
    "paddle.squeeze": 11,
    "paddle.linalg.solve": 12,
    "paddle.bitwise_right_shift": 12,
    "paddle.Tensor.masked_select": 12,
    "paddle.ones_like": 12,
    "paddle.zeros_like": 12,
    "paddle.nansum": 13,
    "paddle.cross": 13,
    "paddle.linalg.vector_norm": 13,
    "paddle.var": 13,
    "paddle.Tensor.gather_nd": 13,
    "paddle.incubate.nn.functional.swiglu": 13,
    "paddle.flatten": 13,
    "paddle.nn.functional.unfold": 14,
    "paddle.diagonal": 14,
    "paddle.tensor_split": 14,
    "paddle.logsumexp": 14,
    "paddle.Tensor.min": 14,
    "paddle.Tensor.__lshift__": 14,
    "paddle.Tensor.__rshift__": 14,
    "paddle.shape": 14,
    "paddle.floor_divide": 14,
    "paddle.roll": 14,
    "paddle.nn.functional.hardswish": 15,
    "paddle.sqrt": 15,
    "paddle.nn.utils.vector_to_parameters": 15,
    "paddle.unsqueeze": 15,
    "paddle.Tensor.chunk": 15,
    "paddle.nn.functional.softplus": 16,
    "paddle.any": 16,
    "paddle.linalg.matrix_power": 16,
    "paddle.bmm": 16,
    "paddle.kron": 16,
    "paddle.nn.functional.triplet_margin_with_distance_loss": 16,
    "paddle.heaviside": 17,
    "paddle.Tensor.lerp": 17,
    "paddle.inner": 17,
    "paddle.nn.functional.adaptive_max_pool3d": 17,
    "paddle.nn.functional.glu": 18,
    "paddle.amax": 18,
    "paddle.amin": 18,
    "paddle.as_complex": 18,
    "paddle.nn.functional.adaptive_max_pool2d": 18,
    "paddle.all": 18,
    "paddle.bitwise_not": 18,
    "paddle.Tensor.inner": 19,
    "paddle.gcd": 19,
    "paddle.std": 19,
    "paddle.fmin": 19,
    "paddle.nn.functional.relu6": 19,
    "paddle.nanmean": 20,
    "paddle.argmin": 20,
    "paddle.fmax": 20,
    "paddle.digamma": 20,
    "paddle.prod": 20,
    "paddle.linalg.lu": 20,
    "paddle.slice_scatter": 20,
    "paddle.less_than": 20,
    "paddle.cumprod": 20,
    "paddle.nn.functional.poisson_nll_loss": 20,
    "paddle.searchsorted": 20,
    "paddle.sort": 20,
    "paddle.Tensor.any": 20,
    "paddle.Tensor.detach": 20,
    "paddle.Tensor.fill_diagonal_tensor": 20,
    "paddle.less_equal": 20,
    "paddle.meshgrid": 20,
    "paddle.nn.functional.max_pool1d": 20,
    "paddle.mod": 20,
    "paddle.greater_than": 20,
    "paddle.linalg.lu_unpack": 20,
    "paddle.tanh": 20,
    "paddle.nn.functional.silu": 20,
    "paddle.clone": 21,
    "paddle.logical_not": 22,
    "paddle.ceil": 22,
    "paddle.isnan": 22,
    "paddle.Tensor.clone": 22,
    "paddle.fft.ihfftn": 22,
    "paddle.nn.functional.swish": 22,
    "paddle.log1p": 23,
    "paddle.nn.functional.hardsigmoid": 23,
    "paddle.nn.functional.hardtanh": 23,
    "paddle.Tensor.sqrt": 24,
    "paddle.Tensor.argmax": 24,
    "paddle.Tensor.clip": 24,
    "paddle.Tensor.sin": 25,
    "paddle.lgamma": 25,
    "paddle.isinf": 25,
    "paddle.Tensor.gcd": 25,
    "paddle.log": 25,
    "paddle.linalg.qr": 25,
    "paddle.Tensor.diff": 26,
    "paddle.nn.functional.elu": 26,
    "paddle.numel": 26,
    "paddle.Tensor.prod": 26,
    "paddle.Tensor.__neg__": 26,
    "paddle.triu": 26,
    "paddle.Tensor.bmm": 27,
    "paddle.moveaxis": 27,
    "paddle.unflatten": 27,
    "paddle.Tensor.fill_": 27,
    "paddle.nn.functional.group_norm": 27,
    "paddle.square": 27,
    "paddle.linalg.pinv": 28,
    "paddle.Tensor.__len__": 28,
    "paddle.flip": 29,
    "paddle.repeat_interleave": 29,
    "paddle.nn.functional.sigmoid_focal_loss": 30,
    "paddle.geometric.send_uv": 30,
    "paddle.nn.functional.pixel_shuffle": 30,
    "paddle.nn.functional.instance_norm": 30,
    "paddle.unstack": 30,
    "paddle.Tensor.__rtruediv__": 30,
    "paddle.nn.functional.log_softmax": 30,
    "paddle.logaddexp": 30,
    "paddle.scatter_nd_add": 30,
    "paddle.rot90": 31,
    "paddle.Tensor.rot90": 31,
    "paddle.nn.functional.label_smooth": 31,
    "paddle.nonzero": 31,
    "paddle.nanquantile": 32,
    "paddle.rsqrt": 32,
    "paddle.expm1": 32,
    "paddle.tan": 32,
    "paddle.Tensor.dim": 32,
    "paddle.renorm": 33,
    "paddle.nextafter": 33,
    "paddle.mv": 34,
    "paddle.Tensor.zero_": 34,
    "paddle.neg": 34,
    "paddle.Tensor.all": 34,
    "paddle.tril": 34,
    "paddle.nn.functional.prelu": 35,
    "paddle.Tensor.add": 35,
    "paddle.unbind": 36,
    "paddle.signbit": 37,
    "paddle.index_sample": 37,
    "paddle.bincount": 38,
    "paddle.Tensor.flip": 38,
    "paddle.nn.functional.celu": 38,
    "paddle.linalg.matrix_norm": 38,
    "paddle.Tensor.moveaxis": 38,
    "paddle.Tensor.cos": 38,
    "paddle.Tensor.std": 40,
    "paddle.logical_xor": 40,
    "paddle.Tensor.amin": 40,
    "paddle.Tensor.amax": 40,
    "paddle.nn.functional.temporal_shift": 40,
    "paddle.isfinite": 42,
    "paddle.hsplit": 42,
    "paddle.nn.functional.log_sigmoid": 42,
    "paddle.nn.functional.rrelu": 42,
    "paddle.nn.functional.adaptive_max_pool1d": 42,
    "paddle.fft.ihfft": 43,
    "paddle.isclose": 43,
    "paddle.log2": 43,
    "paddle.fft.ihfft2": 43,
    "paddle.erf": 43,
    "paddle.fft.ifftn": 45,
    "paddle.Tensor.is_complex": 46,
    "paddle.nn.functional.fold": 48,
    "paddle.Tensor.tolist": 48,
    "paddle.select_scatter": 48,
    "paddle.trunc": 48,
    "paddle.nan_to_num": 48,
    "paddle.Tensor.quantile": 48,
    "paddle.Tensor.subtract": 50,
    "paddle.diag_embed": 50,
    "paddle.vsplit": 50,
    "paddle.Tensor.fill_diagonal_": 50,
    "paddle.cumsum": 51,
    "paddle.Tensor.var": 53,
    "paddle.sinh": 53,
    "paddle.as_strided": 53,
    "paddle.diagflat": 53,
    "paddle.Tensor.nansum": 53,
    "paddle.Tensor.abs": 53,
    "paddle.outer": 54,
    "paddle.linalg.norm": 55,
    "paddle.isneginf": 56,
    "paddle.isposinf": 56,
    "paddle.rad2deg": 56,
    "paddle.cdist": 56,
    "paddle.quantile": 57,
    "paddle.min": 58,
    "paddle.Tensor.isclose": 59,
    "paddle.atan": 59,
    "paddle.Tensor.logical_and": 59,
    "paddle.erfinv": 59,
    "paddle.Tensor.norm": 60,
    "paddle.nn.functional.thresholded_relu": 63,
    "paddle.linalg.det": 63,
    "paddle.pdist": 63,
    "paddle.nn.functional.avg_pool1d": 64,
    "paddle.masked_scatter": 65,
    "paddle.linalg.triangular_solve": 65,
    "paddle.logical_or": 66,
    "paddle.unique": 67,
    "paddle.cartesian_prod": 67,
    "paddle.Tensor.signbit": 67,
    "paddle.hypot": 67,
    "paddle.asin": 67,
    "paddle.nn.functional.max_unpool1d": 70,
    "paddle.nn.functional.softsign": 71,
    "paddle.polygamma": 71,
    "paddle.Tensor.tanh": 71,
    "paddle.nn.functional.multi_label_soft_margin_loss": 73,
    "paddle.linalg.inv": 77,
    "paddle.log10": 77,
    "paddle.acos": 77,
    "paddle.dot": 77,
    "paddle.index_add": 78,
    "paddle.fft.rfftn": 81,
    "paddle.fft.ifftshift": 83,
    "paddle.dsplit": 83,
    "paddle.fft.fftshift": 83,
    "paddle.Tensor.sign": 83,
    "paddle.isin": 85,
    "paddle.multiplex": 89,
    "paddle.nn.functional.sequence_mask": 89,
    "paddle.Tensor.lu": 91,
    "paddle.Tensor.sigmoid": 91,
    "paddle.Tensor.logical_or": 91,
    "paddle.floor": 93,
    "paddle.vecdot": 98,
    "paddle.Tensor.index_select": 98,
    "paddle.t": 100,
    "paddle.nn.functional.tanhshrink": 100,
    "paddle.nn.functional.maxout": 105,
    "paddle.histogramdd": 105,
    "paddle.bucketize": 111,
    "paddle.Tensor.set_": 111,
    "paddle.Tensor.mode": 111,
    "paddle.asinh": 111,
    "paddle.nn.functional.hardshrink": 111,
    "paddle.atanh": 111,
    "paddle.Tensor.remainder": 111,
    "paddle.is_complex": 111,
    "paddle.tile": 113,
    "paddle.conj": 125,
    "paddle.Tensor.equal": 125,
    "paddle.logit": 125,
    "paddle.nn.functional.lp_pool2d": 125,
    "paddle.full": 137,
    "paddle.Tensor.tril": 143,
    "paddle.bitwise_invert": 143,
    "paddle.frac": 143,
    "paddle.Tensor.square": 143,
    "paddle.Tensor.trunc": 143,
    "paddle.nn.functional.square_error_cost": 144,
    "paddle.linalg.cov": 146,
    "paddle.nn.functional.bilinear": 156,
    "paddle.gammaincc": 156,
    "paddle.geometric.send_u_recv": 156,
    "paddle.vision.ops.roi_pool": 166,
    "paddle.kthvalue": 166,
    "paddle.sgn": 167,
    "paddle.Tensor.slice_scatter": 167,
    "paddle.isreal": 167,
    "paddle.Tensor.mm": 167,
    "paddle.sinc": 167,
    "paddle.diag": 181,
    "paddle.nn.functional.zeropad2d": 182,
    "paddle.Tensor.__rpow__": 182,
    "paddle.Tensor.isnan": 183,
    "paddle.Tensor.floor": 184,
    "paddle.nn.functional.local_response_norm": 194,
    "paddle.index_fill": 194,
    "paddle.Tensor.repeat_interleave": 196,
    "paddle.deg2rad": 200,
    "paddle.Tensor.rsqrt": 200,
    "paddle.fft.ifft2": 200,
    "paddle.i0": 200,
    "paddle.fft.fft2": 200,
    "paddle.Tensor.dot": 200,
    "paddle.Tensor.round": 200,
    "paddle.Tensor.log": 200,
    "paddle.reverse": 206,
    "paddle.mode": 206,
    "paddle.Tensor.multiply": 211,
    "paddle.nn.functional.normalize": 211,
    "paddle.gammainc": 220,
    "paddle.count_nonzero": 232,
    "paddle.signal.stft": 234,
    "paddle.scatter_nd": 234,
    "paddle.multigammaln": 250,
    "paddle.Tensor.logit": 250,
    "paddle.combinations": 250,
    "paddle.Tensor.diagonal": 250,
    "paddle.Tensor.logical_not": 255,
    "paddle.nn.functional.log_loss": 257,
    "paddle.nn.functional.selu": 257,
    "paddle.equal": 297,
    "paddle.linalg.multi_dot": 306,
    "paddle.Tensor.exp": 311,
    "paddle.Tensor.item": 320,
    "paddle.Tensor.topk": 333,
    "paddle.acosh": 333,
    "paddle.Tensor.outer": 333,
    "paddle.Tensor.diag_embed": 333,
    "paddle.Tensor.kthvalue": 333,
    "paddle.nn.functional.affine_grid": 342,
    "paddle.incubate.nn.functional.fused_matmul_bias": 350,
    "paddle.nn.functional.multi_margin_loss": 350,
    "paddle.nn.functional.cosine_embedding_loss": 350,
    "paddle.copysign": 380,
    "paddle.strided_slice": 409,
    "paddle.Tensor.inverse": 409,
    "paddle.exp": 409,
    "paddle.sin": 424,
    "paddle.Tensor.cumsum": 431,
    "paddle.Tensor.not_equal": 435,
    "paddle.less": 457,
    "paddle.Tensor.conj": 497,
    "paddle.i1e": 500,
    "paddle.i0e": 500,
    "paddle.fft.ifft": 500,
    "paddle.fft.fft": 500,
    "paddle.i1": 500,
    "paddle.crop": 500,
    "paddle.view": 500,
    "paddle.ldexp": 500,
    "paddle.fft.rfft2": 520,
    "paddle.fft.fftn": 617,
    "paddle.fft.rfft": 617,
    "paddle.stanh": 622,
    "paddle.linalg.slogdet": 622,
    "paddle.nn.functional.pixel_unshuffle": 630,
    "paddle.logcumsumexp": 692,
    "paddle.cos": 741,
    "paddle.sign": 755,
    "paddle.Tensor.erfinv": 788,
    "paddle.nn.functional.channel_shuffle": 788,
    "paddle.geometric.segment_mean": 838,
    "paddle.nn.functional.mish": 870,
    "paddle.Tensor.put_along_axis": 893,
    "paddle.Tensor.lgamma": 893,
    "paddle.round": 893,
    "paddle.positive": 909,
    "paddle.diagonal_scatter": 922,
    "paddle.geometric.segment_sum": 963,
    "paddle.Tensor.broadcast_to": 1000,
    "paddle.trapezoid": 1000,
    "paddle.Tensor.unique": 1000,
    "paddle.Tensor.multigammaln": 1000,
    "paddle.nn.functional.tanh": 1000,
    "paddle.unfold": 1185,
    "paddle.Tensor.atanh": 1185,
    "paddle.geometric.segment_min": 1185,
    "paddle.Tensor.rad2deg": 1185,
    "paddle.geometric.segment_max": 1185,
    "paddle.Tensor.__rrshift__": 1250,
    "paddle.negative": 1250,
    "paddle.Tensor.__rlshift__": 1250,
    "paddle.reciprocal": 1285,
    "paddle.nn.functional.adaptive_avg_pool1d": 1351,
    "paddle.take": 1383,
    "paddle.lcm": 1383,
    "paddle.Tensor.digamma": 1383,
    "paddle.shard_index": 1383,
    "paddle.linalg.corrcoef": 1383,
    "paddle.matrix_transpose": 1389,
    "paddle.nn.functional.softshrink": 1518,
    "paddle.vander": 1655,
    "paddle.Tensor.nonzero": 1788,
    "paddle.cummax": 2000,
    "paddle.nn.functional.adaptive_log_softmax_with_loss": 2000,
    "paddle.cummin": 2000,
    "paddle.Tensor.__rmod__": 2467,
    "paddle.Tensor.equal_all": 2467,
    "paddle.Tensor.log1p": 2467,
    "paddle.is_empty": 2467,
    "paddle.linalg.svdvals": 2467,
    "paddle.Tensor.__nonzero__": 3125,
    "paddle.cosh": 3846,
    "paddle.Tensor.reciprocal": 4086,
    "paddle.zeros": 4086,
    "paddle.increment": 4086,
    "paddle.Tensor.ceil": 4086,
    "paddle.view_as": 4086,
    "paddle.unique_consecutive": 5263,
    "paddle.linalg.cholesky": 5567,
    "paddle.Tensor.gather": 5567,
    "paddle.Tensor.__rxor__": 5567,
    "paddle.Tensor.less": 5567,
    "paddle.Tensor.__rmatmul__": 5567,
    "paddle.histogram_bin_edges": 5567,
    "paddle.Tensor.__ror__": 5567,
    "paddle.polar": 5567,
    "paddle.logspace": 8000,
    "paddle.nn.functional.npair_loss": 8000,
    "paddle.tolist": 8000,
    "paddle.reduce_as": 8000,
    "paddle.nn.functional.lp_pool1d": 16667,
    "paddle.ones": 22200,
    "paddle.eye": 22200,
    "paddle.Tensor.__div__": 22200,
    "paddle.Tensor.__abs__": 22200,
    "paddle.Tensor.divide": 22200,
    "paddle.incubate.segment_sum": 22200,
    "paddle.linalg.matrix_transpose": 22200,
    "paddle.incubate.segment_max": 22200,
    "paddle.incubate.segment_min": 22200,
    "paddle.incubate.segment_mean": 22200,
    "paddle.trace": 25000,
    "paddle.Tensor.log10": 50000,
    "paddle.Tensor.cumprod": 50000,
    "paddle.histogram": 50000,
    "paddle.linalg.eigvals": 200000,
    "paddle.linalg.eigvalsh": 200000,
    "paddle.Tensor.neg": 200000,
}


def to_big_tensor_config(api_config):
    if api_config.api_name not in apis_map:
        apis_map[api_config.api_name] = {}

    key = config_key(api_config)

    if key not in apis_map[api_config.api_name]:
        apis_map[api_config.api_name][key] = 1
    else:
        apis_map[api_config.api_name][key] += 1

    if api_config.api_name in count_map:
        if apis_map[api_config.api_name][key] > count_map[api_config.api_name]:
            return []
    else:
        if apis_map[api_config.api_name][key] > 5:
            return []

    tensor_configs = get_tensor_configs(api_config)

    result = []

    if len(tensor_configs) == 0:
        return []

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config) or is_0D_tensor(tensor_config):
            return []
        if tensor_config.dtype in ["complex64", "complex128"]:
            return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    for i in range(len(tensor_configs)):
        for j in range(len(tensor_configs[i].shape)):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            if tmp_tensor_configs[i].dtype in [
                "float8",
                "float16",
                "bfloat16",
                "int16",
                "uint16",
                "int8",
                "uint8",
            ]:
                base_size = 4294967296
            elif tmp_tensor_configs[i].dtype in ["float64"]:
                base_size = 4294967296
                for k in range(len(tmp_tensor_configs)):
                    if tmp_tensor_configs[k].dtype in ["float64"]:
                        tmp_tensor_configs[k].dtype = "float16"
            else:
                base_size = 2281701378
            old_dim = tmp_tensor_configs[i].shape[j]
            new_dim = (
                int(
                    base_size
                    / (tensor_numel(tmp_tensor_configs[i]) / tmp_tensor_configs[i].shape[j])
                )
                + 1
            )
            tmp_tensor_configs[i].shape[j] = new_dim
            config_str = str(tmp_api_config)
            if len(config_str) < 1000:
                result.append(config_str)

            valid = True
            tmp_tensor_configs[i].shape[j] = old_dim
            for m in range(len(tensor_configs)):
                is_first = True
                for n in range(len(tensor_configs[m].shape)):
                    if tmp_tensor_configs[m].shape[n] == old_dim:
                        tmp_tensor_configs[m].shape[n] = new_dim
                        if is_first:
                            is_first = False
                        else:
                            valid = False
            if valid == False:
                continue
            config_str2 = str(tmp_api_config)
            if len(config_str2) < 1000:
                result.append(config_str2)

    if shape_equal:
        for j in range(shape_len):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            for i in range(len(tensor_configs)):
                if tmp_tensor_configs[i].dtype in [
                    "float8",
                    "float16",
                    "bfloat16",
                    "int16",
                    "uint16",
                    "int8",
                    "uint8",
                ]:
                    base_size = 4294967296
                elif tmp_tensor_configs[i].dtype in ["float64"]:
                    base_size = 4294967296
                    tmp_tensor_configs[i].dtype = "float16"
                else:
                    base_size = 2281701378
                tmp_tensor_configs[i].shape[j] = (
                    int(
                        base_size
                        / (tensor_numel(tmp_tensor_configs[0]) / tmp_tensor_configs[0].shape[j])
                    )
                    + 1
                )
            config_str = str(tmp_api_config)
            if len(config_str) < 1000:
                result.append(config_str)
    return result


if __name__ == "__main__":
    config_big_tensor = set()
    api_configs = analyse_configs(
        "/host_home/wanghuan29/APItest/PaddleAPITest/tester/api_config/5_accuracy/17.txt"
    )
    with open(
        "/host_home/wanghuan29/APItest/PaddleAPITest/tester/api_config/8_big_tensor/big_tensor_17.txt",
        "w",
    ) as f:
        for api_config in tqdm(api_configs):
            # print(api_config.config)
            # config_big_tensor = config_big_tensor.union(set(to_big_tensor_config(api_config)))
            try:
                configs = to_big_tensor_config(api_config)
                for a in configs:
                    f.write(str(a) + "\n")
            except Exception:
                continue
    f.close()
