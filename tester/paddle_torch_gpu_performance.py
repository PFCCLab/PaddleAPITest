from __future__ import annotations

import time

import paddle
import torch
from func_timeout import func_set_timeout

from .api_config.config_analyzer import TensorConfig
from .api_config.log_writer import write_to_log
from .base import APITestBase
from .paddle_to_torch import get_converter


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


def total_numel(api_config):
    tensor_configs = get_tensor_configs(api_config)
    numel = 0
    for tensor_config in tensor_configs:
        numel = numel + tensor_numel(tensor_config)
    return numel


def print_performance(
    is_finish,
    api,
    config,
    numel,
    test_loop,
    paddle_forward,
    torch_forward,
    paddle_forward_sync,
    torch_forward_sync,
    paddle_backward,
    torch_backward,
    paddle_backward_sync,
    torch_backward_sync,
    is_torch_combined,
):
    if (
        paddle_forward is not None
        and torch_forward is not None
        and paddle_backward is not None
        and torch_backward is not None
    ):
        print(
            "[Prof]",
            api,
            "\t",
            config,
            "\t",
            numel,
            "\t",
            test_loop,
            "\t",
            paddle_forward,
            "\t",
            torch_forward,
            "\t",
            paddle_forward_sync,
            "\t",
            torch_forward_sync,
            "\t",
            paddle_backward,
            "\t",
            torch_backward,
            "\t",
            paddle_backward_sync,
            "\t",
            torch_backward_sync,
            "\t",
            is_torch_combined,
            flush=True,
        )
    elif paddle_forward is not None and torch_forward is not None:
        print(
            "[Prof]",
            api,
            "\t",
            config,
            "\t",
            numel,
            "\t",
            test_loop,
            "\t",
            paddle_forward,
            "\t",
            torch_forward,
            "\t",
            paddle_forward_sync,
            "\t",
            torch_forward_sync,
            "\t",
            "None",
            "\t",
            "None",
            "\t",
            "None",
            "\t",
            "None",
            "\t",
            is_torch_combined,
            flush=True,
        )


api_loop = {
    "paddle.greater_than": 53046,
    "paddle.cosh": 33925,
    "paddle.log10": 33848,
    "paddle.histogram_bin_edges": 98929,
    "paddle.bucketize": 957494,
    "paddle.triu": 30544,
    "paddle.max": 65890,
    "paddle.abs": 33746,
    "paddle.Tensor.unbind": 85224,
    "paddle.prod": 68328,
    "paddle.Tensor.__truediv__": 33318,
    "paddle.fmin": 33691,
    "paddle.nn.functional.max_unpool3d": 349895,
    "paddle.tril": 32473,
    "paddle.nn.functional.conv2d_transpose": 19463,
    "paddle.linalg.lu": 22178,
    "paddle.Tensor.squeeze": 2295230,
    "paddle.nn.functional.grid_sample": 97873,
    "paddle.complex": 20670,
    "paddle.nn.functional.linear": 62826,
    "paddle.roll": 18327,
    "paddle.full_like": 74569,
    "paddle.rank": 247008,
    "paddle.Tensor.any": 20191,
    "paddle.nn.functional.softmax": 33703,
    "paddle.searchsorted": 998667,
    "paddle.digamma": 10410,
    "paddle.Tensor.all": 21540,
    "paddle.nn.functional.avg_pool3d": 56552,
    "paddle.flatten": 1742182,
    "paddle.row_stack": 32497,
    "paddle.nn.functional.max_unpool2d": 470794,
    "paddle.reverse": 19922,
    "paddle.vsplit": 588277,
    "paddle.incubate.segment_sum": 16032,
    "paddle.count_nonzero": 16875,
    "paddle.Tensor.scale": 33533,
    "paddle.nanmean": 5042,
    "paddle.ones_like": 74688,
    "paddle.logical_not": 12728,
    "paddle.geometric.send_uv": 104455,
    "paddle.Tensor.__xor__": 85302,
    "paddle.Tensor.diff": 12395,
    "paddle.multigammaln": 4159,
    "paddle.scale": 33791,
    "paddle.bmm": 12079,
    "paddle.nn.functional.prelu": 33188,
    "paddle.greater_equal": 53247,
    "paddle.fft.rfftn": 6899,
    "paddle.nn.functional.dropout": 66830,
    "paddle.nn.functional.adaptive_avg_pool1d": 52811,
    "paddle.Tensor.log": 33839,
    "paddle.geometric.send_ue_recv": 40736,
    "paddle.i0e": 25383,
    "paddle.nn.functional.binary_cross_entropy": 9554,
    "paddle.isin": 3671,
    "paddle.unstack": 30584,
    "paddle.Tensor.rank": 242345,
    "paddle.nn.functional.softshrink": 33823,
    "paddle.pdist": 1686,
    "paddle.sin": 33844,
    "paddle.nn.functional.local_response_norm": 2912,
    "paddle.Tensor.gather_nd": 1730,
    "paddle.Tensor.__or__": 85484,
    "paddle.einsum": 12187,
    "paddle.zeros_like": 74631,
    "paddle.tensordot": 20388,
    "paddle.cumprod": 33059,
    "paddle.less": 53034,
    "paddle.dsplit": 587939,
    "paddle.linalg.matrix_power": 2109,
    "paddle.slice_scatter": 54545,
    "paddle.subtract": 28013,
    "paddle.logit": 33608,
    "paddle.Tensor.equal_all": 612074,
    "paddle.Tensor.index_select": 1057032,
    "paddle.geometric.segment_mean": 28038,
    "paddle.nn.functional.multi_margin_loss": 7942,
    "paddle.bitwise_not": 33833,
    "paddle.log2": 33844,
    "paddle.bitwise_and": 84592,
    "paddle.log": 33842,
    "paddle.Tensor.nansum": 10691,
    "paddle.add_n": 21148,
    "paddle.exp": 33820,
    "paddle.moveaxis": 1394939,
    "paddle.Tensor.__add__": 29759,
    "paddle.Tensor.mm": 15341,
    "paddle.rot90": 19442,
    "paddle.Tensor.multiply": 33843,
    "paddle.where": 20615,
    "paddle.Tensor.__rxor__": 33457,
    "paddle.atleast_3d": 4582436,
    "paddle.negative": 33834,
    "paddle.Tensor.outer": 6606,
    "paddle.divide": 33832,
    "paddle.Tensor.rad2deg": 33553,
    "paddle.i1": 19933,
    "paddle.vstack": 32416,
    "paddle.bitwise_left_shift": 22337,
    "paddle.ceil": 33827,
    "paddle.nn.functional.margin_cross_entropy": 3872,
    "paddle.nn.functional.adaptive_max_pool3d": 6398,
    "paddle.nn.functional.max_pool2d": 50852,
    "paddle.put_along_axis": 77440,
    "paddle.Tensor.cumprod": 30751,
    "paddle.nn.functional.conv2d": 9096,
    "paddle.nn.functional.leaky_relu": 33695,
    "paddle.gather": 6316,
    "paddle.nn.functional.log_sigmoid": 33264,
    "paddle.gather_nd": 4935,
    "paddle.tensor_split": 428549,
    "paddle.atan": 33597,
    "paddle.combinations": 804184,
    "paddle.masked_select": 7185,
    "paddle.Tensor.clip": 33863,
    "paddle.Tensor.exp": 33844,
    "paddle.select_scatter": 315923,
    "paddle.linalg.multi_dot": 60322,
    "paddle.Tensor.__rsub__": 33801,
    "paddle.bitwise_xor": 84955,
    "paddle.Tensor.reciprocal": 33834,
    "paddle.nn.functional.pixel_shuffle": 26616,
    "paddle.dstack": 31252,
    "paddle.Tensor.__rpow__": 17155,
    "paddle.nn.functional.relu": 33682,
    "paddle.Tensor.bmm": 42736,
    "paddle.cast": 33565,
    "paddle.isclose": 27502,
    "paddle.gcd": 1588,
    "paddle.Tensor.unsqueeze": 2409550,
    "paddle.nn.functional.relu6": 33340,
    "paddle.nn.functional.smooth_l1_loss": 12325,
    "paddle.mm": 7310,
    "paddle.take_along_axis": 32853,
    "paddle.isfinite": 42769,
    "paddle.cos": 33844,
    "paddle.linalg.cov": 18486,
    "paddle.Tensor.__pow__": 27052,
    "paddle.Tensor.logit": 30800,
    "paddle.Tensor.prod": 25773,
    "paddle.floor_divide": 23479,
    "paddle.bitwise_right_shift": 22351,
    "paddle.clone": 32432,
    "paddle.amax": 50766,
    "paddle.unflatten": 101360,
    "paddle.sgn": 32402,
    "paddle.fft.ihfft": 133166,
    "paddle.remainder": 33680,
    "paddle.Tensor.fill_diagonal_tensor": 31092,
    "paddle.Tensor.cumsum": 28938,
    "paddle.nn.functional.adaptive_avg_pool2d": 33312,
    "paddle.erfinv": 29882,
    "paddle.column_stack": 32438,
    "paddle.median": 3680,
    "paddle.scatter": 14665,
    "paddle.Tensor.is_complex": 2814024,
    "paddle.unsqueeze": 2405680,
    "paddle.Tensor.pow": 26857,
    "paddle.atleast_2d": 5325424,
    "paddle.nn.functional.cosine_similarity": 9454,
    "paddle.nn.functional.margin_ranking_loss": 7473,
    "paddle.nn.functional.max_unpool1d": 283012,
    "paddle.nn.functional.zeropad2d": 31313,
    "paddle.Tensor.__sub__": 33533,
    "paddle.geometric.segment_min": 28537,
    "paddle.fft.rfft2": 6899,
    "paddle.inner": 62291,
    "paddle.isnan": 42698,
    "paddle.Tensor.sqrt": 33926,
    "paddle.nn.functional.cross_entropy": 8585,
    "paddle.nn.functional.sigmoid_focal_loss": 1450,
    "paddle.nn.functional.mse_loss": 13409,
    "paddle.round": 33846,
    "paddle.transpose": 2997001,
    "paddle.atleast_1d": 6288311,
    "paddle.asinh": 33221,
    "paddle.nn.functional.cosine_embedding_loss": 5778,
    "paddle.rad2deg": 33829,
    "paddle.add": 22147,
    "paddle.multiply": 33806,
    "paddle.mv": 64031,
    "paddle.Tensor.cast": 32559,
    "paddle.nn.functional.pad": 8486,
    "paddle.all": 49878,
    "paddle.Tensor.__neg__": 33840,
    "paddle.Tensor.amax": 65883,
    "paddle.hypot": 10387,
    "paddle.Tensor.moveaxis": 1457722,
    "paddle.isreal": 25942,
    "paddle.Tensor.mean": 60303,
    "paddle.Tensor.sign": 32403,
    "paddle.nn.functional.multi_label_soft_margin_loss": 2957,
    "paddle.cross": 22358,
    "paddle.Tensor.masked_fill": 70418,
    "paddle.Tensor.__rshift__": 22342,
    "paddle.mode": 1169,
    "paddle.Tensor.log1p": 33826,
    "paddle.diagonal_scatter": 31122,
    "paddle.index_put": 28767,
    "paddle.logsumexp": 15717,
    "paddle.Tensor.amin": 65865,
    "paddle.allclose": 9683,
    "paddle.Tensor.__radd__": 33511,
    "paddle.log1p": 33832,
    "paddle.Tensor.set_": 272766,
    "paddle.nn.functional.layer_norm": 33222,
    "paddle.Tensor.__div__": 33824,
    "paddle.diff": 12388,
    "paddle.shard_index": 32350,
    "paddle.incubate.nn.functional.fused_linear": 1922,
    "paddle.Tensor.atanh": 33658,
    "paddle.t": 2515777,
    "paddle.Tensor.tanh": 33854,
    "paddle.Tensor.chunk": 29186,
    "paddle.is_empty": 2953527,
    "paddle.reciprocal": 33850,
    "paddle.polar": 8265,
    "paddle.linalg.inv": 1322,
    "paddle.nn.functional.avg_pool2d": 52538,
    "paddle.argmax": 15617,
    "paddle.fft.fftn": 1679,
    "paddle.not_equal": 46945,
    "paddle.tile": 33767,
    "paddle.trace": 153440,
    "paddle.Tensor.round": 33823,
    "paddle.argmin": 22702,
    "paddle.Tensor.sigmoid": 33917,
    "paddle.nn.functional.triplet_margin_with_distance_loss": 4341,
    "paddle.nonzero": 1360,
    "paddle.nn.functional.log_softmax": 32827,
    "paddle.sign": 32506,
    "paddle.Tensor.__floordiv__": 32836,
    "paddle.nansum": 10275,
    "paddle.Tensor.add": 22210,
    "paddle.linalg.slogdet": 1538,
    "paddle.logaddexp": 4307,
    "paddle.cartesian_prod": 27607,
    "paddle.atan2": 22449,
    "paddle.nn.functional.square_error_cost": 16765,
    "paddle.Tensor.abs": 33826,
    "paddle.expm1": 29864,
    "paddle.Tensor.min": 45506,
    "paddle.lgamma": 25074,
    "paddle.less_equal": 53036,
    "paddle.nn.functional.max_pool1d": 37061,
    "paddle.diagonal": 1998070,
    "paddle.tan": 33898,
    "paddle.Tensor.equal": 56162,
    "paddle.atanh": 33693,
    "paddle.cumsum": 28767,
    "paddle.index_add": 32129,
    "paddle.concat": 31651,
    "paddle.Tensor.__rmul__": 33795,
    "paddle.nn.functional.tanhshrink": 33845,
    "paddle.stack": 5521,
    "paddle.Tensor.__ne__": 21223,
    "paddle.Tensor.__getitem__": 1992164,
    "paddle.mean": 54851,
    "paddle.fft.ihfftn": 4772,
    "paddle.Tensor.item": 533796,
    "paddle.index_fill": 29793,
    "paddle.matmul": 12781,
    "paddle.nn.functional.gelu": 29045,
    "paddle.hstack": 32425,
    "paddle.Tensor.matmul": 9746,
    "paddle.Tensor.topk": 26178,
    "paddle.incubate.nn.functional.fused_dropout_add": 22272,
    "paddle.nextafter": 22191,
    "paddle.shape": 2368860,
    "paddle.Tensor.__gt__": 21267,
    "paddle.trunc": 34308,
    "paddle.nn.functional.conv1d_transpose": 14413,
    "paddle.index_sample": 83783,
    "paddle.logical_and": 84691,
    "paddle.Tensor.not_equal": 72509,
    "paddle.var": 10215,
    "paddle.nn.functional.sigmoid": 33850,
    "paddle.nn.functional.celu": 32447,
    "paddle.Tensor.__eq__": 52174,
    "paddle.Tensor.__and__": 84955,
    "paddle.Tensor.transpose": 66690,
    "paddle.bincount": 10040,
    "paddle.nn.functional.silu": 33834,
    "paddle.Tensor.expand_as": 74165,
    "paddle.Tensor.lu": 64078,
    "paddle.nn.functional.lp_pool2d": 18010,
    "paddle.nn.functional.adaptive_max_pool1d": 48051,
    "paddle.Tensor.sin": 33845,
    "paddle.min": 65799,
    "paddle.index_select": 49358,
    "paddle.copysign": 22302,
    "paddle.bitwise_or": 84965,
    "paddle.expand_as": 72722,
    "paddle.Tensor.take_along_axis": 33018,
    "paddle.topk": 4760,
    "paddle.Tensor.max": 61050,
    "paddle.hsplit": 578883,
    "paddle.Tensor.lgamma": 14030,
    "paddle.nn.functional.pairwise_distance": 8769,
    "paddle.minimum": 3827,
    "paddle.asin": 33858,
    "paddle.geometric.send_u_recv": 27790,
    "paddle.take": 172659,
    "paddle.nn.functional.thresholded_relu": 33826,
    "paddle.nn.functional.pixel_unshuffle": 31500,
    "paddle.crop": 547102,
    "paddle.masked_fill": 26391,
    "paddle.Tensor.std": 9109,
    "paddle.maximum": 33726,
    "paddle.nn.functional.bilinear": 2209,
    "paddle.matrix_transpose": 2360594,
    "paddle.Tensor.fill_diagonal_": 432242,
    "paddle.fft.rfft": 7826,
    "paddle.Tensor.__matmul__": 12494,
    "paddle.empty_like": 694973,
    "paddle.bitwise_invert": 33827,
    "paddle.nn.functional.softmax_with_cross_entropy": 29087,
    "paddle.Tensor.__rlshift__": 33460,
    "paddle.nn.functional.poisson_nll_loss": 6305,
    "paddle.sum": 58698,
    "paddle.Tensor.isnan": 55184,
    "paddle.gammainc": 2378,
    "paddle.nn.functional.log_loss": 20178,
    "paddle.less_than": 56808,
    "paddle.Tensor.trunc": 34186,
    "paddle.sinh": 33851,
    "paddle.Tensor.unique": 1487,
    "paddle.positive": 5479885,
    "paddle.nanmedian": 2836,
    "paddle.nn.functional.conv3d_transpose": 3204,
    "paddle.square": 33810,
    "paddle.Tensor.detach": 12614448,
    "paddle.kthvalue": 3230,
    "paddle.mod": 22295,
    "paddle.Tensor.rot90": 19459,
    "paddle.Tensor.dim": 14433255,
    "paddle.ldexp": 12113,
    "paddle.rsqrt": 33838,
    "paddle.Tensor.less": 30607,
    "paddle.Tensor.__rrshift__": 33481,
    "paddle.Tensor.__len__": 2154792,
    "paddle.pow": 27143,
    "paddle.Tensor.__mod__": 22349,
    "paddle.Tensor.masked_select": 7203,
    "paddle.geometric.segment_sum": 20010,
    "paddle.Tensor.inner": 51721,
    "paddle.diag": 199725,
    "paddle.Tensor.lerp": 22191,
    "paddle.linalg.norm": 66479,
    "paddle.nn.functional.rrelu": 20818,
    "paddle.nn.functional.mish": 32808,
    "paddle.Tensor.__lshift__": 22338,
    "paddle.nn.functional.binary_cross_entropy_with_logits": 9646,
    "paddle.fmax": 33694,
    "paddle.squeeze": 2700427,
    "paddle.nn.functional.conv3d": 1559,
    "paddle.addmm": 31761,
    "paddle.Tensor.floor": 33839,
    "paddle.nn.functional.glu": 13909,
    "paddle.Tensor.sum": 68402,
    "paddle.diag_embed": 3661,
    "paddle.polygamma": 1793,
    "paddle.amin": 50777,
    "paddle.sinc": 3389,
    "paddle.view": 750201,
    "paddle.floor": 33815,
    "paddle.Tensor.argmax": 36546,
    "paddle.cdist": 22208,
    "paddle.Tensor.erfinv": 30679,
    "paddle.nn.functional.selu": 33360,
    "paddle.nanquantile": 24384,
    "paddle.isposinf": 1000,
    "paddle.nn.functional.adaptive_max_pool2d": 25828,
    "paddle.Tensor.__lt__": 22019,
    "paddle.linalg.matrix_transpose": 2337440,
    "paddle.linalg.solve": 2715,
    "paddle.unique_consecutive": 6999,
    "paddle.incubate.segment_max": 9957,
    "paddle.Tensor.fill_": 68510,
    "paddle.outer": 1897,
    "paddle.tanh": 33846,
    "paddle.Tensor.__mul__": 33522,
    "paddle.nn.functional.adaptive_avg_pool3d": 19743,
    "paddle.acosh": 33859,
    "paddle.chunk": 29583,
    "paddle.Tensor.conj": 33610,
    "paddle.nn.functional.softplus": 33404,
    "paddle.reduce_as": 56022,
    "paddle.Tensor.norm": 65593,
    "paddle.sqrt": 33937,
    "paddle.linalg.qr": 5519,
    "paddle.nn.functional.batch_norm": 29508,
    "paddle.nn.functional.tanh": 33853,
    "paddle.equal": 56247,
    "paddle.incubate.nn.functional.fused_matmul_bias": 33751,
    "paddle.neg": 33849,
    "paddle.tolist": 11272,
    "paddle.nn.functional.conv1d": 30827,
    "paddle.kron": 1311,
    "paddle.Tensor.flatten": 1816345,
    "paddle.slice": 1331905,
    "paddle.Tensor.quantile": 17789,
    "paddle.dist": 21993,
    "paddle.Tensor.log10": 33813,
    "paddle.lerp": 33528,
    "paddle.erf": 29636,
    "paddle.acos": 33801,
    "paddle.stanh": 33926,
    "paddle.masked_scatter": 22631,
    "paddle.nn.functional.embedding": 66306,
    "paddle.Tensor.zero_": 68861,
    "paddle.nn.functional.normalize": 21309,
    "paddle.Tensor.neg": 33807,
    "paddle.nn.functional.softsign": 33892,
    "paddle.as_strided": 199910,
    "paddle.unbind": 85063,
    "paddle.unique": 1489,
    "paddle.scatter_nd_add": 1437,
    "paddle.gammaincc": 2555,
    "paddle.Tensor.remainder": 22180,
    "paddle.equal_all": 555286,
    "paddle.Tensor.__ror__": 33465,
    "paddle.cummax": 55599,
    "paddle.std": 9126,
    "paddle.fft.ifftn": 41344,
    "paddle.vecdot": 10704,
    "paddle.i0": 21481,
    "paddle.Tensor.astype": 3208126,
    "paddle.nn.functional.gather_tree": 1000,
    "paddle.Tensor.var": 7832,
    "paddle.nn.functional.unfold": 5718,
    "paddle.renorm": 3503,
    "paddle.logical_xor": 52894,
    "paddle.Tensor.gather": 72474,
    "paddle.Tensor.logical_or": 84635,
    "paddle.Tensor.tril": 33390,
    "paddle.nn.functional.elu": 33842,
    "paddle.Tensor.flip": 11097,
    "paddle.nn.functional.label_smooth": 33539,
    "paddle.Tensor.inverse": 2707,
    "paddle.isinf": 42950,
    "paddle.Tensor.ceil": 33828,
    "paddle.is_complex": 2834372,
    "paddle.fft.ihfft2": 99947,
    "paddle.repeat_interleave": 9813,
    "paddle.nn.functional.channel_shuffle": 31913,
    "paddle.Tensor.dot": 33757,
    "paddle.linalg.matrix_norm": 42429,
    "paddle.deg2rad": 33772,
    "paddle.Tensor.reshape": 66716,
    "paddle.Tensor.diag_embed": 5658,
    "paddle.signbit": 4599,
    "paddle.linalg.svdvals": 21631,
    "paddle.logical_or": 85027,
    "paddle.Tensor.slice_scatter": 32593,
    "paddle.heaviside": 33800,
    "paddle.broadcast_tensors": 16272,
    "paddle.linalg.det": 1000,
    "paddle.Tensor.rsqrt": 33809,
    "paddle.nn.functional.nll_loss": 366849,
    "paddle.flip": 10401,
    "paddle.Tensor.digamma": 8544,
    "paddle.Tensor.__rmod__": 22183,
    "paddle.linalg.pinv": 1074,
    "paddle.Tensor.isclose": 27577,
    "paddle.clip": 33866,
    "paddle.nn.functional.lp_pool1d": 10734,
    "paddle.numel": 1155041,
    "paddle.Tensor.clone": 32273,
    "paddle.nn.functional.npair_loss": 6743,
    "paddle.Tensor.subtract": 22223,
    "paddle.cummin": 12631,
    "paddle.conj": 32583,
    "paddle.nn.functional.sequence_mask": 12839,
    "paddle.frac": 13371,
    "paddle.histogramdd": 5336,
    "paddle.scatter_nd": 1000,
    "paddle.Tensor.diagonal": 2017934,
    "paddle.Tensor.tile": 33795,
    "paddle.linalg.lu_unpack": 10765,
    "paddle.Tensor.slice": 198496,
    "paddle.Tensor.repeat_interleave": 16163,
    "paddle.linalg.corrcoef": 5085,
    "paddle.incubate.softmax_mask_fuse_upper_triangle": 38131,
    "paddle.Tensor.kthvalue": 1495,
    "paddle.Tensor.__rtruediv__": 17130,
    "paddle.Tensor.signbit": 1000,
    "paddle.Tensor.cos": 33870,
    "paddle.Tensor.__rmatmul__": 13088,
    "paddle.as_complex": 3339679,
    "paddle.Tensor.logical_not": 12687,
    "paddle.multiplex": 29797,
    "paddle.Tensor.__le__": 21231,
    "paddle.i1e": 31796,
    "paddle.Tensor.square": 33808,
    "paddle.argsort": 1000,
    "paddle.logcumsumexp": 2955,
    "paddle.nan_to_num": 3470,
    "paddle.Tensor.nonzero": 1684,
    "paddle.histogram": 1544,
    "paddle.signal.stft": 1000,
    "paddle.nn.functional.avg_pool1d": 1335,
    "paddle.Tensor.__abs__": 33766,
    "paddle.Tensor.__ge__": 21328,
    "paddle.view_as": 711815,
    "paddle.Tensor.gcd": 1000,
    "paddle.incubate.segment_min": 8591,
    "paddle.incubate.nn.functional.swiglu": 43514,
    "paddle.any": 21580,
    "paddle.dot": 33667,
    "paddle.isneginf": 1000,
    "paddle.nn.functional.adaptive_log_softmax_with_loss": 1469,
    "paddle.unfold": 7292,
    "paddle.lcm": 1000,
    "paddle.geometric.segment_max": 9664,
    "paddle.Tensor.tolist": 1000,
    "paddle.strided_slice": 476161,
    "paddle.linalg.cond": 1000,
    "paddle.incubate.segment_mean": 151362,
    "paddle.Tensor.mode": 1000,
    "paddle.reshape": 66714,
    "paddle.Tensor.logical_and": 84274,
}


class APITestPaddleTorchGPUPerformance(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.converter = get_converter()

    @func_set_timeout(600)
    def test(self):
        if self.need_skip(paddle_only=True):
            print("[Skip]", flush=True)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(
                f"[paddle_to_torch] Convertion failed for {self.api_config.config}: {e!s}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(
                f"[paddle_to_torch] Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.code or not convert_result.code.is_valid():
            print(
                f"[paddle_to_torch] No code generated for {self.api_config.api_name}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return

        numel = total_numel(self.api_config)
        # test_loop = 1000
        test_loop = api_loop.get(self.api_config.api_name, 1000)
        # test_loop = 2147483647 * 20 // numel
        # test_loop = 100000 if test_loop > 100000 else test_loop

        combined = ""
        paddle_forward = None
        torch_forward = None
        paddle_backward = None
        torch_backward = None
        paddle_forward_sync = None
        torch_forward_sync = None
        paddle_backward_sync = None
        torch_backward_sync = None

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err))
            write_to_log("numpy_error", self.api_config.config)
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                return

            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)

            with paddle.no_grad():
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                        start = time.time()
                        for _i in range(test_loop):
                            self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                        before_sync = time.time()
                        paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                        end = time.time()
                        paddle_forward = end - start
                        paddle_forward_sync = end - before_sync
                else:
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    start = time.time()
                    for _i in range(test_loop):
                        self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                    before_sync = time.time()
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    end = time.time()
                    paddle_forward = end - start
                    paddle_forward_sync = end - before_sync
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            # print_performance(False, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_forward_sync, torch_forward_sync, paddle_backward, torch_backward, paddle_backward_sync, torch_backward_sync, combined)
            print("[Error]", str(err))
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return

        try:
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(
                    paddle_output
                )
                if (
                    len(inputs_list) != 0
                    and len(result_outputs) != 0
                    and len(result_outputs_grads) != 0
                ):
                    paddle.grad(
                        result_outputs,
                        inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    start = time.time()
                    for _i in range(test_loop):
                        paddle.grad(
                            result_outputs,
                            inputs_list,
                            grad_outputs=result_outputs_grads,
                            allow_unused=True,
                        )
                    before_sync = time.time()
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    end = time.time()
                    paddle_backward = end - start
                    paddle_backward_sync = end - before_sync
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            # print_performance(False, self.api_config.api_name, self.api_config.config, numel, test_loop, paddle_forward, torch_forward, paddle_forward_sync, torch_forward_sync, paddle_backward, torch_backward, paddle_backward_sync, torch_backward_sync, combined)
            print("[Error]", str(err))
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err

            # return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return

            # torch_args 与 torch_kwargs 是尚未映射的 torch 参数（即按 paddle 的参数顺序与关键字排列的 torch tensors）
            # (弃用)以下代码等价于:
            # torch_output = Paddle2TorchConverter.execute(convert_result, self.torch_args, self.torch_kwargs)
            # 准备执行环境，将参数(torch tensors)直接映射至locals()
            exec_globals = {"torch": torch}
            exec_locals = {
                "args": self.torch_args,
                "kwargs": self.torch_kwargs,
                "result": None,
                **self.torch_kwargs,
            }
            if self.api_config.api_name == "paddle.nn.functional.rnnt_loss":
                if paddle.device.get_device() == "cpu":
                    exec_locals["fused_log_softmax"] = False

            # convert_result.is_torch_corresponding 为 True 时代表有对应的 Torch API
            # 执行 *_compiled 编译好的代码速度更快，定位 compile error 时可删去 _compiled
            code = convert_result.code
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)

            if not convert_result.is_torch_corresponding:
                combined = "combined"

            if code.core_compiled:
                if self.test_amp:
                    with torch.autocast(device_type="cuda"):
                        exec(code.core_compiled, exec_globals, exec_locals)
                else:
                    exec(code.core_compiled, exec_globals, exec_locals)

            # if code.postprocess_compiled:
            #     exec(code.postprocess_compiled, exec_globals, exec_locals)
            output_var = convert_result.output_var or "result"
            torch_output = exec_locals[output_var]

            with torch.no_grad():
                if code.core_compiled:
                    if self.test_amp:
                        with torch.autocast(device_type="cuda"):
                            torch.cuda.synchronize()
                            start = time.time()
                            for i in range(test_loop):
                                exec(code.core_compiled, exec_globals, exec_locals)
                            before_sync = time.time()
                            torch.cuda.synchronize()
                            end = time.time()
                            torch_forward = end - start
                            torch_forward_sync = end - before_sync
                    else:
                        torch.cuda.synchronize()
                        start = time.time()
                        for i in range(test_loop):
                            exec(code.core_compiled, exec_globals, exec_locals)
                        before_sync = time.time()
                        torch.cuda.synchronize()
                        end = time.time()
                        torch_forward = end - start
                        torch_forward_sync = end - before_sync

            del exec_globals, exec_locals, output_var, convert_result, code
        except Exception as err:
            print_performance(
                False,
                self.api_config.api_name,
                self.api_config.config,
                numel,
                test_loop,
                paddle_forward,
                torch_forward,
                paddle_forward_sync,
                torch_forward_sync,
                paddle_backward,
                torch_backward,
                paddle_backward_sync,
                torch_backward_sync,
                combined,
            )
            print("[Error]", str(err))
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return

        try:
            if self.need_check_grad():
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(
                    torch_output
                )
                del self.torch_args, self.torch_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    torch.autograd.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                        retain_graph=True,
                    )
                    torch.cuda.synchronize()
                    start = time.time()
                    for _i in range(test_loop):
                        torch.autograd.grad(
                            outputs=result_outputs,
                            inputs=inputs_list,
                            grad_outputs=result_outputs_grads,
                            retain_graph=True,
                        )
                    before_sync = time.time()
                    torch.cuda.synchronize()
                    end = time.time()
                    torch_backward = end - start
                    torch_backward_sync = end - before_sync
                del inputs_list, result_outputs, result_outputs_grads, torch_output
            else:
                del self.torch_args, self.torch_kwargs, torch_output
        except Exception as err:
            print_performance(
                False,
                self.api_config.api_name,
                self.api_config.config,
                numel,
                test_loop,
                paddle_forward,
                torch_forward,
                paddle_forward_sync,
                torch_forward_sync,
                paddle_backward,
                torch_backward,
                paddle_backward_sync,
                torch_backward_sync,
                combined,
            )
            print("[Error]", str(err))
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return
        print_performance(
            True,
            self.api_config.api_name,
            self.api_config.config,
            numel,
            test_loop,
            paddle_forward,
            torch_forward,
            paddle_forward_sync,
            torch_forward_sync,
            paddle_backward,
            torch_backward,
            paddle_backward_sync,
            torch_backward_sync,
            combined,
        )
