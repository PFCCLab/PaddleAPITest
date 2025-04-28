import collections
import inspect

import numpy
import paddle
import torch

from .api_config import USE_CACHED_NUMPY, TensorConfig, cached_numpy

# Todo: check paddle.linalg.pca_lowrank @cangtianhuang
not_support_api = ["paddle.Tensor.coalesce",
 "paddle.Tensor.is_coalesced",
 "paddle.Tensor.index_put",
 "paddle.Tensor.index_sample",
 "paddle.linalg.pca_lowrank"
 ]

rand_apis = [
    "paddle.bernoulli",
    "paddle.bernoulli_",
    "paddle.binomial",
    "paddle.cauchy_",
    "paddle.geometric_",
    "paddle.log_normal",
    "paddle.log_normal_",
    "paddle.multinomial",
    "paddle.normal",
    "paddle.normal_",
    "paddle.poisson",
    "paddle.rand",
    "paddle.randn",
    "paddle.randint",
    "paddle.randint_like",
    "paddle.randperm",
    "paddle.standard_gamma",
    "paddle.standard_normal", 
    "paddle.uniform",
    "paddle.Tensor.bernoulli_",
    "paddle.Tensor.cauchy_",
    "paddle.Tensor.exponential_",
    "paddle.Tensor.geometric_",
    "paddle.Tensor.log_normal_",
    "paddle.Tensor.multinomial",
    "paddle.Tensor.normal_",
    "paddle.Tensor.uniform_",
    "paddle.empty",
    "paddle.empty_like",
]

stochastic_behavior_apis =[
    "paddle.Tensor.top_p_sampling", 
    "paddle.incubate.nn.functional.fused_bias_dropout_residual_layer_norm",
    "paddle.incubate.nn.functional.fused_dropout_add",
    "paddle.incubate.nn.functional.moe_dispatch",
    "paddle.nn.functional.alpha_dropout", 
    "paddle.nn.functional.fused_feedforward",
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
    "paddle.nn.functional.feature_alpha_dropout",
]

single_op_no_signature_apis = ["__eq__", "__ge__", "__gt__", "__le__", "__lt__", "__add__", "__div__", "__floordiv__", "__matmul__", "__mod__", "__ne__", "__pow__", "__radd__", "__rmatmul__", "__rmod__", "__rmul__", "__rpow__", "__rsub__", "__rtruediv__", "__truediv__", "__mul__", "__sub__"]

no_signature_api_mappings = {
    f"paddle.Tensor.{method}": {
        "self": lambda cfg: get_arg(cfg, 0, "self"),
        "y": lambda cfg: get_arg(cfg, 1, "y")
    }
    for method in single_op_no_signature_apis
}

# Todo: check paddle.prod paddle.cumprod @cangtianhuang
int_too_big_fail_api = [
    "paddle.Tensor.cumprod",
    "paddle.pow",
    "paddle.Tensor.pow",
    "paddle.lcm",
    "paddle.Tensor.lcm",
    "paddle.Tensor.prod",
]

handle_axes_api = [
    "paddle.mean",
    "paddle.max",
    "paddle.min",
    "paddle.sum",
    "paddle.prod",
]


class APITestBase:
    def __init__(self, api_config):
        self.api_config = api_config
        self.outputs_grad_numpy = []

    def need_skip(self):
        # not support
        if "sparse" in self.api_config.api_name:
            return True
        if self.api_config.api_name in not_support_api:
            return True
        if self.api_config.api_name in rand_apis:
            return True
        if self.api_config.api_name in stochastic_behavior_apis:
            return True
        if self.api_config.api_name in int_too_big_fail_api:
            return True
        for i in range(len(self.api_config.args)):
            if isinstance(self.api_config.args[i], TensorConfig):
                if self.api_config.args[i].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                    return True
            elif isinstance(self.api_config.args[i], list):
                tmp = []
                for j in range(len(self.api_config.args[i])):
                    if isinstance(self.api_config.args[i][j], TensorConfig):
                        if self.api_config.args[i][j].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                            return True
            elif isinstance(self.api_config.args[i], tuple):
                tmp = []
                for j in range(len(self.api_config.args[i])):
                    if isinstance(self.api_config.args[i][j], TensorConfig):
                        if self.api_config.args[i][j].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                            return True
            elif self.api_config.args[i] in [paddle.base.core.DataType.FLOAT8_E4M3FN, paddle.base.core.DataType.FLOAT8_E5M2, "float8_e5m2", "float8_e4m3fn"]:
                return True

        for key, arg_config in self.api_config.kwargs.items():
            if isinstance(arg_config, TensorConfig):
                if arg_config.dtype in ["float8_e5m2", "float8_e4m3fn"]:
                    return True
            elif isinstance(arg_config, list):
                value = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        if arg_config[i].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                            return True
            elif isinstance(arg_config, tuple):
                tmp = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        if arg_config[i].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                            return True
            elif arg_config in [paddle.base.core.DataType.FLOAT8_E4M3FN, paddle.base.core.DataType.FLOAT8_E5M2, "float8_e5m2", "float8_e4m3fn"]:
                return True

        return False

    def need_check_grad(self):
        # if self.is_forward_only():
        #     return False
        # if self.api_config.api_name == "paddle.assign":
        #     if (len(self.paddle_args) and isinstance(self.paddle_args[0], list)) or (
        #         len(self.paddle_args) > 1 and self.paddle_args[1] is not None
        #     ):
        #         return False
        # if len(self.api_config.args) > 0 and isinstance(
        #     self.api_config.args[0], TensorConfig
        # ):
        #     dtype = self.api_config.args[0].dtype
        #     if dtype in [
        #         "float32",
        #         "float64",
        #         "float16",
        #         "complex64",
        #         "complex128",
        #         "bfloat16",
        #     ]:
        #         return True
        # return True

        if not self.is_forward_only() and not (self.api_config.api_name == "paddle.assign" and len(self.paddle_args) and isinstance(self.paddle_args[0], list)) and not (self.api_config.api_name == "paddle.assign" and len(self.paddle_args) > 1 and self.paddle_args[1] is not None):
            if len(self.api_config.args) > 0 and isinstance(self.api_config.args[0], TensorConfig):
                dtype = self.api_config.args[0].dtype
                if dtype in ['float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16']:
                    return True
            return True
        return False

    def ana_api_info(self):
        return self.ana_paddle_api_info() and self.ana_torch_api_info()

    def ana_paddle_api_info(self):
        self.paddle_api = eval(self.api_config.api_name)
        self.paddle_args_config = self.api_config.args
        self.paddle_kwargs_config = self.api_config.kwargs
        return True

    def ana_torch_api_info(self):
        self.torch_args_config = []
        self.torch_kwargs_config = collections.OrderedDict()
        self.paddle_merged_kwargs_config = collections.OrderedDict()

        if self.api_config.api_name in ["paddle.Tensor.__getitem__", "paddle.Tensor.__setitem__"]:
            self.torch_args_config = self.api_config.args
            return True

        if self.api_config.api_name not in no_signature_api_mappings:
            # For APIs with signatures, use paddle_sig.bind to get arguments
            paddle_sig = inspect.signature(self.paddle_api)
            paddle_bound_args = paddle_sig.bind(*self.api_config.args, **self.api_config.kwargs)
            paddle_args_dict = paddle_bound_args.arguments
        else:
            # For APIs without signatures, use the external mapping dict
            mapping = no_signature_api_mappings[self.api_config.api_name]
            paddle_args_dict = {}
            for key, get_value_func in mapping.items():
                paddle_args_dict[key] = get_value_func(self.api_config)

        self.paddle_merged_kwargs_config = paddle_args_dict
        self.torch_kwargs_config.update(paddle_args_dict)
        self.torch_kwargs_config.pop('name', None)

        return True

    def _handle_list_or_tuple(self, config_items, is_tuple=False, index=None, key=None, list_index=[]):
        """处理 list 或 tuple """

        need_axes_handling = self.api_config.api_name in handle_axes_api
        need_indices_handling = self.api_config.api_name == "paddle.index_put"

        if need_indices_handling and (index == 1 or key == "indices"):
            return self._handle_indices_arg(config_items, is_tuple)
        elif need_axes_handling and (index == 1 or key == "axis"):
            return self._handle_axis_arg(config_items, is_tuple)

        tmp = []
        for i, item in enumerate(config_items):
            current_list_index = list_index + [i]
            if isinstance(item, (list, tuple)):
                is_nested_tuple = isinstance(item, tuple)
                processed_item = self._handle_list_or_tuple(
                    item, 
                    is_tuple=is_nested_tuple, 
                    index=index, 
                    key=key, 
                    list_index=current_list_index
                )
            elif isinstance(item, TensorConfig):
                processed_item = item.get_paddle_tensor(
                    self.api_config,
                    index=index,
                    key=key,
                    list_index=current_list_index
                )
            else:
                processed_item = item
            tmp.append(processed_item)
        return tuple(tmp) if is_tuple else tmp

    def _handle_axis_arg(self, config_items, is_tuple=False):
        """处理 axis 参数"""
        x = self.paddle_args_config[0] if len(self.paddle_args_config) > 0 else self.paddle_kwargs_config["x"]
        max_dim = max(len(x.shape), 1) # scalar

        tmp = []
        used_axes = set()
        tensor_configs = []

        for item in config_items:
            if isinstance(item, TensorConfig):
                if item.shape not in [[], [1]] or item.dtype not in ["int32", "int64"]:
                    raise ValueError(f"Invalid TensorConfig for axis: shape {item.shape} or dtype {item.dtype}")
                tensor_configs.append(item)
                tmp.append(0) # placeholder
            elif isinstance(item, int):
                if not (-max_dim <= item < max_dim):
                    raise ValueError(f"Axis value {item} out of range [-{max_dim}, {max_dim})")
                positive_axis = item + max_dim if item < 0 else item
                if positive_axis in used_axes:
                    raise ValueError(f"Duplicate axis value: {item}")
                used_axes.add(positive_axis)
                tmp.append(item)
            else:
                raise ValueError(f"Invalid item type for axis: {type(item)}")

        if tensor_configs:
            available_dims = list(set(range(max_dim)) - used_axes)
            if len(available_dims) < len(tensor_configs):
                raise ValueError(f"Not enough available dimensions ({len(available_dims)}) for {tensor_config_count} TensorConfig items")
            selected_dims = numpy.random.choice(available_dims, size=len(tensor_configs), replace=False)
            mask = numpy.random.randint(0, 2, size=len(tensor_configs)).astype(bool)
            final_dims = numpy.where(mask, selected_dims - max_dim, selected_dims)
            tensor_idx = 0
            for i, item in enumerate(config_items):
                if isinstance(item, TensorConfig):
                    item.fill_numpy_tensor(final_dims[tensor_idx])
                    tmp[i] = item.get_paddle_tensor(self.api_config)
                    tensor_idx += 1
        return tuple(tmp) if is_tuple else tmp

    def _handle_indices_arg(self, config_items, is_tuple=False):
        x = self.paddle_args_config[0] if len(self.paddle_args_config) > 0 else self.paddle_kwargs_config["x"]
        value = self.paddle_args_config[2] if len(self.paddle_args_config) > 2 else self.paddle_kwargs_config["value"]
        x_shape = x.shape
        value_shape = value.shape

        tmp = []
        matched_axis = 0
        indices_shape_len = 0
        for item in config_items:
            if item.dtype != "bool":
                matched_axis += 1
                indices_shape_len = max(indices_shape_len, len(item.shape))

        expected = indices_shape_len + len(x_shape) - matched_axis
        reduced = expected - len(value_shape)
        x_shape_index = 0
        value_shape_index = indices_shape_len

        for item in config_items:
            if item.dtype == "bool":
                true_needed = []
                for i in range(len(item.shape)):
                    if reduced > 0:
                        reduced -= 1
                        true_needed.append(1)
                    else:
                        true_needed.append(value_shape[value_shape_index])
                        value_shape_index += 1
                for i in range(len(true_needed) - 1, 0, -1):
                    if true_needed[i] > item.shape[i]:
                        true_needed[i - 1] *= true_needed[i] // item.shape[i]
                        true_needed[i] = item.shape[i]
                mask = numpy.zeros(item.shape, dtype=bool)
                indices = [
                    numpy.random.choice(dim_size, size=needed, replace=False)
                    for dim_size, needed in zip(item.shape, true_needed)
                ]
                mask[numpy.ix_(*indices)] = True
                item.numpy_tensor = mask
                x_shape_index += len(item.shape)
            else:
                x_dim = x_shape[x_shape_index]
                item.numpy_tensor = numpy.random.randint(-x_dim, x_dim, size=item.shape, dtype=item.dtype)
                x_shape_index += 1
            tmp.append(item.get_paddle_tensor(self.api_config))
        return tuple(tmp) if is_tuple else tmp

    def gen_paddle_input(self):
        self.paddle_args = []
        self.paddle_kwargs = collections.OrderedDict()
        self.paddle_merged_kwargs = collections.OrderedDict()

        for i, arg_config in enumerate(self.paddle_args_config):
            if isinstance(arg_config, TensorConfig):
                self.paddle_args.append(arg_config.get_paddle_tensor(self.api_config, index=i))
            elif isinstance(arg_config, list):
                self.paddle_args.append(self._handle_list_or_tuple(arg_config, index=i))
            elif isinstance(arg_config, tuple):
                self.paddle_args.append(self._handle_list_or_tuple(arg_config, is_tuple=True, index=i))
            else:
                self.paddle_args.append(arg_config)

        for key, kwarg_config in self.paddle_kwargs_config.items():
            if isinstance(kwarg_config, TensorConfig):
                self.paddle_kwargs[key] = kwarg_config.get_paddle_tensor(self.api_config, key=key)
            elif isinstance(kwarg_config, list):
                self.paddle_kwargs[key] = self._handle_list_or_tuple(kwarg_config, key=key)
            elif isinstance(kwarg_config, tuple):
                self.paddle_kwargs[key] = self._handle_list_or_tuple(kwarg_config, is_tuple=True, key=key)
            else:
                self.paddle_kwargs[key] = kwarg_config

        if len(self.paddle_args) == 0 and "paddle.Tensor." in self.api_config.api_name:
            self.paddle_args.append(self.paddle_kwargs.popitem(last=False)[1])

        if self.api_config.api_name == "paddle.linalg.lstsq" and 'gpu' in paddle.device.get_device():
            if len(self.paddle_args) > 3:
                self.paddle_args[3] = "gels"
            elif "driver" in self.paddle_kwargs:
                self.paddle_kwargs["driver"] = "gels"

        if self.need_check_grad():
            if (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__") or self.api_config.api_name == "paddle.Tensor.__setitem__":
                self.paddle_args, self.paddle_kwargs = self.copy_paddle_input()
        self.clear_paddle_tensor()
        return True

    def copy_paddle_input(self):
        args = []
        kwargs = collections.OrderedDict()

        for i in range(len(self.paddle_args)):
            if isinstance(self.paddle_args[i], paddle.Tensor):
                args.append(paddle.assign(self.paddle_args[i]))
            elif isinstance(self.paddle_args[i], list) and len(self.paddle_args[i]) > 0 and isinstance(self.paddle_args[i][0], paddle.Tensor):
                tmp = []
                for j in range(len(self.paddle_args[i])):
                    tmp.append(paddle.assign(self.paddle_args[i][j]))
                args.append(tmp)
            elif isinstance(self.paddle_args[i], tuple) and len(self.paddle_args[i]) > 0 and isinstance(self.paddle_args[i][0], paddle.Tensor):
                tmp = []
                for j in range(len(self.paddle_args[i])):
                    tmp.append(paddle.assign(self.paddle_args[i][j]))
                args.append(tmp)
            else:
                args.append(self.paddle_args[i])

        for key, value in self.paddle_kwargs.items():
            if isinstance(value, paddle.Tensor):
                kwargs[key] = paddle.assign(value)
            elif isinstance(value, list):
                tmp = []
                for j in range(len(value)):
                    tmp.append(paddle.assign(value[j]))
                kwargs[key] = tmp
            elif isinstance(value, tuple):
                tmp = []
                for j in range(len(value)):
                    tmp.append(paddle.assign(value[j]))
                kwargs[key] = tmp
            else:
                kwargs[key] = value
        return args, kwargs

    def get_paddle_input_list(self):
        result = []

        for i in range(len(self.paddle_args)):
            if isinstance(self.paddle_args[i], paddle.Tensor):
                result.append(self.paddle_args[i])
            elif isinstance(self.paddle_args[i], tuple) or isinstance(self.paddle_args[i], list):
                for item in self.paddle_args[i]:
                    if isinstance(item, paddle.Tensor):
                        result.append(item)

        for key, value in self.paddle_kwargs.items():
            if isinstance(value, paddle.Tensor):
                result.append(value)
            elif isinstance(value, tuple) or isinstance(value, list):
                for item in value:
                    if isinstance(item, paddle.Tensor):
                        result.append(item)

        return result

    def get_torch_input_list(self):
        result = []
        for i in range(len(self.torch_args)):
            if isinstance(self.torch_args[i], torch.Tensor):
                result.append(self.torch_args[i])
            elif isinstance(self.torch_args[i], tuple) or isinstance(self.torch_args[i], list):
                for item in self.paddle_args[i]:
                    if isinstance(item, torch.Tensor):
                        result.append(item)

        for key, value in self.torch_kwargs.items():
            if isinstance(value, torch.Tensor):
                result.append(value)
            elif isinstance(value, tuple) or isinstance(value, list):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        result.append(item)
        return result

    def get_cached_numpy(self, dtype, shape):
        numel = 1
        for i in shape:
            numel = numel * i

        start = (4300000000 - numel - 100) if (4300000000 - numel - 100) > 0 else 0
        if dtype in cached_numpy:
            tensor = cached_numpy[dtype][start:start+numel].reshape(shape)
        else:
            if "int" in dtype:
                cached_numpy[dtype] = numpy.random.randint(-65535, 65535, size=4300000000, dtype="int64").astype(dtype)
                tensor = cached_numpy[dtype][start:start+numel].reshape(shape)
            else:
                cached_numpy[dtype] = (numpy.random.random([4300000000]) - 0.5).astype(dtype)
                tensor = cached_numpy[dtype][start:start+numel].reshape(shape)
        return tensor

    def gen_paddle_output_and_output_grad(self, outputs):
        result_outputs = []
        if isinstance(outputs, paddle.Tensor):
            result_outputs.append(outputs)
        elif isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], paddle.Tensor):
            result_outputs = outputs
        elif isinstance(outputs, paddle.autograd.autograd.Hessian) or \
                isinstance(outputs, paddle.autograd.autograd.Jacobian):
            result_outputs.append(outputs[:])
        elif isinstance(outputs, tuple):
            for output in outputs:
                if isinstance(output, paddle.Tensor):
                    result_outputs.append(output)
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, paddle.Tensor):
                            result_outputs.append(item)
                    else:
                        raise ValueError("outputs format not support")
                elif isinstance(output, paddle.autograd.autograd.Hessian) or \
                        isinstance(output, paddle.autograd.autograd.Jacobian):
                    result_outputs.extend(output[:])
                elif isinstance(output, tuple) and len(output) > 0 and \
                        (isinstance(output[0], paddle.autograd.autograd.Hessian) or \
                        isinstance(output[0], paddle.autograd.autograd.Jacobian)):
                    for lazy_obj in output:
                        result_outputs.append(lazy_obj[:])
                else:
                    raise ValueError("outputs format not support")
                # elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], paddle.Tensor):
                #     result_outputs.append(output)
                # elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], paddle.Tensor):
                #     result_outputs.append(output)

        result_outputs_grads = []

        if len(self.outputs_grad_numpy) == 0:
            for output in result_outputs:
                dtype = str(output.dtype)[7:]
                if USE_CACHED_NUMPY:
                    dtype = "float32" if dtype == "bfloat16" else dtype
                    numpy_tensor = self.get_cached_numpy(dtype, output.shape)
                else:
                    if "int" in dtype:
                        numpy_tensor = (numpy.random.randint(-65535, 65535, size=output.shape)).astype(dtype)
                    else:
                        dtype = "float32" if dtype == "bfloat16" else dtype
                        numpy_tensor = (numpy.random.random(output.shape) - 0.5).astype(dtype)
                self.outputs_grad_numpy.append(numpy_tensor)
        for numpy_tensor in self.outputs_grad_numpy:
            dtype = str(numpy_tensor.dtype)
            result_output_grad = paddle.to_tensor(
                numpy_tensor,
                dtype=dtype if dtype != 'bfloat16' else "float32",
            )
            result_output_grad.stop_gradient = False
            if dtype == "bfloat16":
                result_output_grad = paddle.cast(result_output_grad, dtype="uint16")
            result_outputs_grads.append(result_output_grad)
        return result_outputs, result_outputs_grads

    def convert_dtype_to_torch_type(self, dtype):
        if dtype in ["float32", numpy.float32]:
            return torch.float32
        elif dtype in ['float16', numpy.float16]:
            return torch.float16
        elif dtype in ['float64', numpy.float64]:
            return torch.float64
        elif dtype in ['int16', numpy.int16]:
            return torch.int16
        elif dtype in ['int8', numpy.int8]:
            return torch.int8
        elif dtype in ['bool', numpy.bool_]:
            return torch.bool
        elif dtype in ['bfloat16', numpy.uint16]:
            return torch.bfloat16
        elif dtype in ['uint8', numpy.uint8]:
            return torch.uint8
        elif dtype in ['int32', numpy.int32]:
            return torch.int32
        elif dtype in ['int64', numpy.int64]:
            return torch.int64
        elif dtype in ['complex64', numpy.complex64]:
            return torch.complex64
        elif dtype in ['complex128', numpy.complex128]:
            return torch.complex128
        else:
            raise ValueError(f'Unsupport dtype: {dtype}')

    def gen_torch_output_and_output_grad(self, outputs):
        result_outputs = []
        if isinstance(outputs, torch.Tensor):
            result_outputs.append(outputs)
        elif isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            result_outputs = outputs
        elif isinstance(outputs, tuple):
            for output in outputs:
                if isinstance(output, torch.Tensor):
                    result_outputs.append(output)
                else:
                    raise ValueError("outputs format not support")
                # elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                #     result_outputs.append(output)
                # elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                #     result_outputs.append(output)

        result_outputs_grads = []
        if len(self.outputs_grad_numpy) == 0:
            for output in result_outputs:
                dtype = str(output.dtype)[6:]
                if USE_CACHED_NUMPY:
                    dtype = "float32" if dtype == "bfloat16" else dtype
                    numpy_tensor = self.get_cached_numpy(dtype, output.shape)
                else:
                    if "int" in dtype:
                        numpy_tensor = (numpy.random.randint(-65535, 65535, size=output.shape)).astype(dtype)
                    else:
                        dtype = "float32" if dtype == "bfloat16" else dtype
                        numpy_tensor = (numpy.random.random(output.shape) - 0.5).astype(dtype)
                self.outputs_grad_numpy.append(numpy_tensor)
        for numpy_tensor in self.outputs_grad_numpy:
            dtype = str(numpy_tensor.dtype)
            result_output_grad = torch.tensor(
                numpy_tensor,
                dtype=self.convert_dtype_to_torch_type(dtype)
                if dtype != 'bfloat16'
                else torch.float32,
            )

            if dtype == "bfloat16":
                result_output_grad = result_output_grad.to(dtype=torch.bfloat16)
            result_outputs_grads.append(result_output_grad)

        return result_outputs, result_outputs_grads

    def gen_paddle_input_with_merged_kwargs(self):
        self.paddle_args = []
        self.paddle_kwargs = collections.OrderedDict()
        self.paddle_merged_kwargs = collections.OrderedDict()

        for i in range(len(self.paddle_args_config)):
            if isinstance(self.paddle_args_config[i], TensorConfig):
                self.paddle_args.append(self.paddle_args_config[i].get_paddle_tensor(self.api_config))
            elif isinstance(self.paddle_args_config[i], list):
                tmp = []
                for j in range(len(self.paddle_args_config[i])):
                    if isinstance(self.paddle_args_config[i][j], TensorConfig):
                        tmp.append(self.paddle_args_config[i][j].get_paddle_tensor(self.api_config))
                    else:
                        tmp.append(self.paddle_args_config[i][j])
                self.paddle_args.append(tmp)
            elif isinstance(self.paddle_args_config[i], tuple):
                tmp = []
                for j in range(len(self.paddle_args_config[i])):
                    if isinstance(self.paddle_args_config[i][j], TensorConfig):
                        tmp.append(self.paddle_args_config[i][j].get_paddle_tensor(self.api_config))
                    else:
                        tmp.append(self.paddle_args_config[i][j])
                self.paddle_args.append(tuple(tmp))
            else:
                self.paddle_args.append(self.paddle_args_config[i])

        for key, arg_config in self.paddle_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.paddle_kwargs[key] = arg_config.get_paddle_tensor(self.api_config)
            elif isinstance(arg_config, list):
                value = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        value.append(arg_config[i].get_paddle_tensor(self.api_config))
                    else:
                        value.append(arg_config[i])
                self.paddle_kwargs[key] = value
            elif isinstance(arg_config, tuple):
                tmp = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        tmp.append(arg_config[i].get_paddle_tensor(self.api_config))
                    else:
                        tmp.append(arg_config[i])
                self.paddle_kwargs[key] = tuple(tmp)
            else:
                self.paddle_kwargs[key] = arg_config

        for key, arg_config in self.paddle_merged_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.paddle_merged_kwargs[key] = arg_config.get_paddle_tensor(self.api_config)
            elif isinstance(arg_config, list):
                value = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        value.append(arg_config[i].get_paddle_tensor(self.api_config))
                    else:
                        value.append(arg_config[i])
                self.paddle_merged_kwargs[key] = value
            elif isinstance(arg_config, tuple):
                tmp = []
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        tmp.append(arg_config[i].get_paddle_tensor(self.api_config))
                    else:
                        tmp.append(arg_config[i])
                self.paddle_merged_kwargs[key] = tuple(tmp)
            else:
                self.paddle_merged_kwargs[key] = arg_config
        return True

    def copy_torch_input(self):
        args = []
        kwargs = collections.OrderedDict()

        for arg in self.torch_args:
            if isinstance(arg, torch.Tensor):
                args.append(torch.clone(arg))
            elif isinstance(arg, (list, tuple)):
                args.append([
                    torch.clone(x) if isinstance(x, torch.Tensor) else x
                    for x in arg
                ])
            else:
                args.append(arg)

        for key, value in self.torch_kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = torch.clone(value)
            elif isinstance(value, (list, tuple)):
                kwargs[key] = [
                    torch.clone(x) if isinstance(x, torch.Tensor) else x
                    for x in value
                ]
            else:
                kwargs[key] = value

        return args, kwargs
    def _handle_list_or_tuple_torch(self, config_items, is_tuple=False):
        """处理 list 或 tuple """
        tmp = []
        for item in config_items:
            if isinstance(item, (list, tuple)):
                is_nested_tuple = isinstance(item, tuple)
                processed_item = self._handle_list_or_tuple_torch(
                    item, 
                    is_tuple=is_nested_tuple)
            elif isinstance(item, TensorConfig):
                processed_item = item.get_torch_tensor(self.api_config)
            else:
                processed_item = item
            tmp.append(processed_item)
        return tuple(tmp) if is_tuple else tmp

    def gen_torch_input(self):
        self.torch_args = []
        self.torch_kwargs = collections.OrderedDict()
        for arg_config in self.torch_args_config:
            if isinstance(arg_config, TensorConfig):
                self.torch_args.append(arg_config.get_torch_tensor(self.api_config))
            elif isinstance(arg_config, list):
                self.torch_args.append(self._handle_list_or_tuple_torch(arg_config))
            elif isinstance(arg_config, tuple):
                self.torch_args.append(self._handle_list_or_tuple_torch(arg_config, is_tuple=True))
            else:
                self.torch_args.append(arg_config)

        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.torch_kwargs[key] = arg_config.get_torch_tensor(self.api_config)
            elif isinstance(arg_config, list):
                self.torch_kwargs[key] = self._handle_list_or_tuple_torch(arg_config)
            elif isinstance(arg_config, tuple):
                self.torch_kwargs[key] = self._handle_list_or_tuple_torch(arg_config, is_tuple=True)
            else:
                self.torch_kwargs[key] = arg_config

        if self.need_check_grad():
            if (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__") or self.api_config.api_name == "paddle.Tensor.__setitem__":
                self.torch_args, self.torch_kwargs = self.copy_torch_input()

        self.clear_torch_tensor()
        return True

    def np_assert_accuracy(
        self,
        np_paddle,
        np_torch,
        atol,
        rtol,
        api,
    ):
        if np_paddle.dtype == numpy.bool_:
            numpy.testing.assert_equal(np_paddle, np_torch)
            return
        # max_atol_idx = numpy.argmax(numpy.abs(np_paddle - np_torch))
        # np_paddle_flatten = np_paddle.flatten()
        # np_torch_flatten = np_torch.flatten()
        # sub_res = np_paddle_flatten - np_torch_flatten
        # nonzero_idx = numpy.nonzero(np_torch_flatten)
        # sub_res = sub_res.take(nonzero_idx)
        # np_torch_flatten_nonzero = np_torch_flatten.take(nonzero_idx).flatten()
        # np_paddle_flatten_nonzero = np_paddle_flatten.take(nonzero_idx).flatten()
        # if sub_res.size ==0:
        #     max_rtol_idx = 0
        # else:
        #     max_rtol_idx = numpy.argmax(numpy.abs(sub_res / np_torch_flatten_nonzero))
        numpy.testing.assert_allclose(
            np_paddle,
            np_torch,
            atol,
            rtol,
            # err_msg=(
            #     '{api}: compare failed,\n'.format(
            #         api=api,
            #     )
            #     + 'max_atol value, paddle_value: {value_paddle}, torch_value: {value_torch},\n'.format(
            #         value_paddle=str(np_paddle_flatten[max_atol_idx].item()),
            #         value_torch=str(np_torch_flatten[max_atol_idx].item()),
            #     )
            #     + 'max_rtol value , torch_value: {value_paddle}, paddle_value: {value_torch},\n'.format(
            #         value_paddle=str(np_paddle_flatten_nonzero[max_rtol_idx].item()) if max_rtol_idx < len(np_paddle_flatten_nonzero) else '',
            #         value_torch=str(np_torch_flatten_nonzero[max_rtol_idx].item()) if max_rtol_idx < len(np_torch_flatten_nonzero) else '',
            #     )
            # ),
        )

    def test(self):
        pass

    def clear_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_tensor()
            elif isinstance(arg_config, list):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_tensor()
            elif isinstance(arg_config, tuple):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_tensor()
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_paddle_tensor()
            elif isinstance(arg_config, list):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_paddle_tensor()
            elif isinstance(arg_config, tuple):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_paddle_tensor()
        paddle.device.cuda.empty_cache()

    def clear_torch_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_torch_tensor()
            elif isinstance(arg_config, list):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_torch_tensor()
            elif isinstance(arg_config, tuple):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_torch_tensor()
        torch.cuda.empty_cache()

    def clear_numpy_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_numpy_tensor()
            elif isinstance(arg_config, list):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_numpy_tensor()
            elif isinstance(arg_config, tuple):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_numpy_tensor()

    def is_forward_only(self):
        forward_only_apis = [
            "accuracy",
            "accuracy_check",
            "adadelta_",
            "adagrad_",
            "adam_",
            "adamax_",
            "adamw_",
            "add_act_xpu",
            "add_group_norm_silu",
            "add_layernorm_xpu",
            "add_n",
            "addcmul_xpu",
            "all",
            "all_reduce",
            "allclose",
            "any",
            "apply_per_channel_scale",
            "arange",
            "argmax",
            "argmin",
            "asgd_",
            "assign_pos",
            "assign_value",
            "assign_value_",
            "auc",
            "average_accumulates_",
            "barrier",
            "batch_fc",
            "bernoulli",
            "bincount",
            "binomial",
            "bipartite_match",
            "bitwise_and",
            "bitwise_left_shift",
            "bitwise_not",
            "bitwise_invert",
            "bitwise_or",
            "bitwise_right_shift",
            "bitwise_xor",
            "blha_get_max_len",
            "block_multihead_attention_",
            "block_multihead_attention_xpu",
            "bn_act_xpu",
            "box_clip",
            "box_coder",
            "c_allgather",
            "c_allreduce_avg",
            "c_allreduce_max",
            "c_allreduce_min",
            "c_allreduce_prod",
            "c_allreduce_sum",
            "c_broadcast",
            "c_concat",
            "c_identity",
            "c_reduce_avg",
            "c_reduce_max",
            "c_reduce_min",
            "c_reduce_prod",
            "c_reduce_sum",
            "c_reducescatter",
            "c_scatter",
            "c_split",
            "c_sync_calc_stream",
            "c_sync_comm_stream",
            "check_finite_and_unscale_",
            "check_numerics",
            "chunk_eval",
            "class_center_sample",
            "clip_by_norm",
            "coalesce",
            "coalesce_tensor",
            "coalesce_tensor_",
            "conv1d_xpu",
            "conv2d_transpose_bias",
            "conv2d_transpose_xpu",
            "conv2d_xpu",
            "conv3d_implicit_gemm",
            "copy_to",
            "crf_decoding",
            "cross_attention_xpu",
            "ctc_align",
            "data",
            "decayed_adagrad",
            "decode_jpeg",
            "depend",
            "dequantize_abs_max",
            "dequantize_linear",
            "dequantize_log",
            "dequantize_xpu",
            "detection_map",
            "dgc",
            "dgc_momentum",
            "diag_embed",
            "dirichlet",
            "distribute_fpn_proposals",
            "distributed_fused_lamb",
            "distributed_fused_lamb_init",
            "distributed_lookup_table",
            "distributed_push_sparse",
            "dpsgd",
            "edit_distance",
            "eigvals",
            "embedding_grad_dense",
            "embedding_with_eltwise_add_xpu",
            "empty",
            "empty_like",
            "equal",
            "isreal",
            "equal_all",
            "eye",
            "fake_channel_wise_dequantize_max_abs",
            "fake_channel_wise_quantize_abs_max",
            "fake_dequantize_max_abs",
            "fake_quantize_abs_max",
            "fake_quantize_moving_average_abs_max",
            "fake_quantize_range_abs_max",
            "fast_layernorm_xpu",
            "fast_where_xpu",
            "fc",
            "fc_xpu",
            "feed",
            "fetch",
            "floor_divide",
            "__floordiv__",
            "__rfloordiv__",
            "ftrl",
            "full",
            "full_",
            "full_batch_size_like",
            "full_int_array",
            "full_like",
            "full_with_tensor",
            "fused_adam_",
            "fused_bias_act",
            "fused_bias_residual_layernorm",
            "fused_conv2d_add_act",
            "fused_dconv_drelu_dbn",
            "fused_elementwise_add",
            "fused_elementwise_div",
            "fused_elementwise_mul",
            "fused_elementwise_sub",
            "fused_embedding_eltwise_layernorm",
            "fused_fc_elementwise_layernorm",
            "fused_linear_param_grad_add",
            "fused_multi_transformer",
            "fused_multi_transformer_int8_xpu",
            "fused_multi_transformer_xpu",
            "fused_scale_bias_add_relu",
            "fused_scale_bias_relu_conv_bn",
            "fused_token_prune",
            "fusion_group",
            "fusion_gru",
            "fusion_lstm",
            "fusion_repeated_fc_relu",
            "fusion_seqconv_eltadd_relu",
            "fusion_seqexpand_concat_fc",
            "fusion_seqpool_cvm_concat",
            "fusion_squared_mat_sub",
            "fusion_transpose_flatten_concat",
            "gather_tree",
            "gaussian",
            "standard_normal",
            "gemm_epilogue",
            "generate_proposals",
            "generate_sequence_xpu",
            "get_tensor_from_selected_rows",
            "graph_khop_sampler",
            "graph_sample_neighbors",
            "sample_neighbors",
            "greater_equal",
            "greater_than",
            "group_norm_silu_xpu",
            "histogram",
            "increment",
            "indices",
            "is_empty",
            "isclose",
            "isfinite",
            "isinf",
            "isnan",
            "lamb_",
            "lars_momentum",
            "layer_norm_act_xpu",
            "less_equal",
            "less_than",
            "less",
            "limit_by_capacity",
            "linspace",
            "histogram_bin_edges",
            "block_multihead_attention",
            "llm_int8_linear",
            "load_combine",
            "lod_array_length",
            "logical_and",
            "isneginf",
            "isposinf",
            "logical_not",
            "logical_or",
            "logical_xor",
            "diff",
            "logspace",
            "lower",
            "lstsq",
            "mask_adaptive_xpu",
            "masked_multihead_attention_",
            "masked_multihead_attention",
            "matrix_nms",
            "matrix_rank",
            "matrix_rank_tol",
            "memcpy",
            "memcpy_d2h",
            "memcpy_h2d",
            "merge_selected_rows",
            "merged_adam_",
            "merged_momentum_",
            "moe",
            "momentum_",
            "moving_average_abs_max_scale",
            "multi_encoder_xpu",
            "multiclass_nms3",
            "multihead_matmul",
            "multinomial",
            "nadam_",
            "nextafter",
            "nms",
            "nonzero",
            "nop",
            "not_equal",
            "npu_identity",
            "number_count",
            "numel",
            "one_hot",
            "onednn_to_paddle_layout",
            "ones",
            "ones_like",
            "pad2d_xpu",
            "partial_allgather",
            "partial_recv",
            "partial_send",
            "print",
            "prior_box",
            "prune_gate_by_capacity",
            "pull_box_sparse",
            "push_dense",
            "push_sparse_v2",
            "qkv_attention_xpu",
            "qkv_unpack_mha",
            "quantize_linear",
            "quantize_xpu",
            "radam_",
            "randint",
            "random_routing",
            "randperm",
            "read_file",
            "recv_v2",
            "reindex_graph",
            "remainder",
            "rmsprop_",
            "roformer_relative_embedding_xpu",
            "row_conv",
            "rprop_",
            "save_combine",
            "searchsorted",
            "bucketize",
            "seed",
            "self_dp_attention",
            "send_and_recv",
            "send_v2",
            "sequence_mask",
            "sequence_unpad_xpu",
            "sgd_",
            "shadow_feed",
            "shadow_feed_tensors",
            "shape",
            "shard_index",
            "share_data_",
            "sine_pos_xpu",
            "skip_layernorm",
            "sparse_momentum",
            "spatial_transformer_resblock_xpu",
            "squeeze_excitation_block",
            "standard_gamma",
            "tdm_child",
            "tdm_sampler",
            "to_sparse_csr",
            "top_p_sampling",
            "tril_indices",
            "triu_indices",
            "truncated_gaussian_random",
            "uniform",
            "uniform_random_batch_size_like",
            "unique",
            "unique_consecutive",
            "update_loss_scaling_",
            "upper",
            "variable_length_memory_efficient_attention",
            "viterbi_decode",
            "weight_dequantize",
            "weight_only_linear_xpu",
            "weight_quantize",
            "weighted_sample_neighbors",
            "write_to_array",
            "yolo_box",
            "yolo_box_head",
            "yolo_box_post",
            "yolo_box_xpu",
            "zeros",
            "zeros_like",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
            "add_act_xpu",
            "add_layernorm_xpu",
            "addcmul_xpu",
            "blha_get_max_len",
            "block_multihead_attention_",
            "block_multihead_attention_xpu",
            "bn_act_xpu",
            "conv1d_xpu",
            "conv2d_transpose_xpu",
            "conv2d_xpu",
            "cross_attention_xpu",
            "dequantize_xpu",
            "distributed_fused_lamb_init",
            "embedding_with_eltwise_add_xpu",
            "fast_layernorm_xpu",
            "fast_where_xpu",
            "fc",
            "fc_xpu",
            "fp8_fp8_half_gemm_fused",
            "fused_bias_act",
            "fused_bias_residual_layernorm",
            "fused_conv2d_add_act",
            "fused_dconv_drelu_dbn",
            "fused_elementwise_add",
            "fused_elementwise_div",
            "fused_elementwise_mul",
            "fused_elementwise_sub",
            "fused_embedding_eltwise_layernorm",
            "fused_fc_elementwise_layernorm",
            "fused_linear_param_grad_add",
            "fused_multi_transformer_",
            "fused_multi_transformer_int8_xpu",
            "fused_multi_transformer_xpu",
            "fused_scale_bias_add_relu",
            "fused_scale_bias_relu_conv_bn",
            "fused_token_prune",
            "fusion_group",
            "fusion_gru",
            "fusion_lstm",
            "fusion_repeated_fc_relu",
            "fusion_seqconv_eltadd_relu",
            "fusion_seqpool_concat",
            "fusion_seqpool_cvm_concat",
            "fusion_squared_mat_sub",
            "fusion_transpose_flatten_concat",
            "gemm_epilogue",
            "generate_sequence_xpu",
            "group_norm_silu_xpu",
            "layer_norm_act_xpu",
            "layer_norm_relu_xpu",
            "mask_adaptive_xpu",
            "multi_encoder_xpu",
            "multihead_matmul",
            "pad2d_xpu",
            "qkv_attention_xpu",
            "qkv_unpack_mha",
            "quantize_xpu",
            "roformer_relative_embedding_xpu",
            "self_dp_attention",
            "sequence_unpad_xpu",
            "sine_pos_xpu",
            "skip_layernorm",
            "spatial_transformer_resblock_xpu",
            "squeeze_excitation_block",
            "variable_length_memory_efficient_attention",
            "weight_only_linear_xpu",
            "yolo_box_xpu",
            "add_group_norm_silu",
            "fused_embedding_fc_lstm",
            "fused_moe",
            "histogramdd",
            "fused_layer_norm",
            "isin",
            "qr",
            "svd_lowrank",
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
            "__and__",
            "__rand__",
            "__rand__",
            "__or__",
            "__ror__",
            "__ror__",
            "__xor__",
            "__rxor__",
            "__rxor__",
            "__invert__",
            "__lshift__",
            "__rlshift__",
            "__rrshift__",
            "__rshift__",
        ]

        api = self.api_config.api_name[self.api_config.api_name.rindex(".")+1:]

        return api in forward_only_apis

def get_arg(api_config, arg_pos, arg_name, default=None):
    if 0 <= arg_pos < len(api_config.args):
        return api_config.args[arg_pos]
    if arg_name in api_config.kwargs:
        return api_config.kwargs[arg_name]
    return default
