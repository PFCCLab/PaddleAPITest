from __future__ import annotations

import collections
import inspect

import numpy
import paddle
import torch
import yaml

from .api_config import USE_CACHED_NUMPY, TensorConfig, cached_numpy
from .api_config.log_writer import log_accuracy_tolerance

with open("tester/base_config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

forward_only_apis = frozenset(config.get("forward_only_apis", []))
handle_axes_api = frozenset(config.get("handle_axes_api", []))
not_check_dtype = frozenset(config.get("not_check_dtype", []))
rand_apis = frozenset(config.get("rand_apis", []))
stochastic_behavior_apis = frozenset(config.get("stochastic_behavior_apis", []))
single_op_no_signature_apis = frozenset(config.get("single_op_no_signature_apis", []))

paddle_error_dismiss = config.get("paddle_error_dismiss", {})
special_accuracy_atol_rtol = config.get("special_accuracy_atol_rtol", {})

with open("tester/api_config/torch_error_skip.txt") as f:
    torch_error_skip = frozenset(line.strip() for line in f if line.strip())

del config

CUDA_ERROR = frozenset(
    [
        "CUDA error",
        "memory corruption",
    ]
)

CUDA_OOM = frozenset(
    [
        "CUDA out of memory",
        "Out of memory error",
    ]
)


def get_arg(api_config, arg_pos, arg_name, default=None):
    if 0 <= arg_pos < len(api_config.args):
        return api_config.args[arg_pos]
    if arg_name in api_config.kwargs:
        return api_config.kwargs[arg_name]
    return default


no_signature_api_mappings = {
    f"paddle.Tensor.{method}": {
        "self": lambda cfg: get_arg(cfg, 0, "self"),
        "y": lambda cfg: get_arg(cfg, 1, "y"),
    }
    for method in single_op_no_signature_apis
}


class APITestBase:
    def __init__(self, api_config):
        self.api_config = api_config
        self.outputs_grad_numpy = []
        torch.set_num_threads(8)
        torch.set_printoptions(threshold=100, linewidth=120)

    def need_skip(self, paddle_only=False):
        # not support
        if "sparse" in self.api_config.api_name:
            return True
        # if self.api_config.api_name in not_support_api:
        #     return True
        # if not paddle_only and self.api_config.api_name in rand_apis:
        #     return True
        # if not paddle_only and self.api_config.api_name in stochastic_behavior_apis:
        #     return True
        if not paddle_only and self.api_config.config in torch_error_skip:
            return True
        for i in range(len(self.api_config.args)):
            if isinstance(self.api_config.args[i], TensorConfig):
                if self.api_config.args[i].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                    return True
            elif isinstance(self.api_config.args[i], list) or isinstance(
                self.api_config.args[i], tuple
            ):
                for j in range(len(self.api_config.args[i])):
                    if isinstance(self.api_config.args[i][j], TensorConfig):
                        if self.api_config.args[i][j].dtype in [
                            "float8_e5m2",
                            "float8_e4m3fn",
                        ]:
                            return True
            elif self.api_config.args[i] in [
                paddle.base.core.DataType.FLOAT8_E4M3FN,
                paddle.base.core.DataType.FLOAT8_E5M2,
                "float8_e5m2",
                "float8_e4m3fn",
            ]:
                return True

        for _key, arg_config in self.api_config.kwargs.items():
            if isinstance(arg_config, TensorConfig):
                if arg_config.dtype in ["float8_e5m2", "float8_e4m3fn"]:
                    return True
            elif isinstance(arg_config, (list, tuple)):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        if arg_config[i].dtype in ["float8_e5m2", "float8_e4m3fn"]:
                            return True
            elif arg_config in [
                paddle.base.core.DataType.FLOAT8_E4M3FN,
                paddle.base.core.DataType.FLOAT8_E5M2,
                "float8_e5m2",
                "float8_e4m3fn",
            ]:
                return True

        return False

    def need_check_grad(self):
        if self.is_forward_only():
            return False

        if self.api_config.api_name == "paddle.assign":
            has_list_arg = len(self.paddle_args_config) and isinstance(
                self.paddle_args_config[0], list
            )
            has_second_arg = (
                len(self.paddle_args_config) > 1 and self.paddle_args_config[1] is not None
            )
            if has_list_arg or has_second_arg:
                return False

        return True

        # This part seems unused in any case:
        #
        # valid_dtypes = {'float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16'}
        # if len(self.api_config.args) > 0 and isinstance(self.api_config.args[0], TensorConfig):
        #     dtype = self.api_config.args[0].dtype
        #     if dtype in valid_dtypes:
        #         return True
        # return True

        # Original implementation:
        #
        # if not self.is_forward_only() and not (self.api_config.api_name == "paddle.assign" and len(self.paddle_args_config) and isinstance(self.paddle_args_config[0], list)) and not (self.api_config.api_name == "paddle.assign" and len(self.paddle_args_config) > 1 and self.paddle_args_config[1] is not None):
        #     if len(self.api_config.args) > 0 and isinstance(self.api_config.args[0], TensorConfig):
        #         dtype = self.api_config.args[0].dtype
        #         if dtype in ['float32', 'float64', 'float16', 'complex64', 'complex128', 'bfloat16']:
        #             return True
        #     return True
        # return False

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

        api_name = self.api_config.api_name
        if api_name == "paddle.Tensor.__getitem__" or api_name == "paddle.Tensor.__setitem__":
            self.torch_args_config = self.api_config.args
            return True

        if api_name not in no_signature_api_mappings:
            # For APIs with signatures, use paddle_sig.bind to get arguments
            paddle_sig = inspect.signature(self.paddle_api)
            paddle_bound_args = paddle_sig.bind(*self.api_config.args, **self.api_config.kwargs)
            paddle_args_dict = paddle_bound_args.arguments
            # fix paddle.arange wrong binding
            if self.api_config.api_name == "paddle.arange":
                # if end is not provided, use the 'start' kwargs as end
                if "end" not in paddle_args_dict:
                    paddle_args_dict["end"] = paddle_args_dict["start"]
                    paddle_args_dict["start"] = 0
        else:
            # For APIs without signatures, use the external mapping dict
            mapping = no_signature_api_mappings[api_name]
            paddle_args_dict = {}
            for key, get_value_func in mapping.items():
                paddle_args_dict[key] = get_value_func(self.api_config)

        self.paddle_merged_kwargs_config = paddle_args_dict
        self.torch_kwargs_config.update(paddle_args_dict)
        self.torch_kwargs_config.pop("name", None)

        return True

    def _handle_list_or_tuple(
        self, config_items, is_tuple=False, index=None, key=None, list_index=None
    ):
        """处理 list 或 tuple"""
        if list_index is None:
            list_index = []
        need_axes_handling = self.api_config.api_name in handle_axes_api
        need_indices_handling = self.api_config.api_name == "paddle.index_put"

        if need_indices_handling and (index == 1 or key == "indices"):
            return self._handle_indices_arg(config_items, is_tuple)
        elif need_axes_handling and (index == 1 or key == "axis"):
            return self._handle_axis_arg(config_items, is_tuple)

        tmp = []
        for i, item in enumerate(config_items):
            current_list_index = [*list_index, i]
            if isinstance(item, (list, tuple)):
                is_nested_tuple = isinstance(item, tuple)
                processed_item = self._handle_list_or_tuple(
                    item,
                    is_tuple=is_nested_tuple,
                    index=index,
                    key=key,
                    list_index=current_list_index,
                )
            elif isinstance(item, TensorConfig):
                processed_item = item.get_numpy_tensor(
                    self.api_config, index=index, key=key, list_index=current_list_index
                )
            else:
                processed_item = item
            tmp.append(processed_item)
        return tuple(tmp) if is_tuple else tmp

    def _handle_axis_arg(self, config_items, is_tuple=False):
        """处理 axis 参数"""
        x = (
            self.paddle_args_config[0]
            if len(self.paddle_args_config) > 0
            else self.paddle_kwargs_config["x"]
        )
        max_dim = max(len(x.shape), 1)  # scalar

        tmp = []
        used_axes = set()
        tensor_configs = []

        for item in config_items:
            if isinstance(item, TensorConfig):
                if item.shape not in [[], [1]] or item.dtype not in ["int32", "int64"]:
                    raise ValueError(
                        f"Invalid TensorConfig for axis: shape {item.shape} or dtype {item.dtype}"
                    )
                tensor_configs.append(item)
                tmp.append(0)  # placeholder
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
                raise ValueError(
                    f"Not enough available dimensions ({len(available_dims)}) for {len(tensor_configs)} TensorConfig items"
                )
            selected_dims = numpy.random.choice(
                available_dims, size=len(tensor_configs), replace=False
            )
            mask = numpy.random.randint(0, 2, size=len(tensor_configs)).astype(bool)
            final_dims = numpy.where(mask, selected_dims - max_dim, selected_dims)
            tensor_idx = 0
            for i, item in enumerate(config_items):
                if isinstance(item, TensorConfig):
                    item.fill_numpy_tensor(final_dims[tensor_idx])
                    tmp[i] = item.get_numpy_tensor(self.api_config)
                    tensor_idx += 1
        return tuple(tmp) if is_tuple else tmp

    def _generate_int_indices(self, item_shape, dim_size):
        num_elements = numpy.prod(item_shape).item()
        if num_elements > dim_size:
            indices_flat = numpy.random.randint(-dim_size, dim_size, size=num_elements)
        else:
            indices_flat = numpy.random.choice(dim_size, size=num_elements, replace=False)
        return indices_flat.reshape(item_shape)

    def _generate_constrained_bool_mask(self, shape, num_true):
        mask_size = numpy.prod(shape).item()
        if mask_size < num_true:
            raise ValueError(
                f"Cannot generate a mask with {num_true} true values in a {mask_size} element mask"
            )
        mask_flat = numpy.zeros(mask_size, dtype="bool")
        true_indices = numpy.random.choice(mask_size, num_true, replace=False)
        mask_flat[true_indices] = True
        return mask_flat.reshape(shape)

    def _broadcast_or_raise(self, shapes):
        return numpy.broadcast_shapes(*[tuple(s) for s in shapes])

    def _handle_indices_arg(self, config_items, is_tuple=False):
        x = (
            self.paddle_args_config[0]
            if len(self.paddle_args_config) > 0
            else self.paddle_kwargs_config["x"]
        )
        value = (
            self.paddle_args_config[2]
            if len(self.paddle_args_config) > 2
            else self.paddle_kwargs_config["value"]
        )
        x_shape = x.shape
        value_shape = value.shape
        int_index_shapes = []
        has_bool_index = False
        dims_consumed = 0
        for item in config_items:
            if item.dtype == "bool":
                b_rank = len(item.shape)
                has_bool_index = True
                dims_consumed += b_rank
            else:
                int_index_shapes.append(tuple(item.shape))
                dims_consumed += 1

        if dims_consumed > len(x_shape):
            raise ValueError(
                f"Too many indices: consume {dims_consumed} dims but x has {len(x_shape)} dims"
            )
        num_true_needed = -1
        num_remaining_dims = len(x_shape) - dims_consumed
        advanced_shape = ()
        if int_index_shapes:
            try:
                advanced_shape = self._broadcast_or_raise(int_index_shapes)
                # give a default 1
                if (
                    has_bool_index
                    and len(value_shape) > num_remaining_dims
                    and advanced_shape[-1] == 1
                    and value_shape[-num_remaining_dims - 1] != 1
                ):
                    advanced_shape = (
                        *advanced_shape[:-1],
                        value_shape[-num_remaining_dims - 1],
                    )
                num_true_needed = advanced_shape[-1]
            except Exception:
                raise ValueError(
                    f"Incompatible integer index shapes for broadcasting: {int_index_shapes}"
                )
        elif has_bool_index:
            if len(value_shape) > num_remaining_dims:
                advanced_shape = (value_shape[0],)
                num_true_needed = value_shape[0]
            else:
                # give a default 1, other valid(not out of bound) shape also can.
                advanced_shape = (1,)
                num_true_needed = 1
        res_dims = advanced_shape + tuple(x_shape[dims_consumed:])
        try:
            # only for checking.
            self._broadcast_or_raise([value_shape, res_dims])
        except ValueError:
            raise ValueError(
                f"Value shape {value_shape} cannot be broadcast to the indexed shape {res_dims}."
            )
        processed_indices = []
        x_dim_cursor = 0
        for item in config_items:
            if item.dtype == "bool":
                if num_true_needed < 0:
                    raise ValueError(
                        "Cannot determine the number of True elements for the boolean mask."
                    )
                item.numpy_tensor = self._generate_constrained_bool_mask(
                    item.shape, num_true_needed
                )
                x_dim_cursor += len(item.shape)
            else:
                x_dim_to_index = x_shape[x_dim_cursor]
                indices = self._generate_int_indices(item.shape, x_dim_to_index)
                item.numpy_tensor = indices.astype(item.dtype)
                x_dim_cursor += 1

            processed_indices.append(item.numpy_tensor)

        return tuple(processed_indices) if is_tuple else processed_indices

    def gen_numpy_input(self):
        for i, arg_config in enumerate(self.paddle_args_config):
            if isinstance(arg_config, (list, tuple)):
                is_tuple = isinstance(arg_config, tuple)
                self._handle_list_or_tuple(arg_config, is_tuple=is_tuple, index=i)
            elif isinstance(arg_config, TensorConfig):
                arg_config.get_numpy_tensor(self.api_config, index=i)
        for key, kwarg_config in self.paddle_kwargs_config.items():
            if isinstance(kwarg_config, (list, tuple)):
                is_tuple = isinstance(kwarg_config, tuple)
                self._handle_list_or_tuple(kwarg_config, is_tuple=is_tuple, key=key)
            elif isinstance(kwarg_config, TensorConfig):
                kwarg_config.get_numpy_tensor(self.api_config, key=key)
        return True

    def _handle_list_or_tuple_paddle(self, config_items, is_tuple=False):
        """处理 list 或 tuple"""
        tmp = []
        for item in config_items:
            if isinstance(item, (list, tuple)):
                is_nested_tuple = isinstance(item, tuple)
                processed_item = self._handle_list_or_tuple_paddle(item, is_tuple=is_nested_tuple)
            elif isinstance(item, TensorConfig):
                processed_item = item.get_paddle_tensor(self.api_config)
                item.clear_paddle_tensor()
            else:
                processed_item = item
            tmp.append(processed_item)
        return tuple(tmp) if is_tuple else tmp

    def gen_paddle_input(self):
        """Generate paddle input by config, for tensor config initlize paddle tensor by get_paddle_tensor()

        be sure to call gen_numpy_input() before use gen_paddle_input() since gen_paddle_input() do not pass index or key to get_paddle_tensor() or get_numpy_tensor() while gen_numpy_input() pass.
        """
        self.paddle_args = []
        self.paddle_kwargs = collections.OrderedDict()
        self.paddle_merged_kwargs = collections.OrderedDict()

        for arg_config in self.paddle_args_config:
            if isinstance(arg_config, TensorConfig):
                self.paddle_args.append(arg_config.get_paddle_tensor(self.api_config))
                arg_config.clear_paddle_tensor()
            elif isinstance(arg_config, (list, tuple)):
                is_tuple = isinstance(arg_config, tuple)
                self.paddle_args.append(self._handle_list_or_tuple_paddle(arg_config, is_tuple))
            else:
                self.paddle_args.append(arg_config)

        for key, kwarg_config in self.paddle_kwargs_config.items():
            if isinstance(kwarg_config, TensorConfig):
                self.paddle_kwargs[key] = kwarg_config.get_paddle_tensor(self.api_config)
                kwarg_config.clear_paddle_tensor()
            elif isinstance(kwarg_config, (list, tuple)):
                is_tuple = isinstance(kwarg_config, tuple)
                self.paddle_kwargs[key] = self._handle_list_or_tuple_paddle(kwarg_config, is_tuple)
            else:
                self.paddle_kwargs[key] = kwarg_config

        if len(self.paddle_args) == 0 and self.api_config.api_name.startswith("paddle.Tensor."):
            self.paddle_args.append(self.paddle_kwargs.popitem(last=False)[1])

        if (
            self.api_config.api_name == "paddle.linalg.lstsq"
            and "gpu" in paddle.device.get_device()
        ):
            if len(self.paddle_args) > 3:
                self.paddle_args[3] = "gels"
            elif "driver" in self.paddle_kwargs:
                self.paddle_kwargs["driver"] = "gels"
        elif self.api_config.api_name == "paddle.nn.functional.cross_entropy":
            use_softmax = get_arg(self.api_config, 8, "use_softmax", True)
            if not use_softmax:
                axis = get_arg(self.api_config, 7, "axis", -1)
                if len(self.paddle_args) > 0:
                    self.paddle_args[0] = paddle.nn.functional.softmax(
                        self.paddle_args[0], axis=axis
                    )
                else:
                    self.paddle_kwargs["input"] = paddle.nn.functional.softmax(
                        self.paddle_kwargs["input"], axis=axis
                    )

        if self.need_check_grad() and (
            (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__")
            or self.api_config.api_name == "paddle.Tensor.__setitem__"
        ):
            self.paddle_args, self.paddle_kwargs = self.copy_paddle_input()

        return True

    def copy_paddle_input(self):
        def _deep_copy(data):
            if isinstance(data, paddle.Tensor):
                return paddle.assign(data)
            elif isinstance(data, (list, tuple)):
                return type(data)(_deep_copy(x) for x in data)
            return data

        args = [_deep_copy(arg) for arg in self.paddle_args]
        kwargs = collections.OrderedDict((k, _deep_copy(v)) for k, v in self.paddle_kwargs.items())
        return args, kwargs

    def get_paddle_input_list(self):
        result = []

        for arg in self.paddle_args:
            if isinstance(arg, paddle.Tensor):
                result.append(arg)
            elif isinstance(arg, (tuple, list)):
                result.extend(item for item in arg if isinstance(item, paddle.Tensor))

        # 按 merged_kwargs 顺序遍历，确保 paddle 关键字参数与 torch 参数顺序一致，避免反向比较无法对应
        # torch 参数顺序通过 paddle_sig.bind 绑定，见 ana_torch_api_info()
        if hasattr(self, "paddle_merged_kwargs_config"):
            for key in self.paddle_merged_kwargs_config:
                if key in self.paddle_kwargs:
                    value = self.paddle_kwargs[key]
                    if isinstance(value, paddle.Tensor):
                        result.append(value)
                    elif isinstance(value, (tuple, list)):
                        result.extend(item for item in value if isinstance(item, paddle.Tensor))
        else:  #  paddle_only
            for key, value in self.paddle_kwargs.items():
                if isinstance(value, paddle.Tensor):
                    result.append(value)
                elif isinstance(value, (tuple, list)):
                    result.extend(item for item in value if isinstance(item, paddle.Tensor))

        return result

    def get_torch_input_list(self):
        result = []
        for i in range(len(self.torch_args)):
            if isinstance(self.torch_args[i], torch.Tensor):
                result.append(self.torch_args[i])
            elif isinstance(self.torch_args[i], (tuple, list)):
                for item in self.torch_args[i]:
                    if isinstance(item, torch.Tensor):
                        result.append(item)

        for _key, value in self.torch_kwargs.items():
            if isinstance(value, torch.Tensor):
                result.append(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        result.append(item)
        return result

    def get_cached_numpy(self, dtype, shape):
        numel = 1
        for i in shape:
            numel = numel * i
        if numel > 4300000000:
            raise RuntimeError(f"Too large tensor to get cached numpy: {numel}")

        start = (4300000000 - numel - 100) if (4300000000 - numel - 100) > 0 else 0
        if dtype in cached_numpy:
            tensor = cached_numpy[dtype][start : start + numel].reshape(shape)
        else:
            if "int" in dtype:
                cached_numpy[dtype] = numpy.random.randint(
                    -65535, 65535, size=4300000000, dtype="int64"
                ).astype(dtype)
                tensor = cached_numpy[dtype][start : start + numel].reshape(shape)
            else:
                cached_numpy[dtype] = (numpy.random.random([4300000000]) - 0.5).astype(dtype)
                tensor = cached_numpy[dtype][start : start + numel].reshape(shape)
        return tensor

    def gen_paddle_output_and_output_grad(self, outputs):
        result_outputs = []
        if isinstance(outputs, paddle.Tensor):
            result_outputs.append(outputs)
        elif isinstance(outputs, list):
            result_outputs = [
                output
                for output in outputs
                if isinstance(output, paddle.Tensor)
                and (output._is_initialized() or output.numel() == 0)
            ]
        elif isinstance(
            outputs, (paddle.autograd.autograd.Hessian, paddle.autograd.autograd.Jacobian)
        ):
            result_outputs.append(outputs[:])
        elif isinstance(outputs, tuple):
            for output in outputs:
                if output is None or (
                    isinstance(output, paddle.Tensor)
                    and not (output._is_initialized() or output.numel() == 0)
                ):
                    continue
                elif isinstance(output, paddle.Tensor):
                    result_outputs.append(output)
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, paddle.Tensor):
                            result_outputs.append(item)
                elif isinstance(
                    output, (paddle.autograd.autograd.Hessian, paddle.autograd.autograd.Jacobian)
                ):
                    result_outputs.extend(output[:])
                elif (
                    isinstance(output, tuple)
                    and len(output) > 0
                    and (
                        isinstance(output[0], paddle.autograd.autograd.Hessian)
                        or isinstance(output[0], paddle.autograd.autograd.Jacobian)
                    )
                ):
                    for lazy_obj in output:
                        result_outputs.append(lazy_obj[:])
                else:
                    raise ValueError("outputs format not support")

        result_outputs_grads = []
        if len(self.outputs_grad_numpy) == 0:
            for output in result_outputs:
                dtype = str(output.dtype).split(".")[-1]
                if USE_CACHED_NUMPY:
                    dtype = "float32" if dtype == "bfloat16" else dtype
                    numpy_tensor = self.get_cached_numpy(dtype, output.shape)
                else:
                    if "int" in dtype:
                        numpy_tensor = (
                            numpy.random.randint(-65535, 65535, size=output.shape)
                        ).astype(dtype)
                    else:
                        dtype = "float32" if dtype == "bfloat16" else dtype
                        numpy_tensor = (numpy.random.random(output.shape) - 0.5).astype(dtype)
                self.outputs_grad_numpy.append(numpy_tensor)
        for i, numpy_tensor in enumerate(self.outputs_grad_numpy):
            dtype = str(result_outputs[i].dtype).split(".")[-1]
            result_output_grad = paddle.to_tensor(
                numpy_tensor,
                dtype=dtype if dtype != "bfloat16" else "float32",
            )
            result_output_grad.stop_gradient = False
            if dtype == "bfloat16":
                result_output_grad = paddle.cast(result_output_grad, dtype="bfloat16")
            result_outputs_grads.append(result_output_grad)
        return result_outputs, result_outputs_grads

    def gen_torch_output_and_output_grad(self, outputs):
        result_outputs = []
        if isinstance(outputs, torch.Tensor):
            result_outputs.append(outputs)
        elif isinstance(outputs, torch.Size):
            result_outputs.append(torch.tensor(outputs))
        elif isinstance(outputs, list):
            result_outputs = [output for output in outputs if isinstance(output, torch.Tensor)]
        elif isinstance(outputs, tuple):
            for output in outputs:
                if output is None:
                    continue
                elif isinstance(output, torch.Tensor):
                    result_outputs.append(output)
                else:
                    raise ValueError("outputs format not support")

        result_outputs_grads = []
        if len(self.outputs_grad_numpy) == 0:
            for output in result_outputs:
                dtype = str(output.dtype).split(".")[-1]
                if USE_CACHED_NUMPY:
                    dtype = "float32" if dtype == "bfloat16" else dtype
                    numpy_tensor = self.get_cached_numpy(dtype, output.shape)
                else:
                    if "int" in dtype:
                        numpy_tensor = (
                            numpy.random.randint(-65535, 65535, size=output.shape)
                        ).astype(dtype)
                    else:
                        dtype = "float32" if dtype == "bfloat16" else dtype
                        numpy_tensor = (numpy.random.random(output.shape) - 0.5).astype(dtype)
                self.outputs_grad_numpy.append(numpy_tensor)
        for i, numpy_tensor in enumerate(self.outputs_grad_numpy):
            dtype = str(result_outputs[i].dtype).split(".")[1]
            result_output_grad = torch.tensor(
                numpy_tensor,
                dtype=self.convert_dtype_to_torch_type(dtype)
                if dtype != "bfloat16"
                else torch.float32,
            )
            if dtype == "bfloat16":
                result_output_grad = result_output_grad.to(dtype=torch.bfloat16)
            result_outputs_grads.append(result_output_grad)
        return result_outputs, result_outputs_grads

    def convert_dtype_to_torch_type(self, dtype):
        # for python built-in types, mappings are int -> torch.int64, bool -> torch.bool, float -> torch.float64, complex -> torch.complex128, None -> None
        if dtype in [
            "float32",
            numpy.float32,
            paddle.float32,
            paddle.base.libpaddle.VarDesc.VarType.FP32,
        ]:
            return torch.float32
        elif dtype in [
            "float16",
            numpy.float16,
            paddle.float16,
            paddle.base.libpaddle.VarDesc.VarType.FP16,
        ]:
            return torch.float16
        elif dtype in [
            "float64",
            "float",
            "double",
            numpy.float64,
            paddle.float64,
            paddle.base.libpaddle.VarDesc.VarType.FP64,
            float,
        ]:
            return torch.float64
        elif dtype in [
            "int16",
            numpy.int16,
            paddle.int16,
            paddle.base.libpaddle.VarDesc.VarType.INT16,
        ]:
            return torch.int16
        elif dtype in [
            "int8",
            numpy.int8,
            paddle.int8,
            paddle.base.libpaddle.VarDesc.VarType.INT8,
        ]:
            return torch.int8
        elif dtype in [
            "bool",
            numpy.bool_,
            paddle.bool,
            paddle.base.libpaddle.VarDesc.VarType.BOOL,
            bool,
        ]:
            return torch.bool
        elif dtype in [
            "bfloat16",
            "uint16",
            numpy.uint16,
            paddle.bfloat16,
            paddle.base.libpaddle.VarDesc.VarType.BF16,
        ]:
            return torch.bfloat16
        elif dtype in [
            "uint8",
            numpy.uint8,
            paddle.uint8,
            paddle.base.libpaddle.VarDesc.VarType.UINT8,
        ]:
            return torch.uint8
        elif dtype in [
            "int32",
            numpy.int32,
            paddle.int32,
            paddle.base.libpaddle.VarDesc.VarType.INT32,
        ]:
            return torch.int32
        elif dtype in [
            "int64",
            "int",
            numpy.int64,
            paddle.int64,
            paddle.base.libpaddle.VarDesc.VarType.INT64,
            int,
        ]:
            return torch.int64
        elif dtype in [
            "complex64",
            numpy.complex64,
            paddle.complex64,
            paddle.base.libpaddle.VarDesc.VarType.COMPLEX64,
        ]:
            return torch.complex64
        elif dtype in [
            "complex128",
            numpy.complex128,
            paddle.complex128,
            paddle.base.libpaddle.VarDesc.VarType.COMPLEX128,
            complex,
        ]:
            return torch.complex128
        elif dtype is None:
            return None
        else:
            raise ValueError(f"Unsupport dtype: {dtype}")

    def gen_paddle_input_with_merged_kwargs(self):
        self.paddle_args = []
        self.paddle_kwargs = collections.OrderedDict()
        self.paddle_merged_kwargs = collections.OrderedDict()

        for i in range(len(self.paddle_args_config)):
            if isinstance(self.paddle_args_config[i], TensorConfig):
                self.paddle_args.append(
                    self.paddle_args_config[i].get_paddle_tensor(self.api_config)
                )
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
        def _deep_copy(data):
            if isinstance(data, torch.Tensor):
                return torch.clone(data)
            elif isinstance(data, (list, tuple)):
                return type(data)(_deep_copy(x) for x in data)
            return data

        args = [_deep_copy(arg) for arg in self.torch_args]
        kwargs = collections.OrderedDict((k, _deep_copy(v)) for k, v in self.torch_kwargs.items())
        return args, kwargs

    def _handle_list_or_tuple_torch(self, config_items, is_tuple=False):
        """处理 list 或 tuple"""
        tmp = []
        for item in config_items:
            if isinstance(item, (list, tuple)):
                is_nested_tuple = isinstance(item, tuple)
                processed_item = self._handle_list_or_tuple_torch(item, is_tuple=is_nested_tuple)
            elif isinstance(item, TensorConfig):
                processed_item = item.get_torch_tensor(self.api_config)
                item.clear_torch_tensor()
            else:
                processed_item = item
            tmp.append(processed_item)
        return tuple(tmp) if is_tuple else tmp

    def gen_torch_input(self):
        """Generate torch input by config, for tensor config initlize torch tensor by get_torch_tensor()

        be sure to call gen_numpy_input() before use gen_torch_input() since gen_torch_input() do not pass index or key to get_torch_tensor() or get_numpy_tensor() while gen_numpy_input() pass.
        """
        self.torch_args = []
        self.torch_kwargs = collections.OrderedDict()
        for arg_config in self.torch_args_config:
            if isinstance(arg_config, TensorConfig):
                self.torch_args.append(arg_config.get_torch_tensor(self.api_config))
                arg_config.clear_torch_tensor()
            elif isinstance(arg_config, (list, tuple)):
                is_tuple = isinstance(arg_config, tuple)
                self.torch_args.append(self._handle_list_or_tuple_torch(arg_config, is_tuple))
            elif isinstance(arg_config, (paddle.dtype, paddle.base.libpaddle.VarDesc.VarType)):
                self.torch_args.append(self.convert_dtype_to_torch_type(arg_config))
            else:
                self.torch_args.append(arg_config)

        for key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                self.torch_kwargs[key] = arg_config.get_torch_tensor(self.api_config)
                arg_config.clear_torch_tensor()
            elif isinstance(arg_config, (list, tuple)):
                is_tuple = isinstance(arg_config, tuple)
                self.torch_kwargs[key] = self._handle_list_or_tuple_torch(arg_config, is_tuple)
            elif (
                isinstance(arg_config, (paddle.dtype, paddle.base.libpaddle.VarDesc.VarType))
                or key == "dtype"
            ):
                self.torch_kwargs[key] = self.convert_dtype_to_torch_type(arg_config)
            else:
                self.torch_kwargs[key] = arg_config

        if self.need_check_grad() and (
            (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__")
            or self.api_config.api_name == "paddle.Tensor.__setitem__"
        ):
            self.torch_args, self.torch_kwargs = self.copy_torch_input()

        torch.cuda.empty_cache()
        return True

    def np_assert_accuracy(self, np_paddle, np_torch, atol=1e-2, rtol=1e-2):
        if np_paddle.dtype == numpy.bool_:
            numpy.testing.assert_equal(np_paddle, np_torch)
            return
        bitwise_alignment = getattr(self, "bitwise_alignment", False)
        if not bitwise_alignment and self.api_config.api_name in special_accuracy_atol_rtol:
            atol, rtol = special_accuracy_atol_rtol[self.api_config.api_name]

        numpy.testing.assert_allclose(
            np_paddle,
            np_torch,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

    def paddle_assert_accuracy(
        self, actual_paddle_tensor, expected_paddle_tensor, atol=1e-2, rtol=1e-2
    ):
        is_check_dtype = self.api_config.api_name not in not_check_dtype

        if not actual_paddle_tensor.is_contiguous():
            actual_paddle_tensor = actual_paddle_tensor.contiguous()
        actual_paddle_tensor = actual_paddle_tensor.cpu().detach()

        if not expected_paddle_tensor.is_contiguous():
            expected_paddle_tensor = expected_paddle_tensor.contiguous()
        expected_paddle_tensor = expected_paddle_tensor.cpu().detach()

        actual_paddle_dlpack = paddle.utils.dlpack.to_dlpack(actual_paddle_tensor)  # type: ignore[reportGeneralTypeIssues]
        torch.utils.dlpack.from_dlpack(actual_paddle_dlpack)  # type: ignore[reportGeneralTypeIssues]

        expected_paddle_dlpack = paddle.utils.dlpack.to_dlpack(expected_paddle_tensor)  # type: ignore[reportGeneralTypeIssues]
        converted_paddle_tensor = torch.utils.dlpack.from_dlpack(expected_paddle_dlpack)  # type: ignore[reportGeneralTypeIssues]

        def error_msg(msg):
            return (
                f"Not equal to tolerance rtol={rtol}, atol={atol}\n"
                f"{msg}\n"
                f"ACTUAL: (shape={converted_paddle_tensor.shape}, dtype={converted_paddle_tensor.dtype})\n"
                f"{converted_paddle_tensor}\n"
                f"DESIRED: (shape={converted_paddle_tensor.shape}, dtype={converted_paddle_tensor.dtype})\n"
                f"{converted_paddle_tensor}"
            )

        bitwise_alignment = getattr(self, "bitwise_alignment", False)
        if not bitwise_alignment and self.api_config.api_name in special_accuracy_atol_rtol:
            atol, rtol = special_accuracy_atol_rtol[self.api_config.api_name]

        try:
            torch.testing.assert_close(
                converted_paddle_tensor,
                converted_paddle_tensor,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                check_dtype=is_check_dtype,
                msg=error_msg,
            )
        except Exception as e:
            error_str = str(e)
            if error_str.startswith("Comparing"):
                print("torch_assert failed, try np_assert", flush=True)
                self.np_assert_accuracy(
                    actual_paddle_tensor.numpy(),
                    expected_paddle_tensor.numpy(),
                    atol,
                    rtol,
                )
            else:
                raise

    def torch_assert_accuracy(self, paddle_tensor, torch_tensor, atol=1e-2, rtol=1e-2):
        is_check_dtype = self.api_config.api_name not in not_check_dtype

        if not paddle_tensor.is_contiguous():
            paddle_tensor = paddle_tensor.contiguous()
        paddle_tensor = paddle_tensor.cpu().detach()

        if not torch_tensor.is_contiguous():
            torch_tensor = torch_tensor.contiguous()
        torch_tensor = torch_tensor.cpu().detach()

        paddle_dlpack = paddle.utils.dlpack.to_dlpack(paddle_tensor)  # type: ignore[reportGeneralTypeIssues]
        converted_paddle_tensor = torch.utils.dlpack.from_dlpack(paddle_dlpack)  # type: ignore[reportGeneralTypeIssues]

        def error_msg(msg):
            return (
                f"Not equal to tolerance rtol={rtol}, atol={atol}\n"
                f"{msg}\n"
                f"ACTUAL: (shape={converted_paddle_tensor.shape}, dtype={converted_paddle_tensor.dtype})\n"
                f"{converted_paddle_tensor}\n"
                f"DESIRED: (shape={torch_tensor.shape}, dtype={torch_tensor.dtype})\n"
                f"{torch_tensor}"
            )

        bitwise_alignment = getattr(self, "bitwise_alignment", False)

        if not bitwise_alignment and self.api_config.api_name in special_accuracy_atol_rtol:
            atol, rtol = special_accuracy_atol_rtol[self.api_config.api_name]
        test_tol = getattr(self, "test_tol", False)
        is_backward = getattr(self, "is_backward", False)
        if test_tol:
            atol, rtol = 0.0, 0.0
        try:
            torch.testing.assert_close(
                converted_paddle_tensor,
                torch_tensor,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                check_dtype=is_check_dtype,
                msg=error_msg,
            )
            if test_tol:
                api_name = self.api_config.api_name
                config = self.api_config.config[:120000]
                log_accuracy_tolerance(
                    "Identical",
                    api_name,
                    config,
                    str(paddle_tensor.dtype),
                    is_backward,
                )
        except Exception as e:
            error_str = str(e)
            if error_str.startswith("Comparing"):
                print("torch_assert failed, try np_assert", flush=True)
                self.np_assert_accuracy(
                    paddle_tensor.numpy(),
                    torch_tensor.numpy(),
                    atol,
                    rtol,
                )
            elif test_tol:
                error_info = error_str.split("\n", maxsplit=2)[1] if "\n" in error_str else None
                if error_info and (
                    error_info.startswith("Tensor-likes") or error_info.startswith("Scalars")
                ):
                    api_name = self.api_config.api_name
                    config = self.api_config.config[:120000]
                    log_accuracy_tolerance(
                        error_str,
                        api_name,
                        config,
                        str(paddle_tensor.dtype),
                        is_backward,
                    )
            else:
                raise

    def test(self):
        pass

    def clear_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for _key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_tensor()
            elif isinstance(arg_config, (list, tuple)):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_tensor()
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

    def clear_paddle_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for _key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_paddle_tensor()
            elif isinstance(arg_config, (list, tuple)):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_paddle_tensor()
        paddle.device.cuda.empty_cache()

    def clear_torch_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for _key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_torch_tensor()
            elif isinstance(arg_config, (list, tuple)):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_torch_tensor()
        torch.cuda.empty_cache()

    def clear_numpy_tensor(self):
        if not hasattr(self, "torch_kwargs_config"):
            return
        for _key, arg_config in self.torch_kwargs_config.items():
            if isinstance(arg_config, TensorConfig):
                arg_config.clear_numpy_tensor()
            elif isinstance(arg_config, (list, tuple)):
                for i in range(len(arg_config)):
                    if isinstance(arg_config[i], TensorConfig):
                        arg_config[i].clear_numpy_tensor()

    def is_forward_only(self):
        api = self.api_config.api_name[self.api_config.api_name.rindex(".") + 1 :]
        return api in forward_only_apis

    def should_ignore_paddle_error(self, error_msg):
        dismiss_errors = paddle_error_dismiss.get(self.api_config.api_name, None)
        if dismiss_errors is None:
            return False
        if isinstance(dismiss_errors, str):
            return dismiss_errors in error_msg
        elif isinstance(dismiss_errors, (list, tuple)):
            return any(error in error_msg for error in dismiss_errors)
        return False

    def should_check_dtype(self):
        return self.api_config.api_name not in not_check_dtype
