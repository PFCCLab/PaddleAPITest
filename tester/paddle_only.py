from __future__ import annotations

import contextlib
import itertools
import math

import paddle

from .api_config.parameter_binding import bind_input_parameters
from .base import APITestBase, GpuMemoryGuardSkip
from .input_generation.tensor_config import TensorConfig

# from func_timeout import func_set_timeout


class APITestPaddleOnly(APITestBase):
    # 内部常量白名单只解决 kernel 构造 Inf 的误报，最终输出仍必须满足独立契约。
    _INTERNAL_NONFINITE_APIS = {
        "paddle.nan_to_num",
        "paddle.Tensor.nan_to_num",
        "paddle.linalg.pinv",
        "paddle.Tensor.pinv",
    }
    # 空规约必须同时命中 API 白名单和实际规约轴为 0，未规约的 0 维不在此范围。
    _EMPTY_REDUCTION_APIS = {
        "paddle.amax",
        "paddle.amin",
        "paddle.logsumexp",
        "paddle.median",
        "paddle.mean",
        "paddle.min",
        "paddle.max",
        "paddle.nanmean",
        "paddle.nanmedian",
        "paddle.var",
        "paddle.std",
        "paddle.Tensor.amax",
        "paddle.Tensor.amin",
        "paddle.Tensor.logsumexp",
        "paddle.Tensor.median",
        "paddle.Tensor.mean",
        "paddle.Tensor.min",
        "paddle.Tensor.max",
        "paddle.Tensor.nanmean",
        "paddle.Tensor.nanmedian",
        "paddle.Tensor.var",
        "paddle.Tensor.std",
    }
    # loss 白名单只包含可以从静态 TensorConfig 严格推导逐元素输出规模的 API。
    _EMPTY_MEAN_LOSS_APIS = {
        "paddle.nn.functional.binary_cross_entropy_with_logits",
        "paddle.nn.functional.cross_entropy",
        "paddle.nn.functional.dice_loss",
        "paddle.nn.functional.gaussian_nll_loss",
        "paddle.nn.functional.kl_div",
        "paddle.nn.functional.l1_loss",
        "paddle.nn.functional.margin_ranking_loss",
        "paddle.nn.functional.mse_loss",
        "paddle.nn.functional.poisson_nll_loss",
        "paddle.nn.functional.smooth_l1_loss",
        "paddle.nn.functional.soft_margin_loss",
        "paddle.nn.functional.triplet_margin_with_distance_loss",
    }
    input_operation_mode = "paddle_only"

    def __init__(self, api_config, **kwargs):
        super().__init__(
            api_config,
            use_torch=False,
            runtime_config=kwargs.get("runtime_config"),
        )
        self.test_amp = kwargs.get("test_amp", False)

    # @func_set_timeout(600)
    def _is_mutating_paddle_api(self):
        return (
            self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__"
        ) or self.api_config.api_name == "paddle.Tensor.__setitem__"

    def _get_paddle_output_owner(self):
        if len(self.paddle_args) > 0:
            return self.paddle_args[0]
        return next(iter(self.paddle_kwargs.values()))

    def _invoke_paddle_api(self):
        if self.test_amp:
            with paddle.amp.auto_cast():
                return self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
        return self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)

    def _normalize_paddle_forward_output(self, paddle_output):
        if self._is_mutating_paddle_api():
            return self._get_paddle_output_owner()
        return paddle_output

    def _run_paddle_forward(self):
        self.reset_random_state()
        self.dump_event("paddle_forward_start")
        paddle_output = self._normalize_paddle_forward_output(self._invoke_paddle_api())
        self.dump_save("paddle_forward_output", paddle_output, framework="paddle")
        self.dump_event("paddle_forward_done")
        return paddle_output

    @staticmethod
    def _broadcast_tensor_config_shapes(*configs):
        """校验 TensorConfig 广播协议并返回广播后的静态 shape。"""
        # 动态负维无法证明最终结果为空，因此不得据此关闭数值检查。
        shapes = [tuple(int(dim) for dim in config.shape) for config in configs]
        if not shapes or any(dimension < 0 for shape in shapes for dimension in shape):
            return None
        result = []
        for dimensions in itertools.zip_longest(
            *[reversed(shape) for shape in shapes], fillvalue=1
        ):
            non_unit = {dimension for dimension in dimensions if dimension != 1}
            # 0 与 1 可广播为 0；两个不同的非 1 维度属于非法协议。
            if len(non_unit) > 1:
                return None
            result.append(non_unit.pop() if non_unit else 1)
        return tuple(reversed(result))

    def _is_valid_empty_mean_loss_case(self, bound):
        """仅识别参数协议合法且逐元素 loss 为空的 mean case。"""
        api_name = self.api_config.api_name
        if api_name not in self._EMPTY_MEAN_LOSS_APIS:
            return False
        if bound.source == "unresolved":
            return False
        arguments = bound.arguments
        # dice_loss 的公开语义固定对 batch 求均值，其他 loss 必须显式解析为 mean。
        if api_name.endswith(".dice_loss"):
            if "reduction" in arguments:
                return False
        elif arguments.get("reduction") != "mean":
            return False

        def tensor_config(name):
            value = arguments.get(name)
            return value if isinstance(value, TensorConfig) else None

        def broadcast_result(*names):
            configs = [tensor_config(name) for name in names]
            if any(config is None for config in configs):
                return None
            return self._broadcast_tensor_config_shapes(*configs)

        if api_name.endswith(".cross_entropy"):
            # class 轴不是普通广播轴，hard label 可删除该轴或保留长度 1。
            input_config = tensor_config("input")
            label_config = tensor_config("label")
            if input_config is None or label_config is None or len(input_config.shape) < 2:
                return False
            rank = len(input_config.shape)
            axis = arguments.get("axis", -1)
            if not isinstance(axis, int) or isinstance(axis, bool):
                return False
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank or int(input_config.shape[axis]) <= 0:
                return False
            input_shape = tuple(int(dim) for dim in input_config.shape)
            label_shape = tuple(int(dim) for dim in label_config.shape)
            if arguments.get("soft_label", False):
                output_shape = input_shape if input_shape == label_shape else None
            else:
                squeezed_shape = input_shape[:axis] + input_shape[axis + 1 :]
                retained_axis = (
                    len(label_shape) == rank
                    and label_shape[axis] == 1
                    and all(
                        label_shape[index] == input_shape[index]
                        for index in range(rank)
                        if index != axis
                    )
                )
                output_shape = (
                    squeezed_shape if label_shape == squeezed_shape or retained_axis else None
                )
            weight_value = arguments.get("weight")
            if weight_value is not None:
                weight = tensor_config("weight")
                if weight is None or tuple(weight.shape) != (input_shape[axis],):
                    return False
        elif api_name.endswith(".dice_loss"):
            # dice_loss 固定压缩类别轴，最终 mean 的未规约结果只保留 batch 轴。
            input_config = tensor_config("input")
            label_config = tensor_config("label")
            if input_config is None or label_config is None or len(input_config.shape) < 2:
                return False
            input_shape = tuple(int(dim) for dim in input_config.shape)
            label_shape = tuple(int(dim) for dim in label_config.shape)
            if label_shape != (*input_shape[:-1], 1) or input_shape[-1] <= 0:
                return False
            output_shape = (input_shape[0],)
        elif api_name.endswith(".triplet_margin_with_distance_loss"):
            # 自定义距离函数的输出 shape 无法静态推断，因此不参与豁免。
            if arguments.get("distance_function") is not None:
                return False
            configs = [tensor_config(name) for name in ("input", "positive", "negative")]
            if any(config is None for config in configs):
                return False
            shapes = [tuple(int(dim) for dim in config.shape) for config in configs]
            if len(shapes[0]) < 1 or any(shape != shapes[0] for shape in shapes[1:]):
                return False
            output_shape = shapes[0][:-1]
        else:
            short_name = api_name.rsplit(".", 1)[-1]
            same_shape_names = {
                "binary_cross_entropy_with_logits": ("logit", "label"),
                "kl_div": ("input", "label"),
                "margin_ranking_loss": ("input", "other", "label"),
                "poisson_nll_loss": ("input", "label"),
                "smooth_l1_loss": ("input", "label"),
                "soft_margin_loss": ("input", "label"),
            }
            broadcast_names = {
                "l1_loss": ("input", "label"),
                "mse_loss": ("input", "label"),
            }
            if short_name == "gaussian_nll_loss":
                # variance 只支持同 shape、末维为 1 或省略末维三种专用广播形式。
                input_config = tensor_config("input")
                label_config = tensor_config("label")
                variance_config = tensor_config("variance")
                if input_config is None or label_config is None or variance_config is None:
                    return False
                input_shape = tuple(int(dim) for dim in input_config.shape)
                label_result = self._broadcast_tensor_config_shapes(input_config, label_config)
                variance_shape = tuple(int(dim) for dim in variance_config.shape)
                variance_valid = variance_shape == input_shape or (
                    len(input_shape) > 0
                    and (
                        variance_shape == input_shape[:-1]
                        or variance_shape == (*input_shape[:-1], 1)
                    )
                )
                output_shape = (
                    input_shape if label_result == input_shape and variance_valid else None
                )
            elif short_name in same_shape_names:
                # 这些 kernel 的逐元素协议要求参与计算的主 Tensor 完全同 shape。
                configs = [tensor_config(name) for name in same_shape_names[short_name]]
                if any(config is None for config in configs):
                    return False
                shapes = [tuple(int(dim) for dim in config.shape) for config in configs]
                output_shape = (
                    shapes[0] if all(shape == shapes[0] for shape in shapes[1:]) else None
                )
            else:
                # 仅保留 Paddle 已声明支持广播的逐元素距离 loss。
                output_shape = broadcast_result(*broadcast_names[short_name])

            if output_shape is not None and short_name == "binary_cross_entropy_with_logits":
                # 可选权重不能把原本的空 loss 扩展成不同输出 shape。
                for optional_name in ("weight", "pos_weight"):
                    optional_value = arguments.get(optional_name)
                    if optional_value is None:
                        continue
                    optional = tensor_config(optional_name)
                    if optional is None:
                        return False
                    optional_shape = self._broadcast_tensor_config_shapes(
                        TensorConfig(list(output_shape), optional.dtype), optional
                    )
                    if optional_shape != output_shape:
                        return False

        return output_shape is not None and math.prod(output_shape) == 0

    @staticmethod
    def _is_empty_reduction(arguments):
        # 这里解析的是配置 shape，而不是运行时 Tensor，避免为分类提前物化输入。
        x_config = arguments.get("x")
        if not isinstance(x_config, TensorConfig):
            return False
        rank = len(x_config.shape)
        axis = arguments.get("axis")
        if axis is None or (isinstance(axis, (list, tuple)) and not axis):
            # Paddle 将 None、空 list 和空 tuple 都解释为 reduce-all。
            axes = tuple(range(rank))
        elif isinstance(axis, int) and not isinstance(axis, bool):
            axes = (axis,)
        elif isinstance(axis, (list, tuple)) and all(
            isinstance(item, int) and not isinstance(item, bool) for item in axis
        ):
            axes = tuple(axis)
        else:
            return False
        normalized_axes = tuple(item + rank if item < 0 else item for item in axes)
        # 非法轴和重复轴必须继续由 Paddle 报错，不能被数值豁免覆盖。
        if any(item < 0 or item >= rank for item in normalized_axes) or len(
            set(normalized_axes)
        ) != len(normalized_axes):
            return False
        return any(int(x_config.shape[item]) == 0 for item in normalized_axes)

    def _nonfinite_exemption_scope(self):
        """返回 all、forward 或 None；豁免只控制检查 flag，不跳过测试流程。"""
        api_name = self.api_config.api_name
        bound = bind_input_parameters(
            api_name,
            self.api_config.args,
            self.api_config.kwargs,
            api=self.paddle_api,
            apply_defaults=True,
        )
        if bound.source == "unresolved":
            return None
        # 这些 API 只有在固定参数下才关闭检查，输入、前向和反向仍完整执行。
        if api_name in self._INTERNAL_NONFINITE_APIS:
            return "forward"

        # 显式填充值属于 API 语义；不将 norm 的 p=math.inf 等控制参数误判为填充值。
        for name in ("value", "fill_value", "padding_value"):
            value = bound.arguments.get(name)
            if isinstance(value, float) and not math.isfinite(value):
                return "all"

        if api_name in ("paddle.view", "paddle.Tensor.view"):
            target = bound.arguments.get("shape_or_dtype")
            dtype_names = {
                "bool",
                "uint8",
                "int8",
                "uint16",
                "int16",
                "uint32",
                "int32",
                "uint64",
                "int64",
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "complex64",
                "complex128",
            }
            # 只有位模式重解释允许产生任意浮点位型，普通 shape view 仍保持检查。
            if isinstance(target, str) and target.removeprefix("paddle.") in dtype_names:
                return "all"
            if isinstance(target, paddle.base.core.DataType):
                return "all"

        if api_name.endswith(".fused_layer_norm"):
            x_config = bound.arguments.get("x")
            if isinstance(x_config, TensorConfig) and any(int(dim) == 0 for dim in x_config.shape):
                return "all"
        has_zero_input = any(
            isinstance(value, TensorConfig) and any(int(dim) == 0 for dim in value.shape)
            for value in bound.arguments.values()
        )
        if api_name in self._EMPTY_REDUCTION_APIS and has_zero_input:
            return "all"
        if (
            api_name in self._EMPTY_MEAN_LOSS_APIS
            and bound.arguments.get("reduction") == "mean"
            and has_zero_input
        ):
            return "forward"
        return None

    @contextlib.contextmanager
    def _nan_inf_check_disabled(self, disabled):
        """按调用阶段临时关闭 Paddle 检查，并恢复同一 worker 的原状态。"""
        if not disabled:
            # 普通 case 不读取或写入全局 flag，保持原有错误分类路径。
            yield
            return
        flag_name = "FLAGS_check_nan_inf"
        original_flags = paddle.get_flags([flag_name])
        paddle.set_flags({flag_name: False})
        try:
            yield
        finally:
            # worker 会连续执行 case，异常退出时也必须恢复进入作用域前的状态。
            paddle.set_flags(original_flags)

    def _run_paddle_backward(self, paddle_output):
        if not self.need_check_grad():
            self.dump_event("paddle_backward_skipped")
            return

        self.dump_event("paddle_backward_start")
        inputs_list = self.get_paddle_input_list()
        result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
        self.enforce_paddle_backward_capacity(
            inputs_list,
            result_outputs,
            result_outputs_grads,
        )
        self.dump_save(
            "paddle_backward",
            {
                "inputs": inputs_list,
                "outputs": result_outputs,
                "grad_outputs": result_outputs_grads,
            },
            framework="paddle",
        )
        if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
            input_grads = paddle.grad(
                result_outputs,
                inputs_list,
                grad_outputs=result_outputs_grads,
                allow_unused=True,
            )
            self.dump_save("paddle_input_grads", input_grads, framework="paddle")
        self.dump_event("paddle_backward_done")

    def _finalize_paddle_only(self, status):
        self.clear_runtime_inputs("paddle")
        self.dump_finalize(status)

    def _report_paddle_only_error(
        self,
        err,
        default_log_type,
        phase,
        *,
        allow_ignore_paddle=False,
    ):
        log_type, fatal = self.report_runtime_error(
            err,
            default_log_type,
            phase,
            allow_ignore_paddle=allow_ignore_paddle,
        )
        self._finalize_paddle_only(log_type or default_log_type)
        return log_type, fatal

    def test(self):
        self.dump_event("api_analyze_start", mode="paddle_only")
        if self.need_skip(paddle_only=True):
            self.report_case_result("skip")
            self._finalize_paddle_only("skip")
            return

        if not self.ana_paddle_api_info():
            self.report_case_result("config_parse", "ana_paddle_api_info failed")
            self._finalize_paddle_only("config_parse")
            return
        self.dump_event("api_analyze_done", api_name=self.api_config.api_name)
        if not self.run_gpu_memory_preflight("paddle_only"):
            return

        try:
            self.dump_event("numpy_input_start")
            if not self.generate_input_values():
                self.report_case_result("config_input", "generate_input_values failed")
                self._finalize_paddle_only("config_input")
                return
            self.dump_event("numpy_input_done")
        except Exception as err:
            _, fatal = self._report_paddle_only_error(err, "config_input", "input")
            if fatal:
                raise
            return

        try:
            exemption_scope = self._nonfinite_exemption_scope()
            with self._nan_inf_check_disabled(exemption_scope == "all"):
                self.dump_event("paddle_input_start")
                if not self.build_paddle_input():
                    self.report_case_result("paddle_error", "build_paddle_input failed")
                    self._finalize_paddle_only("paddle_error")
                    return
                self.clear_generated_input_values()
                self.dump_save(
                    "paddle_inputs",
                    {"args": self.paddle_args, "kwargs": self.paddle_kwargs},
                    framework="paddle",
                )
                self.dump_event("paddle_input_done")

                with self._nan_inf_check_disabled(exemption_scope in {"all", "forward"}):
                    paddle_output = self._run_paddle_forward()
                self._run_paddle_backward(paddle_output)
        except GpuMemoryGuardSkip as err:
            self.report_case_result("oom", phase="memory_guard", message=str(err))
            self._finalize_paddle_only("oom")
            return
        except Exception as err:
            _, fatal = self._report_paddle_only_error(
                err,
                "paddle_error",
                "paddle_only",
                allow_ignore_paddle=True,
            )
            if fatal:
                raise
            return

        try:
            self.check_operator_cuda_error()
        except Exception as err:
            _, fatal = self._report_paddle_only_error(
                err,
                "paddle_cuda",
                "paddle_only cuda check",
            )
            if fatal:
                raise
            return

        self.report_case_result("pass")
        self._finalize_paddle_only("pass")
