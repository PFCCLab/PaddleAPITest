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
    # 这些 case 的合法非有限值可能出现在物化或反向阶段，因此沿用全流程豁免。
    _FULL_SCOPE_NONFINITE_CASES = {
        "explicit_fill",
        "dtype_view",
        "empty_reduction",
        "empty_fused_norm",
    }
    # 内部常量和空 mean 的例外只属于前向，不应影响输入物化或梯度检查。
    _FORWARD_SCOPE_NONFINITE_CASES = {"internal_constant", "empty_loss_mean"}

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

    def _classify_nonfinite_case(self):
        """一次性识别当前 Paddle-only case 的非有限值契约。"""
        # 返回绑定结果供后验复用，避免分类和验证看到不同的默认参数。
        api_name = self.api_config.api_name
        bound = bind_input_parameters(
            api_name,
            self.api_config.args,
            self.api_config.kwargs,
            api=self.paddle_api,
            apply_defaults=True,
        )
        if bound.source == "unresolved":
            # 参数无法可靠绑定时禁止豁免，让原执行路径暴露配置或 Paddle 错误。
            return None, bound
        if api_name in self._INTERNAL_NONFINITE_APIS:
            # 内部常量类别优先，确保 nan_to_num 和 pinv 使用严格的最终输出检查。
            return "internal_constant", bound
        if self._is_valid_empty_mean_loss_case(bound):
            # 空 loss 需要标量 NaN 后验，不能退化为一般的空规约豁免。
            return "empty_loss_mean", bound

        # 仅识别填充值参数，避免将 norm 的 p=math.inf 等控制参数误判为输出契约。
        for name in ("value", "fill_value", "padding_value"):
            value = bound.arguments.get(name)
            # 只接受 Python float，Tensor 值必须继续经过正常的输入数值检查。
            if isinstance(value, float) and not math.isfinite(value):
                return "explicit_fill", bound

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
                return "dtype_view", bound
            if isinstance(target, paddle.base.core.DataType):
                return "dtype_view", bound

        if api_name.endswith(".fused_layer_norm"):
            # 前缀维为空但归一化后缀非空不会形成空统计量，不能获得该豁免。
            x_config = bound.arguments.get("x")
            begin_axis = bound.arguments.get("begin_norm_axis")
            if isinstance(x_config, TensorConfig) and isinstance(begin_axis, int):
                rank = len(x_config.shape)
                axis = begin_axis + rank if begin_axis < 0 else begin_axis
                # 只豁免归一化后缀明确包含 0 的空统计量。
                if 0 <= axis < rank and any(int(dim) == 0 for dim in x_config.shape[axis:]):
                    return "empty_fused_norm", bound

        if api_name in self._EMPTY_REDUCTION_APIS and self._is_empty_reduction(bound.arguments):
            return "empty_reduction", bound
        return None, bound

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

    @staticmethod
    def _assert_allowed_nonfinite(tensors, allowed, api_name):
        # allowed 描述非有限值类型，而不是整体放行输出；有限元素不受该检查影响。
        for tensor in tensors:
            invalid = ~paddle.isfinite(tensor)
            if "nan" in allowed:
                invalid &= ~paddle.isnan(tensor)
            if "posinf" in allowed:
                invalid &= ~paddle.isposinf(tensor)
            if "neginf" in allowed:
                invalid &= ~paddle.isneginf(tensor)
            if bool(paddle.any(invalid).item()):
                raise RuntimeError(f"{api_name} returned an unexpected non-finite output")

    def _validate_nan_to_num_output(self, leaves, bound):
        api_name = self.api_config.api_name
        if len(leaves) != 1:
            # nan_to_num 的公开返回协议是单 Tensor，复合输出视为 Paddle 行为异常。
            raise RuntimeError(f"{api_name} returned an unexpected output structure")
        input_tensor = self.paddle_args[0] if self.paddle_args else self.paddle_kwargs.get("x")
        if not isinstance(input_tensor, paddle.Tensor):
            # 缺少运行时输入时无法验证 replacement 位置，宁可失败也不扩大豁免。
            raise RuntimeError(f"{api_name} input tensor is unavailable for output validation")
        input_masks = {
            "nan": paddle.isnan(input_tensor),
            "posinf": paddle.isposinf(input_tensor),
            "neginf": paddle.isneginf(input_tensor),
        }
        output = leaves[0]
        unexpected = ~paddle.isfinite(output)
        # 非有限 replacement 只能影响输入中对应的 NaN、+Inf 或 -Inf 位置。
        for parameter, input_mask in input_masks.items():
            replacement = bound.arguments.get(parameter)
            if not isinstance(replacement, (int, float)) or math.isfinite(replacement):
                continue
            if math.isnan(replacement):
                output_mask = paddle.isnan(output)
            elif replacement > 0:
                output_mask = paddle.isposinf(output)
            else:
                output_mask = paddle.isneginf(output)
            unexpected &= ~(input_mask & output_mask)
        if bool(paddle.any(unexpected).item()):
            raise RuntimeError(f"{api_name} returned an unexpected non-finite output")

    def _validate_nonfinite_case_output(self, case_kind, bound, paddle_output):
        # 验证只消费分类阶段的 bound，不重新解析配置或扩大豁免范围。
        if case_kind is None or case_kind == "dtype_view":
            # dtype view 的语义就是重解释任意位模式，无法对 NaN/Inf 类型增加约束。
            return
        leaves = list(self._iter_tensor_tree_leaves(paddle_output, tensor_type=paddle.Tensor))
        api_name = self.api_config.api_name
        if case_kind == "empty_loss_mean":
            # 空 mean loss 的契约严格限定为唯一的 0 维 NaN。
            if len(leaves) != 1 or leaves[0].ndim != 0 or not bool(paddle.isnan(leaves[0]).item()):
                raise RuntimeError(f"{api_name} did not return the expected scalar NaN")
            return
        if case_kind == "explicit_fill":
            # 输出只允许出现配置填充值明确指定的 NaN 或对应符号的 Inf。
            allowed = set()
            for name in ("value", "fill_value", "padding_value"):
                value = bound.arguments.get(name)
                if isinstance(value, float) and math.isnan(value):
                    allowed.add("nan")
                elif value == math.inf:
                    allowed.add("posinf")
                elif value == -math.inf:
                    allowed.add("neginf")
            self._assert_allowed_nonfinite(leaves, allowed, api_name)
            return
        if case_kind == "internal_constant":
            if api_name in {"paddle.nan_to_num", "paddle.Tensor.nan_to_num"}:
                self._validate_nan_to_num_output(leaves, bound)
                return
            # pinv 的内部常量可豁免，但最终输出不允许包含任何非有限值。
            self._assert_allowed_nonfinite(leaves, set(), api_name)
            return
        if case_kind == "empty_fused_norm":
            # 空统计量可能产生 NaN，但 Inf 不属于该协议。
            self._assert_allowed_nonfinite(leaves, {"nan"}, api_name)
            return
        if case_kind == "empty_reduction":
            # 每类规约只接受其数学空集结果，防止宽泛豁免隐藏其他非有限值。
            # 空输出没有元素可验证；非空输出中的非有限类型必须匹配 API 身份元。
            short_name = api_name.rsplit(".", 1)[-1]
            if short_name == "logsumexp":
                allowed = {"neginf"}
            elif short_name in {"min", "amin"}:
                allowed = {"posinf"}
            elif short_name in {"max", "amax"}:
                allowed = {"neginf"}
            else:
                allowed = {"nan"}
            self._assert_allowed_nonfinite(leaves, allowed, api_name)

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
            nonfinite_case, bound = self._classify_nonfinite_case()
            # 显式输出和空规约沿用全流程豁免，内部常量与空 loss 仅覆盖前向。
            with self._nan_inf_check_disabled(nonfinite_case in self._FULL_SCOPE_NONFINITE_CASES):
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

                with self._nan_inf_check_disabled(
                    nonfinite_case in self._FORWARD_SCOPE_NONFINITE_CASES
                ):
                    paddle_output = self._run_paddle_forward()
                self._validate_nonfinite_case_output(nonfinite_case, bound, paddle_output)
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
