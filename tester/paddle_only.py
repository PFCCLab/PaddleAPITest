from __future__ import annotations

import contextlib
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
    # loss 白名单只限制 API 范围，是否豁免再由 reduction 和空输入配置决定。
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
        stage,
    ):
        log_type, fatal = self.report_runtime_error(
            err,
            default_log_type,
            stage,
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
            _, fatal = self._report_paddle_only_error(
                err,
                "config_input",
                self.STAGE_INPUT,
            )
            if fatal:
                raise
            return

        # 豁免范围在阶段边界内恢复，避免吞掉后续阶段的数值检查。
        exemption_scope = self._nonfinite_exemption_scope()
        # 输入失败必须单独归类，否则会被误报为前向执行失败。
        try:
            with self._nan_inf_check_disabled(exemption_scope == "all"):
                self.dump_event("paddle_input_start")
                if not self.build_paddle_input():
                    self.report_case_result(
                        "paddle_error",
                        "build_paddle_input failed",
                        stage=self.STAGE_INPUT,
                    )
                    self._finalize_paddle_only("paddle_error")
                    return
                self.clear_generated_input_values()
                self.dump_save(
                    "paddle_inputs",
                    {"args": self.paddle_args, "kwargs": self.paddle_kwargs},
                    framework="paddle",
                )
                self.dump_event("paddle_input_done")
        except Exception as err:
            _, fatal = self._report_paddle_only_error(
                err,
                "paddle_error",
                self.STAGE_INPUT,
            )
            if fatal:
                raise
            return

        # 前向异常不应与反向梯度异常共享同一个错误阶段。
        try:
            with self._nan_inf_check_disabled(exemption_scope in {"all", "forward"}):
                paddle_output = self._run_paddle_forward()
        except Exception as err:
            _, fatal = self._report_paddle_only_error(
                err,
                "paddle_error",
                self.STAGE_PADDLE_FORWARD,
            )
            if fatal:
                raise
            return

        # 反向阶段单独处理显存保护异常和框架异常。
        try:
            with self._nan_inf_check_disabled(exemption_scope == "all"):
                self._run_paddle_backward(paddle_output)
        except GpuMemoryGuardSkip as err:
            self.report_case_result(
                "oom",
                stage=self.STAGE_PADDLE_BACKWARD,
                message=str(err),
            )
            self._finalize_paddle_only("oom")
            return
        except Exception as err:
            _, fatal = self._report_paddle_only_error(
                err,
                "paddle_error",
                self.STAGE_PADDLE_BACKWARD,
            )
            if fatal:
                raise
            return

        # 同步错误保留所属方向，便于定位异步 CUDA 错误。
        try:
            self.check_operator_cuda_error()
        except Exception as err:
            _, fatal = self._report_paddle_only_error(
                err,
                "paddle_cuda",
                self.STAGE_PADDLE_BACKWARD_SYNC
                if self.need_check_grad()
                else self.STAGE_PADDLE_FORWARD_SYNC,
            )
            if fatal:
                raise
            return

        self.report_case_result("pass")
        self._finalize_paddle_only("pass")
