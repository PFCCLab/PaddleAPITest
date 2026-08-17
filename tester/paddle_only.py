from __future__ import annotations

import math

import paddle

from .api_config.parameter_binding import bind_input_parameters
from .base import APITestBase, GpuMemoryGuardSkip

# from func_timeout import func_set_timeout


class APITestPaddleOnly(APITestBase):
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

    def _check_internal_nonfinite_api_output(self, paddle_output):
        allows_internal_constants = self._allows_internal_nonfinite_constants()
        allows_empty_mean = self._allows_empty_mean_loss()
        if not allows_internal_constants and not allows_empty_mean:
            return
        if allows_empty_mean:
            leaves = list(self._iter_tensor_tree_leaves(paddle_output, tensor_type=paddle.Tensor))
            # 豁免契约只接受唯一的 0 维 NaN；Inf、有限值和非标量结果都暴露为框架错误。
            if len(leaves) != 1 or leaves[0].ndim != 0 or not bool(paddle.isnan(leaves[0]).item()):
                raise RuntimeError(
                    f"{self.api_config.api_name} did not return the expected scalar NaN"
                )
            return
        if self.api_config.api_name in {"paddle.nan_to_num", "paddle.Tensor.nan_to_num"}:
            bound = bind_input_parameters(
                self.api_config.api_name,
                self.api_config.args,
                self.api_config.kwargs,
                api=getattr(self, "paddle_api", None),
            )
            for name in ("nan", "posinf", "neginf"):
                replacement = bound.arguments.get(name)
                if isinstance(replacement, (int, float)) and not math.isfinite(replacement):
                    # 调用者明确要求保留非有限 replacement 时，非有限最终结果属于 API 合法语义。
                    return
        # flag 恢复后检查白名单 API 的最终结果，pinv 不允许任何非有限输出例外。
        for tensor in self._iter_tensor_tree_leaves(paddle_output, tensor_type=paddle.Tensor):
            if not bool(paddle.all(paddle.isfinite(tensor)).item()):
                raise RuntimeError(f"{self.api_config.api_name} returned a non-finite output")

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
            # 合法的无穷填充和 0size 规约需覆盖输入物化及前后向全过程。
            with self._nan_inf_check_scope():
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

                # 白名单 API 内部会短暂构造 Inf，豁免严格限定在本次前向生命周期。
                with self._nan_inf_check_scope(
                    allow_internal_nonfinite_constants=True,
                    allow_empty_mean_loss=True,
                ):
                    paddle_output = self._run_paddle_forward()
                self._check_internal_nonfinite_api_output(paddle_output)
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
