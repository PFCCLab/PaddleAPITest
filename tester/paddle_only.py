from __future__ import annotations

import paddle

from .base import APITestBase, GpuMemoryGuardSkip

# from func_timeout import func_set_timeout


class APITestPaddleOnly(APITestBase):
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

            paddle_output = self._run_paddle_forward()
            self._run_paddle_backward(paddle_output)
        except GpuMemoryGuardSkip as err:
            self.report_case_result("skip", phase="memory_guard", message=str(err))
            self._finalize_paddle_only("skip")
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
