from __future__ import annotations

import paddle

from .base import APITestBase

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
    def test(self):
        self.dump_event("api_analyze_start", mode="paddle_only")
        if self.need_skip(paddle_only=True):
            self.report_case_result("skip")
            self.dump_finalize("skip")
            return

        if not self.ana_paddle_api_info():
            self.report_case_result("config_parse", "ana_paddle_api_info failed")
            self.dump_finalize("config_parse")
            return
        self.dump_event("api_analyze_done", api_name=self.api_config.api_name)

        try:
            self.dump_event("numpy_input_start")
            if not self.gen_numpy_input():
                self.report_case_result("config_input", "gen_numpy_input failed")
                self.dump_finalize("config_input")
                return
            self.dump_event("numpy_input_done")
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", "input")
            self.dump_finalize(log_type or "config_input")
            if fatal:
                raise
            return

        try:
            self.dump_event("paddle_input_start")
            if not self.gen_paddle_input():
                self.report_case_result("paddle_error", "gen_paddle_input failed")
                self.dump_finalize("paddle_error")
                return
            self.dump_save(
                "paddle_inputs",
                {"args": self.paddle_args, "kwargs": self.paddle_kwargs},
                framework="paddle",
            )
            self.dump_event("paddle_input_done")

            self.dump_event("paddle_forward_start")
            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            self.dump_save("paddle_forward_output", paddle_output, framework="paddle")
            self.dump_event("paddle_forward_done")

            if self.need_check_grad():
                self.dump_event("paddle_backward_start")
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(
                    paddle_output
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
                if (
                    len(inputs_list) != 0
                    and len(result_outputs) != 0
                    and len(result_outputs_grads) != 0
                ):
                    input_grads = paddle.grad(
                        result_outputs,
                        inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
                    self.dump_save("paddle_input_grads", input_grads, framework="paddle")
                self.dump_event("paddle_backward_done")
            else:
                self.dump_event("paddle_backward_skipped")
        except Exception as err:
            _, fatal = self.report_runtime_error(
                err, "paddle_error", "paddle_only", allow_ignore_paddle=True
            )
            self.dump_finalize("paddle_error")
            if fatal:
                raise
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            self.report_runtime_error(err, "paddle_cuda", "paddle_only cuda check")
            self.dump_finalize("paddle_cuda")
            raise

        self.report_case_result("pass")
        self.dump_finalize("pass")
