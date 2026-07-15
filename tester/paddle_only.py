from __future__ import annotations

import paddle

from .api_config.log_writer import write_to_log
from .base import APITestBase

# from func_timeout import func_set_timeout


class APITestPaddleOnly(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config, use_torch=False)
        self.test_amp = kwargs.get("test_amp", False)

    # @func_set_timeout(600)
    def test(self):
        if self.need_skip(paddle_only=True):
            print(f"[skip] {self.api_config.config}", flush=True)
            write_to_log("skip", self.api_config.config)
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed", flush=True)
            write_to_log("config_parse", self.api_config.config)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed", flush=True)
                write_to_log("config_input", self.api_config.config)
                return
        except Exception as err:
            print(f"[config_input] {self.api_config.config}\n{err!s}", flush=True)
            write_to_log("config_input", self.api_config.config)
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                write_to_log("paddle_error", self.api_config.config)
                return
            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
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
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            _, fatal = self.report_runtime_error(
                err, "paddle_error", "paddle_only", allow_ignore_paddle=True
            )
            if fatal:
                raise
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            self.report_runtime_error(err, "paddle_cuda", "paddle_only_cuda_check")
            raise

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
        print(f"[pass] {self.api_config.config}", flush=True)
        write_to_log("pass", self.api_config.config)
