from __future__ import annotations

import time

import paddle

from .base import APITestBase
from .input_generation.materialization import tensor_config_tree_numel


class APITestPaddleGPUPerformance(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(
            api_config,
            use_torch=False,
            runtime_config=kwargs.get("runtime_config"),
        )
        self.test_amp = kwargs.get("test_amp", False)

    def test(self):
        if self.need_skip(paddle_only=True):
            self.report_case_result("skip")
            return

        if not self.ana_paddle_api_info():
            self.report_case_result("config_parse", "ana_paddle_api_info failed")
            return

        try:
            if not self.generate_input_values():
                self.report_case_result("config_input", "generate_input_values failed")
                return
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", self.STAGE_INPUT)
            if fatal:
                raise
            return

        try:
            if not self.build_paddle_input():
                self.report_case_result(
                    "paddle_error", "build_paddle_input failed", stage=self.STAGE_INPUT
                )
                return
            numel = tensor_config_tree_numel(self.api_config.args, self.api_config.kwargs)
            test_loop = 2147483647 * 20 // numel
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
                        paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                        end = time.time()
                        timeused = end - start
                        print(
                            self.api_config.api_name,
                            "\t",
                            self.api_config.config,
                            "\tforward\t",
                            numel,
                            "\t",
                            test_loop,
                            "\t",
                            timeused,
                        )
                else:
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    start = time.time()
                    for _i in range(test_loop):
                        self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    end = time.time()
                    timeused = end - start
                    print(
                        self.api_config.api_name,
                        "\t",
                        self.api_config.config,
                        "\tforward\t",
                        numel,
                        "\t",
                        test_loop,
                        "\t",
                        timeused,
                    )
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            print(
                self.api_config.api_name,
                "\t",
                self.api_config.config,
                "\tforward\t",
                numel,
                "\t",
                test_loop,
                "\t",
                "failed",
            )
            _, fatal = self.report_runtime_error(err, "paddle_error", self.STAGE_PADDLE_FORWARD)
            if fatal:
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
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    start = time.time()
                    for _i in range(test_loop):
                        paddle.grad(
                            result_outputs,
                            inputs_list,
                            grad_outputs=result_outputs_grads,
                            allow_unused=True,
                        )
                    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
                    end = time.time()
                    timeused = end - start
                    print(
                        self.api_config.api_name,
                        "\t",
                        self.api_config.config,
                        "\tbackward\t",
                        numel,
                        "\t",
                        test_loop,
                        "\t",
                        timeused,
                    )
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            print(
                self.api_config.api_name,
                "\t",
                self.api_config.config,
                "\tbackward\t",
                numel,
                "\t",
                test_loop,
                "\t",
                "failed",
            )
            _, fatal = self.report_runtime_error(err, "paddle_error", self.STAGE_PADDLE_BACKWARD)
            if fatal:
                raise err
            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
