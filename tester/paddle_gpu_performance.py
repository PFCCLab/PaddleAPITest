from __future__ import annotations

import time

import paddle

from .api_config.config_analyzer import TensorConfig
from .api_config.log_writer import write_to_log
from .base import APITestBase


def tensor_numel(tensor_config):
    numel = 1
    for i in tensor_config.shape:
        numel = numel * i
    return numel


def get_tensor_configs(api_config):
    tensor_configs = []
    for arg_config in api_config.args:
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])

    for _key, arg_config in api_config.kwargs.items():
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
    return tensor_configs


def total_numel(api_config):
    tensor_configs = get_tensor_configs(api_config)
    numel = 0
    for tensor_config in tensor_configs:
        numel = numel + tensor_numel(tensor_config)
    return numel


class APITestPaddleGPUPerformance(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)

    def test(self):
        if self.need_skip(paddle_only=True):
            print("[Skip]", flush=True)
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed", flush=True)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err))
            write_to_log("numpy_error", self.api_config.config)
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed", flush=True)
                return
            numel = total_numel(self.api_config)
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
                "faild",
            )
            if self.should_ignore_paddle_error(str(err)):
                return
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
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
                "faild",
            )
            if self.should_ignore_paddle_error(str(err)):
                return
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
