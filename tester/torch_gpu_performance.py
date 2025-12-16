from __future__ import annotations

import time

import torch

from .api_config.config_analyzer import TensorConfig
from .api_config.log_writer import write_to_log
from .base import APITestBase
from .paddle_to_torch import get_converter


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


class APITestTorchGPUPerformance(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.converter = get_converter()

    def test(self):
        if self.need_skip(paddle_only=True):
            print("[Skip]", flush=True)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(
                f"[paddle_to_torch] Convertion failed for {self.api_config.config}: {e!s}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(
                f"[paddle_to_torch] Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.code or not convert_result.code.is_valid():
            print(
                f"[paddle_to_torch] No code generated for {self.api_config.api_name}",
                flush=True,
            )
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                return
        except Exception as err:
            print("[numpy error]", self.api_config.config, "\n", str(err))
            write_to_log("numpy_error", self.api_config.config)
            return

        numel = total_numel(self.api_config)
        test_loop = 2147483647 * 20 // numel
        combined = ""

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return

            # torch_args 与 torch_kwargs 是尚未映射的 torch 参数（即按 paddle 的参数顺序与关键字排列的 torch tensors）
            # (弃用)以下代码等价于:
            # torch_output = Paddle2TorchConverter.execute(convert_result, self.torch_args, self.torch_kwargs)
            # 准备执行环境，将参数(torch tensors)直接映射至locals()
            exec_globals = {"torch": torch}
            exec_locals = {
                "args": self.torch_args,
                "kwargs": self.torch_kwargs,
                "result": None,
                **self.torch_kwargs,
            }
            if self.api_config.api_name == "paddle.nn.functional.rnnt_loss":
                if paddle.device.get_device() == "cpu":
                    exec_locals["fused_log_softmax"] = False

            # convert_result.is_torch_corresponding 为 True 时代表有对应的 Torch API
            # 执行 *_compiled 编译好的代码速度更快，定位 compile error 时可删去 _compiled
            code = convert_result.code
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)

            if not convert_result.is_torch_corresponding:
                combined = "combined"

            if code.core_compiled:
                if self.test_amp:
                    with torch.autocast(device_type="cuda"):
                        exec(code.core_compiled, exec_globals, exec_locals)
                else:
                    exec(code.core_compiled, exec_globals, exec_locals)

            if code.postprocess_compiled:
                exec(code.postprocess_compiled, exec_globals, exec_locals)
            output_var = convert_result.output_var or "result"
            torch_output = exec_locals[output_var]

            with torch.no_grad():
                if code.core_compiled:
                    if self.test_amp:
                        with torch.autocast(device_type="cuda"):
                            torch.cuda.synchronize()
                            start = time.time()
                            for i in range(test_loop):
                                exec(code.core_compiled, exec_globals, exec_locals)
                            torch.cuda.synchronize()
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
                                "\tTorch\t",
                                combined,
                            )
                    else:
                        torch.cuda.synchronize()
                        start = time.time()
                        for i in range(test_loop):
                            exec(code.core_compiled, exec_globals, exec_locals)
                        torch.cuda.synchronize()
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
                            "\tTorch\t",
                            combined,
                        )

            del exec_globals, exec_locals, output_var, convert_result, code
        except Exception as err:
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
                "\tTorch\t",
                combined,
            )
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return

        try:
            if self.need_check_grad():
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(
                    torch_output
                )
                del self.torch_args, self.torch_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    torch.cuda.synchronize()
                    start = time.time()
                    for _i in range(test_loop):
                        torch.autograd.grad(
                            outputs=result_outputs,
                            inputs=inputs_list,
                            grad_outputs=result_outputs_grads,
                            retain_graph=True,
                        )
                    torch.cuda.synchronize()
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
                        "\tTorch\t",
                        combined,
                    )
                del inputs_list, result_outputs, result_outputs_grads, torch_output
            else:
                del self.torch_args, self.torch_kwargs, torch_output
        except Exception as err:
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
                "\tTorch\t",
                combined,
            )
            print(str(err))
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return
