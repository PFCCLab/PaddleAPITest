from __future__ import annotations

import time
from contextlib import nullcontext

import torch

from .base import APITestBase
from .input_generation.tensor_config import tensor_config_tree_numel
from .paddle_to_torch import ConversionKind, get_converter
from .paddle_to_torch.arguments import bind_paddle_arguments


class APITestTorchGPUPerformance(APITestBase):
    input_operation_mode = "torch_gpu_performance"

    def __init__(self, api_config, **kwargs):
        super().__init__(api_config, runtime_config=kwargs.get("runtime_config"))
        self.test_amp = kwargs.get("test_amp", False)
        self.converter = get_converter()

    def test(self):
        if self.need_skip(paddle_only=True):
            self.report_case_result("skip")
            return

        if not self.ana_api_info():
            self.report_case_result("config_parse", "ana_api_info failed")
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            self.report_case_result("config_convert", f"Conversion failed: {e!s}")
            return
        if convert_result.kind is ConversionKind.UNSUPPORTED:
            self.report_case_result(
                "config_convert",
                f"Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
            )
            return

        try:
            if not self.generate_input_values():
                self.report_case_result("config_input", "generate_input_values failed")
                return
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", "input")
            if fatal:
                raise
            return

        numel = tensor_config_tree_numel(self.api_config.args, self.api_config.kwargs)
        test_loop = 2147483647 * 20 // numel
        combined = ""

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            if not self.build_torch_input():
                self.report_case_result("torch_error", "build_torch_input failed")
                return

            bound_arguments = bind_paddle_arguments(
                self.api_config.api_name,
                self.torch_args,
                self.torch_kwargs,
            )
            context = self.converter.prepare_execution(
                convert_result,
                self.torch_args,
                bound_arguments,
                execution_locals=self._torch_execution_locals(),
            )
            self.converter.run_preprocess(context)

            # Only direct Torch mappings are eligible for performance comparison.
            if convert_result.kind is not ConversionKind.DIRECT:
                combined = "combined"

            amp_context = torch.autocast(device_type="cuda") if self.test_amp else nullcontext()
            with amp_context:
                self.converter.run_core(context)
            self.converter.run_postprocess(context)
            torch_output = self.converter.get_output(context)

            with torch.no_grad():
                amp_context = torch.autocast(device_type="cuda") if self.test_amp else nullcontext()
                with amp_context:
                    torch.cuda.synchronize()
                    start = time.time()
                    self.converter.run_core(context, repeat=test_loop)
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

            del context, convert_result
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
                "failed",
                "\tTorch\t",
                combined,
            )
            _, fatal = self.report_runtime_error(err, "torch_error", "forward")
            if fatal:
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
                "failed",
                "\tTorch\t",
                combined,
            )
            _, fatal = self.report_runtime_error(err, "torch_error", "backward")
            if fatal:
                raise err
            return
