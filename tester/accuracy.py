import gc
import traceback

import paddle
import torch
from func_timeout import func_set_timeout

from .api_config.log_writer import write_to_log
from .base import APITestBase
from .paddle_to_torch import get_converter


class APITestAccuracy(APITestBase):
    def __init__(self, api_config, test_amp):
        super().__init__(api_config)
        self.test_amp = test_amp
        self.converter = get_converter()
    
    @func_set_timeout(600)
    def test(self):
        
        if self.need_skip():
            print("[Skip]", flush=True)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(f"[paddle_to_torch] Convertion failed for {self.api_config.config}: {str(e)}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(f"[paddle_to_torch] Unsupported API {self.api_config.api_name}: {convert_result.error_message}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return
        if not convert_result.code or not convert_result.compiled_code:
            print(f"[paddle_to_torch] No code generated for {self.api_config.api_name}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err))
            write_to_log("paddle_error", self.api_config.config)
            return

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return
        
            # torch_args 与 torch_kwargs 是尚未映射的 torch 参数（即按 paddle 的参数顺序与关键字排列的 torch tensors）
            # 以下代码等价于:
            # torch_output = Paddle2TorchConverter.execute(convert_result, self.torch_args, self.torch_kwargs)
            # 准备执行环境，将参数(torch tensors)直接映射至locals
            exec_globals = {"torch": torch}
            exec_locals = {
                "args": self.torch_args,
                "kwargs": self.torch_kwargs,
                "result": None,
                **self.torch_kwargs
            }

            if self.test_amp:
                with torch.autocast(device_type="cuda"):
                    exec(convert_result.compiled_code, exec_globals, exec_locals)
            else:
                exec(convert_result.compiled_code, exec_globals, exec_locals)

            output_var = convert_result.output_var or "result"
            torch_output = exec_locals[output_var]

            # if "paddle.Tensor." in self.api_config.api_name:
            #     api = getattr(self.torch_args[0], self.torch_api_str[self.torch_api_str.rindex(".")+1:])
            #     args = []
            #     if len(self.torch_args) > 1:
            #         args = self.torch_args[1:]
            #     if self.test_amp:
            #         with torch.autocast(device_type="cuda"):
            #             torch_output = api(*tuple(args), **self.torch_kwargs)
            #     else:
            #         torch_output = api(*tuple(args), **self.torch_kwargs)
            #     del args
            # else:
            #     if self.test_amp:
            #         with torch.autocast(device_type="cuda"):
            #             torch_output = self.torch_api(*tuple(self.torch_args), **self.torch_kwargs)
            #     else:
            #         torch_output = self.torch_api(*tuple(self.torch_args), **self.torch_kwargs)
            # if (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__") or self.api_config.api_name == "paddle.Tensor.__setitem__":
            #     torch_output = self.torch_args[0] if len(self.torch_args) > 0 else next(iter(self.torch_kwargs.values()))
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[torch error]", self.api_config.config, flush=True)
            traceback.print_exc()
            write_to_log("torch_error", self.api_config.config)
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise Exception(err)
            return

        if self.need_check_grad():
            inputs_list = self.get_torch_input_list()
            result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(torch_output)
            try:
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    del self.torch_args
                    del self.torch_kwargs
                    torch_out_grads = torch.autograd.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads)
                    torch_grad_success = True
                else:
                    torch_grad_success = False
                    torch_out_grads = None
                    del self.torch_args
                    del self.torch_kwargs

                del inputs_list
                del result_outputs
                del result_outputs_grads
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(str(err), flush=True)
                torch_grad_success = False
                if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                    raise Exception(err)
        else:
            del self.torch_args
            del self.torch_kwargs

        if isinstance(torch_output, torch.Tensor):
            if torch_output.dtype == torch.bfloat16:
                torch_output = torch_output.to(dtype=torch.float32)
            torch_output = torch_output.cpu().detach()
        elif isinstance(torch_output, (torch.return_types.max, torch.return_types.min)):
            torch_output = torch_output.values
            if torch_output.dtype == torch.bfloat16:
                torch_output = torch_output.to(dtype=torch.float32)
            torch_output = torch_output.cpu().detach()
        elif isinstance(torch_output, (list, tuple)):
            if isinstance(torch_output, tuple):
                torch_output = list(torch_output)
            for i in range(len(torch_output)):
                if isinstance(torch_output[i], torch.Tensor):
                    if torch_output[i].dtype == torch.bfloat16:
                        torch_output[i] = torch_output[i].to(dtype=torch.float32)
                    torch_output[i] = torch_output[i].cpu().detach()

        if self.need_check_grad() and torch_grad_success:
            if isinstance(torch_out_grads, torch.Tensor):
                if torch_out_grads.dtype == torch.bfloat16:
                    torch_out_grads = torch_out_grads.to(dtype=torch.float32)
                torch_out_grads = torch_out_grads.cpu().detach()
            elif isinstance(torch_out_grads, (torch.return_types.max, torch.return_types.min)):
                torch_out_grads = torch_out_grads.values
                if torch_out_grads.dtype == torch.bfloat16:
                    torch_out_grads = torch_out_grads.to(dtype=torch.float32)
                torch_out_grads = torch_out_grads.cpu().detach()
            elif isinstance(torch_out_grads, (list, tuple)):
                if isinstance(torch_out_grads, tuple):
                    torch_out_grads = list(torch_out_grads)
                for i in range(len(torch_out_grads)):
                    if isinstance(torch_out_grads[i], torch.Tensor):
                        if torch_out_grads[i].dtype == torch.bfloat16:
                            torch_out_grads[i] = torch_out_grads[i].to(dtype=torch.float32)
                        torch_out_grads[i] = torch_out_grads[i].cpu().detach()

        self.clear_torch_tensor()
        gc.collect()
        torch.cuda.empty_cache()

        try:
            if "paddle.Tensor." in self.api_config.api_name:
                api = getattr(self.paddle_args[0], self.api_config.api_name[self.api_config.api_name.rindex(".")+1:])
                args = []
                if len(self.paddle_args) > 1:
                    args = self.paddle_args[1:]

                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = api(*tuple(args), **self.paddle_kwargs)
                else:
                    paddle_output = api(*tuple(args), **self.paddle_kwargs)
            else:
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
                else:
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            if (self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__") or self.api_config.api_name == "paddle.Tensor.__setitem__":
                paddle_output = self.paddle_args[0] if len(self.paddle_args) > 0 else next(iter(self.paddle_kwargs.values()))
        except Exception as err:
            print("[paddle error]", self.api_config.config, "\n", str(err), flush=True)
            write_to_log("paddle_error", self.api_config.config)
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise Exception(err)
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err), flush=True)
            write_to_log("paddle_error", self.api_config.config)
            return

        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                try:
                    if paddle_output.dtype == paddle.bfloat16:
                        paddle_output = paddle.cast(paddle_output, dtype="float32")
                        torch_output = torch_output.to(dtype=torch.float32)
                    self.np_assert_accuracy(paddle_output.numpy(), torch_output.numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            elif isinstance(torch_output, bool):
                try:
                    assert paddle_output.dtype == paddle.bool, "paddle_output dtype is not bool"
                    assert paddle_output.shape == [], "paddle_output shape is not []"
                    assert bool(paddle_output) == torch_output, f"paddle_output{bool(paddle_output)} is not equal to torch_output{torch_output}"
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            elif isinstance(torch_output, (torch.return_types.max, torch.return_types.min)):
                try:
                    torch_output = torch_output.values
                    if paddle_output.dtype == paddle.bfloat16:
                        paddle_output = paddle.cast(paddle_output, dtype="float32")
                        torch_output = torch_output.to(dtype=torch.float32)
                    self.np_assert_accuracy(paddle_output.numpy(), torch_output.numpy(), 1e-2, 1e-2, self.api_config)
                except Exception as err:
                    print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            else:
                print("[accuracy error]", self.api_config.config, "\n[output type diff error1], ", type(torch_output), flush=True)
                write_to_log("accuracy_error", self.api_config.config)
                return
        elif isinstance(paddle_output, (list, tuple)):
            if isinstance(paddle_output, tuple):
                paddle_output = list(paddle_output)
            if not isinstance(torch_output, (list, tuple)):
                print("[output type diff error]", self.api_config.config, flush=True)
                return
            if isinstance(torch_output, tuple):
                torch_output = list(torch_output)
            if len(paddle_output) != len(torch_output):
                print("[accuracy error]", self.api_config.config, "\n[output type diff error2], ", len(paddle_output), len(torch_output), flush=True)
                write_to_log("accuracy_error", self.api_config.config)
                return
            for i in range(len(paddle_output)):
                if not isinstance(paddle_output[i], paddle.Tensor):
                    print("[not compare] ", paddle_output[i], torch_output[i], flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
                elif not isinstance(torch_output[i], torch.Tensor):
                    print("[accuracy error]", self.api_config.config, "\n[output type diff error3], ", type(torch_output[i]), flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
                else:
                    try:
                        if paddle_output[i].dtype == paddle.bfloat16:
                            paddle_output[i] = paddle.cast(paddle_output[i], dtype="float32")
                            torch_output[i] = torch_output[i].to(dtype=torch.float32)
                        self.np_assert_accuracy(paddle_output[i].numpy(), torch_output[i].numpy(), 1e-2, 1e-2, self.api_config)
                    except Exception as err:
                        print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                        write_to_log("accuracy_error", self.api_config.config)
                        return

        if self.need_check_grad() and torch_grad_success:
            paddle_out_grads = None
            inputs_list = self.get_paddle_input_list()
            result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
            try:
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    del self.paddle_args
                    del self.paddle_kwargs
                    paddle_out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
                del inputs_list
                del result_outputs
                del result_outputs_grads
            except Exception as err:
                print("[paddle error]", self.api_config.config, "\n", str(err), flush=True)
                write_to_log("paddle_error", self.api_config.config)
                if "CUDA error" in str(err) or "memory corruption" in str(err):
                    raise Exception(err)
                return

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print("[cuda error]", self.api_config.config, "\n", str(err), flush=True)
                write_to_log("paddle_error", self.api_config.config)
                return

            if isinstance(paddle_out_grads, paddle.Tensor):
                if isinstance(torch_out_grads, torch.Tensor):
                    try:
                        if paddle_out_grads.dtype == paddle.bfloat16:
                            paddle_out_grads = paddle.cast(paddle_out_grads, dtype="float32")
                            torch_out_grads = torch_out_grads.to(dtype=torch.float32)
                        self.np_assert_accuracy(paddle_out_grads.numpy(), torch_out_grads.numpy(), 1e-2, 1e-2, self.api_config)
                    except Exception as err:
                        print("[accuracy error] backward ", self.api_config.config, "\n", str(err), flush=True)
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                else:
                    print("[accuracy error] backward ", self.api_config.config, "\n[output type diff error1], ", type(torch_out_grads), flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            elif isinstance(paddle_out_grads, (list, tuple)):
                if isinstance(paddle_out_grads, tuple):
                    paddle_out_grads = list(paddle_out_grads)
                if not isinstance(torch_out_grads, (list, tuple)):
                    print("[output type diff error]", self.api_config.config, flush=True)
                    return
                if isinstance(torch_out_grads, tuple):
                    torch_out_grads = list(torch_out_grads)
                if len(paddle_out_grads) != len(torch_out_grads):
                    print("[accuracy error] backward ", self.api_config.config, "\n[output type diff error2], ", len(paddle_out_grads), len(torch_out_grads), flush=True)
                    write_to_log("accuracy_error", self.api_config.config)
                    return
                for i in range(len(paddle_out_grads)):
                    if not isinstance(paddle_out_grads[i], paddle.Tensor):
                        print("[not compare] ", paddle_out_grads[i], torch_out_grads[i], flush=True)
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                    elif not isinstance(torch_out_grads[i], torch.Tensor):
                        print("[accuracy error] backward ", self.api_config.config, "\n[output type diff error3], ", type(torch_out_grads[i]), flush=True)
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                    else:
                        try:
                            if paddle_out_grads[i].dtype == paddle.bfloat16:
                                paddle_out_grads[i] = paddle.cast(paddle_out_grads[i], dtype="float32")
                                torch_out_grads[i] = torch_out_grads[i].to(dtype=torch.float32)
                            self.np_assert_accuracy(paddle_out_grads[i].numpy(), torch_out_grads[i].numpy(), 1e-2, 1e-2, self.api_config)
                        except Exception as err:
                            print("[accuracy error] backward ", self.api_config.config, "\n", str(err), flush=True)
                            write_to_log("accuracy_error", self.api_config.config)
                            return

        print("[Pass]", self.api_config.config, flush=True)
        write_to_log("pass", self.api_config.config)
