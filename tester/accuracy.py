import gc
import traceback

import numpy
import paddle
import torch
# from func_timeout import func_set_timeout

from .api_config.log_writer import write_to_log
from .base import APITestBase
from .paddle_to_torch import get_converter


class APITestAccuracy(APITestBase):
    def __init__(self, api_config, test_amp):
        super().__init__(api_config)
        self.test_amp = test_amp
        self.converter = get_converter()

    #@func_set_timeout(600)
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
        if not convert_result.code or not convert_result.code.is_valid():
            print(f"[paddle_to_torch] No code generated for {self.api_config.api_name}", flush=True)
            write_to_log("paddle_to_torch_failed", self.api_config.config)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
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
                **self.torch_kwargs,
            }

            # convert_result.is_torch_corresponding 为 True 时代表有对应的 Torch API
            # 执行 *_compiled 编译好的代码速度更快
            code = convert_result.code
            if code.preprocess_compiled:
                exec(code.preprocess_compiled, exec_globals, exec_locals)

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
            del exec_globals, exec_locals, output_var, convert_result, code

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
            if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                raise err
            return

        if self.need_check_grad():
            try:
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(torch_output)
                del self.torch_args, self.torch_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    torch_out_grads = torch.autograd.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads
                    )
                    torch_grad_success = True
                else:
                    torch_out_grads = None
                    torch_grad_success = False
                del inputs_list, result_outputs, result_outputs_grads
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(str(err), flush=True)
                torch_grad_success = False
                if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                    raise err
        else:
            del self.torch_args, self.torch_kwargs

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

        gc.collect()
        torch.cuda.empty_cache()

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return
            if "paddle.Tensor." in self.api_config.api_name:
                api = getattr(self.paddle_args[0], self.api_config.api_name[self.api_config.api_name.rindex(".")+1:])
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
                else:
                    paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
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
                raise err
            if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                raise err
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err), flush=True)
            write_to_log("paddle_error", self.api_config.config)
            return
        
        if self.api_config.api_name == "paddle.incubate.nn.functional.fused_rms_norm": 
            paddle_output = paddle_output[0]

        if self.api_config.api_name == "paddle.unique": 
            if "return_index=True" in self.api_config.config:
                paddle_output = list(paddle_output)
                del paddle_output[1]

        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                try:
                    if paddle_output.dtype == paddle.bfloat16:
                        paddle_output = paddle.cast(paddle_output, dtype="float32")
                        torch_output = torch_output.to(dtype=torch.float32)
                    # self.np_assert_accuracy(paddle_output.numpy(), torch_output.numpy(), 1e-2, 1e-2, self.api_config)
                    self.torch_assert_accuracy(paddle_output, torch_output, 1e-2, 1e-2)
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
                    # self.np_assert_accuracy(paddle_output.numpy(), torch_output.numpy(), 1e-2, 1e-2, self.api_config)
                    self.torch_assert_accuracy(paddle_output, torch_output, 1e-2, 1e-2)
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
                flag = False
                if isinstance(paddle_output[i], (tuple,list)) and isinstance(torch_output[i], (tuple,list)):
                    flag =True
                    for item in paddle_output[i]:
                        if not isinstance(item, paddle.Tensor):
                            flag = False
                            break
                    for item in torch_output[i]:
                        if not isinstance(item, torch.Tensor):
                            flag = False
                            break
                if flag:
                    try:
                        for index, item_paddle in enumerate(paddle_output[i]):
                            item_torch = torch_output[i][index]
                            if item_paddle.dtype == paddle.bfloat16:
                                item_paddle = paddle.cast(item_paddle, dtype="float32")
                                item_torch = item_torch.to(dtype=torch.float32)
                            # self.np_assert_accuracy(item_paddle.numpy(), item_torch.cpu().detach().numpy(), 1e-2, 1e-2, self.api_config)
                            self.torch_assert_accuracy(item_paddle, item_torch, 1e-2, 1e-2)
                    except Exception as err:
                        print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                        write_to_log("accuracy_error", self.api_config.config)
                        return                    
                else:
                    if isinstance(paddle_output[i], int):
                        self.np_assert_accuracy(numpy.array(paddle_output[i]), numpy.array(torch_output[i]), 1e-2, 1e-2, self.api_config)
                    elif 'tolist' in self.api_config.api_name:
                        self.np_assert_accuracy(numpy.array(paddle_output[i]), numpy.array(torch_output[i]), 1e-2, 1e-2, self.api_config)
                    elif not isinstance(paddle_output[i], paddle.Tensor):
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
                            # self.np_assert_accuracy(paddle_output[i].numpy(), torch_output[i].numpy(), 1e-2, 1e-2, self.api_config)
                            self.torch_assert_accuracy(paddle_output[i], torch_output[i], 1e-2, 1e-2)
                        except Exception as err:
                            print("[accuracy error]", self.api_config.config, "\n", str(err), flush=True)
                            write_to_log("accuracy_error", self.api_config.config)
                            return

        if self.need_check_grad() and torch_grad_success:
            try:
                paddle_out_grads = None
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                del self.paddle_args, self.paddle_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    paddle_out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
                del inputs_list, result_outputs, result_outputs_grads
            except Exception as err:
                print("[paddle error]", self.api_config.config, "\n", str(err), flush=True)
                write_to_log("paddle_error", self.api_config.config)
                if "CUDA error" in str(err) or "memory corruption" in str(err):
                    raise err
                if "CUDA out of memory" in str(err) or "Out of memory error" in str(err):
                    raise err
                return

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print("[cuda error]", self.api_config.config, "\n", str(err), flush=True)
                write_to_log("paddle_error", self.api_config.config)
                return

            if self.api_config.api_name == "paddle.Tensor.__setitem__":
                torch_out_grads = torch_out_grads[0]
                paddle_out_grads = paddle_out_grads[0]
            
            # All configs that not compared with torch 
            # should be moved to tester/api_config/5_accuracy/grads_diff.txt
            if self.api_config.api_name == "paddle.tensordot":
                paddle_out_grads = paddle_out_grads[:2]
                
            if isinstance(paddle_out_grads, paddle.Tensor):
                if isinstance(torch_out_grads, torch.Tensor):
                    try:
                        if paddle_out_grads.dtype == paddle.bfloat16:
                            paddle_out_grads = paddle.cast(paddle_out_grads, dtype="float32")
                            torch_out_grads = torch_out_grads.to(dtype=torch.float32)
                        # self.np_assert_accuracy(paddle_out_grads.numpy(), torch_out_grads.numpy(), 1e-2, 1e-2, self.api_config)
                        self.torch_assert_accuracy(paddle_out_grads, torch_out_grads, 1e-2, 1e-2)
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
                    if isinstance(paddle_out_grads[i], int):
                        self.np_assert_accuracy(numpy.array(paddle_out_grads[i]), numpy.array(torch_out_grads[i]), 1e-2, 1e-2, self.api_config)
                    elif not isinstance(paddle_out_grads[i], paddle.Tensor):
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
                            # self.np_assert_accuracy(paddle_out_grads[i].numpy(), torch_out_grads[i].numpy(), 1e-2, 1e-2, self.api_config)
                            self.torch_assert_accuracy(paddle_out_grads[i], torch_out_grads[i], 1e-2, 1e-2)
                        except Exception as err:
                            print("[accuracy error] backward ", self.api_config.config, "\n", str(err), flush=True)
                            write_to_log("accuracy_error", self.api_config.config)
                            return

        print("[Pass]", self.api_config.config, flush=True)
        write_to_log("pass", self.api_config.config)
