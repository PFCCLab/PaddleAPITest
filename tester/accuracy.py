from __future__ import annotations

import gc
import traceback

import numpy
import paddle
import torch

from .api_config.log_writer import write_to_log
from .base import CUDA_ERROR, CUDA_OOM, APITestBase
from .paddle_to_torch import get_converter

# from func_timeout import func_set_timeout


class APITestAccuracy(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config)
        self.test_amp = kwargs.get("test_amp", False)
        self.atol = kwargs.get("atol", 1e-2)
        self.rtol = kwargs.get("rtol", 1e-2)
        self.test_tol = kwargs.get("test_tol", False)
        self.exit_on_error = kwargs.get("exit_on_error", False)
        self.bitwise_alignment = kwargs.get("bitwise_alignment", False)
        if self.test_tol:
            torch.set_printoptions(profile="short")
        self.converter = get_converter()

    # @func_set_timeout(600)
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
            print(f"[numpy error] {self.api_config.config}\n{err!s}")
            traceback.print_exc()
            write_to_log("numpy_error", self.api_config.config)
            return

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
            print(f"[torch error] {self.api_config.config}\n{err!s}", flush=True)
            traceback.print_exc()
            write_to_log("torch_error", self.api_config.config)
            if any(cuda_err in str(err) for cuda_err in CUDA_ERROR) or any(
                cuda_err in str(err) for cuda_err in CUDA_OOM
            ):
                raise
            return

        torch_grad_success = False
        torch_out_grads = None
        if self.need_check_grad():
            try:
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(
                    torch_output
                )
                del self.torch_args, self.torch_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    torch_out_grads = torch.autograd.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                    )
                    torch_grad_success = True
                del inputs_list, result_outputs, result_outputs_grads
            except Exception as err:
                if str(err).startswith("Too large tensor to get cached numpy: "):
                    print(f"[numpy error] {self.api_config.config}\n{err!s}")
                    write_to_log("numpy_error", self.api_config.config)
                    return
                # some torch backward error can be tolerable, so we catch cuda error here
                if any(cuda_err in str(err) for cuda_err in CUDA_ERROR) or any(
                    cuda_err in str(err) for cuda_err in CUDA_OOM
                ):
                    print(
                        f"[torch error] backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("torch_error", self.api_config.config)
                    raise
                print(str(err), flush=True)
            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(
                    f"[torch error] backward {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                write_to_log("torch_error", self.api_config.config)
                raise
        else:
            del self.torch_args, self.torch_kwargs

        def process_torch_outputs(obj):
            if isinstance(obj, (torch.return_types.max, torch.return_types.min)):
                obj = obj.values
            if isinstance(obj, torch.Tensor):
                obj = obj.cpu().detach()
            elif isinstance(obj, (list, tuple)):
                obj = list(obj)
                for i in range(len(obj)):
                    if isinstance(obj[i], torch.Tensor):
                        obj[i] = obj[i].cpu().detach()
            return obj

        torch_output = process_torch_outputs(torch_output)
        if torch_grad_success:
            torch_out_grads = process_torch_outputs(torch_out_grads)

        gc.collect()
        torch.cuda.empty_cache()

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return
            if "paddle.Tensor." in self.api_config.api_name:
                api = getattr(
                    self.paddle_args[0],
                    self.api_config.api_name[self.api_config.api_name.rindex(".") + 1 :],
                )
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
                else:
                    paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
            else:
                if self.test_amp:
                    with paddle.amp.auto_cast():
                        paddle_output = self.paddle_api(
                            *tuple(self.paddle_args), **self.paddle_kwargs
                        )
                else:
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            if (
                self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__"
            ) or self.api_config.api_name == "paddle.Tensor.__setitem__":
                paddle_output = (
                    self.paddle_args[0]
                    if len(self.paddle_args) > 0
                    else next(iter(self.paddle_kwargs.values()))
                )
        except Exception as err:
            if self.should_ignore_paddle_error(str(err)):
                print(f"[Pass] {self.api_config.config}", flush=True)
                write_to_log("pass", self.api_config.config)
                return
            if any(cuda_err in str(err) for cuda_err in CUDA_ERROR):
                print(f"[cuda error] {self.api_config.config}\n{err!s}", flush=True)
                write_to_log("cuda_error", self.api_config.config)
                raise
            if any(cuda_err in str(err) for cuda_err in CUDA_OOM):
                print(f"[oom] {self.api_config.config}\n{err!s}", flush=True)
                write_to_log("oom", self.api_config.config)
                raise
            print(f"[paddle error] {self.api_config.config}\n{err!s}", flush=True)
            write_to_log("paddle_error", self.api_config.config)
            if self.exit_on_error:
                raise
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print(f"[cuda error] {self.api_config.config}\n{err!s}", flush=True)
            write_to_log("cuda_error", self.api_config.config)
            raise

        paddle_output, torch_output = process_output(self.api_config, paddle_output, torch_output)

        self.is_backward = False

        def compare_paddle_and_torch(paddle_tensor, torch_tensor, idx=0) -> bool:
            try:
                # if paddle_tensor.dtype == paddle.bfloat16:
                #     paddle_tensor = paddle.cast(paddle_tensor, dtype="float32")
                # if torch_tensor.dtype == torch.bfloat16:
                #     torch_tensor = torch_tensor.to(dtype=torch.float32)
                # self.np_assert_accuracy(paddle_tensor.numpy(), torch_tensor.numpy(), atol=self.atol, rtol=self.rtol)
                self.torch_assert_accuracy(
                    paddle_tensor, torch_tensor, atol=self.atol, rtol=self.rtol
                )
            except Exception as err:
                if self.is_backward:
                    print(
                        f"[accuracy error] backward at {idx} {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                else:
                    print(
                        f"[accuracy error] at {idx} {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                write_to_log("accuracy_error", self.api_config.config)
                if self.exit_on_error:
                    raise
                return False
            return True

        # Forward output check:
        if isinstance(paddle_output, paddle.Tensor):
            if isinstance(torch_output, torch.Tensor):
                if not compare_paddle_and_torch(paddle_output, torch_output):
                    return
            elif isinstance(torch_output, bool):
                try:
                    assert paddle_output.dtype == paddle.bool, "paddle_output dtype is not bool"
                    assert paddle_output.shape == [], "paddle_output shape is not []"
                    assert bool(paddle_output) == torch_output, (
                        f"paddle_output {bool(paddle_output)} is not equal to torch_output {torch_output}"
                    )
                except Exception as err:
                    print(
                        f"[not compare] {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            elif isinstance(torch_output, (torch.return_types.max, torch.return_types.min)):
                torch_output = torch_output.values
                if not compare_paddle_and_torch(paddle_output, torch_output):
                    return
            else:
                print(
                    f"[not compare] {self.api_config.config}\n"
                    f"torch is {type(torch_output)} but paddle is {type(paddle_output)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return
        elif isinstance(paddle_output, (list, tuple)):
            if not isinstance(torch_output, (list, tuple)):
                print(
                    f"[not compare] {self.api_config.config}\n"
                    f"torch is {type(torch_output)} but paddle is {type(paddle_output)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return
            paddle_output = list(paddle_output)
            torch_output = list(torch_output)
            if len(paddle_output) != len(torch_output):
                print(
                    f"[not compare] {self.api_config.config}\n"
                    f"torch len is {len(torch_output)} but paddle len is {len(paddle_output)}",
                    flush=True,
                )
                write_to_log("accuracy_error", self.api_config.config)
                return
            for i, (paddle_item, torch_item) in enumerate(
                zip(paddle_output, torch_output, strict=False)
            ):
                if isinstance(paddle_item, int) or self.api_config.api_name.endswith("tolist"):
                    self.np_assert_accuracy(
                        numpy.array(paddle_item),
                        numpy.array(torch_item),
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                # especially for paddle.vision.ops.distribute_fpn_proposals
                elif isinstance(paddle_item, list) and isinstance(torch_item, list):
                    if any(isinstance(x, paddle.Tensor) for x in paddle_item) and any(
                        isinstance(x, torch.Tensor) for x in torch_item
                    ):
                        for paddle_item_sub, torch_item_sub in zip(
                            paddle_item, torch_item, strict=False
                        ):
                            if not compare_paddle_and_torch(paddle_item_sub, torch_item_sub, i):
                                return
                    else:
                        print(
                            f"[not compare] at {i} {self.api_config.config}\n"
                            f"torch is {type(torch_item)} but paddle is {type(paddle_item)}",
                            flush=True,
                        )
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                elif (
                    paddle_item is None
                    or (
                        isinstance(paddle_item, paddle.Tensor)
                        and not (paddle_item._is_initialized() or paddle_item.numel() == 0)
                    )
                ) and torch_item is None:
                    # paddle is None and torch is None
                    # paddle is Tensor but uninitialized and torch is None
                    pass
                elif not isinstance(paddle_item, paddle.Tensor) or not isinstance(
                    torch_item, torch.Tensor
                ):
                    print(
                        f"[not compare] at {i} {self.api_config.config}\n"
                        f"torch is {type(torch_item)} but paddle is {type(paddle_item)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
                else:
                    if not compare_paddle_and_torch(paddle_item, torch_item, i):
                        return

        # Forward check now pass.
        # Then do paddle backward and backward result check.
        if torch_grad_success:
            self.is_backward = True
            try:
                paddle_out_grads = None
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(
                    paddle_output
                )
                del self.paddle_args, self.paddle_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    paddle_out_grads = paddle.grad(
                        result_outputs,
                        inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
                del inputs_list, result_outputs, result_outputs_grads
            except Exception as err:
                if str(err).startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[numpy error] backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("numpy_error", self.api_config.config)
                    return
                if self.should_ignore_paddle_error(str(err)):
                    print(f"[Pass] {self.api_config.config}", flush=True)
                    write_to_log("pass", self.api_config.config)
                    return
                if any(cuda_err in str(err) for cuda_err in CUDA_ERROR):
                    print(
                        f"[cuda error] backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("cuda_error", self.api_config.config)
                    raise
                if any(cuda_err in str(err) for cuda_err in CUDA_OOM):
                    print(
                        f"[oom] backward {self.api_config.config}\n{err!s}",
                        flush=True,
                    )
                    write_to_log("oom", self.api_config.config)
                    raise
                print(
                    f"[paddle error] backward {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                if self.exit_on_error:
                    raise
                write_to_log("paddle_error", self.api_config.config)
                return

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(
                    f"[cuda error] backward {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                write_to_log("cuda_error", self.api_config.config)
                raise

            paddle_out_grads, torch_out_grads = process_grad_output(
                self.api_config, paddle_out_grads, torch_out_grads
            )

            # Backward output check:
            if isinstance(paddle_out_grads, paddle.Tensor):
                if isinstance(torch_out_grads, torch.Tensor):
                    if not compare_paddle_and_torch(paddle_out_grads, torch_out_grads):
                        return
                else:
                    print(
                        f"[not compare] backward {self.api_config.config}\n"
                        f"torch is {type(torch_out_grads)} but paddle is {type(paddle_out_grads)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
            elif isinstance(paddle_out_grads, (list, tuple)):
                if not isinstance(torch_out_grads, (list, tuple)):
                    print(
                        f"[not compare] backward {self.api_config.config}\n"
                        f"torch is {type(torch_out_grads)} but paddle is {type(paddle_out_grads)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
                paddle_out_grads = list(paddle_out_grads)
                torch_out_grads = list(torch_out_grads)
                if len(paddle_out_grads) != len(torch_out_grads):
                    print(
                        f"[not compare] backward {self.api_config.config}\n"
                        f"torch len is {len(torch_out_grads)} but paddle len is {len(paddle_out_grads)}",
                        flush=True,
                    )
                    write_to_log("accuracy_error", self.api_config.config)
                    return
                for i, (paddle_item, torch_item) in enumerate(
                    zip(paddle_out_grads, torch_out_grads, strict=False)
                ):
                    if isinstance(paddle_item, int):
                        self.np_assert_accuracy(
                            numpy.array(paddle_item),
                            numpy.array(torch_item),
                            atol=self.atol,
                            rtol=self.rtol,
                        )
                    elif (
                        paddle_item is None
                        or (
                            isinstance(paddle_item, paddle.Tensor)
                            and not (paddle_item._is_initialized() or paddle_item.numel() == 0)
                        )
                    ) and torch_item is None:
                        # paddle is None and torch is None
                        # paddle is Tensor but uninitialized and torch is None
                        pass
                    elif not isinstance(paddle_item, paddle.Tensor) or not isinstance(
                        torch_item, torch.Tensor
                    ):
                        print(
                            f"[not compare] backward at {i} {self.api_config.config}\n"
                            f"torch is {type(torch_item)} but paddle is {type(paddle_item)}",
                            flush=True,
                        )
                        write_to_log("accuracy_error", self.api_config.config)
                        return
                    else:
                        if not compare_paddle_and_torch(paddle_item, torch_item, i):
                            return

        print(f"[Pass] {self.api_config.config}", flush=True)
        write_to_log("pass", self.api_config.config)


def process_output(api_config, paddle_output, torch_output):
    if api_config.api_name == "paddle.unique":
        if "return_index=True" in api_config.config:
            paddle_output = list(paddle_output)
            paddle_output.pop(1)
    elif api_config.api_name in {
        "paddle.mode",
        "paddle.Tensor.mode",
        "paddle.incubate.nn.functional.fused_layer_norm",
        "paddle.incubate.nn.functional.fused_rms_norm",
        "paddle.kthvalue",
        "paddle.Tensor.kthvalue",
        "paddle.topk",
    }:
        paddle_output = paddle_output[:1]
        torch_output = torch_output[:1]
    elif api_config.api_name in {
        "paddle.strided_slice",
        "paddle.vander",
    }:
        if any(s < 0 for s in paddle_output.strides):
            # torch's from_dlpack now don't support negative strides
            paddle_output = paddle_output.contiguous()
    elif api_config.api_name == "paddle.linalg.eigh":
        # The output of eigen vectors are not unique, because multiplying an eigen vector by -1 in the real case
        # or by e^(i*\theta) in the complex case produces another set of valid eigen vectors of the matrix.
        # So we test whether the elements of each coef_vector (i.e. paddle_output / torch_output for each eigen vector)
        # are all the same and whether the |coef| == 1 for simplicity.
        paddle_output, torch_output = list(paddle_output), list(torch_output)
        eigvector_len = paddle_output[1].shape[-2]
        paddle_eigvectors = paddle_output.pop(1).matrix_transpose().reshape([-1, eigvector_len])
        torch_eigvectors = torch_output.pop(1).transpose(-1, -2).reshape((-1, eigvector_len))
        paddle_output, torch_output = [], []
        for i in range(paddle_eigvectors.shape[0]):
            coef_vector = paddle.to_tensor(
                paddle_eigvectors[i].numpy() / torch_eigvectors[i].numpy(),
                dtype=paddle_eigvectors[i].dtype,
            )
            coef_vector = coef_vector.round(2)
            coef_0 = paddle_eigvectors[i].numpy()[0] / torch_eigvectors[i].numpy()[0]
            coef_vector_approx = torch.tensor([coef_0] * eigvector_len)
            abs_coef = coef_vector.abs().astype("float64")[0]
            one = torch.tensor(1.0, dtype=torch.float64)
            paddle_output.append([coef_vector, abs_coef])
            torch_output.append([coef_vector_approx, one])
    return paddle_output, torch_output


def process_grad_output(api_config, paddle_out_grads, torch_out_grads):
    # All configs that not compared with torch should be copied
    # to tester/api_config/5_accuracy/accuracy_gpu_error_grads_diff.txt
    if api_config.api_name in {
        "paddle.nn.functional.scaled_dot_product_attention",
    }:
        paddle_out_grads = paddle_out_grads[:3]
        torch_out_grads = torch_out_grads[:3]
    elif api_config.api_name in {
        "paddle.lerp",
        "paddle.tensordot",
    }:
        paddle_out_grads = paddle_out_grads[:2]
        torch_out_grads = torch_out_grads[:2]
    elif api_config.api_name in {
        "paddle.Tensor.__setitem__",
        "paddle.Tensor.fill_diagonal_tensor",
        "paddle.diagonal_scatter",
        "paddle.incubate.softmax_mask_fuse",
        "paddle.nn.functional.binary_cross_entropy",
        "paddle.nn.functional.binary_cross_entropy_with_logits",
        "paddle.nn.functional.cross_entropy",
        "paddle.nn.functional.gaussian_nll_loss",
        "paddle.nn.functional.kl_div",
        "paddle.nn.functional.sigmoid_focal_loss",
        "paddle.scale",
    }:
        paddle_out_grads = paddle_out_grads[:1]
        torch_out_grads = torch_out_grads[:1]
    elif api_config.api_name in {
        "paddle.combinations",
        "paddle.nn.utils.parameters_to_vector",
        "paddle.cdist",
    }:
        paddle_out_grads = []
        torch_out_grads = []
    elif api_config.api_name == "paddle.linalg.cholesky_solve":
        if len(api_config.args) > 2:
            is_upper = api_config.args[2]
        elif "is_upper" in api_config.kwargs:
            is_upper = api_config.kwargs["is_upper"]
        else:
            is_upper = False
        torch_out_grads[1] = (
            torch.triu(torch_out_grads[1]) if is_upper else torch.tril(torch_out_grads[1])
        )
    elif api_config.api_name == "paddle.incubate.nn.functional.fused_rotary_position_embedding":
        # Paddle only has 3 outputs/grads Q, K, V
        valid_out_num = len([out for out in paddle_out_grads if out is not None])
        paddle_out_grads = paddle_out_grads[:valid_out_num]
        torch_out_grads = torch_out_grads[:valid_out_num]
    return paddle_out_grads, torch_out_grads
