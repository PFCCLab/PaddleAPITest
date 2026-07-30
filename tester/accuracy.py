from __future__ import annotations

import gc

import numpy
import paddle
import torch
import yaml

from .base import APITestBase, gpu_mode_maybe_empty_cache
from .paddle_to_torch import adaptive_workspace_bytes, get_converter


def _paddle_tensor_to_torch(value):
    return torch.utils.dlpack.from_dlpack(
        paddle.utils.dlpack.to_dlpack(value.detach())  # type: ignore[reportGeneralTypeIssues]
    )


class APITestAccuracy(APITestBase):
    def __init__(self, api_config, **kwargs):
        super().__init__(api_config, runtime_config=kwargs.get("runtime_config"))
        self.test_amp = kwargs.get("test_amp", False)
        self.atol = kwargs.get("atol", 0)
        self.rtol = kwargs.get("rtol", 0)
        self.test_tol = kwargs.get("test_tol", False)
        self.exit_on_error = kwargs.get("exit_on_error", self.runtime_config.exit_on_error)
        self.bitwise_alignment = kwargs.get(
            "bitwise_alignment", self.runtime_config.bitwise_alignment
        )
        self.use_gpu_mode = self.gpu_mode_config.enabled
        self.manual_threshold_config_file = kwargs.get("manual_threshold_config_file", "")
        self.manual_threshold_config = self._load_manual_threshold_config(
            self.manual_threshold_config_file
        )
        if self.test_tol:
            torch.set_printoptions(profile="short")
        self.converter = get_converter()

    def _load_manual_threshold_config(self, manual_threshold_config_file):
        if not manual_threshold_config_file:
            return {}
        with open(manual_threshold_config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config.get("manual_threshold_config") or {}

    def get_atol(self):
        api_name = (
            self.paddle_args[0]
            if self.api_config.api_name == "paddle._C_ops._run_custom_op"
            else self.api_config.api_name
        )
        threshold = self.manual_threshold_config.get(api_name)
        if threshold is not None:
            return threshold[0]
        return self.atol

    def get_rtol(self):
        api_name = (
            self.paddle_args[0]
            if self.api_config.api_name == "paddle._C_ops._run_custom_op"
            else self.api_config.api_name
        )
        threshold = self.manual_threshold_config.get(api_name)
        if threshold is not None:
            return threshold[1]
        return self.rtol

    def _should_spill_torch_result_tree(
        self, convert_result, torch_output, torch_out_grads, probe_bytes
    ):
        retained_tree_bytes = self.tensor_tree_nbytes((torch_output, torch_out_grads))
        reference_workspace_bytes = self._reference_workspace_bytes(convert_result)
        return gpu_mode_maybe_empty_cache(
            self.gpu_mode_config,
            request_spill=True,
            probe_bytes=probe_bytes,
            retained_tree_bytes=retained_tree_bytes,
            required_headroom_bytes=(
                probe_bytes
                + retained_tree_bytes
                + max(retained_tree_bytes, reference_workspace_bytes)
            ),
        )

    def _prepare_torch_result_tree(self, value, *, keep_on_device):
        if isinstance(value, (torch.return_types.max, torch.return_types.min)):
            value = value.values
        if isinstance(value, torch.Tensor):
            value = value.detach()
            return value if keep_on_device else value.cpu()
        if isinstance(value, list):
            return [
                self._prepare_torch_result_tree(item, keep_on_device=keep_on_device)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._prepare_torch_result_tree(item, keep_on_device=keep_on_device)
                for item in value
            )
        if isinstance(value, dict):
            return type(value)(
                (key, self._prepare_torch_result_tree(item, keep_on_device=keep_on_device))
                for key, item in value.items()
            )
        return value

    def _report_runtime_error_and_finalize(
        self,
        err,
        default_log_type,
        phase,
        *,
        allow_ignore_paddle=False,
        force_log_type=None,
    ):
        log_type, fatal = self.report_runtime_error(
            err,
            default_log_type,
            phase,
            allow_ignore_paddle=allow_ignore_paddle,
            force_log_type=force_log_type,
        )
        self.dump_finalize(log_type or default_log_type)
        return log_type, fatal

    def _report_comparison_error(self, err, tensor_index=0, tensor_count=1):
        phase = "backward" if self.is_backward else "forward"
        log_type, fatal = self.report_runtime_error(
            err,
            "paddle_accuracy",
            phase,
            tensor_position=f"{tensor_index + 1}/{tensor_count}",
        )
        self.dump_finalize(log_type or "paddle_accuracy")
        if fatal or self.exit_on_error:
            raise err

    def _report_structure_error(
        self,
        reason,
        *,
        tensor_position=None,
        **details,
    ):
        phase = "backward" if self.is_backward else "forward"
        detail_text = " | ".join(
            f"{key.replace('_', ' ')} {value}" for key, value in details.items()
        )
        self.report_case_result(
            "paddle_accuracy",
            reason.replace("_", " "),
            phase=phase,
            tensor_position=tensor_position,
            error=detail_text or None,
        )
        self.dump_finalize("paddle_accuracy")

    def _compare_accuracy_tree(self, actual, expected, tensor_index=0, tensor_count=None):
        tensor_types = (paddle.Tensor, torch.Tensor)

        def is_missing(value):
            return value is None or (
                isinstance(value, paddle.Tensor)
                and not value._is_initialized()
                and int(value.numel()) != 0
            )

        def leaf_count(left, right):
            if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
                return sum(leaf_count(left[key], right[key]) for key in left)
            if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                if len(left) != len(right):
                    return max(len(left), len(right), 1)
                return sum(
                    leaf_count(left_item, right_item)
                    for left_item, right_item in zip(left, right, strict=False)
                )
            return 1

        if tensor_count is None:
            tensor_count = leaf_count(actual, expected)

        def compare_leaf(left, right, index):
            position = f"{index + 1}/{tensor_count}"
            if is_missing(left) and is_missing(right):
                return True
            if isinstance(left, paddle.Tensor) and isinstance(right, bool):
                try:
                    assert left.dtype == paddle.bool, "paddle_output dtype is not bool"
                    assert left.shape == [], "paddle_output shape is not []"
                    assert bool(left) == right, (
                        f"paddle_output {bool(left)} is not equal to torch_output {right}"
                    )
                except Exception as err:
                    self._report_structure_error(
                        "value_mismatch", tensor_position=position, message=err
                    )
                    return False
                return True
            if isinstance(left, tensor_types) and isinstance(right, tensor_types):
                try:
                    self.torch_assert_accuracy(
                        left,
                        right,
                        atol=self.get_atol(),
                        rtol=self.get_rtol(),
                        tensor_index=index,
                        tensor_count=tensor_count,
                    )
                except Exception as err:
                    self._report_comparison_error(err, index, tensor_count)
                    return False
                return True
            if isinstance(left, tensor_types) or isinstance(right, tensor_types):
                self._report_structure_error(
                    "type_mismatch",
                    tensor_position=position,
                    actual_type=type(left).__name__,
                    expected_type=type(right).__name__,
                )
                return False
            try:
                self.np_assert_accuracy(
                    numpy.array(left),
                    numpy.array(right),
                    atol=self.get_atol(),
                    rtol=self.get_rtol(),
                )
            except Exception as err:
                self._report_comparison_error(err, index, tensor_count)
                return False
            return True

        def visit(left, right, index):
            if isinstance(left, dict):
                if not isinstance(right, dict):
                    self._report_structure_error(
                        "type_mismatch",
                        tensor_position=f"{index + 1}/{tensor_count}",
                        actual_type=type(left).__name__,
                        expected_type=type(right).__name__,
                    )
                    return 1, False
                if left.keys() != right.keys():
                    self._report_structure_error(
                        "key_mismatch",
                        tensor_position=f"{index + 1}/{tensor_count}",
                        actual_keys=list(left.keys()),
                        expected_keys=list(right.keys()),
                    )
                    return max(len(left), len(right), 1), False
                consumed = 0
                for key in left:
                    child_consumed, ok = visit(left[key], right[key], index + consumed)
                    consumed += child_consumed
                    if not ok:
                        return max(consumed, 1), False
                return max(consumed, 1), True
            if isinstance(left, (list, tuple)):
                if not isinstance(right, (list, tuple)):
                    self._report_structure_error(
                        "type_mismatch",
                        tensor_position=f"{index + 1}/{tensor_count}",
                        actual_type=type(left).__name__,
                        expected_type=type(right).__name__,
                    )
                    return 1, False
                if len(left) != len(right):
                    self._report_structure_error(
                        "count_mismatch",
                        tensor_position=f"{index + 1}/{tensor_count}",
                        actual_count=len(left),
                        expected_count=len(right),
                    )
                    return max(len(left), len(right), 1), False
                consumed = 0
                for left_item, right_item in zip(left, right, strict=False):
                    child_consumed, ok = visit(left_item, right_item, index + consumed)
                    consumed += child_consumed
                    if not ok:
                        return max(consumed, 1), False
                return max(consumed, 1), True
            if isinstance(right, (dict, list, tuple)):
                self._report_structure_error(
                    "type_mismatch",
                    tensor_position=f"{index + 1}/{tensor_count}",
                    actual_type=type(left).__name__,
                    expected_type=type(right).__name__,
                )
                return 1, False
            return 1, compare_leaf(left, right, index)

        _, ok = visit(actual, expected, tensor_index)
        return ok

    def test(self):
        self.dump_event("api_analyze_start", mode="accuracy")
        if self.need_skip():
            self.report_case_result("skip")
            self.dump_finalize("skip")
            return

        if not self.ana_api_info():
            self.report_case_result("config_parse", "ana_api_info failed")
            self.dump_finalize("config_parse")
            return
        self.dump_event("api_analyze_done", api_name=self.api_config.api_name)

        try:
            self.dump_event("config_convert_start")
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            self.dump_error("config_convert_error", e)
            self.report_case_result("config_convert", f"Conversion failed: {e!s}")
            self.dump_finalize("config_convert")
            return
        if not convert_result.is_supported:
            self.report_case_result(
                "config_convert",
                f"Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
            )
            self.dump_event("config_convert_error", error=convert_result.error_message)
            self.dump_finalize("config_convert")
            return
        self.dump_event("config_convert_done")
        if not convert_result.code or not convert_result.code.is_valid():
            self.report_case_result(
                "config_convert",
                f"No code generated for {self.api_config.api_name}",
            )
            self.dump_event("config_convert_error", error="no code generated")
            self.dump_finalize("config_convert")
            return

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

        probe_bytes = self.estimate_input_bytes()

        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            self.dump_event("torch_input_start")
            if not self.gen_torch_input():
                self.report_case_result("torch_error", "gen_torch_input failed")
                self.dump_finalize("torch_error")
                return
            self.dump_save(
                "torch_inputs",
                {"args": self.torch_args, "kwargs": self.torch_kwargs},
                framework="torch",
            )
            self.dump_event("torch_input_done")

            # Reseed before executing torch, so that random APIs
            # (e.g. torch.rand / uniform / normal / dropout) produce
            # deterministic outputs across runs when --random_seed is set.
            self.reset_random_state()
            self.dump_event("torch_forward_start")

            # torch_args 与 torch_kwargs 是尚未映射的 torch 参数（即按 paddle 的参数顺序与关键字排列的 torch tensors）
            # (弃用)以下代码等价于:
            # torch_output = Paddle2TorchConverter.execute(convert_result, self.torch_args, self.torch_kwargs)
            # 准备执行环境，将参数(torch tensors)直接映射至locals()
            exec_globals = {"torch": torch, "_adaptive_workspace_bytes": adaptive_workspace_bytes}
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
            self.dump_save("torch_forward_output", torch_output, framework="torch")
            self.dump_event("torch_forward_done")
            del exec_globals, exec_locals, output_var, code

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
            _, fatal = self._report_runtime_error_and_finalize(err, "torch_error", "forward")
            if fatal:
                raise
            return

        torch_grad_success = False
        torch_out_grads = None
        if self.need_check_grad():
            try:
                self.dump_event("torch_backward_start")
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(
                    torch_output
                )
                self.dump_save(
                    "torch_backward",
                    {
                        "inputs": inputs_list,
                        "outputs": result_outputs,
                        "grad_outputs": result_outputs_grads,
                    },
                    framework="torch",
                )
                del self.torch_args, self.torch_kwargs
                if inputs_list and result_outputs and result_outputs_grads:
                    torch_out_grads = torch.autograd.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
                    torch_grad_success = True
                    self.dump_save("torch_input_grads", torch_out_grads, framework="torch")
                self.dump_event("torch_backward_done", grad_success=torch_grad_success)
                del inputs_list, result_outputs, result_outputs_grads
            except Exception as err:
                if str(err).startswith("Too large tensor to get cached numpy: "):
                    self._report_runtime_error_and_finalize(
                        err,
                        "config_input",
                        "backward",
                        force_log_type="config_input",
                    )
                    return
                _, fatal = self._report_runtime_error_and_finalize(err, "torch_error", "backward")
                if fatal:
                    raise
                return
            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                self._report_runtime_error_and_finalize(err, "torch_error", "backward cuda check")
                raise
        else:
            del self.torch_args, self.torch_kwargs

        spill_torch_outputs = False
        if self.use_gpu_mode:
            spill_torch_outputs = self._should_spill_torch_result_tree(
                convert_result,
                torch_output,
                torch_out_grads,
                probe_bytes,
            )
        keep_torch_outputs_on_device = self.use_gpu_mode and not spill_torch_outputs

        torch_output = self._prepare_torch_result_tree(
            torch_output,
            keep_on_device=keep_torch_outputs_on_device,
        )
        if torch_grad_success:
            torch_out_grads = self._prepare_torch_result_tree(
                torch_out_grads,
                keep_on_device=keep_torch_outputs_on_device,
            )

        gc.collect()
        if self.use_gpu_mode:
            self.clear_torch_tensor(probe_bytes=probe_bytes)
            gpu_mode_maybe_empty_cache(
                self.gpu_mode_config,
                force=not keep_torch_outputs_on_device,
                probe_bytes=probe_bytes,
            )
        else:
            torch.cuda.empty_cache()

        try:
            if not self.gen_paddle_input():
                self.report_case_result("paddle_error", "gen_paddle_input failed")
                self.dump_finalize("paddle_error")
                return

            # Reseed before executing paddle so that random APIs
            # (paddle.uniform / normal / randn / bernoulli / dropout ...)
            # match the torch run with the same seed.
            self.reset_random_state()
            self.dump_event("paddle_forward_start")
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
            log_type, fatal = self._report_runtime_error_and_finalize(
                err, "paddle_error", "forward", allow_ignore_paddle=True
            )
            if fatal or (self.exit_on_error and log_type == "paddle_error"):
                raise
            return

        try:
            self.dump_save("paddle_forward_output", paddle_output, framework="paddle")
            self.dump_event("paddle_forward_done")
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            self._report_runtime_error_and_finalize(err, "paddle_cuda", "forward")
            raise

        try:
            paddle_output, torch_output = process_output(
                self.api_config, paddle_output, torch_output
            )
        except Exception as err:
            _, fatal = self._report_runtime_error_and_finalize(err, "paddle_accuracy", "forward")
            if fatal:
                raise
            return

        self.is_backward = False
        if self.use_gpu_mode:
            gpu_mode_maybe_empty_cache(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
            )
        if not self._compare_accuracy_tree(paddle_output, torch_output):
            return

        # Forward check now pass.
        # Then do paddle backward and backward result check.
        if self.use_gpu_mode:
            del torch_output
            gpu_mode_maybe_empty_cache(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
            )
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
                    self._report_runtime_error_and_finalize(
                        err,
                        "config_input",
                        "backward",
                        force_log_type="config_input",
                    )
                    return
                log_type, fatal = self._report_runtime_error_and_finalize(
                    err, "paddle_error", "backward", allow_ignore_paddle=True
                )
                if fatal or (self.exit_on_error and log_type == "paddle_error"):
                    raise
                return

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                self._report_runtime_error_and_finalize(err, "paddle_cuda", "backward cuda check")
                raise

            try:
                paddle_out_grads, torch_out_grads = process_grad_output(
                    self.api_config, paddle_out_grads, torch_out_grads
                )
            except Exception as err:
                _, fatal = self._report_runtime_error_and_finalize(
                    err, "paddle_accuracy", "backward"
                )
                if fatal:
                    raise
                return

            if self.use_gpu_mode:
                gpu_mode_maybe_empty_cache(
                    self.gpu_mode_config,
                    probe_bytes=probe_bytes,
                )
            if not self._compare_accuracy_tree(paddle_out_grads, torch_out_grads):
                return

        self.report_case_result("pass")
        self.dump_finalize("pass")


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
            paddle_vector = _paddle_tensor_to_torch(paddle_eigvectors[i]).to(
                device=torch_eigvectors.device
            )
            coef_vector = paddle_vector / torch_eigvectors[i]
            if coef_vector.is_complex():
                coef_vector = torch.complex(
                    coef_vector.real.round(decimals=2),
                    coef_vector.imag.round(decimals=2),
                )
            else:
                coef_vector = coef_vector.round(decimals=2)
            coef_vector_approx = torch.ones_like(coef_vector) * coef_vector[0]
            abs_coef = coef_vector.abs().to(dtype=torch.float64)[0]
            one = torch.ones_like(abs_coef)
            paddle_output.append([coef_vector, abs_coef])
            torch_output.append([coef_vector_approx, one])
    elif api_config.api_name == "paddle._C_ops.fused_linear_param_grad_add":
        # When has_bias=False, Paddle returns an uninitialized tensor for dbias (2nd output).
        # Only compare the first output (dweight).
        if isinstance(paddle_output, (list, tuple)) and len(paddle_output) > 1:
            paddle_output = paddle_output[:1]
        if isinstance(torch_output, (list, tuple)) and len(torch_output) > 1:
            torch_output = torch_output[:1]
    elif api_config.api_name == "paddle._C_ops.swiglu_grad":
        # When y is None, Paddle returns an uninitialized placeholder tensor for dy.
        # Only compare dx to avoid converting the uninitialized tensor to DLPack.
        if len(api_config.args) > 1 and api_config.args[1] is None:
            if isinstance(paddle_output, (list, tuple)) and len(paddle_output) > 1:
                paddle_output = paddle_output[:1]
            if isinstance(torch_output, (list, tuple)) and len(torch_output) > 1:
                torch_output = torch_output[:1]
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
