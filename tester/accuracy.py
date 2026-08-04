from __future__ import annotations

import gc

import numpy
import paddle
import torch
import yaml

from .accuracy_common import process_grad_output, process_output
from .base import APITestBase, gpu_mode_maybe_empty_cache
from .paddle_to_torch import ConversionKind, get_converter
from .paddle_to_torch.arguments import bind_paddle_arguments


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

        def compare_leaf(left, right, index, count):
            position = f"{index + 1}/{count}"
            if self.is_missing_compare_value(left) and self.is_missing_compare_value(right):
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
                        tensor_count=count,
                    )
                except Exception as err:
                    self._report_comparison_error(err, index, count)
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
                self._report_comparison_error(err, index, count)
                return False
            return True

        def report_structure_error(reason, *, tensor_position=None, **details):
            details.pop("tensor_index", None)
            details.pop("tensor_count", None)
            self._report_structure_error(reason, tensor_position=tensor_position, **details)

        return self.compare_tensor_tree(
            actual,
            expected,
            compare_leaf,
            report_structure_error,
            tensor_index=tensor_index,
            tensor_count=tensor_count,
        )

    def _convert_api(self):
        try:
            self.dump_event("config_convert_start")
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            self.dump_error("config_convert_error", e)
            self.report_case_result("config_convert", f"Conversion failed: {e!s}")
            self.dump_finalize("config_convert")
            return None
        if convert_result.kind is ConversionKind.UNSUPPORTED:
            self.report_case_result(
                "config_convert",
                f"Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
            )
            self.dump_event("config_convert_error", error=convert_result.error_message)
            self.dump_finalize("config_convert")
            return None
        self.dump_event("config_convert_done")
        return convert_result

    def _generate_input_values(self):
        try:
            self.dump_event("numpy_input_start")
            if not self.generate_input_values():
                self.report_case_result("config_input", "generate_input_values failed")
                self.dump_finalize("config_input")
                return False
            self.dump_event("numpy_input_done")
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", "input")
            self.dump_finalize(log_type or "config_input")
            if fatal:
                raise
            return False
        return True

    def get_torch_output(self, convert_result):
        try:
            device = torch.device("cuda:0")
            torch.set_default_device(device)
            self.dump_event("torch_input_start")
            if not self.build_torch_input():
                self.report_case_result("torch_error", "build_torch_input failed")
                self.dump_finalize("torch_error")
                return False, None, None, False
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

            bound_arguments = bind_paddle_arguments(
                self.api_config.api_name,
                self.torch_args,
                self.torch_kwargs,
            )

            def execute_core(compiled, exec_globals, exec_locals):
                if self.test_amp:
                    with torch.autocast(device_type="cuda"):
                        exec(compiled, exec_globals, exec_locals)
                else:
                    exec(compiled, exec_globals, exec_locals)

            torch_output = self.converter.execute(
                convert_result,
                self.torch_args,
                bound_arguments,
                execution_locals=self._torch_execution_locals(),
                core_executor=execute_core,
            )
            self.dump_save("torch_forward_output", torch_output, framework="torch")
            self.dump_event("torch_forward_done")

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
            return False, None, None, False

        torch_grad_success = False
        torch_out_grads = None
        if not self.need_check_grad():
            del self.torch_args, self.torch_kwargs
            return True, torch_output, torch_out_grads, torch_grad_success

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
                return False, None, None, False
            _, fatal = self._report_runtime_error_and_finalize(err, "torch_error", "backward")
            if fatal:
                raise
            return False, None, None, False
        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            self._report_runtime_error_and_finalize(err, "torch_error", "backward cuda check")
            raise
        return True, torch_output, torch_out_grads, torch_grad_success

    def _prepare_torch_results_for_paddle(
        self, convert_result, torch_output, torch_out_grads, torch_grad_success, probe_bytes
    ):
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
        return torch_output, torch_out_grads

    def get_paddle_output(self):
        try:
            if not self.build_paddle_input():
                self.report_case_result("paddle_error", "build_paddle_input failed")
                self.dump_finalize("paddle_error")
                return False, None
            # Torch 已完成，Paddle 输入已持有数据；生成源不再有后续消费者。
            self.clear_generated_input_values()

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
            return False, None

        try:
            self.dump_save("paddle_forward_output", paddle_output, framework="paddle")
            self.dump_event("paddle_forward_done")
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            self._report_runtime_error_and_finalize(err, "paddle_cuda", "forward")
            raise
        return True, paddle_output

    def get_paddle_grad(self, paddle_output):
        paddle_out_grads = None
        try:
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
                return False, None
            log_type, fatal = self._report_runtime_error_and_finalize(
                err, "paddle_error", "backward", allow_ignore_paddle=True
            )
            if fatal or (self.exit_on_error and log_type == "paddle_error"):
                raise
            return False, None

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            self._report_runtime_error_and_finalize(err, "paddle_cuda", "backward cuda check")
            raise
        return True, paddle_out_grads

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

        convert_result = self._convert_api()
        if convert_result is None:
            return
        if not self.run_gpu_memory_preflight("accuracy"):
            return
        if not self._generate_input_values():
            return
        probe_bytes = self.estimate_input_bytes()

        torch_success, torch_output, torch_out_grads, torch_grad_success = self.get_torch_output(
            convert_result
        )
        if not torch_success:
            return
        torch_output, torch_out_grads = self._prepare_torch_results_for_paddle(
            convert_result,
            torch_output,
            torch_out_grads,
            torch_grad_success,
            probe_bytes,
        )

        paddle_success, paddle_output = self.get_paddle_output()
        if not paddle_success:
            return

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
            paddle_grad_success, paddle_out_grads = self.get_paddle_grad(paddle_output)
            if not paddle_grad_success:
                return

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
