from __future__ import annotations

import gc
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy
import paddle
import torch

from .accuracy import process_grad_output, process_output
from .api_config.logging import log_comparison, log_worker
from .base import CUDA_ERROR, CUDA_OOM, APITestBase, gpu_mode_memory_decision
from .paddle_to_torch import adaptive_workspace_bytes, get_converter


class APITestAccuracyStable(APITestBase):
    # 执行阶段错误广播映射: (iter_idx, source) -> 受影响的 comp 列表
    _TORCH_AFFECTED_COMPS = {
        0: ["P1T1", "P2T1", "T1T2", "P1T1B", "P2T1B", "T1T2B"],
        1: ["P2T2", "P1T2", "T1T2", "P2T2B", "P1T2B", "T1T2B"],
    }
    _PADDLE_AFFECTED_COMPS = {
        0: ["P1T1", "P1T2", "P1P2", "P1T1B", "P1T2B", "P1P2B"],
        1: ["P2T2", "P2T1", "P1P2", "P2T2B", "P2T1B", "P1P2B"],
    }

    def __init__(self, api_config, **kwargs):
        super().__init__(api_config, runtime_config=kwargs.get("runtime_config"))
        self.test_amp = kwargs.get("test_amp", False)
        self.use_gpu_mode = self.gpu_mode_config.enabled
        self.use_dual_gpu = self.use_gpu_mode and self.gpu_mode_config.dual_gpu
        self.comparison_device_id = self.gpu_mode_config.comparison_device_id
        self.converter = get_converter()
        torch.set_printoptions(profile="short", edgeitems=2, threshold=100, linewidth=120)
        torch.set_default_device("cuda")

    def move_tensor_tree_to_comparison_gpu(self, value):
        if self.comparison_device_id is None:
            raise RuntimeError("dual-GPU comparison device is not configured")
        return self.move_tensor_tree_to_gpu(value, self.comparison_device_id)

    def release_compute_gpu_cache(self, framework=None, *, collect_cycles=False):
        if collect_cycles:
            gc.collect()
        if framework in (None, "torch"):
            with torch.cuda.device(0):
                torch.cuda.empty_cache()
        if framework in (None, "paddle"):
            paddle.device.cuda.empty_cache()

    def compare_first_pair(self, paddle_output, torch_output, paddle_grad, torch_grad):
        with torch.cuda.device(self.comparison_device_id):
            self.compare(paddle_output, torch_output, "P1T1")
            self.compare(paddle_grad, torch_grad, "P1T1B")

    def release_comparison_gpu_cache(self):
        with torch.cuda.device(self.comparison_device_id):
            torch.cuda.empty_cache()

        compute_device = paddle.device.get_device()
        try:
            paddle.device.set_device(f"gpu:{self.comparison_device_id}")
            paddle.device.cuda.empty_cache()
        finally:
            paddle.device.set_device(compute_device)

    def _reference_workspace_bytes(self, convert_result):
        if not self.gpu_mode_config.enabled:
            return 0
        code = convert_result.code
        source_lines = (*code.preprocess, *code.core, *code.postprocess)
        if not any("_workspace_bytes =" in str(line) for line in source_lines):
            return 0
        return adaptive_workspace_bytes(torch)

    def _broadcast_to_comp_dimensions(self, log_type, affected_comps):
        """将执行阶段错误广播到所有受影响的 comp 维度"""
        for comp in affected_comps:
            log_worker.write_to_comp_log(comp, log_type, self.api_config.config)

    def test(self):
        if self.need_skip():
            print(f"[skip] {self.api_config.config}", flush=True)
            log_worker.write_to_log("skip", self.api_config.config)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            log_worker.write_to_log("config_parse", self.api_config.config)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(
                f"[config_convert] Conversion failed for {self.api_config.config}: {e!s}",
                flush=True,
            )
            log_worker.write_to_log("config_convert", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(
                f"[config_convert] Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
                flush=True,
            )
            log_worker.write_to_log("config_convert", self.api_config.config)
            return
        if not convert_result.code or not convert_result.code.is_valid():
            print(
                f"[config_convert] No code generated for {self.api_config.api_name}",
                flush=True,
            )
            log_worker.write_to_log("config_convert", self.api_config.config)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                log_worker.write_to_log("config_input", self.api_config.config)
                return
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", "input")
            if fatal:
                raise
            return

        try:
            self.save_original_inputs_to_cpu()
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", "input cache")
            if fatal:
                raise
            return

        probe_bytes = self.estimate_input_bytes()

        torch_output_pair = []
        torch_grad_pair = []
        paddle_output_pair = []
        paddle_grad_pair = []
        first_pair_comparison = None
        first_pair_comparison_finished = False
        comparison_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="accuracy-stable-compare")
            if self.use_dual_gpu
            else None
        )

        def finish_first_pair_comparison():
            nonlocal first_pair_comparison_finished
            if first_pair_comparison_finished:
                return
            try:
                if first_pair_comparison is not None:
                    first_pair_comparison.result()
            finally:
                if comparison_executor is not None:
                    comparison_executor.shutdown(wait=True)
                first_pair_comparison_finished = True

        # Every execution recreates its input from the same immutable CPU copy.
        for _i in range(2):
            # ======== torch ========
            self.reset_random_state()
            torch_output, torch_out_grads, torch_grad_success = self.get_torch_output(
                convert_result, _i
            )
            if torch_output is None:
                finish_first_pair_comparison()
                return
            torch_output = self.detach_tensor_tree(torch_output)
            torch_out_grads = self.detach_tensor_tree(torch_out_grads)
            self.clear_runtime_inputs("torch")
            if self.use_dual_gpu:
                torch_output = self.move_tensor_tree_to_comparison_gpu(torch_output)
                torch_out_grads = self.move_tensor_tree_to_comparison_gpu(torch_out_grads)
                self.release_compute_gpu_cache("torch")
            elif self.use_gpu_mode:
                torch_live_bytes = self.tensor_tree_nbytes((torch_output, torch_out_grads))
                # Release idle Torch blocks before the next Paddle execution;
                # the two frameworks do not share caching allocators.
                gpu_mode_memory_decision(
                    self.gpu_mode_config,
                    probe_bytes=probe_bytes,
                    retained_tree_bytes=torch_live_bytes,
                    required_headroom_bytes=probe_bytes + torch_live_bytes,
                )

            if self.use_dual_gpu and _i == 1:
                # Bound compare-card peak: overlap P1T1 with T2 Torch, then
                # retire its scratch before P2 Paddle results arrive.
                finish_first_pair_comparison()

            # ======== paddle ========
            self.reset_random_state()
            paddle_output, paddle_out_grads = self.get_paddle_output(torch_grad_success, _i)
            if paddle_output is None:
                finish_first_pair_comparison()
                return
            paddle_output = self.detach_tensor_tree(paddle_output)
            paddle_out_grads = self.detach_tensor_tree(paddle_out_grads)
            self.clear_runtime_inputs("paddle")
            if self.use_gpu_mode and not self.use_dual_gpu:
                gpu_mode_memory_decision(
                    self.gpu_mode_config,
                    probe_bytes=probe_bytes,
                    required_headroom_bytes=probe_bytes,
                )

            # ======== format ========
            paddle_output, torch_output = process_output(
                self.api_config, paddle_output, torch_output
            )
            paddle_out_grads, torch_out_grads = process_grad_output(
                self.api_config, paddle_out_grads, torch_out_grads
            )

            if self.use_dual_gpu:
                # Formatting may create replacement tensors, so normalize all
                # four complete result trees onto the comparison GPU here.
                torch_output = self.move_tensor_tree_to_comparison_gpu(torch_output)
                torch_out_grads = self.move_tensor_tree_to_comparison_gpu(torch_out_grads)
                paddle_output = self.move_tensor_tree_to_comparison_gpu(paddle_output)
                paddle_out_grads = self.move_tensor_tree_to_comparison_gpu(paddle_out_grads)
                self.release_compute_gpu_cache("paddle")

            # ======== add to pair ========
            # if torch_grad_success = False, out_grads = [] and compare return
            torch_output_pair.append(torch_output)
            torch_grad_pair.append(torch_out_grads)
            paddle_output_pair.append(paddle_output)
            paddle_grad_pair.append(paddle_out_grads)

            # Pair lists own the results from here onward. Drop loop-local aliases
            # before comparison or D2H spill so source trees can be released serially.
            torch_output = None
            paddle_output = None
            torch_out_grads = None
            paddle_out_grads = None
            if _i != 0:
                continue

            if comparison_executor is not None:
                first_pair_comparison = comparison_executor.submit(
                    self.compare_first_pair,
                    paddle_output_pair[0],
                    torch_output_pair[0],
                    paddle_grad_pair[0],
                    torch_grad_pair[0],
                )
            else:
                self.compare(paddle_output_pair[0], torch_output_pair[0], "P1T1")
                self.compare(paddle_grad_pair[0], torch_grad_pair[0], "P1T1B")

            if self.use_dual_gpu:
                continue
            if not self.use_gpu_mode:
                continue

            torch_phase_bytes = self.tensor_tree_nbytes((torch_output_pair[0], torch_grad_pair[0]))
            paddle_phase_bytes = self.tensor_tree_nbytes(
                (paddle_output_pair[0], paddle_grad_pair[0])
            )
            retained_tree_bytes = torch_phase_bytes + paddle_phase_bytes
            required_headroom_bytes = (
                probe_bytes
                + max(torch_phase_bytes, paddle_phase_bytes)
                + self._reference_workspace_bytes(convert_result)
            )
            decision = gpu_mode_memory_decision(
                self.gpu_mode_config,
                request_spill=True,
                probe_bytes=probe_bytes,
                retained_tree_bytes=retained_tree_bytes,
                required_headroom_bytes=required_headroom_bytes,
            )
            if decision.should_spill:
                self.spill_tensor_tree_slot_to_cpu(torch_output_pair)
                self.spill_tensor_tree_slot_to_cpu(paddle_output_pair)
                self.spill_tensor_tree_slot_to_cpu(torch_grad_pair)
                self.spill_tensor_tree_slot_to_cpu(paddle_grad_pair)

        self.clear_original_cpu_inputs()
        if self.use_dual_gpu:
            # Output-gradient seeds are reused by all four backward executions,
            # but are no longer needed once P2 has been offloaded.
            self.outputs_grad_numpy.clear()
            self.outputs_grad_paddleonly.clear()
            self.release_compute_gpu_cache()

        finish_first_pair_comparison()

        # ======== summary ========
        self.compare(paddle_output_pair[1], torch_output_pair[1], "P2T2")
        self.compare(paddle_grad_pair[1], torch_grad_pair[1], "P2T2B")
        self.compare(paddle_output_pair[1], torch_output_pair[0], "P2T1")
        self.compare(paddle_grad_pair[1], torch_grad_pair[0], "P2T1B")
        self.compare(paddle_output_pair[0], torch_output_pair[1], "P1T2")
        self.compare(paddle_grad_pair[0], torch_grad_pair[1], "P1T2B")
        self.compare(torch_output_pair[0], torch_output_pair[1], "T1T2")
        torch_output_pair.clear()
        self.compare(torch_grad_pair[0], torch_grad_pair[1], "T1T2B")
        torch_grad_pair.clear()
        self.compare(paddle_output_pair[0], paddle_output_pair[1], "P1P2")
        paddle_output_pair.clear()
        self.compare(paddle_grad_pair[0], paddle_grad_pair[1], "P1P2B")
        paddle_grad_pair.clear()

        if self.use_dual_gpu:
            torch_output = None
            paddle_output = None
            torch_out_grads = None
            paddle_out_grads = None
            self.release_comparison_gpu_cache()

        log_worker.write_stable_passes(self.api_config.config)

    def get_torch_output(self, convert_result, iter_idx=0):
        # ======== run torch forward ========:
        torch_output = None
        try:
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return None, None, None

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

            code = convert_result.code
            with torch.set_grad_enabled(self.need_check_grad()):
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
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "torch_error", "forward")
            self._broadcast_to_comp_dimensions(log_type, self._TORCH_AFFECTED_COMPS[iter_idx])
            if fatal:
                raise
            return None, None, None

        # ======== run torch backward ========
        torch_grad_success = False
        torch_out_grads = []
        if self.need_check_grad():
            try:
                inputs_list = self.get_torch_input_list()
                result_outputs, result_outputs_grads = self.gen_torch_output_and_output_grad(
                    torch_output
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    torch_out_grads = torch.autograd.grad(
                        outputs=result_outputs,
                        inputs=inputs_list,
                        grad_outputs=result_outputs_grads,
                    )
                    torch_grad_success = True
            except Exception as err:
                err_str = str(err)
                if err_str.startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[config_input] {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions(
                        "config_input", self._TORCH_AFFECTED_COMPS[iter_idx]
                    )
                    return None, None, None
                if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                    print(
                        f"[oom] backward | {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions("oom", self._TORCH_AFFECTED_COMPS[iter_idx])
                    raise
                if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
                    print(
                        f"[torch_error] backward | {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions(
                        "torch_error", self._TORCH_AFFECTED_COMPS[iter_idx]
                    )
                    raise
                print(err_str, flush=True)

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                err_str = str(err)
                print(
                    f"[torch_error] backward | {self.api_config.config}\n{err_str}",
                    flush=True,
                )
                traceback.print_exc()
                self._broadcast_to_comp_dimensions(
                    "torch_error", self._TORCH_AFFECTED_COMPS[iter_idx]
                )
                raise

        def process_torch_outputs(obj):
            if isinstance(obj, (torch.return_types.max, torch.return_types.min)):
                obj = obj.values
            if isinstance(obj, (list, tuple)):
                obj = list(obj)
            return obj

        torch_output = process_torch_outputs(torch_output)
        torch_out_grads = process_torch_outputs(torch_out_grads)
        return torch_output, torch_out_grads, torch_grad_success

    def get_paddle_output(self, torch_grad_success, iter_idx=0):
        # ======== run paddle forward ========
        paddle_output = None
        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return None, None

            # determine the dtype
            self.api_config.dtype = None
            for arg in self.paddle_args:
                if isinstance(arg, paddle.Tensor):
                    self.api_config.dtype = arg.dtype
                    break
            if self.api_config.dtype is None:
                for arg in self.paddle_kwargs.values():
                    if isinstance(arg, paddle.Tensor):
                        self.api_config.dtype = arg.dtype
                        break
            # if there is no tensor in args and kwargs, use float32 as default
            if self.api_config.dtype is None:
                self.api_config.dtype = paddle.float32

            # find the first arg
            first_arg = (
                self.paddle_args[0]
                if len(self.paddle_args) > 0
                else next(iter(self.paddle_kwargs.values()))
            )
            with paddle.set_grad_enabled(self.need_check_grad()):
                if self.api_config.api_name.startswith("paddle.Tensor."):
                    api_name = self.api_config.api_name.split(".")[-1]
                    api = getattr(self.paddle_args[0], api_name)
                    if self.test_amp:
                        with paddle.amp.auto_cast():
                            paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
                    else:
                        paddle_output = api(*self.paddle_args[1:], **self.paddle_kwargs)
                else:
                    if self.test_amp:
                        with paddle.amp.auto_cast():
                            paddle_output = self.paddle_api(*self.paddle_args, **self.paddle_kwargs)
                    else:
                        paddle_output = self.paddle_api(*self.paddle_args, **self.paddle_kwargs)
            if (
                self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__"
            ) or self.api_config.api_name == "paddle.Tensor.__setitem__":
                paddle_output = first_arg
        except Exception as err:
            log_type, fatal = self.report_runtime_error(
                err,
                "paddle_error",
                "forward",
                allow_ignore_paddle=True,
            )
            self._broadcast_to_comp_dimensions(log_type, self._PADDLE_AFFECTED_COMPS[iter_idx])
            if fatal:
                raise
            return None, None

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print(f"[paddle_cuda] {self.api_config.config}\n{err!s}", flush=True)
            self._broadcast_to_comp_dimensions("paddle_cuda", self._PADDLE_AFFECTED_COMPS[iter_idx])
            raise

        # ======== run paddle backward ========
        paddle_out_grads = []
        if torch_grad_success:
            try:
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(
                    paddle_output
                )
                if inputs_list and result_outputs and result_outputs_grads:
                    paddle_out_grads = paddle.grad(
                        result_outputs,
                        inputs_list,
                        grad_outputs=result_outputs_grads,
                        allow_unused=True,
                    )
            except Exception as err:
                err_str = str(err)
                if err_str.startswith("Too large tensor to get cached numpy: "):
                    print(
                        f"[config_input] {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions(
                        "config_input", self._PADDLE_AFFECTED_COMPS[iter_idx]
                    )
                    return None, None
                if self.should_ignore_paddle_error(err_str):
                    print(f"[pass] {self.api_config.config}", flush=True)
                    self._broadcast_to_comp_dimensions(
                        "pass", self._PADDLE_AFFECTED_COMPS[iter_idx]
                    )
                    return None, None
                if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
                    print(
                        f"[paddle_cuda] backward | {self.api_config.config}\n{err_str}",
                    )
                    self._broadcast_to_comp_dimensions(
                        "paddle_cuda", self._PADDLE_AFFECTED_COMPS[iter_idx]
                    )
                    raise
                if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                    print(
                        f"[oom] backward | {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions("oom", self._PADDLE_AFFECTED_COMPS[iter_idx])
                    raise
                print(
                    f"[paddle_error] backward | {self.api_config.config}\n{err_str}",
                    flush=True,
                )
                traceback.print_exc()
                self._broadcast_to_comp_dimensions(
                    "paddle_error", self._PADDLE_AFFECTED_COMPS[iter_idx]
                )
                return None, None

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                print(
                    f"[paddle_cuda] backward | {self.api_config.config}\n{err!s}",
                    flush=True,
                )
                self._broadcast_to_comp_dimensions(
                    "paddle_cuda", self._PADDLE_AFFECTED_COMPS[iter_idx]
                )
                raise

        def process_paddle_outputs(obj):
            if isinstance(obj, (list, tuple)):
                obj = list(obj)
            return obj

        paddle_output = process_paddle_outputs(paddle_output)
        paddle_out_grads = process_paddle_outputs(paddle_out_grads)
        return paddle_output, paddle_out_grads

    def compare(self, input1, input2, comp):
        if isinstance(input1, (paddle.Tensor, torch.Tensor)):
            if isinstance(input2, (paddle.Tensor, torch.Tensor)):
                try:
                    self.assert_accuracy(
                        input1,
                        input2,
                        comp,
                        tensor_index=0,
                        tensor_count=1,
                    )
                except Exception as err:
                    self.report_compare_error(
                        err,
                        comp,
                        tensor_position="1/1",
                    )
                    return
            else:
                log_comparison.log_comp_issue(
                    comp,
                    "paddle_accuracy",
                    self.api_config.config,
                    tensor_index=None,
                    tensor_count=None,
                    reason="type_mismatch",
                    actual_type=type(input1).__name__,
                    expected_type=type(input2).__name__,
                )
                return
        elif isinstance(input1, (list, tuple)):
            if not isinstance(input2, (list, tuple)):
                log_comparison.log_comp_issue(
                    comp,
                    "paddle_accuracy",
                    self.api_config.config,
                    tensor_index=None,
                    tensor_count=None,
                    reason="type_mismatch",
                    actual_type=type(input1).__name__,
                    expected_type=type(input2).__name__,
                )
                return
            if len(input1) != len(input2):
                log_comparison.log_comp_issue(
                    comp,
                    "paddle_accuracy",
                    self.api_config.config,
                    tensor_index=None,
                    tensor_count=None,
                    reason="count_mismatch",
                    actual_count=len(input1),
                    expected_count=len(input2),
                )
                return
            tensor_count = len(input1)
            for idx, (item1, item2) in enumerate(zip(input1, input2, strict=False)):
                if isinstance(item1, (paddle.Tensor, torch.Tensor)) and isinstance(
                    item2, (paddle.Tensor, torch.Tensor)
                ):
                    try:
                        self.assert_accuracy(
                            item1,
                            item2,
                            comp,
                            tensor_index=idx,
                            tensor_count=tensor_count,
                        )
                    except Exception as err:
                        self.report_compare_error(
                            err,
                            comp,
                            tensor_position=f"{idx + 1}/{tensor_count}",
                        )
                        return
                elif not isinstance(item1, (paddle.Tensor, torch.Tensor)) and not isinstance(
                    item2, (paddle.Tensor, torch.Tensor)
                ):
                    try:
                        self.assert_accuracy(
                            torch.tensor(item1),
                            torch.tensor(item2),
                            comp,
                            tensor_index=idx,
                            tensor_count=tensor_count,
                        )
                    except Exception as err:
                        self.report_compare_error(
                            err,
                            comp,
                            tensor_position=f"{idx + 1}/{tensor_count}",
                        )
                        return
                else:
                    log_comparison.log_comp_issue(
                        comp,
                        "paddle_accuracy",
                        self.api_config.config,
                        tensor_index=idx,
                        tensor_count=tensor_count,
                        reason="type_mismatch",
                        actual_type=type(item1).__name__,
                        expected_type=type(item2).__name__,
                    )
                    return
        else:
            try:
                self.assert_accuracy(
                    torch.tensor(input1),
                    torch.tensor(input2),
                    comp,
                    tensor_index=0,
                    tensor_count=1,
                )
            except Exception as err:
                self.report_compare_error(
                    err,
                    comp,
                    tensor_position="1/1",
                )
                return

    def assert_accuracy(
        self,
        tensor1,
        tensor2,
        comp,
        tensor_index=0,
        tensor_count=1,
    ):
        api_name = self.api_config.api_name
        config = self.api_config.config
        dtype = self.api_config.dtype
        check_dtype = self.should_check_dtype()
        framework_names = {"P": "Paddle", "T": "Torch"}
        actual_source = framework_names[comp[0]]
        expected_source = framework_names[comp.removesuffix("B")[2]]

        try:
            self.torch_assert_accuracy(
                tensor1,
                tensor2,
                atol=0.0,
                rtol=0.0,
                check_dtype=check_dtype,
                actual_name=actual_source,
                expected_name=expected_source,
                apply_special_tolerance=False,
            )
            log_comparison.log_accuracy_stable(
                "Identical",
                api_name,
                config,
                dtype,
                comp,
                tensor_index=tensor_index,
                tensor_count=tensor_count,
            )
        except Exception as err:
            err_str = str(err)
            is_acc_err = False
            err_list = err_str.split("\n", maxsplit=1)
            if len(err_list) > 1 and (
                err_list[1].startswith("Tensor-likes") or err_list[1].startswith("Scalars")
            ):
                is_acc_err = True
            if is_acc_err:
                log_comparison.log_accuracy_stable(
                    err_str,
                    api_name,
                    config,
                    dtype,
                    comp,
                    tensor_index=tensor_index,
                    tensor_count=tensor_count,
                )
            else:
                raise
