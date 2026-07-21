from __future__ import annotations

import gc
import traceback

import numpy
import paddle
import torch

from .accuracy import process_grad_output, process_output
from .api_config.config_analyzer import TensorConfig
from .api_config.log_writer import (
    ALL_DIMENSIONS,
    COMP_TO_DIMENSION,
    has_comp_terminal_log,
    has_terminal_log,
    log_accuracy_stable,
    write_to_comp_log,
    write_to_log,
)
from .base import CUDA_ERROR, CUDA_OOM, APITestBase, gpu_mode_maybe_empty_cache
from .paddle_to_torch import get_converter


class APITestAccuracyStable(APITestBase):
    # 执行阶段错误广播映射: (iter_idx, source) -> 受影响的 comp 列表
    _TORCH_AFFECTED_COMPS = {
        0: ["T1P1", "T1P2", "T1T2", "T1P1B", "T1P2B", "T1T2B"],
        1: ["T2P2", "T2P1", "T1T2", "T2P2B", "T2P1B", "T1T2B"],
    }
    _PADDLE_AFFECTED_COMPS = {
        0: ["T1P1", "T2P1", "P1P2", "T1P1B", "T2P1B", "P1P2B"],
        1: ["T2P2", "T1P2", "P1P2", "T2P2B", "T1P2B", "P1P2B"],
    }

    def __init__(self, api_config, **kwargs):
        super().__init__(api_config, runtime_config=kwargs.get("runtime_config"))
        self.test_amp = kwargs.get("test_amp", False)
        self.use_gpu_mode = self.gpu_mode_config.enabled
        self.converter = get_converter()
        torch.set_printoptions(profile="short", edgeitems=2, threshold=100, linewidth=120)
        torch.set_default_device("cuda")

    def _detach_result(self, value):
        if isinstance(value, (torch.Tensor, paddle.Tensor)):
            return value.detach()
        if isinstance(value, list):
            return [self._detach_result(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._detach_result(item) for item in value)
        if isinstance(value, dict):
            return {key: self._detach_result(item) for key, item in value.items()}
        return value

    def _move_result_to_cpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.cpu()
        if isinstance(value, paddle.Tensor):
            return value.cpu()
        if isinstance(value, list):
            return [self._move_result_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_result_to_cpu(item) for item in value)
        if isinstance(value, dict):
            return {key: self._move_result_to_cpu(item) for key, item in value.items()}
        return value

    def _result_num_bytes(self, value):
        if isinstance(value, (torch.Tensor, paddle.Tensor)):
            try:
                return int(value.numel()) * int(value.element_size())
            except Exception:
                return 0
        if isinstance(value, (list, tuple)):
            return sum(self._result_num_bytes(item) for item in value)
        if isinstance(value, dict):
            return sum(self._result_num_bytes(item) for item in value.values())
        return 0

    def _move_result_to_gpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(device="cuda", non_blocking=True)
        if isinstance(value, paddle.Tensor):
            return value.cuda()
        if isinstance(value, list):
            return [self._move_result_to_gpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_result_to_gpu(item) for item in value)
        if isinstance(value, dict):
            return {key: self._move_result_to_gpu(item) for key, item in value.items()}
        return value

    def _try_restore_spilled_results_to_gpu(self, results):
        required_bytes = sum(self._result_num_bytes(value) for value in results)
        if required_bytes <= 0:
            return None
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            available_bytes = free_bytes
            memory_budget = float(getattr(self.gpu_mode_config, "memory_budget", 0.0) or 0.0)
            if memory_budget > 0:
                reserved_bytes = torch.cuda.memory_reserved()
                budget_headroom = max(
                    0,
                    int(memory_budget * (1024**3)) - reserved_bytes,
                )
                available_bytes = min(available_bytes, budget_headroom)
            # Leave 20% headroom for allocator and comparison temporaries.
            if required_bytes * 5 > available_bytes * 4:
                return None
            restored = tuple(self._move_result_to_gpu(value) for value in results)
            return restored
        except Exception:
            return None

    def _clear_runtime_inputs(self, framework):
        for attr_name in (f"{framework}_args", f"{framework}_kwargs"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        gc.collect()
        if self.use_gpu_mode:
            gpu_mode_maybe_empty_cache(
                self.gpu_mode_config,
                f"accuracy_stable_after_{framework}",
            )
        elif framework == "torch":
            torch.cuda.empty_cache()
        else:
            paddle.device.cuda.empty_cache()

    def _clear_config_tensor_cache(self):
        def clear(value):
            if isinstance(value, TensorConfig):
                value.paddle_tensor = None
                value.torch_tensor = None
            elif isinstance(value, (list, tuple)):
                for item in value:
                    clear(item)

        for value in self.api_config.args:
            clear(value)
        for value in self.api_config.kwargs.values():
            clear(value)

    def _broadcast_to_comp_dimensions(self, log_type, affected_comps):
        """将执行阶段错误广播到所有受影响的 comp 维度"""
        for comp in affected_comps:
            write_to_comp_log(comp, log_type, self.api_config.config)

    def _reset_random_state(self, seed: int = 42):
        """Reset numpy / paddle / torch (CPU+CUDA) RNGs so random APIs
        (uniform, normal, randn, bernoulli, dropout, ...) produce
        reproducible outputs across the torch run and the paddle run."""
        numpy.random.seed(seed)
        try:
            paddle.seed(seed)
            if paddle.device.is_compiled_with_cuda():
                try:
                    for i in range(paddle.device.cuda.device_count()):
                        paddle.framework.core.default_cuda_generator(i).manual_seed(seed)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def test(self):
        if self.need_skip():
            print(f"[skip] {self.api_config.config}", flush=True)
            write_to_log("skip", self.api_config.config)
            return

        if not self.ana_api_info():
            print("ana_api_info failed", flush=True)
            write_to_log("config_parse", self.api_config.config)
            return

        try:
            convert_result = self.converter.convert(self.api_config.api_name)
        except Exception as e:
            print(
                f"[config_convert] Conversion failed for {self.api_config.config}: {e!s}",
                flush=True,
            )
            write_to_log("config_convert", self.api_config.config)
            return
        if not convert_result.is_supported:
            print(
                f"[config_convert] Unsupported API {self.api_config.api_name}: {convert_result.error_message}",
                flush=True,
            )
            write_to_log("config_convert", self.api_config.config)
            return
        if not convert_result.code or not convert_result.code.is_valid():
            print(
                f"[config_convert] No code generated for {self.api_config.api_name}",
                flush=True,
            )
            write_to_log("config_convert", self.api_config.config)
            return

        try:
            if not self.gen_numpy_input():
                print("gen_numpy_input failed")
                write_to_log("config_input", self.api_config.config)
                return
        except Exception as err:
            log_type, fatal = self.report_runtime_error(err, "config_input", "gen_numpy_input")
            if fatal:
                raise
            return

        # GPU-mode input analysis may materialize a paired Paddle/Torch tensor.
        # Discard that pre-seed cache so both stable-test iterations generate
        # their input pair after _reset_random_state().
        self._clear_config_tensor_cache()

        torch_output_pair = []
        torch_grad_pair = []
        paddle_output_pair = []
        paddle_grad_pair = []

        # iter twice
        for _i in range(2):
            # ======== torch ========
            self._reset_random_state()
            torch_output, torch_out_grads, torch_grad_success = self.get_torch_output(
                convert_result, _i
            )
            if torch_output is None:
                return
            torch_output = self._detach_result(torch_output)
            torch_out_grads = self._detach_result(torch_out_grads)
            self._clear_runtime_inputs("torch")

            # ======== paddle ========
            self._reset_random_state()
            paddle_output, paddle_out_grads = self.get_paddle_output(torch_grad_success, _i)
            if paddle_output is None:
                return
            paddle_output = self._detach_result(paddle_output)
            paddle_out_grads = self._detach_result(paddle_out_grads)
            self._clear_runtime_inputs("paddle")

            # ======== format ========
            paddle_output, torch_output = process_output(
                self.api_config, paddle_output, torch_output
            )
            paddle_out_grads, torch_out_grads = process_grad_output(
                self.api_config, paddle_out_grads, torch_out_grads
            )

            # ======== add to pair ========
            # if torch_grad_success = False, out_grads = [] and compare return
            torch_output_pair.append(torch_output)
            torch_grad_pair.append(torch_out_grads)
            paddle_output_pair.append(paddle_output)
            paddle_grad_pair.append(paddle_out_grads)

            if _i == 0:
                self.compare(torch_output_pair[0], paddle_output_pair[0], "T1P1")
                self.compare(torch_grad_pair[0], paddle_grad_pair[0], "T1P1B")
                if self.use_gpu_mode:
                    torch_output_pair[0] = self._move_result_to_cpu(torch_output_pair[0])
                    paddle_output_pair[0] = self._move_result_to_cpu(paddle_output_pair[0])
                    torch_grad_pair[0] = self._move_result_to_cpu(torch_grad_pair[0])
                    paddle_grad_pair[0] = self._move_result_to_cpu(paddle_grad_pair[0])
                    torch_output = None
                    paddle_output = None
                    torch_out_grads = None
                    paddle_out_grads = None
                    gc.collect()
                    gpu_mode_maybe_empty_cache(
                        self.gpu_mode_config,
                        "accuracy_stable_after_first_compare_spill",
                        force=True,
                    )

        if self.use_gpu_mode:
            restored_results = self._try_restore_spilled_results_to_gpu(
                (
                    torch_output_pair[0],
                    paddle_output_pair[0],
                    torch_grad_pair[0],
                    paddle_grad_pair[0],
                )
            )
            if restored_results is not None:
                (
                    torch_output_pair[0],
                    paddle_output_pair[0],
                    torch_grad_pair[0],
                    paddle_grad_pair[0],
                ) = restored_results

        # ======== summary ========
        self.compare(torch_output_pair[1], paddle_output_pair[1], "T2P2")
        self.compare(torch_grad_pair[1], paddle_grad_pair[1], "T2P2B")
        self.compare(torch_output_pair[0], paddle_output_pair[1], "T1P2")
        self.compare(torch_grad_pair[0], paddle_grad_pair[1], "T1P2B")
        self.compare(torch_output_pair[1], paddle_output_pair[0], "T2P1")
        self.compare(torch_grad_pair[1], paddle_grad_pair[0], "T2P1B")
        self.compare(torch_output_pair[0], torch_output_pair[1], "T1T2")
        torch_output_pair.clear()
        self.compare(torch_grad_pair[0], torch_grad_pair[1], "T1T2B")
        torch_grad_pair.clear()
        gc.collect()
        if self.use_gpu_mode:
            gpu_mode_maybe_empty_cache(
                self.gpu_mode_config,
                "accuracy_stable_after_torch_compare",
            )
        self.compare(paddle_output_pair[0], paddle_output_pair[1], "P1P2")
        paddle_output_pair.clear()
        self.compare(paddle_grad_pair[0], paddle_grad_pair[1], "P1P2B")
        paddle_grad_pair.clear()

        # 逐维度写 pass
        for dimension in ALL_DIMENSIONS:
            if not has_comp_terminal_log(dimension, self.api_config.config):
                # 取该维度的任一 comp 代表写 pass
                rep_comp = next(c for c, d in COMP_TO_DIMENSION.items() if d == dimension)
                write_to_comp_log(rep_comp, "pass", self.api_config.config)
        # 主日志 pass（让 engine 的 has_terminal_log 能正确判断）
        if not has_terminal_log(self.api_config.config):
            print(f"[pass] {self.api_config.config}", flush=True)
            write_to_log("pass", self.api_config.config)

    def get_torch_output(self, convert_result, iter_idx=0):
        # ======== run torch forward ========:
        torch_output = None
        try:
            if not self.gen_torch_input():
                print("gen_torch_input failed", flush=True)
                return None, None, None

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
            err_str = str(err)
            if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                print(f"[oom] {self.api_config.config}\n{err_str}", flush=True)
                self._broadcast_to_comp_dimensions("oom", self._TORCH_AFFECTED_COMPS[iter_idx])
                raise
            print(f"[torch_error] {self.api_config.config}\n{err_str}", flush=True)
            traceback.print_exc()
            self._broadcast_to_comp_dimensions("torch_error", self._TORCH_AFFECTED_COMPS[iter_idx])
            if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
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
                        f"[oom] phase=backward {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions("oom", self._TORCH_AFFECTED_COMPS[iter_idx])
                    raise
                if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
                    print(
                        f"[torch_error] phase=backward {self.api_config.config}\n{err_str}",
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
                    f"[torch_error] phase=backward {self.api_config.config}\n{err_str}",
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
            err_str = str(err)
            if self.should_ignore_paddle_error(err_str):
                print(f"[pass] {self.api_config.config}", flush=True)
                self._broadcast_to_comp_dimensions("pass", self._PADDLE_AFFECTED_COMPS[iter_idx])
                return None, None
            if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
                print(f"[paddle_cuda] {self.api_config.config}\n{err_str}", flush=True)
                self._broadcast_to_comp_dimensions(
                    "paddle_cuda", self._PADDLE_AFFECTED_COMPS[iter_idx]
                )
                raise
            if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                print(f"[oom] {self.api_config.config}\n{err_str}", flush=True)
                self._broadcast_to_comp_dimensions("oom", self._PADDLE_AFFECTED_COMPS[iter_idx])
                raise
            print(f"[paddle_error] {self.api_config.config}\n{err_str}", flush=True)
            traceback.print_exc()
            self._broadcast_to_comp_dimensions(
                "paddle_error", self._PADDLE_AFFECTED_COMPS[iter_idx]
            )
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
                        f"[paddle_cuda] phase=backward {self.api_config.config}\n{err_str}",
                    )
                    self._broadcast_to_comp_dimensions(
                        "paddle_cuda", self._PADDLE_AFFECTED_COMPS[iter_idx]
                    )
                    raise
                if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                    print(
                        f"[oom] phase=backward {self.api_config.config}\n{err_str}",
                        flush=True,
                    )
                    self._broadcast_to_comp_dimensions("oom", self._PADDLE_AFFECTED_COMPS[iter_idx])
                    raise
                print(
                    f"[paddle_error] phase=backward {self.api_config.config}\n{err_str}",
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
                    f"[paddle_cuda] phase=backward {self.api_config.config}\n{err!s}",
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
                    self.assert_accuracy(input1, input2, comp)
                except Exception as err:
                    self.report_compare_error(err, f"comp={comp}")
                    return
            else:
                print(
                    f"[paddle_accuracy] comp={comp} {self.api_config.config}\nreason=not_compare,",
                    f"{type(input1)} / {type(input2)}",
                    flush=True,
                )
                write_to_comp_log(comp, "paddle_accuracy", self.api_config.config)
                return
        elif isinstance(input1, (list, tuple)):
            if not isinstance(input2, (list, tuple)):
                print(
                    f"[paddle_accuracy] comp={comp} {self.api_config.config}\nreason=not_compare,",
                    f"{type(input1)} / {type(input2)}",
                    flush=True,
                )
                write_to_comp_log(comp, "paddle_accuracy", self.api_config.config)
                return
            if len(input1) != len(input2):
                print(
                    f"[paddle_accuracy] comp={comp} {self.api_config.config}\nreason=not_compare,",
                    f"{type(input1)} : {len(input1)} /",
                    f"{type(input2)} : {len(input2)}",
                    flush=True,
                )
                write_to_comp_log(comp, "paddle_accuracy", self.api_config.config)
                return
            for idx, (item1, item2) in enumerate(zip(input1, input2, strict=False)):
                if isinstance(item1, (paddle.Tensor, torch.Tensor)) and isinstance(
                    item2, (paddle.Tensor, torch.Tensor)
                ):
                    try:
                        self.assert_accuracy(item1, item2, comp, idx)
                    except Exception as err:
                        self.report_compare_error(err, f"comp={comp} idx={idx}")
                        return
                elif not isinstance(item1, (paddle.Tensor, torch.Tensor)) and not isinstance(
                    item2, (paddle.Tensor, torch.Tensor)
                ):
                    try:
                        self.assert_accuracy(torch.tensor(item1), torch.tensor(item2), comp, idx)
                    except Exception as err:
                        self.report_compare_error(err, f"comp={comp} idx={idx}")
                        return
                else:
                    print(
                        f"[paddle_accuracy] comp={comp} {self.api_config.config}\nreason=not_compare",
                        f"{type(item1)} / {type(item2)}",
                        flush=True,
                    )
                    write_to_comp_log(comp, "paddle_accuracy", self.api_config.config)
                    return
        else:
            try:
                self.assert_accuracy(torch.tensor(input1), torch.tensor(input2), comp)
            except Exception as err:
                self.report_compare_error(err, f"comp={comp}")
                return

    def assert_accuracy(self, tensor1, tensor2, comp, idx=0):
        api_name = self.api_config.api_name
        config = self.api_config.config[:120000]
        dtype = self.api_config.dtype
        check_dtype = self.should_check_dtype()

        first = "Paddle" if comp[0] == "P" else "Torch"
        second = "Paddle" if comp[2] == "P" else "Torch"

        try:
            self.torch_assert_accuracy(
                tensor1,
                tensor2,
                atol=0.0,
                rtol=0.0,
                check_dtype=check_dtype,
                actual_name=first,
                expected_name=second,
                apply_special_tolerance=False,
            )
            log_accuracy_stable(
                "Identical",
                api_name,
                config,
                dtype,
                comp,
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
                log_accuracy_stable(
                    err_str,
                    api_name,
                    config,
                    dtype,
                    comp,
                )
                write_to_comp_log(comp, "paddle_bitwise", config)
            else:
                raise
