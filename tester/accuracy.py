from __future__ import annotations

import gc

import numpy
import paddle
import torch
import yaml

from .accuracy_common import process_grad_output, process_output
from .base import CUDA_OOM, APITestBase, GpuMemoryGuardSkip, gpu_mode_memory_decision
from .paddle_to_torch import ConversionKind, get_converter
from .paddle_to_torch.arguments import bind_paddle_arguments

_ACCURACY_COMPARISON_WORKSPACE_BYTES = 256 * 1024**2


class APITestAccuracy(APITestBase):
    input_operation_mode = "accuracy"

    def __init__(self, api_config, **kwargs):
        super().__init__(api_config, runtime_config=kwargs.get("runtime_config"))
        self.test_amp = kwargs.get("test_amp", False)
        self.atol = kwargs.get("atol", 0)
        self.rtol = kwargs.get("rtol", 0)
        self.record_accuracy_tolerance = kwargs.get("record_accuracy_tolerance", False)
        self.bitwise_alignment = kwargs.get(
            "bitwise_alignment", self.runtime_config.bitwise_alignment
        )
        self.use_gpu_mode = self.gpu_mode_config.enabled
        self.use_dual_gpu = self.use_gpu_mode and self.gpu_mode_config.dual_gpu
        self.comparison_device_id = self.gpu_mode_config.comparison_device_id
        self.accuracy_manual_threshold_config = kwargs.get("accuracy_manual_threshold_config", "")
        self.bitwise_knows_threshold_config = self._load_manual_threshold_config(
            # 文件缺失或格式错误必须显式失败，不能静默跳过。
            self.accuracy_manual_threshold_config
        )
        if self.accuracy_manual_threshold_config:
            # 手动阈值只用于严格比较失败后的已知差异复核。
            self.atol = 0.0
            self.rtol = 0.0
            self.bitwise_alignment = True
        # known 结果延迟到整个 case 成功后写入，避免后续梯度失败造成重复终态。
        self._bitwise_knows_detected = False
        # Torch 输入/前向失败的延迟终态；合并判定完成后才写盘，保证每个 case 只有一条终态。
        self._torch_failure = None
        # Paddle 侧在合并模式下的延迟失败；fatal 错误仍按既有机制立即上抛。
        self._paddle_failure = None
        # 合并路径中若某一步已写出终态（fatal/skip），后续不得再合并，否则产生重复分类。
        self._merge_terminal_written = False
        if self.record_accuracy_tolerance:
            torch.set_printoptions(profile="short")
        self.converter = get_converter()

    def _accuracy_comparison_memory_state(self):
        return self.gpu_memory_state(
            self.comparison_device_id,
            budget_gib=self.gpu_mode_config.comparison_memory_budget,
        )

    def _release_accuracy_gpu_cache(self, device_id):
        # accuracy 的清理边界覆盖两个 framework，避免一侧 allocator 阻塞另一侧后续分配。
        self.release_framework_gpu_cache(device_id=device_id, collect_cycles=True)

    def _ensure_accuracy_comparison_headroom(self, value):
        # 已在比较卡上的叶子不产生新副本，预算只统计实际需要搬运的唯一 Tensor。
        copy_leaves = tuple(
            tensor
            for tensor in self.iter_unique_tensor_tree_leaves(value)
            if not self.tensor_is_gpu(tensor)
            or self.tensor_gpu_device_id(tensor) != self.comparison_device_id
        )
        value_bytes = self.tensor_tree_nbytes(copy_leaves)
        if value_bytes <= 0:
            return 0

        memory_state = self._accuracy_comparison_memory_state()
        target_free_bytes = (
            memory_state.reserve_bytes + value_bytes + _ACCURACY_COMPARISON_WORKSPACE_BYTES
        )
        if memory_state.free_bytes < target_free_bytes:
            self._release_accuracy_gpu_cache(self.comparison_device_id)
            memory_state = self._accuracy_comparison_memory_state()
        if memory_state.free_bytes <= memory_state.reserve_bytes + value_bytes:
            raise GpuMemoryGuardSkip(
                "accuracy comparison GPU capacity guard: result copy would consume the safe reserve"
            )
        return value_bytes

    def _move_accuracy_result_to_comparison_gpu(self, value):
        """搬运一组 accuracy 结果；复制型 OOM 不应污染算子正确性分类。"""
        if self._ensure_accuracy_comparison_headroom(value) <= 0:
            return value
        try:
            return self.move_tensor_tree_to_gpu(value, self.comparison_device_id)
        except Exception as err:
            if not any(marker.lower() in str(err).lower() for marker in CUDA_OOM):
                raise
            try:
                self._release_accuracy_gpu_cache(self.comparison_device_id)
            except Exception:
                pass
            raise GpuMemoryGuardSkip(
                "accuracy comparison GPU capacity guard: result copy failed after cache release"
            ) from err

    def _compare_accuracy_tree_on_comparison_gpu(self, actual, expected):
        # Torch comparison kernel 的 current device 必须与结果卡一致，临时量才能落在正确预算中。
        with torch.cuda.device(int(self.comparison_device_id)):
            return self._compare_accuracy_tree(actual, expected)

    def _cleanup_accuracy_dual_gpu(self, *, force_cache_release):
        """清除单次 accuracy 的跨卡资源；终态日志仍由原执行路径负责。"""
        for attr_name in ("torch_args", "torch_kwargs", "paddle_args", "paddle_kwargs"):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.clear_output_grad_cache()
        self.clear_generated_input_values()
        gc.collect()

        # 失败可能发生在部分搬运之后，必须强制归还两张卡的空闲 allocator block。
        if force_cache_release:
            for device_id in (0, self.comparison_device_id):
                try:
                    self._release_accuracy_gpu_cache(device_id)
                except Exception:
                    pass
            return

        # 成功路径仅在比较卡接近安全余量时清 cache，避免小 case 每轮都同步 allocator。
        try:
            memory_state = self._accuracy_comparison_memory_state()
            if memory_state.free_bytes <= memory_state.reserve_bytes:
                self._release_accuracy_gpu_cache(self.comparison_device_id)
        except Exception:
            pass

    @staticmethod
    def _load_manual_threshold_config(accuracy_manual_threshold_config):
        if not accuracy_manual_threshold_config:
            return {}
        with open(accuracy_manual_threshold_config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        thresholds = config.get("manual_threshold_config") or {}
        if not isinstance(thresholds, dict):
            raise ValueError("manual_threshold_config must be a mapping")
        normalized = {}
        for api_name, threshold in thresholds.items():
            if (
                not isinstance(api_name, str)
                or not isinstance(threshold, (list, tuple))
                or len(threshold) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in threshold
                )
                or any(value < 0 for value in threshold)
            ):
                raise ValueError(
                    f"invalid manual threshold for {api_name!r}: expected [atol, rtol] "
                    "with two non-negative numbers"
                )
            normalized[api_name] = (float(threshold[0]), float(threshold[1]))
        return normalized

    def _threshold_api_name(self):
        if self.api_config.api_name == "paddle._C_ops._run_custom_op":
            return self.paddle_args[0]
        return self.api_config.api_name

    def _assert_with_bitwise_knows(self, assertion, *, atol, rtol):
        """严格比较失败后，仅对 YAML 命中的 API 使用已知容差复核。"""
        # known 分类要求严格阈值和有效的非零手动阈值。
        threshold = self.bitwise_knows_threshold_config.get(self._threshold_api_name())
        if (atol, rtol) != (0.0, 0.0) or threshold is None or threshold == (0.0, 0.0):
            assertion(atol, rtol)
            return

        try:
            assertion(atol, rtol)
        except AssertionError as strict_error:
            try:
                assertion(*threshold)
            except AssertionError:
                # 超出已知容差仍保留严格比较的诊断，维持原有失败语义。
                raise strict_error
            self._bitwise_knows_detected = True

    def _should_spill_torch_result_tree(
        self, convert_result, torch_output, torch_out_grads, probe_bytes
    ):
        # CPU 算子结果不会占用 GPU allocator，不应触发 GPU mode 的输出迁移决策。
        if not self.tensor_tree_has_gpu_tensor((torch_output, torch_out_grads)):
            return False
        # 保留输出树与参考工作区的联合预算，避免比较阶段瞬时超出显存余量。
        retained_tree_bytes = self.tensor_tree_nbytes((torch_output, torch_out_grads))
        reference_workspace_bytes = self._reference_workspace_bytes(convert_result)
        return gpu_mode_memory_decision(
            self.gpu_mode_config,
            request_spill=True,
            probe_bytes=probe_bytes,
            retained_tree_bytes=retained_tree_bytes,
            required_headroom_bytes=(
                probe_bytes
                + retained_tree_bytes
                + max(retained_tree_bytes, reference_workspace_bytes)
            ),
        ).should_spill

    def _prepare_torch_result_tree(self, value, *, keep_on_device):
        # Torch 结果树的驻留策略必须先于递归转换确定，避免中途产生不可回收副本。
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
                (
                    key,
                    self._prepare_torch_result_tree(item, keep_on_device=keep_on_device),
                )
                for key, item in value.items()
            )
        return value

    def _report_runtime_error_and_finalize(
        self,
        err,
        default_log_type,
        stage,
        *,
        force_log_type=None,
    ):
        log_type, fatal = self.report_runtime_error(
            err,
            default_log_type,
            stage,
            force_log_type=force_log_type,
        )
        self.dump_finalize(log_type or default_log_type)
        # 终态上报后立即释放共享 output-grad，避免错误 case 延长 GPU 生命周期。
        self.clear_output_grad_cache()
        return log_type, fatal

    def _report_comparison_error(self, err, tensor_index=0, tensor_count=1):
        # 比较阶段不携带 Paddle 或 Torch 前缀，避免混淆执行端与比较端。
        stage = self.STAGE_COMPARE_BACKWARD if self.is_backward else self.STAGE_COMPARE_FORWARD
        log_type, fatal = self.report_runtime_error(
            err,
            "paddle_accuracy",
            stage,
            tensor_position=f"{tensor_index + 1}/{tensor_count}",
        )
        self.dump_finalize(log_type or "paddle_accuracy")
        # 比较失败也属于终态，不能把 output-grad seed 留给后续 case。
        self.clear_output_grad_cache()
        if fatal:
            raise err

    def _report_structure_error(
        self,
        reason,
        *,
        tensor_position=None,
        **details,
    ):
        # 比较输出沿用同一阶段协议，错误和精度结果保持一致。
        stage = self.STAGE_COMPARE_BACKWARD if self.is_backward else self.STAGE_COMPARE_FORWARD
        detail_text = " | ".join(
            f"{key.replace('_', ' ')} {value}" for key, value in details.items()
        )
        self.report_case_result(
            "paddle_accuracy",
            reason.replace("_", " "),
            stage=stage,
            tensor_position=tensor_position,
            error=detail_text or None,
        )
        self.dump_finalize("paddle_accuracy")
        # 结构不匹配不会进入统一 compare 收尾，因此这里单独清理缓存。
        self.clear_output_grad_cache()

    def _compare_accuracy_tree(self, actual, expected, tensor_index=0, tensor_count=None):
        # 递归比较沿用统一的索引上下文，便于错误日志定位到嵌套输出位置。
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
                    self._assert_with_bitwise_knows(
                        lambda atol, rtol: self.torch_assert_accuracy(
                            left,
                            right,
                            atol=atol,
                            rtol=rtol,
                            tensor_index=index,
                            tensor_count=count,
                        ),
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                except GpuMemoryGuardSkip:
                    # 比较卡容量不足属于资源准入结果，不能写成数值精度失败。
                    raise
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
                self._assert_with_bitwise_knows(
                    lambda atol, rtol: self.np_assert_accuracy(
                        numpy.array(left),
                        numpy.array(right),
                        atol=atol,
                        rtol=rtol,
                    ),
                    atol=self.atol,
                    rtol=self.rtol,
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
            log_type, fatal = self.report_runtime_error(err, "config_input", self.STAGE_INPUT)
            self.dump_finalize(log_type or "config_input")
            if fatal:
                raise
            return False
        return True

    def _record_torch_failure_or_raise(
        self,
        err,
        default_log_type,
        stage,
        *,
        force_log_type=None,
    ):
        """记录 Torch 失败，返回 "raise"、"done" 或 "deferred"。

        fatal 维持既有立即上抛语义；skip 表示无法有效验证，写终态后直接结束；
        其余延迟到 Paddle 执行后由合并判定给出唯一终态。
        """
        log_type, fatal, err_msg = self.describe_runtime_error(err, default_log_type, stage)
        if force_log_type is not None:
            log_type = force_log_type
        if fatal:
            # OOM 与 CUDA 错误必须保持立即终态：显存和 worker 状态在失败后不可信。
            self.report_case_result(log_type, stage=stage, error=err_msg)
            self.dump_finalize(log_type)
            self.clear_output_grad_cache()
            self._merge_terminal_written = True
            return "raise"
        if log_type == "skip":
            # 未实现（(Unimplemented)）不是可比较的失败，继续跑 Paddle 也无法得出有效结论。
            self.report_case_result("skip", stage=stage, error=err_msg)
            self.dump_finalize("skip")
            self.clear_output_grad_cache()
            self._merge_terminal_written = True
            return "done"
        # 延迟到 paddle 执行后再写唯一终态，避免 torch_error 与合并分类同时落盘。
        self._torch_failure = (log_type, err_msg, stage)
        return "deferred"

    def get_torch_output(self, convert_result):
        try:
            device = self.torch_operator_device()
            torch.set_default_device(device)
            self.dump_event("torch_input_start")
            if not self.build_torch_input():
                # 延迟终态：torch 失败后仍需执行 paddle 才能判定 torch_only_error。
                self._torch_failure = ("torch_error", "build_torch_input failed", self.STAGE_INPUT)
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
                    with torch.autocast(device_type=device.type):
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

        except Exception as err:
            # Torch 前向和同步分开记录，异步错误仍能保留方向。
            outcome = self._record_torch_failure_or_raise(
                err, "torch_error", self.STAGE_TORCH_FORWARD
            )
            if outcome == "raise":
                raise
            return False, None, None, False

        # 单独同步才能把异步 Torch CUDA 错误归入正确的日志和重试分类。
        try:
            self.check_torch_operator_cuda_error()
        except Exception as err:
            # 前向同步失败与输入/前向失败共用延迟终态，fatal 仍立即上抛。
            outcome = self._record_torch_failure_or_raise(
                err,
                "torch_error",
                self.STAGE_TORCH_FORWARD_SYNC,
                force_log_type="torch_error",
            )
            if outcome == "raise":
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
            (
                result_outputs,
                result_outputs_grads,
            ) = self.gen_torch_output_and_output_grad(torch_output)
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
                # 与输入/前向一致：先延迟，再由 Paddle 结果决定 config_input 还是 both_error。
                outcome = self._record_torch_failure_or_raise(
                    err,
                    "config_input",
                    self.STAGE_TORCH_BACKWARD,
                    force_log_type="config_input",
                )
                if outcome == "raise":
                    raise
                return False, None, None, False
            # 反向失败不再立即终态，继续执行 Paddle 后合并出唯一分类。
            outcome = self._record_torch_failure_or_raise(
                err, "torch_error", self.STAGE_TORCH_BACKWARD
            )
            if outcome == "raise":
                raise
            return False, None, None, False
        try:
            self.check_torch_operator_cuda_error()
        except Exception as err:
            outcome = self._record_torch_failure_or_raise(
                err,
                "torch_error",
                self.STAGE_TORCH_BACKWARD_SYNC,
                force_log_type="torch_error",
            )
            if outcome == "raise":
                raise
            return False, None, None, False
        return True, torch_output, torch_out_grads, torch_grad_success

    def _prepare_torch_results_for_paddle(
        self,
        convert_result,
        torch_output,
        torch_out_grads,
        torch_grad_success,
        probe_bytes,
    ):
        if self.use_dual_gpu:
            # 双卡模式不走 CPU spill：Torch 完整结果迁到比较卡，计算卡只保留后续输入源。
            torch_output = self._prepare_torch_result_tree(torch_output, keep_on_device=True)
            if torch_grad_success:
                torch_out_grads = self._prepare_torch_result_tree(
                    torch_out_grads,
                    keep_on_device=True,
                )
            torch_output, torch_out_grads = self._move_accuracy_result_to_comparison_gpu(
                (torch_output, torch_out_grads)
            )
            gc.collect()
            self.clear_torch_tensor(
                probe_bytes=probe_bytes,
                required_headroom_bytes=probe_bytes,
            )
            return torch_output, torch_out_grads

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
            self.clear_torch_tensor(
                force=not keep_torch_outputs_on_device,
                probe_bytes=probe_bytes,
            )
        else:
            self.release_framework_gpu_cache("torch")
        return torch_output, torch_out_grads

    def get_paddle_output(self):
        # Paddle 侧必须在 Torch 结果清理后开始，避免两套 runtime 同时持有峰值数据。
        try:
            if not self.build_paddle_input():
                if self._torch_failure is not None:
                    # 合并模式下先记录，由统一出口决定 both_error/torch_only_error。
                    self._paddle_failure = (
                        "paddle_error",
                        "build_paddle_input failed",
                        self.STAGE_INPUT,
                    )
                    return False, None
                self.report_case_result(
                    "paddle_error", "build_paddle_input failed", stage=self.STAGE_INPUT
                )
                self.clear_output_grad_cache()
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
            if self._torch_failure is not None:
                outcome = self._record_paddle_failure_or_raise(
                    err, "paddle_error", self.STAGE_PADDLE_FORWARD
                )
                if outcome == "raise":
                    raise
                return False, None
            _, fatal = self._report_runtime_error_and_finalize(
                err,
                "paddle_error",
                self.STAGE_PADDLE_FORWARD,
                force_log_type=self.paddle_refusal_log_type(err),
            )
            if fatal:
                raise
            return False, None

        try:
            self.dump_save("paddle_forward_output", paddle_output, framework="paddle")
            self.dump_event("paddle_forward_done")
            self.check_paddle_kernel_cuda_error()
        except Exception as err:
            if self._torch_failure is not None:
                # CUDA 同步错误属于 fatal，helper 内部会先写终态再要求上抛。
                outcome = self._record_paddle_failure_or_raise(
                    err, "paddle_cuda", self.STAGE_PADDLE_FORWARD_SYNC
                )
                if outcome == "raise":
                    raise
                return False, None
            self._report_runtime_error_and_finalize(
                err, "paddle_cuda", self.STAGE_PADDLE_FORWARD_SYNC
            )
            raise
        return True, paddle_output

    def _record_paddle_failure_or_raise(
        self,
        err,
        default_log_type,
        stage,
        *,
        force_log_type=None,
    ):
        """合并模式下记录 Paddle 失败，返回 "raise" 或 "deferred"。

        OOM/CUDA 等 fatal 写唯一终态后上抛，交给既有 worker 协议；其余（含 Paddle
        的 skip/未实现）一律延迟到统一出口参与合并判定，因此归入 both_error。
        """
        # 显存守卫不属于算子结果，必须走 test() 的既有 OOM 入口，不能吞进合并分类。
        if isinstance(err, GpuMemoryGuardSkip):
            raise err
        log_type, fatal, err_msg = self.describe_runtime_error(err, default_log_type, stage)
        if force_log_type is not None:
            log_type = force_log_type
        if fatal or log_type == "oom":
            self.report_case_result(log_type, stage=stage, error=err_msg)
            self.dump_finalize(log_type)
            self.clear_output_grad_cache()
            self._merge_terminal_written = True
            return "raise"
        self._paddle_failure = (log_type, err_msg, stage)
        return "deferred"

    def _check_paddle_backward_outputs(self, inputs_list, result_outputs):
        """输入需要梯度却没有可求导输出时，拒绝把跳过反向当作反向成功。"""
        if not inputs_list or result_outputs:
            return
        # grad 开关被关闭时所有输出都被标记为 stop_gradient，反向实际没有执行。
        if not paddle.is_grad_enabled():
            raise RuntimeError(
                "Paddle grad mode is disabled, all outputs are stop_gradient and backward was skipped"
            )
        # Torch 已产出输入梯度时，Paddle 无可求导输出本身就是两侧分歧。
        if self._torch_failure is None:
            raise RuntimeError(
                "Paddle produced no differentiable outputs while Torch produced input grads"
            )

    def get_paddle_grad(self, paddle_output):
        paddle_out_grads = None
        try:
            inputs_list = self.get_paddle_input_list()
            result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(
                paddle_output
            )
            self.enforce_paddle_backward_capacity(
                inputs_list,
                result_outputs,
                result_outputs_grads,
            )
            del self.paddle_args, self.paddle_kwargs
            self._check_paddle_backward_outputs(inputs_list, result_outputs)
            if inputs_list and result_outputs and result_outputs_grads:
                paddle_out_grads = paddle.grad(
                    result_outputs,
                    inputs_list,
                    grad_outputs=result_outputs_grads,
                    allow_unused=True,
                )
            del inputs_list, result_outputs, result_outputs_grads
        except GpuMemoryGuardSkip as err:
            # 合并模式下不能在此写终态，否则会与随后合并出的终态重复；交给 test() 统一上报。
            if self._torch_failure is not None:
                raise
            self.report_case_result("oom", stage=self.STAGE_PADDLE_BACKWARD, message=str(err))
            self.clear_runtime_inputs("paddle")
            self.clear_output_grad_cache()
            self.dump_finalize("oom", memory_guard=str(err))
            return False, None
        except Exception as err:
            if str(err).startswith("Too large tensor to get cached numpy: "):
                if self._torch_failure is not None:
                    outcome = self._record_paddle_failure_or_raise(
                        err,
                        "config_input",
                        self.STAGE_PADDLE_BACKWARD,
                        force_log_type="config_input",
                    )
                    if outcome == "raise":
                        raise
                    return False, None
                self._report_runtime_error_and_finalize(
                    err,
                    "config_input",
                    self.STAGE_PADDLE_BACKWARD,
                    force_log_type="config_input",
                )
                return False, None
            if self._torch_failure is not None:
                outcome = self._record_paddle_failure_or_raise(
                    err,
                    "paddle_error",
                    self.STAGE_PADDLE_BACKWARD,
                )
                if outcome == "raise":
                    raise
                return False, None
            _, fatal = self._report_runtime_error_and_finalize(
                err,
                "paddle_error",
                self.STAGE_PADDLE_BACKWARD,
                force_log_type=self.paddle_refusal_log_type(err),
            )
            if fatal:
                raise
            return False, None

        try:
            self.check_paddle_kernel_cuda_error()
        except Exception as err:
            if self._torch_failure is not None:
                # 与前向同步保持一致：合并模式下统一走 handler，fatal 由它写终态并上抛。
                outcome = self._record_paddle_failure_or_raise(
                    err, "paddle_cuda", self.STAGE_PADDLE_BACKWARD_SYNC
                )
                if outcome == "raise":
                    raise
                return False, None
            self._report_runtime_error_and_finalize(
                err, "paddle_cuda", self.STAGE_PADDLE_BACKWARD_SYNC
            )
            raise
        return True, paddle_out_grads

    def test(self):
        dual_gpu_completed = False
        try:
            dual_gpu_completed = bool(self._test_accuracy())
        except GpuMemoryGuardSkip as err:
            self.report_case_result("oom", stage=self.STAGE_MEMORY_PREFLIGHT, message=str(err))
            self.dump_finalize("oom", memory_guard=str(err))
        finally:
            # _test_accuracy 返回后局部结果引用已经销毁，此时才能可靠清理 allocator cache。
            if self.use_dual_gpu:
                self._cleanup_accuracy_dual_gpu(force_cache_release=not dual_gpu_completed)

    def _test_paddle_after_torch_failure(self, probe_bytes):
        """Torch 输入/前向失败后继续执行 Paddle，并在最后合并出唯一终态。

        Torch 未产出参考值，因此本路径不比较精度，只区分「Paddle 能否执行」：
        失败侧若为 fatal（OOM/CUDA/崩溃/超时）由既有机制立即写终态并上抛，其余组合：
        - Paddle 成功 → torch_only_error（Torch 独有分歧，Paddle 更宽松或 Torch 有 gap）
        - 两侧均 config_input → config_input（两框架一致认定该配置非法）
        - 其余两者都失败 → both_error
        """
        # Torch 失败后其输入缓存不再需要，先释放再物化 Paddle 输入，避免双份驻留。
        self.clear_runtime_inputs("torch")
        if self.use_gpu_mode:
            # 与正常路径一致：进入 Paddle 前按当前压力决定是否回收 allocator cache。
            gpu_mode_memory_decision(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
            )
        paddle_success, paddle_output = self.get_paddle_output()
        if paddle_success and self.need_check_grad():
            self.is_backward = True
            paddle_success, _ = self.get_paddle_grad(paddle_output)

        if self._merge_terminal_written:
            # fatal/skip 已写唯一终态，不允许再合并出第二条分类。
            self._torch_failure = None
            self._paddle_failure = None
            return
        final_log_type = self._merge_torch_paddle_failure(paddle_success)
        # 合并终态必须同时保留两侧原始错误，否则日志只剩分类名无法定位。
        merge_error = self._format_merge_error()
        # stage 取 torch 失败阶段：合并终态的归因主体是 torch 侧，paddle 细节保留在 error 文本里。
        merge_stage = self._torch_failure[2]
        self._torch_failure = None
        self._paddle_failure = None
        self.clear_output_grad_cache()
        self.report_case_result(final_log_type, stage=merge_stage, error=merge_error)
        self.dump_finalize(final_log_type)

    def _format_merge_error(self):
        """把两侧错误拼成一条可追溯的诊断文本，不改变分类本身。"""
        torch_log_type, torch_msg, torch_stage = self._torch_failure
        lines = [f"[torch] {torch_stage} {torch_log_type}: {torch_msg}"]
        if self._paddle_failure is not None:
            paddle_log_type, paddle_msg, paddle_stage = self._paddle_failure
            lines.append(f"[paddle] {paddle_stage} {paddle_log_type}: {paddle_msg}")
        else:
            lines.append("[paddle] forward/backward succeeded")
        return "\n".join(lines)

    def _merge_torch_paddle_failure(self, paddle_success):
        """按优先级合并两侧结果；fatal 已在各自阶段写终态，不会到达这里。"""
        if paddle_success:
            return "torch_only_error"
        torch_log_type = self._torch_failure[0]
        paddle_log_type = self._paddle_failure[0] if self._paddle_failure else "paddle_error"
        # 只有两侧都被明确识别为配置非法时才归 config_input，保持既有信号语义。
        if torch_log_type == "config_input" and paddle_log_type == "config_input":
            return "config_input"
        return "both_error"

    def _test_accuracy(self):
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
        memory_mode = "accuracy_dual_gpu" if self.use_dual_gpu else "accuracy"
        if not self.run_gpu_memory_preflight(memory_mode):
            return
        if not self._generate_input_values():
            return
        probe_bytes = self.estimate_input_bytes()

        (
            torch_success,
            torch_output,
            torch_out_grads,
            torch_grad_success,
        ) = self.get_torch_output(convert_result)
        if not torch_success:
            if self._torch_failure is not None:
                # Torch 没有参考值，无法比较精度；继续执行 Paddle 只能判定其能否运行。
                self._test_paddle_after_torch_failure(probe_bytes)
            return
        torch_output, torch_out_grads = self._prepare_torch_results_for_paddle(
            convert_result,
            torch_output,
            torch_out_grads,
            torch_grad_success,
            probe_bytes,
        )
        if self.use_gpu_mode:
            # caller 重绑定后旧 graph 才失去最后引用，仅在此时出现压力才清 allocator cache。
            gpu_mode_memory_decision(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
            )
        paddle_success, paddle_output = self.get_paddle_output()
        if not paddle_success:
            return

        try:
            paddle_output, torch_output = process_output(
                self.api_config, paddle_output, torch_output
            )
        except Exception as err:
            _, fatal = self._report_runtime_error_and_finalize(
                err, "paddle_accuracy", self.STAGE_COMPARE_FORWARD
            )
            if fatal:
                raise
            return

        self.is_backward = False
        if self.use_gpu_mode:
            gpu_mode_memory_decision(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
            )
        if self.use_dual_gpu:
            # Paddle 原结果继续持有 backward graph；比较卡只接收 detached 快照。
            paddle_compare_output = self._move_accuracy_result_to_comparison_gpu(
                self.detach_tensor_tree(paddle_output)
            )
            if not self._compare_accuracy_tree_on_comparison_gpu(
                paddle_compare_output,
                torch_output,
            ):
                return
            # 前向结果在 backward 前结束生命周期，比较卡只继续保留 Torch 输入梯度。
            paddle_compare_output = None
            torch_output = None
            gc.collect()
        elif not self._compare_accuracy_tree(paddle_output, torch_output):
            return

        # Forward check now pass.
        # Then do paddle backward and backward result check.
        if self.use_gpu_mode and not self.use_dual_gpu:
            del torch_output
            gpu_mode_memory_decision(
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
                    err, "paddle_accuracy", self.STAGE_COMPARE_BACKWARD
                )
                if fatal:
                    raise
                return

            # Paddle backward 已完成，原输出 graph 不应与梯度比较副本同时占用计算卡。
            if self.use_dual_gpu:
                paddle_output = None
                paddle_out_grads = self._move_accuracy_result_to_comparison_gpu(
                    self.detach_tensor_tree(paddle_out_grads)
                )
                gc.collect()
            if self.use_gpu_mode:
                gpu_mode_memory_decision(
                    self.gpu_mode_config,
                    probe_bytes=probe_bytes,
                )
            if self.use_dual_gpu:
                if not self._compare_accuracy_tree_on_comparison_gpu(
                    paddle_out_grads,
                    torch_out_grads,
                ):
                    return
            elif not self._compare_accuracy_tree(paddle_out_grads, torch_out_grads):
                return

        # backward compare 已经消费完两侧共享的 output-grad seed。
        self.clear_output_grad_cache()
        final_log_type = "paddle_bitwise_knows" if self._bitwise_knows_detected else "pass"
        self.report_case_result(final_log_type)
        self.dump_finalize(final_log_type)
        return True
