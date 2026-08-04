from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy
import paddle
import torch

from .accuracy_common import process_grad_output, process_output
from .base import (
    CUDA_ERROR,
    CUDA_OOM,
    APITestBase,
    GpuMemoryGuardSkip,
    gpu_mode_memory_decision,
)
from .log_writer import log_comparison, log_worker
from .paddle_to_torch import ConversionKind, get_converter
from .paddle_to_torch.arguments import bind_paddle_arguments

_GIB = 1024**3
_RESULT_STREAM_WORKSPACE_BYTES = 256 * 1024**2
_SUMMARY_COMPARISON_WORKSPACE_BYTES = _GIB


@dataclass
class _StableResultPairs:
    torch_outputs: list = field(default_factory=list)
    torch_grads: list = field(default_factory=list)
    paddle_outputs: list = field(default_factory=list)
    paddle_grads: list = field(default_factory=list)

    def append(self, torch_output, torch_grads, paddle_output, paddle_grads):
        self.torch_outputs.append(torch_output)
        self.torch_grads.append(torch_grads)
        self.paddle_outputs.append(paddle_output)
        self.paddle_grads.append(paddle_grads)

    def clear_all(self):
        self.torch_outputs.clear()
        self.torch_grads.clear()
        self.paddle_outputs.clear()
        self.paddle_grads.clear()

    def clear_forward(self):
        self.torch_outputs.clear()
        self.paddle_outputs.clear()


@dataclass
class _StableExecutionState:
    first_pair_comparison: object | None = None
    first_pair_finished: bool = False
    phased_result_residency: bool = False
    first_iteration_spilled: bool = False
    probe_bytes: int = 0


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

    @staticmethod
    def _normalize_torch_result(value):
        if isinstance(value, (torch.return_types.max, torch.return_types.min)):
            value = value.values
        if isinstance(value, (list, tuple)):
            value = list(value)
        return value

    @staticmethod
    def _normalize_paddle_result(value):
        if isinstance(value, (list, tuple)):
            value = list(value)
        return value

    def compare_first_pair(self, paddle_output, torch_output, paddle_grad, torch_grad):
        def run():
            self.compare(paddle_output, torch_output, "P1T1")
            self.compare(paddle_grad, torch_grad, "P1T1B")

        self._run_with_torch_device(self.comparison_device_id, run)

    def _compare_many_on_comparison_gpu(self, comparisons):
        def run():
            for left, right, comp in comparisons:
                self.compare(left, right, comp)

        self._run_with_torch_device(self.comparison_device_id, run)

    def _comparison_gpu_memory_state(self):
        return self.gpu_memory_state(
            self.comparison_device_id,
            budget_gib=self.gpu_mode_config.comparison_memory_budget,
        )

    def _release_compute_gpu_cache(self, framework=None):
        self.release_framework_gpu_cache(framework, device_id=0, collect_cycles=True)

    def _release_comparison_gpu_cache(self):
        self.release_framework_gpu_cache(device_id=self.comparison_device_id, collect_cycles=True)

    def _run_with_torch_device(self, device_id, callback):
        device_id = int(device_id)
        if not torch.cuda.is_available():
            current_device = None
        else:
            try:
                current_device = torch.cuda.current_device()
            except Exception:
                current_device = None
        if current_device == device_id:
            return callback()
        with torch.cuda.device(device_id):
            return callback()

    def _manage_compute_headroom(
        self,
        required_headroom_bytes,
        framework=None,
        *,
        enforce=False,
    ):
        """按计算卡 headroom 释放 cache，并可在分配前强制校验。"""
        memory_state = self.gpu_memory_state(0, budget_gib=self.gpu_mode_config.memory_budget)
        required_headroom_bytes = int(required_headroom_bytes or 0)
        under_pressure = (
            memory_state.free_bytes <= memory_state.reserve_bytes + required_headroom_bytes
        )
        if under_pressure:
            # 只在命中压力阈值时释放，正常小 Tensor 路径不承担跨框架 cache 开销。
            self._release_compute_gpu_cache(framework)
            if enforce:
                # 强校验必须在释放后重新读取物理 free，首次快照不能决定最终分类。
                memory_state = self.gpu_memory_state(
                    0,
                    budget_gib=self.gpu_mode_config.memory_budget,
                )
                under_pressure = (
                    memory_state.free_bytes <= memory_state.reserve_bytes + required_headroom_bytes
                )
        if enforce and under_pressure:
            raise GpuMemoryGuardSkip(
                "compute GPU capacity guard: known inputs exceed current safe headroom"
            )

    def _move_tensor_tree_to_comparison_gpu(self, value):
        # 所有双卡结果搬运集中经过此入口，避免某一结果族绕过物理 headroom 检查。
        if self._ensure_comparison_copy_headroom(value) <= 0:
            return value
        try:
            return self.move_tensor_tree_to_gpu(value, self.comparison_device_id)
        except Exception as err:
            err_str = str(err).lower()
            if not any(marker.lower() in err_str for marker in CUDA_OOM):
                raise
            # 复制 OOM 不涉及算子正确性；清 cache 后转换为可审计的容量 skip。
            try:
                self._release_comparison_gpu_cache()
            except Exception as cleanup_error:
                self._log_dual_cleanup_error(cleanup_error)
            raise GpuMemoryGuardSkip(
                "comparison GPU capacity guard: result copy failed after cache release"
            ) from err

    def _clear_execution_resources(self):
        for attr_name in (
            "torch_args",
            "torch_kwargs",
            "paddle_args",
            "paddle_kwargs",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.clear_output_grad_cache()
        self.clear_original_cpu_inputs()

    def _projected_results_need_phasing(self, projected_bytes):
        if not self.use_dual_gpu:
            return False
        memory_state = self._comparison_gpu_memory_state()
        allocator_margin = max(_GIB, int(projected_bytes) // 20)
        return (
            int(projected_bytes) + allocator_margin + _RESULT_STREAM_WORKSPACE_BYTES
            > memory_state.live_budget_bytes
        )

    def _prepare_second_torch_results(self, state, torch_result_bytes):
        """Reserve comparison-card space before copying the T2 result family."""
        memory_state = self._comparison_gpu_memory_state()
        free_bytes = memory_state.free_bytes
        reserve_bytes = memory_state.reserve_bytes
        target_free = reserve_bytes + int(torch_result_bytes) + _RESULT_STREAM_WORKSPACE_BYTES
        if free_bytes < target_free:
            # P1T1 has completed, so its reduction blocks are no longer live.
            self._release_comparison_gpu_cache()
            memory_state = self._comparison_gpu_memory_state()
            free_bytes = memory_state.free_bytes
            reserve_bytes = memory_state.reserve_bytes

        if free_bytes <= reserve_bytes + int(torch_result_bytes):
            raise GpuMemoryGuardSkip(
                "comparison GPU capacity guard: retained first results leave no room "
                "for the second Torch result family"
            )
        if free_bytes < target_free:
            if not state.phased_result_residency:
                self.record_memory_governance_metric("phased_result_residency")
            state.phased_result_residency = True

    def _place_second_iteration_results(self, pairs):
        output_bytes = self.tensor_tree_nbytes(pairs.paddle_outputs[1])
        grad_bytes = self.tensor_tree_nbytes(pairs.paddle_grads[1])
        memory_state = self._comparison_gpu_memory_state()
        free_bytes = memory_state.free_bytes
        reserve_bytes = memory_state.reserve_bytes
        new_bytes = output_bytes + grad_bytes
        if free_bytes < reserve_bytes + new_bytes + _RESULT_STREAM_WORKSPACE_BYTES:
            # A completed first comparison may leave allocator cache behind. A
            # smaller chunk is valid when the 256 MiB performance target cannot
            # be met, but the second result copy must not consume the reserve.
            self._release_comparison_gpu_cache()
            memory_state = self._comparison_gpu_memory_state()
            free_bytes = memory_state.free_bytes
            reserve_bytes = memory_state.reserve_bytes
        if free_bytes <= reserve_bytes + new_bytes:
            raise GpuMemoryGuardSkip(
                "comparison GPU capacity guard: second result family exceeds safe capacity"
            )
        self._stream_second_slot_to_comparison_gpu(
            pairs.paddle_outputs,
            1,
            release_compute_cache=False,
        )
        self._stream_second_slot_to_comparison_gpu(
            pairs.paddle_grads,
            1,
            release_compute_cache=False,
        )
        self._release_compute_gpu_cache()

    def _finish_first_pair_comparison(self, state):
        if state.first_pair_finished:
            return
        try:
            if state.first_pair_comparison is not None:
                state.first_pair_comparison.result()
        finally:
            state.first_pair_finished = True

    def _log_dual_cleanup_error(self, cleanup_error):
        print(
            f"[dual_gpu_cleanup_error] {self.api_config.config}\n{cleanup_error!s}",
            flush=True,
        )

    def _cleanup_comparison_stage(
        self,
        state,
        pairs,
        *,
        discard_results,
        cache_release_policy="none",
    ):
        """Join compare, drop result references, then release allocator cache when needed."""
        valid_cache_release_policies = {"none", "if_pressure", "always"}
        if cache_release_policy not in valid_cache_release_policies:
            raise ValueError(
                f"Unsupported cache release policy: {cache_release_policy!r}. "
                f"Expected one of {sorted(valid_cache_release_policies)!r}"
            )
        if not self.use_dual_gpu:
            if discard_results:
                pairs.clear_all()
            if cache_release_policy != "none" and self.use_gpu_mode:
                probe_bytes = getattr(state, "probe_bytes", 0)
                gpu_mode_memory_decision(
                    self.gpu_mode_config,
                    probe_bytes=probe_bytes,
                    required_headroom_bytes=probe_bytes,
                )
            return
        try:
            if state.first_pair_comparison is not None:
                self._finish_first_pair_comparison(state)
        except Exception:
            pairs.clear_all()
            try:
                self._release_comparison_gpu_cache()
            except Exception as cleanup_error:
                self._log_dual_cleanup_error(cleanup_error)
            raise
        if discard_results:
            pairs.clear_all()
        if cache_release_policy == "always" or (
            cache_release_policy == "if_pressure"
            and self._comparison_gpu_cache_needs_release(getattr(state, "probe_bytes", 0))
        ):
            self._release_comparison_gpu_cache()

    def _comparison_gpu_cache_needs_release(self, required_headroom_bytes):
        memory_state = self._comparison_gpu_memory_state()
        return memory_state.free_bytes <= memory_state.reserve_bytes + int(
            required_headroom_bytes or 0
        )

    def _abort_case_resources(self, state, pairs):
        try:
            self._cleanup_comparison_stage(
                state,
                pairs,
                discard_results=True,
                cache_release_policy="always",
            )
        except GpuMemoryGuardSkip:
            pass
        except Exception as cleanup_error:
            self._log_dual_cleanup_error(cleanup_error)
        try:
            self._clear_execution_resources()
        except Exception as cleanup_error:
            self._log_dual_cleanup_error(cleanup_error)
        if self.use_dual_gpu:
            try:
                self._release_compute_gpu_cache()
            except Exception as cleanup_error:
                self._log_dual_cleanup_error(cleanup_error)

    def _start_first_pair_comparison(self, pairs, state):
        comparison_executor = None
        try:
            comparison_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="accuracy-stable-compare",
            )
            state.first_pair_comparison = comparison_executor.submit(
                self.compare_first_pair,
                pairs.paddle_outputs[0],
                pairs.torch_outputs[0],
                pairs.paddle_grads[0],
                pairs.torch_grads[0],
            )
        finally:
            # Only one task is ever submitted. Closing submission immediately
            # lets the worker thread retire by itself when the future completes.
            if comparison_executor is not None:
                comparison_executor.shutdown(wait=False)

    def _spill_first_iteration_if_needed(self, pairs, convert_result, probe_bytes, state):
        torch_phase_bytes = self.tensor_tree_nbytes((pairs.torch_outputs[0], pairs.torch_grads[0]))
        paddle_phase_bytes = self.tensor_tree_nbytes(
            (pairs.paddle_outputs[0], pairs.paddle_grads[0])
        )
        retained_tree_bytes = torch_phase_bytes + paddle_phase_bytes
        projected_summary_bytes = 2 * retained_tree_bytes
        allocator_margin = max(_GIB, projected_summary_bytes // 20)
        reference_workspace = self._reference_workspace_bytes(convert_result)
        comparison_workspace = max(_SUMMARY_COMPARISON_WORKSPACE_BYTES, reference_workspace)
        required_headroom_bytes = (
            probe_bytes + max(torch_phase_bytes, paddle_phase_bytes) + reference_workspace
        )
        required_headroom_bytes = max(
            required_headroom_bytes,
            projected_summary_bytes + allocator_margin + comparison_workspace,
        )
        decision = gpu_mode_memory_decision(
            self.gpu_mode_config,
            request_spill=True,
            probe_bytes=probe_bytes,
            retained_tree_bytes=retained_tree_bytes,
            required_headroom_bytes=required_headroom_bytes,
        )
        if decision.should_spill or state.first_iteration_spilled:
            spill_results = [
                self.spill_tensor_tree_slot_to_cpu(pairs.torch_outputs, release_cache=False),
                self.spill_tensor_tree_slot_to_cpu(pairs.paddle_outputs, release_cache=False),
                self.spill_tensor_tree_slot_to_cpu(pairs.torch_grads, release_cache=False),
                self.spill_tensor_tree_slot_to_cpu(pairs.paddle_grads, release_cache=False),
            ]
            if any(spill_results):
                gpu_mode_memory_decision(self.gpu_mode_config, force=True)
                state.first_iteration_spilled = True

    def _compare_on_compute_gpu(self, input1, input2, comp):
        def run():
            self.compare(input1, input2, comp)

        self._run_with_torch_device(0, run)

    def _ensure_comparison_copy_headroom(self, value):
        # 已在目标卡上的结果再次经过归一化入口时是 no-op，不能重复占用复制预算。
        copy_leaves = tuple(
            tensor
            for tensor in self.iter_unique_tensor_tree_leaves(value)
            if not self.tensor_is_gpu(tensor)
            or self.tensor_gpu_device_id(tensor) != self.comparison_device_id
        )
        if not copy_leaves:
            return 0
        value_bytes = self.tensor_tree_nbytes(copy_leaves)
        if value_bytes <= 0:
            return 0
        memory_state = self._comparison_gpu_memory_state()
        free_bytes = memory_state.free_bytes
        reserve_bytes = memory_state.reserve_bytes
        if free_bytes < reserve_bytes + value_bytes + _RESULT_STREAM_WORKSPACE_BYTES:
            self._release_comparison_gpu_cache()
            memory_state = self._comparison_gpu_memory_state()
            free_bytes = memory_state.free_bytes
            reserve_bytes = memory_state.reserve_bytes
        if free_bytes <= reserve_bytes + value_bytes:
            raise GpuMemoryGuardSkip(
                "comparison GPU capacity guard: result copy would consume the safe reserve"
            )
        return value_bytes

    def _stream_second_slot_to_comparison_gpu(self, values, index, *, release_compute_cache=True):
        source = values[index]
        values[index] = self._move_tensor_tree_to_comparison_gpu(source)
        del source
        if release_compute_cache:
            self._release_compute_gpu_cache()

    def _drop_second_comparison_slot(self, values, index):
        source = values[index]
        values[index] = None
        del source

    def _run_phased_dual_summary_comparisons(self, pairs):
        self._compare_on_compute_gpu(pairs.paddle_outputs[1], pairs.torch_outputs[1], "P2T2")

        self._stream_second_slot_to_comparison_gpu(pairs.paddle_outputs, 1)
        self._compare_many_on_comparison_gpu(
            [
                (pairs.paddle_outputs[1], pairs.torch_outputs[0], "P2T1"),
                (pairs.paddle_outputs[0], pairs.paddle_outputs[1], "P1P2"),
            ]
        )
        self._drop_second_comparison_slot(pairs.paddle_outputs, 1)

        self._stream_second_slot_to_comparison_gpu(pairs.torch_outputs, 1)
        self._compare_many_on_comparison_gpu(
            [
                (pairs.paddle_outputs[0], pairs.torch_outputs[1], "P1T2"),
                (pairs.torch_outputs[0], pairs.torch_outputs[1], "T1T2"),
            ]
        )
        self._drop_second_comparison_slot(pairs.torch_outputs, 1)
        pairs.clear_forward()

        self._compare_on_compute_gpu(pairs.paddle_grads[1], pairs.torch_grads[1], "P2T2B")

        self._stream_second_slot_to_comparison_gpu(pairs.paddle_grads, 1)
        self._compare_many_on_comparison_gpu(
            [
                (pairs.paddle_grads[1], pairs.torch_grads[0], "P2T1B"),
                (pairs.paddle_grads[0], pairs.paddle_grads[1], "P1P2B"),
            ]
        )
        self._drop_second_comparison_slot(pairs.paddle_grads, 1)

        self._stream_second_slot_to_comparison_gpu(pairs.torch_grads, 1)
        self._compare_many_on_comparison_gpu(
            [
                (pairs.paddle_grads[0], pairs.torch_grads[1], "P1T2B"),
                (pairs.torch_grads[0], pairs.torch_grads[1], "T1T2B"),
            ]
        )
        self._drop_second_comparison_slot(pairs.torch_grads, 1)
        pairs.torch_grads.clear()
        pairs.paddle_grads.clear()

    def _run_summary_comparisons(self, pairs, state):
        if self.use_dual_gpu and state.phased_result_residency:
            self._run_phased_dual_summary_comparisons(pairs)
            return

        if self.use_gpu_mode and state.first_iteration_spilled:

            def restore_first_slot(values):
                source = values[0]
                value_bytes = self.tensor_tree_nbytes(source)
                if value_bytes > 0:
                    memory_state = self.gpu_memory_state(
                        0,
                        budget_gib=self.gpu_mode_config.memory_budget,
                    )
                    if memory_state.free_bytes < (
                        memory_state.reserve_bytes + value_bytes + _RESULT_STREAM_WORKSPACE_BYTES
                    ):
                        self._release_compute_gpu_cache()
                        memory_state = self.gpu_memory_state(
                            0,
                            budget_gib=self.gpu_mode_config.memory_budget,
                        )
                    if memory_state.free_bytes > memory_state.reserve_bytes + value_bytes:
                        try:
                            values[0] = self.move_tensor_tree_to_gpu(source, 0)
                            del source
                            self.record_memory_governance_metric("cpu_restore")
                        except Exception as err:
                            err_str = str(err).lower()
                            if not any(marker.lower() in err_str for marker in CUDA_OOM):
                                raise
                            self._release_compute_gpu_cache()
                return values[0]

            def clear_first_slot(values):
                source = values[0]
                values[0] = None
                del source
                self._release_compute_gpu_cache()

            self.compare(pairs.paddle_outputs[1], pairs.torch_outputs[1], "P2T2")
            restore_first_slot(pairs.paddle_outputs)
            self.compare(pairs.paddle_outputs[0], pairs.torch_outputs[1], "P1T2")
            self.compare(pairs.paddle_outputs[0], pairs.paddle_outputs[1], "P1P2")
            clear_first_slot(pairs.paddle_outputs)
            restore_first_slot(pairs.torch_outputs)
            self.compare(pairs.paddle_outputs[1], pairs.torch_outputs[0], "P2T1")
            self.compare(pairs.torch_outputs[0], pairs.torch_outputs[1], "T1T2")
            clear_first_slot(pairs.torch_outputs)
            pairs.clear_forward()

            self.compare(pairs.paddle_grads[1], pairs.torch_grads[1], "P2T2B")
            restore_first_slot(pairs.paddle_grads)
            self.compare(pairs.paddle_grads[0], pairs.torch_grads[1], "P1T2B")
            self.compare(pairs.paddle_grads[0], pairs.paddle_grads[1], "P1P2B")
            clear_first_slot(pairs.paddle_grads)
            restore_first_slot(pairs.torch_grads)
            self.compare(pairs.paddle_grads[1], pairs.torch_grads[0], "P2T1B")
            self.compare(pairs.torch_grads[0], pairs.torch_grads[1], "T1T2B")
            clear_first_slot(pairs.torch_grads)
            pairs.torch_grads.clear()
            pairs.paddle_grads.clear()
            return

        # Finish all forward comparisons before touching the backward result
        # family, so output residency does not overlap later grad diagnostics.
        self.compare(pairs.paddle_outputs[1], pairs.torch_outputs[1], "P2T2")
        self.compare(pairs.paddle_outputs[1], pairs.torch_outputs[0], "P2T1")
        self.compare(pairs.paddle_outputs[0], pairs.torch_outputs[1], "P1T2")
        self.compare(pairs.torch_outputs[0], pairs.torch_outputs[1], "T1T2")
        self.compare(pairs.paddle_outputs[0], pairs.paddle_outputs[1], "P1P2")
        pairs.clear_forward()

        # Backward comparisons run after the forward result family is gone.
        self.compare(pairs.paddle_grads[1], pairs.torch_grads[1], "P2T2B")
        self.compare(pairs.paddle_grads[1], pairs.torch_grads[0], "P2T1B")
        self.compare(pairs.paddle_grads[0], pairs.torch_grads[1], "P1T2B")
        self.compare(pairs.torch_grads[0], pairs.torch_grads[1], "T1T2B")
        pairs.torch_grads.clear()
        self.compare(pairs.paddle_grads[0], pairs.paddle_grads[1], "P1P2B")
        pairs.paddle_grads.clear()

    def _run_stable_iteration(self, iter_idx, convert_result, probe_bytes, pairs, state):
        state.probe_bytes = probe_bytes

        # ======== torch ========
        self.reset_random_state()
        try:
            if self.use_dual_gpu:
                self._manage_compute_headroom(probe_bytes, "torch", enforce=True)
            torch_output, torch_out_grads, torch_grad_success = self.get_torch_output(
                convert_result, iter_idx
            )
        except Exception:
            self._abort_case_resources(state, pairs)
            raise
        if torch_output is None:
            self._abort_case_resources(state, pairs)
            return False

        torch_output = self.detach_tensor_tree(torch_output)
        torch_out_grads = self.detach_tensor_tree(torch_out_grads)
        self.clear_runtime_inputs("torch")
        keep_second_results_on_compute = (
            self.use_dual_gpu and iter_idx == 1 and state.phased_result_residency
        )
        if self.use_dual_gpu and iter_idx == 1:
            # Do not put T2 results on the comparison card while P1 compare
            # still owns its reduction workspace. If the case is not already
            # phased, preflight the complete copy before moving either tree.
            try:
                self._cleanup_comparison_stage(
                    state,
                    pairs,
                    discard_results=False,
                    cache_release_policy="none",
                )
                if not state.phased_result_residency:
                    self._prepare_second_torch_results(
                        state,
                        self.tensor_tree_nbytes((torch_output, torch_out_grads)),
                    )
                    keep_second_results_on_compute = state.phased_result_residency
            except Exception:
                self._abort_case_resources(state, pairs)
                raise

        if self.use_dual_gpu:
            try:
                if not keep_second_results_on_compute:
                    torch_output = self._move_tensor_tree_to_comparison_gpu(torch_output)
                    torch_out_grads = self._move_tensor_tree_to_comparison_gpu(torch_out_grads)
            except Exception:
                self._abort_case_resources(state, pairs)
                raise
            self._manage_compute_headroom(probe_bytes, "torch")
        elif self.use_gpu_mode:
            torch_live_bytes = self.tensor_tree_nbytes((torch_output, torch_out_grads))
            # Release idle Torch blocks before the next Paddle execution;
            # the two frameworks do not share caching allocators.
            decision = gpu_mode_memory_decision(
                self.gpu_mode_config,
                request_spill=iter_idx == 0,
                probe_bytes=probe_bytes,
                retained_tree_bytes=torch_live_bytes,
                required_headroom_bytes=(
                    probe_bytes
                    + torch_live_bytes
                    + max(torch_live_bytes, self._reference_workspace_bytes(convert_result))
                ),
            )
            if iter_idx == 0 and decision.should_spill:
                torch_output = self.move_tensor_tree_to_cpu(torch_output)
                torch_out_grads = self.move_tensor_tree_to_cpu(torch_out_grads)
                state.first_iteration_spilled = True
                gpu_mode_memory_decision(self.gpu_mode_config, force=True)

        # ======== paddle ========
        self.reset_random_state()
        try:
            if self.use_dual_gpu:
                self._manage_compute_headroom(probe_bytes, "paddle", enforce=True)
            paddle_output, paddle_out_grads = self.get_paddle_output(torch_grad_success, iter_idx)
        except Exception:
            self._abort_case_resources(state, pairs)
            raise
        if paddle_output is None:
            self._abort_case_resources(state, pairs)
            return False

        try:
            # Normalize API-specific output quirks before detach/move/byte
            # accounting touches uninitialized Paddle placeholders.
            paddle_output, torch_output = process_output(
                self.api_config, paddle_output, torch_output
            )
            paddle_out_grads, torch_out_grads = process_grad_output(
                self.api_config, paddle_out_grads, torch_out_grads
            )
        except Exception:
            self._abort_case_resources(state, pairs)
            raise

        paddle_output = self.detach_tensor_tree(paddle_output)
        paddle_out_grads = self.detach_tensor_tree(paddle_out_grads)
        self.clear_runtime_inputs("paddle")
        if self.use_gpu_mode and not self.use_dual_gpu:
            gpu_mode_memory_decision(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
                required_headroom_bytes=probe_bytes,
            )

        if self.use_dual_gpu:
            # Formatting may create replacement tensors. Normalize the first
            # result set immediately; the second Paddle result set is placed
            # only after the residency planner sees its exact byte sizes.
            try:
                if not keep_second_results_on_compute:
                    torch_output = self._move_tensor_tree_to_comparison_gpu(torch_output)
                    torch_out_grads = self._move_tensor_tree_to_comparison_gpu(torch_out_grads)
                if iter_idx == 0:
                    paddle_output = self._move_tensor_tree_to_comparison_gpu(paddle_output)
                    paddle_out_grads = self._move_tensor_tree_to_comparison_gpu(paddle_out_grads)
            except Exception:
                self._abort_case_resources(state, pairs)
                raise
            self._manage_compute_headroom(probe_bytes, "paddle")

        # if torch_grad_success = False, out_grads = [] and compare return
        pairs.append(torch_output, torch_out_grads, paddle_output, paddle_out_grads)

        # Pair lists own the results from here onward. Drop loop-local aliases
        # before residency planning so relocated sources can actually be freed.
        torch_output = None
        paddle_output = None
        torch_out_grads = None
        paddle_out_grads = None

        if self.use_dual_gpu:
            try:
                if iter_idx == 0:
                    first_result_bytes = self.tensor_tree_nbytes(
                        (
                            pairs.torch_outputs[0],
                            pairs.torch_grads[0],
                            pairs.paddle_outputs[0],
                            pairs.paddle_grads[0],
                        )
                    )
                    projected_phased = self._projected_results_need_phasing(2 * first_result_bytes)
                    if projected_phased and not state.phased_result_residency:
                        self.record_memory_governance_metric("phased_result_residency")
                    state.phased_result_residency = projected_phased
                else:
                    # P2 backward is the last consumer of the shared output-grad
                    # seeds. Release them before summary streams second results.
                    self.clear_output_grad_cache()
                    self._release_compute_gpu_cache()
                    if not state.phased_result_residency:
                        self._place_second_iteration_results(pairs)
            except Exception:
                self._abort_case_resources(state, pairs)
                raise

        if iter_idx != 0:
            return True

        if self.use_dual_gpu:
            try:
                self._start_first_pair_comparison(pairs, state)
            except Exception:
                self._abort_case_resources(state, pairs)
                raise
            return True

        if self.use_gpu_mode:
            self._spill_first_iteration_if_needed(pairs, convert_result, probe_bytes, state)

        self.compare(pairs.paddle_outputs[0], pairs.torch_outputs[0], "P1T1")
        self.compare(pairs.paddle_grads[0], pairs.torch_grads[0], "P1T1B")
        return True

    def test(self):
        if self.need_skip():
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

        memory_mode = "accuracy_stable_dual_gpu" if self.use_dual_gpu else "accuracy_stable"
        if not self.run_gpu_memory_preflight(memory_mode):
            return

        try:
            if not self.generate_input_values():
                self.report_case_result("config_input", "generate_input_values failed")
                return
        except Exception as err:
            _, fatal = self.report_runtime_error(err, "config_input", "input")
            if fatal:
                raise
            return

        try:
            self.save_original_inputs_to_cpu()
            # 后续两轮只从不可变 CPU 副本重建，GPU 生成源在此结束生命周期。
            self.clear_generated_input_values()
        except Exception as err:
            _, fatal = self.report_runtime_error(err, "config_input", "input cache")
            if fatal:
                raise
            return

        probe_bytes = self.estimate_input_bytes()

        pairs = _StableResultPairs()
        execution_state = _StableExecutionState()

        # Every execution recreates its input from the same immutable CPU copy.
        try:
            for iter_idx in range(2):
                if not self._run_stable_iteration(
                    iter_idx, convert_result, probe_bytes, pairs, execution_state
                ):
                    return
        except GpuMemoryGuardSkip as err:
            self.report_case_result("skip", phase="memory_guard", message=str(err))
            return

        summary_failed = False
        try:
            self.clear_original_cpu_inputs()

            self._cleanup_comparison_stage(
                execution_state,
                pairs,
                discard_results=False,
                cache_release_policy="none",
            )

            self._run_summary_comparisons(pairs, execution_state)
            log_worker.write_stable_passes(self.api_config.config)
        except GpuMemoryGuardSkip as err:
            summary_failed = True
            self.report_case_result("skip", phase="memory_guard", message=str(err))
            return
        except Exception:
            summary_failed = True
            raise
        finally:
            # Summary comparisons can fail before the success-path cleanup.
            # Clear result families first, then release both comparison allocators.
            if summary_failed:
                self._abort_case_resources(execution_state, pairs)
            else:
                self._cleanup_comparison_stage(
                    execution_state,
                    pairs,
                    discard_results=True,
                    cache_release_policy="if_pressure",
                )

    def get_torch_output(self, convert_result, iter_idx=0):
        torch_output = None
        try:
            if not self.build_torch_input():
                self.report_case_result(
                    "torch_error",
                    "build_torch_input failed",
                    affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
                )
                return None, None, None

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

            with torch.set_grad_enabled(self.need_check_grad()):
                torch_output = self.converter.execute(
                    convert_result,
                    self.torch_args,
                    bound_arguments,
                    execution_locals=self._torch_execution_locals(),
                    core_executor=execute_core,
                )
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            _, fatal = self.report_runtime_error(
                err,
                "torch_error",
                "forward",
                affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
            )
            if fatal:
                raise
            return None, None, None

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
                        allow_unused=True,
                    )
                    torch_grad_success = True
            except Exception as err:
                err_str = str(err)
                if err_str.startswith("Too large tensor to get cached numpy: "):
                    self.report_runtime_error(
                        err,
                        "config_input",
                        "backward",
                        force_log_type="config_input",
                        affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
                    )
                    return None, None, None
                if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                    self.report_runtime_error(
                        err,
                        "oom",
                        "backward",
                        force_log_type="oom",
                        affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
                    )
                    raise
                if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
                    self.report_runtime_error(
                        err,
                        "torch_error",
                        "backward",
                        force_log_type="torch_error",
                        affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
                    )
                    raise
                _, fatal = self.report_runtime_error(
                    err,
                    "torch_error",
                    "backward",
                    affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
                )
                if fatal:
                    raise
                return None, None, None

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                self.report_runtime_error(
                    err,
                    "torch_error",
                    "backward cuda check",
                    force_log_type="torch_error",
                    affected_comps=self._TORCH_AFFECTED_COMPS[iter_idx],
                )
                raise

        torch_output = self._normalize_torch_result(torch_output)
        torch_out_grads = self._normalize_torch_result(torch_out_grads)
        return torch_output, torch_out_grads, torch_grad_success

    def get_paddle_output(self, torch_grad_success, iter_idx=0):
        paddle_output = None
        try:
            if not self.build_paddle_input():
                self.report_case_result(
                    "paddle_error",
                    "build_paddle_input failed",
                    affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                )
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
            if self.api_config.dtype is None:
                self.api_config.dtype = paddle.float32

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
                            paddle_output = self.paddle_api(
                                *self.paddle_args,
                                **self.paddle_kwargs,
                            )
                    else:
                        paddle_output = self.paddle_api(*self.paddle_args, **self.paddle_kwargs)
            if (
                self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__"
            ) or self.api_config.api_name == "paddle.Tensor.__setitem__":
                paddle_output = first_arg
        except Exception as err:
            _, fatal = self.report_runtime_error(
                err,
                "paddle_error",
                "forward",
                allow_ignore_paddle=True,
                affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
            )
            if fatal:
                raise
            return None, None

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            self.report_runtime_error(
                err,
                "paddle_cuda",
                "forward cuda check",
                force_log_type="paddle_cuda",
                affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
            )
            raise

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
                    self.report_runtime_error(
                        err,
                        "config_input",
                        "backward",
                        force_log_type="config_input",
                        affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                    )
                    return None, None
                if self.should_ignore_paddle_error(err_str):
                    self.report_runtime_error(
                        err,
                        "paddle_error",
                        "backward",
                        allow_ignore_paddle=True,
                        affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                    )
                    return None, None
                if any(cuda_err in err_str for cuda_err in CUDA_ERROR):
                    self.report_runtime_error(
                        err,
                        "paddle_cuda",
                        "backward",
                        force_log_type="paddle_cuda",
                        affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                    )
                    raise
                if any(cuda_err in err_str for cuda_err in CUDA_OOM):
                    self.report_runtime_error(
                        err,
                        "oom",
                        "backward",
                        force_log_type="oom",
                        affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                    )
                    raise
                self.report_runtime_error(
                    err,
                    "paddle_error",
                    "backward",
                    affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                )
                return None, None

            try:
                paddle.base.core.eager._for_test_check_cuda_error()
            except Exception as err:
                self.report_runtime_error(
                    err,
                    "paddle_cuda",
                    "backward cuda check",
                    force_log_type="paddle_cuda",
                    affected_comps=self._PADDLE_AFFECTED_COMPS[iter_idx],
                )
                raise

        paddle_output = self._normalize_paddle_result(paddle_output)
        paddle_out_grads = self._normalize_paddle_result(paddle_out_grads)
        return paddle_output, paddle_out_grads

    def report_stable_compare_error(
        self,
        err,
        comp,
        *,
        tensor_position=None,
        tensor_count=1,
    ):
        log_type, fatal = self.report_runtime_error(
            err,
            "paddle_accuracy",
            comp,
            tensor_position=tensor_position,
            affected_comps=(comp,),
        )
        log_worker._record_case_comparison(comp, log_type, 0, max(1, int(tensor_count or 1)))
        if fatal:
            raise err
        return log_type, fatal

    def _log_stable_identical_missing(self, input1, input2, comp, tensor_index, tensor_count):
        dtype = "None"
        if isinstance(input1, (paddle.Tensor, torch.Tensor)):
            dtype = str(input1.dtype)
        elif isinstance(input2, (paddle.Tensor, torch.Tensor)):
            dtype = str(input2.dtype)
        log_comparison.log_accuracy_stable(
            "Identical",
            self.api_config.api_name,
            self.api_config.config,
            dtype,
            comp,
            tensor_index=tensor_index,
            tensor_count=tensor_count,
        )

    def _compare_stable_leaf(self, input1, input2, comp, tensor_index, tensor_count):
        input1_missing = self.is_missing_compare_value(input1)
        input2_missing = self.is_missing_compare_value(input2)
        tensor_position = f"{tensor_index + 1}/{tensor_count}"
        if input1_missing and input2_missing:
            self._log_stable_identical_missing(
                input1,
                input2,
                comp,
                tensor_index,
                tensor_count,
            )
            return True
        if isinstance(input1, (paddle.Tensor, torch.Tensor)):
            if isinstance(input2, (paddle.Tensor, torch.Tensor)):
                try:
                    self.assert_accuracy(
                        input1,
                        input2,
                        comp,
                        tensor_index=tensor_index,
                        tensor_count=tensor_count,
                    )
                except GpuMemoryGuardSkip:
                    raise
                except Exception as err:
                    self.report_stable_compare_error(
                        err,
                        comp,
                        tensor_position=tensor_position,
                        tensor_count=tensor_count,
                    )
                    return False
                return True
            log_comparison.log_comp_issue(
                comp,
                "paddle_accuracy",
                self.api_config.config,
                tensor_index=tensor_index,
                tensor_count=tensor_count,
                reason="type_mismatch",
                actual_type=type(input1).__name__,
                expected_type=type(input2).__name__,
            )
            return False
        if isinstance(input2, (paddle.Tensor, torch.Tensor)):
            log_comparison.log_comp_issue(
                comp,
                "paddle_accuracy",
                self.api_config.config,
                tensor_index=tensor_index,
                tensor_count=tensor_count,
                reason="type_mismatch",
                actual_type=type(input1).__name__,
                expected_type=type(input2).__name__,
            )
            return False
        try:
            self.assert_accuracy(
                torch.tensor(input1),
                torch.tensor(input2),
                comp,
                tensor_index=tensor_index,
                tensor_count=tensor_count,
            )
        except Exception as err:
            self.report_stable_compare_error(
                err,
                comp,
                tensor_position=tensor_position,
                tensor_count=tensor_count,
            )
            return False
        return True

    def compare(self, input1, input2, comp):
        def compare_leaf(left, right, tensor_index, tensor_count):
            return self._compare_stable_leaf(left, right, comp, tensor_index, tensor_count)

        def report_structure_error(reason, *, tensor_index=0, tensor_count=1, **details):
            details.pop("tensor_position", None)
            log_comparison.log_comp_issue(
                comp,
                "paddle_accuracy",
                self.api_config.config,
                tensor_index=tensor_index,
                tensor_count=tensor_count,
                reason=reason,
                **details,
            )

        # stable 模式只复用结构遍历；叶子比较仍强制 exact compare（atol/rtol 均为 0）。
        self.compare_tensor_tree(
            input1,
            input2,
            compare_leaf,
            report_structure_error,
        )

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
