from __future__ import annotations

import collections
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import numpy
import paddle
import yaml

from .api_config.dtype_utils import to_torch_dtype
from .api_config.parameter_binding import bind_input_parameters, split_tensor_method_arguments
from .input_generation.generation_rules import input_rules
from .input_generation.materialization import (
    clear_tensor_configs,
    materialize_config_tree,
    tensor_config_tree_nbytes,
)
from .input_generation.tensor_config import (
    AUTOGRAD_DTYPES,
    TensorConfig,
    dtype_element_size,
    dtype_name,
)
from .reporting import log_worker
from .reporting.dump_writer import DEFAULT_DUMP_DIR, DumpContext, dump_enabled
from .reporting.log_comparison import log_accuracy_tolerance
from .reporting.log_schema import MAX_CSV_CONFIG_LENGTH
from .runtime.gpu_memory_preflight import (
    GpuMemoryDeferred,
    decide_gpu_memory_preflight,
    requires_inplace_input_copy,
    should_check_grad,
)

_TESTER_DIR = Path(__file__).resolve().parent

with (_TESTER_DIR / "base_config.yaml").open(encoding="utf-8") as f:
    config = yaml.safe_load(f)

forward_only_apis = frozenset(config.get("forward_only_apis", []))
not_check_dtype = frozenset(config.get("not_check_dtype", []))
rand_apis = frozenset(config.get("rand_apis", []))
stochastic_behavior_apis = frozenset(config.get("stochastic_behavior_apis", []))

paddle_error_dismiss = {}  # disabled: covered by the unified runtime error reporter
# paddle_error_dismiss = config.get("paddle_error_dismiss", {})
special_accuracy_atol_rtol = config.get("special_accuracy_atol_rtol", {})

with (_TESTER_DIR / "api_config" / "torch_error_skip.txt").open(encoding="utf-8") as f:
    torch_error_skip = frozenset(line.strip() for line in f if line.strip())

del config


class _LazyTorch:
    def __getattr__(self, name):
        import torch

        globals()["torch"] = torch
        return getattr(torch, name)


torch = _LazyTorch()


CUDA_ERROR = frozenset(
    [
        "CUDA error",
        "memory corruption",
    ]
)

CUDA_OOM = frozenset(
    [
        "CUDA out of memory",
        "Out of memory error",
        "ResourceExhaustedError",
        "out of memory",
        "OutOfMemoryError",
    ]
)


GPU_MEMORY_PROBE_MIN_BYTES = 256 << 20
COMPARISON_WORKSPACE_FAST_PATH_BYTES = 256 << 20
DEFAULT_COMPARISON_WORKSPACE_BYTES = 1 << 30
_GIB = 1024**3


def _dtype_element_size(dtype):
    return dtype_element_size(dtype, default=4)


def _tensor_element_size(value):
    try:
        return int(value.element_size())
    except (AttributeError, TypeError, ValueError):
        return _dtype_element_size(value.dtype)


def _dtype_name(value):
    return dtype_name(value)


@dataclass(frozen=True)
class GpuMemoryDecision:
    cleanup_performed: bool = False
    should_spill: bool = False
    free_before_bytes: int | None = None
    free_after_bytes: int | None = None
    required_headroom_bytes: int = 0
    pressure_before: bool = False
    pressure_after: bool = False


@dataclass(frozen=True)
class GpuMemoryState:
    device_id: int
    free_bytes: int
    total_bytes: int
    budget_bytes: int
    reserve_bytes: int
    live_budget_bytes: int


class GpuMemoryGuardSkip(RuntimeError):
    """运行时已知驻留集合无法安全放入目标 GPU。"""


@dataclass(frozen=True)
class OutputGradSlot:
    seed_numpy: object | None = None
    paddle_grad: object | None = None
    torch_grad: object | None = None


def _gpu_memory_is_under_pressure(gpu_config, free_bytes, total_bytes, required_headroom_bytes):
    workers_on_gpu = max(1, int(gpu_config.workers_on_gpu or 1))
    memory_budget_bytes = max(0, int(float(gpu_config.memory_budget or 0.0) * _GIB))
    device_budget_bytes = (
        memory_budget_bytes * workers_on_gpu if memory_budget_bytes > 0 else total_bytes
    )
    device_used_bytes = max(0, total_bytes - free_bytes)
    over_budget = bool(
        device_budget_bytes > 0
        and device_used_bytes >= int(device_budget_bytes * float(gpu_config.cleanup_used_ratio))
    )
    low_free = bool(
        device_budget_bytes > 0
        and free_bytes <= int(device_budget_bytes * float(gpu_config.cleanup_pressure_ratio))
    )
    insufficient_headroom = bool(
        required_headroom_bytes > 0 and free_bytes < required_headroom_bytes
    )
    return over_budget or low_free or insufficient_headroom


def _release_gpu_allocator_caches(torch_module=None):
    # torch_module=None 是 Paddle-only 协议值，不表示 Torch 查询失败后的隐式降级。
    gc.collect()
    if torch_module is not None:
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _query_gpu_memory(torch_module=None):
    try:
        if torch_module is not None:
            # Accuracy 需要查询 Tensor 实际所在的 CUDA runtime，仍由 Torch 提供快照。
            free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        else:
            # Paddle-only 不能借显存探测间接加载 Torch，统一使用当前 Paddle 设备。
            free_bytes = paddle.base.core.gpu_memory_available()
            total_bytes = paddle.device.cuda.get_device_properties().total_memory
        return int(free_bytes), int(total_bytes)
    except Exception:
        return None


def gpu_mode_memory_decision(
    gpu_config,
    force=False,
    request_spill=False,
    probe_bytes=None,
    retained_tree_bytes=0,
    required_headroom_bytes=None,
    use_torch=True,
):
    # 该入口统一承载 Paddle-only 与 Accuracy 两种显存治理协议。
    """Release idle allocator blocks and decide whether live result trees must spill."""
    if not gpu_config.enabled:
        return GpuMemoryDecision()

    probe_bytes = max(0, int(probe_bytes or 0))
    retained_tree_bytes = max(0, int(retained_tree_bytes or 0))
    if required_headroom_bytes is None:
        required_headroom_bytes = probe_bytes
    required_headroom_bytes = max(0, int(required_headroom_bytes or 0))
    decision_probe_bytes = max(probe_bytes, retained_tree_bytes, required_headroom_bytes)
    if not force and decision_probe_bytes < GPU_MEMORY_PROBE_MIN_BYTES:
        return GpuMemoryDecision(required_headroom_bytes=required_headroom_bytes)

    torch_module = None
    if use_torch:
        # 只有声明使用 Torch 的模式才允许触发依赖加载。
        try:
            import torch as torch_module
        except (ImportError, OSError):
            torch_module = None
    if use_torch and torch_module is None:
        if force:
            gc.collect()
            try:
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
            return GpuMemoryDecision(
                cleanup_performed=True,
                required_headroom_bytes=required_headroom_bytes,
            )
        return GpuMemoryDecision(required_headroom_bytes=required_headroom_bytes)

    if force:
        _release_gpu_allocator_caches(torch_module)
        return GpuMemoryDecision(
            cleanup_performed=True,
            required_headroom_bytes=required_headroom_bytes,
        )

    before = _query_gpu_memory(torch_module)
    pressure_before = before is None or _gpu_memory_is_under_pressure(
        gpu_config, *before, required_headroom_bytes
    )
    cleanup_performed = pressure_before
    after = before
    if cleanup_performed:
        _release_gpu_allocator_caches(torch_module)
        after = _query_gpu_memory(torch_module)

    # Unknown whole-device headroom is treated as pressure after cleanup.
    pressure_after = after is None or _gpu_memory_is_under_pressure(
        gpu_config, *after, required_headroom_bytes
    )
    should_spill = bool(request_spill and retained_tree_bytes > 0 and pressure_after)
    return GpuMemoryDecision(
        cleanup_performed=cleanup_performed,
        should_spill=should_spill,
        free_before_bytes=before[0] if before is not None else None,
        free_after_bytes=after[0] if after is not None else None,
        required_headroom_bytes=required_headroom_bytes,
        pressure_before=pressure_before,
        pressure_after=pressure_after,
    )


def classify_runtime_error(error_msg):
    """Classify runtime errors without printing or mutating log state."""
    error_msg_lower = error_msg.lower()
    if error_msg.startswith("[torch_assert_OOM]"):
        return "oom", False
    oom_markers = tuple(marker.lower() for marker in CUDA_OOM) + (
        "cannot allocate memory",
        "std::bad_alloc",
        "bad allocation",
        "memoryerror",
        "cublas_status_alloc_failed",
    )
    if any(marker in error_msg_lower for marker in oom_markers):
        return "oom", True
    cuda_markers = tuple(marker.lower() for marker in CUDA_ERROR) + (
        "illegal memory access",
        "invalid configuration argument",
        "invalid resource handle",
    )
    if any(marker in error_msg_lower for marker in cuda_markers):
        return "paddle_cuda", True
    # (Unimplemented): Paddle 当前功能无法由该 config 有效验证
    if "(unimplemented)" in error_msg_lower:
        return "skip", False
    # Paddle 输出数值检查失败
    if "there are nan or inf" in error_msg_lower or "check_numerics" in error_msg_lower:
        return "paddle_error", False
    # (InvalidArgument) / (PreconditionNotMet) / (OutOfRange): 输入/配置不满足前提
    if (
        "(invalidargument)" in error_msg_lower
        or "(preconditionnotmet)" in error_msg_lower
        or "(outofrange)" in error_msg_lower
    ):
        return "config_input", False
    # Torch-side equivalents of invalid configs (accuracy runs torch before paddle).
    torch_invalid_config_markers = (
        "out of bounds for dimension",
        "is invalid for input of size",
        "must match the size of tensor b",
        "does not match the shape of the indexed tensor",
    )
    if any(marker in error_msg_lower for marker in torch_invalid_config_markers):
        return "config_input", False
    return None, False


class APITestBase:
    def __init__(self, api_config, use_torch=True, runtime_config=None):
        # case 只能消费主进程冻结后的策略，不能在 worker 内根据环境重新选择 backend。
        if runtime_config is None:
            raise ValueError("runtime_config is required")
        self.api_config = api_config
        self.api_config.use_torch = use_torch
        self.use_torch = bool(use_torch)
        self.runtime_config = runtime_config
        self.gpu_mode_config = self.runtime_config.gpu_mode
        # TensorConfig 通过 case 配置读取算子设备，避免把 GPU mode 当成执行设备开关。
        self.api_config.test_cpu = self.runtime_config.test_cpu
        self.memory_governance_metrics = collections.Counter()
        self.dump_context = (
            DumpContext(
                os.environ.get("DUMP_DIR") or DEFAULT_DUMP_DIR,
                api_config=api_config.config,
            )
            if dump_enabled()
            else None
        )
        self.output_grad_slots = []
        self.output_grad_slots_signature = None
        from .input_generation.backend_runtime import reset_output_grad_streams

        reset_output_grad_streams()
        # 同一 case 的结果树共享设备快照，避免每个叶子重复查询 CUDA allocator。
        self.comparison_workspace_cache = {}
        if use_torch:
            torch.set_num_threads(8)
            torch.set_printoptions(threshold=100, linewidth=120)

    def torch_operator_device(self):
        """Torch reference 始终在 worker 绑定的计算卡执行。"""
        return torch.device(f"{self.runtime_config.torch_operator_device_type}:0")

    def check_paddle_kernel_cuda_error(self):
        """只为 Paddle GPU kernel 查询异步 CUDA 错误状态。"""
        if not self.runtime_config.test_cpu:
            paddle.base.core.eager._for_test_check_cuda_error()

    def check_torch_operator_cuda_error(self):
        """同步 Torch reference 的计算流，使异步 CUDA 错误在所属阶段上报。"""
        torch.cuda.synchronize(self.torch_operator_device())

    def check_operator_cuda_error(self):
        """仅执行 Paddle 算子的 tester。"""
        self.check_paddle_kernel_cuda_error()

    def requires_gpu_runtime(self):
        """算子执行或 GPU mode 任一需要 GPU 时返回 True。"""
        # use_torch 表示 tester 会执行 Torch reference，而不是仅用 Torch 做 CPU 结果表示。
        return self.use_torch or not self.runtime_config.test_cpu or self.gpu_mode_config.enabled

    def run_with_dump(self):
        """Execute the test with dump output capture and lifecycle reporting."""
        if self.dump_context is None:
            raise RuntimeError("run_with_dump() requires dump mode to be enabled")
        with self.dump_context.tee_output():
            try:
                return self.test()
            except GpuMemoryDeferred:
                # deferred 属于 worker 重试协议，不能固化为本次 case 的错误终态。
                raise
            except Exception as err:
                if self.dump_context._data.get("status") is None:
                    self.dump_finalize("engine_error", error=str(err))
                raise

    def dump_event(self, name, **data):
        if self.dump_context:
            self.dump_context.event(name, **data)

    def dump_error(self, name, err):
        if self.dump_context:
            self.dump_context.error_event(name, err)

    def dump_save(self, stem, obj, framework=None):
        if self.dump_context:
            self.dump_context.save_tensors(stem, obj, framework=framework)

    def dump_finalize(self, status, **data):
        if self.dump_context:
            memory_governance_metrics = getattr(self, "memory_governance_metrics", None)
            if memory_governance_metrics:
                data.setdefault(
                    "memory_governance_metrics",
                    dict(memory_governance_metrics),
                )
            self.dump_context.finalize(status, **data)

    def record_memory_governance_metric(self, name, amount=1):
        if not hasattr(self, "memory_governance_metrics"):
            self.memory_governance_metrics = collections.Counter()
        self.memory_governance_metrics[name] += int(amount)

    def report_runtime_error(
        self,
        err,
        default_log_type,
        phase,
        allow_ignore_paddle=False,
        *,
        tensor_position=None,
        force_log_type=None,
        affected_comps=None,
    ):
        """Report one runtime error and optionally mirror its log type to stable comp logs.

        force_log_type changes the emitted log bucket only; fatal/nonfatal still comes
        from classify_runtime_error().
        """
        err_msg = str(err)
        if phase:
            self.dump_error(f"{phase}_error", err)
        log_type, fatal = classify_runtime_error(err_msg)
        if log_type is None and allow_ignore_paddle and self.should_ignore_paddle_error(err_msg):
            log_type = "pass"
            self.report_case_result(log_type, affected_comps=affected_comps)
            return log_type, False
        if force_log_type is not None:
            log_type = force_log_type
        if log_type is None:
            log_type = default_log_type
        self.report_case_result(
            log_type,
            phase=phase,
            error=err_msg,
            tensor_position=tensor_position,
            affected_comps=affected_comps,
        )
        return log_type, fatal

    def report_case_result(
        self,
        log_type,
        message=None,
        *,
        phase=None,
        error=None,
        tensor_position=None,
        affected_comps=None,
        write_main_log=True,
    ):
        """Emit one standardized config-level result and write the matching logs."""
        log_worker.emit_case_result(
            log_type,
            self.api_config.config,
            phase=phase,
            message=message,
            error=error,
            tensor_position=tensor_position,
            affected_comps=affected_comps,
            write_main_log=write_main_log,
        )
        return log_type

    def run_gpu_memory_preflight(self, mode):
        """在输入生成前统一执行 GPU 容量准入，并记录可审计终态。"""
        input_backend = getattr(self.runtime_config, "input_backend_resolved", None)
        # 预检和实际生成必须共享同一个 resolved 名称，缺失时不能用默认值低估显存。
        if input_backend is None:
            raise ValueError("runtime config has no resolved input backend")
        if not self.gpu_mode_config.enabled:
            return True
        precomputed = getattr(self.runtime_config, "gpu_memory_estimate", None)
        if precomputed is None:
            # 单 case 入口或主进程估算失败时保留完整静态预检语义。
            decision = decide_gpu_memory_preflight(
                self.api_config,
                mode,
                self.gpu_mode_config,
                check_grad=self.need_check_grad(),
                paddle_kernel_on_gpu=not self.runtime_config.test_cpu,
                torch_operator_on_gpu=self.use_torch,
                input_backend=input_backend,
                input_source_on_gpu=(
                    input_backend != "numpy" and self.runtime_config.input_logical_device != "cpu"
                ),
            )
            if decision.should_skip:
                message = decision.message()
                self.record_memory_governance_metric("memory_preflight_oom")
                self.report_case_result("oom", phase="preflight", message=message)
                self.dump_finalize("oom", memory_preflight=message)
                return False
            compute_stages = tuple(
                stage
                for stage in getattr(decision.estimate, "stages", ())
                if stage.device == "compute" and stage.plan is None
            )
            required_headroom_bytes = max(
                (stage.total_bytes for stage in compute_stages),
                default=0,
            )
        else:
            # admission budget 已由相同估算派生，worker 只需确认当前物理 headroom。
            required_headroom_bytes = max(
                0,
                int(getattr(precomputed, "compute_headroom_bytes", 0)),
            )
        if required_headroom_bytes <= 0:
            return True
        runtime_decision = gpu_mode_memory_decision(
            self.gpu_mode_config,
            required_headroom_bytes=required_headroom_bytes,
            use_torch=self.use_torch,
        )
        if runtime_decision.cleanup_performed:
            self.record_memory_governance_metric("preflight_cache_release")
        physical_free_bytes = runtime_decision.free_after_bytes
        if physical_free_bytes is None or required_headroom_bytes <= physical_free_bytes:
            return True
        message = (
            f"mode={mode}, stage=runtime_headroom, device=compute, "
            f"estimated_peak={required_headroom_bytes / _GIB:.2f} GiB, "
            f"physical_free={physical_free_bytes / _GIB:.2f} GiB, "
            "basis=post_cleanup_physical_headroom"
        )
        # 动态物理显存竞争仍走 worker 重试协议，不能固化为本次 case 的 OOM 终态。
        self.record_memory_governance_metric("memory_preflight_defer")
        raise GpuMemoryDeferred(message)

    def reset_random_state(self, seed=None):
        """Reset NumPy and framework RNGs for reproducible executions."""
        if seed is None:
            seed = getattr(self.runtime_config, "random_seed", 42)
        seed = int(seed)
        numpy.random.seed(seed)
        if self.requires_gpu_runtime():
            paddle.seed(seed)
        else:
            # paddle.seed 会遍历所有 CUDA generator；纯 CPU 路径只播种 CPU generator。
            paddle.framework.core.default_cpu_generator().manual_seed(seed)
        if self.requires_gpu_runtime() and paddle.device.is_compiled_with_cuda():
            for device_id in range(paddle.device.cuda.device_count()):
                # 框架 generator 播种失败必须上抛，静默继续会破坏可复现性。
                paddle.framework.core.default_cuda_generator(device_id).manual_seed(seed)
        if self.use_torch:
            if self.requires_gpu_runtime():
                torch.manual_seed(seed)
            else:
                # torch.manual_seed 覆盖全部设备，纯 CPU 路径只更新默认 CPU generator。
                torch.random.default_generator.manual_seed(seed)
            if self.requires_gpu_runtime() and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def clear_runtime_inputs(self, framework):
        """Release one framework's generated inputs after an execution."""
        attr_names = [f"{framework}_args", f"{framework}_kwargs"]
        for attr_name in attr_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        if not self.gpu_mode_config.enabled:
            self.release_framework_gpu_cache(framework)

    def gpu_memory_state(self, device_id, *, budget_gib=0.0):
        """Return physical headroom and configured live budget for one CUDA device."""
        device_id = int(device_id)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        budget_bytes = max(0, int(float(budget_gib or 0.0) * _GIB))
        reserve_bytes = self._gpu_safety_reserve_bytes(total_bytes, budget_bytes)
        return GpuMemoryState(
            device_id=device_id,
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            budget_bytes=budget_bytes,
            reserve_bytes=reserve_bytes,
            live_budget_bytes=max(0, int(total_bytes) - reserve_bytes),
        )

    def release_framework_gpu_cache(
        self,
        framework=None,
        *,
        device_id=None,
        collect_cycles=False,
    ):
        """Release idle allocator blocks for one or both framework runtimes."""
        if framework not in (None, "torch", "paddle"):
            raise ValueError(f"Unsupported framework for GPU cache release: {framework!r}")
        if not self.requires_gpu_runtime():
            return
        self.record_memory_governance_metric("cache_release")
        if collect_cycles:
            gc.collect()
        if framework in (None, "torch"):
            if device_id is None:
                torch.cuda.empty_cache()
            else:
                with torch.cuda.device(int(device_id)):
                    torch.cuda.empty_cache()
        if framework in (None, "paddle"):
            if device_id is None:
                paddle.device.cuda.empty_cache()
                return
            current_device = paddle.device.get_device()
            try:
                paddle.device.set_device(f"gpu:{int(device_id)}")
                paddle.device.cuda.empty_cache()
            finally:
                paddle.device.set_device(current_device)

    def report_compare_error(
        self,
        err,
        phase,
        default_log_type="paddle_accuracy",
        *,
        tensor_position=None,
    ):
        log_type, fatal = self.report_runtime_error(
            err,
            default_log_type,
            phase,
            tensor_position=tensor_position,
        )
        if fatal:
            raise err
        return log_type, fatal

    def need_skip(self, paddle_only=False):
        if "sparse" in self.api_config.api_name:
            return True

        if not paddle_only and self.api_config.config in torch_error_skip:
            return True
        api_name = self.api_config.api_name
        if not paddle_only and (api_name in rand_apis or api_name in stochastic_behavior_apis):
            return True
        # float8 dtypes are handled by TensorConfig / paddle_to_torch Rules;
        # do not skip accuracy-mode comparison solely because of float8 inputs.

        return False

    def need_check_grad(self):
        # 主调度与 worker 共用纯配置判定，不能在两个生命周期各维护一份规则。
        return should_check_grad(self.api_config)

    def ana_api_info(self):
        return self.ana_paddle_api_info() and self.ana_torch_api_info()

    def ana_paddle_api_info(self):
        self.paddle_api = eval(self.api_config.api_name)
        self.paddle_args_config = self.api_config.args
        self.paddle_kwargs_config = self.api_config.kwargs
        return True

    def ana_torch_api_info(self):
        self.torch_args_config = []
        self.torch_kwargs_config = collections.OrderedDict()
        self.paddle_merged_kwargs_config = collections.OrderedDict()

        def finish(paddle_args_dict):
            self.paddle_merged_kwargs_config = paddle_args_dict
            self.torch_kwargs_config.update(paddle_args_dict)
            self.torch_kwargs_config.pop("name", None)
            return True

        api_name = self.api_config.api_name
        parameter_binding = bind_input_parameters(
            api_name,
            self.api_config.args,
            self.api_config.kwargs,
            api=self.paddle_api,
        )
        if parameter_binding.source == "unresolved":
            return True
        arguments = parameter_binding.arguments
        if api_name.startswith("paddle.Tensor."):
            self.torch_args_config, arguments = split_tensor_method_arguments(api_name, arguments)
        return finish(arguments)

    def generate_input_values(self):
        return input_rules.generate(self)

    def _torch_execution_locals(self):
        if (
            self.api_config.api_name == "paddle.nn.functional.rnnt_loss"
            and paddle.device.get_device() == "cpu"
        ):
            return {"fused_log_softmax": False}
        return None

    def _materialize_paddle_config_value(self, config_value, *, clear_tensor=True):
        """Materialize a Paddle config tree into live values, optionally clearing cache."""
        return materialize_config_tree(
            config_value,
            self.api_config,
            "paddle",
            clear_tensor=clear_tensor,
        )

    def build_paddle_input(self):
        """Generate Paddle inputs from config and materialize TensorConfig leaves.

        Call generate_input_values() first because build_paddle_input() only materializes
        TensorConfig leaves and does not generate logical input values.
        """
        self.paddle_args = []
        self.paddle_kwargs = collections.OrderedDict()

        for arg_config in self.paddle_args_config:
            self.paddle_args.append(
                self._materialize_paddle_config_value(arg_config, clear_tensor=True)
            )

        for key, kwarg_config in self.paddle_kwargs_config.items():
            self.paddle_kwargs[key] = self._materialize_paddle_config_value(
                kwarg_config,
                clear_tensor=True,
            )

        if len(self.paddle_args) == 0 and self.api_config.api_name.startswith("paddle.Tensor."):
            self.paddle_args.append(self.paddle_kwargs.popitem(last=False)[1])

        if (
            self.api_config.api_name == "paddle.linalg.lstsq"
            and "gpu" in paddle.device.get_device()
        ):
            if len(self.paddle_args) > 3:
                self.paddle_args[3] = "gels"
            elif "driver" in self.paddle_kwargs:
                self.paddle_kwargs["driver"] = "gels"

        # In-place ops require a non-leaf tensor as target.  Always copy inputs
        # so the in-place op doesn't hit "Leaf Var can't use inplace strategy".
        # (Previously guarded by need_check_grad(), but forward_only in-place ops
        # still need the copy to avoid Paddle's leaf-variable restriction.)
        if requires_inplace_input_copy(self.api_config):
            self.paddle_args, self.paddle_kwargs = self.copy_paddle_input()

        return True

    def copy_paddle_input(self):
        def _deep_copy(data):
            if isinstance(data, paddle.Tensor):
                return paddle.assign(data)
            elif isinstance(data, (list, tuple)):
                return type(data)(_deep_copy(x) for x in data)
            return data

        args = [_deep_copy(arg) for arg in self.paddle_args]
        kwargs = collections.OrderedDict((k, _deep_copy(v)) for k, v in self.paddle_kwargs.items())
        return args, kwargs

    def _iter_tensor_tree_leaves(self, value, *, tensor_type=None, unique=False):
        if tensor_type is None:
            # Paddle-only 的默认遍历类型不能为了构造 tuple 而解析 Torch lazy proxy。
            tensor_type = (
                (paddle.Tensor, torch.Tensor) if getattr(self, "use_torch", True) else paddle.Tensor
            )
        seen = set()

        def visit(item):
            if isinstance(item, tensor_type):
                if unique and id(item) in seen:
                    return
                if unique:
                    seen.add(id(item))
                yield item
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    yield from visit(child)
                return
            if isinstance(item, dict):
                for child in item.values():
                    yield from visit(child)

        yield from visit(value)

    def _collect_tensor_leaves(self, value, result, tensor_type):
        result.extend(self._iter_tensor_tree_leaves(value, tensor_type=tensor_type))

    def get_paddle_input_list(self):
        result = []

        for arg in self.paddle_args:
            self._collect_tensor_leaves(arg, result, paddle.Tensor)

        # 按 merged_kwargs 顺序遍历，确保 paddle 关键字参数与 torch 参数顺序一致，避免反向比较无法对应
        # torch 参数顺序通过 paddle_sig.bind 绑定，见 ana_torch_api_info()
        merged_kwargs_config = getattr(self, "paddle_merged_kwargs_config", None)
        if merged_kwargs_config is not None:
            for key in merged_kwargs_config:
                if key in self.paddle_kwargs:
                    self._collect_tensor_leaves(self.paddle_kwargs[key], result, paddle.Tensor)
        else:  #  paddle_only
            for key, value in self.paddle_kwargs.items():
                self._collect_tensor_leaves(value, result, paddle.Tensor)

        return [t for t in result if not t.stop_gradient]

    def get_torch_input_list(self):
        result = []
        for i in range(len(self.torch_args)):
            self._collect_tensor_leaves(self.torch_args[i], result, torch.Tensor)

        for _key, value in self.torch_kwargs.items():
            self._collect_tensor_leaves(value, result, torch.Tensor)

        return [t for t in result if t.requires_grad]

    def _make_numpy_output_grad(self, output):
        dtype = _dtype_name(output.dtype)
        from .input_generation.backend_runtime import generate_output_grad_for_runtime

        return generate_output_grad_for_runtime(
            dtype=dtype,
            shape=output.shape,
            device="cpu",
            runtime_config=self.runtime_config,
            config_fingerprint=self.api_config.config,
        )

    @staticmethod
    def _output_grad_intermediate_dtype(dtype):
        if dtype in ["float8_e5m2", "float8_e4m3fn"]:
            return "float16"
        if dtype == "bfloat16":
            return "float32"
        return dtype

    def _numpy_output_grad_to_paddle(self, numpy_tensor, output):
        dtype = _dtype_name(output.dtype)
        intermediate_dtype = self._output_grad_intermediate_dtype(dtype)
        result_output_grad = paddle.to_tensor(
            numpy_tensor,
            dtype=intermediate_dtype,
            place=output.place,
        )
        result_output_grad.stop_gradient = False
        if dtype in ["bfloat16", "float8_e5m2", "float8_e4m3fn"]:
            result_output_grad = paddle.cast(result_output_grad, dtype=dtype)
        return result_output_grad

    def _numpy_output_grad_to_torch(self, numpy_tensor, output):
        dtype = _dtype_name(output.dtype)
        result_output_grad = torch.tensor(
            numpy_tensor,
            dtype=self.to_torch_dtype(dtype) if dtype != "bfloat16" else torch.float32,
            device=output.device,
        )
        if dtype == "bfloat16":
            result_output_grad = result_output_grad.to(dtype=torch.bfloat16)
        return result_output_grad

    def _make_torch_output_grad(self, shape, dtype, device=None):
        device = device or torch.device("cuda", torch.cuda.current_device())
        from .input_generation.backend_runtime import generate_output_grad_for_runtime

        return generate_output_grad_for_runtime(
            dtype=dtype,
            shape=shape,
            device=str(device),
            runtime_config=self.runtime_config,
            config_fingerprint=self.api_config.config,
        )

    def _make_paddle_output_grad(self, shape, dtype, place=None):
        # grad 必须直接生成到输出所在 place，避免跨设备复制改变峰值显存语义。
        if place is not None and place.is_gpu_place():
            device = f"gpu:{place.gpu_device_id()}"
        else:
            device = "cpu"
        from .input_generation.backend_runtime import generate_output_grad_for_runtime

        return generate_output_grad_for_runtime(
            dtype=dtype,
            shape=shape,
            device=device,
            runtime_config=self.runtime_config,
            config_fingerprint=self.api_config.config,
        )

    @staticmethod
    def _output_grad_signature(outputs):
        return tuple((_dtype_name(output.dtype), tuple(output.shape)) for output in outputs)

    @staticmethod
    def _tensor_matches_output_place(tensor, output):
        if isinstance(tensor, paddle.Tensor) and isinstance(output, paddle.Tensor):
            if tensor.place.is_cpu_place() and output.place.is_cpu_place():
                return True
            if tensor.place.is_gpu_place() and output.place.is_gpu_place():
                return int(tensor.place.gpu_device_id()) == int(output.place.gpu_device_id())
            return str(tensor.place) == str(output.place)
        if isinstance(tensor, torch.Tensor) and isinstance(output, torch.Tensor):
            tensor_device = 0 if tensor.device.index is None else int(tensor.device.index)
            output_device = 0 if output.device.index is None else int(output.device.index)
            return tensor.device.type == output.device.type and tensor_device == output_device
        return False

    def _make_output_grad_slot(self, output):
        """由 resolved input backend 创建反向种子，再交给消费框架物化。"""
        dtype = _dtype_name(output.dtype)
        backend_name = getattr(self.runtime_config, "input_backend_resolved", None)
        if backend_name is None:
            raise ValueError("runtime config has no resolved input backend")
        if backend_name == "numpy":
            # NumPy seed 是该 backend 的正式反向输入，随后才按消费框架执行物化。
            return OutputGradSlot(seed_numpy=self._make_numpy_output_grad(output))

        if isinstance(output, paddle.Tensor):
            # 设备只决定原生 seed 的生成位置，不能反向改变 resolved backend。
            device_id = int(output.place.gpu_device_id()) if output.place.is_gpu_place() else None
        else:
            device_id = int(output.device.index or 0) if output.device.type == "cuda" else None

        if backend_name == "torch":
            device = (
                torch.device("cuda", device_id) if device_id is not None else torch.device("cpu")
            )
            # Torch seed 保持为唯一所有者，Paddle 消费时通过 DLPack 延迟共享。
            torch_grad = self._make_torch_output_grad(output.shape, dtype, device=device).detach()
            return OutputGradSlot(torch_grad=torch_grad)

        if backend_name == "paddle":
            place = paddle.CUDAPlace(device_id) if device_id is not None else paddle.CPUPlace()
            paddle_grad = self._make_paddle_output_grad(output.shape, dtype, place=place)
            paddle_grad.stop_gradient = False
            return OutputGradSlot(paddle_grad=paddle_grad)
        raise ValueError(f"unsupported output-grad backend: {backend_name!r}")

    @staticmethod
    def _torch_grad_to_paddle(torch_grad, output):
        if output.place.is_gpu_place():
            device = torch.device("cuda", int(output.place.gpu_device_id()))
        else:
            device = torch.device("cpu")
        torch_grad = torch_grad.detach().to(device=device)
        paddle_grad = paddle.utils.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch_grad))
        if not (output.place.is_cpu_place() or output.place.is_gpu_place()):
            # 自定义设备不支持直接 DLPack 接收，先在 CPU 建立 Paddle 所有权再复制。
            paddle_grad = paddle_grad._copy_to(output.place, False)
        paddle_grad.stop_gradient = False
        return paddle_grad

    @staticmethod
    def _paddle_grad_to_torch(paddle_grad, output):
        if output.device.type == "cuda":
            place = paddle.CUDAPlace(int(output.device.index or 0))
        else:
            place = paddle.CPUPlace()
        paddle_grad = paddle_grad.detach()._copy_to(place, False)
        return torch.utils.dlpack.from_dlpack(paddle.utils.dlpack.to_dlpack(paddle_grad))

    def clear_output_grad_cache(self):
        """Drop all cached output-grad seeds for the current execution."""
        self.output_grad_slots.clear()
        self.output_grad_slots_signature = None

    def _output_grad_slot_to_paddle(self, slot, output):
        if slot.paddle_grad is not None and self._tensor_matches_output_place(
            slot.paddle_grad, output
        ):
            return slot.paddle_grad
        if slot.paddle_grad is not None:
            # Paddle backend 仍是唯一值所有者，place 转换不引入其他随机源。
            paddle_grad = slot.paddle_grad.detach()._copy_to(output.place, False)
            paddle_grad.stop_gradient = False
            return paddle_grad
        if slot.seed_numpy is not None:
            return self._numpy_output_grad_to_paddle(slot.seed_numpy, output)
        if slot.torch_grad is not None:
            return self._torch_grad_to_paddle(slot.torch_grad, output)
        raise RuntimeError("output grad slot is empty")

    def _output_grad_slot_to_torch(self, slot, output):
        if slot.torch_grad is not None and self._tensor_matches_output_place(
            slot.torch_grad, output
        ):
            return slot.torch_grad
        if slot.torch_grad is not None:
            return slot.torch_grad.detach().to(device=output.device)
        if slot.seed_numpy is not None:
            return self._numpy_output_grad_to_torch(slot.seed_numpy, output)
        if slot.paddle_grad is not None:
            return self._paddle_grad_to_torch(slot.paddle_grad, output)
        raise RuntimeError("output grad slot is empty")

    def _prepare_output_grad_slots(self, result_outputs):
        signature = self._output_grad_signature(result_outputs)
        if self.output_grad_slots_signature != signature:
            self.output_grad_slots = []
            self.output_grad_slots_signature = signature
        if len(self.output_grad_slots) != len(result_outputs):
            self.output_grad_slots = []
            for output in result_outputs:
                # 输出所在框架不再决定随机源；它只决定最终梯度必须到达的设备。
                self.output_grad_slots.append(self._make_output_grad_slot(output))

    def gen_paddle_output_and_output_grad(self, outputs):
        """Normalize Paddle outputs and align them with cached output-grad seeds."""
        result_outputs = []
        if isinstance(outputs, paddle.Tensor):
            result_outputs.append(outputs)
        elif isinstance(outputs, list):
            result_outputs = [
                output
                for output in outputs
                if isinstance(output, paddle.Tensor)
                and (output._is_initialized() or output.numel() == 0)
            ]
        elif isinstance(
            outputs,
            (paddle.autograd.autograd.Hessian, paddle.autograd.autograd.Jacobian),
        ):
            result_outputs.append(outputs[:])
        elif isinstance(outputs, tuple):
            for output in outputs:
                if output is None or (
                    isinstance(output, paddle.Tensor)
                    and not (output._is_initialized() or output.numel() == 0)
                ):
                    continue
                elif isinstance(output, paddle.Tensor):
                    result_outputs.append(output)
                elif isinstance(output, list):
                    for item in output:
                        if isinstance(item, paddle.Tensor):
                            result_outputs.append(item)
                elif isinstance(
                    output,
                    (
                        paddle.autograd.autograd.Hessian,
                        paddle.autograd.autograd.Jacobian,
                    ),
                ):
                    result_outputs.extend(output[:])
                elif (
                    isinstance(output, tuple)
                    and len(output) > 0
                    and (
                        isinstance(output[0], paddle.autograd.autograd.Hessian)
                        or isinstance(output[0], paddle.autograd.autograd.Jacobian)
                    )
                ):
                    for lazy_obj in output:
                        result_outputs.append(lazy_obj[:])
                else:
                    raise ValueError("outputs format not support")

        result_outputs = [
            output
            for output in result_outputs
            if not output.stop_gradient and str(output.dtype).split(".")[-1] in AUTOGRAD_DTYPES
        ]

        self._prepare_output_grad_slots(result_outputs)
        result_outputs_grads = []
        for output, slot in zip(result_outputs, self.output_grad_slots, strict=True):
            result_outputs_grads.append(self._output_grad_slot_to_paddle(slot, output))
        return result_outputs, result_outputs_grads

    def gen_torch_output_and_output_grad(self, outputs):
        result_outputs = []
        if isinstance(outputs, torch.Tensor):
            result_outputs.append(outputs)
        elif isinstance(outputs, torch.Size):
            result_outputs.append(torch.tensor(outputs))
        elif isinstance(outputs, list):
            result_outputs = [output for output in outputs if isinstance(output, torch.Tensor)]
        elif isinstance(outputs, tuple):
            for output in outputs:
                if output is None:
                    continue
                elif isinstance(output, torch.Tensor):
                    result_outputs.append(output)
                else:
                    raise ValueError("outputs format not support")

        result_outputs = [output for output in result_outputs if output.requires_grad]

        self._prepare_output_grad_slots(result_outputs)
        result_outputs_grads = []
        for output, slot in zip(result_outputs, self.output_grad_slots, strict=True):
            result_outputs_grads.append(self._output_grad_slot_to_torch(slot, output))
        return result_outputs, result_outputs_grads

    def to_torch_dtype(self, dtype):
        return to_torch_dtype(dtype)

    def copy_torch_input(self):
        def _deep_copy(data):
            if isinstance(data, torch.Tensor):
                return torch.clone(data)
            elif isinstance(data, (list, tuple)):
                return type(data)(_deep_copy(x) for x in data)
            return data

        args = [_deep_copy(arg) for arg in self.torch_args]
        kwargs = collections.OrderedDict((k, _deep_copy(v)) for k, v in self.torch_kwargs.items())
        return args, kwargs

    def _materialize_torch_config_value(self, config_value, *, convert_dtype=False):
        """Materialize a Torch config tree and translate explicit dtype values."""
        return materialize_config_tree(
            config_value,
            self.api_config,
            "torch",
            convert_dtype=convert_dtype,
        )

    def build_torch_input(self):
        """Generate Torch inputs from config and materialize TensorConfig leaves.

        Call generate_input_values() first because build_torch_input() only materializes
        TensorConfig leaves and does not generate logical input values.
        """
        self.torch_args = []
        self.torch_kwargs = collections.OrderedDict()
        for arg_config in self.torch_args_config:
            self.torch_args.append(self._materialize_torch_config_value(arg_config))

        for key, arg_config in self.torch_kwargs_config.items():
            self.torch_kwargs[key] = self._materialize_torch_config_value(
                arg_config,
                convert_dtype=key == "dtype",
            )

        if requires_inplace_input_copy(self.api_config):
            self.torch_args, self.torch_kwargs = self.copy_torch_input()

        if not self.gpu_mode_config.enabled:
            self.release_framework_gpu_cache("torch")
        return True

    def np_assert_accuracy(self, np_paddle, np_torch, atol=1e-2, rtol=1e-2):
        if np_paddle.dtype == numpy.bool_:
            numpy.testing.assert_equal(np_paddle, np_torch)
            return
        bitwise_alignment = getattr(self, "bitwise_alignment", False)
        if not bitwise_alignment and self.api_config.api_name in special_accuracy_atol_rtol:
            atol, rtol = special_accuracy_atol_rtol[self.api_config.api_name]

        numpy.testing.assert_allclose(
            np_paddle,
            np_torch,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )

    def paddle_assert_accuracy(
        self, actual_paddle_tensor, expected_paddle_tensor, atol=1e-2, rtol=1e-2
    ):
        # Paddle/Paddle 与 Paddle/Torch 共用同一比较设备协议，避免 CINN 固定退回 CPU。
        return self.torch_assert_accuracy(
            actual_paddle_tensor,
            expected_paddle_tensor,
            atol=atol,
            rtol=rtol,
        )

    def _torch_assert_accuracy_in_chunks(
        self, actual, expected, atol, rtol, error_msg, working_bytes
    ):
        temp_bytes_per_element = max(
            32, 4 * max(actual.element_size(), expected.element_size()) + 16
        )
        chunk_numel = max(1, working_bytes // temp_bytes_per_element)
        actual_flat = actual.reshape(-1)
        expected_flat = expected.reshape(-1)
        actual_numel = actual_flat.numel()

        def chunks():
            for start in range(0, actual_numel, chunk_numel):
                end = min(actual_numel, start + chunk_numel)
                yield start, actual_flat[start:end], expected_flat[start:end]

        self._torch_assert_accuracy_from_chunks(
            chunks(),
            actual_numel,
            tuple(actual.shape),
            actual.dtype,
            expected.dtype,
            atol,
            rtol,
            error_msg,
        )

    def _torch_assert_accuracy_from_chunks(
        self,
        chunks,
        actual_numel,
        actual_shape,
        actual_dtype,
        expected_dtype,
        atol,
        rtol,
        error_msg,
    ):
        actual_is_float8 = "float8" in str(actual_dtype)
        expected_is_float8 = "float8" in str(expected_dtype)
        mismatch_count = 0
        max_abs_diff = -1.0
        max_abs_index = 0
        max_rel_diff = -1.0
        max_rel_index = 0
        exact_compare = atol == 0.0 and rtol == 0.0
        for start, actual_chunk, expected_chunk in chunks:
            if actual_chunk.dtype != expected_chunk.dtype:
                # Match TensorLikePair._equalize_attributes without promoting
                # either complete tensor. PyTorch does not define mixed FP8
                # promotion, while the existing small-tensor path promotes FP8
                # to float32 before assert_close.
                if "float8" in str(actual_chunk.dtype):
                    actual_chunk = actual_chunk.float()
                if "float8" in str(expected_chunk.dtype):
                    expected_chunk = expected_chunk.float()
                actual_promote_dtype = actual_chunk.dtype
                expected_promote_dtype = expected_chunk.dtype
                unsigned_dtypes = (torch.uint16, torch.uint32, torch.uint64)
                if actual_promote_dtype in unsigned_dtypes:
                    actual_promote_dtype = torch.int64
                if expected_promote_dtype in unsigned_dtypes:
                    expected_promote_dtype = torch.int64
                compare_dtype = torch.promote_types(actual_promote_dtype, expected_promote_dtype)
                actual_chunk = actual_chunk.to(compare_dtype)
                expected_chunk = expected_chunk.to(compare_dtype)

            if exact_compare:
                equal = actual_chunk == expected_chunk
                if actual_chunk.dtype.is_floating_point or actual_chunk.dtype.is_complex:
                    equal |= torch.isnan(actual_chunk) & torch.isnan(expected_chunk)
                mismatch = equal.logical_not_()
            else:
                if actual_is_float8:
                    actual_chunk = actual_chunk.float()
                if expected_is_float8:
                    expected_chunk = expected_chunk.float()
                mismatch = torch.isclose(
                    actual_chunk,
                    expected_chunk,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                ).logical_not_()

            chunk_mismatch_count = int(mismatch.sum().item())
            mismatch_count += chunk_mismatch_count
            if chunk_mismatch_count == 0:
                continue

            chunk_actual_is_complex = actual_chunk.dtype.is_complex
            chunk_expected_is_complex = expected_chunk.dtype.is_complex
            if chunk_actual_is_complex or chunk_expected_is_complex:
                actual_for_diff = actual_chunk
                expected_for_diff = expected_chunk
            elif actual_chunk.dtype == torch.float64 or expected_chunk.dtype == torch.float64:
                actual_for_diff = actual_chunk.to(torch.float64)
                expected_for_diff = expected_chunk.to(torch.float64)
            elif "float8" in str(actual_chunk.dtype) or "float8" in str(expected_chunk.dtype):
                actual_for_diff = actual_chunk.float()
                expected_for_diff = expected_chunk.float()
            elif (
                not actual_chunk.dtype.is_floating_point
                and not chunk_actual_is_complex
                and not expected_chunk.dtype.is_floating_point
                and not chunk_expected_is_complex
            ):
                actual_for_diff = actual_chunk.to(torch.int64)
                expected_for_diff = expected_chunk.to(torch.int64)
            else:
                actual_for_diff = actual_chunk
                expected_for_diff = expected_chunk
            abs_diff = torch.abs(actual_for_diff - expected_for_diff)
            matched = ~mismatch
            abs_diff.masked_fill_(matched, -1.0)
            chunk_max_abs, chunk_abs_index = torch.max(abs_diff, dim=0)
            rel_diff = abs_diff / torch.abs(expected_for_diff)
            rel_diff.masked_fill_(matched, -1.0)
            chunk_max_rel, chunk_rel_index = torch.max(rel_diff, dim=0)

            # Transfer all reduction results in one synchronization.  Calling
            # .item() for every value dominates scans with hundreds of chunks.
            (
                chunk_max_abs_value,
                chunk_abs_index_value,
                chunk_max_rel_value,
                chunk_rel_index_value,
            ) = (
                torch.stack(
                    (
                        chunk_max_abs.to(torch.float64),
                        chunk_abs_index.to(torch.float64),
                        chunk_max_rel.to(torch.float64),
                        chunk_rel_index.to(torch.float64),
                    )
                )
                .cpu()
                .tolist()
            )
            if chunk_max_abs_value > max_abs_diff:
                max_abs_diff = chunk_max_abs_value
                max_abs_index = start + int(chunk_abs_index_value)
            if chunk_max_rel_value > max_rel_diff:
                max_rel_diff = chunk_max_rel_value
                max_rel_index = start + int(chunk_rel_index_value)

        if mismatch_count == 0:
            return

        abs_index = tuple(int(value) for value in numpy.unravel_index(max_abs_index, actual_shape))
        rel_index = tuple(int(value) for value in numpy.unravel_index(max_rel_index, actual_shape))
        mismatch_percent = 100.0 * mismatch_count / actual_numel
        raise AssertionError(
            error_msg(
                "Tensor-likes are not equal!\n\n"
                f"Mismatched elements: {mismatch_count} / {actual_numel} "
                f"({mismatch_percent:.1f}%)\n"
                f"Greatest absolute difference: {max_abs_diff} at index {abs_index}\n"
                f"Greatest relative difference: {max_rel_diff} at index {rel_index}"
            )
        )

    @staticmethod
    def _logical_slab_indices(shape, max_numel):
        shape = tuple(int(dim) for dim in shape)
        if not shape:
            yield ()
            return
        total_numel = 1
        for dim in shape:
            total_numel *= dim
        if total_numel == 0:
            return

        max_numel = max(1, int(max_numel))
        for axis in range(len(shape)):
            suffix_numel = 1
            for dim in shape[axis + 1 :]:
                suffix_numel *= dim
            if suffix_numel <= max_numel:
                break
        block_size = max(1, max_numel // suffix_numel)
        suffix = (slice(None),) * (len(shape) - axis - 1)
        for prefix in numpy.ndindex(shape[:axis]):
            for start in range(0, shape[axis], block_size):
                end = min(start + block_size, shape[axis])
                yield (*prefix, slice(start, end), *suffix)

    @staticmethod
    def _framework_tensor_torch_dtype(value):
        if isinstance(value, torch.Tensor):
            return value.dtype
        dtype_name = str(value.dtype).split(".")[-1]
        return getattr(torch, dtype_name)

    @staticmethod
    def _logical_slab_to_torch(value, index, comparison_device):
        slab = value[index].detach()
        if not slab.is_contiguous():
            slab = slab.contiguous()
        if comparison_device.type == "cpu":
            slab = slab.cpu()
        if isinstance(slab, paddle.Tensor):
            slab = torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(slab)  # type: ignore[reportGeneralTypeIssues]
            )
        if slab.device != comparison_device:
            slab = slab.to(device=comparison_device)
        return slab

    def _torch_assert_accuracy_in_logical_slabs(
        self,
        actual,
        expected,
        atol,
        rtol,
        error_msg,
        working_bytes,
        comparison_device,
        check_dtype,
    ):
        actual_dtype = self._framework_tensor_torch_dtype(actual)
        expected_dtype = self._framework_tensor_torch_dtype(expected)
        temp_bytes_per_element = max(
            32,
            4 * max(_tensor_element_size(actual), _tensor_element_size(expected)) + 16,
        )
        max_numel = max(1, working_bytes // temp_bytes_per_element)
        shape = tuple(actual.shape)
        actual_numel = int(actual.numel())

        if not check_dtype and actual_dtype != expected_dtype:
            for index in self._logical_slab_indices(shape, max_numel):
                actual_chunk = self._logical_slab_to_torch(actual, index, comparison_device)
                expected_chunk = self._logical_slab_to_torch(expected, index, comparison_device)

                def slab_error_msg(msg, *, slab_index=index):
                    return error_msg(f"logical slab {slab_index}: {msg}")

                torch.testing.assert_close(
                    actual_chunk,
                    expected_chunk,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    check_device=False,
                    check_dtype=False,
                    msg=slab_error_msg,
                )
            return

        def chunks():
            offset = 0
            for index in self._logical_slab_indices(shape, max_numel):
                actual_chunk = self._logical_slab_to_torch(
                    actual, index, comparison_device
                ).reshape(-1)
                expected_chunk = self._logical_slab_to_torch(
                    expected, index, comparison_device
                ).reshape(-1)
                yield offset, actual_chunk, expected_chunk
                offset += actual_chunk.numel()

        self._torch_assert_accuracy_from_chunks(
            chunks(),
            actual_numel,
            shape,
            actual_dtype,
            expected_dtype,
            atol,
            rtol,
            error_msg,
        )

    @staticmethod
    def _comparison_cuda_device_id(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cuda":
                return None
            return 0 if value.device.index is None else int(value.device.index)
        if isinstance(value, paddle.Tensor) and value.place.is_gpu_place():
            return int(value.place.gpu_device_id())
        return None

    @staticmethod
    def _comparison_temporary_bytes(actual, expected):
        return int(actual.numel()) * max(
            32,
            4 * max(_tensor_element_size(actual), _tensor_element_size(expected)) + 16,
        )

    def _resolve_comparison_workspace(
        self,
        actual,
        estimated_temp_bytes,
        dual_gpu,
        *,
        comparison_device_id=None,
    ):
        """按实际比较设备解析并缓存 bounded compare 工作区。"""
        estimated_temp_bytes = max(0, int(estimated_temp_bytes))
        device_id = comparison_device_id
        if device_id is None:
            device_id = self._comparison_cuda_device_id(actual)
        if device_id is None:
            return DEFAULT_COMPARISON_WORKSPACE_BYTES
        if estimated_temp_bytes <= COMPARISON_WORKSPACE_FAST_PATH_BYTES:
            # 独占设备上的小比较不做驱动探测，避免大量小叶子产生同步查询。
            return DEFAULT_COMPARISON_WORKSPACE_BYTES

        cache = getattr(self, "comparison_workspace_cache", None)
        if cache is None:
            cache = self.comparison_workspace_cache = {}
        cache_key = (device_id, bool(dual_gpu))
        # estimated/numel 与比较内核的 per-element 估算同源，避免另设固定最小块。
        min_working_bytes = max(
            1,
            estimated_temp_bytes // max(1, int(actual.numel())),
        )
        if cache_key in cache:
            # dtype 不同会改变单元素临时量，因此复用设备快照时仍保留当前最小推进量。
            return max(min_working_bytes, cache[cache_key])

        try:
            # 同一 case 首次遇到大比较时读取一次物理快照，后续叶子只复用结果。
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
            memory_budget = float(getattr(self.gpu_mode_config, "memory_budget", 0.0) or 0.0)
            comparison_device_id = getattr(self.gpu_mode_config, "comparison_device_id", None)
            if device_id == comparison_device_id:
                memory_budget = float(
                    getattr(
                        self.gpu_mode_config,
                        "comparison_memory_budget",
                        memory_budget,
                    )
                    or memory_budget
                )

            if dual_gpu:
                # comparison 卡独占一个 worker，但仍要保留运行时和框架 allocator reserve。
                budget_bytes = int(memory_budget * _GIB) if memory_budget > 0 else 0
                reserve_bytes = self._gpu_safety_reserve_bytes(total_bytes, budget_bytes)
                working_bytes = self._comparison_workspace_bytes(
                    free_bytes,
                    reserve_bytes,
                    dual_gpu=True,
                )
            else:
                # memory_reserved 代表本 worker 的 Torch allocator footprint，用于约束新增块。
                reserved_bytes = torch.cuda.memory_reserved(torch.device("cuda", device_id))
                working_bytes = self._comparison_workspace_bytes(free_bytes)
                if memory_budget > 0:
                    # 单卡只允许预算余量的四分之一用于比较；零余量仅保留单元素推进量。
                    budget_headroom = max(0, int(memory_budget * _GIB) - reserved_bytes)
                    working_bytes = min(
                        working_bytes,
                        max(min_working_bytes, budget_headroom // 4),
                    )
        except Exception:
            # 查询失败不能放大到 1 GiB；保守分块仍允许 best-effort 继续执行。
            # 失败结果也写入 case cache，避免后续每个叶子重复触发同一个驱动异常。
            # 该降级路径不再依赖额外的驱动调用，优先保证比较可以继续推进。
            working_bytes = max(
                min_working_bytes,
                min(COMPARISON_WORKSPACE_FAST_PATH_BYTES, estimated_temp_bytes),
            )
            cache[cache_key] = working_bytes
            return working_bytes

        if dual_gpu and working_bytes < min_working_bytes:
            # 双卡结果已经搬运到 comparison 卡，无法退回 CPU 或切换设备继续比较。
            raise GpuMemoryGuardSkip(
                "comparison GPU capacity guard: no reserved headroom for bounded compare"
            )
        working_bytes = max(min_working_bytes, working_bytes)
        cache[cache_key] = working_bytes
        return working_bytes

    def _comparison_device(self, actual, expected, *, record_accuracy_tolerance, dual_gpu):
        """解析比较设备；该策略独立于两侧算子的执行设备。"""
        # record_accuracy_tolerance 只改变容差与日志，不得绕过 GPU mode 的比较设备协议。
        # 非 GPU mode 使用 CPU compare，保持低显存路径。
        # 单卡 GPU mode 允许 CPU kernel 结果显式搬到 GPU 0 后比较。
        # 双卡模式必须从结果设备解析比较卡，不允许 CPU tensor 隐式降级。
        if not self.gpu_mode_config.enabled:
            return torch.device("cpu")
        if dual_gpu:
            device_id = self._comparison_cuda_device_id(actual)
            if device_id is None:
                device_id = self._comparison_cuda_device_id(expected)
            if device_id is None:
                raise RuntimeError("dual GPU comparisons require CUDA tensors on both sides")
            return torch.device("cuda", device_id)
        device_id = self.gpu_mode_config.comparison_device_id
        return torch.device("cuda", 0 if device_id is None else int(device_id))

    def torch_assert_accuracy(
        self,
        actual,
        expected,
        atol=1e-2,
        rtol=1e-2,
        check_dtype=None,
        actual_name="ACTUAL",
        expected_name="DESIRED",
        apply_special_tolerance=True,
        tensor_index=0,
        tensor_count=1,
    ):
        is_check_dtype = self.should_check_dtype() if check_dtype is None else check_dtype
        bitwise_alignment = getattr(self, "bitwise_alignment", False)

        if (
            apply_special_tolerance
            and not bitwise_alignment
            and self.api_config.api_name in special_accuracy_atol_rtol
        ):
            atol, rtol = special_accuracy_atol_rtol[self.api_config.api_name]
        record_accuracy_tolerance = getattr(self, "record_accuracy_tolerance", False)
        is_backward = getattr(self, "is_backward", False)
        if record_accuracy_tolerance:
            atol, rtol = 0.0, 0.0

        def is_cpu_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.device.type == "cpu"
            if isinstance(value, paddle.Tensor):
                return value.place.is_cpu_place()
            return False

        dual_gpu = bool(getattr(self.gpu_mode_config, "dual_gpu", False))
        tensor_types = (paddle.Tensor, torch.Tensor)
        if not isinstance(actual, tensor_types):
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(actual)}")
        if not isinstance(expected, tensor_types):
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(expected)}")
        if dual_gpu and (is_cpu_tensor(actual) or is_cpu_tensor(expected)):
            raise RuntimeError("dual GPU comparisons require CUDA tensors on both sides")
        comparison_device = self._comparison_device(
            actual,
            expected,
            record_accuracy_tolerance=record_accuracy_tolerance,
            dual_gpu=dual_gpu,
        )
        comparison_device_id = comparison_device.index if comparison_device.type == "cuda" else None

        # DLPack 只负责跨框架表示转换，转换后的 Tensor 还要统一搬到比较设备。
        # 因此 CPU kernel 与 GPU compare 的组合不会因源 Tensor 在 CPU 而退回 CPU compare。

        if not actual.is_contiguous() or not expected.is_contiguous():
            actual_shape = tuple(actual.shape)
            expected_shape = tuple(expected.shape)
            actual_dtype = self._framework_tensor_torch_dtype(actual)
            expected_dtype = self._framework_tensor_torch_dtype(expected)
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"shape mismatch: {actual_name} {actual_shape}, "
                    f"{expected_name} {expected_shape}"
                )
            if is_check_dtype and actual_dtype != expected_dtype:
                raise AssertionError(
                    f"dtype mismatch: {actual_name} {actual_dtype}, "
                    f"{expected_name} {expected_dtype}"
                )

            def slab_error_msg(msg):
                return (
                    f"Not equal to tolerance rtol={rtol}, atol={atol}\n"
                    f"{msg}\n"
                    f"{actual_name}: (shape={actual_shape}, dtype={actual_dtype})\n"
                    f"{actual}\n"
                    f"{expected_name}: (shape={expected_shape}, dtype={expected_dtype})\n"
                    f"{expected}"
                )

            try:
                estimated_temp_bytes = self._comparison_temporary_bytes(actual, expected)
                working_bytes = self._resolve_comparison_workspace(
                    actual,
                    estimated_temp_bytes,
                    dual_gpu,
                    comparison_device_id=comparison_device_id,
                )
                self._torch_assert_accuracy_in_logical_slabs(
                    actual,
                    expected,
                    atol,
                    rtol,
                    slab_error_msg,
                    working_bytes,
                    comparison_device,
                    is_check_dtype,
                )
                if record_accuracy_tolerance:
                    log_accuracy_tolerance(
                        "Identical",
                        self.api_config.api_name,
                        self.api_config.config[:MAX_CSV_CONFIG_LENGTH],
                        str(actual.dtype),
                        is_backward,
                        tensor_index=tensor_index,
                        tensor_count=tensor_count,
                    )
                return
            except Exception as err:
                error_str = str(err)
                if record_accuracy_tolerance:
                    error_info = error_str.split("\n", maxsplit=2)[1] if "\n" in error_str else None
                    if error_info and (
                        error_info.startswith("Tensor-likes") or error_info.startswith("Scalars")
                    ):
                        log_accuracy_tolerance(
                            error_str,
                            self.api_config.api_name,
                            self.api_config.config[:MAX_CSV_CONFIG_LENGTH],
                            str(actual.dtype),
                            is_backward,
                            tensor_index=tensor_index,
                            tensor_count=tensor_count,
                        )
                        return
                raise

        if isinstance(actual, paddle.Tensor):
            if not actual.is_contiguous():
                actual = actual.contiguous()
            actual = actual.detach()
            if comparison_device.type == "cpu":
                actual = actual.cpu()
            actual_tensor = torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(actual)  # type: ignore[reportGeneralTypeIssues]
            )
        elif isinstance(actual, torch.Tensor):
            if not actual.is_contiguous():
                actual = actual.contiguous()
            actual_tensor = actual.detach()
            if comparison_device.type == "cpu":
                actual_tensor = actual_tensor.cpu()
        else:
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(actual)}")

        if isinstance(expected, paddle.Tensor):
            if not expected.is_contiguous():
                expected = expected.contiguous()
            expected = expected.detach()
            if comparison_device.type == "cpu":
                expected = expected.cpu()
            expected_tensor = torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(expected)  # type: ignore[reportGeneralTypeIssues]
            )
        elif isinstance(expected, torch.Tensor):
            if not expected.is_contiguous():
                expected = expected.contiguous()
            expected_tensor = expected.detach()
            if comparison_device.type == "cpu":
                expected_tensor = expected_tensor.cpu()
        else:
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(expected)}")

        if actual_tensor.device != comparison_device:
            actual_tensor = actual_tensor.to(device=comparison_device)
        if expected_tensor.device != comparison_device:
            expected_tensor = expected_tensor.to(device=comparison_device)

        if actual_tensor.shape != expected_tensor.shape:
            raise AssertionError(
                f"shape mismatch: {actual_name} {actual_tensor.shape}, "
                f"{expected_name} {expected_tensor.shape}"
            )
        if is_check_dtype and actual_tensor.dtype != expected_tensor.dtype:
            raise AssertionError(
                f"dtype mismatch: {actual_name} {actual_tensor.dtype}, "
                f"{expected_name} {expected_tensor.dtype}"
            )

        def error_msg(msg):
            return (
                f"Not equal to tolerance rtol={rtol}, atol={atol}\n"
                f"{msg}\n"
                f"{actual_name}: (shape={actual_tensor.shape}, dtype={actual_tensor.dtype})\n"
                f"{actual_tensor}\n"
                f"{expected_name}: (shape={expected_tensor.shape}, dtype={expected_tensor.dtype})\n"
                f"{expected_tensor}"
            )

        try:
            estimated_temp_bytes = self._comparison_temporary_bytes(
                actual_tensor,
                expected_tensor,
            )
            working_bytes = self._resolve_comparison_workspace(
                actual_tensor,
                estimated_temp_bytes,
                dual_gpu,
                comparison_device_id=comparison_device_id,
            )
            if self._should_chunk_accuracy_compare(estimated_temp_bytes, working_bytes):
                self.record_memory_governance_metric("chunk_compare")
                self._torch_assert_accuracy_in_chunks(
                    actual_tensor,
                    expected_tensor,
                    atol,
                    rtol,
                    error_msg,
                    working_bytes,
                )
                if record_accuracy_tolerance:
                    log_accuracy_tolerance(
                        "Identical",
                        self.api_config.api_name,
                        self.api_config.config[:MAX_CSV_CONFIG_LENGTH],
                        str(actual.dtype),
                        is_backward,
                        tensor_index=tensor_index,
                        tensor_count=tensor_count,
                    )
                return

            # Keep FP8 diagnostics consistent with the chunked path by comparing
            # promoted float32 values even for exact comparisons.
            if "float8" in str(actual_tensor.dtype):
                actual_tensor = actual_tensor.float()
            if "float8" in str(expected_tensor.dtype):
                expected_tensor = expected_tensor.float()
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                check_device=False,
                check_dtype=is_check_dtype,
                msg=error_msg,
            )
            if record_accuracy_tolerance:
                log_accuracy_tolerance(
                    "Identical",
                    self.api_config.api_name,
                    self.api_config.config[:MAX_CSV_CONFIG_LENGTH],
                    str(actual.dtype),
                    is_backward,
                    tensor_index=tensor_index,
                    tensor_count=tensor_count,
                )
        except Exception as err:
            error_str = str(err)
            if error_str.startswith("Comparing"):
                if dual_gpu or os.environ.get("PADDLEAPITEST_NP_FALLBACK", "0") != "1":
                    raise RuntimeError(
                        "[torch_assert_OOM] torch.testing.assert_close OOM on large tensor comparison"
                    ) from err
                print(
                    "[torch_assert_OOM] torch.testing.assert_close OOM, fallback to np_assert",
                    flush=True,
                )
                actual_cpu = actual_tensor.cpu()
                expected_cpu = expected_tensor.cpu()
                if actual_cpu.dtype == torch.bfloat16 or "float8" in str(actual_cpu.dtype):
                    actual_cpu = actual_cpu.float()
                if expected_cpu.dtype == torch.bfloat16 or "float8" in str(expected_cpu.dtype):
                    expected_cpu = expected_cpu.float()
                self.np_assert_accuracy(
                    actual_cpu.numpy(), expected_cpu.numpy(), atol=atol, rtol=rtol
                )
                return
            if record_accuracy_tolerance:
                error_info = error_str.split("\n", maxsplit=2)[1] if "\n" in error_str else None
                if error_info and (
                    error_info.startswith("Tensor-likes") or error_info.startswith("Scalars")
                ):
                    log_accuracy_tolerance(
                        error_str,
                        self.api_config.api_name,
                        self.api_config.config[:MAX_CSV_CONFIG_LENGTH],
                        str(actual.dtype),
                        is_backward,
                        tensor_index=tensor_index,
                        tensor_count=tensor_count,
                    )
                    return
            raise

    def _should_chunk_accuracy_compare(self, estimated_temp_bytes, working_bytes):
        """Use bounded GPU workspaces when a full comparison would exceed the budget."""
        # 只有预估临时量超过当前工作区时才切 slab，避免小比较引入额外 kernel 调度。
        return bool(estimated_temp_bytes > working_bytes)

    @staticmethod
    def _comparison_workspace_bytes(free_bytes, reserve_bytes=0, dual_gpu=False):
        """Pick a conservative comparison workspace from the remaining headroom."""
        if dual_gpu:
            return APITestBase._dual_comparison_workspace_bytes(free_bytes, reserve_bytes)
        max_workspace = 32 * 1024**3
        return max(1, min(max_workspace, int(free_bytes) // 4))

    @staticmethod
    def _dual_comparison_workspace_bytes(free_bytes, reserve_bytes):
        """Choose a bounded comparison workspace from physical card headroom."""
        max_workspace = 64 * 1024**3
        min_workspace = 256 * 1024**2
        available_bytes = max(0, int(free_bytes) - int(reserve_bytes))
        if available_bytes <= 0:
            return 0
        target_workspace = (available_bytes * 2) // 3
        if target_workspace < min_workspace:
            return target_workspace
        return min(max_workspace, max(min_workspace, target_workspace))

    @staticmethod
    def _gpu_safety_reserve_bytes(total_bytes, budget_bytes):
        """仅保留固定运行时余量，不再按大卡容量百分比扣减。"""
        minimum_reserve = 256 * 1024**2
        total_bytes = max(0, int(total_bytes))
        budget_bytes = max(0, int(budget_bytes))
        if budget_bytes > 0 and budget_bytes < total_bytes:
            return max(minimum_reserve, total_bytes - budget_bytes)
        return minimum_reserve

    def test(self):
        pass

    def clear_tensor(self):
        if not self._clear_tensor_config_cache("clear_tensor", self.api_config):
            return
        if self.gpu_mode_config.enabled:
            gpu_mode_memory_decision(self.gpu_mode_config, use_torch=self.use_torch)
        else:
            self.release_framework_gpu_cache()

    def _tensor_config_roots(self):
        return (
            getattr(self, "paddle_args_config", ()),
            getattr(self, "paddle_kwargs_config", {}),
            getattr(self, "paddle_merged_kwargs_config", {}),
            getattr(self, "torch_args_config", ()),
            getattr(self, "torch_kwargs_config", {}),
        )

    def _clear_tensor_config_cache(self, clear_method_name, *args):
        # 共享物化模块按对象身份去重，避免 merged/torch/paddle 别名重复释放。
        return clear_tensor_configs(
            *self._tensor_config_roots(),
            clear_method=clear_method_name,
            clear_args=args,
        )

    def _map_tensor_tree(self, value, tensor_mapper):
        if isinstance(value, paddle.Tensor):
            return tensor_mapper(value)
        if self.use_torch and isinstance(value, torch.Tensor):
            return tensor_mapper(value)
        if isinstance(value, list):
            return [self._map_tensor_tree(item, tensor_mapper) for item in value]
        if isinstance(value, tuple):
            return tuple(self._map_tensor_tree(item, tensor_mapper) for item in value)
        if isinstance(value, dict):
            return type(value)(
                (key, self._map_tensor_tree(item, tensor_mapper)) for key, item in value.items()
            )
        return value

    def move_tensor_tree_to_cpu(self, value):
        """Recursively move framework tensors to CPU without changing containers."""
        return self._map_tensor_tree(value, lambda tensor: tensor.cpu())

    def move_tensor_tree_to_gpu(self, value, device_id):
        """Recursively move framework tensors to one logical GPU."""

        def move_tensor(tensor):
            if isinstance(tensor, torch.Tensor):
                return tensor.to(
                    device=torch.device("cuda", device_id),
                    non_blocking=False,
                )
            return tensor.cuda(device_id=device_id, blocking=True)

        return self._map_tensor_tree(value, move_tensor)

    def iter_unique_tensor_tree_leaves(self, value):
        """Yield unique framework tensor leaves from a result tree."""
        yield from self._iter_tensor_tree_leaves(value, unique=True)

    @staticmethod
    def tensor_is_gpu(value):
        if isinstance(value, paddle.Tensor):
            return value.place.is_gpu_place()
        if isinstance(value, torch.Tensor):
            return value.device.type != "cpu"
        return False

    @staticmethod
    def tensor_gpu_device_id(value):
        if isinstance(value, paddle.Tensor):
            return int(value.place.gpu_device_id())
        if isinstance(value, torch.Tensor):
            return 0 if value.device.index is None else int(value.device.index)
        raise TypeError(f"Expected Paddle or Torch tensor, but got {type(value)}")

    def tensor_tree_nbytes(self, value):
        """Estimate logical bytes held by unique tensor leaves in a result tree."""
        return sum(
            int(tensor.numel()) * _tensor_element_size(tensor)
            for tensor in self.iter_unique_tensor_tree_leaves(value)
        )

    def _unique_gpu_storage_bytes(self, *trees):
        """返回多棵运行时 Tensor 树当前实际占用的唯一 GPU storage 下界。"""
        storage_bytes = {}
        for tree in trees:
            for tensor in self.iter_unique_tensor_tree_leaves(tree):
                if not self.tensor_is_gpu(tensor):
                    continue
                framework = "paddle" if isinstance(tensor, paddle.Tensor) else "torch"
                try:
                    if isinstance(tensor, paddle.Tensor):
                        # Paddle offset 以字节表示；holder 基址在非零 offset view 间保持一致。
                        pointer = int(tensor.data_ptr()) - int(tensor._offset())
                        allocated_bytes = int(tensor._holder_size())
                    else:
                        storage = tensor.untyped_storage()
                        pointer = int(storage.data_ptr())
                        allocated_bytes = int(storage.nbytes())
                    if allocated_bytes < 0:
                        raise ValueError("negative tensor storage size")
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    # 非标准 Tensor 不一定暴露 storage；降级时不推断未知 view 关系。
                    try:
                        pointer = int(tensor.data_ptr())
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pointer = id(tensor)
                    allocated_bytes = int(tensor.numel()) * _tensor_element_size(tensor)
                key = (framework, self.tensor_gpu_device_id(tensor), pointer)
                storage_bytes[key] = max(storage_bytes.get(key, 0), allocated_bytes)
        return sum(storage_bytes.values())

    def enforce_paddle_backward_capacity(self, inputs, outputs, output_grads):
        """在真实输出 shape 已知后拒绝物理容量必然无法完成的 Paddle backward。"""
        if not self.gpu_mode_config.enabled or not inputs or not outputs or not output_grads:
            return
        memory = _query_gpu_memory(None)
        if memory is None:
            return
        _, total_bytes = memory
        capacity_bytes = max(
            0,
            int(total_bytes) - self._gpu_safety_reserve_bytes(total_bytes, 0),
        )
        live_bytes = self._unique_gpu_storage_bytes(inputs, outputs, output_grads)
        output_grad_bytes = self._unique_gpu_storage_bytes(output_grads)
        input_sizes = [
            # 新输入梯度只覆盖当前 view 的逻辑元素，不会复制完整 backing storage。
            self.tensor_tree_nbytes(input_tensor)
            for input_tensor in inputs
            if self.tensor_is_gpu(input_tensor)
        ]
        if not input_sizes:
            return
        # GradTensorHolder 会复制 seed，且有效 backward 至少形成一个输入梯度 storage。
        required_bytes = live_bytes + output_grad_bytes + min(input_sizes)
        if required_bytes <= capacity_bytes:
            return
        raise GpuMemoryGuardSkip(
            "Paddle backward protocol exceeds physical GPU capacity: "
            f"required={required_bytes / _GIB:.2f} GiB, "
            f"capacity={capacity_bytes / _GIB:.2f} GiB, "
            "basis=actual_input_output_grad_storage"
        )

    @staticmethod
    def is_missing_compare_value(value):
        """Return whether a compare leaf means "no value" on both accuracy paths."""
        return value is None or (
            isinstance(value, paddle.Tensor)
            and not value._is_initialized()
            and int(value.numel()) != 0
        )

    def tensor_tree_leaf_count(self, actual, expected):
        """Count comparable leaves with the same structure rules used by compare_tensor_tree."""
        if (
            isinstance(actual, dict)
            and isinstance(expected, dict)
            and actual.keys() == expected.keys()
        ):
            return sum(self.tensor_tree_leaf_count(actual[key], expected[key]) for key in actual)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return max(len(actual), len(expected), 1)
            return sum(
                self.tensor_tree_leaf_count(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=False)
            )
        return 1

    def compare_tensor_tree(
        self,
        actual,
        expected,
        compare_leaf,
        report_structure_error,
        *,
        tensor_index=0,
        tensor_count=None,
    ):
        """遍历嵌套输出结构；调用方只负责叶子比较和错误落盘。"""
        if tensor_count is None:
            tensor_count = max(1, self.tensor_tree_leaf_count(actual, expected))

        def report(reason, index, **details):
            report_structure_error(
                reason,
                tensor_position=f"{index + 1}/{tensor_count}",
                tensor_index=index,
                tensor_count=tensor_count,
                **details,
            )

        def visit(left, right, index):
            if isinstance(left, dict):
                if not isinstance(right, dict):
                    report(
                        "type_mismatch",
                        index,
                        actual_type=type(left).__name__,
                        expected_type=type(right).__name__,
                    )
                    return 1, False
                if left.keys() != right.keys():
                    report(
                        "key_mismatch",
                        index,
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
                    report(
                        "type_mismatch",
                        index,
                        actual_type=type(left).__name__,
                        expected_type=type(right).__name__,
                    )
                    return 1, False
                if len(left) != len(right):
                    report(
                        "count_mismatch",
                        index,
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
                report(
                    "type_mismatch",
                    index,
                    actual_type=type(left).__name__,
                    expected_type=type(right).__name__,
                )
                return 1, False
            return 1, compare_leaf(left, right, index, tensor_count)

        _, ok = visit(actual, expected, tensor_index)
        return ok

    def _reference_workspace_bytes(self, convert_result):
        if not self.gpu_mode_config.enabled:
            return 0
        if not convert_result.code.workspace_required:
            return 0

        from .paddle_to_torch import adaptive_workspace_bytes

        return adaptive_workspace_bytes(torch)

    def tensor_tree_has_gpu_tensor(self, value):
        """Return whether a result tree contains at least one GPU tensor leaf."""
        return any(
            self.tensor_is_gpu(tensor) for tensor in self.iter_unique_tensor_tree_leaves(value)
        )

    def tensor_tree_gpu_device_ids(self, value):
        """Collect unique CUDA device ids used by a result tree."""
        device_ids = []
        seen_device_ids = set()

        def add_device_id(device_id):
            device_id = int(device_id)
            if device_id not in seen_device_ids:
                seen_device_ids.add(device_id)
                device_ids.append(device_id)

        for tensor in self.iter_unique_tensor_tree_leaves(value):
            if self.tensor_is_gpu(tensor):
                add_device_id(self.tensor_gpu_device_id(tensor))
        return device_ids

    def spill_tensor_tree_slot_to_cpu(self, values, index=0, release_cache=False):
        """Replace one result-tree slot with a CPU copy, then optionally release its GPU source."""
        source = values[index]
        if not self.tensor_tree_has_gpu_tensor(source):
            return False
        device_ids = self.tensor_tree_gpu_device_ids(source)
        values[index] = self.move_tensor_tree_to_cpu(source)
        del source
        if release_cache and self.gpu_mode_config.enabled:
            for device_id in device_ids:
                try:
                    self.release_framework_gpu_cache(
                        device_id=device_id,
                        collect_cycles=True,
                    )
                except Exception:
                    pass
        self.record_memory_governance_metric("spill_result_tree")
        return True

    def detach_tensor_tree(self, value):
        """Detach every framework tensor while preserving the argument tree."""
        return self._map_tensor_tree(value, lambda tensor: tensor.detach())

    def save_original_inputs_to_cpu(self):
        """Save config inputs on CPU before either framework can mutate them."""
        # 快照和物化共用 identity 去重，保证共享 TensorConfig 只保存一次。
        clear_tensor_configs(
            *self._tensor_config_roots(),
            clear_method="save_cpu_copy",
            clear_args=(self.api_config,),
        )

    def clear_original_cpu_inputs(self):
        # CPU 快照释放也走同一 owning module，避免生命周期分散回调用方。
        clear_tensor_configs(
            *self._tensor_config_roots(),
            clear_method="clear_cpu_copy",
        )

    def clear_generated_input_values(self):
        """框架输入取得所有权后释放 GPU 生成源，避免跨阶段长期驻留。"""
        # 生成值清理必须传 api_config，values.py 才能同步删除路径索引。
        clear_tensor_configs(
            *self._tensor_config_roots(),
            clear_method="clear_generated_input_value",
            clear_args=(self.api_config,),
        )

    def estimate_input_bytes(self):
        """Estimate unique configured input storage bytes for memory probe gating."""
        return tensor_config_tree_nbytes(*self._tensor_config_roots(), storage=True)

    def clear_paddle_tensor(self):
        if not self._clear_tensor_config_cache("clear_paddle_tensor"):
            return
        if self.gpu_mode_config.enabled:
            gpu_mode_memory_decision(self.gpu_mode_config, use_torch=self.use_torch)
        else:
            self.release_framework_gpu_cache("paddle")

    def clear_torch_tensor(
        self,
        probe_bytes=None,
        *,
        force=False,
        required_headroom_bytes=None,
    ):
        # force/headroom 与配置缓存释放合并为一次 allocator 决策，避免同阶段重复同步。
        cache_cleared = self._clear_tensor_config_cache("clear_torch_tensor")
        if not cache_cleared and not force and required_headroom_bytes is None:
            return
        if self.gpu_mode_config.enabled:
            return gpu_mode_memory_decision(
                self.gpu_mode_config,
                force=force,
                probe_bytes=probe_bytes,
                required_headroom_bytes=required_headroom_bytes,
            )
        if cache_cleared:
            self.release_framework_gpu_cache("torch")

    def is_forward_only(self):
        api = self.api_config.api_name[self.api_config.api_name.rindex(".") + 1 :]
        return api in forward_only_apis

    def should_ignore_paddle_error(self, error_msg):
        dismiss_errors = paddle_error_dismiss.get(self.api_config.api_name, None)
        if dismiss_errors is None:
            return False
        if isinstance(dismiss_errors, str):
            return dismiss_errors in error_msg
        elif isinstance(dismiss_errors, (list, tuple)):
            return any(error in error_msg for error in dismiss_errors)
        return False

    def should_check_dtype(self):
        return self.api_config.api_name not in not_check_dtype
