from __future__ import annotations

import collections
import contextlib
import gc
import inspect
import os
from dataclasses import dataclass

import numpy
import paddle
import yaml

from .api_config.config_analyzer import (
    USE_CACHED_NUMPY,
    TensorConfig,
    get_cached_numpy_array,
)
from .api_config.dump_writer import DEFAULT_DUMP_DIR, DumpContext, dump_enabled
from .log_writer import log_worker
from .log_writer.log_comparison import log_accuracy_tolerance
from .log_writer.log_schema import MAX_CSV_CONFIG_LENGTH
from .log_writer.log_worker import write_to_log
from .runtime_config import TestRuntimeConfig

with open("tester/base_config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

forward_only_apis = frozenset(config.get("forward_only_apis", []))
handle_axes_api = frozenset(config.get("handle_axes_api", []))
not_check_dtype = frozenset(config.get("not_check_dtype", []))
rand_apis = frozenset(config.get("rand_apis", []))
stochastic_behavior_apis = frozenset(config.get("stochastic_behavior_apis", []))
single_op_no_signature_apis = frozenset(config.get("single_op_no_signature_apis", []))

paddle_error_dismiss = {}  # disabled: covered by the unified runtime error reporter
# paddle_error_dismiss = config.get("paddle_error_dismiss", {})
special_accuracy_atol_rtol = config.get("special_accuracy_atol_rtol", {})

with open("tester/api_config/torch_error_skip.txt") as f:
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
_GIB = 1024**3
_TENSOR_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}

_OUTPUT_GRAD_DTYPES = frozenset(
    [
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex64",
        "complex128",
    ]
)


def _dtype_element_size(dtype):
    return _TENSOR_DTYPE_BYTES.get(_dtype_name(dtype), 4)


def _tensor_element_size(value):
    try:
        return int(value.element_size())
    except (AttributeError, TypeError, ValueError):
        return _dtype_element_size(value.dtype)


def _dtype_name(value):
    return str(value).split(".")[-1]


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


def _release_gpu_allocator_caches(torch_module):
    gc.collect()
    try:
        torch_module.cuda.empty_cache()
    except Exception:
        pass
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _query_gpu_memory(torch_module):
    try:
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
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
):
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

    try:
        import torch as torch_module
    except (ImportError, OSError):
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


def gpu_mode_maybe_empty_cache(
    gpu_config,
    force=False,
    request_spill=False,
    probe_bytes=None,
    retained_tree_bytes=0,
    required_headroom_bytes=None,
):
    decision = gpu_mode_memory_decision(
        gpu_config,
        force=force,
        request_spill=request_spill,
        probe_bytes=probe_bytes,
        retained_tree_bytes=retained_tree_bytes,
        required_headroom_bytes=required_headroom_bytes,
    )
    return decision.should_spill if request_spill else decision.cleanup_performed


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
    # (Unimplemented): Paddle 已知不支持的功能，当前 case 无法有效验证
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


def get_arg(api_config, arg_pos, arg_name, default=None):
    if 0 <= arg_pos < len(api_config.args):
        return api_config.args[arg_pos]
    if arg_name in api_config.kwargs:
        return api_config.kwargs[arg_name]
    return default


no_signature_api_mappings = {
    f"paddle.Tensor.{method}": {
        "self": lambda cfg: get_arg(cfg, 0, "self"),
        "y": lambda cfg: get_arg(cfg, 1, "y"),
    }
    for method in single_op_no_signature_apis
}

# For _C_ops builtins that share the same parameter names as a public Paddle API,
# we reuse the public API's signature for argument binding instead of a manual mapping.
# Positional args beyond the public API's parameter count (e.g. a trailing `place`)
# are silently dropped.
_COPS_API_PUBLIC_ALIAS: dict[str, str] = {
    "paddle._C_ops.add_": "paddle.add",
    "paddle._C_ops.bitwise_not": "paddle.bitwise_not",
    "paddle._C_ops.clip": "paddle.clip",
    "paddle._C_ops.concat": "paddle.concat",
    "paddle._C_ops.flatten_": "paddle.flatten",
    "paddle._C_ops.matmul": "paddle.matmul",
    "paddle._C_ops.multiply_": "paddle.multiply",
    "paddle._C_ops.numel": "paddle.numel",
    "paddle._C_ops.put_along_axis_": "paddle.put_along_axis",
    "paddle._C_ops.reshape_": "paddle.reshape",
    "paddle._C_ops.scale_": "paddle.scale",
    "paddle._C_ops.subtract_": "paddle.subtract",
    "paddle._C_ops.transpose": "paddle.transpose",
    "paddle._C_ops.uniform": "paddle.uniform",
}

# Manual mappings for _C_ops that have no public API counterpart or whose signature
# differs materially from any public API.
no_signature_api_mappings.update(
    {
        # adamw_(param, grad, lr, moment1, moment2, moment2_max,
        #        beta1_pow, beta2_pow, master_param, skip_update,
        #        beta1, beta2, epsilon, lr_ratio, coeff, with_decay,
        #        lazy_mode, min_row_size, multi_precision, use_global_beta_pow, amsgrad)
        "paddle._C_ops.adamw_": {
            "param": lambda cfg: get_arg(cfg, 0, "param"),
            "grad": lambda cfg: get_arg(cfg, 1, "grad"),
            "learning_rate": lambda cfg: get_arg(cfg, 2, "learning_rate"),
            "moment1": lambda cfg: get_arg(cfg, 3, "moment1"),
            "moment2": lambda cfg: get_arg(cfg, 4, "moment2"),
            "moment2_max": lambda cfg: get_arg(cfg, 5, "moment2_max"),
            "beta1_pow": lambda cfg: get_arg(cfg, 6, "beta1_pow"),
            "beta2_pow": lambda cfg: get_arg(cfg, 7, "beta2_pow"),
            "master_param": lambda cfg: get_arg(cfg, 8, "master_param"),
            "skip_update": lambda cfg: get_arg(cfg, 9, "skip_update"),
            "beta1": lambda cfg: get_arg(cfg, 10, "beta1"),
            "beta2": lambda cfg: get_arg(cfg, 11, "beta2"),
            "epsilon": lambda cfg: get_arg(cfg, 12, "epsilon"),
            "lr_ratio": lambda cfg: get_arg(cfg, 13, "lr_ratio"),
            "coeff": lambda cfg: get_arg(cfg, 14, "coeff"),
            "with_decay": lambda cfg: get_arg(cfg, 15, "with_decay"),
            "lazy_mode": lambda cfg: get_arg(cfg, 16, "lazy_mode"),
            "min_row_size_to_use_multithread": lambda cfg: get_arg(
                cfg, 17, "min_row_size_to_use_multithread"
            ),
            "multi_precision": lambda cfg: get_arg(cfg, 18, "multi_precision"),
            "use_global_beta_pow": lambda cfg: get_arg(cfg, 19, "use_global_beta_pow"),
            "amsgrad": lambda cfg: get_arg(cfg, 20, "amsgrad"),
        },
        # full_(x, shape, value, dtype, place) — fills x in-place; no public API equivalent
        "paddle._C_ops.full_": {
            "x": lambda cfg: get_arg(cfg, 0, "x"),
            "shape": lambda cfg: get_arg(cfg, 1, "shape"),
            "value": lambda cfg: get_arg(cfg, 2, "value"),
            "dtype": lambda cfg: get_arg(cfg, 3, "dtype"),
        },
        # fused_linear_param_grad_add(x, dout, dweight, dbias, multi_precision, has_bias)
        "paddle._C_ops.fused_linear_param_grad_add": {
            "x": lambda cfg: get_arg(cfg, 0, "x"),
            "dout": lambda cfg: get_arg(cfg, 1, "dout"),
            "dweight": lambda cfg: get_arg(cfg, 2, "dweight"),
            "dbias": lambda cfg: get_arg(cfg, 3, "dbias"),
            "multi_precision": lambda cfg: get_arg(cfg, 4, "multi_precision"),
            "has_bias": lambda cfg: get_arg(cfg, 5, "has_bias"),
        },
        # gaussian(shape, mean, std, seed, dtype, place) — no paddle.gaussian public API
        "paddle._C_ops.gaussian": {
            "shape": lambda cfg: get_arg(cfg, 0, "shape"),
            "mean": lambda cfg: get_arg(cfg, 1, "mean"),
            "std": lambda cfg: get_arg(cfg, 2, "std"),
            "seed": lambda cfg: get_arg(cfg, 3, "seed"),
            "dtype": lambda cfg: get_arg(cfg, 4, "dtype"),
        },
        # matmul_grad(x, y, dout, transpose_x, transpose_y) -> (dx, dy)
        "paddle._C_ops.matmul_grad": {
            "x": lambda cfg: get_arg(cfg, 0, "x"),
            "y": lambda cfg: get_arg(cfg, 1, "y"),
            "dout": lambda cfg: get_arg(cfg, 2, "dout"),
            "transpose_x": lambda cfg: get_arg(cfg, 3, "transpose_x"),
            "transpose_y": lambda cfg: get_arg(cfg, 4, "transpose_y"),
        },
        # squared_l2_norm(x)
        "paddle._C_ops.squared_l2_norm": {
            "x": lambda cfg: get_arg(cfg, 0, "x"),
        },
        # swiglu_grad(x, y, dout) -> (dx, dy); y may be None (then x is split along last dim)
        "paddle._C_ops.swiglu_grad": {
            "x": lambda cfg: get_arg(cfg, 0, "x"),
            "y": lambda cfg: get_arg(cfg, 1, "y"),
            "dout": lambda cfg: get_arg(cfg, 2, "dout"),
        },
        # _run_custom_op(op_name, *args) — op_name dispatches to per-op sub-rules
        "paddle._C_ops._run_custom_op": {
            "op_name": lambda cfg: get_arg(cfg, 0, "op_name"),
            "arg1": lambda cfg: get_arg(cfg, 1, "arg1"),
            "arg2": lambda cfg: get_arg(cfg, 2, "arg2"),
            "arg3": lambda cfg: get_arg(cfg, 3, "arg3"),
            "arg4": lambda cfg: get_arg(cfg, 4, "arg4"),
        },
    }
)


class APITestBase:
    def __init__(self, api_config, use_torch=True, runtime_config=None):
        self.api_config = api_config
        self.api_config.use_torch = use_torch
        self.runtime_config = runtime_config or TestRuntimeConfig()
        self.gpu_mode_config = self.runtime_config.gpu_mode
        self.memory_governance_metrics = collections.Counter()
        self.dump_context = (
            DumpContext(
                os.environ.get("DUMP_DIR") or DEFAULT_DUMP_DIR, api_config=api_config.config
            )
            if dump_enabled()
            else None
        )
        self.output_grad_slots = []
        self.output_grad_slots_signature = None
        if use_torch:
            torch.set_num_threads(8)
            torch.set_printoptions(threshold=100, linewidth=120)

    def run_with_dump(self):
        """Execute the test with dump output capture and lifecycle reporting."""
        if self.dump_context is None:
            raise RuntimeError("run_with_dump() requires dump mode to be enabled")
        with self.dump_context.tee_output():
            try:
                return self.test()
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
        """Emit one standardized case-level result and write the matching logs."""
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

    def reset_random_state(self, seed=None):
        """Reset NumPy and framework RNGs for reproducible executions."""
        if seed is None:
            seed = getattr(self.runtime_config, "random_seed", 42)
        seed = int(seed)
        numpy.random.seed(seed)
        try:
            paddle.seed(seed)
            if paddle.device.is_compiled_with_cuda():
                try:
                    for device_id in range(paddle.device.cuda.device_count()):
                        paddle.framework.core.default_cuda_generator(device_id).manual_seed(seed)
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

    def clear_runtime_inputs(self, framework):
        """Release one framework's generated inputs after an execution."""
        attr_names = [f"{framework}_args", f"{framework}_kwargs"]
        for attr_name in attr_names:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        if not self.gpu_mode_config.enabled and framework == "torch":
            torch.cuda.empty_cache()
        elif not self.gpu_mode_config.enabled:
            paddle.device.cuda.empty_cache()

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
        # float8 dtypes are handled by TensorConfig / paddle_to_torch Rules;
        # do not skip accuracy-mode comparison solely because of float8 inputs.

        return False

    def _has_float8_tensor_config(self):
        """True if any TensorConfig arg/kwarg uses float8 dtype."""
        float8 = ("float8_e5m2", "float8_e4m3fn")

        def _check(obj):
            if isinstance(obj, TensorConfig):
                return obj.dtype in float8
            if isinstance(obj, (list, tuple)):
                return any(_check(x) for x in obj)
            return False

        if any(_check(a) for a in self.api_config.args):
            return True
        return any(_check(v) for v in self.api_config.kwargs.values())

    def need_check_grad(self):
        if self.is_forward_only():
            return False
        # float8 autograd / numpy grad path is unsupported in current torch tooling
        if self._has_float8_tensor_config():
            return False

        if self.api_config.api_name == "paddle.assign":
            has_list_arg = len(self.paddle_args_config) and isinstance(
                self.paddle_args_config[0], list
            )
            has_second_arg = (
                len(self.paddle_args_config) > 1 and self.paddle_args_config[1] is not None
            )
            has_output_kwarg = self.paddle_kwargs_config.get("output") is not None
            if has_list_arg or has_second_arg or has_output_kwarg:
                return False

        return True

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

        signature_cache = getattr(APITestBase.ana_torch_api_info, "_signature_cache", None)
        if signature_cache is None:
            signature_cache = {}
            APITestBase.ana_torch_api_info._signature_cache = signature_cache

        def get_signature(cache_key, api):
            if cache_key not in signature_cache:
                try:
                    signature_cache[cache_key] = inspect.signature(api)
                except ValueError:
                    signature_cache[cache_key] = None
            return signature_cache[cache_key]

        api_name = self.api_config.api_name
        if api_name in ("paddle.Tensor.__getitem__", "paddle.Tensor.__setitem__"):
            self.torch_args_config = self.api_config.args
            return True

        if api_name in ("paddle.Tensor.view", "paddle.view"):
            # paddle.view supports variadic int args: view(-1, 4096) in addition to view([-1, 4096])
            # inspect.bind incorrectly maps the second int to `name` param, so handle manually.
            args = self.api_config.args
            rest = args[1:]
            if len(rest) > 1 and all(isinstance(arg, int) for arg in rest):
                return finish({"x": args[0], "shape_or_dtype": list(rest)})

            paddle_sig = get_signature(api_name, self.paddle_api)
            if paddle_sig is None:
                raise ValueError(f"API {api_name} has no inspectable signature")
            paddle_bound_args = paddle_sig.bind(*args, **self.api_config.kwargs)
            return finish(paddle_bound_args.arguments)

        if api_name in ("paddle.Tensor.reshape", "paddle.reshape"):
            # paddle.reshape supports variadic int args: reshape(1, 2048, -1) in addition to reshape([1, 2048, -1])
            # Tensor.reshape signature is (x, shape, name=None), so bind() rejects multiple ints.
            args = self.api_config.args
            rest = args[1:]
            if len(rest) >= 1 and all(isinstance(arg, int) for arg in rest):
                return finish({"x": args[0], "shape": list(rest)})

        if api_name in no_signature_api_mappings:
            # For APIs without signatures, use the external mapping dict
            mapping = no_signature_api_mappings[api_name]
            return finish(
                {key: get_value_func(self.api_config) for key, get_value_func in mapping.items()}
            )

        # For APIs with signatures, use paddle_sig.bind to get arguments.
        paddle_sig = get_signature(api_name, self.paddle_api)
        if paddle_sig is None:
            # _C_ops builtins have no inspectable signature. If the op has a public
            # API counterpart with the same params, use its signature.
            public_api_name = _COPS_API_PUBLIC_ALIAS.get(api_name)
            if public_api_name is None:
                # No alias and no manual mapping — forward-only, skip torch comparison.
                return True

            paddle_sig = get_signature(public_api_name, eval(public_api_name))
            if paddle_sig is None:
                raise ValueError(f"API {public_api_name} has no inspectable signature")
            positional_count = sum(
                1
                for param in paddle_sig.parameters.values()
                if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            )
            valid_names = set(paddle_sig.parameters)
            filtered_kwargs = {
                key: value for key, value in self.api_config.kwargs.items() if key in valid_names
            }
            paddle_bound_args = paddle_sig.bind(
                *self.api_config.args[:positional_count], **filtered_kwargs
            )
            return finish(paddle_bound_args.arguments)

        paddle_bound_args = paddle_sig.bind(*self.api_config.args, **self.api_config.kwargs)
        paddle_args_dict = paddle_bound_args.arguments
        # fix paddle.arange wrong binding
        if api_name == "paddle.arange":
            # if end is not provided, use the 'start' kwargs as end
            if "end" not in paddle_args_dict:
                paddle_args_dict["end"] = paddle_args_dict["start"]
                paddle_args_dict["start"] = 0

        return finish(paddle_args_dict)

    def _handle_list_or_tuple(
        self, config_items, is_tuple=False, index=None, key=None, list_index=None
    ):
        """处理 list 或 tuple"""
        if list_index is None:
            list_index = []
        need_axes_handling = self.api_config.api_name in handle_axes_api
        need_indices_handling = self.api_config.api_name == "paddle.index_put"

        if need_indices_handling and (index == 1 or key == "indices"):
            return self._handle_indices_arg(config_items, is_tuple)
        elif need_axes_handling and (index == 1 or key == "axis"):
            return self._handle_axis_arg(config_items, is_tuple)

        tmp = []
        for i, item in enumerate(config_items):
            current_list_index = [*list_index, i]
            if isinstance(item, (list, tuple)):
                is_nested_tuple = isinstance(item, tuple)
                processed_item = self._handle_list_or_tuple(
                    item,
                    is_tuple=is_nested_tuple,
                    index=index,
                    key=key,
                    list_index=current_list_index,
                )
            elif isinstance(item, TensorConfig):
                processed_item = item.get_numpy_tensor(
                    self.api_config, index=index, key=key, list_index=current_list_index
                )
            else:
                processed_item = item
            tmp.append(processed_item)
        return tuple(tmp) if is_tuple else tmp

    def _handle_axis_arg(self, config_items, is_tuple=False):
        """处理 axis 参数"""
        x = (
            self.paddle_args_config[0]
            if len(self.paddle_args_config) > 0
            else self.paddle_kwargs_config["x"]
        )
        max_dim = max(len(x.shape), 1)  # scalar

        tmp = []
        used_axes = set()
        tensor_configs = []

        for item in config_items:
            if isinstance(item, TensorConfig):
                if item.shape not in [[], [1]] or item.dtype not in ["int32", "int64"]:
                    raise ValueError(
                        f"Invalid TensorConfig for axis: shape {item.shape} or dtype {item.dtype}"
                    )
                tensor_configs.append(item)
                tmp.append(0)  # placeholder
            elif isinstance(item, int):
                if not (-max_dim <= item < max_dim):
                    raise ValueError(f"Axis value {item} out of range [-{max_dim}, {max_dim})")
                positive_axis = item + max_dim if item < 0 else item
                if positive_axis in used_axes:
                    raise ValueError(f"Duplicate axis value: {item}")
                used_axes.add(positive_axis)
                tmp.append(item)
            else:
                raise ValueError(f"Invalid item type for axis: {type(item)}")

        if tensor_configs:
            available_dims = list(set(range(max_dim)) - used_axes)
            if len(available_dims) < len(tensor_configs):
                raise ValueError(
                    f"Not enough available dimensions ({len(available_dims)}) for {len(tensor_configs)} TensorConfig items"
                )
            selected_dims = numpy.random.choice(
                available_dims, size=len(tensor_configs), replace=False
            )
            mask = numpy.random.randint(0, 2, size=len(tensor_configs)).astype(bool)
            final_dims = numpy.where(mask, selected_dims - max_dim, selected_dims)
            tensor_idx = 0
            for i, item in enumerate(config_items):
                if isinstance(item, TensorConfig):
                    item.fill_numpy_tensor(final_dims[tensor_idx])
                    tmp[i] = item.get_numpy_tensor(self.api_config)
                    tensor_idx += 1
        return tuple(tmp) if is_tuple else tmp

    def _generate_int_indices(self, item_shape, dim_size):
        num_elements = numpy.prod(item_shape).item()
        if num_elements > dim_size:
            indices_flat = numpy.random.randint(-dim_size, dim_size, size=num_elements)
        else:
            indices_flat = numpy.random.choice(dim_size, size=num_elements, replace=False)
        return indices_flat.reshape(item_shape)

    def _generate_constrained_bool_mask(self, shape, num_true):
        mask_size = numpy.prod(shape).item()
        if mask_size < num_true:
            raise ValueError(
                f"Cannot generate a mask with {num_true} true values in a {mask_size} element mask"
            )
        mask_flat = numpy.zeros(mask_size, dtype="bool")
        true_indices = numpy.random.choice(mask_size, num_true, replace=False)
        mask_flat[true_indices] = True
        return mask_flat.reshape(shape)

    def _broadcast_or_raise(self, shapes):
        return numpy.broadcast_shapes(*[tuple(s) for s in shapes])

    def _handle_indices_arg(self, config_items, is_tuple=False):
        x = (
            self.paddle_args_config[0]
            if len(self.paddle_args_config) > 0
            else self.paddle_kwargs_config["x"]
        )
        value = (
            self.paddle_args_config[2]
            if len(self.paddle_args_config) > 2
            else self.paddle_kwargs_config["value"]
        )
        x_shape = x.shape
        value_shape = value.shape
        int_index_shapes = []
        has_bool_index = False
        dims_consumed = 0
        for item in config_items:
            if item.dtype == "bool":
                b_rank = len(item.shape)
                has_bool_index = True
                dims_consumed += b_rank
            else:
                int_index_shapes.append(tuple(item.shape))
                dims_consumed += 1

        if dims_consumed > len(x_shape):
            raise ValueError(
                f"Too many indices: consume {dims_consumed} dims but x has {len(x_shape)} dims"
            )
        num_true_needed = -1
        num_remaining_dims = len(x_shape) - dims_consumed
        advanced_shape = ()
        if int_index_shapes:
            try:
                advanced_shape = self._broadcast_or_raise(int_index_shapes)
                # give a default 1
                if (
                    has_bool_index
                    and len(value_shape) > num_remaining_dims
                    and advanced_shape[-1] == 1
                    and value_shape[-num_remaining_dims - 1] != 1
                ):
                    advanced_shape = (
                        *advanced_shape[:-1],
                        value_shape[-num_remaining_dims - 1],
                    )
                num_true_needed = advanced_shape[-1]
            except Exception:
                raise ValueError(
                    f"Incompatible integer index shapes for broadcasting: {int_index_shapes}"
                )
        elif has_bool_index:
            if len(value_shape) > num_remaining_dims:
                advanced_shape = (value_shape[0],)
                num_true_needed = value_shape[0]
            else:
                # give a default 1, other valid(not out of bound) shape also can.
                advanced_shape = (1,)
                num_true_needed = 1
        res_dims = advanced_shape + tuple(x_shape[dims_consumed:])
        try:
            # only for checking.
            self._broadcast_or_raise([value_shape, res_dims])
        except ValueError:
            raise ValueError(
                f"Value shape {value_shape} cannot be broadcast to the indexed shape {res_dims}."
            )
        processed_indices = []
        x_dim_cursor = 0
        for item in config_items:
            if item.dtype == "bool":
                if num_true_needed < 0:
                    raise ValueError(
                        "Cannot determine the number of True elements for the boolean mask."
                    )
                item.numpy_tensor = self._generate_constrained_bool_mask(
                    item.shape, num_true_needed
                )
                x_dim_cursor += len(item.shape)
            else:
                x_dim_to_index = x_shape[x_dim_cursor]
                indices = self._generate_int_indices(item.shape, x_dim_to_index)
                item.numpy_tensor = indices.astype(item.dtype)
                x_dim_cursor += 1

            processed_indices.append(item.numpy_tensor)

        return tuple(processed_indices) if is_tuple else processed_indices

    def gen_numpy_input(self):
        for i, arg_config in enumerate(self.paddle_args_config):
            if isinstance(arg_config, (list, tuple)):
                is_tuple = isinstance(arg_config, tuple)
                self._handle_list_or_tuple(arg_config, is_tuple=is_tuple, index=i)
            elif isinstance(arg_config, TensorConfig):
                arg_config.get_numpy_tensor(self.api_config, index=i)
        for key, kwarg_config in self.paddle_kwargs_config.items():
            if isinstance(kwarg_config, (list, tuple)):
                is_tuple = isinstance(kwarg_config, tuple)
                self._handle_list_or_tuple(kwarg_config, is_tuple=is_tuple, key=key)
            elif isinstance(kwarg_config, TensorConfig):
                kwarg_config.get_numpy_tensor(self.api_config, key=key)
        return True

    def _materialize_paddle_config_value(self, config_value, *, clear_tensor=True):
        """Materialize a Paddle config tree into live values, optionally clearing cache."""
        if isinstance(config_value, TensorConfig):
            paddle_tensor = config_value.get_paddle_tensor(self.api_config)
            if clear_tensor:
                config_value.clear_paddle_tensor()
            return paddle_tensor
        if isinstance(config_value, list):
            return [
                self._materialize_paddle_config_value(item, clear_tensor=clear_tensor)
                for item in config_value
            ]
        if isinstance(config_value, tuple):
            return tuple(
                self._materialize_paddle_config_value(item, clear_tensor=clear_tensor)
                for item in config_value
            )
        return config_value

    def gen_paddle_input(self):
        """Generate Paddle inputs from config and materialize TensorConfig leaves.

        Call gen_numpy_input() first because gen_paddle_input() does not pass
        index or key to get_paddle_tensor() or get_numpy_tensor().
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
        elif self.api_config.api_name == "paddle.nn.functional.cross_entropy":
            use_softmax = get_arg(self.api_config, 8, "use_softmax", True)
            if not use_softmax:
                axis = get_arg(self.api_config, 7, "axis", -1)
                if len(self.paddle_args) > 0:
                    self.paddle_args[0] = paddle.nn.functional.softmax(
                        self.paddle_args[0], axis=axis
                    )
                else:
                    self.paddle_kwargs["input"] = paddle.nn.functional.softmax(
                        self.paddle_kwargs["input"], axis=axis
                    )

        # In-place ops require a non-leaf tensor as target.  Always copy inputs
        # so the in-place op doesn't hit "Leaf Var can't use inplace strategy".
        # (Previously guarded by need_check_grad(), but forward_only in-place ops
        # still need the copy to avoid Paddle's leaf-variable restriction.)
        if (
            self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__"
        ) or self.api_config.api_name == "paddle.Tensor.__setitem__":
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

    def _collect_tensor_leaves(self, value, result, tensor_type):
        if isinstance(value, tensor_type):
            result.append(value)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._collect_tensor_leaves(item, result, tensor_type)
            return
        if isinstance(value, dict):
            for item in value.values():
                self._collect_tensor_leaves(item, result, tensor_type)

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

    def get_cached_numpy(self, dtype, shape, generation_kind="output_grad", scale=1.0):
        return get_cached_numpy_array(dtype, shape, generation_kind=generation_kind, scale=scale)

    def _make_numpy_output_grad(self, output):
        dtype = _dtype_name(output.dtype)
        if USE_CACHED_NUMPY:
            dtype = "float32" if dtype == "bfloat16" else dtype
            return self.get_cached_numpy(dtype, output.shape)
        if "int" in dtype:
            return numpy.random.randint(-65535, 65535, size=output.shape).astype(dtype)
        if dtype in ["float8_e5m2", "float8_e4m3fn"]:
            numpy_dtype = "float16"
        elif dtype == "bfloat16":
            numpy_dtype = "float32"
        else:
            numpy_dtype = dtype
        return (numpy.random.random(output.shape) - 0.5).astype(numpy_dtype)

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
            dtype=self.convert_dtype_to_torch_type(dtype) if dtype != "bfloat16" else torch.float32,
        )
        if dtype == "bfloat16":
            result_output_grad = result_output_grad.to(dtype=torch.bfloat16)
        return result_output_grad

    def _make_torch_output_grad(self, shape, dtype, device=None):
        torch_dtype = self.convert_dtype_to_torch_type(dtype)
        device = device or torch.device("cuda", torch.cuda.current_device())
        if "int" in dtype:
            return torch.randint(-65535, 65535, tuple(shape), device=device, dtype=torch.int64).to(
                dtype=torch_dtype
            )
        if dtype in ["float8_e5m2", "float8_e4m3fn"]:
            base_dtype = torch.float16
        elif dtype == "bfloat16":
            base_dtype = torch.float32
        elif dtype.startswith("complex"):
            real_dtype = torch.float32 if dtype == "complex64" else torch.float64
            real_part = torch.rand(tuple(shape), device=device, dtype=real_dtype) - 0.5
            imag_part = torch.rand(tuple(shape), device=device, dtype=real_dtype) - 0.5
            return (real_part + 1j * imag_part).to(dtype=torch_dtype)
        else:
            base_dtype = torch_dtype
        return (torch.rand(tuple(shape), device=device, dtype=base_dtype) - 0.5).to(
            dtype=torch_dtype
        )

    def _make_paddle_output_grad(self, shape, dtype, place=None):
        if "int" in dtype:
            return paddle.randint(-65535, 65535, tuple(shape), dtype="int64", device=place).cast(
                dtype
            )
        if dtype in ["float8_e5m2", "float8_e4m3fn"]:
            base_dtype = "float16"
        elif dtype == "bfloat16":
            base_dtype = "float32"
        elif dtype.startswith("complex"):
            real_dtype = "float32" if dtype == "complex64" else "float64"
            real_part = paddle.rand(tuple(shape), dtype=real_dtype, device=place) - 0.5
            imag_part = paddle.rand(tuple(shape), dtype=real_dtype, device=place) - 0.5
            return (real_part + 1j * imag_part).cast(dtype)
        else:
            base_dtype = dtype
        return (paddle.rand(tuple(shape), dtype=base_dtype, device=place) - 0.5).cast(dtype)

    def _use_gpu_output_grad(self, output):
        if not self.gpu_mode_config.enabled:
            return False
        dtype = _dtype_name(output.dtype)
        if isinstance(output, torch.Tensor):
            return output.device.type != "cpu" and dtype in _OUTPUT_GRAD_DTYPES
        if isinstance(output, paddle.Tensor):
            return output.place.is_gpu_place() and dtype in _OUTPUT_GRAD_DTYPES
        return False

    @staticmethod
    def _output_grad_signature(outputs):
        return tuple((_dtype_name(output.dtype), tuple(output.shape)) for output in outputs)

    @staticmethod
    def _tensor_matches_output_place(tensor, output):
        if isinstance(tensor, torch.Tensor) and isinstance(output, torch.Tensor):
            tensor_device = 0 if tensor.device.index is None else int(tensor.device.index)
            output_device = 0 if output.device.index is None else int(output.device.index)
            return tensor.device.type == output.device.type and tensor_device == output_device
        if isinstance(tensor, paddle.Tensor) and isinstance(output, paddle.Tensor):
            if tensor.place.is_cpu_place() and output.place.is_cpu_place():
                return True
            if tensor.place.is_gpu_place() and output.place.is_gpu_place():
                return int(tensor.place.gpu_device_id()) == int(output.place.gpu_device_id())
        return False

    def _make_gpu_paddle_output_grad_slot(self, output):
        dtype = _dtype_name(output.dtype)
        paddle_grad = self._make_paddle_output_grad(output.shape, dtype, place=output.place)
        paddle_grad.stop_gradient = False
        return OutputGradSlot(seed_numpy=paddle_grad.numpy(), paddle_grad=paddle_grad)

    def _make_gpu_output_grad_pair_slot(self, output):
        dtype = _dtype_name(output.dtype)
        torch_grad = self._make_torch_output_grad(
            output.shape,
            dtype,
            device=output.device,
        ).detach()
        paddle_grad = paddle.utils.dlpack.from_dlpack(
            torch.utils.dlpack.to_dlpack(torch_grad.clone())
        )
        paddle_grad.stop_gradient = False
        return OutputGradSlot(
            seed_numpy=torch_grad.detach().cpu().numpy(),
            paddle_grad=paddle_grad,
            torch_grad=torch_grad,
        )

    def clear_output_grad_cache(self):
        """Drop all cached output-grad seeds for the current execution."""
        self.output_grad_slots.clear()
        self.output_grad_slots_signature = None

    def _output_grad_slot_to_paddle(self, slot, output):
        if slot.paddle_grad is not None and self._tensor_matches_output_place(
            slot.paddle_grad, output
        ):
            return slot.paddle_grad
        if slot.seed_numpy is not None:
            return self._numpy_output_grad_to_paddle(slot.seed_numpy, output)
        if slot.torch_grad is not None:
            return self._numpy_output_grad_to_paddle(slot.torch_grad.detach().cpu().numpy(), output)
        raise RuntimeError("output grad slot is empty")

    def _output_grad_slot_to_torch(self, slot, output):
        if slot.torch_grad is not None and self._tensor_matches_output_place(
            slot.torch_grad, output
        ):
            return slot.torch_grad
        if slot.seed_numpy is not None:
            return self._numpy_output_grad_to_torch(slot.seed_numpy, output)
        if slot.paddle_grad is not None:
            return self._numpy_output_grad_to_torch(slot.paddle_grad.numpy(), output)
        raise RuntimeError("output grad slot is empty")

    def _prepare_output_grad_slots(self, result_outputs):
        signature = self._output_grad_signature(result_outputs)
        if self.output_grad_slots_signature != signature:
            self.output_grad_slots = []
            self.output_grad_slots_signature = signature
        if len(self.output_grad_slots) != len(result_outputs):
            self.output_grad_slots = []
            for output in result_outputs:
                if self._use_gpu_output_grad(output):
                    if isinstance(output, torch.Tensor):
                        self.output_grad_slots.append(self._make_gpu_output_grad_pair_slot(output))
                    else:
                        self.output_grad_slots.append(
                            self._make_gpu_paddle_output_grad_slot(output)
                        )
                else:
                    self.output_grad_slots.append(
                        OutputGradSlot(seed_numpy=self._make_numpy_output_grad(output))
                    )

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
            outputs, (paddle.autograd.autograd.Hessian, paddle.autograd.autograd.Jacobian)
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
                    output, (paddle.autograd.autograd.Hessian, paddle.autograd.autograd.Jacobian)
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

        paddle_autograd_dtypes = {
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "complex64",
            "complex128",
        }
        result_outputs = [
            output
            for output in result_outputs
            if not output.stop_gradient
            and str(output.dtype).split(".")[-1] in paddle_autograd_dtypes
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

    def convert_dtype_to_torch_type(self, dtype):
        # for python built-in types, mappings are int -> torch.int64, bool -> torch.bool, float -> torch.float64, complex -> torch.complex128, None -> None
        if dtype in [
            "float32",
            numpy.float32,
            paddle.float32,
            paddle.base.libpaddle.VarDesc.VarType.FP32,
        ]:
            return torch.float32
        elif dtype in [
            "float16",
            numpy.float16,
            paddle.float16,
            paddle.base.libpaddle.VarDesc.VarType.FP16,
        ]:
            return torch.float16
        elif dtype in [
            "float64",
            "float",
            "double",
            numpy.float64,
            paddle.float64,
            paddle.base.libpaddle.VarDesc.VarType.FP64,
            float,
        ]:
            return torch.float64
        elif dtype in [
            "int16",
            numpy.int16,
            paddle.int16,
            paddle.base.libpaddle.VarDesc.VarType.INT16,
        ]:
            return torch.int16
        elif dtype in [
            "int8",
            numpy.int8,
            paddle.int8,
            paddle.base.libpaddle.VarDesc.VarType.INT8,
        ]:
            return torch.int8
        elif dtype in [
            "bool",
            numpy.bool_,
            paddle.bool,
            paddle.base.libpaddle.VarDesc.VarType.BOOL,
            bool,
        ]:
            return torch.bool
        elif dtype in [
            "bfloat16",
            "uint16",
            numpy.uint16,
            paddle.bfloat16,
            paddle.base.libpaddle.VarDesc.VarType.BF16,
        ]:
            return torch.bfloat16
        elif dtype in [
            "uint8",
            numpy.uint8,
            paddle.uint8,
            paddle.base.libpaddle.VarDesc.VarType.UINT8,
        ]:
            return torch.uint8
        elif dtype in [
            "int32",
            numpy.int32,
            paddle.int32,
            paddle.base.libpaddle.VarDesc.VarType.INT32,
        ]:
            return torch.int32
        elif dtype in [
            "int64",
            "int",
            numpy.int64,
            paddle.int64,
            paddle.base.libpaddle.VarDesc.VarType.INT64,
            int,
        ]:
            return torch.int64
        elif dtype in [
            "complex64",
            numpy.complex64,
            paddle.complex64,
            paddle.base.libpaddle.VarDesc.VarType.COMPLEX64,
        ]:
            return torch.complex64
        elif dtype in [
            "complex128",
            numpy.complex128,
            paddle.complex128,
            paddle.base.libpaddle.VarDesc.VarType.COMPLEX128,
            complex,
        ]:
            return torch.complex128
        elif dtype is None:
            return None
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

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
        if isinstance(config_value, TensorConfig):
            torch_tensor = config_value.get_torch_tensor(self.api_config)
            config_value.clear_torch_tensor()
            return torch_tensor
        if isinstance(config_value, list):
            return [self._materialize_torch_config_value(item) for item in config_value]
        if isinstance(config_value, tuple):
            return tuple(self._materialize_torch_config_value(item) for item in config_value)
        if convert_dtype or isinstance(
            config_value, (paddle.dtype, paddle.base.libpaddle.VarDesc.VarType)
        ):
            return self.convert_dtype_to_torch_type(config_value)
        return config_value

    def gen_torch_input(self):
        """Generate Torch inputs from config and materialize TensorConfig leaves.

        Call gen_numpy_input() first because gen_torch_input() does not pass
        index or key to get_torch_tensor() or get_numpy_tensor().
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

        if (
            self.api_config.api_name[-1] == "_" and self.api_config.api_name[-2:] != "__"
        ) or self.api_config.api_name == "paddle.Tensor.__setitem__":
            self.torch_args, self.torch_kwargs = self.copy_torch_input()

        if not self.gpu_mode_config.enabled:
            torch.cuda.empty_cache()
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
        is_check_dtype = self.api_config.api_name not in not_check_dtype

        if not actual_paddle_tensor.is_contiguous():
            actual_paddle_tensor = actual_paddle_tensor.contiguous()
        actual_paddle_tensor = actual_paddle_tensor.cpu().detach()

        if not expected_paddle_tensor.is_contiguous():
            expected_paddle_tensor = expected_paddle_tensor.contiguous()
        expected_paddle_tensor = expected_paddle_tensor.cpu().detach()

        actual_paddle_dlpack = paddle.utils.dlpack.to_dlpack(actual_paddle_tensor)  # type: ignore[reportGeneralTypeIssues]
        converted_actual_tensor = torch.utils.dlpack.from_dlpack(actual_paddle_dlpack)  # type: ignore[reportGeneralTypeIssues]

        expected_paddle_dlpack = paddle.utils.dlpack.to_dlpack(expected_paddle_tensor)  # type: ignore[reportGeneralTypeIssues]
        converted_expected_tensor = torch.utils.dlpack.from_dlpack(expected_paddle_dlpack)  # type: ignore[reportGeneralTypeIssues]

        def error_msg(msg):
            return (
                f"Not equal to tolerance rtol={rtol}, atol={atol}\n"
                f"{msg}\n"
                f"ACTUAL: (shape={converted_actual_tensor.shape}, dtype={converted_actual_tensor.dtype})\n"
                f"{converted_actual_tensor}\n"
                f"DESIRED: (shape={converted_expected_tensor.shape}, dtype={converted_expected_tensor.dtype})\n"
                f"{converted_expected_tensor}"
            )

        bitwise_alignment = getattr(self, "bitwise_alignment", False)
        if not bitwise_alignment and self.api_config.api_name in special_accuracy_atol_rtol:
            atol, rtol = special_accuracy_atol_rtol[self.api_config.api_name]

        try:
            torch.testing.assert_close(
                converted_actual_tensor,
                converted_expected_tensor,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                check_dtype=is_check_dtype,
                msg=error_msg,
            )
        except Exception as e:
            error_str = str(e)
            if error_str.startswith("Comparing"):
                if os.environ.get("PADDLEAPITEST_NP_FALLBACK", "0") == "1":
                    # torch_assert OOM or internal error, fallback to np_assert
                    print(
                        "[torch_assert_OOM] torch.testing.assert_close OOM, fallback to np_assert",
                        flush=True,
                    )
                    # numpy does not support bfloat16/float8, cast to float32 for comparison
                    actual_np = actual_paddle_tensor
                    expected_np = expected_paddle_tensor
                    if hasattr(actual_np, "dtype"):
                        dtype_str = str(actual_np.dtype)
                        if "bfloat16" in dtype_str or "float8" in dtype_str:
                            actual_np = (
                                actual_np.astype("float32")
                                if hasattr(actual_np, "astype")
                                else actual_np.float()
                            )
                    if hasattr(expected_np, "dtype"):
                        dtype_str = str(expected_np.dtype)
                        if "bfloat16" in dtype_str or "float8" in dtype_str:
                            expected_np = (
                                expected_np.astype("float32")
                                if hasattr(expected_np, "astype")
                                else expected_np.float()
                            )
                    self.np_assert_accuracy(
                        actual_np.numpy() if hasattr(actual_np, "numpy") else actual_np,
                        expected_np.numpy() if hasattr(expected_np, "numpy") else expected_np,
                        atol,
                        rtol,
                    )
                else:
                    raise RuntimeError(
                        "[torch_assert_OOM] torch.testing.assert_close OOM on large tensor comparison"
                    )
            else:
                raise

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
    def _logical_slab_to_torch(value, index, compare_on_cpu):
        slab = value[index].detach()
        if not slab.is_contiguous():
            slab = slab.contiguous()
        if compare_on_cpu:
            slab = slab.cpu()
        if isinstance(slab, paddle.Tensor):
            return torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(slab)  # type: ignore[reportGeneralTypeIssues]
            )
        return slab

    def _torch_assert_accuracy_in_logical_slabs(
        self,
        actual,
        expected,
        atol,
        rtol,
        error_msg,
        working_bytes,
        compare_on_cpu,
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
                actual_chunk = self._logical_slab_to_torch(actual, index, compare_on_cpu)
                expected_chunk = self._logical_slab_to_torch(expected, index, compare_on_cpu)
                if actual_chunk.device != expected_chunk.device:
                    expected_chunk = expected_chunk.to(device=actual_chunk.device)

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
                actual_chunk = self._logical_slab_to_torch(actual, index, compare_on_cpu).reshape(
                    -1
                )
                expected_chunk = self._logical_slab_to_torch(
                    expected, index, compare_on_cpu
                ).reshape(-1)
                if actual_chunk.device != expected_chunk.device:
                    expected_chunk = expected_chunk.to(device=actual_chunk.device)
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
        test_tol = getattr(self, "test_tol", False)
        is_backward = getattr(self, "is_backward", False)
        if test_tol:
            atol, rtol = 0.0, 0.0

        def is_cpu_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.device.type == "cpu"
            if isinstance(value, paddle.Tensor):
                return value.place.is_cpu_place()
            return False

        dual_gpu = bool(getattr(self.gpu_mode_config, "dual_gpu", False))
        compare_on_cpu = not dual_gpu and (
            test_tol
            or not self.gpu_mode_config.enabled
            or is_cpu_tensor(actual)
            or is_cpu_tensor(expected)
        )
        tensor_types = (paddle.Tensor, torch.Tensor)
        if not isinstance(actual, tensor_types):
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(actual)}")
        if not isinstance(expected, tensor_types):
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(expected)}")
        if dual_gpu and (is_cpu_tensor(actual) or is_cpu_tensor(expected)):
            raise RuntimeError("dual GPU comparisons require CUDA tensors on both sides")

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
                working_bytes = int(getattr(self, "comparison_working_bytes", 1024**3))
                self._torch_assert_accuracy_in_logical_slabs(
                    actual,
                    expected,
                    atol,
                    rtol,
                    slab_error_msg,
                    working_bytes,
                    compare_on_cpu,
                    is_check_dtype,
                )
                if test_tol:
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
                if test_tol:
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
            if compare_on_cpu:
                actual = actual.cpu()
            actual_tensor = torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(actual)  # type: ignore[reportGeneralTypeIssues]
            )
        elif isinstance(actual, torch.Tensor):
            if not actual.is_contiguous():
                actual = actual.contiguous()
            actual_tensor = actual.detach()
            if compare_on_cpu:
                actual_tensor = actual_tensor.cpu()
        else:
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(actual)}")

        if isinstance(expected, paddle.Tensor):
            if not expected.is_contiguous():
                expected = expected.contiguous()
            expected = expected.detach()
            if compare_on_cpu:
                expected = expected.cpu()
            expected_tensor = torch.utils.dlpack.from_dlpack(
                paddle.utils.dlpack.to_dlpack(expected)  # type: ignore[reportGeneralTypeIssues]
            )
        elif isinstance(expected, torch.Tensor):
            if not expected.is_contiguous():
                expected = expected.contiguous()
            expected_tensor = expected.detach()
            if compare_on_cpu:
                expected_tensor = expected_tensor.cpu()
        else:
            raise TypeError(f"Expected Paddle or Torch tensor, but got {type(expected)}")

        if actual_tensor.device != expected_tensor.device:
            expected_tensor = expected_tensor.to(device=actual_tensor.device)

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
            working_bytes = 1024**3
            estimated_temp_bytes = actual_tensor.numel() * max(
                32,
                4 * max(actual_tensor.element_size(), expected_tensor.element_size()) + 16,
            )
            # Small tensors always use torch.testing.assert_close directly. Avoid
            # querying the CUDA driver for a working budget on that hot path.
            needs_chunk_budget = estimated_temp_bytes > working_bytes
            if actual_tensor.is_cuda and needs_chunk_budget:
                insufficient_dual_headroom = False
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(actual_tensor.device.index)
                    min_working_bytes = 256 * 1024**2
                    memory_budget = float(
                        getattr(self.gpu_mode_config, "memory_budget", 0.0) or 0.0
                    )
                    comparison_device_id = getattr(
                        self.gpu_mode_config, "comparison_device_id", None
                    )
                    if actual_tensor.device.index == comparison_device_id:
                        memory_budget = float(
                            getattr(
                                self.gpu_mode_config,
                                "comparison_memory_budget",
                                memory_budget,
                            )
                            or memory_budget
                        )

                    if dual_gpu:
                        budget_bytes = int(memory_budget * (1024**3)) if memory_budget > 0 else 0
                        reserve_bytes = self._gpu_safety_reserve_bytes(total_bytes, budget_bytes)
                        working_bytes = self._comparison_workspace_bytes(
                            free_bytes, reserve_bytes, dual_gpu=True
                        )
                        insufficient_dual_headroom = working_bytes == 0
                    else:
                        reserved_bytes = torch.cuda.memory_reserved(actual_tensor.device)
                        working_bytes = self._comparison_workspace_bytes(free_bytes)

                    if memory_budget > 0 and not dual_gpu:
                        budget_headroom = max(
                            0,
                            int(memory_budget * (1024**3)) - reserved_bytes,
                        )
                        working_bytes = min(
                            working_bytes,
                            max(min_working_bytes, budget_headroom // 4),
                        )
                except Exception:
                    pass
                if insufficient_dual_headroom:
                    raise RuntimeError(
                        "[torch_assert_OOM] comparison card lacks the reserved headroom "
                        "for a bounded GPU comparison"
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
                if test_tol:
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
            if test_tol:
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
            if test_tol:
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
        """Keep a small physical reserve without double-counting allocator state."""
        minimum_reserve = 1 * 1024**3
        total_bytes = max(0, int(total_bytes))
        budget_bytes = max(0, int(budget_bytes))
        if budget_bytes > 0 and budget_bytes < total_bytes:
            return max(minimum_reserve, total_bytes - budget_bytes)
        return max(minimum_reserve, total_bytes // 20)

    def test(self):
        pass

    def clear_tensor(self):
        if not self._clear_tensor_config_cache("clear_tensor"):
            return
        if self.gpu_mode_config.enabled:
            gpu_mode_maybe_empty_cache(self.gpu_mode_config)
        else:
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()

    def _for_each_tensor_config(self, callback, *roots):
        """Apply callback to every TensorConfig in the parsed argument tree."""
        seen = set()

        def visit(value):
            if isinstance(value, TensorConfig):
                if id(value) not in seen:
                    seen.add(id(value))
                    callback(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)

        if not roots:
            roots = (
                getattr(self, "paddle_args_config", ()),
                getattr(self, "paddle_kwargs_config", {}),
            )
        for value in roots:
            visit(value)

    def _tensor_config_roots(self):
        return (
            getattr(self, "paddle_args_config", ()),
            getattr(self, "paddle_kwargs_config", {}),
            getattr(self, "paddle_merged_kwargs_config", {}),
            getattr(self, "torch_args_config", ()),
            getattr(self, "torch_kwargs_config", {}),
        )

    def _clear_tensor_config_cache(self, clear_method_name):
        roots = self._tensor_config_roots()
        if not any(roots):
            return False
        self._for_each_tensor_config(
            lambda config: getattr(config, clear_method_name)(),
            *roots,
        )
        return True

    def _map_tensor_tree(self, value, tensor_mapper):
        if isinstance(value, (torch.Tensor, paddle.Tensor)):
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
        seen = set()

        def visit(item):
            if isinstance(item, (torch.Tensor, paddle.Tensor)):
                if id(item) in seen:
                    return
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

    @staticmethod
    def tensor_is_gpu(value):
        if isinstance(value, torch.Tensor):
            return value.device.type != "cpu"
        if isinstance(value, paddle.Tensor):
            return value.place.is_gpu_place()
        return False

    @staticmethod
    def tensor_gpu_device_id(value):
        if isinstance(value, torch.Tensor):
            return 0 if value.device.index is None else int(value.device.index)
        return int(value.place.gpu_device_id())

    def tensor_tree_nbytes(self, value):
        """Estimate logical bytes held by unique tensor leaves in a result tree."""
        return sum(
            int(tensor.numel()) * _tensor_element_size(tensor)
            for tensor in self.iter_unique_tensor_tree_leaves(value)
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
        code = convert_result.code
        source_lines = (*code.preprocess, *code.core, *code.postprocess)
        if not any("_workspace_bytes =" in str(line) for line in source_lines):
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
        self._for_each_tensor_config(
            lambda config: config.save_original_tensor_to_cpu(self.api_config),
            *self._tensor_config_roots(),
        )

    def clear_original_cpu_inputs(self):
        self._for_each_tensor_config(
            lambda config: config.clear_original_cpu_tensor(),
            *self._tensor_config_roots(),
        )

    def estimate_input_bytes(self):
        """Estimate total configured input bytes for memory probe gating."""
        total_bytes = 0

        def visit(config):
            nonlocal total_bytes
            dtype = getattr(config, "dtype", "float32")
            total_bytes += int(config.numel()) * _dtype_element_size(dtype)

        self._for_each_tensor_config(visit, *self._tensor_config_roots())
        return total_bytes

    def clear_paddle_tensor(self):
        if not self._clear_tensor_config_cache("clear_paddle_tensor"):
            return
        if self.gpu_mode_config.enabled:
            gpu_mode_maybe_empty_cache(self.gpu_mode_config)
        else:
            paddle.device.cuda.empty_cache()

    def clear_torch_tensor(self, probe_bytes=None):
        if not self._clear_tensor_config_cache("clear_torch_tensor"):
            return
        if self.gpu_mode_config.enabled:
            gpu_mode_maybe_empty_cache(
                self.gpu_mode_config,
                probe_bytes=probe_bytes,
            )
        else:
            torch.cuda.empty_cache()

    def clear_numpy_tensor(self):
        if not self._clear_tensor_config_cache("clear_numpy_tensor"):
            return

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
