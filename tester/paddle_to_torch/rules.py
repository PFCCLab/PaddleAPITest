from __future__ import annotations

import os
import threading
import time
import types
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

IMPLEMENTATION_ENV_VAR = "PADDLEAPITEST_IMPL"
WORKERS_ON_GPU_ENV_VAR = "PADDLEAPITEST_WORKERS_ON_GPU"
# 这些值同时约束环境变量和 Rule 的 SUPPORTED_IMPLEMENTATIONS 声明。
VALID_IMPLEMENTATIONS = frozenset({"apex", "te", "torch"})
_WORKSPACE_PROBE_TTL = 0.25
_WORKSPACE_PROBE_CACHE: dict[tuple[str | None, int], tuple[float, int]] = {}
_WORKSPACE_PROBE_LOCK = threading.Lock()
_RULE_REGISTRY: dict[str, type] = {}
_RULE_REGISTRY_VIEW = MappingProxyType(_RULE_REGISTRY)
_RULE_REGISTRY_FROZEN = False


@dataclass(frozen=True)
class ConversionEnvironment:
    implementation: str | None


def read_conversion_environment() -> ConversionEnvironment:
    implementation = os.environ.get(IMPLEMENTATION_ENV_VAR)
    # 非法显式选择必须在转换入口失败，不能被 Rule 默认值掩盖。
    if implementation is not None and implementation not in VALID_IMPLEMENTATIONS:
        expected = ", ".join(sorted(VALID_IMPLEMENTATIONS))
        raise ValueError(
            f"{IMPLEMENTATION_ENV_VAR} must be one of {expected}, got {implementation!r}"
        )
    return ConversionEnvironment(implementation=implementation)


def read_workers_on_gpu() -> int:
    # 未配置时按单 worker 预算。
    raw_workers_on_gpu = os.environ.get(WORKERS_ON_GPU_ENV_VAR, "1")
    # workspace 按 worker 数切分，零值或非整数会破坏显存预算。
    try:
        workers_on_gpu = int(raw_workers_on_gpu)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{WORKERS_ON_GPU_ENV_VAR} must be a positive integer, got {raw_workers_on_gpu!r}"
        ) from exc
    if workers_on_gpu < 1:
        raise ValueError(
            f"{WORKERS_ON_GPU_ENV_VAR} must be a positive integer, got {raw_workers_on_gpu!r}"
        )
    return workers_on_gpu


def adaptive_workspace_bytes(
    torch_module, execution_values: Mapping[str, Any] | None = None
) -> int:
    """Return a cached, free-memory-aware workspace size for generated code."""
    workers_on_gpu = read_workers_on_gpu()

    def find_cuda_device(value, seen: set[int]):
        value_id = id(value)
        if value_id in seen:
            return None
        seen.add(value_id)
        if torch_module.is_tensor(value):
            return value.device if value.is_cuda else None
        if isinstance(value, Mapping):
            for nested in value.values():
                device = find_cuda_device(nested, seen)
                if device is not None:
                    return device
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                device = find_cuda_device(nested, seen)
                if device is not None:
                    return device
        return None

    device = find_cuda_device(execution_values, set()) if execution_values is not None else None
    if device is None:
        try:
            device = torch_module.cuda.current_device()
        except Exception:
            device = None
    cache_device = str(device) if device is not None else None
    cache_key = (cache_device, workers_on_gpu)
    now = time.monotonic()
    with _WORKSPACE_PROBE_LOCK:
        cached = _WORKSPACE_PROBE_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _WORKSPACE_PROBE_TTL:
            return cached[1]
    workspace = 1
    if device is not None:
        try:
            free_bytes, _ = torch_module.cuda.mem_get_info(device)
            workspace = max(1, min(32 << 30, int(free_bytes) // (5 * workers_on_gpu)))
        except Exception:
            pass
    with _WORKSPACE_PROBE_LOCK:
        cached = _WORKSPACE_PROBE_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _WORKSPACE_PROBE_TTL:
            return cached[1]
        _WORKSPACE_PROBE_CACHE[cache_key] = (now, workspace)
    return workspace


@dataclass(frozen=True)
class Code:
    """Immutable generated code and its compiled execution stages."""

    preprocess: Sequence[str] = field(default_factory=tuple)
    core: Sequence[str] = field(default_factory=tuple)
    postprocess: Sequence[str] = field(default_factory=tuple)
    workspace_required: bool = False
    preprocess_compiled: types.CodeType | None = field(init=False, default=None)
    core_compiled: types.CodeType | None = field(init=False, default=None)
    postprocess_compiled: types.CodeType | None = field(init=False, default=None)

    def __post_init__(self):
        object.__setattr__(self, "preprocess", tuple(self.preprocess))
        object.__setattr__(self, "core", tuple(self.core))
        object.__setattr__(self, "postprocess", tuple(self.postprocess))
        object.__setattr__(
            self,
            "preprocess_compiled",
            self._compile(self.preprocess, "preprocess"),
        )
        object.__setattr__(self, "core_compiled", self._compile(self.core, "core"))
        object.__setattr__(
            self,
            "postprocess_compiled",
            self._compile(self.postprocess, "postprocess"),
        )

    @staticmethod
    def _compile(code_lines: Sequence[str], stage: str) -> types.CodeType | None:
        if not code_lines:
            return None
        source = "\n".join(code_lines)
        return compile(source, f"<paddle_to_torch:{stage}>", "exec")


class ConversionKind(Enum):
    UNSUPPORTED = "unsupported"
    DIRECT = "direct"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class ConvertResult:
    """Immutable result of converting one Paddle API."""

    paddle_api: str
    kind: ConversionKind
    code: Code | None = None
    output_var: str | None = None
    error_message: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, ConversionKind):
            raise TypeError(
                f"Conversion kind must be ConversionKind, got {type(self.kind).__name__}"
            )
        if self.kind is ConversionKind.UNSUPPORTED:
            if self.code is not None:
                raise ValueError(
                    f"Unsupported conversion for {self.paddle_api} cannot contain code"
                )
            if not self.error_message:
                raise ValueError(
                    f"Unsupported conversion for {self.paddle_api} requires an error message"
                )
            return
        if self.code is None:
            raise ValueError(f"Supported conversion for {self.paddle_api} requires code")
        if not isinstance(self.code, Code):
            raise TypeError(
                f"Conversion code for {self.paddle_api} must be Code, "
                f"got {type(self.code).__name__}"
            )
        if self.error_message is not None:
            raise ValueError(
                f"Supported conversion for {self.paddle_api} cannot contain an error message"
            )

    @classmethod
    def success(
        cls,
        paddle_api: str,
        code: Code,
        output_var: str = "result",
        kind: ConversionKind = ConversionKind.DIRECT,
    ) -> ConvertResult:
        return cls(
            paddle_api,
            kind=kind,
            code=code,
            output_var=output_var,
        )

    @classmethod
    def error(cls, paddle_api: str, message: str) -> ConvertResult:
        return cls(
            paddle_api,
            kind=ConversionKind.UNSUPPORTED,
            error_message=message,
        )


class BaseRule(ABC):
    """转换规则的抽象基类"""

    PADDLE_APIS: tuple[str, ...] = ()
    SUPPORTED_IMPLEMENTATIONS: frozenset[str] = frozenset()
    DEFAULT_IMPLEMENTATION: str | None = None

    def __init__(self, conversion_environment: ConversionEnvironment):
        self._conversion_environment = conversion_environment

    def __init_subclass__(cls, *, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if not register:
            return
        if not cls.PADDLE_APIS:
            raise RuntimeError(f"{cls.__name__} must declare PADDLE_APIS")
        if not isinstance(cls.PADDLE_APIS, tuple):
            raise RuntimeError(f"{cls.__name__}.PADDLE_APIS must be a tuple")
        if len(set(cls.PADDLE_APIS)) != len(cls.PADDLE_APIS):
            raise RuntimeError(f"{cls.__name__}.PADDLE_APIS contains duplicates")
        for paddle_api in cls.PADDLE_APIS:
            if not isinstance(paddle_api, str) or not paddle_api.startswith("paddle."):
                raise RuntimeError(f"Invalid Paddle API {paddle_api!r} on {cls.__name__}")
            existing = _RULE_REGISTRY.get(paddle_api)
            if existing is not None and existing is not cls:
                raise RuntimeError(f"Duplicate Rule ownership for {paddle_api}")
        if _RULE_REGISTRY_FROZEN:
            raise RuntimeError(
                "The built-in Rule registry is frozen; pass additional Rules to the converter"
            )
        for paddle_api in cls.PADDLE_APIS:
            _RULE_REGISTRY[paddle_api] = cls

    def build_implementation_code(self) -> tuple[str, str]:
        """选择参考实现，并按 ``_<实现名>_code`` 约定生成代码。"""
        default = self.DEFAULT_IMPLEMENTATION
        if default is None:
            raise RuntimeError(f"{type(self).__name__} does not declare a default implementation")
        if default not in self.SUPPORTED_IMPLEMENTATIONS:
            raise ValueError(f"default implementation {default!r} is not supported")
        # 支持集必须与环境变量协议一致，避免新增实现被错误地静默回退。
        if not self.SUPPORTED_IMPLEMENTATIONS <= VALID_IMPLEMENTATIONS:
            invalid = ", ".join(sorted(self.SUPPORTED_IMPLEMENTATIONS - VALID_IMPLEMENTATIONS))
            raise ValueError(f"unknown supported implementations: {invalid}")
        requested = self._conversion_environment.implementation
        implementation = requested if requested in self.SUPPORTED_IMPLEMENTATIONS else default
        # 声明支持却缺少生成函数属于 Rule 协议错误，必须在转换阶段明确失败。
        builder = getattr(self, f"_{implementation}_code", None)
        if not callable(builder):
            raise RuntimeError(f"{type(self).__name__} does not implement _{implementation}_code()")
        return implementation, builder()

    @abstractmethod
    def apply(self, paddle_api: str) -> ConvertResult:
        """将 Paddle API 调用转换为 PyTorch 等效代码形式
        code 中可包含输入变量的占位符(如 {input}、{x}), 这些变量将被自动填充为 torch tensor

        Args:
            paddle_api (str): Paddle API 名称

        Returns:
            ConvertResult: 包含代码和输出变量的 ConvertResult 对象, 或错误信息
        """
        pass

    def read_mapping(self, mapping: Mapping[str, Any]) -> None:
        self.mapping = mapping
        self.torch_api = mapping.get("torch_api")
        self.is_attribute = mapping.get("is_attribute", False)

    def _build_default_code(self) -> list[str]:
        code = []
        for name, value in self.mapping.get("set_defaults", {}).items():
            expression = value if isinstance(value, str) else repr(value)
            code.append(f"{name} = locals().get({name!r}, {expression})")
        return code

    def _build_argument_map_code(self, *, ensure_args: bool) -> list[str]:
        map_code = []
        if ensure_args or "torch_args" in self.mapping:
            map_code.append("_args = []")
            for arg in self.mapping.get("torch_args", ()):
                map_code.append(f"_args.extend([{arg}])")
        if (
            ensure_args
            or "torch_args" in self.mapping
            or "torch_kwargs" in self.mapping
            or "paddle_torch_args_map" in self.mapping
        ):
            map_code.append("_kwargs = {}")
        for key, value in self.mapping.get("torch_kwargs", {}).items():
            expression = value if isinstance(value, str) else repr(value)
            map_code.append(f"_kwargs[{key!r}] = {expression}")
        args_map = self.mapping.get("paddle_torch_args_map", {})
        if args_map:
            map_code.append("for paddle_param, torch_param in {")
            for paddle_param, torch_param in args_map.items():
                map_code.append(f"    {paddle_param!r}: {torch_param!r},")
            map_code.append("}.items():")
            map_code.append("    if paddle_param in locals():")
            map_code.append("        _kwargs[torch_param] = locals()[paddle_param]")
        return map_code

    def build_result(
        self,
        paddle_api: str,
        *,
        kind: ConversionKind,
        preprocess: str | Sequence[str] = (),
        core: str | Sequence[str] = (),
        postprocess: str | Sequence[str] = (),
        output_var: str = "result",
        workspace_required: bool = False,
    ) -> ConvertResult:
        def code_lines(source: str | Sequence[str]) -> Sequence[str]:
            return source.splitlines() if isinstance(source, str) else source

        core_code = code_lines(core)
        generate_standard_call = not core_code
        if generate_standard_call:
            if not self.torch_api:
                raise ValueError(
                    f"Rule {type(self).__name__} for {paddle_api} requires core or torch_api"
                )
            if self.torch_api.startswith("torch.Tensor."):
                method_name = self.torch_api.removeprefix("torch.Tensor.")
                core_code = [f"result = x.{method_name}(*_args, **_kwargs)"]
            else:
                core_code = [f"result = {self.torch_api}(*_args, **_kwargs)"]

        return ConvertResult.success(
            paddle_api,
            Code(
                preprocess=[
                    # Keep Torch deterministic algorithms in lockstep with Paddle.
                    "import paddle",
                    "if paddle.get_flags('FLAGS_cudnn_deterministic')"
                    "['FLAGS_cudnn_deterministic']:",
                    "    torch.use_deterministic_algorithms(True)",
                    *self._build_default_code(),
                    *code_lines(preprocess),
                    *self._build_argument_map_code(ensure_args=generate_standard_call),
                ],
                core=core_code,
                postprocess=code_lines(postprocess),
                workspace_required=workspace_required,
            ),
            output_var=output_var,
            kind=kind,
        )


def get_rule_registry() -> Mapping[str, type[BaseRule]]:
    return _RULE_REGISTRY_VIEW


class GenericRule(BaseRule, register=False):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = []
        is_tensor_method = paddle_api.startswith("paddle.Tensor.")
        if is_tensor_method:
            if not self.torch_api.startswith("torch.Tensor."):
                return ConvertResult.error(
                    paddle_api,
                    "The torch api should start with 'torch.Tensor.' when direct mapping a paddle api that starts with 'paddle.Tensor.'",
                )
            pre.append("_tmp_tensor = x")
            pre.append("_args = list(positional_arguments[1:])")
            if self.is_attribute:
                core = [f"result = _tmp_tensor.{self.torch_api.split('.')[-1]}"]
                return self.build_result(
                    paddle_api,
                    kind=ConversionKind.DIRECT,
                    preprocess=pre,
                    core=core,
                )
        is_inplace = (
            paddle_api.endswith("_") and not paddle_api.endswith("__")
        ) or paddle_api == "paddle.Tensor.__setitem__"

        if not is_tensor_method and "torch_args" not in self.mapping:
            pre.append("_args = []")
        has_argument_mapping = any(
            field in self.mapping
            for field in ("torch_args", "torch_kwargs", "paddle_torch_args_map")
        )
        if not has_argument_mapping:
            pre.append("_kwargs = {}")

        post = []
        if is_tensor_method:
            torch_method = self.torch_api.replace("torch.Tensor.", "")
            if is_inplace:
                core = [f"_tmp_tensor.{torch_method}(*_args, **_kwargs)"]
                post = ["result = _tmp_tensor"]
            else:
                core = [f"result = _tmp_tensor.{torch_method}(*_args, **_kwargs)"]
        else:
            if is_inplace:
                core = [f"{self.torch_api}(*_args, **_kwargs)"]
                post = ["result = next(iter(bound_arguments.values()))"]
            else:
                core = [f"result = {self.torch_api}(*_args, **_kwargs)"]
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre,
            core=core,
            postprocess=post,
        )


# a
class AsComplexRule(BaseRule):
    PADDLE_APIS = ("paddle.as_complex",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dtype = x.dtype
if dtype == torch.bfloat16:
    x = x.to(torch.float32)
"""
        core = f"result = {self.torch_api}(input=x)"
        post = """
if dtype == torch.bfloat16:
    result = result.to(torch.bfloat16)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre,
            core=core,
            postprocess=post,
        )


class AddNRule(BaseRule):
    PADDLE_APIS = ("paddle.add_n",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
inputs = [inputs] if torch.is_tensor(inputs) else inputs
expanded_inputs = torch.broadcast_tensors(*inputs)
"""
        core = "result = torch.sum(torch.stack(expanded_inputs), dim=0)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre,
            core=core,
        )


class AdaptiveLogSoftmaxWithLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.adaptive_log_softmax_with_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
target_dim = label.dim()
is_batched = target_dim > 0
if not is_batched:
    input = input.unsqueeze(0)
    label = label.unsqueeze(0)

batch_size = label.shape[0]
output = input.new_zeros((batch_size,))
gather_inds = input.new_empty((batch_size,), dtype=torch.long)

cutoff_values = [0] + list(cutoffs)
used_rows = 0

for i in range(len(cutoff_values) - 1):
    low_idx = cutoff_values[i]
    high_idx = cutoff_values[i + 1]
    label_mask = (label >= low_idx) & (label < high_idx)  # shape: (B,)
    row_indices = label_mask.nonzero(as_tuple=False).squeeze()
    if row_indices.numel() == 0:
        continue
    if row_indices.dim() == 0:
        row_indices = row_indices.unsqueeze(0)

    if i == 0:
        gather_inds[row_indices] = label[label_mask]
    else:

        relative_label = label[label_mask] - low_idx
        input_subset = input[row_indices]


        cluster_hidden = torch.nn.functional.linear(
            input_subset, tail_weights[i-1][0].t()
        )
        cluster_output = torch.nn.functional.linear(
            cluster_hidden, tail_weights[i-1][1].t()
        )

        cluster_index = cutoffs[0] + i - 1
        gather_inds[row_indices] = cluster_index

        cluster_logprob = torch.log_softmax(cluster_output, dim=1)
        local_logprob = cluster_logprob.gather(1, relative_label.unsqueeze(1)).squeeze(1)
        output[row_indices] = local_logprob
    used_rows += row_indices.numel()

if used_rows != batch_size:
    raise ValueError(
        f"label values should be in [0, n_classes - 1], "
        f"but values in range [{label.min().item()}, {label.max().item()}] "
        "were found. "
    )

head_output = torch.nn.functional.linear(input, head_weight.t(), head_bias)
head_logprob = torch.log_softmax(head_output, dim=1)
output = output + head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze(1)
loss = (-output).mean()

if not is_batched:
    output = output.squeeze(0)

result = [output, loss]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class AllRule(BaseRule):
    PADDLE_APIS = ("paddle.all",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """

if (isinstance(axis, (list, tuple)) and len(axis) == 0):
    result = torch.tensor([True])
else:
    result = torch.all(input=x, dim=axis, keepdim=keepdim)
"""
        post = """
if (isinstance(axis, (list, tuple)) and len(axis) == 0) and keepdim:
    shape = []
    for i in range(x.dim()):
        shape.append(1)
    result = result.reshape(shape)
elif (isinstance(axis, (list, tuple)) and len(axis) == 0) and not keepdim:
    result = True
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
            postprocess=post.splitlines(),
        )


class AllcloseRule(BaseRule):
    PADDLE_APIS = ("paddle.allclose",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(x, tuple):
    x = x[0]
if isinstance(y, tuple):
    y = y[0]
rtol = max(0.0, rtol)
atol = max(0.0, atol)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
        )


class AdaptiveAvgPoolRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nn.functional.adaptive_avg_pool2d",
        "paddle.nn.functional.adaptive_avg_pool3d",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre_2d = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
"""
        pre_3d = """
if data_format == 'NDHWC':
    x = x.permute(0, 4, 1, 2, 3)
"""
        post_2d = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        post_3d = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        if self.torch_api.endswith("2d"):
            preprocess = pre_2d
            postprocess = post_2d
        elif self.torch_api.endswith("3d"):
            preprocess = pre_3d
            postprocess = post_3d
        else:
            return ConvertResult.error(paddle_api, "Unsupported adaptive_avg_pool API")
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class ArgmaxRule(BaseRule):
    PADDLE_APIS = (
        "paddle.argmax",
        "paddle.Tensor.argmax",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        post = "result = result.to(dtype=torch.int32 if dtype == torch.int32 else torch.int64)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            postprocess=post,
        )


class ArgminRule(BaseRule):
    PADDLE_APIS = ("paddle.argmin",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if keepdim == None:
    keepdim = False
if not isinstance(axis, int) and axis != None:
    axis = int(axis)
"""
        post = "result = result.to(dtype=dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            postprocess=[post],
        )


class AssignRule(BaseRule):
    PADDLE_APIS = ("paddle.assign",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def convert_seq2tensor_wrap_scalar(tlist):
    if isinstance(tlist, (list, tuple)):
        result = []
        appear_float = False
        appear_int = False
        appear_bool = False

        for x in tlist:
            mid_result = []
            if isinstance(x, (list, tuple)):
                for y in x:
                    if isinstance(y, (list, tuple)):
                        inner_result = []
                        for z in y:
                            if isinstance(z, (list, tuple)):
                                raise NotImplementedError("Nested list (depth > 3) is not supported")
                            else:
                                if isinstance(z, bool):
                                    appear_bool = True
                                elif isinstance(z, int):
                                    appear_int = True
                                elif isinstance(z, float):
                                    appear_float = True
                                inner_result.append(torch.tensor(z))
                        mid_result.append(torch.stack(inner_result))
                    else:
                        if isinstance(y, bool):
                            appear_bool = True
                        elif isinstance(y, int):
                            appear_int = True
                        elif isinstance(y, float):
                            appear_float = True
                        mid_result.append(torch.tensor(y))
                result.append(torch.stack(mid_result))
            else:
                if isinstance(x, bool):
                    appear_bool = True
                elif isinstance(x, int):
                    appear_int = True
                elif isinstance(x, float):
                    appear_float = True
                result.append(torch.tensor(x))
        result = torch.stack(result)
        if appear_float:
            result = result.to(torch.float64)
        elif appear_int:
            result = result.to(torch.int64)
        elif appear_bool:
            result = result.to(torch.bool)
        return result
    # handle scalar input: wrap by list
    elif isinstance(tlist, (int, float, bool)):
        py2torch_type_mapping = {float: torch.float64, int: torch.int64, bool: torch.bool}
        dtype = py2torch_type_mapping[type(tlist)]
        return torch.tensor([tlist], dtype=dtype)
    elif isinstance(tlist, torch.Tensor):
        return tlist
    else:
        return torch.tensor(tlist)

x = convert_seq2tensor_wrap_scalar(x)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
        )


# b
class BlhaGetMaxLenRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.blha_get_max_len",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
bsz = batch_size.shape[0]
if bsz == 0:
    result = (torch.zeros([1], dtype=seq_lens_encoder.dtype), torch.zeros([1], dtype=seq_lens_decoder.dtype))
else:
    result = (torch.max(seq_lens_encoder[:bsz]).unsqueeze(0), torch.max(seq_lens_decoder[:bsz]).unsqueeze(0))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class BinomialRule(BaseRule):
    PADDLE_APIS = ("paddle.binomial",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = "distribution = torch.distributions.binomial.Binomial(total_count=count, probs=prob)"
        core = "result = distribution.sample()"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class BmmRule(BaseRule):
    PADDLE_APIS = (
        "paddle.bmm",
        "paddle.Tensor.bmm",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if x.dtype != y.dtype:
    target = torch.promote_types(x.dtype, y.dtype)
    x, y = x.to(target), y.to(target)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
        )


class BroadcastShapeRule(BaseRule):
    PADDLE_APIS = ("paddle.broadcast_shape",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.broadcast_shapes(x_shape, y_shape)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class BroadcastTensorsRule(BaseRule):
    PADDLE_APIS = ("paddle.broadcast_tensors",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.broadcast_tensors(*input)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class BatchNormRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.batch_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if data_format == 'NHWC':
    x = x.permute(0, 3, 1, 2)
if running_mean is not None:
    running_mean.requires_grad = False
if running_var is not None:
    running_var.requires_grad = False
"""
        post = """
if data_format == 'NHWC':
    result = result.permute(0, 2, 3, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            postprocess=post.splitlines(),
        )


# c
class CastRule(BaseRule):
    PADDLE_APIS = (
        "paddle.cast",
        "paddle.Tensor.cast",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(dtype, str) and hasattr(torch, dtype):
    dtype = getattr(torch, dtype)
"""
        core = "result = x.to(dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess,
            core=core,
        )


class CorrcoefRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.corrcoef",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dtype = x.dtype
if dtype == torch.float16:
    x = x.to(torch.float)
"""
        core = """
if rowvar:
    result = torch.corrcoef(x)
else:
    x = x.t()
    result = torch.corrcoef(x).t()
"""
        postprocess = """
if dtype == torch.float16:
    result = result.to(torch.float16)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=postprocess.splitlines(),
        )


class CosineEmbeddingLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.cosine_embedding_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if input1.dim() == 1:
    input1 = input1.unsqueeze(1)
if input2.dim() == 1:
    input2 = input2.unsqueeze(1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
        )


class CrossEntropyRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.cross_entropy",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
shp = label.shape
if len(input.shape) > 2:
    perm = [0] + [len(input.shape)-1]+ [i for i in range(1,len(input.shape)-1)]
    input = input.permute(*perm)
label = label.squeeze(-1)
if weight is not None:
    weight.requires_grad = False
if label.dtype == torch.int32:
    label = label.long()
if soft_label and weight is not None and shp == input.shape:
    reduction_original = reduction
    weight_original = weight
    reduction = "none"
    weight = None
"""
        core = """
if not use_softmax and not soft_label and label_smoothing == 0.0:
    result = torch.nn.functional.nll_loss(
        torch.log(input),
        label,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )
else:
    result = torch.nn.functional.cross_entropy(
        input if use_softmax else torch.log(input),
        label,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
"""
        postprocess = """
if reduction_original is not None:
    reduction = reduction_original
    loss_weight = label@weight_original
    sum_weight = loss_weight.sum()
    result *= loss_weight
else:
    sum_weight = result.numel()

if reduction == "none":
    if soft_label:
        result = result.unsqueeze(-1)
    else:
        result = result.reshape(shp)
elif reduction == "sum":
    result = result.sum()
else:
    result = result.sum()/sum_weight
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
            postprocess=postprocess,
        )


class ChunkRule(BaseRule):
    PADDLE_APIS = ("paddle.chunk",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if not isinstance(axis, int) and axis != None:
    axis = int(axis)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
        )


class CovRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.cov",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if rowvar is False:
    if torch.is_tensor(x) and x.dim() > 1:
        x = torch.transpose(x, 0, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
        )


class CropRule(BaseRule):
    PADDLE_APIS = ("paddle.crop",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
ndim = x.dim()
if offsets is None:
    offsets = [0] * ndim
elif isinstance(offsets, (list, tuple)):
    new_offsets = []
    for o in offsets:
        if isinstance(o, torch.Tensor):
            new_offsets.append(o.item())
        else:
            new_offsets.append(int(o))
    offsets = new_offsets
elif isinstance(offsets, torch.Tensor):
    offsets = offsets.tolist()

if shape is None:
    new_shape = []
    for i in range(ndim):
        new_shape.append(x.size(i) - offsets[i])
    shape = new_shape
elif isinstance(shape, (list, tuple)):
    new_shape = []
    for s in shape:
        if isinstance(s, torch.Tensor):
            new_shape.append(s.item())
        else:
            new_shape.append(int(s))
    shape = new_shape
elif isinstance(shape, torch.Tensor):
    shape = shape.tolist()

new_shape = []
for i, s in enumerate(shape):
    if s == -1:
        new_shape.append(x.size(i) - offsets[i])
    else:
        new_shape.append(s)
shape = new_shape

slices = []
for i in range(ndim):
    slices.append(slice(offsets[i], offsets[i] + shape[i]))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess,
            core="result = x[tuple(slices)]",
        )


class CtcLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.ctc_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
        )


class CumRule(BaseRule):
    PADDLE_APIS = (
        "paddle.cummax",
        "paddle.cummin",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        torch_api = paddle_api.replace("paddle.", "torch.")
        pre = """
if axis is None:
    x = x.flatten()
    axis = 0
"""
        core = f"result = {torch_api}(input=x, dim=axis)"
        post = "result.values.to(dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post.splitlines(),
        )


class CumprodRule(BaseRule):
    PADDLE_APIS = ("paddle.cumprod",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if dtype is None:
    dtype = x.dtype
"""
        core = "result = torch.cumprod(input=x, dim=dim, dtype=dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class CumsumRule(BaseRule):
    PADDLE_APIS = (
        "paddle.cumsum",
        "paddle.Tensor.cumsum",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if axis == None:
    axis = 0
    x = x.flatten()
if not isinstance(axis, int) and axis != None:
    axis = int(axis)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
        )


class CumulativeTrapezoidRule(BaseRule):
    PADDLE_APIS = ("paddle.cumulative_trapezoid",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if dx is not None:
    if hasattr(dx, 'numel'):
        if dx.numel() == 0:
            dx = None
        elif dx.numel() == 1:
            dx = dx.item()
        else:
            dx = dx.flatten()[0].item()
"""
        core = f"""
if x is not None:
    result = {self.torch_api}(y, x, dim=axis)
elif dx is not None:
    result = {self.torch_api}(y, dx=dx, dim=axis)
else:
    result = torch.cumulative_trapezoid(y, dim=axis)
"""

        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class ClassCenterSampleRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.class_center_sample",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
unique_pos_classes = torch.unique(label)
num_pos_classes = unique_pos_classes.size(0)
if num_pos_classes >= num_samples:
    sampled_classes = unique_pos_classes
    remapped_label = torch.zeros_like(label)
    for new_idx, old_class in enumerate(sampled_classes):
        remapped_label[label == old_class] = new_idx
else:
    all_classes = torch.arange(num_classes, device=label.device)
    neg_classes = all_classes[~torch.isin(all_classes, unique_pos_classes)]
    num_neg_needed = num_samples - num_pos_classes
    if num_neg_needed > 0:
        if neg_classes.numel() >= num_neg_needed:
            selected_neg = neg_classes[torch.randperm(neg_classes.size(0))[:num_neg_needed]]
        else:
            selected_neg = neg_classes
        sampled_classes = torch.cat([unique_pos_classes, selected_neg])
    else:
        sampled_classes = unique_pos_classes
    remapped_label = torch.zeros_like(label)
    for new_idx, old_class in enumerate(sampled_classes):
        remapped_label[label == old_class] = new_idx
result = (remapped_label, sampled_classes)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class ClipRule(BaseRule):
    PADDLE_APIS = (
        "paddle.clip",
        "paddle.Tensor.clip",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(min, torch.Tensor):
    min = min.item()
if isinstance(max, torch.Tensor):
    max = max.item()
"""
        if paddle_api == "paddle.clip":
            core = """
if min is None and max is None:
    result = x
else:
    result = torch.clamp(input=x, min=min, max=max)
"""
        elif paddle_api == "paddle.Tensor.clip":
            core = """
if min is None and max is None:
    result = x
else:
    result = x.clamp(min=min, max=max)
"""
        else:
            return ConvertResult.error(paddle_api, f"Unsupported clip api: {paddle_api}")
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class Conv1dTransposeRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv1d_transpose",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = stride[0] if isinstance(stride, (list, tuple)) else stride
output_padding = output_padding[0] if isinstance(output_padding, (list, tuple)) else output_padding
dilation = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
output_size = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
if data_format == "NLC":
    x = x.transpose(1, 2)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        kernel_size = weight.size(-1)
        padding = (dilation * (kernel_size - 1)) // 2
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 1:
        padding = padding[0]
    elif len(padding) == 2:
        crop = padding
        padding = 0
    elif len(padding) == 3:
        crop = padding[1] if data_format == "NLC" else padding[2]
        padding = 0
if output_size is not None:
    L_in = x.size(-1)
    kernel_size = weight.size(-1)
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    output_padding = output_size - L_out
"""
        postprocess = """
if crop:
    result = result[:, :, crop[0]:result.size(-1) - crop[1]]
if data_format == "NLC":
    result = result.transpose(1, 2)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class Conv2dTransposeRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv2d_transpose",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = tuple(stride) if isinstance(stride, list) else stride
output_padding = tuple(output_padding) if isinstance(output_padding, list) else output_padding
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        padding = []
        for i in range(2):
            dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
            kernel_size = weight.size(2 + i)
            padding.append((dilation_i * (kernel_size - 1)) // 2)
        padding = tuple(padding)
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 2:
        padding = tuple(padding)
    elif len(padding) == 4:
        is_all_int = True
        for p in padding:
            if not isinstance(p, int):
                is_all_int = False
                break
        if is_all_int:
            crop = padding
        else:
            crop = []
            if data_format == "NHWC":
                for i in range(1, 3):
                    crop.extend(padding[i])
            else:
                for i in range(2, 4):
                    crop.extend(padding[i])
        padding = 0
if output_size is not None:
    output_padding = []
    for i in range(2):
        L_in = x.size(2 + i)
        kernel_size = weight.size(2 + i)
        stride_i = stride[i] if isinstance(stride, tuple) else stride
        padding_i = padding[i] if isinstance(padding, tuple) else padding
        dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
        L_out = (L_in - 1) * stride_i - 2 * padding_i + dilation_i * (kernel_size - 1) + 1
        output_padding.append(output_size[i] - L_out)
    output_padding = tuple(output_padding)
"""
        postprocess = """
if crop:
    result = result[:, :, crop[0]:result.size(-1) - crop[1], crop[2]:result.size(-2) - crop[3]]
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class Conv3dTransposeRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv3d_transpose",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = tuple(stride) if isinstance(stride, list) else stride
output_padding = tuple(output_padding) if isinstance(output_padding, list) else output_padding
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        padding = []
        for i in range(3):
            dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
            kernel_size = weight.size(2 + i)
            padding.append((dilation_i * (kernel_size - 1)) // 2)
        padding = tuple(padding)
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 3:
        padding = tuple(padding)
    elif len(padding) == 6:
        crop = padding
        padding = 0
    elif len(padding) == 5:
        crop = []
        if data_format == "NDHWC":
            for i in range(1, 4):
                crop.extend(padding[i])
        else:
            for i in range(2, 5):
                crop.extend(padding[i])
        padding = 0
if output_size is not None:
    output_padding = []
    for i in range(3):
        L_in = x.size(2 + i)
        kernel_size = weight.size(2 + i)
        stride_i = stride[i] if isinstance(stride, tuple) else stride
        padding_i = padding[i] if isinstance(padding, tuple) else padding
        dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
        L_out = (L_in - 1) * stride_i - 2 * padding_i + dilation_i * (kernel_size - 1) + 1
        output_padding.append(output_size[i] - L_out)
    output_padding = tuple(output_padding)
"""
        postprocess = """
if crop:
    result = result[:, :, crop[0]:result.size(-3) - crop[1], crop[2]:result.size(-2) - crop[3], crop[4]:result.size(-1) - crop[5]]
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class Conv1dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv1d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if data_format == "NLC":
    x = x.permute(0, 2, 1)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
    else:
        padding = tuple(padding)
"""
        postprocess = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class Conv2dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv2d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:  # [pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 4:
        is_all_int = True
        for p in padding:
            if not isinstance(p, int):
                is_all_int = False
                break
        if is_all_int: # [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        else: # Paddle 的 4D 填充格式(NCHW 或 NHWC)
            if data_format == "NCHW":
                pad_top, pad_bottom = padding[2][0], padding[2][1]
                pad_left, pad_right = padding[3][0], padding[3][1]
            else:  # NHWC
                pad_top, pad_bottom = padding[1][0], padding[1][1]
                pad_left, pad_right = padding[2][0], padding[2][1]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        padding = 0
"""
        postprocess = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class Conv3dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.conv3d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 3:  # [pad_depth, pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 6:  # [front, back, top, bottom, left, right]
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
    elif len(padding) == 5: # Paddle 的 5D 填充格式
        if data_format == "NCDHW":
            pad_front, pad_back = padding[2][0], padding[2][1]
            pad_top, pad_bottom = padding[3][0], padding[3][1]
            pad_left, pad_right = padding[4][0], padding[4][1]
        else: # NDHWC
            pad_front, pad_back = padding[1][0], padding[1][1]
            pad_top, pad_bottom = padding[2][0], padding[2][1]
            pad_left, pad_right = padding[3][0], padding[3][1]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
"""
        postprocess = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class CountNonzeroRule(BaseRule):
    PADDLE_APIS = ("paddle.count_nonzero",)

    def apply(self, paddle_api: str) -> ConvertResult:
        post = """
if keepdim:
    shape = list(x.shape)
    if axis is None:
        for i in range(len(shape)):
            shape[i] = 1
    else:
        if not isinstance(axis,(list,tuple)):
            axis = [axis]
        for i in range(len(axis)):
            shape[axis[i]] = 1
    result = result.reshape(shape)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            postprocess=post.splitlines(),
        )


class CopyRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.copy_",)

    def apply(self, paddle_api: str) -> ConvertResult:
        # paddle 的 copy_ 是双态算子：phi::Copy 先执行 dst->Resize(src.dims())，所以 shape
        # 相同时才是缓冲区级原地拷贝，shape 不同时 dst 整体接管 src 的形状；且 dtype 必须
        # 完全一致（Tensor::copy_ 的 PADDLE_ENFORCE_EQ）。torch.Tensor.copy_ 相反：dst 保留
        # 自己的 shape 并把 src 广播过来，dtype 自动 cast。两者仅在 shape+dtype 全同时等价，
        # 因此按 shape 分派，shape 不同时改用 set_ 才能同时换形状和内容。
        core = """
if x.dtype != other.dtype:
    raise RuntimeError(
        f"Tensor has different data type ({other.dtype} vs {x.dtype}), "
        f"Tensor Copy cannot be performed!"
    )
with torch.no_grad():
    if list(x.shape) == list(other.shape):
        x.copy_(other, non_blocking=not blocking)
    else:
        # paddle 的梯度只累到 dst，不建立回传 src 的边，因此 detach 后再 clone。
        x.set_(other.detach().clone().to(device=x.device))
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


# d
class DotRule(BaseRule):
    PADDLE_APIS = (
        "paddle.dot",
        "paddle.Tensor.dot",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x_dtype = x.dtype
if x.dtype in {torch.int32, torch.int64, torch.bool}:
    x = x.to(torch.float32)
    y = y.to(torch.float32)
"""
        if paddle_api == "paddle.dot":
            core = """
if x.ndim == 2:
    result = []
    for xi, yi in zip(x, y):
        _sum = 0
        for xi_j, yi_j in zip(xi, yi):
            _sum += xi_j * yi_j
        result.append(_sum)
    result = torch.tensor(result)
else:
    result = torch.dot(x, y)
"""
        elif paddle_api == "paddle.Tensor.dot":
            core = """
if x.ndim == 2:
    dot_results = []
    for i in range(x.shape[0]):
        dot_results.append(x[i].dot(y[i]))
    result = torch.stack(dot_results)
else:
    result = x.dot(y)
"""
        else:
            return ConvertResult.error(paddle_api, f"Unsupported dot API: {paddle_api}")
        post = "result = result.to(x_dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=[post],
        )


class DeformConv2dRule(BaseRule):
    PADDLE_APIS = ("paddle.vision.ops.deform_conv2d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import torchvision
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
        )


class DataFormatRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nn.functional.channel_shuffle",
        "paddle.nn.functional.pixel_shuffle",
        "paddle.nn.functional.pixel_unshuffle",
        "paddle.nn.functional.prelu",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if data_format == "NLC":
    x = x.transpose(1, 2)
elif data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
elif data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
"""
        postprocess = """
if data_format == "NLC":
    result = result.transpose(1, 2)
elif data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
elif data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            postprocess=postprocess,
        )


class DiceLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.dice_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
label = label.squeeze(-1)
label = torch.nn.functional.one_hot(label, input.size()[-1])
intersection = (input * label).sum(dim=1)
union = input.sum(dim=1) + label.sum(dim=1)

dice = (2 * intersection + epsilon) / (union + epsilon)

loss = 1 - dice  # shape: [N, C]
result = loss.mean()
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class DistributeFpnProposalsRule(BaseRule):
    PADDLE_APIS = ("paddle.vision.ops.distribute_fpn_proposals",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
import math
def BBoxArea(box, pixel_offset):
    w = box[2] - box[0]
    h = box[3] - box[1]
    if pixel_offset:
        return (w+1) * (h+1)
    else:
        return w * h

pixel_offset = pixel_offset
rois_num = rois_num
num_level = max_level - min_level + 1
if rois_num is not None:
    for i in range(1, rois_num.numel()):
        rois_num[i] += rois_num[i-1]
    fpn_rois_lod = torch.concat([torch.tensor([0]), rois_num])
else:
    fpn_rois_lod = torch.tensor([0, fpn_rois.shape[0]])

size = fpn_rois_lod.numel() - 1
fpn_rois_num = (int)(fpn_rois_lod[size])
# 计算roi所属的level
num_rois_level = torch.zeros([num_level])
target_level = []
for i in range(fpn_rois_lod.numel() - 1):
    fpn_rois_slice = fpn_rois[fpn_rois_lod[i]:fpn_rois_lod[i+1]]
    for rois_data in fpn_rois_slice:
        roi_scale = math.sqrt(BBoxArea(rois_data, pixel_offset))
        tgt_lvl = math.floor(math.log2(roi_scale / refer_scale) + refer_level)
        tgt_lvl = min(max_level, max(tgt_lvl, min_level))
        target_level.append(tgt_lvl)
        num_rois_level[tgt_lvl - min_level] += 1
# 初始化结果
multi_rois = []
for i in range(num_level):
    multi_rois.append([])
restore_ind = torch.empty(fpn_rois.shape[0], 1)
rois_num_per_level = []
for i in range(num_level):
    rois_num_per_level.append(
        torch.zeros([rois_num.numel()]).to(torch.int32)
    )
# 计算结果
index = 0
for i in range(fpn_rois_lod.numel() - 1):
    fpn_rois_slice = fpn_rois[fpn_rois_lod[i]:fpn_rois_lod[i+1]]
    for rois_data in fpn_rois_slice:
        level = target_level[index]
        if multi_rois[level-min_level] == []:
            multi_rois[level-min_level].append(rois_data)
        else:
            multi_rois[level-min_level].append(rois_data)
        rois_num_per_level[level - min_level][i] += 1
        index += 1
for i in range(num_level):
    if multi_rois[i] == []:
        multi_rois[i] = torch.zeros([0,4])
    else:
        multi_rois[i] = torch.stack(multi_rois[i])
index = 0
for i in range(num_level):
    for j in range(fpn_rois.shape[0]):
        if target_level[j] == i + min_level:
            restore_ind[j] = index
            index += 1

result = (multi_rois, restore_ind.to(torch.int32), rois_num_per_level)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class DiagRule(BaseRule):
    PADDLE_APIS = (
        "paddle.diag",
        "paddle.Tensor.diag",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        if paddle_api == "paddle.diag":
            core = "result = torch.diag(x, diagonal=offset)"
        else:
            core = "result = x.diag(diagonal=offset)"
        post = """
if x.ndim == 1 and padding_value != 0:
    padding_value = torch.tensor(padding_value, dtype=torch.float32)
    diag_mask = torch.diag(torch.ones_like(x), diagonal=offset)
    result = torch.where(diag_mask.bool(), result, padding_value)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            core=core,
            postprocess=post.splitlines(),
        )


class DropoutRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.dropout",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
def axis_dropout(x, p, axis, training=True, mode='upscale_in_train'):
    if isinstance(axis, int):
        axis = [axis]
    mask_shape = []
    for i in range(x.dim()):
        if i in axis:
            mask_shape.append(x.shape[i])
        else:
            mask_shape.append(1)
    mask = torch.bernoulli(torch.full(mask_shape, 1-p)).to(x.device)
    if mode == 'upscale_in_train':
        if training:
            return x * mask / (1.0 - p)
        else:
            return x
    elif mode == 'downscale_in_infer':
        if training:
            return x * mask
        else:
            return x * (1.0 - p)
    else:
        raise ValueError(f"Invalid mode: {mode}")

x = x
p = p
axis = axis
training = training
mode = mode
"""
        core = "result = axis_dropout(x, p, axis, training, mode) if axis is not None else torch.nn.functional.dropout(input=x, p=float(p), training=training)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess,
            core=core,
        )


class Dropout2dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.dropout2d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
p = p
training = training
data_format = data_format

if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
"""
        core = "result = torch.nn.functional.dropout2d(input=x, p=float(p), training=training)"
        postprocess = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
            core=core,
            postprocess=postprocess,
        )


class Dropout3dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.dropout3d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
p = p
training = training
data_format = data_format

if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
"""
        core = "result = torch.nn.functional.dropout3d(input=x, p=float(p), training=training)"
        postprocess = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
            core=core,
            postprocess=postprocess,
        )


# e
class EinsumRule(BaseRule):
    PADDLE_APIS = ("paddle.einsum",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
tensor_args = list(operands) if isinstance(operands, (list, tuple)) else [operands]
output_pattern = equation.split('->')[-1] if '->' in equation else ""
has_repeats = len(output_pattern.replace('...', '')) != len(set(output_pattern.replace('...', '')))
is_ii_pattern = output_pattern == 'ii'
special_handling = has_repeats and is_ii_pattern
if special_handling:
    interim_eq = equation.split('->')[0] + '->i'
    interim_result = None
"""
        core = f"""
if special_handling:
    interim_result = {self.torch_api}(interim_eq, *tensor_args)
    n = interim_result.shape[0]
    result = torch.zeros((n, n), dtype=interim_result.dtype, device=interim_result.device)
    result.diagonal().copy_(interim_result)
else:
    result = {self.torch_api}(equation, *tensor_args)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class EmbeddingRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.embedding",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
if torch.is_complex(weight):
    result = torch.nn.functional.embedding(x, weight.real, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    result = result + 1j * torch.nn.functional.embedding(x, weight.imag, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
else:
    result = torch.nn.functional.embedding(x, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class EmptyRule(BaseRule):
    PADDLE_APIS = ("paddle.empty",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(shape, torch.Tensor):
    size_list = shape.tolist()
elif isinstance(shape, (list, tuple)):
    size_list = []
    for s in shape:
        if isinstance(s, torch.Tensor):
            if s.numel() == 1:
                size_list.append(s.item())
            else:
                size_list.append(s.flatten()[0].item())
        else:
            size_list.append(s)
"""
        core = """
if dtype is not None:
    result = torch.empty(size_list, dtype=dtype)
else:
    result = torch.empty(size_list)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class ExpandRule(BaseRule):
    PADDLE_APIS = (
        "paddle.expand",
        "paddle.Tensor.expand",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
if (isinstance(shape, list) or isinstance(shape, tuple)) and len(shape) == 0:
    result = x
else:
    result = x.expand(*shape)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=impl.splitlines(),
        )


class ExpandAsRule(BaseRule):
    PADDLE_APIS = ("paddle.expand_as",)

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.expand_as(y)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=impl.splitlines(),
        )


class EyeRule(BaseRule):
    PADDLE_APIS = ("paddle.eye",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if not isinstance(num_rows, int) and num_rows != None:
    num_rows = int(num_rows)
if not isinstance(num_columns, int) and num_columns != None:
    num_columns = int(num_columns)
"""
        core = """
if num_columns is None:
    result = torch.eye(n=num_rows, dtype=dtype)
else:
    result = torch.eye(n=num_rows, m=num_columns, dtype=dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


# f
class FillDiagonalTensorRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.fill_diagonal_tensor",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
diag = torch.diagonal(x, offset=offset, dim1=dim1, dim2=dim2)
result = x.clone()
"""
        core = """
result = torch.diagonal_scatter(result, y, offset=offset, dim1=dim1, dim2=dim2)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class FillDiagonalRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.fill_diagonal_",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def fill_diagonal(x, value, offset=0, wrap=True):
    if offset == 0:
        x.fill_diagonal_(value, wrap)
    else:
        # roll along dim 1, but do not re-introduce the roll-out elements
        # fidx is the flattened index of the value to be filled in the diagonal on row i
        fidx = 0
        for i in range(x.shape[0]):
            if wrap:
                if fidx < (i + 1) * x.shape[1]:
                    vidx = fidx % x.shape[1]
                    fidx += x.shape[1] + 1
                else:
                    # this row has no elements to fill
                    vidx = -1
            else:
                if i < x.shape[1]:
                    vidx = i
                else:
                    vidx = -1
            if vidx != -1 and vidx + offset >= 0 and vidx + offset < x.shape[1]:
                x[i, vidx + offset] = value
    return x
"""
        core = """
result = fill_diagonal(x, value, offset, wrap)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class FracRule(BaseRule):
    PADDLE_APIS = ("paddle.frac",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(x, torch.Tensor):
    src_dtype = x.dtype
    if x.dtype not in [torch.float16, torch.float32, torch.float64]:
        x = x.to(torch.float64)
else:
    raise ValueError(f"x must be a tensor, but got {type(x)}")
"""
        core = "result = torch.frac(input=x)"
        post_process = """
if src_dtype != result.dtype:
    result = result.to(src_dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post_process.splitlines(),
        )


class FractionalMaxPoolRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nn.functional.fractional_max_pool2d",
        "paddle.nn.functional.fractional_max_pool3d",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre1 = """
batch_size, C = x.shape[0], x.shape[1]
if random_u is not None:
    random_u = torch.tensor([[[random_u] * 2] * C] * batch_size, dtype=x.dtype, device=x.device)

def compute_kernel_size(x, output_size):
    H_in, W_in = x.shape[2], x.shape[3]
    if isinstance(output_size, int):
        H_out = W_out = output_size
    else:
        H_out, W_out = output_size

    def compute_k(input_size, output_size):
        if output_size is None or output_size == input_size:
            return 1  # No pooling
        else:
            return (input_size + output_size - 1) // output_size  # ceil(input_size / output_size)

    kH = compute_k(H_in, H_out)
    kW = compute_k(W_in, W_out)
    return (kH, kW)
"""
        pre2 = """
batch_size, C = x.shape[0], x.shape[1]
if random_u is not None:
    random_u = torch.tensor([[[random_u] * 3] * C] * batch_size, dtype=x.dtype, device=x.device)

def compute_kernel_size(x, output_size):
    D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
    if isinstance(output_size, int):
        D_out = H_out = W_out = output_size
    else:
        D_out, H_out, W_out = output_size

    def compute_k(input_size, output_size):
        if output_size is None or output_size == input_size:
            return 1  # No pooling
        else:
            return (input_size + output_size - 1) // output_size  # ceil(input_size / output_size)

    kD = compute_k(D_in, D_out)
    kH = compute_k(H_in, H_out)
    kW = compute_k(W_in, W_out)
    return (kD, kH, kW)
"""
        pre3 = """
kernel_size = kernel_size
if kernel_size is None:
    kernel_size = compute_kernel_size(x, output_size)
elif isinstance(kernel_size, list):
    kernel_size = tuple(kernel_size)
if isinstance(output_size, (list, tuple)):
    new_output_size = []
    for i, size in enumerate(output_size):
        if size is None:
            new_output_size.append(x.shape[i + 2])
        else:
            new_output_size.append(size)
    output_size = tuple(new_output_size)
"""
        if paddle_api == "paddle.nn.functional.fractional_max_pool2d":
            pre = pre1
        else:
            pre = pre2
        pre += pre3
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
        )


class Fp8QuantBlockwiseRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fp8_quant_blockwise",)

    """Torch reference for paddle.incubate.nn.functional.fp8_quant_blockwise.

    Aligns with phi fp8_quant_blockwise kernel:
    quant_scale = fp8_max / amax (optionally power-of-2), stored scale = 1/quant_scale,
    quantized = cast(x * quant_scale, float8_e4m3fn).

    Reference implementation is selected via PADDLEAPITEST_IMPL env var:
      - "torch" (default): Shape-generic manual Torch implementation
      - "te":              Transformer Engine Float8BlockQuantizer
    """

    SUPPORTED_IMPLEMENTATIONS = frozenset({"te", "torch"})
    DEFAULT_IMPLEMENTATION = "torch"

    def apply(self, paddle_api: str) -> ConvertResult:
        # TE's blockwise transpose kernel rejects valid very-large Paddle
        # inputs with CUDA_INVALID_ARGUMENT. Use the shape-generic Torch
        # reference by default; TE remains available for explicit testing.
        impl, core = self.build_implementation_code()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=(),
            core=core,
            workspace_required=impl == "torch",
        )

    @staticmethod
    def _te_code() -> str:
        return """
import os as _os
_fp8_max = 448.0
_eps = float(epsilon) if epsilon is not None else 0.0
_input_transpose = bool(input_transpose)
_output_scale_transpose = bool(output_scale_transpose)
_return_transpose_only = bool(return_transpose_only)
_using_pow2_scale = bool(using_pow2_scale)
_using_ue8m0_scale = bool(using_ue8m0_scale)
_quant_method = quant_method
if _quant_method not in ("1x128", "128x128"):
    raise ValueError(f"Unsupported quantization method: {_quant_method}")
if output_type != "e4m3":
    raise ValueError(f"Unsupported output type: {output_type}")

try:
    import transformer_engine.pytorch as _te
    from transformer_engine.pytorch.quantization import DType as _teDType
except Exception as _err:
    raise RuntimeError(
        "PADDLEAPITEST_IMPL=te: cannot import transformer_engine; "
        f"error={type(_err).__name__}: {_err}"
    ) from _err

def _te_fp8_quant_blockwise(inp, eps, power2, scale_transpose, ue8m0, method):
    import transformer_engine.pytorch as _te
    from transformer_engine.pytorch.quantization import DType as _teDType
    m, n = inp.shape
    if inp.numel() == 0:
        q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=inp.device)
        if method == "1x128":
            scale_cols = (n + 127) // 128
            if ue8m0:
                packed_cols = (scale_cols + 3) // 4
                scale_shape = (packed_cols, m) if scale_transpose else (m, packed_cols)
                scale = torch.empty(scale_shape, dtype=torch.int32, device=inp.device)
            else:
                scale_shape = (scale_cols, m) if scale_transpose else (m, scale_cols)
                scale = torch.empty(scale_shape, dtype=torch.float32, device=inp.device)
        else:
            sm, sn = (m + 127) // 128, (n + 127) // 128
            if ue8m0:
                packed_sn = (sn + 3) // 4
                scale_shape = (packed_sn, sm) if scale_transpose else (sm, packed_sn)
                scale = torch.empty(scale_shape, dtype=torch.int32, device=inp.device)
            else:
                scale_shape = (sn, sm) if scale_transpose else (sm, sn)
                scale = torch.empty(scale_shape, dtype=torch.float32, device=inp.device)
        return q, scale
    _block_dim = 1 if method == "1x128" else 2
    _quantizer = _te.Float8BlockQuantizer(
        fp8_dtype=_teDType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=power2,
        amax_epsilon=eps,
        block_scaling_dim=_block_dim,
    )
    _result = _quantizer.quantize(inp.to(torch.bfloat16) if inp.dtype not in (torch.bfloat16, torch.float16, torch.float32) else inp)
    q = _result._rowwise_data.view(torch.float8_e4m3fn)
    deq_scale = _result._rowwise_scale_inv  # float32

    # Handle scale transpose:
    # TE 1D: already (scale_cols, m) = transposed format
    # TE 2D: (sm, sn) = non-transposed format
    if method == "1x128":
        # TE gives (scale_cols, m) which IS the transposed layout
        scale = deq_scale if scale_transpose else deq_scale.t().contiguous()
    else:
        # TE gives (sm, sn) which is NON-transposed layout
        scale = deq_scale.t().contiguous() if scale_transpose else deq_scale

    # Handle ue8m0 packing: 4 float32 exponents -> 1 packed int32
    if ue8m0:
        base = deq_scale.contiguous() if method == "1x128" else (deq_scale.t().contiguous() if scale_transpose else deq_scale.contiguous())
        # base is in the final layout orientation already; pack along last dim
        # Actually re-derive from the non-transposed deq_scale for consistent packing
        if method == "1x128":
            _base = deq_scale.t().contiguous()  # (m, scale_cols)
        else:
            _base = deq_scale.contiguous()  # (sm, sn)
        cols_s = _base.shape[-1]
        pad_c = (4 - (cols_s % 4)) % 4
        if pad_c:
            _base = torch.nn.functional.pad(_base, (0, pad_c))
        b = _base.reshape(*_base.shape[:-1], -1, 4)
        safe = torch.clamp(b, min=torch.finfo(torch.float32).tiny)
        exp = torch.floor(torch.log2(safe)).to(torch.int32) + 127
        exp = torch.clamp(exp, 0, 255)
        packed = (exp[..., 0] | (exp[..., 1] << 8) | (exp[..., 2] << 16) | (exp[..., 3] << 24)).to(torch.int32)
        scale = packed.transpose(0, 1).contiguous() if scale_transpose else packed.contiguous()
    return q, scale

if not _input_transpose:
    result = _te_fp8_quant_blockwise(x, _eps, _using_pow2_scale, _output_scale_transpose, _using_ue8m0_scale, _quant_method)
else:
    x_t = x.transpose(0, 1).contiguous()
    q_t, s_t = _te_fp8_quant_blockwise(x_t, _eps, _using_pow2_scale, _output_scale_transpose, _using_ue8m0_scale, _quant_method)
    if _return_transpose_only:
        result = (q_t, s_t)
    else:
        q, s = _te_fp8_quant_blockwise(x, _eps, _using_pow2_scale, _output_scale_transpose, _using_ue8m0_scale, _quant_method)
        result = (q, s, q_t, s_t)
"""

    @staticmethod
    def _torch_code() -> str:
        return """
_fp8_max = 448.0
_eps = float(epsilon) if epsilon is not None else 0.0
_input_transpose = bool(input_transpose)
_output_scale_transpose = bool(output_scale_transpose)
_return_transpose_only = bool(return_transpose_only)
_using_pow2_scale = bool(using_pow2_scale)
_using_ue8m0_scale = bool(using_ue8m0_scale)
_quant_method = quant_method
if _quant_method not in ("1x128", "128x128"):
    raise ValueError(f"Unsupported quantization method: {_quant_method}")
if output_type != "e4m3":
    raise ValueError(f"Unsupported output type: {output_type}")

def _fp8_quant_blockwise_impl(inp, eps, power2, scale_transpose, ue8m0, method, fp8_max):
    m, n = inp.shape
    if inp.numel() == 0:
        q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=inp.device)
        if method == "1x128":
            scale_cols = (n + 127) // 128
        else:
            sm, sn = (m + 127) // 128, (n + 127) // 128
            scale_cols = sn if scale_transpose else sm
            # for 128x128: scale shape before transpose is (sm, sn)
            scale_rows_128 = sm
        if ue8m0:
            if method == "1x128":
                packed_cols = (scale_cols + 3) // 4
                scale = torch.empty((packed_cols, m) if scale_transpose else (m, packed_cols), dtype=torch.int32, device=inp.device)
            else:
                packed_sm = (sm + 3) // 4 if not scale_transpose else sm
                packed_sn = (sn + 3) // 4 if scale_transpose else sn
                if scale_transpose:
                    scale = torch.empty((packed_sn, sm), dtype=torch.int32, device=inp.device)
                else:
                    scale = torch.empty((sm, packed_sn), dtype=torch.int32, device=inp.device)
        else:
            if method == "1x128":
                scale = torch.empty((scale_cols, m) if scale_transpose else (m, scale_cols), dtype=torch.float32, device=inp.device)
            else:
                scale = torch.empty((sn, sm) if scale_transpose else (sm, sn), dtype=torch.float32, device=inp.device)
        return q, scale

    _need_chunk = (m * n * 4) > (2 * 1024 * 1024 * 1024)

    if method == "1x128":
        scale_cols = (n + 127) // 128
        if not _need_chunk:
            x_f = inp.to(torch.float32)
            pad_n = (128 - (n % 128)) % 128
            if pad_n:
                x_f = torch.nn.functional.pad(x_f, (0, pad_n))
            x_blk = x_f.view(m, -1, 128)
            amax = x_blk.abs().amax(dim=-1).to(torch.float32)
            amax_mod = torch.clamp(amax, min=eps)
            zero_mask = amax_mod == 0
            q_scale = torch.where(zero_mask, torch.ones_like(amax_mod), fp8_max / amax_mod)
            inf_mask = torch.isinf(q_scale)
            if inf_mask.any():
                q_scale = torch.where(inf_mask, torch.full_like(q_scale, torch.finfo(torch.float32).max), q_scale)
            if power2:
                safe = torch.clamp(q_scale, min=torch.finfo(torch.float32).tiny)
                exp = torch.floor(torch.log2(safe))
                q_scale = torch.pow(torch.tensor(2.0, device=q_scale.device, dtype=q_scale.dtype), exp)
                q_scale = torch.where(zero_mask, torch.ones_like(q_scale), q_scale)
            deq_scale = torch.where(q_scale == 0, torch.zeros_like(q_scale), 1.0 / q_scale)
            q = (x_blk * q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(m, -1)[:, :n]
            scale = deq_scale.transpose(0, 1).contiguous() if scale_transpose else deq_scale.contiguous()
        else:
            _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
            _per_row = max(1, n * 4 * 2)
            _chunk = max(1, min(m, _workspace_bytes // _per_row))
            q = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=inp.device)
            deq_scale = torch.empty((m, scale_cols), dtype=torch.float32, device=inp.device)
            for _rs in range(0, m, _chunk):
                _re = min(m, _rs + _chunk)
                _x_f = inp[_rs:_re].to(torch.float32)
                pad_n = (128 - (n % 128)) % 128
                if pad_n:
                    _x_f = torch.nn.functional.pad(_x_f, (0, pad_n))
                _x_blk = _x_f.view(_re - _rs, -1, 128)
                del _x_f
                _amax = _x_blk.abs().amax(dim=-1).to(torch.float32)
                _amax_mod = torch.clamp(_amax, min=eps)
                _zero_mask = _amax_mod == 0
                _q_scale = torch.where(_zero_mask, torch.ones_like(_amax_mod), fp8_max / _amax_mod)
                _inf_mask = torch.isinf(_q_scale)
                if _inf_mask.any():
                    _q_scale = torch.where(_inf_mask, torch.full_like(_q_scale, torch.finfo(torch.float32).max), _q_scale)
                if power2:
                    _safe = torch.clamp(_q_scale, min=torch.finfo(torch.float32).tiny)
                    _exp = torch.floor(torch.log2(_safe))
                    _q_scale = torch.pow(torch.tensor(2.0, device=_q_scale.device, dtype=_q_scale.dtype), _exp)
                    _q_scale = torch.where(_zero_mask, torch.ones_like(_q_scale), _q_scale)
                _deq_scale = torch.where(_q_scale == 0, torch.zeros_like(_q_scale), 1.0 / _q_scale)
                _q_chunk = (_x_blk * _q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(_re - _rs, -1)[:, :n]
                del _x_blk
                q[_rs:_re] = _q_chunk
                deq_scale[_rs:_re] = _deq_scale
                del _q_chunk, _deq_scale
            scale = deq_scale.transpose(0, 1).contiguous() if scale_transpose else deq_scale.contiguous()
    else:
        pad_m = (128 - (m % 128)) % 128
        pad_n = (128 - (n % 128)) % 128
        if not _need_chunk:
            x_f = inp.to(torch.float32)
            if pad_m or pad_n:
                x_f = torch.nn.functional.pad(x_f, (0, pad_n, 0, pad_m))
            pm, pn = x_f.shape
            x_blk = x_f.view(pm // 128, 128, pn // 128, 128).permute(0, 2, 1, 3).contiguous()
            amax = x_blk.abs().amax(dim=(-1, -2)).to(torch.float32)
            amax_mod = torch.clamp(amax, min=eps)
            zero_mask = amax_mod == 0
            q_scale = torch.where(zero_mask, torch.ones_like(amax_mod), fp8_max / amax_mod)
            inf_mask = torch.isinf(q_scale)
            if inf_mask.any():
                q_scale = torch.where(inf_mask, torch.full_like(q_scale, torch.finfo(torch.float32).max), q_scale)
            if power2:
                safe = torch.clamp(q_scale, min=torch.finfo(torch.float32).tiny)
                exp = torch.floor(torch.log2(safe))
                q_scale = torch.pow(torch.tensor(2.0, device=q_scale.device, dtype=q_scale.dtype), exp)
                q_scale = torch.where(zero_mask, torch.ones_like(q_scale), q_scale)
            deq_scale = torch.where(q_scale == 0, torch.zeros_like(q_scale), 1.0 / q_scale)
            q = (x_blk * q_scale.unsqueeze(-1).unsqueeze(-1)).to(torch.float8_e4m3fn)
            q = q.permute(0, 2, 1, 3).contiguous().view(x_f.shape[0], x_f.shape[1])[:m, :n]
            scale = deq_scale.transpose(0, 1).contiguous() if scale_transpose else deq_scale.contiguous()
        else:
            sm = (m + 127) // 128
            sn = (n + 127) // 128
            _padded_n = n + pad_n
            _q_chunks = []
            _deq_scale = torch.empty((sm, sn), dtype=torch.float32, device=inp.device)
            for _rb in range(sm):
                _r_start = _rb * 128
                _r_end = min(m, _r_start + 128)
                _actual_rows = _r_end - _r_start
                _x_f = inp[_r_start:_r_end].to(torch.float32)
                if _actual_rows < 128:
                    _x_f = torch.nn.functional.pad(_x_f, (0, 0, 0, 128 - _actual_rows))
                if pad_n:
                    _x_f = torch.nn.functional.pad(_x_f, (0, pad_n))
                _x_blk = _x_f.view(1, 128, sn, 128).permute(0, 2, 1, 3).reshape(sn, 128, 128)
                _amax = _x_blk.abs().amax(dim=(-1, -2)).to(torch.float32)
                _amax_mod = torch.clamp(_amax, min=eps)
                _zero_mask = _amax_mod == 0
                _q_scale = torch.where(_zero_mask, torch.ones_like(_amax_mod), fp8_max / _amax_mod)
                _inf_mask = torch.isinf(_q_scale)
                if _inf_mask.any():
                    _q_scale = torch.where(_inf_mask, torch.full_like(_q_scale, torch.finfo(torch.float32).max), _q_scale)
                if power2:
                    _safe = torch.clamp(_q_scale, min=torch.finfo(torch.float32).tiny)
                    _exp = torch.floor(torch.log2(_safe))
                    _q_scale = torch.pow(torch.tensor(2.0, device=_q_scale.device, dtype=_q_scale.dtype), _exp)
                    _q_scale = torch.where(_zero_mask, torch.ones_like(_q_scale), _q_scale)
                _deq_row = torch.where(_q_scale == 0, torch.zeros_like(_q_scale), 1.0 / _q_scale)
                _deq_scale[_rb] = _deq_row
                _q_blk = (_x_blk * _q_scale.unsqueeze(-1).unsqueeze(-1)).to(torch.float8_e4m3fn)
                _q_row = _q_blk.view(1, sn, 128, 128).permute(0, 2, 1, 3).reshape(128, _padded_n)[:_actual_rows, :n]
                del _x_f, _x_blk, _q_blk
                _q_chunks.append(_q_row)
            q = torch.cat(_q_chunks, dim=0)
            del _q_chunks
            deq_scale = _deq_scale
            scale = deq_scale.transpose(0, 1).contiguous() if scale_transpose else deq_scale.contiguous()

    if ue8m0:
        base = deq_scale.contiguous()
        cols_s = base.shape[-1]
        pad_c = (4 - (cols_s % 4)) % 4
        if pad_c:
            base = torch.nn.functional.pad(base, (0, pad_c))
        b = base.reshape(*base.shape[:-1], -1, 4)
        safe = torch.clamp(b, min=torch.finfo(torch.float32).tiny)
        exp = torch.floor(torch.log2(safe)).to(torch.int32) + 127
        exp = torch.clamp(exp, 0, 255)
        packed = (exp[..., 0] | (exp[..., 1] << 8) | (exp[..., 2] << 16) | (exp[..., 3] << 24)).to(torch.int32)
        scale = packed.transpose(0, 1).contiguous() if scale_transpose else packed.contiguous()
    return q, scale

if not _input_transpose:
    result = _fp8_quant_blockwise_impl(x, _eps, _using_pow2_scale, _output_scale_transpose, _using_ue8m0_scale, _quant_method, _fp8_max)
else:
    x_t = x.transpose(0, 1).contiguous()
    q_t, s_t = _fp8_quant_blockwise_impl(x_t, _eps, _using_pow2_scale, _output_scale_transpose, _using_ue8m0_scale, _quant_method, _fp8_max)
    if _return_transpose_only:
        result = (q_t, s_t)
    else:
        q, s = _fp8_quant_blockwise_impl(x, _eps, _using_pow2_scale, _output_scale_transpose, _using_ue8m0_scale, _quant_method, _fp8_max)
        result = (q, s, q_t, s_t)
"""


class FusedActDequantRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_act_dequant",)

    """Torch reference for paddle.incubate.nn.functional.fused_act_dequant.

    Dequantizes float8_e4m3fn activations with per-128-column scales:
    out = (x.float() * broadcast(scale)) -> bfloat16.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
_x = x
_scale = x_scale
_rows, _cols = _x.shape
_tile_size = 128
_num_scale_blocks = (_cols + _tile_size - 1) // _tile_size
_full_blocks = _cols // _tile_size
_full_cols = _full_blocks * _tile_size
_result = torch.empty((_rows, _cols), dtype=torch.bfloat16, device=_x.device)

if _scale is None:
    _result.copy_(_x)
else:
    if _scale.dim() == 1:
        _scale = _scale.unsqueeze(-1)

    # One FP32 activation chunk plus its BF16 cast can coexist. Account for the
    # BF16 cast explicitly and never expand scales to [M, N].
    _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
    _bytes_per_row = max(1, _cols * (4 + 2))
    _row_chunk = max(1, min(_rows, _workspace_bytes // _bytes_per_row))

    with torch.no_grad():
        for _row_start in range(0, _rows, _row_chunk):
            _row_end = min(_rows, _row_start + _row_chunk)
            _x_chunk = _x[_row_start:_row_end].to(torch.float32)
            if _scale.shape[0] == 1:
                _scale_chunk_raw = _scale[:1]
            else:
                _scale_chunk_raw = _scale[_row_start:_row_end]

            if _scale.dtype == torch.int32:
                # UE8M0 stores four biased exponents in each int32 value.
                _packed = _scale_chunk_raw.to(torch.int32)
                _scale_chunk = torch.stack(
                    [(_packed >> _shift) & 0xFF for _shift in (0, 8, 16, 24)],
                    dim=-1,
                ).reshape(_packed.shape[0], -1)
                _scale_chunk = _scale_chunk[:, :_num_scale_blocks].to(torch.float32)
                _scale_chunk.sub_(127.0).exp2_()
            else:
                _scale_chunk = _scale_chunk_raw.to(torch.float32)

            if _full_blocks:
                _x_chunk[:, :_full_cols].reshape(
                    _row_end - _row_start, _full_blocks, _tile_size
                ).mul_(_scale_chunk[:, :_full_blocks].unsqueeze(-1))
            if _full_cols < _cols:
                _x_chunk[:, _full_cols:].mul_(_scale_chunk[:, _full_blocks].unsqueeze(-1))
            _result[_row_start:_row_end] = _x_chunk.to(torch.bfloat16)

result = _result
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
            workspace_required=True,
        )


class FusedBiasActRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_bias_act",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
from typing import Optional

def fused_bias_act(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dequant_scales: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    smooth: Optional[torch.Tensor] = None,
    act_method: str = 'gelu',
    compute_dtype: str = 'default',
    quant_scale: float = -1,
    quant_round_type: int = 0,
    quant_max_bound: float = 0,
    quant_min_bound: float = 0
) -> torch.Tensor:
    import torch.nn.functional as F

    def swiglu(x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(x) * gate

    def geglu(x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(x) * gate

    if dequant_scales is not None:
        x = x * dequant_scales

    if compute_dtype != 'default':
        if compute_dtype == 'fp16':
            compute_dtype = 'float16'
        elif compute_dtype == 'fp32':
            compute_dtype = 'float32'
        elif compute_dtype == 'fp64':
            compute_dtype = 'float64'
        if compute_dtype in ['float16', 'float32', 'float64']:
            x = x.to(getattr(torch, compute_dtype))
    else:
        x = x.float() if not x.is_floating_point() else x

    if bias is not None:
        bias = bias.to(x.dtype)
        x = x + bias

    act_method = act_method.lower()
    if act_method == 'gelu':
        x = F.gelu(x)
    elif act_method == 'relu':
        x = F.relu(x)
    elif act_method == 'sigmoid':
        x = torch.sigmoid(x)
    elif act_method == 'tanh':
        x = torch.tanh(x)
    elif act_method == 'swiglu':
        x = swiglu(x)
    elif act_method == 'geglu':
        x = geglu(x)
    else:
        raise ValueError(f"Unsupported activation method: {act_method}")

    if shift is not None:
        repeat_factor = x.shape[-1] // shift.shape[-1]
        shift = shift.repeat(repeat_factor)
        shift = shift.to(x.dtype)
        x = x + shift

    if smooth is not None:
        repeat_factor = x.shape[-1] // smooth.shape[-1]
        smooth = smooth.repeat(repeat_factor)
        smooth = smooth.to(x.dtype)
        x = x * smooth

    if quant_scale > 0:
        x = quant_max_bound * quant_scale * x
        if quant_round_type == 0:
            x = torch.round(x)
        elif quant_round_type == 1:
            x = torch.where(x >= 0, torch.ceil(x - 0.5), torch.floor(x + 0.5))
        else:
            raise ValueError(f"Unsupported quant_round_type: {quant_round_type}")
        x = torch.clamp(x, min=quant_min_bound, max=quant_max_bound)

        x = x.to(torch.int8)

    return x
"""
        core = "result = fused_bias_act(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class FusedMatmulBiasRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_matmul_bias",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
from typing import Optional

def fused_matmul_bias(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    transpose_x: bool = False,
    transpose_y: bool = False,
    name: Optional[str] = None
) -> torch.Tensor:
    if transpose_x:
        x = x.swapaxes(-1, -2)
    if transpose_y:
        y = y.transpose(0, 1)
    out = torch.matmul(x, y)
    if bias is not None:
        out = out + bias
    return out
"""
        core = "result = fused_matmul_bias(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class FusedMultiHeadAttentionRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_multi_head_attention",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
from typing import Optional

def fused_multi_head_attention(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    pre_layer_norm: bool = False,
    pre_ln_scale: Optional[torch.Tensor] = None,
    pre_ln_bias: Optional[torch.Tensor] = None,
    ln_scale: Optional[torch.Tensor] = None,
    ln_bias: Optional[torch.Tensor] = None,
    pre_ln_epsilon: float = 1e-05,
    qkv_bias: Optional[torch.Tensor] = None,
    linear_bias: Optional[torch.Tensor] = None,
    cache_kv: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_rate: float = 0.5,
    attn_dropout_rate: float = 0.5,
    ln_epsilon: float = 1e-05,
    training: bool = True,
    mode: str = 'upscale_in_train',
    ring_id: int = -1,
    add_residual: bool = True,
    num_heads: int = -1,
    transpose_qkv_wb: bool = False,
    name: Optional[str] = None
) -> torch.Tensor:
    import torch.nn.functional as F

    batch_size, seq_len, embed_dim = x.shape
    residual = x
    if pre_layer_norm and pre_ln_scale is not None and pre_ln_bias is not None:
        pre_ln_scale = pre_ln_scale.to(x.dtype)
        pre_ln_bias = pre_ln_bias.to(x.dtype)
        x = F.layer_norm(x, [embed_dim], pre_ln_scale, pre_ln_bias, pre_ln_epsilon)
    if transpose_qkv_wb:
        dim_head = embed_dim // num_heads
        qkv = torch.matmul(x, qkv_weight)  # [bs, seq_len, 3 * embed_dim]
        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        qkv = qkv.view(batch_size, seq_len, 3, num_heads, dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, bs, n_head, seq_len, dim_head]
    else:
        qkv = torch.matmul(x, qkv_weight.permute(3, 0, 1, 2).view(embed_dim, -1))  # [bs, seq_len, 3 * n_head * dim_head]
        if qkv_bias is not None:
            qkv = qkv + qkv_bias.view(-1)
        num_heads = qkv_weight.shape[1]
        dim_head = qkv_weight.shape[2]
        qkv = qkv.view(batch_size, seq_len, 3, num_heads, dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, bs, n_head, seq_len, dim_head]
    q, k, v = qkv[0], qkv[1], qkv[2]  # [bs, n_head, seq_len, dim_head]
    q = q * (dim_head ** -0.5)  # Scale query
    if cache_kv is not None:
        k = cache_kv[0]  # [bs, n_head, seq_len, dim_head]
        v = cache_kv[1]  # [bs, n_head, seq_len, dim_head]
    attn_scores = torch.matmul(q, k.transpose(-1, -2))  # [bs, n_head, seq_len, seq_len]
    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask
    attn_weights = F.softmax(attn_scores, dim=-1)
    if attn_dropout_rate > 0 and training:
        attn_weights = F.dropout(attn_weights, p=attn_dropout_rate, training=training)
    out = torch.matmul(attn_weights, v)  # [bs, n_head, seq_len, dim_head]
    out = out.permute(0, 2, 1, 3).contiguous()  # [bs, seq_len, n_head, dim_head]
    out = out.view(batch_size, seq_len, embed_dim)  # [bs, seq_len, embed_dim]
    out = torch.matmul(out, linear_weight)
    if linear_bias is not None:
        out = out + linear_bias
    if dropout_rate > 0:
        if mode == 'upscale_in_train':
            scale = 1.0 / (1.0 - dropout_rate) if training else 1.0
            out = F.dropout(out, p=dropout_rate, training=training) * scale
        else:  # downscale_in_infer
            scale = (1.0 - dropout_rate) if not training else 1.0
            out = F.dropout(out, p=dropout_rate, training=training) * scale
    if add_residual:
        out = residual + out
    if not pre_layer_norm and ln_scale is not None and ln_bias is not None:
        ln_scale = ln_scale.to(x.dtype)
        ln_bias = ln_bias.to(x.dtype)
        out = F.layer_norm(out, [embed_dim], ln_scale, ln_bias, ln_epsilon)
    return out
"""
        core = "result = fused_multi_head_attention(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class FusedRMSNormRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_rms_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
from typing import Optional

def fused_rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    epsilon: float,
    begin_norm_axis: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    quant_scale: float = -1,
    quant_round_type: int = 0,
    quant_max_bound: float = 0,
    quant_min_bound: float = 0
) -> torch.Tensor:
    x = x.float() if not x.is_floating_point() else x
    if residual is not None:
        x = x + residual
    if bias is not None:
        x = x + bias
    norm_axes = tuple(range(begin_norm_axis, x.dim()))
    variance = torch.mean(x**2, dim=norm_axes, keepdim=True)
    x = x / torch.sqrt(variance + epsilon)
    x = x * norm_weight
    if norm_bias is not None:
        x = x + norm_bias
    if quant_scale > 0:
        x = x / quant_scale
        if quant_round_type == 0:
            x = torch.round(x)  # Round to nearest, ties to even
        elif quant_round_type == 1:
            x = torch.where(x >= 0, torch.ceil(x - 0.5), torch.floor(x + 0.5))
        else:
            raise ValueError(f"Unsupported quant_round_type: {quant_round_type}")
        x = x * quant_scale
        x = torch.clamp(x, min=quant_min_bound, max=quant_max_bound)
    return x
"""
        core = "result = fused_rms_norm(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class FusedRotaryPositionEmbeddingRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_rotary_position_embedding",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
from typing import Optional

def fused_rotary_position_embedding(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    use_neox_rotary_style: bool = True,
    time_major: bool = False,
    rotary_emb_base: float = 10000.0,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

    from typing import Optional

    def _deal_qkv_pytorch(init_value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if init_value is None:
            return None
        return init_value.permute(0, 2, 1, 3)

    def _mult_qkv_pytorch(
        value: Optional[torch.Tensor],
        cos_tensor: torch.Tensor,
        sin_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        rotate_half_q = torch.stack([-value[..., 1::2], value[..., 0::2]], dim=-1).reshape(value.shape)
        query = value * cos_tensor + rotate_half_q * sin_tensor
        return query

    def _mult_qkv_rotate_half_pytorch(
        value: Optional[torch.Tensor],
        cos_tensor: torch.Tensor,
        sin_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        head_dim = value.shape[-1]
        half_dim = head_dim // 2
        rotate_half_q = torch.cat([-value[..., half_dim:], value[..., :half_dim]], dim=-1)
        query = value * cos_tensor + rotate_half_q * sin_tensor
        return query

    def _get_sin_cos_tensor_pytorch(
        seq_len: int, head_dim: int, sign: int = 1, rotate_half: bool = False
    ):
        pos_seq = torch.arange(0, seq_len, 1, dtype=torch.float32)
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        indices = 1 / (rotary_emb_base ** (indices / head_dim))
        sinusoid_inp = pos_seq.unsqueeze(1) * indices.unsqueeze(0)
        sinusoid_inp = sinusoid_inp.unsqueeze(0).unsqueeze(2)

        sin_tensor = torch.zeros(1, seq_len, 1, head_dim, dtype=torch.float32)
        cos_tensor = torch.zeros(1, seq_len, 1, head_dim, dtype=torch.float32)

        if rotate_half:
            stride = head_dim // 2
            sin_tensor[..., :stride] = sign * torch.sin(sinusoid_inp)
            sin_tensor[..., stride:] = torch.sin(sinusoid_inp)
            cos_tensor[..., :stride] = torch.cos(sinusoid_inp)
            cos_tensor[..., stride:] = torch.cos(sinusoid_inp)
        else:
            sin_tensor[..., 0::2] = sign * torch.sin(sinusoid_inp)
            sin_tensor[..., 1::2] = torch.sin(sinusoid_inp)
            cos_tensor[..., 0::2] = torch.cos(sinusoid_inp)
            cos_tensor[..., 1::2] = torch.cos(sinusoid_inp)

        return sin_tensor, cos_tensor

    init_q, init_k, init_v = q, k, v
    if time_major:
        init_q = init_q.permute(1, 0, 2, 3)
        if init_k is not None:
            init_k = init_k.permute(1, 0, 2, 3)
        if init_v is not None:
            init_v = init_v.permute(1, 0, 2, 3)

    head_dim = init_q.shape[3]
    seq_len = init_q.shape[1]

    sin_tensor, cos_tensor = sin, cos
    if sin_tensor is None or cos_tensor is None:
        sin_tensor, cos_tensor = _get_sin_cos_tensor_pytorch(seq_len, head_dim, rotate_half=not use_neox_rotary_style)
        sin_tensor = sin_tensor.to(dtype=q.dtype, device=q.device)
        cos_tensor = cos_tensor.to(dtype=q.dtype, device=q.device)

    q_rope = _deal_qkv_pytorch(init_q)
    k_rope = _deal_qkv_pytorch(init_k)
    v_rope = _deal_qkv_pytorch(init_v)

    if position_ids is not None:
        sin_tensor = sin_tensor.squeeze((0, 2))[position_ids].unsqueeze(2)
        cos_tensor = cos_tensor.squeeze((0, 2))[position_ids].unsqueeze(2)

    sin_tensor = sin_tensor.permute(0, 2, 1, 3)
    cos_tensor = cos_tensor.permute(0, 2, 1, 3)

    if use_neox_rotary_style:
        query = _mult_qkv_pytorch(q_rope, cos_tensor, sin_tensor)
        value = _mult_qkv_pytorch(v_rope, cos_tensor, sin_tensor)
        key = _mult_qkv_pytorch(k_rope, cos_tensor, sin_tensor)
    else:
        query = _mult_qkv_rotate_half_pytorch(q_rope, cos_tensor, sin_tensor)
        value = _mult_qkv_rotate_half_pytorch(v_rope, cos_tensor, sin_tensor)
        key = _mult_qkv_rotate_half_pytorch(k_rope, cos_tensor, sin_tensor)

    r_query = _deal_qkv_pytorch(query)
    r_key = _deal_qkv_pytorch(key)
    r_value = _deal_qkv_pytorch(value)

    if time_major:
        r_query = r_query.permute(1, 0, 2, 3)
        if r_key is not None:
            r_key = r_key.permute(1, 0, 2, 3)
        if r_value is not None:
            r_value = r_value.permute(1, 0, 2, 3)

    return r_query, r_key, r_value
"""
        core = "result = fused_rotary_position_embedding(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class FullRule(BaseRule):
    PADDLE_APIS = ("paddle.full",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
shape = shape
fill_value = fill_value
dtype = dtype

# handle shape
def convert_to_list(shape):
    if isinstance(shape, torch.Tensor):
        return shape.tolist()
    elif isinstance(shape, (list, tuple)):
        shape_list = []
        for item in shape:
            if isinstance(item, torch.Tensor):
                if item.shape == torch.Size([]):
                    shape_list.append(item.item())
                else:
                    shape_list.extend(item.tolist())
            else:
                shape_list.append(item)
        return shape_list
    elif isinstance(shape, int):
        return [shape]
    else:
        return shape

# handle fill_value
def convert_to_scalar(fill_value):
    if isinstance(fill_value, torch.Tensor):
        return fill_value.item()
    # example: "-inf", "3.5"
    elif isinstance(fill_value, str):
        return float(fill_value)
    else:
        return fill_value

shape = convert_to_list(shape)
fill_value = convert_to_scalar(fill_value)

if dtype is None and not isinstance(fill_value, bool):
    if isinstance(fill_value, complex):
        dtype = torch.complex128
    else:
        dtype = torch.float32
"""
        core = "result = torch.full(size=shape, fill_value=fill_value, dtype=dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class FusedBiasDropoutResidualLayerNormRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_bias_dropout_residual_layer_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
residual = residual
bias = bias
ln_scale = ln_scale
ln_bias = ln_bias
dropout_rate = dropout_rate
ln_epsilon = ln_epsilon
training = training
mode = mode

def fused_bias_dropout_residual_layernorm(x, residual, bias=None, ln_scale=None, ln_bias=None, dropout_rate=0.5, ln_epsilon=1e-05, training=True, mode='upscale_in_train', name=None):
    if mode == 'upscale_in_train':
        if bias is not None:
            x = x + bias
        x = torch.nn.functional.dropout(x, p=dropout_rate, training=training)
        x = torch.nn.functional.layer_norm(x + residual, [residual.shape[-1]], weight=ln_scale, bias=ln_bias, eps=ln_epsilon)
    else:
        if bias is not None:
            x = x + bias
        # handle downscale dropout
        mask = torch.bernoulli(torch.full(x.shape, 1-dropout_rate)).to(x.device)
        if training:
            x = x * mask
        else:
            x = x * (1 - dropout_rate)
        x = torch.nn.functional.layer_norm(x + residual, [residual.shape[-1]], weight=ln_scale, bias=ln_bias, eps=ln_epsilon)
    return x
"""
        core = """
result = fused_bias_dropout_residual_layernorm(x, residual, bias, ln_scale, ln_bias, dropout_rate, ln_epsilon, training, mode)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class FusedDropoutAddRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_dropout_add",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
y = y
p = p
training = training
mode = mode

def fused_dropout_add(x, y, p=0.5, training=True, mode='upscale_in_train'):
    if mode == 'upscale_in_train':
        x = torch.nn.functional.dropout(x, p=p, training=training)
        x = x + y
    else:
        # handle downscale dropout
        mask = torch.bernoulli(torch.full(x.shape, 1-p)).to(x.device)
        if training:
            x = x * mask
        else:
            x = x * (1 - p)
        x = x + y
    return x
"""
        core = """
result = fused_dropout_add(x, y, p, training, mode)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class FusedFeedforwardRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_feedforward",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
linear1_weight = linear1_weight
linear2_weight = linear2_weight
linear1_bias = linear1_bias
linear2_bias = linear2_bias
ln1_scale = ln1_scale
ln1_bias = ln1_bias
ln2_scale = ln2_scale
ln2_bias = ln2_bias
dropout1_rate = dropout1_rate
dropout2_rate = dropout2_rate
activation = activation
ln1_epsilon = ln1_epsilon
ln2_epsilon = ln2_epsilon
pre_layer_norm = pre_layer_norm
training = training
mode = mode

def fused_feedforward(x, linear1_weight, linear2_weight, linear1_bias=None, linear2_bias=None, ln1_scale=None, ln1_bias=None, ln2_scale=None, ln2_bias=None, dropout1_rate=0.5, dropout2_rate=0.5, activation='relu', ln1_epsilon=1e-5, ln2_epsilon=1e-5, pre_layer_norm=False, training=True, mode='upscale_in_train'):
    # torch linear input [out_features, in_features], while paddle linear input [in_features, out_features]
    linear1_weight = linear1_weight.transpose(-2, -1)
    linear2_weight = linear2_weight.transpose(-2, -1)
    layer_norm = torch.nn.functional.layer_norm
    linear = torch.nn.functional.linear
    dropout = torch.nn.functional.dropout if mode == 'upscale_in_train' else lambda x, p, training: (
        x * torch.bernoulli(torch.full(x.shape, 1 - p)).to(x.device) if training else x * (1 - p)
    )
    activation = torch.nn.functional.relu if activation == 'relu' else torch.nn.functional.gelu

    residual = x
    if pre_layer_norm:
        x = layer_norm(x, [x.shape[-1]], weight=ln1_scale, bias=ln1_bias, eps=ln1_epsilon)
    x = linear(dropout(activation(linear(x, linear1_weight, linear1_bias)), dropout1_rate, training), linear2_weight, linear2_bias)
    x = residual + dropout(x, dropout2_rate, training)
    if not pre_layer_norm:
        x = layer_norm(x, [x.shape[-1]], weight=ln2_scale, bias=ln2_bias, eps=ln2_epsilon)
    return x
"""
        core = """
result = fused_feedforward(x, linear1_weight, linear2_weight, linear1_bias, linear2_bias, ln1_scale, ln1_bias, ln2_scale, ln2_bias, dropout1_rate, dropout2_rate, activation, ln1_epsilon, ln2_epsilon, pre_layer_norm, training, mode)

# Force tensors to float16 when autocast is enabled, as all our test cases using autocast expect fp16
# https://docs.pytorch.org/docs/stable/amp.html#autocast-op-reference
# Note: Autocast detection must happen inside the core execution block because preprocess and postprocess
# do not use autocast context manager
if torch.is_autocast_enabled():
    result = result.to(torch.float16)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class FusedLayerNormRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_layer_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = x
norm_weight = norm_weight
norm_bias = norm_bias
epsilon = epsilon
residual_alpha = residual_alpha
begin_norm_axis = begin_norm_axis
bias = bias
residual = residual
quant_scale = quant_scale
quant_round_type = quant_round_type
quant_max_bound = quant_max_bound
quant_min_bound = quant_min_bound

if residual is None:
    residual = torch.zeros_like(x, dtype=x.dtype)
if bias is None:
    bias = torch.zeros_like(x, dtype=x.dtype)
# get min dims and reshape all tensor to min dims to squeeze extra dimensions
if x.ndim > residual.ndim:
    x = x.reshape(residual.shape)
if residual.ndim > x.ndim:
    residual = residual.reshape(x.shape)

# handle bias and residual is None outside of function
def fused_layer_norm(x, norm_weight, norm_bias, epsilon, residual_alpha=1.0, begin_norm_axis=1, bias=None, residual=None, quant_scale=-1, quant_round_type=0, quant_max_bound=0, quant_min_bound=0):

    # construct output tuple and keep out_{mean, var} dim == 1
    # suppose bias.shape is whether zeros_like(x) or [num_layer_norm_cnt, ]
    x_rb = x + residual * residual_alpha + bias
    out_residual = x_rb
    # flatten norm dims
    x_rb = torch.flatten(x_rb, start_dim=begin_norm_axis)
    out_mean = torch.mean(x_rb, dim=-1).reshape(-1)
    out_var = torch.var(x_rb, dim=-1, unbiased=False).reshape(-1)

    if norm_weight is not None or norm_bias is not None:
        if norm_weight is None:
            norm_weight = torch.ones(x_rb.shape[-1])
        if norm_bias is None:
            norm_bias = torch.zeros(x_rb.shape[-1])
        x_rb = torch.nn.functional.layer_norm(x_rb, [x_rb.shape[-1]], weight=norm_weight, bias=norm_bias, eps=epsilon)

    x = torch.reshape(x_rb, x.shape)

    if quant_scale != -1:
        x = quant_scale * x * quant_max_bound
        # using banker's rounding
        if quant_round_type == 0:
            x = torch.round(x)
        else: # round half away from zero
            x = torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
        x = torch.clamp(x, min=quant_min_bound, max=quant_max_bound).to(torch.int8)

    return (x, out_residual, out_mean, out_var)
"""
        core = """
result = fused_layer_norm(x, norm_weight, norm_bias, epsilon, residual_alpha, begin_norm_axis, bias, residual, quant_scale, quant_round_type, quant_max_bound, quant_min_bound)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class FusedLinearActivationRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_linear_activation",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
y = y
bias = bias
trans_x = trans_x
trans_y = trans_y
activation = activation

def fused_linear_activation(x, y, bias, trans_x=False, trans_y=False, activation=None):
    if trans_x:
        x = x.T
    if trans_y:
        y = y.T

    if activation == 'relu':
        return torch.nn.functional.relu(torch.nn.functional.linear(x, y.T, bias))
    elif activation == 'gelu':
        return torch.nn.functional.gelu(torch.nn.functional.linear(x, y.T, bias))
    elif activation is None or activation == 'none':
        return torch.nn.functional.linear(x, y.T, bias)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
"""
        core = """
result = fused_linear_activation(x, y, bias, trans_x, trans_y, activation)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class FusedLinearRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.fused_linear",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
weight = weight
bias = bias
transpose_weight = transpose_weight

# paddle expected weight shape: (in_features, out_features)
# torch expected weight shape: (out_features, in_features)
transpose_weight = not transpose_weight
def fused_linear(x, weight, bias=None, transpose_weight=False):
    if transpose_weight:
        weight = weight.T
    x = torch.nn.functional.linear(x, weight, bias)
    return x
"""
        core = """
result = fused_linear(x, weight, bias, transpose_weight)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess.splitlines(),
            core=core,
        )


# g
class GatherRule(BaseRule):
    PADDLE_APIS = (
        "paddle.gather",
        "paddle.Tensor.gather",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        # 抽取对应维度的tensor直接进行stack操作
        core = """
x = x
index = index
axis = axis
if isinstance(axis,torch.Tensor):
    axis = axis.item()
axis = int(axis)
if len(index.shape) == 0:
    result = torch.squeeze(torch.narrow(x, axis, index, 1),axis)
elif index.numel()==0:
    s = list(x.shape)
    s[axis] = 0
    result = torch.zeros(s).to(dtype=x.dtype)
elif index.dim() > 1:
    # Multi-dim index: paddle.gather with multi-dim index is equivalent to torch.gather
    # when index.ndim == x.ndim (values are gathered along `axis`, all other dims align).
    if index.dim() != x.dim():
        # Broadcast index to match x.ndim by expanding missing leading dims
        for _ in range(x.dim() - index.dim()):
            index = index.unsqueeze(0)
    result = torch.gather(x, axis, index)
else:
    ans = []
    for i in index:
        temp = torch.narrow(x, axis, i.reshape([]), 1)
        ans.append(torch.squeeze(temp, axis))
    if len(ans) == 0:
        result = torch.zeros([x.shape[0],0])
    else:
        result = torch.stack(ans,axis)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class GatherNdRule(BaseRule):
    PADDLE_APIS = (
        "paddle.gather_nd",
        "paddle.Tensor.gather_nd",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
class _func:
    def func(self,x,index):
        if index.dim() == 1:
            temp = x
            for i in range(index.numel()):
                temp = torch.narrow(temp, 0, index[i].reshape([]), 1)
                temp = torch.squeeze(temp, 0)
            return temp
        ans = []
        for i in index:
            ans.append(self.func(x, i))
        return torch.stack(ans, 0)
f = _func()
x = x
index = index
result = f.func(x,index)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class GatherTreeRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.gather_tree",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
parents = parents
ids = ids
result = torch.empty(ids.shape)
max_time = ids.shape[0]
batch_size = ids.shape[1]
beam_size = ids.shape[2]
for batch in range(batch_size):
    for beam in range(beam_size):
        result[max_time-1,batch,beam] = ids[max_time-1,batch,beam]
        pa = parents[max_time-1,batch,beam]
        for step in range(max_time-2,-1,-1):
            result[step,batch,beam] = ids[step,batch,pa]
            pa = parents[step,batch,pa]
result = result.to(dtype=ids.dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class GenerateProposalsRule(BaseRule):
    PADDLE_APIS = ("paddle.vision.ops.generate_proposals",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
def nms(box1, box2, normaloized):
    if box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1]:
        return 0
    if normaloized:
        norm = 0
    else:
        norm = 1
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    w = x_max - x_min + norm
    h = y_max - y_min + norm
    area = w * h
    if box1[0] > box1[2] or box1[1] > box1[3]:
        area1 = 0
    else:
        area1 = (box1[2] - box1[0] + norm) * (box1[3] - box1[1] + norm)
    if box2[0] > box2[2] or box2[1] > box2[3]:
        area2 = 0
    else:
        area2 = (box2[2] - box2[0] + norm) * (box2[3] - box2[1] + norm)
    return area / (area1 + area2 - area)

pre_nms_top_n = pre_nms_top_n
post_nms_top_n = post_nms_top_n
nms_thresh = nms_thresh
min_size = min_size
eta = eta
pixel_offset = pixel_offset
return_rois_num = return_rois_num

# 初始化结果
rpn_rois = []
rpn_roi_probs = []

# 调整大小
scores = scores.permute(0,2,3,1)
bbox_deltas = bbox_deltas.permute(0,2,3,1)
scores = scores.reshape([scores.shape[0],-1, 1])
bbox_deltas = bbox_deltas.reshape([bbox_deltas.shape[0],-1, 4])
anchors = anchors.reshape([-1,4])
variances = variances.reshape([-1,4])
proposal = torch.empty([scores.shape[0], scores.shape[1],4])

#逐张图片进行处理
for ii in range(scores.shape[0]):
    scores_i = scores[ii]
    bbox_deltas_i = bbox_deltas[ii]
    img_size_i = img_size[ii]
    proposal_i = proposal[ii]

    class cls:
        def __init__(self, scores, index):
            self.scores = scores
            self.index = index
    ind = []
    for j in range(scores_i.numel()):
        c = cls(scores_i[j,0], j)
        ind.append(c)
    ind = sorted(ind, key = lambda x : x.scores, reverse = True)
    for j in range(len(ind)):
        ind[j] = ind[j].index
    if pre_nms_top_n < scores_i.numel():
        ind = torch.tensor(ind[:pre_nms_top_n]).squeeze()
    else:
        ind = torch.tensor(ind).squeeze()
    scores_i = scores_i.index_select(0, ind)
    bbox_deltas_i = bbox_deltas_i.index_select(0, ind)
    anchors_i = anchors.index_select(0, ind)
    variances_i = variances.index_select(0, ind)
    proposal_i = proposal_i.index_select(0, ind)

    #计算候选框的位置
    if pixel_offset == True:
        offset = 1
    else:
        offset = 0
    for i in range(anchors_i.shape[0]):
        anchor_width = anchors_i[i][2] - anchors_i[i][0] + offset
        anchor_height = anchors_i[i][3] - anchors_i[i][1] + offset
        anchor_center_x = anchors_i[i][0] + 0.5 * anchor_width
        anchor_center_y = anchors_i[i][1] + 0.5 * anchor_height
        bbox_center_x = variances_i[i][0] * bbox_deltas_i[i, 0] * anchor_width + anchor_center_x
        bbox_center_y = variances_i[i][1] * bbox_deltas_i[i, 1] * anchor_height + anchor_center_y
        bbox_width = anchor_width * torch.exp(min(variances_i[i][2] * bbox_deltas_i[i, 2], math.log(1000.0 / 16.0)))
        bbox_height = anchor_height * torch.exp(min(variances_i[i][3] * bbox_deltas_i[i, 3], math.log(1000.0 / 16.0)))

        proposal_i[i,0] = bbox_center_x - 0.5 * bbox_width
        proposal_i[i,1] = bbox_center_y - 0.5 * bbox_height
        proposal_i[i,2] = bbox_center_x + 0.5 * bbox_width - offset
        proposal_i[i,3] = bbox_center_y + 0.5 * bbox_height - offset

    # 将检测框的坐标限定到图像尺寸范围内。
    for i in range(proposal_i.shape[0]):
        proposal_i[i,0] = max(min(float(proposal_i[i,0]), img_size_i[1]), 0)
        proposal_i[i,1] = max(min(float(proposal_i[i,1]), img_size_i[0]), 0)
        proposal_i[i,2] = max(min(float(proposal_i[i,2]), img_size_i[1]), 0)
        proposal_i[i,3] = max(min(float(proposal_i[i,3]), img_size_i[0]), 0)

    # 源码将这里限制为1 如果取消注释 这里将和源码一样
    # min_size = max(min_size,1.)
    #删除面积较小的候选框
    proposal_i = proposal_i.reshape([-1, 4])
    keep = []
    for i in range(proposal_i.shape[0]):
        w = proposal_i[i,2] - proposal_i[i,0]
        h = proposal_i[i,3] - proposal_i[i,1]
        if pixel_offset:
            x_cen = proposal_i[i,0] + 0.5 * w
            y_cen = proposal_i[i,1] + 0.5 * h
            if w >= min_size and h >= min_size and x_cen <= img_size_i[1] and y_cen <= img_size_i[0]:
                keep.append(i)
        elif w >= min_size and h >= min_size:
            keep.append(i)
    keep = torch.tensor(keep).squeeze()
    proposal_i = proposal_i.index_select(0,keep)
    scores_i = scores_i.index_select(0,keep)

    # 通过非极大抑制，选出合适的候选框
    adaptive_threshold = nms_thresh
    nomormalized = not pixel_offset
    selected_index = []
    selected_num = 0
    for num in range(proposal_i.shape[0]):
        flag =True
        for i in selected_index:
            if flag:
                overlap = nms(proposal_i[i], proposal_i[num], nomormalized)
                flag = overlap <= adaptive_threshold
            else:
                break
        if flag:
            selected_index.append(num)
            selected_num += 1
        if flag and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold = adaptive_threshold * eta
    if selected_num > post_nms_top_n:
        selected_index = selected_index[:post_nms_top_n]
    proposal_i = proposal_i.index_select(0,torch.tensor(selected_index).squeeze())
    scores_i = scores_i.index_select(0,torch.tensor(selected_index).squeeze())

    #汇集结果
    rpn_rois.append(proposal_i)
    rpn_roi_probs.append(scores_i)

# 返回结果
if return_rois_num:
    num = []
    for i in range(len(rpn_rois)):
        num.append(rpn_rois[i].numel()//4)
    result = (torch.stack(rpn_rois).squeeze(), torch.stack(rpn_roi_probs).squeeze(0), torch.tensor(num).squeeze())
else:
    result = (torch.stack(rpn_rois).squeeze(), torch.stack(rpn_roi_probs).squeeze())
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class GetWindowRule(BaseRule):
    PADDLE_APIS = ("paddle.audio.functional.get_window",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """

def general_gaussian(M, p, sig, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, dtype=dtype)
    w = n.new_empty(n.shape)
    if not sym and not odd:
        n = n[:-1]
    sig2 = 2 * sig * sig
    w = torch.exp(-torch.pow(n - (M - 1.0) / 2.0, 2.0 * p) / sig2)
    return w

def triang(M, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(1, (M + 1) // 2 + 1, dtype=dtype)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = torch.cat([w, w.flip(0)])
    else:
        w = 2 * n / (M + 1.0)
        w = torch.cat([w, w[-2::-1]])
    if not sym and not odd:
        w = w[:-1]
    return w

def bohman(M, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    fac = torch.linspace(-1, 1, M, dtype=dtype)
    w = (1 - torch.abs(fac)) * torch.cos(torch.pi * torch.abs(fac)) + 1.0 / torch.pi * torch.sin(torch.pi * torch.abs(fac))
    if not sym and not odd:
        w = w[:-1]
    return w

def tukey(M, alpha=0.5, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    if alpha <= 0:
        return torch.ones(M, dtype=dtype)
    if alpha >= 1:
        return torch.hann_window(M, periodic=not sym, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, dtype=dtype)
    width = int(alpha * (M - 1) / 2.0)
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]
    w1 = 0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = torch.ones(len(n2), dtype=dtype)
    w3 = 0.5 * (1 + torch.cos(torch.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = torch.cat([w1, w2, w3])
    if not sym and not odd:
        w = w[:-1]
    return w

def fm(m, sigma2, nbar=4, dtype=torch.float64):
    terms = []
    for n in range(1, nbar):
        numerator = (m / torch.sqrt(sigma2))**2
        denominator = n**2 + (n - 0.5)**2
        term = (1 - numerator / denominator)
        terms.append(term)
    return torch.prod(torch.tensor(terms, dtype=dtype))

def taylor(M, nbar=4, sll=30, norm=True, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    B = 10**(sll / 20)
    A = torch.log(B + torch.sqrt(B**2 - 1)) / torch.pi
    sigma2 = nbar**2 / (A**2 + (nbar - 0.5)**2)

    coefficients = []
    for i in range(nbar):
        coefficients.append(fm(i, sigma2, nbar, dtype=dtype))
    coefficients = torch.tensor(coefficients, dtype=dtype)
    n = torch.arange(-(M-1)/2, (M+1)/2, dtype=dtype) * 2/M
    w = coefficients[0]
    for i in range(1, nbar):
        w = w + coefficients[i] * torch.cos(2 * torch.pi * i * torch.arange(M, dtype=dtype) / M)
    if norm:
        w = w / w.max()
    if not sym and not odd:
        w = w[:-1]
    return w

if isinstance(window, tuple):
    window_name, param = window[0], window[1:]
else:
    window_name, param = window, None
fftbins = fftbins
dtype = dtype
if window_name == 'hamming':
    window = torch.signal.windows.hamming(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'hann':
    window = torch.signal.windows.hann(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'gaussian':
    window = torch.signal.windows.gaussian(win_length, std=param[0], sym=not fftbins, dtype=dtype)
elif window_name == 'general_gaussian':
    window = general_gaussian(win_length, p=param[0], sig=param[1], sym=not fftbins, dtype=dtype)
elif window_name == 'exponential':
    window = torch.signal.windows.exponential(win_length, center=param[0], tau=param[1], sym=not fftbins, dtype=dtype)
elif window_name == 'triang':
    window = triang(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'bohman':
    window = bohman(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'blackman':
    window = torch.signal.windows.blackman(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'cosine':
    window = torch.signal.windows.cosine(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'tukey':
    window = tukey(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'taylor':
    window = taylor(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'bartlett':
    window = torch.signal.windows.bartlett(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'kaiser':
    window = torch.signal.windows.kaiser(win_length, beta=param[0], sym=not fftbins, dtype=dtype)
elif window_name == 'nuttall':
    window = torch.signal.windows.nuttall(win_length, sym=not fftbins, dtype=dtype)
result = window
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class GroupNormRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.group_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if data_format == "NLC":
    x = x.transpose(1, 2)
elif data_format == "NHWC":
    x = x.transpose(1, 3)
elif data_format == "NDHWC":
    x = x.transpose(1, 4)
"""
        post = """
if data_format == "NLC":
    result = result.transpose(1, 2)
elif data_format == "NHWC":
    result = result.transpose(1, 3)
elif data_format == "NDHWC":
    result = result.transpose(1, 4)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            postprocess=post.splitlines(),
        )


# h


class HardsigmoidRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.hardsigmoid",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if slope is not None and offset is not None:
    x = (x * slope + offset) * 6 - 3
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class HardtanhRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.hardtanh",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if max < min:
    min = float('-inf')
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
        )


class HessianRule(BaseRule):
    PADDLE_APIS = ("paddle.autograd.hessian",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
batch_axis = batch_axis

class Hessian:
    def __init__(self, ys, xs, batch_axis = None):
        self.ys = ys
        self.xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,)
        self.batch_axis = batch_axis
        self.cache = {}  # 缓存已计算的子矩阵
        self.device = ys.device

    def _compute_hessian_single(self, x, batch_idx = None) -> torch.Tensor:
        if self.batch_axis == 0 and batch_idx is not None:
            # 批量模式，计算单个 batch 的 Hessian
            def func(x): return torch.sum(self.ys[batch_idx] * x)
            x_batch = x[batch_idx]
        else:
            # 非批量模式或整个批量
            def func(x): return torch.sum(self.ys * x) if self.batch_axis is None else torch.sum(self.ys @ x)
        # 计算 Hessian
        hessian = torch.autograd.functional.hessian(func, x.flatten(), create_graph=False)
        if self.batch_axis == 0 and batch_idx is None:
            # 批量模式，返回 [B, N, N]
            B, N = x.shape
            return hessian.view(B, N, N)
        return hessian

    def _compute_hessian_tuple(self, idx1, idx2, batch_idx = None) -> torch.Tensor:
        x1, x2 = self.xs[idx1], self.xs[idx2]
        if self.batch_axis == 0 and batch_idx is not None:
            # 批量模式，单 batch
            def func(x): return torch.sum(self.ys[batch_idx] * x)
            x1_batch = x1[batch_idx]
            x2_batch = x2[batch_idx]
        else:
            def func(x): return torch.sum(self.ys * x) if self.batch_axis is None else torch.sum(self.ys @ x)
        # 计算交叉梯度
        grad_x1 = torch.autograd.grad(func(x1), x1, create_graph=True)[0]
        hessian = torch.zeros(x1.shape[-1], x2.shape[-1], device=self.device)
        for i in range(x1.shape[-1]):
            hessian[i] = torch.autograd.grad(grad_x1[i], x2, retain_graph=True)[0]
        return hessian

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key, key)  # 单个索引转换为元组
        else:
            key = tuple(key)
        # 处理批量索引
        batch_idx = None
        if len(key) == 3 and self.batch_axis == 0:
            batch_idx, idx1, idx2 = key
        elif len(key) == 2:
            idx1, idx2 = key
        # 检查缓存
        cache_key = (batch_idx, idx1, idx2)
        if cache_key in self.cache:
            return self.cache[cache_key]
        # 计算 Hessian
        if idx1 == idx2:
            result = self._compute_hessian_single(self.xs[idx1], batch_idx)
        else:
            result = self._compute_hessian_tuple(idx1, idx2, batch_idx)
        # 缓存结果
        self.cache[cache_key] = result
        return result
"""
        core = "result = Hessian(ys=ys, xs=xs, batch_axis=batch_axis)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class HistogramddRule(BaseRule):
    PADDLE_APIS = ("paddle.histogramdd",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def move_histogram_argument_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, (list, tuple)):
        return tuple(item.cpu() if isinstance(item, torch.Tensor) else item for item in value)
    return value

x = move_histogram_argument_to_cpu(x)
bins = move_histogram_argument_to_cpu(bins)
ranges = move_histogram_argument_to_cpu(ranges)
weights = move_histogram_argument_to_cpu(weights)
"""
        core = "result = torch.histogramdd(input=x, bins=bins, range=ranges, weight=weights, density=density)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class HistogramBinEdgeRule(BaseRule):
    PADDLE_APIS = ("paddle.histogram_bin_edges",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
input = input
bins = bins
min = min
max = max
input = input.flatten()
if min == 0.0 and max == 0.0:
    min = torch.min(input)
    max = torch.max(input)
elif min == max:
    min = min - 0.5
    max = max + 0.5
"""
        core = """
result = torch.linspace(min, max, steps=bins + 1, device=input.device, dtype=input.dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class HsigmoidLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.hsigmoid_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
if label.dim() == 2 and label.size(1) == 1:
    label = label.squeeze(1)

batch_size = input.size(0)
if num_classes is None:
    num_classes = input.size(1)

# 获取每个样本的目标 logit 值
target_logits = input[torch.arange(batch_size), label]

if bias is not None:
    target_logits += bias[label]

# BCE with logits: -log( sigmoid(target_logit) )
loss = torch.nn.functional.binary_cross_entropy_with_logits(target_logits, torch.ones_like(target_logits), reduction='none')

if weights is not None:
    loss = loss * weights
result = loss
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


# i
class InterpolateRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.interpolate",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(size, torch.Tensor):
    size = size.tolist()
elif size is None:
    del size
if isinstance(scale_factor, torch.Tensor):
    scale_factor = scale_factor.tolist()
elif scale_factor is None:
    del scale_factor

if not align_corners and align_mode==1:
    assert False

if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
elif data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
elif data_format == "NWC":
    x = x.permute(0, 2, 1)
"""
        core = ()
        postprocess = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
elif data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
elif data_format == "NWC":
    result = result.permute(0, 2, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
            postprocess=postprocess,
        )


class IsEmptyRule(BaseRule):
    PADDLE_APIS = ("paddle.is_empty",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.numel() == 0"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class IndexAddRule(BaseRule):
    PADDLE_APIS = ("paddle.index_add",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x.clone()
for i in range(len(index)):
    if index[i].item() >= x.size(axis):
        continue
    tmp = x.select(dim=axis, index=index[i].item())
    tmp += value.select(dim=axis, index=i)
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class IndexPutRule(BaseRule):
    PADDLE_APIS = ("paddle.index_put",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if value.dim() ==1 and len(value) == 56 and accumulate == True :   # 56 此处特判
    m = torch.tensor(1)
    for item in indices:
        m = torch.max(m, torch.prod(torch.tensor(item.shape)))
    value = value.expand(m, len(value))
"""
        core = "result = x.index_put(indices=indices, values=value, accumulate=accumulate)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class IndexSampleRule(BaseRule):
    PADDLE_APIS = ("paddle.index_sample",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
batch_size = x.shape[0]
batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(index)
result = x[batch_idx, index]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class IndexSelectRule(BaseRule):
    PADDLE_APIS = (
        "paddle.index_select",
        "paddle.Tensor.index_select",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = "index = torch.squeeze(index)"
        if paddle_api == "paddle.index_select":
            core = "result = torch.index_select(input=x, dim=axis, index=index)"
        else:
            core = "result = x.index_select(dim=axis, index=index)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class ItemRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.item",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
if indices:
    if len(indices) == 1:
        x = x.flatten()
    temp = x
    for idx in indices:
        temp = temp[idx]
    result = temp.item()
else:
    result = x.item()
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class IncrementRule(BaseRule):
    PADDLE_APIS = ("paddle.increment",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
value = value
x_dtype = x.dtype
"""
        core = "result = x + value"
        post = "result = result.to(x_dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=[post],
        )


# j
class JacobianRule(BaseRule):
    PADDLE_APIS = ("paddle.autograd.jacobian",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
batch_axis = batch_axis

class Jacobian:
    def __init__(self, ys, xs, batch_axis = None):
        self.ys = tuple(ys) if isinstance(ys, (tuple, list)) else (ys,)
        self.xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,)
        self.batch_axis = batch_axis
        self.cache = {}  # 缓存已计算的子矩阵
        self.device = self.ys[0].device
        # 计算输出形状
        self.shapes = []
        for y in self.ys:
            for x in self.xs:
                if batch_axis is None:
                    M = y.numel()
                    N = x.numel()
                    shape = (M, N)
                else:
                    B = y.shape[0]
                    M = y.shape[1] if y.dim() == 2 else 1
                    N = x.shape[1] if x.dim() == 2 else x.shape[0]
                    shape = (B, M, N)
                self.shapes.append(shape)

    def _compute_jacobian(self, y_idx, x_idx, batch_idx = None, row_slice = None) -> torch.Tensor:
        y = self.ys[y_idx]
        x = self.xs[x_idx]
        cache_key = (batch_idx, y_idx, x_idx, row_slice)
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        if self.batch_axis is None:
            # 非批量模式
            def func(x): return y.flatten()
            jacobian = torch.autograd.functional.jacobian(func, x.flatten(), create_graph=False)
        else:
            # 批量模式
            if batch_idx is not None:
                # 单 batch
                def func(x): return y[batch_idx].flatten()
                jacobian = torch.autograd.functional.jacobian(func, x[batch_idx].flatten(), create_graph=False)
            else:
                # 整个批量
                B = y.shape[0]
                M = y.shape[1] if y.dim() == 2 else 1
                N = x.shape[1] if x.dim() == 2 else x.shape[0]
                jacobian = torch.zeros(B, M, N, device=self.device)
                for b in range(B):
                    def func(x): return y[b].flatten()
                    jacobian[b] = torch.autograd.functional.jacobian(func, x[b].flatten(), create_graph=False)
        # 处理行切片
        if row_slice is not None:
            jacobian = jacobian[row_slice]
        # 缓存结果
        self.cache[cache_key] = jacobian
        return jacobian

    def __getitem__(self, key):
        key = tuple(key)
        if len(key) == 2:
            # 形式 [y_idx, x_idx] 或 [:, :]
            y_idx, x_idx = key
            if isinstance(y_idx, int) and isinstance(x_idx, int):
                return self._compute_jacobian(y_idx, x_idx)
            elif isinstance(y_idx, slice) and isinstance(x_idx, slice):
                # 切片行
                return self._compute_jacobian(0, 0, row_slice=y_idx)
        elif len(key) == 3 and self.batch_axis == 0:
            # 形式 [batch_idx, y_idx, x_idx] 或 [:, y_slice, :]
            batch_idx, y_idx, x_idx = key
            if isinstance(batch_idx, int) and isinstance(y_idx, int) and isinstance(x_idx, int):
                return self._compute_jacobian(y_idx, x_idx, batch_idx=batch_idx)
            elif isinstance(batch_idx, slice) and isinstance(y_idx, slice) and isinstance(x_idx, slice):
                # 切片行
                return self._compute_jacobian(0, 0, row_slice=y_idx)
"""
        core = "result = Jacobian(ys=ys, xs=xs, batch_axis=batch_axis)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


# k


# l
class LabelSmoothRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.label_smooth",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
num_classes = label.size(-1)
prior_dist = prior_dist
if prior_dist is None:
    prior_dist = torch.full((1, num_classes,), 1.0 / num_classes)
result = (1 - epsilon) * label + epsilon * prior_dist
result = result.to(dtype=label.dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class LayerNormRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.layer_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(normalized_shape, int):
    normalized_shape = (normalized_shape,)
elif isinstance(normalized_shape, list):
    normalized_shape = tuple(normalized_shape)
if weight is not None:
    weight = weight.view(normalized_shape).to(x.dtype)
if bias is not None:
    bias = bias.view(normalized_shape).to(x.dtype)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class LcmRule(BaseRule):
    PADDLE_APIS = ("paddle.lcm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x, y = torch.broadcast_tensors(x, y)
x_abs = torch.abs(x)
y_abs = torch.abs(y)
gcd = torch.gcd(x_abs, y_abs)
lcm = torch.zeros_like(gcd)
nonzero_mask = gcd != 0
lcm[nonzero_mask] = (x_abs[nonzero_mask] * y_abs[nonzero_mask]) // gcd[nonzero_mask]
result = torch.abs(lcm)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=impl.splitlines(),
        )


class LinearRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.linear",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import paddle
if paddle.get_flags("FLAGS_use_accuracy_compatible_kernel")["FLAGS_use_accuracy_compatible_kernel"]:
    weight = weight.T.contiguous()
else:
    # Keep transpose as a view to avoid changing the GEMM layout path.
    weight = weight.T
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class LocalResponseNormRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.local_response_norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
data_format = data_format
if data_format == "NLC":
    x = x.permute(0,2,1)
elif data_format == "NHWC":
    x = x.permute(0,3,1,2)
elif data_format == "NDHWC":
    x = x.permute(0,4,1,2,3)
"""
        core = ()
        post = """
if data_format == "NLC":
    result = result.permute(0,2,1)
elif data_format == "NHWC":
    result = result.permute(0,2,3,1)
elif data_format == "NDHWC":
    result = result.permute(0,2,3,4,1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post.splitlines(),
        )


class LogcumsumexpRule(BaseRule):
    PADDLE_APIS = ("paddle.logcumsumexp",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = x
axis = axis

if axis is None:
    x = x.flatten()
    axis = 0
"""
        core = "result = torch.logcumsumexp(x, dim=axis)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class LogaddexpRule(BaseRule):
    PADDLE_APIS = ("paddle.logaddexp",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def to_float_if_needed(tensor):
    if tensor.dtype in [torch.int32, torch.int64]:
        return tensor.to(torch.float32)
    return tensor
x = to_float_if_needed(x)
y = to_float_if_needed(y)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class LogLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.log_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
label = label.detach()
epsilon = epsilon
result = -label * torch.log(input + epsilon) - (1 - label) * torch.log(1 - input + epsilon)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class LogSoftMaxRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.log_softmax",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(dtype, str):
    dtype = getattr(torch, dtype)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class LogNormalRule(BaseRule):
    PADDLE_APIS = ("paddle.log_normal",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.exp(torch.normal(mean=mean, std=std, size=shape))"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=(),
            core=core,
        )


class LstsqRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.lstsq",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
# if decvice is GPU then only use gels.
current_device = x.device
if(current_device.type == 'cuda'):
    driver = "gels"
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


# m
class MarginCrossEntropyRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.margin_cross_entropy",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
margin1 = margin1
margin2 = margin2
margin3 = margin3
scale = scale
group = group
return_softmax = return_softmax
reduction = reduction
if reduction is None:
    reduction = 'none'

theta = torch.acos(logits)

target_theta = theta[torch.arange(logits.size(0)), label]
modified_theta = margin1 * target_theta + margin2
modified_cos = torch.cos(modified_theta) - margin3

logits_modified = logits.clone()
logits_modified[torch.arange(logits.size(0)), label] = modified_cos

logits_modified *= scale

if return_softmax:
    probs = torch.nn.functional.softmax(logits_modified, dim=1)
loss = torch.nn.functional.cross_entropy(logits_modified, label, reduction=reduction)

if reduction == 'none':
    loss = loss.unsqueeze(-1)

if return_softmax:
    result = loss, probs
else:
    result = loss

"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class MatmulRule(BaseRule):
    PADDLE_APIS = ("paddle.matmul",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if x.dtype != y.dtype:
    if x.dtype.is_complex:
        y = y.to(x.dtype)
    elif y.dtype.is_complex:
        x = x.to(y.dtype)
transpose_x = transpose_x
transpose_y = transpose_y
if transpose_x == True and x.dim() >=2:
    x = x.transpose(-2, -1)
if transpose_y == True and y.dim() >=2:
    y = y.transpose(-2, -1)
"""
        core = f"result = {self.torch_api}(x, y)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
            core=core,
        )


class MatrixTransposeRule(BaseRule):
    PADDLE_APIS = (
        "paddle.linalg.matrix_transpose",
        "paddle.matrix_transpose",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = x.transpose(-1, -2)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class MatrixRankRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.matrix_rank",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if atol is not None and rtol is not None:
    if isinstance(rtol, float):
        rtol = torch.tensor(rtol)
    if isinstance(atol, float):
        atol = torch.tensor(atol)
"""
        core = """
if tol is None:
    result = torch.linalg.matrix_rank(input=x, hermitian=hermitian, atol=atol, rtol=rtol)
else:
    result = torch.linalg.matrix_rank(input=x, tol=tol, hermitian=hermitian)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class MatmulTensorRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.matmul",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = r"""
if transpose_x == True:
    x = x.transpose(-1, -2)
if transpose_y == True:
    y = y.transpose(-1, -2)

def extract_bit(dtype):
    import re
    s = str(dtype)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else 0

x_bits = extract_bit(x.dtype)
y_bits = extract_bit(y.dtype)
target_dtype = x.dtype if x_bits >= y_bits else y.dtype
x, y = x.to(target_dtype), y.to(target_dtype)

"""
        core = "result = x.matmul(y)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class MaskedScatterRule(BaseRule):
    PADDLE_APIS = (
        "paddle.masked_scatter",
        "paddle.Tensor.masked_scatter",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            core=core,
        )


class MaxoutRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.maxout",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
axis = axis
axis = axis if axis >= 0 else x.dim() + axis
in_channels = x.shape[axis]
channels_per_group = in_channels // groups
shape = list(x.shape)
new_shape = shape[:axis] + [channels_per_group, groups] + shape[axis+1:]

x = x.reshape(*new_shape)
result = x.max(dim=axis+1).values
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class MedianRule(BaseRule):
    PADDLE_APIS = (
        "paddle.median",
        "paddle.Tensor.median",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
axis = axis
keepdim = keepdim
mode = mode
if axis is None:
    x_flat = x.flatten()
    length = x_flat.numel()
    if length % 2 == 0 and mode == 'avg':
        sorted_x = torch.sort(x_flat, stable=True).values
        mid = length // 2
        median = (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        median = torch.median(x_flat)
    if keepdim:
        median = median.reshape([1] * x.ndim)
else:
    if mode == 'avg':
        length = x.shape[axis] if x.ndim > 0 else 1
        if length % 2 == 0:
            sorted_x = torch.sort(x, dim=axis, stable=True).values
            mid = length // 2
            median = (sorted_x.index_select(axis, torch.tensor([mid - 1])) +
                      sorted_x.index_select(axis, torch.tensor([mid]))) / 2
            if not keepdim:
                median = median.squeeze(axis)
        else:
            median = torch.median(x, dim=axis, keepdim=keepdim).values
    else:
        median = torch.median(x, dim=axis, keepdim=keepdim)
if mode == 'avg' and x.dtype != torch.float64:
    median = median.to(torch.float32)
result = median
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class MultiplexRule(BaseRule):
    PADDLE_APIS = ("paddle.multiplex",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
input = inputs
index = index
temp = []
for i in range(index.shape[0]):
    j = index[i].item()
    temp.append(input[j][i])
result = torch.stack(temp)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class MaskedMultiheadAttentionRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.masked_multihead_attention",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
from typing import Optional

def masked_multihead_attention(
    x: torch.Tensor,
    cache_kv: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    src_mask: Optional[torch.Tensor] = None,
    cum_offsets: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[torch.Tensor] = None,
    rotary_tensor: Optional[torch.Tensor] = None,
    beam_cache_offset: Optional[torch.Tensor] = None,
    qkv_out_scale: Optional[torch.Tensor] = None,
    out_shift: Optional[torch.Tensor] = None,
    out_smooth: Optional[torch.Tensor] = None,
    seq_len: int = 1,
    rotary_emb_dims: int = 0,
    use_neox_rotary_style: bool = False,
    compute_dtype: str = 'default',
    out_scale: float = -1.0,
    quant_round_type: int = 1,
    quant_max_bound: float = 127.0,
    quant_min_bound: float = -127.0
):
    # Infer dimensions from input
    _, batch_size, num_head, max_seq_len, head_dim = cache_kv.shape
    # Reshape and split QKV: [batch_size, 3 * num_head * head_dim] -> [batch_size, 3, num_head, head_dim]
    x = x.view(batch_size, 3, num_head, head_dim)
    q, k, v = x[:, 0], x[:, 1], x[:, 2]  # Each is [batch_size, num_head, head_dim]
    # Apply bias if provided
    if bias is not None:
        q = q + bias[0]
        k = k + bias[1]
        v = v + bias[2]
    # Apply QKV quantization if qkv_out_scale is provided
    if qkv_out_scale is not None:
        q = q / qkv_out_scale[0]
        k = k / qkv_out_scale[1]
        v = v / qkv_out_scale[2]
        # Apply quantization
        if quant_round_type == 1:
            q = torch.round(q).clamp(quant_min_bound, quant_max_bound)
            k = torch.round(k).clamp(quant_min_bound, quant_max_bound)
            v = torch.round(v).clamp(quant_min_bound, quant_max_bound)
    # Handle rotary embeddings
    if rotary_tensor is not None and rotary_emb_dims > 0:
        # Apply rotary embeddings to q and k
        def apply_rotary_emb(x, rotary):
            if use_neox_rotary_style:
                # Neox-style: split head_dim into pairs and apply rotation
                half_dim = rotary_emb_dims // 2
                x1, x2 = x[..., :half_dim], x[..., half_dim:2 * half_dim]
                rot1, rot2 = rotary[..., :half_dim], rotary[..., half_dim:2 * half_dim]
                x_rot = torch.cat((-x2 * rot2 + x1 * rot1, x1 * rot2 + x2 * rot1), dim=-1)
                return torch.cat((x_rot, x[..., 2 * half_dim:]), dim=-1)
            else:
                # Standard rotary: apply cosine and sine rotations
                cos, sin = rotary[..., ::2], rotary[..., 1::2]
                x1, x2 = x[..., ::2], x[..., 1::2]
                x_rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
                return x_rot
        q = apply_rotary_emb(q, rotary_tensor.squeeze(2))
        k = apply_rotary_emb(k, rotary_tensor.squeeze(2))
    # Prepare key and value with cache
    if cache_kv is not None:
        cache_k, cache_v = cache_kv[0], cache_kv[1]  # [batch_size, num_head, max_seq_len, head_dim]
        # Concatenate new k, v to cache
        k = torch.cat((cache_k, k.unsqueeze(-2)), dim=2)  # Add seq_len dim
        v = torch.cat((cache_v, v.unsqueeze(-2)), dim=2)
        cache_kvs_out = torch.stack((k, v), dim=0)
    else:
        k = k.unsqueeze(-2)  # [batch_size, num_head, 1, head_dim]
        v = v.unsqueeze(-2)
        cache_kvs_out = torch.stack((k, v), dim=0)
    # Reshape for attention: [batch_size, num_head, seq_len, head_dim]
    q = q.unsqueeze(2)  # [batch_size, num_head, 1, head_dim]
    seq_len_kv = k.shape[2]
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [batch_size, num_head, 1, seq_len_kv]
    # Apply source mask
    if src_mask is not None:
        expected_mask_shape = (batch_size, 1, 1, seq_len_kv)
        if src_mask.shape[-1] != seq_len_kv:
            # Pad src_mask with zeros to match seq_len_kv
            padding = torch.zeros(
                batch_size, 1, 1, seq_len_kv - src_mask.shape[-1],
                device=src_mask.device, dtype=src_mask.dtype
            )
            src_mask = torch.cat((src_mask, padding), dim=-1)
        attn_scores = attn_scores + src_mask  # Broadcasting applies mask
    # Apply sequence lengths if provided
    if sequence_lengths is not None:
        mask = torch.arange(seq_len_kv, device=x.device).expand(batch_size, seq_len_kv)
        mask = mask < sequence_lengths.squeeze(-1).unsqueeze(-1)
        mask = mask[:, None, None, :]  # [batch_size, 1, 1, seq_len_kv]
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    # Softmax to get attention weights
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    # Compute attention output
    attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_head, 1, head_dim]
    attn_output = attn_output.squeeze(-2)  # [batch_size, num_head, head_dim]
    # Reshape output: [batch_size, num_head * head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_head * head_dim)
    # Apply output quantization
    if out_scale > 0:
        attn_output = attn_output / out_scale
        if out_shift is not None:
            attn_output = attn_output + out_shift
        if out_smooth is not None:
            attn_output = attn_output * out_smooth
        if quant_round_type == 1:
            attn_output = torch.round(attn_output).clamp(quant_min_bound, quant_max_bound)
    # Handle beam_cache_offset
    beam_cache_offset_out = beam_cache_offset
    if beam_cache_offset is not None:
        # In Paddle, beam_cache_offset is typically updated in beam search; here we return as-is
        # If specific updates are needed, they should be implemented based on model logic
        pass
    # Return based on beam_cache_offset presence
    if beam_cache_offset is not None:
        return attn_output, cache_kvs_out, beam_cache_offset_out
    return attn_output, cache_kvs_out, None
"""
        core = "result = masked_multihead_attention(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class MmRule(BaseRule):
    PADDLE_APIS = ("paddle.mm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if input.dtype == torch.float16 and mat2.dtype != torch.float16:
    input = input.to(torch.float32)
if mat2.dtype == torch.float16 and input.dtype != torch.float16:
    mat2 = mat2.to(torch.float32)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class MoePermuteRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.moe_permute",)

    """Expert-wise token permutation with selectable Torch or TE reference.

    ``PADDLEAPITEST_IMPL=torch`` (the default) preserves the shape-generic
    Torch composition; ``PADDLEAPITEST_IMPL=te`` uses TE's mask/padded path.
    """

    SUPPORTED_IMPLEMENTATIONS = frozenset({"te", "torch"})
    DEFAULT_IMPLEMENTATION = "torch"

    def apply(self, paddle_api: str) -> ConvertResult:
        impl, core = self.build_implementation_code()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
            workspace_required=impl == "torch",
        )

    @staticmethod
    def _te_code() -> str:
        return """
from transformer_engine.pytorch.permutation import moe_permute_and_pad_with_probs

if hidden_states.dtype != torch.bfloat16 or scale is not None:
    raise ValueError('TE reference is limited to BF16 hidden_states with scale=None')
if not do_gather:
    raise ValueError('TE reference requires do_gather=True')
if using_ue8m0_scale:
    raise ValueError('TE reference does not support ue8m0 scale')
if return_expert_indices:
    raise ValueError('TE reference does not expose expert_indices')
if override_buffer_size != -1:
    raise ValueError('TE reference does not support override_buffer_size')

_tokens, _topk = expert_routemap_topk.shape
_dev = hidden_states.device
_route = expert_routemap_topk.to(torch.int32)
_prob = expert_prob_topk.to(torch.float32)
_dense_route = torch.zeros((_tokens, num_experts), dtype=torch.int32, device=_dev)
_dense_prob = torch.zeros((_tokens, num_experts), dtype=torch.float32, device=_dev)
_rows = torch.arange(_tokens, device=_dev)
for _k in range(_topk):
    _expert = _route[:, _k]
    _valid = _expert >= 0
    _dense_route[_rows[_valid], _expert[_valid]] = 1
    _dense_prob[_rows[_valid], _expert[_valid]] = _prob[:, _k][_valid]

_counts = torch.as_tensor(tokens_per_expert, dtype=torch.int64, device=_dev)
if int(_dense_route.sum().item()) != int((_route >= 0).sum().item()):
    raise ValueError('TE dense route cannot represent duplicate token/expert assignments')
if not torch.equal(_dense_route.sum(dim=0).to(torch.int64), _counts):
    raise ValueError('tokens_per_expert must exactly match expert_routemap_topk')
_out, _te_probs, _te_map, _pad_offsets, _target = moe_permute_and_pad_with_probs(
    hidden_states, _dense_prob, _dense_route, _counts, padding_alignment
)

# TE stores unpadded expert-local rows in its map. Paddle exposes padded rows.
_n = _te_map[:, 2 * num_experts].to(torch.int64)
_valid = torch.arange(num_experts, device=_dev).unsqueeze(0) < _n.unsqueeze(1)
_te_rows = _te_map[:, :num_experts].to(torch.int64)
_te_experts = _te_map[:, num_experts : 2 * num_experts].to(torch.int64)
if _pad_offsets is not None:
    _padded_rows = _te_rows.clone()
    _padded_rows[_valid] += _pad_offsets[_te_experts[_valid]].to(torch.int64)
else:
    _padded_rows = _te_rows
_paddle_rowmap = torch.full(
    (_tokens, num_experts), -1, dtype=torch.int32, device=_dev
)
_token_ids = torch.arange(_tokens, device=_dev).unsqueeze(1).expand(-1, num_experts)
_paddle_rowmap[_token_ids[_valid], _te_experts[_valid]] = _padded_rows[_valid].to(torch.int32)
_scale_out = torch.empty(0, dtype=torch.float32, device=_dev)
result = (_out, _paddle_rowmap, _te_probs, _scale_out)
"""

    @staticmethod
    def _torch_code() -> str:
        return """
# FP8 zeros, advanced indexing, and assignment are supported on CUDA. Keep FP8
# throughout this gather to avoid materializing a full BF16 copy of hidden_states.
do_gather = do_gather
using_ue8m0_scale = using_ue8m0_scale
return_expert_indices = return_expert_indices
seqlen, token_dim = hidden_states.shape
topk = expert_routemap_topk.shape[1]
_dev = hidden_states.device

# padded slots per expert: identical to kernel's InferMeta formula
_tpe = torch.tensor(tokens_per_expert, dtype=torch.int64, device=_dev)
_padded = ((_tpe + padding_alignment - 1) // padding_alignment * padding_alignment)
total_rows = int(_padded.sum().item())
_offsets = torch.zeros(num_experts, dtype=torch.int64, device=_dev)
_offsets[1:] = torch.cumsum(_padded[:-1], dim=0)

# --- Vectorized permute fully on GPU (Megatron-style argsort + index_select) ---
_routemap = expert_routemap_topk.detach().to(torch.int64)  # (seqlen, topk) on _dev
_prob = expert_prob_topk.detach()  # (seqlen, topk) on _dev

# Flatten all (token, topk_col) assignments
_token_ids = torch.arange(seqlen, dtype=torch.int64, device=_dev).unsqueeze(1).expand(-1, topk).reshape(-1)
_expert_flat = _routemap.reshape(-1)  # (seqlen*topk,)
_prob_flat = _prob.reshape(-1)  # (seqlen*topk,)
_topk_cols = torch.arange(topk, dtype=torch.int64, device=_dev).unsqueeze(0).expand(seqlen, -1).reshape(-1)

# Filter invalid (negative) expert assignments
_valid = _expert_flat >= 0
_token_ids = _token_ids[_valid]
_expert_flat = _expert_flat[_valid]
_prob_flat = _prob_flat[_valid]
_topk_cols = _topk_cols[_valid]

# Sort by (expert_id, token_idx, topk_col) to replicate kernel's sequential scan order
_sort_keys = _expert_flat * (seqlen * topk) + _token_ids * topk + _topk_cols
_order = torch.argsort(_sort_keys, stable=True)
_token_ids = _token_ids[_order]
_expert_sorted = _expert_flat[_order]
_prob_flat = _prob_flat[_order]
del _sort_keys, _order, _expert_flat, _topk_cols, _valid

# Deduplicate: keep only first (token, expert) pair
_pair_key = _token_ids * num_experts + _expert_sorted
_shift = torch.ones(len(_pair_key), dtype=torch.bool, device=_dev)
_shift[1:] = _pair_key[1:] != _pair_key[:-1]
_token_ids = _token_ids[_shift]
_expert_sorted = _expert_sorted[_shift]
_prob_flat = _prob_flat[_shift]
del _pair_key, _shift

# Per-expert capacity capping & slot assignment
_boundaries = torch.searchsorted(_expert_sorted.contiguous(), torch.arange(num_experts + 1, dtype=torch.int64, device=_dev))

# Vectorized slot numbering within each expert using cumcount trick
# _within_expert_idx[i] = position of entry i within its expert group (0-based)
_expert_start = _boundaries[_expert_sorted]  # start boundary for each entry's expert
_within_expert_idx = torch.arange(len(_expert_sorted), dtype=torch.int64, device=_dev) - _expert_start
# Capacity mask: only keep entries with slot < tokens_per_expert[expert]
_caps = _tpe[_expert_sorted]
_cap_mask = _within_expert_idx < _caps
_token_ids = _token_ids[_cap_mask]
_expert_sorted = _expert_sorted[_cap_mask]
_prob_flat = _prob_flat[_cap_mask]
_within_expert_idx = _within_expert_idx[_cap_mask]
del _boundaries, _expert_start, _caps, _cap_mask

# Compute output row for each kept entry
_row_indices = _offsets[_expert_sorted] + _within_expert_idx  # int64

# Build rowmap (seqlen, num_experts): rowmap[token, expert] = output row
rowmap = torch.full((seqlen, num_experts), -1, dtype=torch.int32, device=_dev)
rowmap[_token_ids, _expert_sorted] = _row_indices.to(torch.int32)

# Build gather arrays
_gather_src = torch.full((total_rows,), -1, dtype=torch.int64, device=_dev)
_gather_prob = torch.zeros(total_rows, dtype=torch.float32, device=_dev)
_gather_expert_id = torch.full((total_rows,), -1, dtype=torch.int32, device=_dev)
_gather_src[_row_indices] = _token_ids
_gather_prob[_row_indices] = _prob_flat
_gather_expert_id[_row_indices] = _expert_sorted.to(torch.int32)

token_prob_unzipped = _gather_prob
scale = scale
if scale is None:
    scale_unzipped = torch.empty(0, dtype=torch.float32, device=_dev)
else:
    if using_ue8m0_scale:
        scale_unzipped = torch.zeros(total_rows, scale.shape[1], dtype=scale.dtype, device=_dev)
    else:
        scale_unzipped = torch.zeros(total_rows, scale.shape[1], dtype=torch.float32, device=_dev)

if do_gather:
    hidden_states_unzipped = torch.zeros(total_rows, token_dim, dtype=hidden_states.dtype, device=_dev)
    # Advanced indexing materializes the gathered payload before assignment.
    # Bound that temporary while leaving routing fully vectorized.
    _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
    _payload_bytes_per_row = max(1, token_dim * hidden_states.element_size())
    _payload_chunk = max(1, _workspace_bytes // _payload_bytes_per_row)
    with torch.no_grad():
        for _item_start in range(0, _row_indices.numel(), _payload_chunk):
            _item_end = min(_row_indices.numel(), _item_start + _payload_chunk)
            _dst_rows = _row_indices[_item_start:_item_end]
            _src_rows = _token_ids[_item_start:_item_end]
            hidden_states_unzipped[_dst_rows] = hidden_states[_src_rows]
            if scale is not None:
                if using_ue8m0_scale:
                    scale_unzipped[_dst_rows] = scale[_src_rows]
                else:
                    scale_unzipped[_dst_rows] = scale[_src_rows].to(torch.float32)
else:
    hidden_states_unzipped = torch.empty(0, token_dim, dtype=hidden_states.dtype, device=_dev)
    if scale is not None:
        with torch.no_grad():
            scale_unzipped[_row_indices] = scale[_token_ids].to(scale_unzipped.dtype)
if return_expert_indices:
    expert_indices_out = _gather_expert_id
    result = (hidden_states_unzipped, rowmap, token_prob_unzipped, scale_unzipped, expert_indices_out)
else:
    result = (hidden_states_unzipped, rowmap, token_prob_unzipped, scale_unzipped)
"""


class MoeUnpermuteRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.moe_unpermute",)

    """Expert output merge with selectable Torch or TE reference.

    ``PADDLEAPITEST_IMPL=torch`` (the default) preserves the existing Torch
    implementation; ``PADDLEAPITEST_IMPL=te`` converts Paddle's row map to
    TE's mask map and calls Transformer Engine.
    """

    SUPPORTED_IMPLEMENTATIONS = frozenset({"te", "torch"})
    DEFAULT_IMPLEMENTATION = "torch"

    def apply(self, paddle_api: str) -> ConvertResult:
        impl, core = self.build_implementation_code()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
            workspace_required=impl == "torch",
        )

    @staticmethod
    def _te_code() -> str:
        return """
from transformer_engine.pytorch.permutation import moe_unpermute

if hidden_states_unzipped.dtype != torch.bfloat16:
    raise ValueError('TE reference requires BF16 hidden_states_unzipped')
_dev = hidden_states_unzipped.device
_tokens, _topk = expert_routemap_topk.shape
_rowmap = zipped_expertwise_rowmap.to(torch.int64)
_valid = _rowmap >= 0
_slot = _valid.to(torch.int64).cumsum(dim=1) - 1
_te_map = torch.zeros((_tokens, 2 * num_experts + 1), dtype=torch.int32, device=_dev)
_te_map[:, :num_experts].fill_(-1)
_te_map[:, num_experts : 2 * num_experts].fill_(-1)
_token_ids = torch.arange(_tokens, device=_dev).unsqueeze(1).expand(-1, num_experts)
_te_map[_token_ids[_valid], _slot[_valid]] = _rowmap[_valid].to(torch.int32)
_expert_ids = torch.arange(num_experts, dtype=torch.int64, device=_dev).expand(_tokens, -1)
_te_map[_token_ids[_valid], num_experts + _slot[_valid]] = _expert_ids[_valid].to(torch.int32)
_te_map[:, 2 * num_experts] = _valid.sum(dim=1).to(torch.int32)

_prob_flat = token_prob_unzipped.to(torch.float32).reshape(-1)
_safe_rows = _rowmap.clamp_min(0)
_dense_probs = torch.zeros((_tokens, num_experts), dtype=torch.float32, device=_dev)
_dense_probs[_valid] = _prob_flat[_safe_rows[_valid]]
if using_weighted_combine:
    # Paddle bypasses weighting for a token routed to one expert.
    _single = _valid.sum(dim=1) == 1
    _dense_probs[_single] = torch.where(_valid[_single], torch.ones_like(_dense_probs[_single]), _dense_probs[_single])
    _merge = _dense_probs
else:
    _merge = None

_zipped_tokens = moe_unpermute(
    hidden_states_unzipped,
    _te_map,
    merging_probs=_merge,
    restore_shape=(total_zipped_tokens, hidden_states_unzipped.shape[1]),
    map_type='mask',
    pad_offsets=None,
)
_zipped_probs = torch.zeros((_tokens, _topk), dtype=torch.float32, device=_dev)
_route = expert_routemap_topk.to(torch.int64)
for _k in range(_topk):
    _expert = _route[:, _k]
    _ok = _expert >= 0
    _zipped_probs[_ok, _k] = _prob_flat[_rowmap[_ok, _expert[_ok]]]
result = (_zipped_tokens, _zipped_probs)
"""

    @staticmethod
    def _torch_code() -> str:
        return """
using_weighted_combine = using_weighted_combine
seqlen = expert_routemap_topk.shape[0]
topk = expert_routemap_topk.shape[1]
token_dim = hidden_states_unzipped.shape[1] if hidden_states_unzipped.dim() > 1 else 1
out_dtype = hidden_states_unzipped.dtype
_dev = hidden_states_unzipped.device

_rowmap = zipped_expertwise_rowmap.detach().to(torch.int64)
_routemap = expert_routemap_topk.detach()
_prob_1d = token_prob_unzipped.detach().float().reshape(-1)
_valid_mask = _rowmap >= 0
_aggreg_counts = _valid_mask.sum(dim=1)
zipped_tokens = torch.empty(seqlen, token_dim, dtype=out_dtype, device=_dev)

# The FP32 accumulator and gathered payload can overlap. Process token rows in
# a 32 GiB workspace and preserve expert-column accumulation order per token.
_workspace_bytes = _adaptive_workspace_bytes(torch, locals())
_bytes_per_token = max(1, token_dim * (4 * 2 + out_dtype.itemsize))
_token_chunk = max(1, min(seqlen, _workspace_bytes // _bytes_per_token))
with torch.no_grad():
    for _token_start in range(0, seqlen, _token_chunk):
        _token_end = min(seqlen, _token_start + _token_chunk)
        _rowmap_chunk = _rowmap[_token_start:_token_end]
        _acc = torch.zeros(
            _token_end - _token_start, token_dim, dtype=torch.float32, device=_dev
        )
        for _expert_col in range(num_experts):
            _rows = _rowmap_chunk[:, _expert_col]
            _valid = _rows >= 0
            if not _valid.any():
                continue
            _source_rows = _rows[_valid]
            _gathered = hidden_states_unzipped[_source_rows].to(torch.float32)
            if using_weighted_combine:
                _gathered.mul_(_prob_1d[_source_rows].unsqueeze(1))
            _valid_tokens = _valid.nonzero(as_tuple=True)[0]
            _acc.index_add_(0, _valid_tokens, _gathered)
            del _source_rows, _gathered, _valid_tokens

        # A token routed to one expert bypasses weighting in the Paddle kernel.
        _single = _aggreg_counts[_token_start:_token_end] == 1
        if _single.any():
            _single_columns = _valid_mask[_token_start:_token_end][_single].to(
                torch.int64
            ).argmax(dim=1)
            _single_rows = _rowmap_chunk[_single, _single_columns]
            _acc[_single] = hidden_states_unzipped[_single_rows].to(torch.float32)
        zipped_tokens[_token_start:_token_end] = _acc.to(out_dtype)
        del _rowmap_chunk, _acc

# --- Vectorized zipped_probs: advanced indexing ---
_flat_token_ids = torch.arange(seqlen, dtype=torch.int64, device=_dev).unsqueeze(1).expand(-1, topk).reshape(-1)
_flat_experts = _routemap.reshape(-1)
_flat_k = torch.arange(topk, dtype=torch.int64, device=_dev).unsqueeze(0).expand(seqlen, -1).reshape(-1)

# Valid: expert >= 0
_ve = _flat_experts >= 0
_vt = _flat_token_ids[_ve]
_vexp = _flat_experts[_ve].to(torch.int64)
_vk = _flat_k[_ve]
# Lookup row from rowmap
_vrows = _rowmap[_vt, _vexp]
# Further filter: row >= 0
_vr = _vrows >= 0
_vt = _vt[_vr]
_vk = _vk[_vr]
_vrows = _vrows[_vr]

zipped_probs = torch.zeros(seqlen, topk, dtype=torch.float32, device=_dev)
with torch.no_grad():
    zipped_probs[_vt, _vk] = _prob_1d[_vrows]
result = (zipped_tokens, zipped_probs)
"""


# n
class NanmedianRule(BaseRule):
    PADDLE_APIS = ("paddle.nanmedian",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
axis = axis
keepdim = keepdim
mode = mode


def single_axis_nanmedian(x, axis, keepdim, mode):
    if mode == "avg":
        valid_mask = ~torch.isnan(x)
        if x.ndim == 0:
            valid_x = x.masked_select(valid_mask).reshape(1)
            length = valid_x.numel()
        else:
            new_shape = []
            for i, s in enumerate(x.shape):
                if i != axis:
                    new_shape.append(s)
                else:
                    new_shape.append(-1)
            valid_x = x.masked_select(valid_mask).reshape(*new_shape)
            length = valid_x.shape[axis]
        if length % 2 == 0:
            sorted_x = torch.sort(valid_x, dim=axis).values
            non_nan_mask = ~torch.isnan(sorted_x)
            new_shape_sorted = []
            for i, s in enumerate(sorted_x.shape):
                if i != axis:
                    new_shape_sorted.append(s)
                else:
                    new_shape_sorted.append(-1)
            sorted_x = sorted_x.masked_select(non_nan_mask).reshape(*new_shape_sorted)
            mid = length // 2
            median = (
                sorted_x.index_select(axis, torch.tensor([mid - 1]))
                + sorted_x.index_select(axis, torch.tensor([mid]))
            ) / 2
            if not keepdim:
                median = median.squeeze(axis)
        else:
            median = torch.nanmedian(x, dim=axis, keepdim=keepdim).values
    else:
        median = torch.nanmedian(x, dim=axis, keepdim=keepdim)
    return median


if axis is None:
    x_flat = x.flatten()
    valid_mask = ~torch.isnan(x_flat)
    valid_x = x_flat[valid_mask]
    length = valid_x.numel()
    if length % 2 == 0 and mode == "avg":
        sorted_x = torch.sort(valid_x).values
        mid = length // 2
        median = (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        median = torch.nanmedian(x_flat)
    if keepdim:
        median = median.reshape([1] * x.ndim)
elif isinstance(axis, int):
    median = single_axis_nanmedian(x, axis, keepdim, mode)
else:
    axes = []
    for ax in axis:
        axes.append(ax % x.ndim)
    non_axes = []
    for i in range(x.ndim):
        if i not in axes:
            non_axes.append(i)
    perm = non_axes + list(axes)
    x_permuted = x.permute(perm)
    non_axes_shape = []
    for i in non_axes:
        non_axes_shape.append(x.shape[i])
    flattened_size = 1
    for ax in axes:
        flattened_size *= x.shape[ax]
    new_shape = non_axes_shape + [flattened_size]
    x_flat = x_permuted.reshape(new_shape)
    median = single_axis_nanmedian(x_flat, -1, False, mode)
    if mode == "min":
        median = median.values
    if keepdim:
        output_shape = []
        for i in range(x.ndim):
            if i in axes:
                output_shape.append(1)
            else:
                output_shape.append(x.shape[i])
        median = median.reshape(output_shape)
result = median
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class NpairlossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.npair_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
l2_reg = l2_reg

l2_loss = (anchor.pow(2).sum(dim=1) + positive.pow(2).sum(dim=1)).mean() * 0.25 * l2_reg

sim_matrix = torch.matmul(anchor, positive.T)

labels = labels.view(-1, 1)  # shape: [N, 1]
mask = labels.eq(labels.T).float()

sim_matrix = torch.nn.functional.log_softmax(sim_matrix, dim=1)

loss_ce = -torch.sum(sim_matrix * mask, dim=1)
loss_ce = loss_ce.mean()

result = loss_ce + l2_loss
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class NmsRule(BaseRule):
    PADDLE_APIS = ("paddle.vision.ops.nms",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
import torchvision
class scores_pair:
    def __init__(self, scores, index):
        self.scores = scores
        self.index = index
scores = scores
top_k = top_k
category_idxs = category_idxs
category = categories
iou_threshold = iou_threshold

# 没有scores时自行生成scores
if scores is None:
    scores = torch.arange(1,0,(0.-1.)/boxes.shape[0])
    scores = scores[:boxes.shape[0]]

# 存在category时, 按照类别进行nms
if category_idxs is not None:
    result = []
    for cls in category:
        sele = []
        for i in range(len(category_idxs)):
            if category_idxs[i] == cls:
                sele.append(i)
        box = boxes.index_select(0, torch.tensor(sele))
        score = scores.index_select(0, torch.tensor(sele))
        result.append(torchvision.ops.nms(box, score, iou_threshold))
    result = torch.concat(result)
else:
    result = torchvision.ops.nms(boxes, scores, iou_threshold)

# 对结果从大到小进行排序输出
ind = []
scores = scores.index_select(0,result)
for j in range(scores.numel()):
    tmp = scores_pair(scores[j], j)
    ind.append(tmp)
ind = sorted(ind, key = lambda x : x.scores, reverse = True)
for j in range(len(ind)):
    ind[j] = ind[j].index
result = result.index_select(0, torch.tensor(ind))
if top_k is not None:
    result = result[:top_k]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class NormRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.norm",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if p == "fro" and x.dim() == 1:
    p = 2
"""
        core = """
import math
if p==0:
    if keepdim:
        result = (x!= 0).sum(dim=axis, keepdim=True).to(x.dtype)
    else:
        result = (x!= 0).sum(dim=axis).to(x.dtype)
elif len(x.shape)>=2 and axis is None:
    if p==math.inf:
        if keepdim:
            result = x.abs().amax().reshape([1] * x.ndim)
        else:
            result = x.abs().amax()
    elif p==-math.inf:
        if keepdim:
            result = x.abs().amin().reshape([1] * x.ndim)
        else:
            result = x.abs().amin()
    else:
        flattened_order = 2 if p == "fro" else p
        result = torch.linalg.norm(input=x.flatten(), ord=flattened_order)
        if keepdim:
            result = result.reshape([1] * x.ndim)
else:
    result = torch.linalg.norm(input=x, ord=p, dim=axis, keepdim=keepdim)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class NumelRule(BaseRule):
    PADDLE_APIS = ("paddle.numel",)

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
num_elements = x.numel()
result = torch.tensor(num_elements, dtype=torch.int64)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=impl.splitlines(),
        )


class NonzeroRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.__nonzero__",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.__gt__(0).item()"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class NormalRule(BaseRule):
    PADDLE_APIS = ("paddle.normal",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
mean = mean * 1.
std = std * 1.
shape = shape
if isinstance(mean,torch.Tensor) or isinstance(std,torch.Tensor):
    if (isinstance(mean,torch.Tensor) and torch.is_complex(mean)) or (isinstance(std,torch.Tensor) and torch.is_complex(std)):
        if isinstance(mean,torch.Tensor) and not torch.is_complex(mean):
            mean = torch.complex(mean,torch.zeros_like(mean))
        if isinstance(std,torch.Tensor) and not torch.is_complex(std):
            std = torch.complex(std,torch.zeros_like(std))
    elif isinstance(mean, complex) or isinstance(std, complex):
        if isinstance(mean,torch.Tensor) and not torch.is_complex(mean):
            mean = torch.complex(mean,torch.zeros_like(mean))
        if isinstance(std,torch.Tensor) and not torch.is_complex(std):
            std = torch.complex(std,torch.zeros_like(std))
else:
    if isinstance(mean, complex) or isinstance(std, complex):
        if not isinstance(mean, complex):
            mean = complex(mean)
        if not isinstance(std, complex):
            std = complex(std)
"""
        core = """
if isinstance(mean,torch.Tensor) or isinstance(std,torch.Tensor):
    if (isinstance(mean,torch.Tensor) and torch.is_complex(mean)) or (isinstance(std,torch.Tensor) and torch.is_complex(std)):
        result = torch.complex(torch.normal(mean.real, std.real),torch.normal(mean.imag,std.imag))
    elif isinstance(mean, complex) or isinstance(std, complex):
            result = torch.complex(torch.normal(mean.real, std.real),torch.normal(mean.imag,std.imag))
    else:
        result = torch.normal(mean,std)
else:
    if isinstance(mean, complex) or isinstance(std, complex):
         result = torch.complex(torch.normal(mean.real, std.real,shape),torch.normal(mean.imag,std.imag,shape))
    else:
        result = torch.normal(mean,std,shape)

"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


# o


class OnesRule(BaseRule):
    PADDLE_APIS = ("paddle.ones",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dtype = dtype
if isinstance(shape,torch.Tensor):
    if shape.numel() == 1:
        shape = shape.item()
    else:
        li = []
        for i in shape:
            li.append(i.item())
        shape = li
"""
        core = """
if dtype is None:
    result = torch.ones(shape)
else:
    result = torch.ones(shape, dtype=dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class OuterRule(BaseRule):
    PADDLE_APIS = ("paddle.outer",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = x.flatten()
y = y.flatten()
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


# p
class PadRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.pad",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(value, torch.Tensor):
    value = value.item()
if data_format == "NLC":
    x = x.permute(0, 2, 1)
elif data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
elif data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
if isinstance(pad, torch.Tensor):
    li = []
    for i in pad:
        li.append(i.item())
    pad = li
if not pad_from_left_axis:
    num_dims = len(pad) // 2
    new_pad = []
    for i in range(num_dims):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        new_pad = [right, left] + new_pad
    pad = new_pad
elif len(pad) == 2 * x.ndim:
    num_dims = len(pad) // 2
    new_pad = []
    for i in range(num_dims):
        left = pad[2 * i]
        right = pad[2 * i + 1]
        new_pad.insert(0, right)
        new_pad.insert(0, left)
    pad = new_pad
"""
        core = "result = torch.nn.functional.pad(input=x, pad=pad, mode=mode, value=value)"
        postprocess = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
elif data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
elif data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
            core=core,
            postprocess=postprocess,
        )


class PolarRule(BaseRule):
    PADDLE_APIS = ("paddle.polar",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
real = abs * torch.cos(angle)
imag = abs * torch.sin(angle)
result = torch.complex(real, imag)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class PositiveRule(BaseRule):
    PADDLE_APIS = ("paddle.positive",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class TensorToRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.to",)

    def apply(self, paddle_api: str) -> ConvertResult:
        # Paddle 使用 gpu/cpu 设备名，Torch 需要 cuda/cpu；其余参数保持原始顺序。
        core = """
to_args = list(args)
to_kwargs = dict(kwargs)
if to_args and isinstance(to_args[0], str) and to_args[0] == "gpu":
    to_args[0] = "cuda"
elif to_args and isinstance(to_args[0], str) and to_args[0] in {
    "bool", "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "complex64", "complex128",
}:
    # Paddle 的字符串 dtype 不能直接交给 Torch，否则会被解释成设备名。
    to_args[0] = getattr(torch, to_args[0])
if isinstance(to_kwargs.get("device"), str) and to_kwargs["device"] == "gpu":
    to_kwargs["device"] = "cuda"
if to_args or to_kwargs:
    result = x.to(*to_args, **to_kwargs)
else:
    result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class PutAlongAxisRule(BaseRule):
    PADDLE_APIS = (
        "paddle.put_along_axis",
        "paddle.Tensor.put_along_axis",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
input = locals().get("arr", locals().get("x"))
dim = axis
index = indices
src = values
reduce = reduce
if reduce == 'add':
    reduce = 'sum'
if reduce == 'mul':
    reduce = 'prod'
include_self = include_self
broadcast = broadcast

def infer_broadcast_shape(input, index, dim):
    broadcast_shape_list = list(input.shape)
    broadcast_shape_list[dim] = list(index.shape)[dim]
    broadcast_shape = tuple(broadcast_shape_list)
    for i in range(len(input.shape)):
        if input.shape[i] < index.shape[i]:
            # if indices matrix has larger size than arr matrix, do not broadcast.
            return None
    return broadcast_shape

if broadcast == True and 0 not in tuple(index.shape):
    # paddle 在 `0 in indices.shape` 时直接返回 arr 的副本：不广播、不校验 values。
    # 空 index 无法 expand 到 broadcast_shape，这里必须同样短路。
    broadcast_shape = infer_broadcast_shape(input, indices, axis)
    if broadcast_shape:
        index = torch.broadcast_to(index, broadcast_shape)
        src = torch.broadcast_to(src, broadcast_shape)
index = index.to(dtype=torch.int64)
"""
        core = """
if index.numel() == 0:
    result = input.clone()
elif reduce == 'assign':
    result = torch.scatter(input, dim, index, src)
else:
    result = torch.scatter_reduce(input, dim, index, src, reduce, include_self=include_self)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class PoolRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nn.functional.avg_pool1d",
        "paddle.nn.functional.avg_pool2d",
        "paddle.nn.functional.avg_pool3d",
        "paddle.nn.functional.lp_pool1d",
        "paddle.nn.functional.lp_pool2d",
        "paddle.nn.functional.max_pool1d",
        "paddle.nn.functional.max_pool2d",
        "paddle.nn.functional.max_pool3d",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre_1d = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride

def _get_same_padding_1d(input_size, kernel_size, stride):
    if stride is None:
        stride = kernel_size
    output_size = (input_size + stride - 1) // stride
    total_pad = max(0, (output_size - 1) * stride + kernel_size - input_size)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right

if isinstance(padding, str):
    if padding.upper() == "VALID":
        padding = 0
    elif padding.upper() == "SAME":
        input_size = x.shape[2]
        pad_left, pad_right = _get_same_padding_1d(input_size, kernel_size, stride)
        padding = pad_left # 对称填充
        if pad_left != pad_right:  # 非对称填充
            # TODO(zrr1999) maybe mode="replicate"
            x = torch.nn.functional.pad(x, (pad_left, pad_right))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 1:  # [pad]
        padding = tuple(padding)
    elif len(padding) == 2:  # [pad_left, pad_right]
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
"""
        pre_2d = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)

def _get_same_padding_2d(input_size, kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    output_size_h = (input_size[0] + stride[0] - 1) // stride[0]
    output_size_w = (input_size[1] + stride[1] - 1) // stride[1]
    total_pad_h = max(0, (output_size_h - 1) * stride[0] + kernel_size[0] - input_size[0])
    total_pad_w = max(0, (output_size_w - 1) * stride[1] + kernel_size[1] - input_size[1])
    pad_h = (total_pad_h // 2, total_pad_h - total_pad_h // 2)
    pad_w = (total_pad_w // 2, total_pad_w - total_pad_w // 2)
    return pad_h, pad_w

if isinstance(padding, str):
    if padding == "VALID":
        padding = 0
    elif padding == "SAME":
        input_size = (x.shape[2], x.shape[3])
        pad_h, pad_w = _get_same_padding_2d(input_size, kernel_size, stride)
        padding = (pad_h[0], pad_w[0]) # 对称填充
        if pad_h[0] != pad_h[1] or pad_w[0] != pad_w[1]: # 非对称填充
            x = torch.nn.functional.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode="replicate")
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 2: # [pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 4:
        is_all_int = True
        for p in padding:
            if not isinstance(p, int):
                is_all_int = False
                break
        if is_all_int: # [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        else: # Paddle 的 4D 填充格式(NCHW 或 NHWC)
            if data_format == "NCHW":
                pad_top, pad_bottom = padding[2]
                pad_left, pad_right = padding[3]
            else:  # NHWC
                pad_top, pad_bottom = padding[1]
                pad_left, pad_right = padding[2]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        padding = 0
"""
        pre_3d = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
if data_format == 'NDHWC':
    x = x.permute(0, 4, 1, 2, 3)

def _get_same_padding_3d(input_size, kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride,) * 3
    output_size_d = (input_size[0] + stride[0] - 1) // stride[0]
    output_size_h = (input_size[1] + stride[1] - 1) // stride[1]
    output_size_w = (input_size[2] + stride[2] - 1) // stride[2]
    total_pad_d = max(0, (output_size_d - 1) * stride[0] + kernel_size[0] - input_size[0])
    total_pad_h = max(0, (output_size_h - 1) * stride[1] + kernel_size[1] - input_size[1])
    total_pad_w = max(0, (output_size_w - 1) * stride[2] + kernel_size[2] - input_size[2])
    pad_d = (total_pad_d // 2, total_pad_d - total_pad_d // 2)
    pad_h = (total_pad_h // 2, total_pad_h - total_pad_h // 2)
    pad_w = (total_pad_w // 2, total_pad_w - total_pad_w // 2)
    return pad_d, pad_h, pad_w

if exclusive:
    padding = 0

if isinstance(padding, str):
    if padding == "VALID":
        padding = 0
    elif padding == "SAME":
        input_size = (x.shape[2], x.shape[3], x.shape[4])  # (D, H, W)
        pad_d, pad_h, pad_w = _get_same_padding_3d(input_size, kernel_size, stride)
        padding = (pad_d[0], pad_h[0], pad_w[0]) # 对称填充
        if pad_d[0] != pad_d[1] or pad_h[0] != pad_h[1] or pad_w[0] != pad_w[1]: # 非对称填充
            # TODO(zrr1999) maybe mode="replicate"
            x = torch.nn.functional.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 3:  # [pad_depth, pad_height, pad_width]
        max_pad = []
        for i in range(3):
            max_pad.append(kernel_size[i] // 2)
        exceeds_max = False
        for p, m in zip(padding, max_pad):
            if p > m:
                exceeds_max = True
                break
        if exceeds_max:
            pad_d, pad_h, pad_w = padding
            x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
            padding = 0
        else:
            padding = tuple(padding)
    elif len(padding) == 6:  # [front, back, top, bottom, left, right]
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
    elif len(padding) == 5: # Paddle 的 5D 填充格式
        if data_format == "NCDHW":
            pad_front, pad_back = padding[2]
            pad_top, pad_bottom = padding[3]
            pad_left, pad_right = padding[4]
        else: # NDHWC
            pad_front, pad_back = padding[1]
            pad_top, pad_bottom = padding[2]
            pad_left, pad_right = padding[3]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
"""
        core = ()
        post_1d = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        post_2d = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        post_3d = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        if paddle_api.endswith("_pool1d"):
            if paddle_api == "paddle.nn.functional.lp_pool1d":
                pre = (
                    """
if data_format == "NLC":
    x = x.permute(0, 2, 1)
"""
                    + pre_1d
                    + """
if isinstance(padding, int) and padding != 0:
    x = torch.nn.functional.pad(x, (padding, padding))
elif isinstance(padding, tuple):
    x = torch.nn.functional.pad(x, (padding[0], padding[0]))
"""
                )
                post = post_1d
            else:
                pre = pre_1d
                post = ""
        elif paddle_api.endswith("_pool2d"):
            if paddle_api == "paddle.nn.functional.lp_pool2d":
                pre = (
                    pre_2d
                    + """
if isinstance(padding, int) and padding != 0:
    x = torch.nn.functional.pad(x, (padding, padding, padding, padding))
elif isinstance(padding, tuple):
    x = torch.nn.functional.pad(x, (padding[1], padding[1], padding[0], padding[0]))
"""
                )
            else:
                pre = pre_2d
            post = post_2d
        elif paddle_api.endswith("_pool3d"):
            pre = pre_3d
            post = post_3d
        else:
            return ConvertResult.error(paddle_api, f"Unsupported pooling api: {paddle_api}")
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post,
        )


# q
class QuantileRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nanquantile",
        "paddle.quantile",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
import numbers
d = x.dim()
axis0 = axis
if isinstance(axis, int) and axis < 0:
    axis = axis + d
if not isinstance(q,numbers.Number):
    q = torch.tensor(q, dtype = x.dtype)
if isinstance(axis, int) and axis < 0:
    axis = axis + d
if isinstance(axis, list):
    if len(axis) > 1:
        axis_adjusted = []
        for a in axis:
            if a < 0:
                axis_adjusted.append(a + d)
            else:
                axis_adjusted.append(a)
        axis = sorted(axis_adjusted)
        remaining_axes = []
        for i in range(d):
            if i not in axis:
                remaining_axes.append(i)
        permute_order = remaining_axes + axis
        x_perm = x.permute(permute_order)
        shape_list = []
        for i in range(len(remaining_axes), d):
            shape_list.append(x_perm.shape[i])
        merged_dim = int(torch.prod(torch.tensor(shape_list)))
        new_shape = x_perm.shape[:len(remaining_axes)] + (merged_dim,)
        x = x_perm.reshape(*new_shape)
        axis0 = axis
        axis = -1
    elif len(axis) ==1 :
        axis = axis[0]
    else:
        axis = None

"""
        core = ()
        postprocess = """
if keepdim and axis0 != axis:
    if not isinstance(q,numbers.Number) and len(q) >1:
        result = result.unsqueeze(-1)
    else:
        result = result.squeeze(0).unsqueeze(-1)

"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
            postprocess=postprocess,
        )


class QrRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.qr",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
mode = mode
"""
        core = """
result = torch.linalg.qr(x, mode)
"""
        post = """
if mode == "r":
    result = result[1]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post.splitlines(),
        )


# r
class RankRule(BaseRule):
    PADDLE_APIS = (
        "paddle.rank",
        "paddle.Tensor.rank",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.tensor(input.dim(),dtype=torch.int64)"
        post = """
result = result.to(torch.int32)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
            postprocess=post.splitlines(),
        )


class ReduceAsRule(BaseRule):
    PADDLE_APIS = ("paddle.reduce_as",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x_shape = list(x.shape)
t_shape = [1] * (x.dim() - target.dim()) + list(target.shape)
reduce_dims = []
for i, (xs, ts) in enumerate(zip(x_shape, t_shape)):
    if ts == 1 and xs != 1:
        reduce_dims.append(i)
out = x.sum(dim=reduce_dims, keepdim=True)
result = out.view(target.shape)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


# paddle/phi/infermeta/unary.cc ValidateShape 的逐条复刻，reshape 系列共用。
# 归一化后 shape 里不再有 0 或 -1，torch.reshape 只需处理确定形状。
# 三处必须在此拦下的语义（torch 的报错文本不在 classify_runtime_error 白名单里，
# 直接交给 torch 会把无效配置误报成 torch_error）：
#   - shape 里的 0 表示抄输入同位置维度，index 必须落在输入维度内，zero-size 输入例外；
#   - zero-size 输入下 -1 按非零维乘积之比推导，不是 in_size // capacity（那样恒为 0）；
#   - -1 只允许一个，其余负数一律非法。
_RESHAPE_VALIDATE_SHAPE = """
if isinstance(shape, torch.Tensor):
    shape = shape.tolist()
else:
    shape = list(shape)
in_size = x.numel()
in_dims = list(x.shape)
out_shape = [0] * len(shape)
capacity = 1
shape_zero_cnt = 0
unk_dim_idx = -1
for i, s in enumerate(shape):
    if s == -1:
        if unk_dim_idx != -1:
            raise ValueError(
                "(InvalidArgument) Only one dimension value of 'shape' in "
                "ReshapeOp can be -1"
            )
        unk_dim_idx = i
        out_shape[i] = -1
    elif s == 0:
        shape_zero_cnt += 1
        if i < len(in_dims):
            out_shape[i] = 0 if in_size == 0 else in_dims[i]
        elif in_size != 0:
            raise ValueError(
                "(InvalidArgument) If The index of 0 in `shape` >= the input "
                "tensor X's dimensions, It can only be Zero-Sized Tensor"
            )
        capacity *= out_shape[i]
    elif s < 0:
        raise ValueError(
            "(InvalidArgument) Each dimension value of 'shape' in ReshapeOp "
            "must not be negative except one unknown dimension"
        )
    else:
        out_shape[i] = s
        capacity *= s
if capacity == 0 and unk_dim_idx != -1:
    in_zero_cnt = 0
    in_pdt = 1
    for d in in_dims:
        if d == 0:
            in_zero_cnt += 1
        else:
            in_pdt *= d
    shape_pdt = 1
    for s in shape:
        if s != 0 and s != -1:
            shape_pdt *= s
    if shape_zero_cnt == in_zero_cnt and in_pdt % shape_pdt == 0:
        out_shape[unk_dim_idx] = in_pdt // shape_pdt
    else:
        raise ValueError(
            "(InvalidArgument) can not reshape, because the unspecified "
            "dimension can be any number and is ambiguous"
        )
elif unk_dim_idx != -1:
    if in_size % capacity != 0:
        raise ValueError(
            "(InvalidArgument) The 'shape' in ReshapeOp is invalid because "
            "the input size is not divisible by the known dimensions"
        )
    out_shape[unk_dim_idx] = in_size // capacity
elif capacity != in_size:
    raise ValueError(
        "(InvalidArgument) The 'shape' in ReshapeOp is invalid because "
        "the input size does not equal the shape capacity"
    )
shape = out_shape
"""


class ReshapeRule(BaseRule):
    PADDLE_APIS = (
        "paddle.reshape",
        "paddle.Tensor.reshape",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = torch.reshape(x, shape)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=_RESHAPE_VALIDATE_SHAPE,
            core=core,
        )


class ReshapeInplaceRule(BaseRule):
    PADDLE_APIS = (
        "paddle.reshape_",
        "paddle.Tensor.reshape_",
        "paddle._C_ops.reshape_",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        # torch 没有 reshape_ kernel，只能组合。Paddle 侧限制 reshape_ 的三道门禁里，
        # 只有门 1 属于单 API 语义，必须在这里对齐：
        #   门 1 paddle/fluid/eager/utils.cc CheckInplace —— 需要梯度的叶子 Tensor 直接抛
        #        InvalidArgument；但 python/paddle/tensor/manipulation.py 在目标 shape 与
        #        原 shape 相同时先短路返回 x，不进 C++，所以 shape 未变时不该报错。
        #   门 2 inplace version 校验取决于上游算子的反向依赖，单 API 参考实现里没有上游。
        #   门 3 静态图退化为 reshape，与动态图单 API 比对无关。
        core = """
if list(x.shape) == shape:
    # 门 1 的短路口：Paddle 在 Python 层直接返回 x，不触发 CheckInplace
    result = x
else:
    if x.requires_grad and x.is_leaf:
        raise RuntimeError(
            "Leaf Var () that doesn't stop gradient can't use inplace strategy."
        )
    # set_ 不能包 no_grad：包了以后 x 保留旧 grad_fn 而 shape 已变，反向会校验失败。
    # 不包时 set_ 把 grad_fn 置为 NotImplemented，而 harness 只对 x 自身求 grad
    # （outputs 与 inputs 同对象），autograd 直接返回 grad_output，与 Paddle 一致。
    x.set_(torch.reshape(x, shape))
    result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=_RESHAPE_VALIDATE_SHAPE,
            core=core,
        )


class ReverseRule(BaseRule):
    PADDLE_APIS = ("paddle.reverse",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dim = []
if isinstance(axis,int):
    dim.append(axis)
else:
    dim = axis
"""
        core = """
result = torch.flip(x,dim)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class RoiAlignRule(BaseRule):
    PADDLE_APIS = ("paddle.vision.ops.roi_align",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import torchvision
ans = []
end = 0
"""
        core = """
for i in range(boxes_num.shape[0]):
    begin = end
    end = end + int(boxes_num[i])
    ans.append(boxes[begin:end,])
result = torchvision.ops.roi_align(
    input=x,
    boxes=ans,
    output_size=output_size,
    spatial_scale=spatial_scale,
    sampling_ratio=sampling_ratio,
    aligned=aligned,
)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class RoiPoolRule(BaseRule):
    PADDLE_APIS = (
        "paddle.vision.ops.psroi_pool",
        "paddle.vision.ops.roi_pool",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import torchvision
ans = []
end = 0
for i in range(boxes_num.shape[0]):
    begin = end
    end = end + int(boxes_num[i])
    ans.append(boxes[begin:end,])
"""
        core = f"result = {self.torch_api}(input=x, boxes=ans, output_size=output_size, spatial_scale=spatial_scale)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class RollRule(BaseRule):
    PADDLE_APIS = (
        "paddle.roll",
        "paddle.Tensor.roll",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(shifts, torch.Tensor):
    if shifts.numel() == 1:
        shifts = shifts.item()
    else:
        shifts = shifts.tolist()
if isinstance(axis, torch.Tensor):
    if axis.numel() == 1:
        axis = axis.item()
    else:
        axis = axis.tolist()
"""
        if paddle_api == "paddle.roll":
            core = "result = torch.roll(input=x, shifts=shifts, dims=axis)"
        else:
            core = "result = x.roll(shifts=shifts, dims=axis)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class ReduceRule(BaseRule):
    PADDLE_APIS = (
        "paddle.max",
        "paddle.Tensor.max",
        "paddle.mean",
        "paddle.Tensor.mean",
        "paddle.min",
        "paddle.prod",
        "paddle.sum",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(axis, (tuple, list)):
    tmp = []
    for a in axis:
        if torch.is_tensor(a):
            tmp.append(a.item())
        else:
            tmp.append(a)
    axis = tuple(tmp)
if torch.is_tensor(axis):
    if axis.dim() == 0:
        axis = axis.item()
    else:
        axis = tuple(axis.tolist())
"""
        if paddle_api == "paddle.mean":
            core = """
if axis is None:
    result = torch.mean(x)
else:
    result = torch.mean(x, dim=axis, keepdim=keepdim)
"""
            post = """
if axis is None and keepdim:
    result = result.view([1] * x.dim())
"""
        elif paddle_api == "paddle.Tensor.mean":
            preprocess += """
x_dtype = x.dtype
if x_dtype in {torch.int64, torch.int32, torch.bool}:
    x = x.to(torch.float32)
"""
            core = """
if axis is None:
    result = x.mean()
else:
    result = x.mean(dim=axis, keepdim=keepdim)
"""
            post = """
if axis is None and keepdim:
    result = result.view([1] * x.dim())
if x_dtype in {torch.int64, torch.int32, torch.bool}:
    result = result.to(x_dtype)
"""
        elif paddle_api == "paddle.prod":
            preprocess += """
if dtype is None:
    dtype = x.dtype
"""
            core = """
if axis is None:
    result = torch.prod(x, dtype = dtype)
elif isinstance(axis, int):
    result = torch.prod(x, dim=axis, keepdim=keepdim, dtype = dtype)
else:
    for a in axis:
        x = torch.prod(x, dim=a, keepdim=True, dtype=dtype)
    result = x
"""
            post = """
if axis is None and keepdim:
    result = result.view([1] * x.dim())
if isinstance(axis, tuple) and not keepdim:
    result = torch.squeeze(result, dim=axis)
"""
        elif paddle_api == "paddle.sum":
            core = "result = torch.sum(x, dim=axis, keepdim=keepdim, dtype=dtype)"
            post = ""
        elif paddle_api == "paddle.Tensor.max":
            core = """
if axis is None:
    result = torch.max(x)
else:
    _max_result = torch.max(x, dim=axis, keepdim=keepdim)
    result = _max_result.values
"""
            post = ""
        else:
            core = f"result = {self.torch_api}(x, dim=axis, keepdim=keepdim)"
            post = ""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
            postprocess=post,
        )


class RnntLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.rnnt_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
import torchaudio
blank = blank
fastemit_lambda = fastemit_lambda
reduction = reduction
fused_log_softmax = fused_log_softmax

result = torchaudio.functional.rnnt_loss(
        logits=input,
        targets=label,
        logit_lengths=input_lengths,
        target_lengths=label_lengths,
        blank=blank,
        reduction=reduction,
        fused_log_softmax=fused_log_softmax,
    )
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


# s
class SetitemRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.__setitem__",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(value, torch.Tensor) and x.dtype == torch.float32 and value.dtype == torch.bfloat16:
    value = value.to(torch.float32)
"""
        core = "x.__setitem__(item, value)"
        post = "result = x"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=[post],
        )


class SampleNeighborsRule(BaseRule):
    PADDLE_APIS = ("paddle.geometric.sample_neighbors",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
eids = eids
sample_size = sample_size
return_ids = return_eids
out_neighbors = []
out_count = []
out_eids = []
for node in input_nodes:
    start = colptr[node]
    end = colptr[node + 1]
    neighbors = row[start:end]
    num_neighbors = neighbors.numel()
    edge_ids = torch.arange(start,end,dtype=torch.int64)

    if num_neighbors == 0:
        sampled = torch.tensor([], dtype=row.dtype)
        sampled_eids = torch.tensor([], dtype=torch.int64)
    elif sample_size == -1 or num_neighbors <= sample_size:
        sampled = neighbors
        sampled_eids = edge_ids
    else:
        sampled = neighbors[:sample_size]
        sampled_eids = edge_ids[:sample_size]

    out_neighbors.append(sampled)
    out_count.append(sampled.numel())
    out_eids.append(sampled_eids)

out_neighbors = torch.cat(out_neighbors) if out_neighbors else torch.tensor([], dtype=row.dtype)
out_count = torch.tensor(out_count, dtype=torch.int64)
if return_ids:
    out_eids = eids.index_select(0,torch.cat(out_eids)) if out_eids else torch.tensor([], dtype=eids.dtype)

if return_ids:
    result = (out_neighbors, out_count, out_eids)
else:
    result = (out_neighbors, out_count)

"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SegmentMaxRule(BaseRule):
    PADDLE_APIS = ("paddle.geometric.segment_max",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
num = int(segment_ids.max().item()) + 1
ans = torch.full((num,)+data.shape[1:], float('-inf'), dtype = data.dtype)
for idx in range(data.shape[0]):
    seg_id = segment_ids[idx]
    val = data[idx]
    ans[seg_id][val > ans[seg_id]] = val[val > ans[seg_id]]
ans[ans == float('-inf')] = 0
result = ans
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SendURecvRule(BaseRule):
    PADDLE_APIS = ("paddle.geometric.send_u_recv",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
out_size = out_size
src_feat = x[src_index]
D = x.size(1)
if out_size is None or out_size <= 0:
    # out_size = int(dst_index.max()) + 1
    out_size = x.size(0)
result = torch.zeros((out_size, D), dtype=x.dtype, device=x.device) * 1.
if reduce_op == 'sum' or reduce_op == 'mean':
    count = torch.zeros(out_size, dtype=x.dtype, device=x.device)

    for i in range(src_feat.size(0)):
        dst = dst_index[i]
        result[dst] += src_feat[i]
        count[dst] += 1

    if reduce_op == 'mean':
        mask = count > 0
        result[mask] = result[mask] / count[mask].unsqueeze(1)
elif reduce_op == 'max':
    result[:] = float('-inf')
    for i in range(src_feat.size(0)):
        dst = dst_index[i]
        result[dst] = torch.maximum(result[dst], src_feat[i])

    # 若某些 dst 没收到消息，设为 0
    result[result == float('-inf')] = 0
elif reduce_op == 'min':
    result[:] = float('inf')
    for i in range(src_feat.size(0)):
        dst = dst_index[i]
        result[dst] = torch.minimum(result[dst], src_feat[i])

    # 若某些 dst 没收到消息，设为 0
    result[result == float('inf')] = 0
result = result.to(dtype=x.dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SendUERecvRule(BaseRule):
    PADDLE_APIS = ("paddle.geometric.send_ue_recv",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
out_size = out_size
# Determine output size
out_shape = list(x.shape)
if out_size is not None:
    out_shape[0] = out_size
if x.dim() > y.dim():
    y = y.unsqueeze(1)
    y = y.expand(-1,x.shape[1],-1)
dtype = x.dtype

# Get messages from src_index
x_src = x[src_index]

# Apply message operation
if message_op == 'add':
    msg = x_src + y
elif message_op == 'sub':
    msg = x_src - y
elif message_op == 'mul':
    msg = x_src * y
elif message_op == 'div':
    msg = x_src / (y + 1e-12)

out_shape[-1] = msg.shape[-1]

# Reduce operation
if reduce_op in ['sum', 'mean']:
    result = torch.zeros(out_shape, dtype=dtype)
    count = torch.zeros(out_shape[:-1], dtype=dtype)

    for i in range(dst_index.shape[0]):
        dst = dst_index[i].item()
        if dst >= result.shape[0]:
            continue
        result[dst] += msg[i]
        count[dst] += 1

    if reduce_op == 'mean':
        mask = count > 0
        result[mask] = result[mask] / count[mask].unsqueeze(1)

elif reduce_op == "max":
    result = torch.full(out_shape, float('-inf'), dtype=msg.dtype)
    for i in range(dst_index.shape[0]):
        dst = dst_index[i].item()
        if dst >= result.shape[0]:
            continue
        result[dst] = torch.max(result[dst], msg[i])
    result[result == float('-inf')] = 0

elif reduce_op == "min":
    result = torch.full(out_shape, float('inf'), dtype=msg.dtype)
    for i in range(dst_index.shape[0]):
        dst = dst_index[i].item()
        if dst >= result.shape[0]:
            continue
        result[dst] = torch.min(result[dst], msg[i])
    result[result == float('inf')] = 0
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class ScatterRule(BaseRule):
    PADDLE_APIS = ("paddle.scatter",)

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
overwrite = overwrite
x = x.clone()
index = index.view(-1, 1)
try:
    updates = updates.expand_as(x)
except:
    pass
if not overwrite:
    for i in range(index.shape[0]):
        x[index[i]] = torch.zeros_like(x[index[i]])
    for i in range(index.shape[0]):
        x[index[i]] += updates[i]
else:
    for i in range(index.shape[0]):
        x[index[i]] = updates[i]
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=impl.splitlines(),
        )


class ScatterndRule(BaseRule):
    PADDLE_APIS = ("paddle.scatter_nd",)

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
output = torch.zeros(shape, dtype=updates.dtype).to(updates.device)
if index.numel() == 0:
    result = output + updates
else:
    flat_index = index.view(-1, index.size(-1))
    flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
    for i in range(flat_index.size(0)):
        idx_tuple = tuple(flat_index[i])
        output[idx_tuple] += flat_updates[i]
    result = output
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=impl.splitlines(),
        )


class ScatterndaddRule(BaseRule):
    PADDLE_APIS = ("paddle.scatter_nd_add",)

    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = x.clone()
if index.numel() == 0:
    result = x + updates
else:
    flat_index = index.view(-1, index.size(-1))
    flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
    for i in range(flat_index.size(0)):
        idx_tuple = tuple(flat_index[i])
        x[idx_tuple] += flat_updates[i]
    result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=impl.splitlines(),
        )


class SeluRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.selu",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = f"""
if scale == 1.0507009873554804934193349852946 and alpha == 1.6732632423543772848170429916717:
    result = torch.nn.functional.selu(input=x)
else:
    result = scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            core=core,
        )


class SigmoidFocalLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.sigmoid_focal_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
normalizer = normalizer
alpha = alpha
gamma = gamma
reduction = reduction
prob = torch.sigmoid(logit)

pos_loss = -label * alpha * ((1 - prob) ** gamma) * torch.log(prob)
neg_loss = -(1 - label) * (1 - alpha) * (prob ** gamma) * torch.log(1 - prob)
loss = pos_loss + neg_loss

if normalizer is not None:
    loss = loss / normalizer

if reduction == 'mean':
    result = loss.mean()
elif reduction == 'sum':
    result = loss.sum()
elif reduction == 'none':
    result = loss
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SliceRule(BaseRule):
    PADDLE_APIS = (
        "paddle.slice",
        "paddle.Tensor.slice",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(starts, torch.Tensor):
    starts = starts.tolist()
elif isinstance(starts, (list, tuple)):
    new_starts = []
    for s in starts:
        if isinstance(s, torch.Tensor):
            new_starts.append(s.item())
        else:
            new_starts.append(s)
    starts = new_starts
if isinstance(ends, torch.Tensor):
    ends = ends.tolist()
elif isinstance(ends, (list, tuple)):
    new_ends = []
    for e in ends:
        if isinstance(e, torch.Tensor):
            new_ends.append(e.item())
        else:
            new_ends.append(e)
    ends = new_ends
"""
        core = """
for i, dim in enumerate(axes):
    if starts[i] < 0:
        starts[i] = starts[i] + input.shape[dim]
    if ends[i] < 0:
        ends[i] = ends[i] + input.shape[dim]
    starts[i] = max(starts[i], 0)
    starts[i] = min(starts[i], input.shape[dim])
    ends[i] = max(ends[i], 0)
    ends[i] = min(ends[i], input.shape[dim])
    input = torch.narrow(input, dim, starts[i], max(0, ends[i] - starts[i]))
result = input
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=preprocess,
            core=core,
        )


class SplitRule(BaseRule):
    PADDLE_APIS = ("paddle.split",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
axis = axis
if isinstance(axis, torch.Tensor):
    axis =axis.item()
if axis < 0:
    axis = len(x.shape) + axis
if not isinstance(num_or_sections, int):
    num = x.shape[axis]
    for i in num_or_sections:
        if i != -1:
            num = num - i
    for i in range(len(num_or_sections)):
        if num_or_sections[i] == -1:
            num_or_sections[i] = num
            break
"""
        core = """
if isinstance(num_or_sections, int):
    result = torch.tensor_split(x, num_or_sections, dim=axis)
else:
    result = torch.split(x, num_or_sections, dim=axis)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess,
            core=core,
        )


class TensorSplitRule(BaseRule):
    PADDLE_APIS = (
        "paddle.dsplit",
        "paddle.hsplit",
        "paddle.vsplit",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = f"""
if isinstance(num_or_indices, int):
    result = {self.torch_api}(x, sections=num_or_indices)
else:
    result = {self.torch_api}(x, indices=tuple(num_or_indices))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            core=core,
        )


class SquareErrorCostRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.square_error_cost",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = (input - label) ** 2
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SqueezeRule(BaseRule):
    PADDLE_APIS = ("paddle.squeeze",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(axis, torch.Tensor):
    if axis.numel() == 1:
        axis = axis.item()
    else:
        axis = tuple(axis.tolist())
elif isinstance(axis, (list, tuple)):
    axis = tuple(axis)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class SequenceMaskRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.sequence_mask",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
maxlen = maxlen
dtype = dtype
if maxlen is None:
    maxlen = int(x.max().item())
elif isinstance(maxlen, torch.Tensor):
    maxlen = int(maxlen.item())
if maxlen <= 0:
    maxlen = int(x.max().item())
range_row = torch.arange(maxlen, device=x.device)
mask = range_row < x.unsqueeze(-1)
result = mask.to(dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SortRule(BaseRule):
    PADDLE_APIS = (
        "paddle.sort",
        "paddle.Tensor.sort",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
axis = axis if axis >= 0 else x.dim() + axis
"""
        if self.torch_api.startswith("torch.Tensor"):
            core = "result, _ = x.sort(dim=axis, descending=descending, stable=stable)"
        else:
            core = "result, _ = torch.sort(input=x, dim=axis, descending=descending, stable=stable)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class SplitTensorRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.split",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
axis = axis if axis >= 0 else x.dim() + axis
if isinstance(num_or_sections, int):
    num_or_sections = x.shape[axis] // num_or_sections
elif isinstance(num_or_sections, list) and -1 in num_or_sections:
    num_or_sections[num_or_sections.index(-1)] = x.shape[axis] - sum(num_or_sections) - 1
"""
        core = "result = x.split(split_size=num_or_sections, dim=axis)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


class SlogdetRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.slogdet",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = torch.linalg.slogdet(x)
"""
        post = """
result = torch.stack(result,0)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
            postprocess=post.splitlines(),
        )


class SliceScatterRule(BaseRule):
    PADDLE_APIS = ("paddle.slice_scatter",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
slices = [slice(None)] * x.dim()
for i, axis in enumerate(axes):
    slices[axis] = slice(starts[i], ends[i], strides[i])
shape = list(x.shape)
for i, axis in enumerate(axes):
    start, end, stride = starts[i], ends[i], strides[i]
    shape[axis] = (end - start + stride - 1) // stride
if list(value.shape) != shape:
    value = value.expand(shape)
result = x.clone()
result[tuple(slices)] = value
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class StandardGammaRule(BaseRule):
    PADDLE_APIS = ("paddle.standard_gamma",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
rate = torch.ones_like(x)
"""
        core = ()
        post = "result = result.sample()"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=[post],
        )


class StanhRule(BaseRule):
    PADDLE_APIS = ("paddle.stanh",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = "x = x * scale_a"
        core = ()
        post = "result = result * scale_b"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=[pre],
            core=core,
            postprocess=[post],
        )


class StridedSliceRule(BaseRule):
    PADDLE_APIS = ("paddle.strided_slice",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
shape = x.shape
index_list = []
for s in shape:
    index_list.append(torch.arange(s))
for axis, start, end, stride in zip(axes, starts, ends, strides):
    dim_len = shape[axis]
    if start < 0:
        start += dim_len
    if end < 0:
        end += dim_len
    if stride > 0:
        start = min(max(start, 0), dim_len)
        end = min(max(end, 0), dim_len)
    else:
        start = min(max(start, -1), dim_len - 1)
        end = min(max(end, -1), dim_len - 1)
    index_list[axis] = torch.arange(start, end, step=stride)
meshgrid_inputs = []
for i, ind in enumerate(index_list):
    if isinstance(ind, torch.Tensor):
        meshgrid_inputs.append(ind)
    else:
        meshgrid_inputs.append(torch.arange(shape[i]))
grids = torch.meshgrid(*meshgrid_inputs, indexing='ij')
result = x[grids]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class ShardIndexRule(BaseRule):
    PADDLE_APIS = ("paddle.shard_index",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
ignore_value = ignore_value
shard_size = (index_num + nshards - 1) // nshards
lower = shard_id * shard_size
upper = (shard_id + 1) * shard_size

mask = (input >= lower) & (input < upper)
output = torch.full_like(input, ignore_value)

output[mask] = input[mask] - lower
result = output
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class ScaleRule(BaseRule):
    PADDLE_APIS = (
        "paddle.scale",
        "paddle.Tensor.scale",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
if bias_after_scale:
    result = scale * x + bias
else:
    result = scale * (x + bias)
if act is not None:
    if act == 'tanh':
        result = torch.tanh(result)
    elif act == 'sigmoid':
        result = torch.sigmoid(result)
    elif act == 'relu':
        result = torch.relu(result)
    elif act == 'softmax':
        result = torch.softmax(result, dim=-1)
result = result.to(x.dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=(),
            core=core,
        )


class ShapeRule(BaseRule):
    PADDLE_APIS = ("paddle.shape",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.tensor(input.shape, dtype=torch.int64)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class ScaledDotProductAttentionRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.scaled_dot_product_attention",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
query = query.permute(0, 2, 1, 3)
key = key.permute(0, 2, 1, 3)
value = value.permute(0, 2, 1, 3)
if query.shape[1] != key.shape[1]:
    repeat_factor = query.shape[1] // key.shape[1]
    key = key.repeat(1, repeat_factor, 1, 1)
    value = value.repeat(1, repeat_factor, 1, 1)
if is_causal:
    attn_mask = None
"""
        core = ()
        post = "result = result.permute(0, 2, 1, 3)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=[post],
        )


class SubtractRule(BaseRule):
    PADDLE_APIS = ("paddle.subtract",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if x.dtype == torch.bool:
    x = torch.tensor(x, dtype=y.dtype)
elif y.dtype == torch.bool:
    y = torch.tensor(y, dtype=x.dtype)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class SwigluRule(BaseRule):
    PADDLE_APIS = (
        "paddle.incubate.nn.functional.swiglu",
        "paddle.nn.functional.swiglu",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = x
y = y

if y == None:
    x, y = torch.chunk(x, 2, dim=-1)
"""
        core = "result = torch.nn.functional.silu(x) * y"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class SwishRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.swish",)

    def apply(self, paddle_api: str) -> ConvertResult:
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core="result = x * torch.sigmoid(x)",
        )


class SegmentRule(BaseRule):
    PADDLE_APIS = (
        "paddle.geometric.segment_mean",
        "paddle.geometric.segment_min",
        "paddle.geometric.segment_sum",
        "paddle.incubate.segment_max",
        "paddle.incubate.segment_mean",
        "paddle.incubate.segment_min",
        "paddle.incubate.segment_sum",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
num_segments = segment_ids.max().item() + 1
output_shape = (num_segments,) + data.shape[1:]
segment_ids = segment_ids.to(dtype=torch.int64)
"""
        core_max = """
result = torch.full(output_shape, float('-inf'), dtype=data.dtype)
result.scatter_reduce_(0, segment_ids.unsqueeze(-1).expand_as(data), data, 'amax')
result = torch.where(result == float('-inf'), torch.tensor(0.0, dtype=data.dtype), result)
"""
        core_min = """
result = torch.full(output_shape, float('inf'), dtype=data.dtype)
result.scatter_reduce_(0, segment_ids.unsqueeze(-1).expand_as(data), data, 'amin')
result = torch.where(result == float('inf'), torch.tensor(0.0, dtype=data.dtype), result)
"""
        core_sum = """
result = torch.zeros(output_shape, dtype=data.dtype)
result.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(data), data)
"""
        core_mean = """
sum_result = torch.zeros(output_shape, dtype=data.dtype)
sum_result.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(data), data)
count = torch.zeros(num_segments, dtype=torch.int64)
count.scatter_add_(0, segment_ids, torch.ones_like(segment_ids, dtype=torch.int64))
count = count.view(num_segments, *[1] * (data.dim() - 1))
count = count.clamp(min=1)
result = sum_result / count.to(sum_result.dtype)
empty_mask = (count == 1) & (sum_result == 0)
result = torch.where(empty_mask, torch.tensor(0.0, dtype=result.dtype), result)
"""
        if paddle_api.endswith("max"):
            core = core_max
        elif paddle_api.endswith("min"):
            core = core_min
        elif paddle_api.endswith("sum"):
            core = core_sum
        elif paddle_api.endswith("mean"):
            core = core_mean
        else:
            return ConvertResult.error(paddle_api, f"Unsupported segment api: {paddle_api}")
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class SoftMaxRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.softmax",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(dtype, str):
    dtype = getattr(torch, dtype)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class SendUvRule(BaseRule):
    PADDLE_APIS = ("paddle.geometric.send_uv",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
message_op = message_op
src_features = x[src_index]
dst_features = y[dst_index]
"""
        core = """
if message_op == 'add':
    result = src_features + dst_features
elif message_op == 'sub':
    result = src_features - dst_features
elif message_op == 'mul':
    result = src_features * dst_features
elif message_op == 'div':
    result = src_features / (dst_features)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class SoftmaxMaskFuseRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.softmax_mask_fuse",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.softmax(x + mask, dim=-1)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SoftmaxMaskFuseUpperTriangleRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.softmax_mask_fuse_upper_triangle",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
batch, heads, seq_len, seq_len2 = x.shape
mask = torch.triu(torch.full((seq_len, seq_len2), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1)
mask = mask.view(1, 1, seq_len, seq_len2)
result = torch.softmax(x + mask, dim=-1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SoftmaxWithCrossEntropyRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.softmax_with_cross_entropy",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
soft_label = soft_label
ignore_index = ignore_index
numeric_stable_mode = numeric_stable_mode
return_softmax = return_softmax
axis = axis

axis = axis if axis >= 0 else logits.dim() + axis

ogits = logits.transpose(axis, -1)
abel = label.transpose(axis, -1)

logits_flat = ogits.reshape(-1, ogits.shape[-1])
label_flat = abel.reshape(-1)
if numeric_stable_mode:
    max_logits = torch.max(logits_flat, dim=-1, keepdim=True).values
    logits_flat = logits_flat - max_logits
log_softmax = torch.nn.functional.log_softmax(logits_flat, dim=-1)

loss = torch.nn.functional.nll_loss(
    input=log_softmax,
    target=label_flat,
    reduction='none',
    ignore_index=ignore_index
)
if loss.ndim < ogits.ndim:
    loss = loss.reshape(*ogits.shape[:-1]).unsqueeze(-1).transpose(-1, axis)
if return_softmax:
    softmax = torch.nn.functional.softmax(ogits, dim=-1)
    softmax = softmax.transpose(-1, axis).contiguous()
    result = (loss, softmax)
else:
    result = loss
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class SoftMarginLossRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.soft_margin_loss",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
label = label.detach()
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


# t
class TrapezoidRule(BaseRule):
    PADDLE_APIS = ("paddle.trapezoid",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if torch.is_tensor(dx):
    dx = dx.item()
"""
        core = """
if x is not None:
    result = torch.trapezoid(y=y, x=x, dim=axis)
else:
    result = torch.trapezoid(y=y, dx=dx, dim=axis)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class TraceRule(BaseRule):
    PADDLE_APIS = ("paddle.trace",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.diagonal(input=x, offset=offset, dim1=axis1, dim2=axis2).sum(dim=-1)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=(),
            core=core,
        )


class TakeRule(BaseRule):
    PADDLE_APIS = ("paddle.take",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def torch_take(x, index, mode='raise'):
    x_flat = x.reshape(-1)
    numel = x_flat.numel()
    if mode == 'raise':
        index_mask = (index >= 0) & (index < numel)
        valid_indices = torch.clamp(index, 0, numel - 1)  # 避免报错，先 clamp
        taken = torch.take(x_flat, valid_indices)
        taken[~index_mask] = 0.0  # 非法 index 位置手动填 0
        return taken.view(index.shape)
    elif mode == 'wrap':
        index_mod = ((index % numel) + numel) % numel
        result = torch.take(x_flat, index_mod)
    elif mode == 'clip':
        index_clipped = torch.clamp(index, 0, numel - 1)
        result = torch.take(x_flat, index_clipped)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return result.view(index.shape)
"""

        core = "result = torch_take(x, index, mode)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class TemporalShiftRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.temporal_shift",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
shift_ratio = shift_ratio
data_format = data_format
if data_format == "NCHW":
    n_t, c, h, w = x.shape
    n = n_t // seg_num
    x = x.view(n, seg_num, c, h, w)

    fold1 = int(c * shift_ratio)
    fold2 = int(c * shift_ratio * 2)
    x_padded = torch.nn.functional.pad(x, pad=(0, 0, 0, 0, 0, 0, 1, 1))  # Pad T-dim: (left=1, right=1)

    slice1 = x_padded[:, 0:seg_num, :fold1, :, :]
    slice2 = x_padded[:, 2:seg_num+2, fold1:fold2, :, :]
    slice3 = x_padded[:, 1:seg_num+1, fold2:, :, :]

    out = torch.cat((slice1, slice2, slice3), dim=2)
    result = out.view(n_t, c, h, w)

elif data_format == "NHWC":
    n_t, h, w, c = x.shape
    n = n_t // seg_num
    x = x.view(n, seg_num, h, w, c)

    fold1 = int(c * shift_ratio)
    fold2 = int(c * shift_ratio * 2)
    x_padded = torch.nn.functional.pad(x, pad=(0, 0, 0, 0, 0, 0, 1, 1))

    slice1 = x_padded[:, 0:seg_num, :, :, :fold1]
    slice2 = x_padded[:, 2:seg_num+2, :, :, fold1:fold2]
    slice3 = x_padded[:, 1:seg_num+1, :, :, fold2:]

    out = torch.cat((slice1, slice2, slice3), dim=4)
    result = out.view(n_t, h, w, c)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class TensordotRule(BaseRule):
    PADDLE_APIS = ("paddle.tensordot",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def to_nested_list(obj):
    if isinstance(obj, int):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (tuple)):
        obj = list(obj)
    for i, item in enumerate(obj):
        if isinstance(item, (tuple)):
            obj[i] = list(item)
    return obj

axes = to_nested_list(axes)
if isinstance(axes, int):
    tmp = axes
    axes = [[], []]
    for i in range(tmp):
        axes[0].append(x.dim() - tmp + i)
        axes[1].append(i)

if isinstance(axes[0], int):
    axes = [axes, axes]

if len(axes) == 1:
    axes.append(axes[0])
if not isinstance(axes[0], list):
    axes[0] = axes[0].tolist()
if not isinstance(axes[1], list):
    axes[1] = axes[1].tolist()
if len(axes[0]) != len(axes[1]):
    if len(axes[0]) < len(axes[1]):
        padding = axes[1][len(axes[0]):]
        axes[0] += padding
    elif len(axes[1]) < len(axes[0]):
        padding = axes[0][len(axes[1]):]
        axes[1] += padding
if len(axes) > 2:
    axes = [axes[0], axes[1]]
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class TensorOuterRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.outer",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = x.flatten()
y = y.flatten()
"""
        core = "result = x.outer(y)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class TriangularSolveRule(BaseRule):
    PADDLE_APIS = ("paddle.linalg.triangular_solve",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
transpose = transpose
upper = upper
unitriangular = unitriangular
if transpose:
    x = x.transpose(-1,-2)
"""
        core = """
result = torch.linalg.solve_triangular(x,y,upper=upper,left=True,unitriangular=unitriangular)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class TruncNormalRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.init.trunc_normal_",)

    def apply(self, paddle_api: str) -> ConvertResult:
        # 两侧签名逐参对应 (tensor, mean, std, a, b)，torch 侧多一个可选 generator；
        # 原地语义也一致（返回值 is 入参）。这里显式返回入参而不依赖 GenericRule 的
        # 名字后缀推断，避免 paddle.nn.init.* 这类非 Tensor 方法走到 receiver 分支。
        core = """
torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
result = tensor
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class TolistRule(BaseRule):
    PADDLE_APIS = ("paddle.tolist",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.tolist()"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


# u
class UniqueRule(BaseRule):
    PADDLE_APIS = ("paddle.unique",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = ()
        post = """
result = list(result)
if return_inverse:
    result[1] = result[1].to(dtype=dtype)
    if result[1].ndim == 0:
        result[1] = result[1].unsqueeze(0)
    if axis is None:
        result[1] = result[1].flatten()
if return_counts:
    if return_inverse:
        result[2] = result[2].to(dtype=dtype)
    else:
        result[1] = result[1].to(dtype=dtype)
result = tuple(result)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            core=core,
            postprocess=post.splitlines(),
        )


class UnflattenRule(BaseRule):
    PADDLE_APIS = ("paddle.unflatten",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(shape, torch.Tensor):
    shape = tuple(shape.tolist())
else:
    new_shape = []
    for s in shape:
        if isinstance(s, torch.Tensor):
            new_shape.append(s.item())
        else:
            new_shape.append(s)
    shape = tuple(new_shape)
"""
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class UniqueConsecutiveRule(BaseRule):
    PADDLE_APIS = ("paddle.unique_consecutive",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
return_inverse = return_inverse
return_counts = return_counts
axis = axis
"""
        core = f"result = {self.torch_api}(x, return_inverse=return_inverse, return_counts=return_counts, dim=axis)"
        post = """
dtype = dtype
if isinstance(result, tuple):
    result = list(result)
    if return_inverse:
        if result[1].shape == torch.Size([]):
            result[1] = result[1].unsqueeze(0)
        else:
            result[1] = result[1].reshape(-1)
    if dtype is not torch.int64:
        for i in range (1, len(result)):
            result[i] = result[i].to(dtype)
    result = tuple(result)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post.splitlines(),
        )


class UnpoolRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nn.functional.max_unpool1d",
        "paddle.nn.functional.max_unpool2d",
        "paddle.nn.functional.max_unpool3d",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre_1d = """
if data_format == "NLC":
    x = x.permute(0, 2, 1)
"""
        pre_2d = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
"""
        pre_3d = """
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
"""
        pre = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
padding = tuple(padding) if isinstance(padding, list) else padding
output_size = list(output_size) if isinstance(output_size, tuple) else output_size
indices = indices.to(torch.int64)
"""
        core = ()
        post_1d = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        post_2d = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        post_3d = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        if self.torch_api.endswith("1d"):
            pre = pre_1d + pre
            post = post_1d
        elif self.torch_api.endswith("2d"):
            pre = pre_2d + pre
            post = post_2d
        elif self.torch_api.endswith("3d"):
            pre = pre_3d + pre
            post = post_3d
        else:
            return ConvertResult.error(paddle_api, f"Unsupported unpool api: {paddle_api}")
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
            postprocess=post.splitlines(),
        )


class UnfoldRule(BaseRule):
    PADDLE_APIS = ("paddle.unfold",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.unfold(dimension=axis, size=size, step=step)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=(),
            core=core,
        )


class UpsampleRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.upsample",)

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(size, torch.Tensor):
    size = [int(item.item()) for item in size]
if isinstance(scale_factor, torch.Tensor):
    scale_factor = [item.item() for item in scale_factor]
if data_format == "NHWC":
    x = x.permute(0,3,1,2)
elif data_format == "NDHWC":
    x = x.permute(0,4,1,2,3)
elif data_format == "NWC":
    x = x.permute(0,2,1)
"""
        core = """
if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
    result = torch.nn.functional.upsample(
        input=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
else:
    result = torch.nn.functional.upsample(
        input=x,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
    )
"""
        postprocess = """
if data_format == "NHWC":
    result = result.permute(0,2,3,1)
elif data_format == "NDHWC":
    result = result.permute(0,2,3,4,1)
elif data_format == "NWC":
    result = result.permute(0,2,1)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
            postprocess=postprocess,
        )


class UnsqueezeRule(BaseRule):
    PADDLE_APIS = (
        "paddle.unsqueeze",
        "paddle.Tensor.unsqueeze",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
if isinstance(axis, torch.Tensor):
    if axis.numel() == 1:
        axis = axis.item()
    else:
        axis = axis.tolist()
if isinstance(axis, tuple):
    axis = list(axis)
"""
        if paddle_api == "paddle.unsqueeze":
            core = """
if isinstance(axis, list):
    result = x
    for ax in axis:
        result = torch.unsqueeze(result, ax)
else:
    result = torch.unsqueeze(x, axis)
"""
        elif paddle_api == "paddle.Tensor.unsqueeze":
            core = """
result = x
if isinstance(axis, list):
    for ax in axis:
        result = result.unsqueeze(ax)
else:
    result = result.unsqueeze(axis)
"""
        elif paddle_api == "paddle.Tensor.unsqueeze_":
            core = """
result = x
if isinstance(axis, list):
    for ax in axis:
        result.unsqueeze_(ax)
else:
    result.unsqueeze_(axis)
"""
        else:
            return ConvertResult.error(paddle_api, f"Unsupported unsqueeze api: {paddle_api}")
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=preprocess.splitlines(),
            core=core,
        )


# v
class VanderRule(BaseRule):
    PADDLE_APIS = ("paddle.vander",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
n = n
increasing = increasing
if n is None:
    n = len(x)
x_size = x.size(0)
dtype = x.dtype
device = x.device
"""
        core = """
if n == 0:
    result = torch.zeros((x_size, 0), dtype=dtype, device=device)
elif n == 1:
    result = torch.ones((x_size, 1), dtype=dtype, device=device)
else:
    powers = torch.arange(n, device=device)
    if not increasing:
        powers = n - 1 - powers
    x_col = x.view(-1, 1)
    result = x_col ** powers
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class VecdotRule(BaseRule):
    PADDLE_APIS = ("paddle.vecdot",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
axis = axis
if x.dtype != y.dtype:
    if torch.is_complex(x) or torch.is_complex(y):
        target_dtype = torch.complex128
    else:
        if x.dtype == torch.float64 or y.dtype == torch.float64:
            target_dtype = torch.float64
        elif x.dtype == torch.float32 or y.dtype == torch.float32:
            target_dtype = torch.float32
        else:
            target_dtype = x.dtype
    x = x.to(target_dtype)
    y = y.to(target_dtype)
"""
        core = "result = torch.linalg.vecdot(x, y, dim=axis)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


class ViewRule(BaseRule):
    PADDLE_APIS = (
        "paddle.view",
        "paddle.Tensor.view",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.view(shape_or_dtype)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class ViewAsRule(BaseRule):
    PADDLE_APIS = ("paddle.view_as",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.view_as(other)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class VariableLengthMemoryEfficientAttentionRule(BaseRule):
    PADDLE_APIS = ("paddle.incubate.nn.functional.variable_length_memory_efficient_attention",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import math
from typing import Optional

def variable_length_memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    causal: bool = False,
    pre_cache_length: int = 0
) -> torch.Tensor:
    batch_size, num_heads, query_seq_len, head_size = query.shape
    key_seq_len = key.shape[2]
    # Broadcast key and value to match query's num_heads if needed
    if key.shape[1] != num_heads:
        # Repeat key and value along the num_heads dimension
        repeat_factor = num_heads // key.shape[1]
        # key = key.repeat(1, repeat_factor, 1, 1)
        # value = value.repeat(1, repeat_factor, 1, 1)
        key = key.unsqueeze(2).expand(-1,-1, repeat_factor, -1, -1).reshape(batch_size, num_heads, key_seq_len, head_size)
        value = value.unsqueeze(2).expand(-1,-1, repeat_factor, -1, -1).reshape(batch_size, num_heads, key_seq_len, head_size)
    # Default scale if not provided
    if scale is None:
        scale = math.sqrt(1.0 / head_size)
    scale = torch.tensor(scale, dtype=query.dtype, device=query.device)
    # Initialize mask if None
    if mask is None:
        mask = torch.zeros(batch_size, 1, query_seq_len, key_seq_len,
                        dtype=query.dtype, device=query.device)
    else:
        mask = mask[:, :, :query_seq_len, :key_seq_len]
    # Apply sequence length masking
    seq_mask = torch.ones(batch_size, 1, query_seq_len, key_seq_len,
                         dtype=torch.bool, device=query.device)
    for b in range(batch_size):
        q_len = seq_lens[b].squeeze().item()
        kv_len = kv_seq_lens[b].squeeze().item() + pre_cache_length
        seq_mask[b, :, q_len:, :] = False
        seq_mask[b, :, :, kv_len:] = False
    # Apply causal masking if enabled
    if causal:
        causal_mask = torch.tril(
            torch.ones(1, 1, query_seq_len, key_seq_len, dtype=torch.bool, device=query.device)
        )
        seq_mask = seq_mask & causal_mask
    # Compute attention scores: QK^T
    qk_res = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, num_heads, query_seq_len, key_seq_len]
    # Apply scale
    attention = qk_res * scale
    # attention = attention.masked_fill(~seq_mask, torch.finfo(attention.dtype).min)
    attention = attention + mask
    attention = attention.masked_fill(~seq_mask, torch.finfo(attention.dtype).min)
    # Softmax over the last dimension
    softmax_result = torch.nn.functional.softmax(attention, dim=-1)
    softmax_result = softmax_result.masked_fill(~seq_mask, 0.0)
    # Compute output: softmax(QK^T)V
    result = torch.matmul(softmax_result, value)  # [batch_size, num_heads, query_seq_len, head_size]
    return result
"""
        core = "result = variable_length_memory_efficient_attention(**bound_arguments)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


# w
class WeightDetachRule(BaseRule):
    PADDLE_APIS = (
        "paddle.nn.functional.binary_cross_entropy",
        "paddle.nn.functional.binary_cross_entropy_with_logits",
        "paddle.nn.functional.multi_margin_loss",
        "paddle.nn.functional.nll_loss",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = []
        if "weight" in self.mapping.get("paddle_torch_args_map", {}):
            pre.append("if weight is not None: weight = weight.detach()")
        if "pos_weight" in self.mapping.get("paddle_torch_args_map", {}):
            pre.append("if pos_weight is not None: pos_weight = pos_weight.detach()")
        core = ()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre,
            core=core,
        )


class WeightOnlyLinearRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.quant.weight_only_linear",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
bias = bias
if weight.shape[0] < weight_scale.shape[0]:
    weight = weight.repeat(weight_scale.shape[0] // weight.shape[0], 1)
weight_float = weight * weight_scale.unsqueeze(1)
out = torch.matmul(x, weight_float.t())
if bias is not None:
        out = out + bias
result = out
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class WhereRule(BaseRule):
    PADDLE_APIS = ("paddle.where",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if x is not None and not isinstance(x, torch.Tensor):
    x = torch.tensor(x)
"""
        core = """
if x is None and y is None:
    result = torch.where(condition)
else:
    result = torch.where(condition, x, y)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


# x


# y


# z
class ZerosRule(BaseRule):
    PADDLE_APIS = ("paddle.zeros",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dtype = dtype
if isinstance(shape,torch.Tensor):
    if shape.numel() == 1:
        shape = shape.item()
    else:
        li = []
        for i in shape:
            li.append(i.item())
        shape = li
"""
        core = """
if dtype is None:
    result = torch.zeros(shape)
else:
    result = torch.zeros(shape, dtype=dtype)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class Zeropad2dRule(BaseRule):
    PADDLE_APIS = ("paddle.nn.functional.zeropad2d",)

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
data_format = data_format
pad_left, pad_right, pad_top, pad_bottom = padding
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
    padded = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    result = padded.permute(0, 2, 3, 1)
elif data_format == "NCHW":
    result = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


# __
class TensorPowRule(BaseRule):
    PADDLE_APIS = ("paddle.Tensor.__pow__",)

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
tensor = x
other = y
"""
        core = "result = tensor.__pow__(other)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class TopkRule(BaseRule):
    PADDLE_APIS = ("paddle.topk", "paddle.Tensor.topk")

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if axis is None:
    axis = -1
if largest is None:
    largest = True
if sorted is None:
    sorted = True
"""
        core = "result = torch.topk(input=x, k=k, dim=axis, largest=largest, sorted=sorted)"
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            preprocess=pre.splitlines(),
            core=core,
        )


class BitwiseRightShiftRule(BaseRule):
    PADDLE_APIS = (
        "paddle.bitwise_right_shift",
        "paddle.Tensor.__rshift__",
    )

    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
tensor = x
other = y
is_arithmetic = is_arithmetic
# setting default value for is_arithmetic
if is_arithmetic is None:
    is_arithmetic = True

def logical_right_shift(x: torch.Tensor, y: torch.Tensor):
    mask = (1 << (x.element_size() * 8 - 1)) - 1
    x_arithmetic, mask = x >> y, mask >> (y - 1)
    shifted = torch.where(y >= 1, x_arithmetic & mask, x)
    shifted = torch.where(y < 0, torch.zeros_like(x), shifted)
    return shifted
"""
        core = """
if is_arithmetic:
    result = tensor.__rshift__(other)
else:
    # logical right shift
    result = logical_right_shift(tensor, other)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            preprocess=pre.splitlines(),
            core=core,
        )


# ──────────────────────────────────────────────────────────────────────────────
# paddle._C_ops Rules
# These rules implement torch-side equivalents for _C_ops builtins that have no
# public API counterpart or whose signature differs materially from any public API.
# Parameter names must match MANUAL_ARGUMENT_NAMES in paddle_to_torch.arguments.
# ──────────────────────────────────────────────────────────────────────────────


class CopsFlattenRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.flatten_", "paddle.Tensor.flatten_")

    """paddle._C_ops.flatten_(x, start_axis, stop_axis) -> torch.flatten in-place.

    Unlike torch.flatten, Paddle clamps stop_axis to x.ndim-1 when it exceeds
    the tensor's actual number of dimensions.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x           = x
start_axis  = start_axis
stop_axis   = stop_axis
# Clamp stop_axis to valid range (Paddle accepts out-of-range, PyTorch does not)
stop_axis = min(int(stop_axis), x.ndim - 1)
result = torch.flatten(x, start_dim=start_axis, end_dim=stop_axis)
x.set_(result)
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsSubtractRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.subtract_", "paddle.Tensor.subtract_")

    """paddle._C_ops.subtract_(x, y) -> in-place x -= y.

    Uses torch.no_grad() to avoid leaf-variable in-place errors.
    COMPOSITE because Paddle's in-place op does not propagate
    gradients, so the backward comparison is skipped.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
y = y
alpha = locals().get("alpha", 1)
with torch.no_grad():
    x.sub_(y.to(x.dtype), alpha=alpha)
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsAddRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.add_", "paddle.Tensor.add_")

    """paddle._C_ops.add_(x, y) -> in-place x += y.

    Paddle's add_ modifies x in-place (possibly with mixed float32/bfloat16 types).
    Using out-of-place torch.add produces a different result because the framework
    hands Paddle's already-modified x to the comparator while Torch holds an
    untouched tensor.  Replicate the in-place semantics with torch.no_grad().
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
y = y
alpha = locals().get("alpha", 1)
with torch.no_grad():
    x.add_(y.to(x.dtype), alpha=alpha)
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsMultiplyRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.multiply_", "paddle.Tensor.multiply_")

    """paddle._C_ops.multiply_(x, y) -> in-place x *= y.

    Same rationale as CopsAdd_Rule: the comparator sees Paddle's post-mutation x,
    so we must mirror the in-place mutation on the Torch side.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
y = y
with torch.no_grad():
    x.mul_(y.to(x.dtype))
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsBitwiseNotRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.bitwise_not",)

    """paddle._C_ops.bitwise_not(x) → torch.bitwise_not(x)"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
result = torch.bitwise_not(x)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class CopsClipRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.clip",)

    """paddle._C_ops.clip(x, min, max) → torch.clamp(x, min, max)

    _C_ops.clip always receives explicit min/max (no None guard needed).
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x   = x
mn  = min
mx  = max
if isinstance(mn, torch.Tensor):
    mn = mn.item()
if isinstance(mx, torch.Tensor):
    mx = mx.item()
result = torch.clamp(x, min=mn, max=mx)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class CopsConcatRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.concat",)

    """paddle._C_ops.concat(x, axis) → torch.cat(x, dim=axis)"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x    = x
axis = axis
result = torch.cat(x, dim=int(axis))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class CopsNumelRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.numel",)

    """paddle._C_ops.numel(x) → scalar int64 tensor of x.numel()"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
result = torch.tensor(x.numel(), dtype=torch.int64)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class CopsPutAlongAxisRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.put_along_axis_", "paddle.Tensor.put_along_axis_")

    """paddle._C_ops.put_along_axis_(arr, indices, values, axis, reduce, include_self, broadcast)
    → torch.scatter / torch.scatter_reduce (in-place on arr)
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
# C ops 使用 arr，Tensor 方法使用 x；分步回退避免缺省表达式提前求值。
arr          = locals().get("arr")
if arr is None:
    arr = x
indices      = indices
values       = values
axis         = axis
reduce       = reduce
include_self = include_self
broadcast    = broadcast
if reduce == "add":
    reduce = "sum"
if reduce == "mul":
    reduce = "prod"

def _infer_broadcast_shape(inp, idx, dim):
    shape = list(inp.shape)
    shape[dim] = list(idx.shape)[dim]
    for i in range(len(inp.shape)):
        if inp.shape[i] < idx.shape[i]:
            return None
    return tuple(shape)

index = indices
src   = values
dim   = axis
# paddle 在 `0 in indices.shape` 时直接原样返回 arr：不广播、不校验 values。
# torch 侧必须同样短路，否则空 index 无法 expand 到 broadcast_shape
# （例如 arr=[8192,32] / index=[0,10] 会报 "expanded size (8192) must match
# the existing size (0)"），而原生 torch.scatter_ 在此本就是 no-op。
if 0 in tuple(index.shape):
    pass
else:
    if broadcast:
        bshape = _infer_broadcast_shape(arr, indices, axis)
        if bshape:
            index = torch.broadcast_to(index, bshape)
            src   = torch.broadcast_to(src, bshape)
    index = index.to(dtype=torch.int64)
    with torch.no_grad():
        if reduce == "assign":
            arr.scatter_(dim, index, src)
        else:
            arr.scatter_reduce_(dim, index, src, reduce, include_self=include_self)
result = arr
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsScaleRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.scale_", "paddle.Tensor.scale_")

    """Map Paddle's in-place scale operation to an in-place Torch composite.

    Paddle casts ``scale`` and ``bias`` to an integral input dtype before the
    arithmetic. Torch rejects float scalars in integral in-place arithmetic,
    so the reference explicitly creates dtype-matched scalar tensors there.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
scale = float(scale)
bias = float(bias)
if not (x.is_floating_point() or x.is_complex()):
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    bias = torch.as_tensor(bias, dtype=x.dtype, device=x.device)
if bias_after_scale:
    x.mul_(scale).add_(bias)
else:
    x.add_(bias).mul_(scale)
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsTransposeRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.transpose",)

    """paddle._C_ops.transpose(x, perm) → torch.permute(x, dims)

    Trailing dimensions not listed in perm are kept in order (same as Paddle).
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x    = x
perm = perm
dims = tuple(perm) + tuple(range(len(perm), x.ndim))
result = torch.permute(x, dims)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.DIRECT,
            core=core,
        )


class CopsAdamwRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.adamw_",)

    """paddle._C_ops.adamw_ → torch.optim.adam._fused_adam single-step update.

    The Paddle kernel receives explicit beta*_pow tensors, while Torch fused Adam
    derives bias correction from state_steps.  Use the fused Torch kernel when
    beta2_pow can be represented as a step and adjust lr to preserve beta1_pow;
    otherwise fall back to the direct Paddle formula.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
param        = param
grad         = grad
lr_t         = learning_rate
moment1      = moment1
moment2      = moment2
moment2_max  = moment2_max
beta1_pow_t  = beta1_pow
beta2_pow_t  = beta2_pow
master_param = master_param
skip_update  = skip_update
beta1        = beta1
beta2        = beta2
epsilon      = epsilon
lr_ratio     = lr_ratio
coeff        = coeff
with_decay   = with_decay
multi_precision = multi_precision
amsgrad      = amsgrad


def _adamw_scalar(value, default=0.0):
    if value is None:
        return default
    return value.item() if torch.is_tensor(value) else float(value)


def _adamw_bool(value):
    if value is None:
        return False
    if torch.is_tensor(value):
        return bool(value.detach().cpu().any().item())
    return bool(value)


lr = _adamw_scalar(lr_t) * _adamw_scalar(lr_ratio, 1.0)
b1 = _adamw_scalar(beta1, 0.9)
b2 = _adamw_scalar(beta2, 0.999)
eps = _adamw_scalar(epsilon, 1e-8)
wd = _adamw_scalar(coeff, 0.0)
b1_pow = _adamw_scalar(beta1_pow_t)
b2_pow = _adamw_scalar(beta2_pow_t)
with_decay = _adamw_bool(with_decay)
multi_precision = _adamw_bool(multi_precision)
amsgrad = _adamw_bool(amsgrad)
work_param = master_param if (multi_precision and master_param is not None) else param


if grad is None or _adamw_bool(skip_update):
    result = param
else:
    with torch.no_grad():
        fused_success = False
        try:
            from torch.optim.adam import _fused_adam
        except (ImportError, AttributeError):
            _fused_adam = None

        can_try_fused = _fused_adam is not None
        can_try_fused = can_try_fused and not torch.is_complex(work_param)
        can_try_fused = can_try_fused and not torch.is_complex(grad)
        can_try_fused = can_try_fused and 0.0 < b1 < 1.0 and 0.0 < b2 < 1.0
        can_try_fused = can_try_fused and 0.0 < b1_pow < 1.0 and 0.0 < b2_pow < 1.0
        can_try_fused = can_try_fused and work_param.dtype == grad.dtype
        can_try_fused = can_try_fused and moment1.dtype == work_param.dtype
        can_try_fused = can_try_fused and moment2.dtype == work_param.dtype
        can_try_fused = can_try_fused and (
            not amsgrad or (moment2_max is not None and moment2_max.dtype == work_param.dtype)
        )

        if can_try_fused:
            import math

            step = math.log(b2_pow) / math.log(b2)
            if math.isfinite(step) and step > 0.0:
                # step is the training iteration count, which is always an
                # integer. Round to undo float32 round-trip noise from log/log.
                step = round(step)
                b1_pow_from_step = b1**step
                bias_correction1 = 1.0 - b1_pow
                fused_bias_correction1 = 1.0 - b1_pow_from_step
                beta1_pow_matches_step = math.isclose(
                    b1_pow_from_step,
                    b1_pow,
                    rel_tol=1e-4,
                    abs_tol=1e-7,
                )
                can_try_fused = bias_correction1 != 0.0 and fused_bias_correction1 != 0.0
                can_try_fused = can_try_fused and (not with_decay or beta1_pow_matches_step)
                if can_try_fused:
                    state_step = torch.tensor(
                        step - 1.0,
                        dtype=torch.float32,
                        device=work_param.device,
                    )
                    fused_lr = lr
                    if not beta1_pow_matches_step:
                        fused_lr = lr * fused_bias_correction1 / bias_correction1
                    try:
                        _fused_adam(
                            [work_param],
                            [grad],
                            [moment1],
                            [moment2],
                            [moment2_max] if amsgrad else [],
                            [state_step],
                            None,
                            None,
                            amsgrad=amsgrad,
                            has_complex=False,
                            beta1=b1,
                            beta2=b2,
                            lr=fused_lr,
                            weight_decay=wd if with_decay else 0.0,
                            eps=eps,
                            maximize=False,
                            capturable=False,
                            differentiable=False,
                            decoupled_weight_decay=True,
                        )
                        fused_success = True
                    except (RuntimeError, TypeError, ValueError) as err:
                        err_msg = str(err)
                        if "out of memory" in err_msg or "CUDA error" in err_msg:
                            raise

        if not fused_success:
            grad_update = grad.to(moment1.dtype)
            if with_decay:
                work_param.add_(work_param, alpha=-(lr * wd))
            moment1.mul_(b1).add_(grad_update, alpha=1.0 - b1)
            moment2.mul_(b2).addcmul_(grad_update, grad_update, value=1.0 - b2)
            if amsgrad and moment2_max is not None:
                torch.maximum(moment2_max, moment2, out=moment2_max)
                denom_state = moment2_max
            else:
                denom_state = moment2
            denom = denom_state.sqrt() / ((1.0 - b2_pow) ** 0.5) + eps
            work_param.addcdiv_(moment1, denom, value=-(lr / (1.0 - b1_pow)))

        if multi_precision and master_param is not None:
            param.copy_(work_param.to(param.dtype))
    result = param
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsMatmulRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.matmul",)

    """paddle._C_ops.matmul(x, y, transpose_x, transpose_y) → torch.matmul with optional transpose

    Paddle's primitive matmul supports:
      - N-D tensors with broadcasting (like torch.matmul)
      - per-operand transpose flags applied to the last two dims
      - 1-D operands (treated as vectors, matching torch.matmul semantics)
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x           = x
y           = y
transpose_x = transpose_x
transpose_y = transpose_y

x_mat = x.mT if (transpose_x and x.dim() >= 2) else x
y_mat = y.mT if (transpose_y and y.dim() >= 2) else y
result = torch.matmul(x_mat, y_mat)
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsFullRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.full_",)

    """paddle._C_ops.full_(x, shape, value, dtype) → x.fill_(value) in-place"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
shape = shape
fill_value = value
dtype = dtype
x     = x

# handle shape
def convert_to_list(shape):
    if isinstance(shape, torch.Tensor):
        return shape.tolist()
    elif isinstance(shape, (list, tuple)):
        shape_list = []
        for item in shape:
            if isinstance(item, torch.Tensor):
                if item.shape == torch.Size([]):
                    shape_list.append(item.item())
                else:
                    shape_list.extend(item.tolist())
            else:
                shape_list.append(item)
        return shape_list
    elif isinstance(shape, int):
        return [shape]
    else:
        return shape

# handle fill_value
def convert_to_scalar(fill_value):
    if isinstance(fill_value, torch.Tensor):
        return fill_value.item()
    # example: "-inf", "3.5"
    elif isinstance(fill_value, str):
        return float(fill_value)
    else:
        return fill_value

shape = convert_to_list(shape)
fill_value = convert_to_scalar(fill_value)

if dtype is None and not isinstance(fill_value, bool):
    if isinstance(fill_value, complex):
        dtype = torch.complex128
    else:
        dtype = torch.float32
tmp = torch.full(size=shape, fill_value=fill_value, dtype=dtype)
with torch.no_grad():
    x.set_(tmp)
result = x
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsFusedLinearParamGradAddRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.fused_linear_param_grad_add",)

    """paddle._C_ops.fused_linear_param_grad_add(x, dout, dweight, dbias, multi_prec, has_bias)

    Computes dweight += x.T @ dout  (and optionally dbias += dout.sum(0)).

    Reference implementation is selected via PADDLEAPITEST_IMPL env var:
      - "te":    Transformer Engine general_gemm (BF16 + FP32 accum)
      - "apex":            Apex/Megatron wgrad_gemm_accum_fp32
      - "torch":           aten.linear_backward + matmul fallback
    """

    SUPPORTED_IMPLEMENTATIONS = frozenset({"apex", "te", "torch"})
    # 融合线性梯度以 TE 为默认对照基线，torch 和 apex 由环境变量显式选择。
    DEFAULT_IMPLEMENTATION = "te"

    def apply(self, paddle_api: str) -> ConvertResult:
        _, core = self.build_implementation_code()
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )

    @staticmethod
    def _te_code() -> str:
        return """
import os as _os
import ctypes as _ctypes
import pathlib as _pathlib

x               = x
dout            = dout
dweight         = dweight
dbias           = dbias
multi_precision = multi_precision
has_bias        = has_bias

# TE general_gemm only supports BF16 input + FP32 accumulation
if x.dtype != torch.bfloat16:
    raise RuntimeError(
        f"PADDLEAPITEST_IMPL=te requires bfloat16 input, got {x.dtype}"
    )

# Preload libcublasLt for Transformer Engine
_ld_preload = _os.environ.get("LD_PRELOAD", "")
if "libcublasLt.so" not in _ld_preload:
    _candidates = [
        _os.environ.get("TE_CUBLASLT_PRELOAD"),
        "/usr/local/cuda/lib64/libcublasLt.so.13",
        "/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublasLt.so.13.3.0.5",
        "/usr/local/cuda-13.2/targets/x86_64-linux/lib/libcublasLt.so.13",
    ]
    for _cand in _candidates:
        if _cand and _pathlib.Path(_cand).exists():
            try:
                _ctypes.CDLL(_cand, mode=_ctypes.RTLD_GLOBAL)
                break
            except Exception:
                pass

try:
    from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm as _te_gemm
except Exception as _err:
    raise RuntimeError(
        "PADDLEAPITEST_IMPL=te: cannot import "
        "transformer_engine.pytorch.cpp_extensions.gemm.general_gemm; "
        f"error={type(_err).__name__}: {_err}"
    ) from _err

_raw = _te_gemm(x, dout, out_dtype=torch.float32, layout="NT")
_gemm_out = _raw[0] if isinstance(_raw, tuple) else _raw
new_dweight = _gemm_out.t()

if dweight is not None:
    new_dweight = new_dweight + dweight.float()
    dweight_out = new_dweight.to(dweight.dtype).detach()
else:
    dweight_out = new_dweight.detach()

if has_bias:
    dout_f = dout.float() if multi_precision else dout
    new_dbias = dout_f.reshape(-1, dout_f.shape[-1]).sum(0)
    if dbias is not None:
        dbias_sum = new_dbias + dbias.float() if multi_precision else new_dbias + dbias
        dbias_out = dbias_sum.to(dbias.dtype).detach()
    else:
        dbias_out = new_dbias.detach()
else:
    dout_f = dout.float() if multi_precision else dout
    dbias_out = torch.zeros(dout_f.shape[-1], dtype=dout_f.dtype, device=dout_f.device)

torch.cuda.synchronize()
result = [dweight_out, dbias_out]
"""

    @staticmethod
    def _apex_code() -> str:
        return """
x               = x
dout            = dout
dweight         = dweight
dbias           = dbias
multi_precision = multi_precision
has_bias        = has_bias

# wgrad_gemm_accum_fp32 only supports BF16 input + FP32 dweight
if x.dtype != torch.bfloat16:
    raise RuntimeError(
        f"PADDLEAPITEST_IMPL=apex requires bfloat16 input, got {x.dtype}"
    )

try:
    import fused_weight_gradient_mlp_cuda as _wgrad_ext
except Exception as _err:
    raise RuntimeError(
        "PADDLEAPITEST_IMPL=apex: cannot import fused_weight_gradient_mlp_cuda; "
        f"error={type(_err).__name__}: {_err}"
    ) from _err

# wgrad extension uses PyTorch Linear weight layout [N, K] and computes dout.T @ x.
# Our target dweight layout is [K, N], so we pass dweight.t() as main_grad,
# then transpose back after the in-place accumulation.
if dweight is not None:
    _main_grad = dweight.float().t().contiguous()
else:
    _main_grad = torch.zeros(
        (dout.shape[-1], x.shape[-1]), dtype=torch.float32, device=x.device
    )

_wgrad_ext.wgrad_gemm_accum_fp32(x, dout, _main_grad)
torch.cuda.synchronize()
dweight_out = _main_grad.t().contiguous().to(dweight.dtype if dweight is not None else torch.float32).detach()

if has_bias:
    dout_f = dout.float() if multi_precision else dout
    new_dbias = dout_f.reshape(-1, dout_f.shape[-1]).sum(0)
    if dbias is not None:
        dbias_sum = new_dbias + dbias.float() if multi_precision else new_dbias + dbias
        dbias_out = dbias_sum.to(dbias.dtype).detach()
    else:
        dbias_out = new_dbias.detach()
else:
    dout_f = dout.float() if multi_precision else dout
    dbias_out = torch.zeros(dout_f.shape[-1], dtype=dout_f.dtype, device=dout_f.device)

result = [dweight_out, dbias_out]
"""

    @staticmethod
    def _torch_code() -> str:
        return """
x               = x
dout            = dout
dweight         = dweight
dbias           = dbias
multi_precision = multi_precision
has_bias        = has_bias

x_f = x.float() if multi_precision else x
dout_f = dout.float() if multi_precision else dout
linear_backward_success = False
try:
    weight = torch.empty(
        (dout_f.shape[-1], x_f.shape[-1]),
        dtype=x_f.dtype,
        device=x_f.device,
    )
    _grad_input, grad_weight, grad_bias = torch.ops.aten.linear_backward(
        x_f,
        dout_f,
        weight,
        [False, True, bool(has_bias)],
    )
    # Torch linear weight layout is [out_features, in_features], while this
    # Paddle fused op accumulates x.T @ dout with shape [in_features, out_features].
    new_dweight = grad_weight.mT
    new_dbias = grad_bias if has_bias else None
    linear_backward_success = True
except (RuntimeError, TypeError, ValueError, AttributeError) as err:
    err_msg = str(err)
    if "out of memory" in err_msg or "CUDA error" in err_msg:
        raise

if not linear_backward_success:
    x_f_2d = x_f.reshape(-1, x_f.shape[-1])
    dout_f_2d = dout_f.reshape(-1, dout_f.shape[-1])
    new_dweight = x_f_2d.t().mm(dout_f_2d)
    new_dbias = dout_f_2d.sum(0) if has_bias else None

if dweight is not None:
    new_dweight = new_dweight + dweight.float() if multi_precision else new_dweight + dweight
    dweight_out = new_dweight.to(dweight.dtype).detach()
else:
    dweight_out = new_dweight.detach()

if has_bias and dbias is not None:
    dbias_sum = new_dbias + dbias.float() if multi_precision else new_dbias + dbias
    dbias_out = dbias_sum.to(dbias.dtype).detach()
elif has_bias:
    dbias_out = new_dbias.detach()
else:
    # Paddle returns a zero tensor for dbias even when has_bias=False
    dbias_out = torch.zeros(dout_f.shape[-1], dtype=dout_f.dtype, device=dout_f.device)

result = [dweight_out, dbias_out]
"""


class CopsGaussianRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.gaussian",)

    """paddle._C_ops.gaussian(shape, mean, std, seed, dtype) → torch.normal"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
shape = shape
mean  = mean
std   = std
dtype_map = {
    "float32": torch.float32, "float64": torch.float64,
    "float16": torch.float16, "bfloat16": torch.bfloat16,
}
dtype_val = dtype
torch_dtype = dtype_map.get(str(dtype_val).split(".")[-1], torch.float32)
result = torch.empty(list(shape), dtype=torch_dtype).normal_(mean=float(mean), std=float(std))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsMatmulGradRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.matmul_grad",)

    """paddle._C_ops.matmul_grad(x, y, dout, transpose_x, transpose_y) → (dx, dy)"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x           = x
y           = y
dout        = dout
transpose_x = transpose_x
transpose_y = transpose_y

x_mat = x.mT if transpose_x else x
y_mat = y.mT if transpose_y else y

# meta 张量只推导广播后的输出形状，不会为真实计算分配显存。
try:
    expected_dout_shape = torch.matmul(
        torch.empty(tuple(x_mat.shape), device="meta"),
        torch.empty(tuple(y_mat.shape), device="meta"),
    ).shape
except (RuntimeError, ValueError) as err:
    raise ValueError(
        f"matmul_grad received incompatible x/y shapes: "
        f"{tuple(x.shape)} and {tuple(y.shape)}"
    ) from err
if tuple(dout.shape) != tuple(expected_dout_shape):
    # 反向输入必须与前向 matmul 的输出逐维一致，避免后续报出误导性错误。
    raise ValueError(
        f"matmul_grad expected dout shape {tuple(expected_dout_shape)}, "
        f"but got {tuple(dout.shape)}"
    )

matmul_backward_success = False
try:
    dx_mat, dy_mat = torch.ops.aten.matmul_backward(dout, x_mat, y_mat, [True, True])
    dx = dx_mat.mT if transpose_x else dx_mat
    dy = dy_mat.mT if transpose_y else dy_mat
    matmul_backward_success = True
except (RuntimeError, TypeError, ValueError, AttributeError) as err:
    err_msg = str(err)
    if "out of memory" in err_msg or "CUDA error" in err_msg:
        raise

if not matmul_backward_success:
    dx_mat = torch.matmul(dout, y_mat.mT)
    dy_mat = torch.matmul(x_mat.mT, dout)
    dx = dx_mat.mT if transpose_x else dx_mat
    dy = dy_mat.mT if transpose_y else dy_mat

if tuple(dx.shape) != tuple(x.shape) or tuple(dy.shape) != tuple(y.shape):
    # aten fallback 可接受部分广播形状，返回前需恢复 Paddle 梯度接口的严格契约。
    raise ValueError(
        "matmul_grad produced invalid gradient shapes: "
        f"dx={tuple(dx.shape)}, expected {tuple(x.shape)}; "
        f"dy={tuple(dy.shape)}, expected {tuple(y.shape)}"
    )

result = [dx, dy]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsSquaredL2NormRule(BaseRule):
    # Python 包装入口在动态图/PIR 下直接委托同一 C++ kernel。
    PADDLE_APIS = (
        "paddle._C_ops.squared_l2_norm",
        "paddle.nn.clip._squared_l2_norm",
    )

    """paddle._C_ops.squared_l2_norm(x) → (x * x).sum() with shape [1] to match Paddle output"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x
result = (x.float() * x.float()).sum().reshape([1])
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsSwigluGradRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.swiglu_grad",)

    """paddle._C_ops.swiglu_grad(x, y, dout) -> (dx, dy)

    Two forward semantics depending on whether y is provided:
        y is not None:  out = silu(x) * y          # x, y share the same shape
        y is None:      out = silu(x[..., :C]) * x[..., C:]   where C = x.size(-1)//2

    Backward derivatives (y given):
        dy = dout * silu(x)
        dx = dout * y * sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    Backward derivatives (y is None — split last dim):
        let a = x[..., :C], b = x[..., C:]
        da = silu_backward(dout * b, a)
        db = dout * silu(a)
        dx = concat([da, db], dim=-1); dy is returned as None to match
        Paddle's uninitialized second output (the framework comparator
        accepts `paddle uninitialized + torch None` as a pass).

    Use Torch's native silu implementation for the corresponding SiLU value.
    The SiLU derivative is kept analytical because Torch does not expose a full
    swiglu_grad op and aten::silu_backward is not bitwise aligned here.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x    = x
y    = y

dout = dout

if y is None:
    C = x.shape[-1] // 2
    a = x[..., :C]
    b = x[..., C:]
    sig_a = torch.sigmoid(a)
    silu_a = torch.nn.functional.silu(a)
    da = (dout * b) * sig_a * (1.0 + a * (1.0 - sig_a))
    db = dout * silu_a
    dx = torch.cat([da, db], dim=-1)
    dy = None
else:
    sig_x = torch.sigmoid(x)
    silu_x = torch.nn.functional.silu(x)
    dy = dout * silu_x
    dx = (dout * y) * sig_x * (1.0 + x * (1.0 - sig_x))

result = [dx, dy]
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsUniformRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops.uniform",)

    """paddle._C_ops.uniform(shape, dtype, min, max, seed) → torch.empty(...).uniform_"""

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
shape = shape
dtype_val = dtype
mn    = min
mx    = max
dtype_map = {
    "float32": torch.float32, "float64": torch.float64,
    "float16": torch.float16, "bfloat16": torch.bfloat16,
}
torch_dtype = dtype_map.get(str(dtype_val).split(".")[-1], torch.float32)
result = torch.empty(list(shape), dtype=torch_dtype).uniform_(float(mn), float(mx))
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
        )


class CopsRunCustomOpRule(BaseRule):
    PADDLE_APIS = ("paddle._C_ops._run_custom_op",)

    """paddle._C_ops._run_custom_op(op_name, *args) dispatcher.

    Dispatches to per-op torch implementations based on op_name (first arg).
    Currently supported:
      - "fused_swiglu_bwd": dy, x → compute grad of swiglu
      - "fused_swiglu_scale_clamp": x, scale, max_val → scaled+clamped swiglu fwd
      - "fused_swiglu_scale_clamp_bwd": x, scale, dy, max_val → scaled+clamped swiglu bwd
      - "fused_swiglu_probs_bwd": o1, do2_s, unzipped_probs, inplace → weighted swiglu bwd
      - "paddlefleet_fused_swiglu_probs_bwd": same semantics as fused_swiglu_probs_bwd
      - "fuse_weighted_swiglu_fp8_quant": x, prob, using_pow2_scaling, use_ue8m0
      - "fuse_weighted_swiglu_fp8_quant_clamp": same plus clamp_value
      - "fused_swiglu_weighted_clamp_bwd": x, probs, d_out, clamp_value
      - "fuse_stack_fp8_quant" / "fuse_stack_transpose_fp8_quant"
    Unsupported op_names will return an error result at runtime.
    """

    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
op_name = op_name
arg1    = arg1
arg2    = arg2
arg3    = arg3
arg4    = arg4
arg5    = arg5
arg6    = arg6
arg7    = arg7
arg8    = arg8
arg9    = arg9

def _cuda_fminf(a, b):
    # CUDA fminf 遇到单侧 NaN 时返回另一侧，不能直接使用 torch.minimum。
    result = torch.minimum(a, b)
    return torch.where(torch.isnan(a), b, result)

def _cuda_fmaxf(a, b):
    # CUDA fmaxf 与 fminf 的 NaN 规则对称，保持 kernel 的异常值语义。
    result = torch.maximum(a, b)
    return torch.where(torch.isnan(a), b, result)

def _cuda_swiglu_clamp(
    gate, value, clamp_value, inplace=False, clamp_tensor=None
):
    if clamp_tensor is None:
        clamp_tensor = torch.as_tensor(
            clamp_value, dtype=gate.dtype, device=gate.device
        )
    # fminf/fmaxf(NaN, scalar) 返回 scalar；此处 scalar NaN 时返回原输入。
    if clamp_value != clamp_value:
        return gate, value
    # 非负 clamp 可以原地复刻 fminf/fmaxf，避免大 shape 的多份 where 临时量。
    if inplace and clamp_value >= 0.0:
        gate_nan = torch.isnan(gate)
        value_nan = torch.isnan(value)
        gate.clamp_(max=clamp_tensor)
        value.clamp_(min=-clamp_tensor, max=clamp_tensor)
        gate.masked_fill_(gate_nan, clamp_tensor)
        value.masked_fill_(value_nan, clamp_tensor)
        return gate, value
    gate_eff = _cuda_fminf(gate, clamp_tensor)
    value_eff = _cuda_fmaxf(
        _cuda_fminf(value, clamp_tensor), -clamp_tensor
    )
    return gate_eff, value_eff

if op_name == "fused_swiglu_bwd":
    # _run_custom_op("fused_swiglu_bwd", dy, x)
    # fwd: out = silu(x1) * x2  where x is split into (x1, x2) along last dim
    # bwd: given dy, need dx = [d_x1, d_x2] concatenated
    dy = arg1
    x = arg2
    _hidden = dy.shape[-1]
    _outer = dy.numel() // _hidden
    _dy_2d = dy.reshape(_outer, _hidden)
    _x_2d = x.reshape(_outer, _hidden * 2)
    _dx = torch.empty_like(_x_2d)
    _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
    _bytes_per_row = max(1, _hidden * (4 * 7 + x.element_size() * 2))
    _row_chunk = max(1, min(_outer, _workspace_bytes // _bytes_per_row))
    with torch.no_grad():
        for _row_start in range(0, _outer, _row_chunk):
            _row_end = min(_outer, _row_start + _row_chunk)
            _dy = _dy_2d[_row_start:_row_end].to(torch.float32)
            _lhs = _x_2d[_row_start:_row_end, :_hidden].to(torch.float32)
            _rhs = _x_2d[_row_start:_row_end, _hidden:].to(torch.float32)
            _sigmoid = torch.sigmoid(_lhs)
            _silu = _lhs * _sigmoid
            _dx[_row_start:_row_end, _hidden:] = (_dy * _silu).to(x.dtype)
            _lhs.sub_(_silu).add_(1.0).mul_(_sigmoid).mul_(_rhs).mul_(_dy)
            _dx[_row_start:_row_end, :_hidden] = _lhs.to(x.dtype)
    result = [_dx.reshape(x.shape)]

elif op_name == "fused_swiglu_scale_clamp":
    # _run_custom_op("fused_swiglu_scale_clamp", x, scale, max_val)
    # clamp 语义对齐 PaddleFleet fusions/fused_swiglu_scale.py::
    # fused_swiglu_scale_forward (gate 只截上限, value 对称截断),
    # 运算顺序按 CUDA kernel VectorizedFusedSwiGLUFwd 复刻；sigmoid/归约仍可能有
    # 约 1--2 ULP 的差异，因此不能承诺逐位一致:
    #   g = min(gate, cv); v = clamp(value, -cv, cv)
    #   out = (T)((g * sigmoid(g)) * v * s)   # 全程 fp32, 只在最后舍入一次
    # 注意: fleet 的 CPU/XPU fallback 写成
    #   (silu(g)*v).cast(x.dtype) * scale.cast(x.dtype)
    # 比 kernel 多两次 bf16 舍入, 与 CUDA 算子本身不一致, 故这里按 kernel 写。
    x       = arg1   # shape [..., 2D]
    scale   = arg2   # scalar or tensor [..., 1] or [...,]
    max_val = arg3   # scalar
    _cv = float(max_val)
    if x.ndim != 2:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_scale_clamp expects 2-D X, "
            f"but got {tuple(x.shape)}"
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_scale_clamp expects an even "
            f"last dimension, but got {tuple(x.shape)}"
        )
    _hidden = x.shape[-1] // 2
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "fused_swiglu_scale_clamp only supports bfloat16 or float32 X, "
            f"but got {x.dtype}"
        )
    _vec_size = 8 if x.dtype == torch.bfloat16 else 4
    if x.shape[0] > 0 and _hidden % _vec_size != 0:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_scale_clamp expects hidden_size "
            f"divisible by {_vec_size}, but got {_hidden}"
        )
    # fp32 输入的 .to(torch.float32) 可能复用叶子存储，必须复制后再原地截断。
    _x_fp32 = x.to(torch.float32, copy=True)
    _gate, _val = _cuda_swiglu_clamp(
        _x_fp32[..., :_hidden],
        _x_fp32[..., _hidden:],
        _cv,
        inplace=True,
    )
    if torch.is_tensor(scale):
        _scale_exp = scale.to(torch.float32)
    else:
        _scale_exp = torch.tensor(float(scale), dtype=torch.float32, device=x.device)
    while _scale_exp.dim() < _gate.dim():
        _scale_exp = _scale_exp.unsqueeze(-1)
    _swiglu = (_gate * torch.sigmoid(_gate)) * _val
    result = [(_swiglu * _scale_exp).to(x.dtype)]


elif op_name == "fused_swiglu_scale_clamp_bwd":
    # _run_custom_op("fused_swiglu_scale_clamp_bwd", x, scale, dy, max_val)
    # clamp 语义对齐 PaddleFleet fusions/fused_swiglu_scale.py::
    # fused_swiglu_scale_backward, 精度/舍入点对齐 CUDA kernel
    # VectorizedFusedSwiGLUBwd (kHasClamp=true):
    #   g_eff = min(g, cv); v_eff = clamp(v, -cv, cv)
    #   g_mask = (g <= cv); v_mask = (-cv <= v <= cv)   # fp32 掩码, 边界处梯度通过
    #   d_u    = dout * s                                # 全程 fp32
    #   d_v    = (T)(d_u * silu_g * v_mask)
    #   d_g    = (T)(d_u * sig * (1 + g_eff*(1-sig)) * v_eff * g_mask)
    #   d_scale = (ScaleT)sum_fp32( (T)swiglu * (ScaleT)dout )
    #             kernel 特意把 fp32 swiglu 先压回 x.dtype 再与 dout 相乘, 用 fp32 累加
    x       = arg1   # shape [..., 2D] (original forward input)
    scale   = arg2   # scalar or tensor [..., 1]
    dy      = arg3   # shape [..., D] (gradient of forward output)
    max_val = arg4   # scalar
    _cv = float(max_val)
    if x.ndim != 2:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_scale_clamp_bwd expects 2-D X, "
            f"but got {tuple(x.shape)}"
        )
    if x.shape[-1] % 2 != 0:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_scale_clamp_bwd expects an even "
            f"last dimension, but got {tuple(x.shape)}"
        )
    _hidden = x.shape[-1] // 2
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "fused_swiglu_scale_clamp_bwd only supports bfloat16 or float32 X, "
            f"but got {x.dtype}"
        )
    _vec_size = 8 if x.dtype == torch.bfloat16 else 4
    if x.shape[0] > 0 and _hidden % _vec_size != 0:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_scale_clamp_bwd expects hidden_size "
            f"divisible by {_vec_size}, but got {_hidden}"
        )
    if x.shape[0] > 0 and _hidden == 0:
        # Host wrapper returns an uninitialized d_scale buffer for this shape.
        raise NotImplementedError(
            "(Unimplemented) fused_swiglu_scale_clamp_bwd has an uninitialized "
            "d_scale output when hidden_size is zero"
        )
    _x_fp32 = x.to(torch.float32)
    _gate_raw = _x_fp32[..., :_hidden]
    _val_raw = _x_fp32[..., _hidden:]
    _gate, _val = _cuda_swiglu_clamp(
        _gate_raw,
        _val_raw,
        _cv,
    )
    # kernel 中掩码是 float, 保持 fp32 避免额外的 dtype 提升
    _g_mask = (_gate_raw <= _cv).to(torch.float32)
    _v_mask = ((_val_raw <= _cv) & (_val_raw >= -_cv)).to(torch.float32)
    _sig = torch.sigmoid(_gate)
    _silu = _gate * _sig
    _swiglu_val = _silu * _val
    if torch.is_tensor(scale):
        _scale_dtype = scale.dtype
        _scale_exp = scale.to(torch.float32)
    else:
        _scale_dtype = torch.float32
        _scale_exp = torch.tensor(float(scale), dtype=torch.float32, device=x.device)
    while _scale_exp.dim() < dy.dim():
        _scale_exp = _scale_exp.unsqueeze(-1)
    _d_u = dy * _scale_exp
    _d_val = _d_u * _silu * _v_mask
    # kernel 里 1.0f + g_eff*(1.0f-sig) 会被 nvcc 收缩成单条 FMA(只舍入一次),
    # 用 addcmul 走同一条 fused 路径, 而不是 mul+add 两次舍入。
    _d_gate = (
        _d_u
        * _sig
        * torch.addcmul(torch.ones((), dtype=torch.float32, device=x.device), _gate, 1.0 - _sig)
        * _val
        * _g_mask
    )
    dx = torch.cat([_d_gate, _d_val], dim=-1).to(x.dtype)
    # d_scale: fp32 swiglu 先压回 x.dtype, dout 转 scale dtype, 乘积用 fp32 累加,
    # 最后一次性 cast 回 scale dtype(与 kernel 的 shared float 归约一致)
    d_scale = (
        (_swiglu_val.to(x.dtype) * dy.to(_scale_dtype))
        .to(torch.float32)
        .sum(dim=-1, keepdim=True)
        .to(_scale_dtype)
    )
    result = [dx, d_scale]

elif op_name in ("fused_swiglu_probs_bwd", "paddlefleet_fused_swiglu_probs_bwd"):
    # _run_custom_op("fused_swiglu_probs_bwd", o1, do2_s, unzipped_probs, inplace)
    # 输出 [do1, probs_grad, o2_s]，语义参考 paddlefleet 的 SwigluProbsGradKernel:
    #   lhs, rhs = chunk(o1, 2, -1); sig = sigmoid(lhs); silu = sig*lhs
    #   do1[..., :H] = (do2_s*probs) * rhs * sig * (1 + lhs - silu)
    #   do1[..., H:] = (do2_s*probs) * silu
    #   o2_s         = silu * rhs * probs
    #   probs_grad   = sum_last_dim(do2_s * silu * rhs),shape [outer_dim],float32
    # 注意：
    #   1) 空输入 (numel==0) 时 Paddle 直接返回占位 tensor,不做 shape 一致性检查；
    #   2) 大 shape + bf16 输入若整体 upcast 到 fp32 会爆显存，分块处理。
    o1     = arg1
    do2_s  = arg2
    probs  = arg3
    inplace_flag = bool(arg4) if arg4 is not None else False

    # 此自定义 CUDA 内核只定义 bf16 激活和 fp32 概率的组合。
    if o1.ndim < 1 or do2_s.ndim < 1:
        raise ValueError("fused_swiglu_probs_bwd expects o1 and do2_s to have rank >= 1")
    if o1.dtype != torch.bfloat16 or do2_s.dtype != torch.bfloat16:
        raise TypeError("fused_swiglu_probs_bwd expects bfloat16 o1 and do2_s")
    if probs.dtype != torch.float32:
        raise TypeError("fused_swiglu_probs_bwd expects float32 unzipped_probs")
    if tuple(o1.shape[:-1]) != tuple(do2_s.shape[:-1]):
        raise ValueError(
            "fused_swiglu_probs_bwd expects matching o1/do2_s outer shapes, "
            f"but got {tuple(o1.shape)} and {tuple(do2_s.shape)}"
        )
    H2 = o1.shape[-1]
    H = H2 // 2
    # 最后一维将 o1 均分为 SwiGLU 的 gate/value，两部分必须与上游梯度对齐。
    if H <= 0:
        raise ValueError(f"moe_intermediate_size must be > 0, but got {H}")
    if H2 != H * 2 or do2_s.shape[-1] != H:
        raise ValueError(
            "fused_swiglu_probs_bwd expects o1.shape[-1] == "
            f"2 * do2_s.shape[-1], but got {H2} and {do2_s.shape[-1]}"
        )
    outer = 1
    for _s in o1.shape[:-1]:
        outer *= _s
    # 每个外层位置恰好使用一个解压概率，禁止依赖隐式广播掩盖配置错误。
    if probs.numel() != outer:
        raise ValueError(
            f"fused_swiglu_probs_bwd expects {outer} probabilities, "
            f"but got {probs.numel()}"
        )

    # Paddle 的 SwigluProbsGradCUDABackward:
    #   do1      = inplace ? o1    : empty_like(o1)
    #   o2_s     = inplace ? do2_s : empty_like(do2_s)
    #   probs_grad = empty({outer}, fp32)   # 始终新分配
    # 当任一输入 numel==0 时，kernel 不执行，直接返回上述（已分配但未写入的）buffer。
    # 因此 inplace=True 时 do1/o2_s 保留 o1/do2_s 的原值；非 inplace 时为未初始化值。
    if o1.numel() == 0 or do2_s.numel() == 0 or probs.numel() == 0:
        if inplace_flag:
            do1_e  = o1.clone()
            o2_s_e = do2_s.clone()
        else:
            do1_e  = torch.zeros_like(o1)
            o2_s_e = torch.zeros_like(do2_s)
        # probs_grad 始终是新分配的未初始化 buffer；用 0 占位
        pg_e = torch.zeros([outer], dtype=torch.float32, device=o1.device)
        result = [do1_e, pg_e, o2_s_e]
    else:
        o1_dtype  = o1.dtype
        do2_dtype = do2_s.dtype
        o1_2d  = o1.reshape(outer, H2)
        do2_2d = do2_s.reshape(outer, H)
        probs_flat = probs.reshape(-1)
        # Accuracy tests create Paddle/Torch input pairs through DLPack, so the
        # two framework tensors can share storage. Even when the custom op asks
        # Paddle to reuse its inputs, the reference must preserve its inputs for
        # the subsequent Paddle run. Chunked temporaries still bound the peak.
        do1_out = torch.empty_like(o1_2d)
        o2_s_out = torch.empty_like(do2_2d)
        pg_out = torch.empty([outer], dtype=torch.float32, device=o1.device)
        # At most nine FP32 [chunk, H] buffers overlap below. Include BF16
        # casts so the temporary working set remains within 32 GiB.
        _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
        bytes_per_row = max(1, H * (4 * 9 + o1.element_size() * 3))
        row_chunk = max(1, min(outer, _workspace_bytes // bytes_per_row))
        # 切片原地写入会触发 autograd 对叶子张量的报错（尤其 fp32 叶子节点），
        # 这里整体走 no_grad：本算子是 bwd kernel 的数值复刻，不需要再次求导。
        with torch.no_grad():
            for row_start in range(0, outer, row_chunk):
                row_end = min(outer, row_start + row_chunk)
                # inplace 时 o1_2d 即将被写入，需先把 lhs/rhs 拷到 fp32 中间变量
                lhs_c = o1_2d[row_start:row_end, :H].float()
                rhs_c = o1_2d[row_start:row_end, H:].float()
                do2_c = do2_2d[row_start:row_end].float()
                prob_c = probs_flat[row_start:row_end].to(torch.float32).unsqueeze(-1)
                sig_c = torch.sigmoid(lhs_c)
                silu_c = lhs_c * sig_c
                o2_c = silu_c * rhs_c

                pg_out[row_start:row_end] = (do2_c * o2_c).sum(dim=-1)
                o2_s_out[row_start:row_end] = (o2_c * prob_c).to(do2_dtype)

                # Reuse do2_c for the probability-weighted upstream gradient.
                do2_c.mul_(prob_c)
                x1g_c = do2_c * silu_c
                lhs_c.sub_(silu_c).add_(1.0).mul_(sig_c).mul_(rhs_c).mul_(do2_c)
                do1_out[row_start:row_end, :H] = lhs_c.to(o1_dtype)
                do1_out[row_start:row_end, H:] = x1g_c.to(o1_dtype)
        result = [
            do1_out.reshape(o1.shape),
            pg_out,
            o2_s_out.reshape(do2_s.shape),
        ]

elif op_name == "fused_swiglu_weighted_clamp_bwd":
    # _run_custom_op("fused_swiglu_weighted_clamp_bwd", x, probs, d_out, clamp_value)
    # 输出 [d_x, d_probs, out]（out 是重算的前向结果）。语义与舍入点对齐 CUDA kernel
    # paddlefleet_ops/_extensions/fuse_swiglu_scale.cu::
    # VectorizedFusedSwiGLUWeightedBwd (kHasClamp=true):
    #   g_eff = min(g, cv); v_eff = clamp(v, -cv, cv)
    #   g_mask = (g <= cv); v_mask = (-cv <= v <= cv)   # fp32 掩码, 边界处梯度通过
    #   sig = sigmoid(g_eff); silu = g_eff * sig; swiglu = silu * v_eff
    #   out     = (T)(swiglu * p)
    #   d_u     = dout * p                              # 全程 fp32
    #   d_v     = (T)(d_u * silu * v_mask)
    #   d_g     = (T)(d_u * sig * (1 + g_eff*(1-sig)) * v_eff * g_mask)
    #   d_probs = (ScaleT)sum_fp32( (T)swiglu * (ScaleT)dout )   # 逐行归约
    # 注意 clamp 分支的 d_g 乘法顺序与非 clamp 分支不同（v_eff 在括号之后），
    # 且 d_probs 会先把 fp32 的 swiglu 压回 x.dtype 再相乘，两处都会影响低位。
    # d_probs 形状固定为 [rows, 1]，见 WeightedBwdClampInferShape。
    x     = arg1
    probs = arg2
    d_out = arg3
    if arg4 is None:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_weighted_clamp_bwd requires "
            "clamp_value"
        )
    _cv = float(arg4)

    # 形状/dtype 约束取自 CheckFusedSwiGLUInputs 与 FusedSwiGLUWeightedBackwardImpl。
    if x.ndim != 2:
        raise ValueError(
            f"fused_swiglu_weighted_clamp_bwd expects 2-D X, but got {tuple(x.shape)}"
        )
    if d_out.ndim != 2:
        raise ValueError(
            f"fused_swiglu_weighted_clamp_bwd expects 2-D DOut, but got {tuple(d_out.shape)}"
        )
    _rows = x.shape[0]
    _hidden2 = x.shape[1]
    if _hidden2 % 2 != 0:
        raise ValueError(
            "fused_swiglu_weighted_clamp_bwd expects X shape [rows, 2 * hidden_size], "
            f"but got {tuple(x.shape)}"
        )
    _hidden = _hidden2 // 2
    if probs.numel() != _rows or tuple(probs.shape) not in ((_rows,), (_rows, 1)):
        raise ValueError(
            f"fused_swiglu_weighted_clamp_bwd expects Probs shape [{_rows}] or "
            f"[{_rows}, 1], but got {tuple(probs.shape)}"
        )
    # kernel 只实例化 (bf16, fp32)、(bf16, bf16)、(fp32, fp32) 三种组合。
    if (x.dtype, probs.dtype) not in (
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ):
        raise TypeError(
            "fused_swiglu_weighted_clamp_bwd only supports (X, Probs) dtypes "
            "(bfloat16, float32) / (bfloat16, bfloat16) / (float32, float32), "
            f"but got ({x.dtype}, {probs.dtype})"
        )
    if d_out.dtype != x.dtype:
        raise TypeError(
            "fused_swiglu_weighted_clamp_bwd expects DOut to share X's dtype, "
            f"but got {d_out.dtype} and {x.dtype}"
        )
    _vec_size = 8 if x.dtype == torch.bfloat16 else 4
    if _rows > 0 and _hidden % _vec_size != 0:
        raise ValueError(
            "(InvalidArgument) fused_swiglu_weighted_clamp_bwd expects "
            f"hidden_size divisible by {_vec_size}, but got {_hidden}"
        )

    _x_dtype = x.dtype
    _probs_dtype = probs.dtype
    _probs_shape = [_rows, 1]
    # 分块循环复用同一个标量 tensor，避免每个 chunk 重建 clamp 参数。
    _clamp_tensor = torch.as_tensor(_cv, dtype=torch.float32, device=x.device)
    if _rows > 0 and _hidden == 0:
        # Paddle host wrapper 对 d_probs 使用 empty，内容未定义，不能伪造零值比较。
        raise NotImplementedError(
            "(Unimplemented) fused_swiglu_weighted_clamp_bwd has an "
            "uninitialized d_probs output when hidden_size is zero"
        )
    if _rows == 0:
        result = [
            torch.zeros_like(x),
            torch.zeros(_probs_shape, dtype=_probs_dtype, device=x.device),
            torch.zeros([_rows, _hidden], dtype=_x_dtype, device=x.device),
        ]
    else:
        if tuple(d_out.shape) != (_rows, _hidden):
            raise ValueError(
                f"fused_swiglu_weighted_clamp_bwd expects DOut shape [{_rows}, "
                f"{_hidden}], but got {tuple(d_out.shape)}"
            )
        _dx_out = torch.empty_like(x)
        _dprobs_out = torch.empty(_probs_shape, dtype=_probs_dtype, device=x.device)
        _fwd_out = torch.empty([_rows, _hidden], dtype=_x_dtype, device=x.device)
        _probs_flat = probs.reshape(-1)
        # 每行同时存活约 14 个 fp32 [chunk, hidden] 中间量，另有 3 份 x.dtype 写出。
        _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
        # 该分支始终使用非原地 helper，按 fminf/fmaxf 的临时量峰值估算。
        _clamp_extra_bytes = 4 * 8
        _bytes_per_row = max(
            1,
            _hidden * (4 * 14 + x.element_size() * 3 + _clamp_extra_bytes),
        )
        _row_chunk = max(1, min(_rows, _workspace_bytes // _bytes_per_row))
        _one = torch.ones((), dtype=torch.float32, device=x.device)
        with torch.no_grad():
            for _row_start in range(0, _rows, _row_chunk):
                _row_end = min(_rows, _row_start + _row_chunk)
                _gate_raw = x[_row_start:_row_end, :_hidden].to(torch.float32)
                _val_raw = x[_row_start:_row_end, _hidden:].to(torch.float32)
                _gate, _val = _cuda_swiglu_clamp(
                    _gate_raw,
                    _val_raw,
                    _cv,
                    clamp_tensor=_clamp_tensor,
                )
                _g_mask = (_gate_raw <= _cv).to(torch.float32)
                _v_mask = ((_val_raw <= _cv) & (_val_raw >= -_cv)).to(torch.float32)
                _sig = torch.sigmoid(_gate)
                _silu = _gate * _sig
                _swiglu = _silu * _val
                _dout_slice = d_out[_row_start:_row_end]
                _dout_c = _dout_slice.to(torch.float32)
                _p_c = _probs_flat[_row_start:_row_end].to(torch.float32).unsqueeze(-1)

                _fwd_out[_row_start:_row_end] = (_swiglu * _p_c).to(_x_dtype)
                # d_probs: fp32 的 swiglu 先压回 x.dtype，dout 转 probs dtype，
                # 乘积用 fp32 累加后一次性 cast（对齐 kernel 的 shared float 归约）。
                _dprobs_out[_row_start:_row_end] = (
                    (_swiglu.to(_x_dtype) * _dout_slice.to(_probs_dtype))
                    .to(torch.float32)
                    .sum(dim=-1, keepdim=True)
                    .to(_probs_dtype)
                )

                _d_u = _dout_c * _p_c
                _dx_out[_row_start:_row_end, _hidden:] = (_d_u * _silu * _v_mask).to(_x_dtype)
                # kernel 里 1.0f + g_eff*(1.0f-sig) 会被 nvcc 收缩成单条 FMA（只舍入
                # 一次），用 addcmul 走同一条 fused 路径而不是 mul+add 两次舍入。
                _dx_out[_row_start:_row_end, :_hidden] = (
                    _d_u * _sig * torch.addcmul(_one, _gate, 1.0 - _sig) * _val * _g_mask
                ).to(_x_dtype)
        result = [_dx_out, _dprobs_out, _fwd_out]

elif op_name in ("fuse_weighted_swiglu_fp8_quant", "fuse_weighted_swiglu_fp8_quant_clamp"):
    # fuse_weighted_swiglu_fp8_quant(x, prob, using_pow2_scaling, use_ue8m0)
    # fuse_weighted_swiglu_fp8_quant_clamp(x, prob, using_pow2_scaling, use_ue8m0, clamp_value)
    # Returns: [output_fp8 (rows, cols/2), scale (rows, ceil(cols/2/128))];
    # UE8M0 packs every four scale exponents into one int32 column.
    # SwiGLU(x[:, :cols/2], x[:, cols/2:]) * prob -> block-wise 1x128 FP8 quant
    # 两个变体共用 FusedWeightedSwigluActQuantImpl, 只差 kHasClamp 模板参数, 见
    # paddlefleet_ops/_extensions/fuse_weighted_swiglu_fp8_quant.cu::fast_swiglu:
    #   gate 只截上限 min(g, cv), value 对称截断 clamp(v, -cv, cv), 之后 SwiGLU 不变。
    _x = arg1
    _prob = arg2
    _using_pow2_scaling = bool(arg3) if arg3 is not None else False
    _use_ue8m0 = bool(arg4) if arg4 is not None else False
    _has_clamp = op_name.endswith("_clamp")
    if _has_clamp and arg5 is None:
        raise ValueError(
            "(InvalidArgument) fuse_weighted_swiglu_fp8_quant_clamp requires "
            "clamp_value"
        )
    _cv = float(arg5) if _has_clamp else 0.0
    _FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    _TILE = 128

    if _x.ndim < 2:
        raise ValueError(
            "fuse_weighted_swiglu_fp8_quant expects X to have at least 2 dimensions, "
            f"but got {tuple(_x.shape)}"
        )
    if _x.dtype != torch.bfloat16:
        raise TypeError(
            "fuse_weighted_swiglu_fp8_quant expects X to be bfloat16, "
            f"but got {_x.dtype}"
        )
    if _prob is not None and torch.is_tensor(_prob) and _prob.dtype != torch.float32:
        raise TypeError(
            "fuse_weighted_swiglu_fp8_quant expects prob to be float32, "
            f"but got {_prob.dtype}"
        )
    if _x.shape[-1] % 2 != 0:
        raise ValueError(
            "fuse_weighted_swiglu_fp8_quant expects the last X dimension to be even, "
            f"but got {tuple(_x.shape)}"
        )
    if _use_ue8m0 and _x.ndim != 2:
        raise ValueError(
            "fuse_weighted_swiglu_fp8_quant with use_ue8m0 expects 2-D X, "
            f"but got {tuple(_x.shape)}"
        )

    _rows = 1
    for _dim in _x.shape[:-1]:
        _rows *= _dim
    _cols = _x.shape[-1]
    _x_2d = _x.reshape(_rows, _cols)
    if _use_ue8m0 and _cols % 1024 != 0:
        raise ValueError(
            "(InvalidArgument) fuse_weighted_swiglu_fp8_quant with use_ue8m0 "
            "requires the last X dimension divisible by 1024"
        )
    if _use_ue8m0 and _rows % 4 != 0:
        # Host wrapper pads scale rows with empty storage; padded values are undefined.
        raise NotImplementedError(
            "(Unimplemented) use_ue8m0 scale has undefined padded rows when "
            "the flattened row count is not divisible by 4"
        )
    _prob_flat = None
    if _prob is not None:
        if not torch.is_tensor(_prob):
            raise TypeError("fuse_weighted_swiglu_fp8_quant expects prob Tensor or None")
        if (
            _prob.ndim < 1
            or _prob.shape[0] != _rows
            or _prob.numel() != _rows
        ):
            raise ValueError(
                "fuse_weighted_swiglu_fp8_quant expects prob.shape[0] == rows "
                f"and one value per row ({_rows}), but got {tuple(_prob.shape)}"
            )
        _prob_flat = _prob.reshape(-1)
    _half_cols = _cols // 2
    _num_col_blocks = (_half_cols + _TILE - 1) // _TILE
    _dev = _x.device
    _clamp_inplace = _has_clamp and _cv >= 0.0
    # clamp 分支的标量参数在所有 row chunk 间共享。
    _clamp_tensor = (
        torch.as_tensor(_cv, dtype=torch.float32, device=_dev)
        if _has_clamp
        else None
    )

    _output_fp8 = torch.empty([_rows, _half_cols], dtype=torch.float8_e4m3fn, device=_dev)
    _scale_out = torch.empty([_rows, _num_col_blocks], dtype=torch.float32, device=_dev)

    _full_blocks = _half_cols // _TILE
    _full_cols = _full_blocks * _TILE
    # LHS, RHS, and the sigmoid temporary can overlap. Include the FP8 cast
    # temporary and account for clamp NaN masks/negative-clamp temporaries.
    _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
    _clamp_extra_bytes = (
        (2 if _clamp_inplace else 4 * 8) if _has_clamp else 0
    )
    _bytes_per_row = max(
        1,
        _half_cols * (4 * 3 + 1 + _clamp_extra_bytes),
    )
    _row_chunk = max(1, min(_rows, _workspace_bytes // _bytes_per_row))

    with torch.no_grad():
        for _row_start in range(0, _rows, _row_chunk):
            _row_end = min(_rows, _row_start + _row_chunk)
            _lhs = _x_2d[_row_start:_row_end, :_half_cols].to(torch.float32)
            _rhs = _x_2d[_row_start:_row_end, _half_cols:].to(torch.float32)
            if _has_clamp:
                _lhs, _rhs = _cuda_swiglu_clamp(
                    _lhs,
                    _rhs,
                    _cv,
                    inplace=_clamp_inplace,
                    clamp_tensor=_clamp_tensor,
                )
            _lhs.mul_(torch.sigmoid(_lhs)).mul_(_rhs)
            del _rhs
            if _prob_flat is not None:
                _lhs.mul_(_prob_flat[_row_start:_row_end].to(torch.float32).unsqueeze(-1))

            if _full_blocks:
                _blocks = _lhs[:, :_full_cols].reshape(
                    _row_end - _row_start, _full_blocks, _TILE
                )
                _amax = _blocks.abs().amax(dim=-1)
                # ComputeScaleImpl(eps=0) defines an all-zero block's scale as 1.
                _amax.masked_fill_(_amax == 0, _FP8_MAX)
                _quant_scale = _FP8_MAX / _amax
                if _using_pow2_scaling:
                    _quant_scale.log2_().floor_().exp2_()
                _scale_out[_row_start:_row_end, :_full_blocks] = _quant_scale.reciprocal()
                _blocks.mul_(_quant_scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX)
                _output_fp8[_row_start:_row_end, :_full_cols] = _blocks.to(
                    torch.float8_e4m3fn
                ).reshape(_row_end - _row_start, _full_cols)
                del _blocks, _amax, _quant_scale

            if _full_cols < _half_cols:
                _tail = _lhs[:, _full_cols:]
                _tail_amax = _tail.abs().amax(dim=-1)
                _tail_amax.masked_fill_(_tail_amax == 0, _FP8_MAX)
                _tail_scale = _FP8_MAX / _tail_amax
                if _using_pow2_scaling:
                    _tail_scale.log2_().floor_().exp2_()
                _scale_out[_row_start:_row_end, _full_blocks] = _tail_scale.reciprocal()
                _tail.mul_(_tail_scale.unsqueeze(-1)).clamp_(-_FP8_MAX, _FP8_MAX)
                _output_fp8[_row_start:_row_end, _full_cols:] = _tail.to(
                    torch.float8_e4m3fn
                )
                del _tail, _tail_amax, _tail_scale
            del _lhs

    if _use_ue8m0:
        # Pack 4 exponent columns into 1 int32 (ue8m0 format)
        # Kernel stores the raw IEEE-754 exponent bits of inv_scale, not rounded log2.
        _log2_inv = (_scale_out.view(torch.int32) >> 23) & 0xFF
        _pack_cols = _log2_inv.shape[-1]
        _pad_c = (4 - (_pack_cols % 4)) % 4
        if _pad_c:
            _log2_inv = torch.nn.functional.pad(_log2_inv, (0, _pad_c))
        _log2_inv = _log2_inv.reshape(_log2_inv.shape[0], -1, 4)
        _scale_packed = (_log2_inv[..., 0] | (_log2_inv[..., 1] << 8) | (_log2_inv[..., 2] << 16) | (_log2_inv[..., 3] << 24)).to(torch.int32)
        result = [_output_fp8, _scale_packed]
    else:
        result = [_output_fp8, _scale_out]

elif op_name in ("fuse_stack_fp8_quant", "fuse_stack_transpose_fp8_quant"):
    # fuse_stack[_transpose]_fp8_quant(X_list, using_pow2_scaling, using_ue8m0_scale, output_scale_transpose)
    # X_list: list of N tensors each [M, K] in bfloat16
    # Returns: [output_fp8, scale]
    import math as _math
    _X_list = arg1
    _using_pow2_scaling = bool(arg2) if arg2 is not None else False
    _using_ue8m0_scale = bool(arg3) if arg3 is not None else False
    _output_scale_transpose = bool(arg4) if arg4 is not None else False
    _do_transpose = ("transpose" in op_name)
    _N = len(_X_list)
    _M, _K = _X_list[0].shape
    _dev = _X_list[0].device
    _FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    _TILE = 128

    # Fill the final FP8 layout directly from each source tensor. Only the
    # active row-block chunk is promoted to FP32.
    if _do_transpose:
        _out_rows = _N * _K
        _out_cols = _M
    else:
        _out_rows = _N * _M
        _out_cols = _K

    _num_row_blocks = (_out_rows + _TILE - 1) // _TILE
    _num_col_blocks = (_out_cols + _TILE - 1) // _TILE
    _output_fp8 = torch.empty(
        (_out_rows, _out_cols), dtype=torch.float8_e4m3fn, device=_dev
    )
    _inv_scale = torch.empty(
        (_num_row_blocks, _num_col_blocks), dtype=torch.float32, device=_dev
    )
    _workspace_bytes = _adaptive_workspace_bytes(torch, locals())
    _bytes_per_row = max(1, _out_cols * (4 * 2 + 1) + _out_cols * 4)
    _row_blocks_per_chunk = max(
        1, min(_num_row_blocks, _workspace_bytes // _bytes_per_row // _TILE)
    )

    with torch.no_grad():
        for _block_start in range(0, _num_row_blocks, _row_blocks_per_chunk):
            _block_end = min(_num_row_blocks, _block_start + _row_blocks_per_chunk)
            _row_start = _block_start * _TILE
            _row_end = min(_out_rows, _block_end * _TILE)
            _valid_rows = _row_end - _row_start
            _padded_rows = (_block_end - _block_start) * _TILE
            _padded_cols = _num_col_blocks * _TILE
            _chunk = torch.zeros(
                (_padded_rows, _padded_cols), dtype=torch.float32, device=_dev
            )
            _cursor = _row_start
            _chunk_offset = 0
            while _cursor < _row_end:
                if _do_transpose:
                    _source_index = _cursor // _K
                    _source_row = _cursor % _K
                    _take = min(_row_end - _cursor, _K - _source_row)
                    _chunk[_chunk_offset : _chunk_offset + _take, :_out_cols] = (
                        _X_list[_source_index].transpose(0, 1)[
                            _source_row : _source_row + _take
                        ]
                    )
                else:
                    _source_index = _cursor // _M
                    _source_row = _cursor % _M
                    _take = min(_row_end - _cursor, _M - _source_row)
                    _chunk[_chunk_offset : _chunk_offset + _take, :_out_cols] = (
                        _X_list[_source_index][_source_row : _source_row + _take]
                    )
                _cursor += _take
                _chunk_offset += _take
            _tiles = _chunk.reshape(
                _block_end - _block_start, _TILE, _num_col_blocks, _TILE
            ).permute(0, 2, 1, 3)
            # fuse_stack kernel passes the default eps=1e-10 to ComputeScale.
            _amax = _tiles.abs().amax(dim=(2, 3)).clamp_(min=1e-10)
            _quant_scale = _FP8_MAX / _amax
            if _using_pow2_scaling or _using_ue8m0_scale:
                _quant_scale.log2_().floor_().exp2_()
            _inv_scale[_block_start:_block_end] = _quant_scale.reciprocal()
            _tiles.mul_(_quant_scale.unsqueeze(-1).unsqueeze(-1)).clamp_(
                -_FP8_MAX, _FP8_MAX
            )
            _quantized = _tiles.permute(0, 2, 1, 3).reshape(
                (_block_end - _block_start) * _TILE, _num_col_blocks * _TILE
            )
            _output_fp8[_row_start:_row_end] = _quantized[
                :_valid_rows, :_out_cols
            ].to(torch.float8_e4m3fn)
            del _chunk, _tiles, _amax, _quant_scale, _quantized

    # Scale output
    if _using_ue8m0_scale:
        # Kernel stores the raw IEEE-754 exponent bits of inv_scale, not rounded log2.
        _log2_inv = (_inv_scale.view(torch.int32) >> 23) & 0xFF
        _scale_out = _log2_inv.unsqueeze(1).expand(-1, _TILE, -1).reshape(
            _num_row_blocks * _TILE, _num_col_blocks
        )[:_out_rows]
        # Pack 4 exponent columns into 1 int32 (ue8m0 format)
        _pack_cols = _scale_out.shape[-1]
        _pad_c = (4 - (_pack_cols % 4)) % 4
        if _pad_c:
            _scale_out = torch.nn.functional.pad(_scale_out, (0, _pad_c))
        _scale_out = _scale_out.reshape(_scale_out.shape[0], -1, 4)
        _scale_out = (_scale_out[..., 0] | (_scale_out[..., 1] << 8) | (_scale_out[..., 2] << 16) | (_scale_out[..., 3] << 24)).to(torch.int32)
        if _output_scale_transpose:
            _scale_out = _scale_out.t().contiguous()
    else:
        _scale_out = _inv_scale
        if _output_scale_transpose:
            _scale_out = _scale_out.t().contiguous()

    result = [_output_fp8, _scale_out]

else:
    raise NotImplementedError(f"CopsRunCustomOpRule: unsupported op_name={op_name!r}")
"""
        return self.build_result(
            paddle_api,
            kind=ConversionKind.COMPOSITE,
            core=core,
            workspace_required=True,
        )


_RULE_REGISTRY_FROZEN = True
