from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

TensorRunner = Callable[["CompareCase"], Any]


@dataclass
class CompareCase:
    id: str
    shape: tuple[int, ...]
    tensors: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationSpec:
    id: str
    display_name: str
    group: str
    runner: TensorRunner
    dtype: str | None = None
    multi_precision: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    max_abs: float
    mean_abs: float
    rmse: float
    p99_abs: float
    max_rel: float
    mean_rel: float
    p99_rel: float
    max_abs_idx: list[int]
    max_rel_idx: list[int]
    actual_at_max_abs: float
    expect_at_max_abs: float


@dataclass
class ImplementationResult:
    case_id: str
    spec: ImplementationSpec
    status: str
    tensor: Any | None = None
    output_dtype: str | None = None
    metrics_vs_standard: MetricResult | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseResult:
    case_id: str
    actual: ImplementationResult
    expect: ImplementationResult
    metrics: MetricResult


@dataclass
class CompareSuite:
    op_name: str
    cases: list[CompareCase]
    implementations: list[ImplementationSpec]
    standard_id: str
    target_groups: set[str]
    reference_groups: set[str]
    metrics_dtype: str = "fp32"
    reference_pairwise_metrics_dtype: str = "fp64"
    enable_fingerprint: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    report_config: dict[str, Any] = field(default_factory=dict)
