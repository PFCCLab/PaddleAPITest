from __future__ import annotations

from typing import Any

import torch

from .metrics import compute_metrics, tensor_fingerprint
from .spec import (
    CompareSuite,
    ImplementationResult,
    ImplementationSpec,
    MetricResult,
    PairwiseResult,
)


def metric_from_dict(metrics: dict[str, Any]) -> MetricResult:
    return MetricResult(**metrics)


def run_implementation(
    case_id: str, spec: ImplementationSpec, case, enable_fingerprint: bool
) -> ImplementationResult:
    metadata = dict(spec.metadata)
    metadata["case_metadata"] = dict(getattr(case, "metadata", {}))
    try:
        case.tensors["_current_spec"] = spec
        tensor = spec.runner(case)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"implementation {spec.id} returned {type(tensor).__name__}, expected torch.Tensor"
            )
        if enable_fingerprint:
            metadata["output_fingerprint"] = tensor_fingerprint(tensor)
        return ImplementationResult(
            case_id=case_id,
            spec=spec,
            status="ok",
            tensor=tensor.detach(),
            output_dtype=str(tensor.dtype),
            metadata=metadata,
        )
    except Exception as err:
        return ImplementationResult(
            case_id=case_id,
            spec=spec,
            status="failed",
            error=f"{type(err).__name__}: {err}",
            metadata=metadata,
        )
    finally:
        case.tensors.pop("_current_spec", None)


def comparable_pair(actual: ImplementationResult, expect: ImplementationResult) -> bool:
    if actual.status != "ok" or expect.status != "ok":
        return False
    if actual.tensor is None or expect.tensor is None:
        return False
    if actual.spec.dtype != expect.spec.dtype:
        return False
    if actual.spec.multi_precision != expect.spec.multi_precision:
        return False
    return True


def run_compare_suite(suite: CompareSuite) -> dict[str, Any]:
    all_results: list[ImplementationResult] = []
    pairwise_results: list[PairwiseResult] = []
    reference_pairwise_results: list[PairwiseResult] = []

    standard_by_case: dict[str, ImplementationResult] = {}
    for case in suite.cases:
        case_results = [
            run_implementation(case.id, spec, case, suite.enable_fingerprint)
            for spec in suite.implementations
        ]
        all_results.extend(case_results)

        standard = next(
            (result for result in case_results if result.spec.id == suite.standard_id), None
        )
        if standard is None:
            raise ValueError(f"standard implementation {suite.standard_id!r} not registered")
        if standard.status != "ok" or standard.tensor is None:
            raise RuntimeError(
                f"standard implementation failed for case {case.id}: {standard.error}"
            )
        standard_by_case[case.id] = standard

        for result in case_results:
            if result.status == "ok" and result.tensor is not None:
                result.metrics_vs_standard = metric_from_dict(
                    compute_metrics(
                        result.tensor, standard.tensor, metrics_dtype=suite.metrics_dtype
                    )
                )

        targets = [result for result in case_results if result.spec.group in suite.target_groups]
        references = [
            result for result in case_results if result.spec.group in suite.reference_groups
        ]
        for actual in targets:
            for expect in references:
                if not comparable_pair(actual, expect):
                    continue
                metrics = metric_from_dict(
                    compute_metrics(actual.tensor, expect.tensor, metrics_dtype=suite.metrics_dtype)
                )
                pairwise_results.append(PairwiseResult(case.id, actual, expect, metrics))

        for actual_index, actual in enumerate(references):
            for expect in references[actual_index + 1 :]:
                if not comparable_pair(actual, expect):
                    continue
                metrics = metric_from_dict(
                    compute_metrics(
                        actual.tensor,
                        expect.tensor,
                        metrics_dtype=suite.reference_pairwise_metrics_dtype,
                    )
                )
                reference_pairwise_results.append(PairwiseResult(case.id, actual, expect, metrics))

    return {
        "suite": suite,
        "results": all_results,
        "pairwise_results": pairwise_results,
        "reference_pairwise_results": reference_pairwise_results,
        "standard_by_case": standard_by_case,
    }
