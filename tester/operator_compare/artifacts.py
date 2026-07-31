from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import pathlib
import platform
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

from .spec import CompareSuite, ImplementationResult, PairwiseResult


def timestamped_output_dir(root: pathlib.Path, op_name: str) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / op_name / timestamp
    out_dir = base
    suffix = 1
    while out_dir.exists():
        out_dir = pathlib.Path(f"{base}_{suffix:02d}")
        suffix += 1
    out_dir.mkdir(parents=True)
    return out_dir


def metric_dict(metric) -> dict[str, Any] | None:
    return None if metric is None else asdict(metric)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def collect_env_info(suite: CompareSuite) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": package_version("numpy"),
        "transformer_engine_version": package_version("transformer-engine"),
        "transformer_engine_torch_version": package_version("transformer-engine-torch"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
        "TE_CUBLASLT_PRELOAD": os.environ.get("TE_CUBLASLT_PRELOAD"),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "config": {
            "op_name": suite.op_name,
            "standard_id": suite.standard_id,
            "metrics_dtype": suite.metrics_dtype,
            "reference_pairwise_metrics_dtype": suite.reference_pairwise_metrics_dtype,
            "enable_fingerprint": suite.enable_fingerprint,
            **suite.metadata,
        },
    }
    try:
        import paddle

        info["paddle_version"] = paddle.__version__
    except Exception as err:
        info["paddle_error"] = f"{type(err).__name__}: {err}"
    try:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "torch_cudnn_version": torch.backends.cudnn.version(),
                "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            }
        )
        try:
            info["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision()
        except Exception as err:
            info["torch_float32_matmul_precision_error"] = f"{type(err).__name__}: {err}"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as err:
        info["torch_error"] = f"{type(err).__name__}: {err}"
    return info


def metadata_value(result: ImplementationResult, key: str, default: Any = None) -> Any:
    return result.metadata.get(key, default)


def case_metadata_value(result: ImplementationResult, key: str, default: Any = None) -> Any:
    return result.metadata.get("case_metadata", {}).get(key, default)


def result_to_dict(result: ImplementationResult, include_tensor: bool = False) -> dict[str, Any]:
    data = {
        "case_id": result.case_id,
        "id": result.spec.id,
        "display_name": result.spec.display_name,
        "group": result.spec.group,
        "dtype": result.spec.dtype,
        "multi_precision": result.spec.multi_precision,
        "status": result.status,
        "output_dtype": result.output_dtype,
        "metrics_vs_standard": metric_dict(result.metrics_vs_standard),
        "error": result.error,
        "metadata": result.metadata,
    }
    if include_tensor and result.tensor is not None:
        data["tensor_shape"] = list(result.tensor.shape)
    return data


def pairwise_to_dict(pairwise: PairwiseResult) -> dict[str, Any]:
    return {
        "case_id": pairwise.case_id,
        "actual_id": pairwise.actual.spec.id,
        "actual_display_name": pairwise.actual.spec.display_name,
        "actual_group": pairwise.actual.spec.group,
        "actual_dtype": pairwise.actual.spec.dtype,
        "actual_multi_precision": pairwise.actual.spec.multi_precision,
        "actual_output_dtype": pairwise.actual.output_dtype,
        "expect_id": pairwise.expect.spec.id,
        "expect_display_name": pairwise.expect.spec.display_name,
        "expect_group": pairwise.expect.spec.group,
        "expect_dtype": pairwise.expect.spec.dtype,
        "expect_multi_precision": pairwise.expect.spec.multi_precision,
        "expect_output_dtype": pairwise.expect.output_dtype,
        "metrics": metric_dict(pairwise.metrics),
    }


def write_json(path: pathlib.Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_summary_csv(
    path: pathlib.Path, results: list[ImplementationResult], suite: CompareSuite
) -> None:
    case_columns = suite.report_config.get("summary_case_metadata_columns", ["m", "k", "n"])
    implementation_columns = suite.report_config.get(
        "summary_implementation_metadata_columns",
        ["category", "implementation", "input_dtype", "dweight_dtype", "output_fingerprint"],
    )
    columns = [
        "case_id",
        *case_columns,
        "id",
        "display_name",
        "group",
        *implementation_columns,
        "dtype",
        "multi_precision",
        "status",
        "output_dtype",
        "max_abs",
        "mean_abs",
        "rmse",
        "p99_abs",
        "max_rel",
        "mean_rel",
        "p99_rel",
        "max_abs_idx",
        "max_rel_idx",
        "actual_at_max_abs",
        "expect_at_max_abs",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            metric = metric_dict(result.metrics_vs_standard) or {}
            row = {
                "case_id": result.case_id,
                "id": result.spec.id,
                "display_name": result.spec.display_name,
                "group": result.spec.group,
                "dtype": result.spec.dtype,
                "multi_precision": result.spec.multi_precision,
                "status": result.status,
                "output_dtype": result.output_dtype,
                "max_abs": metric.get("max_abs"),
                "mean_abs": metric.get("mean_abs"),
                "rmse": metric.get("rmse"),
                "p99_abs": metric.get("p99_abs"),
                "max_rel": metric.get("max_rel"),
                "mean_rel": metric.get("mean_rel"),
                "p99_rel": metric.get("p99_rel"),
                "max_abs_idx": ";".join(str(i) for i in metric.get("max_abs_idx", [])),
                "max_rel_idx": ";".join(str(i) for i in metric.get("max_rel_idx", [])),
                "actual_at_max_abs": metric.get("actual_at_max_abs"),
                "expect_at_max_abs": metric.get("expect_at_max_abs"),
                "error": result.error,
            }
            row.update({key: case_metadata_value(result, key) for key in case_columns})
            row.update({key: metadata_value(result, key) for key in implementation_columns})
            writer.writerow(row)


def write_pairwise_csv(path: pathlib.Path, pairwise_results: list[PairwiseResult]) -> None:
    columns = [
        "case_id",
        "actual_id",
        "actual_display_name",
        "actual_group",
        "actual_dtype",
        "actual_multi_precision",
        "actual_output_dtype",
        "expect_id",
        "expect_display_name",
        "expect_group",
        "expect_dtype",
        "expect_multi_precision",
        "expect_output_dtype",
        "max_abs",
        "mean_abs",
        "rmse",
        "p99_abs",
        "max_rel",
        "mean_rel",
        "p99_rel",
        "max_abs_idx",
        "actual_at_max_abs",
        "expect_at_max_abs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for pairwise in pairwise_results:
            metric = asdict(pairwise.metrics)
            writer.writerow(
                {
                    **{
                        key: value
                        for key, value in pairwise_to_dict(pairwise).items()
                        if key != "metrics"
                    },
                    "max_abs": metric["max_abs"],
                    "mean_abs": metric["mean_abs"],
                    "rmse": metric["rmse"],
                    "p99_abs": metric["p99_abs"],
                    "max_rel": metric["max_rel"],
                    "mean_rel": metric["mean_rel"],
                    "p99_rel": metric["p99_rel"],
                    "max_abs_idx": ";".join(str(i) for i in metric["max_abs_idx"]),
                    "actual_at_max_abs": metric["actual_at_max_abs"],
                    "expect_at_max_abs": metric["expect_at_max_abs"],
                }
            )


def write_artifacts(out_dir: pathlib.Path, run_data: dict[str, Any]) -> dict[str, pathlib.Path]:
    suite: CompareSuite = run_data["suite"]
    results: list[ImplementationResult] = run_data["results"]
    pairwise_results: list[PairwiseResult] = run_data["pairwise_results"]
    reference_pairwise_results: list[PairwiseResult] = run_data["reference_pairwise_results"]

    paths = {
        "env": out_dir / "env.json",
        "results": out_dir / "results.json",
        "summary": out_dir / "summary.csv",
        "pairwise": out_dir / "pairwise_summary.csv",
        "reference_pairwise": out_dir / "reference_pairwise_summary.csv",
    }
    write_json(paths["env"], collect_env_info(suite))
    write_json(
        paths["results"],
        {
            "results": [result_to_dict(result) for result in results],
            "pairwise_results": [pairwise_to_dict(item) for item in pairwise_results],
            "reference_pairwise_results": [
                pairwise_to_dict(item) for item in reference_pairwise_results
            ],
            "profile": run_data.get("profile"),
        },
    )
    write_summary_csv(paths["summary"], results, suite)
    write_pairwise_csv(paths["pairwise"], pairwise_results)
    write_pairwise_csv(paths["reference_pairwise"], reference_pairwise_results)
    return paths
