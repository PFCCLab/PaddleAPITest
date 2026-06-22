from __future__ import annotations

import hashlib
from typing import Any

import torch

METRICS_DTYPES = {
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    byte_tensor = tensor.detach().contiguous().view(torch.uint8)
    digest = hashlib.sha256(byte_tensor.cpu().numpy().tobytes()).hexdigest()
    return f"sha256:{digest}"


def unravel_index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    indices: list[int] = []
    remainder = flat_index
    for size in reversed(shape):
        indices.append(remainder % size)
        remainder //= size
    return list(reversed(indices))


def tensor_value(tensor: torch.Tensor, indices: list[int]) -> float:
    value: Any = tensor
    for index in indices:
        value = value[index]
    return float(value.item())


def compute_metrics(
    actual: torch.Tensor,
    expect: torch.Tensor,
    eps: float = 1e-12,
    metrics_dtype: str = "fp32",
) -> dict[str, Any]:
    if actual.shape != expect.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)}, expect={tuple(expect.shape)}"
        )
    if metrics_dtype not in METRICS_DTYPES:
        raise ValueError(
            f"metrics_dtype must be one of {sorted(METRICS_DTYPES)}, got {metrics_dtype!r}"
        )

    dtype = METRICS_DTYPES[metrics_dtype]
    actual_compute = actual.to(dtype)
    expect_compute = expect.to(dtype)
    diff = actual_compute - expect_compute
    abs_diff = diff.abs()
    rel_diff = abs_diff / torch.clamp(expect_compute.abs(), min=eps)

    flat_abs_idx = int(abs_diff.reshape(-1).argmax().item())
    flat_rel_idx = int(rel_diff.reshape(-1).argmax().item())
    abs_idx = unravel_index(flat_abs_idx, tuple(abs_diff.shape))
    rel_idx = unravel_index(flat_rel_idx, tuple(rel_diff.shape))

    return {
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "p99_abs": float(torch.quantile(abs_diff.reshape(-1), 0.99).item()),
        "max_rel": float(rel_diff.max().item()),
        "mean_rel": float(rel_diff.mean().item()),
        "p99_rel": float(torch.quantile(rel_diff.reshape(-1), 0.99).item()),
        "max_abs_idx": abs_idx,
        "max_rel_idx": rel_idx,
        "actual_at_max_abs": tensor_value(actual_compute, abs_idx),
        "expect_at_max_abs": tensor_value(expect_compute, abs_idx),
    }
