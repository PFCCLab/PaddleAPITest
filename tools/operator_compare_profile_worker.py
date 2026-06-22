#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one operator implementation under NVTX ranges for nsys profiling."
    )
    parser.add_argument("--case", required=True, help="One PaddleAPITest config line to profile.")
    parser.add_argument(
        "--implementation",
        required=True,
        help="Full implementation id, such as paddle|config|default.",
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--metrics-dtype", default="fp64", choices=["fp32", "fp64"])
    return parser.parse_args()


def main() -> None:
    from tester.operator_compare.implementations import build_compare_suite

    args = parse_args()
    implementation_name = args.implementation.split("|", 1)[0]
    suite = build_compare_suite(
        config_lines=[args.case],
        implementation_names=[implementation_name],
        standard=args.implementation,
        dtypes=[None],
        metrics_dtype=args.metrics_dtype,
        enable_fingerprint=False,
    )
    case = suite.cases[0]
    spec = next((item for item in suite.implementations if item.id == args.implementation), None)
    if spec is None:
        raise ValueError(f"implementation not found: {args.implementation}")

    import torch

    spec.runner(case)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for idx in range(args.repeat):
        range_name = f"{args.implementation}|case={case.id}|iter={idx}"
        torch.cuda.nvtx.range_push(range_name)
        out = spec.runner(case)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        print(range_name, tuple(out.shape), out.dtype, flush=True)


if __name__ == "__main__":
    main()
