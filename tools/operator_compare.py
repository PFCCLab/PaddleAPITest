#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "test_log_operator_compare"


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run operator implementation comparisons and render a report."
    )
    parser.add_argument("--case", default=None, help="One PaddleAPITest config line to compare.")
    parser.add_argument(
        "--config-file", default=None, help="PaddleAPITest config txt file to compare."
    )
    parser.add_argument("--op", default=None, help="Optional API name filter for --config-file.")
    parser.add_argument(
        "--implementations",
        default="paddle,torch",
        help="Comma-separated implementations: paddle, torch, or registered custom names.",
    )
    parser.add_argument(
        "--dtypes",
        default=None,
        help="Comma-separated dtype override matrix. Defaults to config dtype.",
    )
    parser.add_argument("--precisions", default="default", help="Comma-separated precision names.")
    parser.add_argument(
        "--standard",
        default=None,
        help="Standard implementation id. Defaults to torch|<dtype>|default when torch is enabled, otherwise first implementation.",
    )
    parser.add_argument("--metrics-dtype", default="fp64", choices=["fp32", "fp64"])
    parser.add_argument(
        "--no-fingerprint", action="store_true", help="Disable output tensor SHA256 fingerprints."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact output directory. Defaults to test_log_operator_compare/<op>/<timestamp>.",
    )
    args = parser.parse_args()
    if not args.case and not args.config_file:
        parser.error("one of --case or --config-file is required")
    return args


def load_config_lines(args: argparse.Namespace) -> list[str]:
    lines = []
    if args.case:
        lines.append(args.case)
    if args.config_file:
        from tester.operator_compare.config_loader import cases_from_config_file

        loaded_cases = cases_from_config_file(args.config_file)
        for case in loaded_cases:
            if args.op is None or case.metadata["api_name"] == args.op:
                lines.append(case.metadata["raw_config"])
    return lines


def default_standard(implementation_names: list[str], dtypes: list[str | None]) -> str:
    dtype = dtypes[0] if dtypes else None
    dtype_part = dtype or "config"
    implementation = "torch" if "torch" in implementation_names else implementation_names[0]
    return f"{implementation}|{dtype_part}|default"


def build_suite(args: argparse.Namespace):
    from tester.operator_compare.implementations import build_compare_suite

    implementation_names = comma_list(args.implementations)
    dtypes = comma_list(args.dtypes) if args.dtypes else [None]
    standard = args.standard or default_standard(implementation_names, dtypes)
    return build_compare_suite(
        config_lines=load_config_lines(args),
        implementation_names=implementation_names,
        standard=standard,
        dtypes=dtypes,
        precisions=comma_list(args.precisions),
        metrics_dtype=args.metrics_dtype,
        enable_fingerprint=not args.no_fingerprint,
    )


def main() -> None:
    args = parse_args()

    from tester.operator_compare.artifacts import timestamped_output_dir, write_artifacts
    from tester.operator_compare.report import render_report
    from tester.operator_compare.runner import run_compare_suite

    suite = build_suite(args)
    out_dir = (
        pathlib.Path(args.output_dir).resolve()
        if args.output_dir
        else timestamped_output_dir(DEFAULT_OUTPUT_ROOT, suite.op_name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_data = run_compare_suite(suite)
    write_artifacts(out_dir, run_data)
    report_path = render_report(out_dir, run_data)

    print(f"Output directory: {out_dir}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
