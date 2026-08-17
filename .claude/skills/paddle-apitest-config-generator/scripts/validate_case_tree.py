#!/usr/bin/env python3
"""Validate a separated PaddleAPITest config tree and its manifests."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from apitest_config_utils import (
    SPECS,
    APIConfig,
    TensorConfig,
    api_key,
    config_tensors,
    import_official_config_types,
)


def validate(args: argparse.Namespace) -> None:
    official_api_config = None
    if args.official_analyzer:
        official_api_config, _ = import_official_config_types(args.apitest_root)
    root = args.case_tree.resolve()
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing {index_path}")
    index_records = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index_records, list) or not index_records:
        raise ValueError("index.json must contain a non-empty list")

    parsed_count = 0
    zero_tensor_count = 0
    manifest_cache: dict[Path, list[dict]] = {}
    for index_record in index_records:
        path = root / index_record["path"]
        if path.name not in {f"{spec}.txt" for spec in SPECS}:
            raise ValueError(f"unexpected spec filename: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if args.expected_per_file is not None and len(lines) != args.expected_per_file:
            raise ValueError(f"{path} has {len(lines)} cases, expected {args.expected_per_file}")
        if len(lines) != len(set(lines)):
            raise ValueError(f"duplicate configs in {path}")

        manifest_path = path.parent / "manifest.jsonl"
        if manifest_path not in manifest_cache:
            manifest_cache[manifest_path] = [
                json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
        records = [
            record
            for record in manifest_cache[manifest_path]
            if record["spec"] == index_record["spec"]
        ]
        if [record["config"] for record in records] != lines:
            raise ValueError(f"manifest/config mismatch for {path}")
        categories = dict(collections.Counter(record["category"] for record in records))
        if categories != index_record["categories"]:
            raise ValueError(f"category count mismatch for {path}")

        for record, line in zip(records, lines):
            config = APIConfig(line)
            if str(config) != line:
                raise ValueError(f"non-round-trip config in {path}: {line}")
            if official_api_config is not None:
                official_config = official_api_config(line)
                if str(official_config) != line:
                    raise ValueError(f"official analyzer round-trip failed in {path}: {line}")
            if api_key(config) != record["api"] or record["api"] != index_record["api"]:
                raise ValueError(f"API mismatch in {path}: {line}")
            parsed_count += 1
            if index_record["spec"] != "0size":
                continue
            tensors = config_tensors(config, TensorConfig)
            zero_tensors = [tensor for tensor in tensors if 0 in tensor.shape]
            if args.require_zero and tensors and not zero_tensors:
                raise ValueError(f"0size case contains no zero-dimension Tensor: {line}")
            if args.materialize_zero:
                for position, tensor in enumerate(zero_tensors):
                    array = tensor.get_numpy_tensor(config, index=position)
                    if array.size != 0:
                        raise ValueError(f"zero Tensor materialized non-empty: {line}")
                    zero_tensor_count += 1

    print(
        f"validated {len(index_records)} files and {parsed_count} configs; "
        f"materialized {zero_tensor_count} zero-dimension Tensors"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_tree", type=Path)
    parser.add_argument("--official-analyzer", action="store_true")
    parser.add_argument(
        "--apitest-root",
        type=Path,
        default=Path.cwd(),
        help="used only with --official-analyzer",
    )
    parser.add_argument("--expected-per-file", type=int)
    parser.add_argument("--require-zero", action="store_true")
    parser.add_argument("--materialize-zero", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
