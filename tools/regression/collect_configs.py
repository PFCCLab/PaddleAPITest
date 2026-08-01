from __future__ import annotations

import argparse
import math
import numbers
import os
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

from tester.api_config.parser import APIConfig
from tester.input_generation.tensor_config import TensorConfig

DEFAULT_OUTPUT = Path("tools/regression/regression_configs.txt")
DEFAULT_SUMMARY = Path("tools/regression/regression_summary.txt")
SOURCE_ENV_VAR = "REGRESSION_CONFIG_SOURCES"
EXCLUDED_PATH_KEYWORDS = ("needfix", "need_fix", "not_monitor")


def source_spec(path: Path) -> str:
    return "1M" if "1m" in path.name.lower() else "4096"


def iter_source_files(sources):
    paths = []
    for source in sources:
        if source.is_file():
            paths.append(source)
            continue
        if not source.exists():
            continue
        paths.extend(source.rglob("*.txt"))

    selected = {
        path
        for path in paths
        if path.suffix.lower() == ".txt"
        and ("4096" in path.name.lower() or "1m" in path.name.lower())
        and not any(keyword in str(path).lower() for keyword in EXCLUDED_PATH_KEYWORDS)
    }
    yield from sorted(selected, key=lambda path: (source_spec(path) == "1M", str(path)))


def custom_op_name(api_config: APIConfig) -> str | None:
    if api_config.api_name != "paddle._C_ops._run_custom_op":
        return None
    if api_config.args and isinstance(api_config.args[0], str) and api_config.args[0]:
        return api_config.args[0]
    value = api_config.kwargs.get("op_name")
    if isinstance(value, str) and value:
        return value
    return "unknown"


def api_key(api_config: APIConfig) -> str:
    op_name = custom_op_name(api_config)
    if op_name is not None:
        return f"{api_config.api_name}:{op_name}"
    return api_config.api_name


def iter_tensor_configs(value):
    if isinstance(value, TensorConfig):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensor_configs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensor_configs(item)


def iter_shape_dimensions(value):
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, numbers.Integral) and not isinstance(item, bool) for item in value)
    ):
        yield [abs(int(item)) for item in value]
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_shape_dimensions(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_shape_dimensions(item)


def config_shape_score(api_config: APIConfig):
    tensors = [
        *iter_tensor_configs(api_config.args),
        *iter_tensor_configs(api_config.kwargs),
    ]
    shapes = [[abs(int(dim)) for dim in tensor.shape] for tensor in tensors]
    if not shapes:
        for value in (*api_config.args, *api_config.kwargs.values()):
            shapes.extend(iter_shape_dimensions(value))
    numels = [math.prod(shape) for shape in shapes]
    dimensions = [dim for shape in shapes for dim in shape]
    return (
        sum(numels),
        max(numels, default=0),
        max(dimensions, default=0),
        str(api_config),
    )


def collect_configs(sources, max_per_api):
    selected: OrderedDict[str, list[str]] = OrderedDict()
    seen_configs: set[str] = set()
    one_m_candidates = defaultdict(dict)
    stats = Counter()

    for path in iter_source_files(sources):
        stats["files"] += 1
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                config_text = raw_line.strip()
                if not config_text or not config_text.startswith("paddle."):
                    continue
                stats["lines"] += 1
                try:
                    api_config = APIConfig(config_text)
                except Exception:
                    stats["parse_error"] += 1
                    continue
                key = api_key(api_config)
                normalized = str(api_config)
                if source_spec(path) == "1M":
                    one_m_candidates[key].setdefault(normalized, api_config)
                    continue
                bucket = selected.setdefault(key, [])
                if len(bucket) >= max_per_api:
                    stats["skip_bucket_full"] += 1
                    continue
                if normalized in seen_configs:
                    stats["skip_duplicate"] += 1
                    continue
                bucket.append(normalized)
                seen_configs.add(normalized)
                stats["selected"] += 1
                stats["selected_4096"] += 1

    for key, candidates in one_m_candidates.items():
        bucket = selected.setdefault(key, [])
        for normalized, _api_config in sorted(
            candidates.items(), key=lambda item: config_shape_score(item[1])
        ):
            if len(bucket) >= max_per_api:
                break
            if normalized in seen_configs:
                stats["skip_duplicate"] += 1
                continue
            bucket.append(normalized)
            seen_configs.add(normalized)
            stats["selected"] += 1
            stats["selected_1M"] += 1
    return selected, stats


def write_outputs(selected, stats, output_path, summary_path, max_per_api):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for key in sorted(selected):
            for config in selected[key]:
                output.write(f"{config}\n")
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("# Project regression config summary\n\n")
        for name, value in sorted(stats.items()):
            summary.write(f"- {name}: {value}\n")
        summary.write(f"- max_per_api: {max_per_api}\n")
        summary.write("- source_file_policy: *4096*.txt, *1M*.txt\n")
        summary.write("- config_filter_policy: exclude needfix, need_fix, not_monitor paths\n")
        summary.write("- selection_policy: 4096 first, then smallest 1M shapes\n")
        summary.write("- incomplete_api_keys_policy: keep available configs\n")
        summary.write(f"- regression_api_keys: {len(selected)}\n")
        summary.write(f"- regression_configs: {sum(len(items) for items in selected.values())}\n")
        summary.write(
            f"- incomplete_api_keys: {sum(1 for items in selected.values() if len(items) < max_per_api)}\n\n"
        )
        summary.write("## API key coverage\n\n")
        for key in sorted(selected):
            summary.write(f"- {key}: {len(selected[key])}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect project regression API configs.")
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=None,
        help="Source file or directory. Can be specified multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-per-api", type=int, default=5)
    return parser.parse_args()


def resolve_sources(cli_sources):
    if cli_sources:
        return tuple(cli_sources)
    env_value = os.environ.get(SOURCE_ENV_VAR)
    if env_value:
        return tuple(Path(value) for value in env_value.split(os.pathsep) if value)
    raise SystemExit(f"no config sources provided; pass --source or set {SOURCE_ENV_VAR}")


def main():
    args = parse_args()
    sources = resolve_sources(args.source)
    selected, stats = collect_configs(sources, args.max_per_api)
    write_outputs(selected, stats, args.output, args.summary, args.max_per_api)
    print(f"regression api keys: {len(selected)}", flush=True)
    print(f"regression configs: {sum(len(items) for items in selected.values())}", flush=True)
    print(f"output: {args.output}", flush=True)
    print(f"summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
