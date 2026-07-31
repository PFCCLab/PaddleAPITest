from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from pathlib import Path

from tester.api_config.parser import APIConfig
from tester.input_generation.registry import API_RULE_REGISTRY
from tester.paddle_to_torch import get_converter

DEFAULT_SOURCES = (
    Path(
        "/root/paddlejob/share-storage/gpfs/system-public/lihaoyang08/"
        "PaddleAPITest-worktrees/opt/tester/api_config/monitor_config/accuracy/GPU"
    ),
    Path(
        "/root/paddlejob/share-storage/gpfs/system-public/lihaoyang08/"
        "baidu/paddle/jelly/case/scripts/api_test/apitest_config"
    ),
)

DEFAULT_OUTPUT = Path("tools/regression/regression_configs.txt")
DEFAULT_SUMMARY = Path("tools/regression/regression_summary.txt")
EXCLUDED_PATH_KEYWORDS = (
    "needfix",
    "need_fix",
    "not_monitor",
    "1m",
    "0size",
)
EXCLUDED_CONFIG_KEYWORDS = ("float8_",)
EXCLUDED_API_NAMES = (
    "paddle.empty",
    "paddle.empty_like",
)


def iter_source_files(sources):
    for source in sources:
        if source.is_file():
            if source.suffix == ".txt":
                yield source
            continue
        if not source.exists():
            continue
        for path in sorted(source.rglob("*.txt")):
            path_text = str(path).lower()
            if any(keyword in path_text for keyword in EXCLUDED_PATH_KEYWORDS):
                continue
            yield path


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


def is_accuracy_supported(api_name: str, converter_cache: dict[str, bool]) -> bool:
    if api_name not in converter_cache:
        result = get_converter().convert(api_name)
        converter_cache[api_name] = bool(
            result.is_supported and result.code and result.code.is_valid()
        )
    return converter_cache[api_name]


def should_skip_config(config_text: str) -> bool:
    lowered = config_text.lower()
    return any(keyword in lowered for keyword in EXCLUDED_CONFIG_KEYWORDS)


def collect_configs(sources, max_per_api):
    selected: OrderedDict[str, list[str]] = OrderedDict()
    seen_configs: set[str] = set()
    converter_cache: dict[str, bool] = {}
    stats = Counter()

    for path in iter_source_files(sources):
        stats["files"] += 1
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                config_text = raw_line.strip()
                if not config_text or not config_text.startswith("paddle."):
                    continue
                stats["lines"] += 1
                if should_skip_config(config_text):
                    stats["skip_config_keyword"] += 1
                    continue
                try:
                    api_config = APIConfig(config_text)
                except Exception:
                    stats["parse_error"] += 1
                    continue
                if api_config.api_name in EXCLUDED_API_NAMES:
                    stats["skip_api_policy"] += 1
                    continue
                if api_config.api_name not in API_RULE_REGISTRY:
                    stats["skip_no_input_rule"] += 1
                    continue
                if not is_accuracy_supported(api_config.api_name, converter_cache):
                    stats["skip_no_accuracy_rule"] += 1
                    continue
                key = api_key(api_config)
                bucket = selected.setdefault(key, [])
                if len(bucket) >= max_per_api:
                    stats["skip_bucket_full"] += 1
                    continue
                normalized = str(api_config)
                if normalized in seen_configs:
                    stats["skip_duplicate"] += 1
                    continue
                bucket.append(normalized)
                seen_configs.add(normalized)
                stats["selected"] += 1
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
        summary.write("- incomplete_api_keys_policy: included\n")
        summary.write("- excluded_size_policies: 1M, 0size\n")
        summary.write(f"- excluded_api_policies: {', '.join(EXCLUDED_API_NAMES)}\n")
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


def main():
    args = parse_args()
    sources = tuple(args.source or DEFAULT_SOURCES)
    selected, stats = collect_configs(sources, args.max_per_api)
    write_outputs(selected, stats, args.output, args.summary, args.max_per_api)
    print(f"regression api keys: {len(selected)}", flush=True)
    print(f"regression configs: {sum(len(items) for items in selected.values())}", flush=True)
    print(f"output: {args.output}", flush=True)
    print(f"summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
