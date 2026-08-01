from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tester.api_config.parser import APIConfig
from tools.regression.collect_configs import api_key

ALLOWED_TYPES = frozenset({"pass", "skip", "checkpoint", "paddle_bitwise"})
DEFAULT_SUMMARY = Path("tools/regression/regression_summary.txt")
RESULT_FILES = {
    "paddle_error": "api_config_paddle_error.txt",
    "paddle_accuracy": "api_config_paddle_accuracy.txt",
    "paddle_bitwise": "api_config_paddle_bitwise.txt",
    "paddle_cuda": "api_config_paddle_cuda.txt",
    "paddle_crash": "api_config_paddle_crash.txt",
    "oom": "api_config_oom.txt",
    "timeout": "api_config_timeout.txt",
    "torch_error": "api_config_torch_error.txt",
    "config_input": "api_config_config_input.txt",
    "config_parse": "api_config_config_parse.txt",
    "config_convert": "api_config_config_convert.txt",
}


def read_configs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return {line.strip() for line in handle if line.strip()}


def blocked_configs(log_dirs: list[Path]) -> dict[str, set[str]]:
    blocked: dict[str, set[str]] = {}
    for log_dir in log_dirs:
        for result_type, file_name in RESULT_FILES.items():
            if result_type in ALLOWED_TYPES:
                continue
            configs = read_configs(log_dir / file_name)
            if configs:
                blocked.setdefault(result_type, set()).update(configs)
    return blocked


def selected_by_api(configs: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for config in configs:
        counts[api_key(APIConfig(config))] += 1
    return dict(sorted(counts.items()))


def write_summary(
    configs: list[str],
    summary_path: Path,
    max_per_api: int,
    blocked_by_type: dict[str, set[str]],
):
    counts = selected_by_api(configs)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("# Project regression config summary\n\n")
        summary.write(f"- max_per_api: {max_per_api}\n")
        summary.write("- source_file_policy: *4096*.txt, *1M*.txt\n")
        summary.write("- config_filter_policy: exclude needfix, need_fix, not_monitor paths\n")
        summary.write("- selection_policy: 4096 first, then smallest 1M shapes\n")
        summary.write("- incomplete_api_keys_policy: keep available configs\n")
        summary.write(f"- regression_api_keys: {len(counts)}\n")
        summary.write(f"- regression_configs: {sum(counts.values())}\n")
        summary.write(
            f"- incomplete_api_keys: {sum(1 for value in counts.values() if value < max_per_api)}\n"
        )
        if blocked_by_type:
            summary.write("- last_refinement_blocked_configs: ")
            summary.write(f"{sum(len(configs) for configs in blocked_by_type.values())}\n")
            for result_type, blocked in sorted(blocked_by_type.items()):
                summary.write(f"- last_refinement_{result_type}: {len(blocked)}\n")
        summary.write("\n## API key coverage\n\n")
        for key, count in counts.items():
            summary.write(f"- {key}: {count}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remove non-allowed regression failures from a config set."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-per-api", type=int, default=5)
    parser.add_argument("--log-dir", action="append", required=True, type=Path)
    args = parser.parse_args()

    original = [line.strip() for line in args.input.read_text().splitlines() if line.strip()]
    blocked_by_type = blocked_configs(args.log_dir)
    blocked_all = set().union(*blocked_by_type.values()) if blocked_by_type else set()
    refined = [config for config in original if config not in blocked_all]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{config}\n" for config in refined), encoding="utf-8")
    write_summary(refined, args.summary, args.max_per_api, blocked_by_type)

    print(f"input configs: {len(original)}", flush=True)
    print(f"blocked configs: {len(blocked_all)}", flush=True)
    for result_type, configs in sorted(blocked_by_type.items()):
        print(f"  {result_type}: {len(configs)}", flush=True)
    print(f"output configs: {len(refined)}", flush=True)
    print(f"output: {args.output}", flush=True)
    print(f"summary: {args.summary}", flush=True)


if __name__ == "__main__":
    main()
