from __future__ import annotations

import argparse
from pathlib import Path

ALLOWED_RESULT_TYPES = frozenset({"pass", "skip", "checkpoint", "paddle_bitwise"})
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


def count_configs(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Validate regression error_stat output.")
    parser.add_argument("--log-dir", required=True, type=Path)
    args = parser.parse_args()

    blocked = {}
    for result_type, file_name in RESULT_FILES.items():
        count = count_configs(args.log_dir / file_name)
        if count and result_type not in ALLOWED_RESULT_TYPES:
            blocked[result_type] = count
        print(f"{result_type}: {count}", flush=True)

    if blocked:
        details = ", ".join(f"{name}={count}" for name, count in sorted(blocked.items()))
        raise SystemExit(f"non-bitwise regression issues found: {details}")
    print("regression gate passed: no non-bitwise issues", flush=True)


if __name__ == "__main__":
    main()
