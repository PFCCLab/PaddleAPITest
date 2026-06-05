# 筛选 skip 配置小工具
# @author: cangtianhuang
# @date: 2025-11-11
from __future__ import annotations

from pathlib import Path

TEST_LOG_PATH = Path("tester/api_config/test_log")
OUTPUT_PATH = TEST_LOG_PATH / "api_config_skip.txt"

LOG_PREFIXES = {
    "checkpoint": "checkpoint",
    "pass": "api_config_pass",
    "skip": "api_config_skip",
    "paddle_error": "api_config_paddle_error",
    "paddle_accuracy": "api_config_paddle_accuracy",
    "paddle_bitwise": "api_config_paddle_bitwise",
    "paddle_cuda": "api_config_paddle_cuda",
    "paddle_crash": "api_config_paddle_crash",
    "oom": "api_config_oom",
    "timeout": "api_config_timeout",
    "torch_error": "api_config_torch_error",
    "config_input": "api_config_config_input",
    "config_parse": "api_config_config_parse",
    "config_convert": "api_config_config_convert",
}

log_counts = {}
checkpoint_configs = set()
api_configs = set()
checkpoint_file = TEST_LOG_PATH / "checkpoint.txt"
if not checkpoint_file.exists():
    print("No checkpoint file found", flush=True)
    exit(0)
try:
    with checkpoint_file.open("r") as f:
        checkpoint_configs = {line.strip() for line in f if line.strip()}
        log_counts["checkpoint"] = len(checkpoint_configs)
except Exception as err:
    print(f"Error reading {checkpoint_file}: {err}", flush=True)
    exit(0)
print(f"Read {len(checkpoint_configs)} api configs from checkpoint", flush=True)

api_configs = checkpoint_configs.copy()
for log_type, prefix in LOG_PREFIXES.items():
    if log_type == "checkpoint":
        continue
    log_file = TEST_LOG_PATH / f"{prefix}.txt"
    if not log_file.exists():
        continue
    try:
        with log_file.open("r") as f:
            lines = {line.strip() for line in f if line.strip()}
            api_configs -= lines
            log_counts[log_type] = len(lines)
    except Exception as err:
        print(f"Error reading {log_file}: {err}", flush=True)
        exit(0)

if api_configs:
    log_counts["skip"] = len(api_configs)
else:
    print("No skip configs found", flush=True)

for log_type, count in log_counts.items():
    print(f"{log_type}: {count}", flush=True)

if api_configs:
    skip_file = OUTPUT_PATH
    try:
        with skip_file.open("w") as f:
            f.writelines(f"{line}\n" for line in sorted(api_configs))
    except Exception as err:
        print(f"Error writing to {skip_file}: {err}", flush=True)
        exit(0)
    print(f"Write {len(api_configs)} skip api configs to {skip_file}", flush=True)

    checkpoint_count = len(checkpoint_configs)
    checkpoint_configs -= api_configs
    print(f"checkpoint removed: {checkpoint_count - len(checkpoint_configs)}", flush=True)
    print(f"checkpoint remaining: {len(checkpoint_configs)}", flush=True)
    try:
        with checkpoint_file.open("w") as f:
            f.writelines(f"{line}\n" for line in sorted(checkpoint_configs))
    except Exception as err:
        print(f"Error writing {checkpoint_file}: {err}", flush=True)
        exit(0)
    print(
        f"Write {len(checkpoint_configs)} checkpoint api configs to {checkpoint_file}",
        flush=True,
    )
