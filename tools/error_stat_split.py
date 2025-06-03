# test_log 一键整理小工具（engineV2有序版）：分类版 log_digester_lite + get_api_set + get_api_config_set
# @author: cangtianhuang
# 整理效果：{log_prefix}_api.txt + {log_prefix}_config.txt + {log_prefix}_log.log ({log_prefix} in LOG_PREFIXES)

from collections import defaultdict
from pathlib import Path
import re

TEST_LOG_PATH = Path("tester/api_config/test_log_cpu")
OUTPUT_PATH = TEST_LOG_PATH
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

LOG_PREFIXES = {
    "pass": "api_config_pass",
    "paddle_error": "api_config_paddle_error",
    "torch_error": "api_config_torch_error",
    "paddle_to_torch_failed": "api_config_paddle_to_torch_failed",
    "accuracy_error": "api_config_accuracy_error",
    "timeout": "api_config_timeout",
    "crash": "api_config_crash",
    "oom": "api_config_oom",
    "skip": "api_config_skip",
}

# log_digester_lite
logs = []
in_test_block = False
current_content = []

LOG_PATH = TEST_LOG_PATH / "log_inorder.log"
try:
    with LOG_PATH.open("r") as f:
        input_text = f.read()
except Exception as err:
    print(f"Error reading {LOG_PATH}: {err}", flush=True)
    exit(1)

for line in input_text.split("\n"):
    if "gpu_resources.cc" in line or "Waiting for available memory" in line:
        continue

    if "test begin" in line:
        if in_test_block and current_content:
            logs.append("\n".join(current_content))
        in_test_block = True
        current_content = [line]
        continue

    if "Worker PID" in line:
        if in_test_block and current_content:
            logs.append("\n".join(current_content))
        in_test_block = False
        current_content = []
        continue

    if in_test_block:
        current_content.append(line)

if current_content:
    logs.append("\n".join(current_content))

log_counts = defaultdict(set)
for prefix, file_name in LOG_PREFIXES.items():
    file_path = TEST_LOG_PATH / f"{file_name}.txt"
    if not file_path.exists():
        continue
    try:
        with open(file_path, "r") as f:
            lines = set(line.strip() for line in f if line.strip())
            log_counts[prefix] = lines
            print(f"Read {len(lines)} log(s) from {file_path}", flush=True)
    except Exception as err:
        print(f"Error reading {file_path}: {err}", flush=True)

def get_sort_key(content):
    lines = content.split("\n")
    match = re.search(r"test begin: (.*)$", lines[0])
    if match:
        return match.group(1).strip()
    return ""

key_logs = defaultdict(dict)
for content in logs:
    key = get_sort_key(content)
    if not key:
        continue
    prefix = "unknown"
    for p, s in log_counts.items():
        if key in s:
            prefix = p
            break
    key_logs[prefix][key] = content

if not key_logs:
    print("No logs found", flush=True)
    exit(0)

for prefix, contents in key_logs.items():
    output_log = OUTPUT_PATH / f"{prefix}_log.log"
    try:
        with open(output_log, "w") as f:
            for key in sorted(contents.keys()):
                content = contents[key]
                f.write(content + "\n\n")
    except Exception as err:
        print(f"Error writing {output_log}: {err}", flush=True)
    print(f"Read and write {len(contents)} log(s) for {prefix}", flush=True)

    # get_api_set + get_api_config_set

    api_names = set()
    api_configs = set()
    for line in log_counts[prefix]:
        line = line.strip()
        if line:
            api_name = line.split("(", 1)[0]
            api_names.add(api_name)
            api_configs.add(line)
    print(f"Read {len(api_names)} api(s) for {prefix}", flush=True)
    print(f"Read {len(api_configs)} api config(s) for {prefix}", flush=True)

    # get_api_set
    API_OUTPUT_PATH = OUTPUT_PATH / f"{prefix}_api.txt"
    try:
        with open(API_OUTPUT_PATH, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(api_names))
    except Exception as err:
        print(f"Error writing {API_OUTPUT_PATH}: {err}", flush=True)
    print(f"Write {len(api_names)} api(s) for {prefix}", flush=True)

    # get_api_config_set
    CONFIG_OUTPUT_PATH = OUTPUT_PATH / f"{prefix}_config.txt"
    try:
        with open(CONFIG_OUTPUT_PATH, "w") as f:
            f.writelines(f"{line}\n" for line in sorted(api_configs))
    except Exception as err:
        print(f"Error writing {CONFIG_OUTPUT_PATH}: {err}", flush=True)
    print(f"Write {len(api_configs)} api config(s) for {prefix}", flush=True)
