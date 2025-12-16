from __future__ import annotations

from pathlib import Path

PATH1 = Path("tester/api_config/test_log_cinn_filtered/pass_config.txt")
PATH2 = Path("tester/api_config/test_log_cinn/api_config_pass.txt")

content1 = PATH1.read_text(encoding="utf-8")
config1 = {line.strip() for line in content1.splitlines() if line.strip()}
content2 = PATH2.read_text(encoding="utf-8")
config2 = {line.strip() for line in content2.splitlines() if line.strip()}

if len(config1) > len(config2):
    print(f"len(config1) > len(config2), {len(config1) - len(config2)} lines removed")
    for config in sorted(config1 - config2):
        print(config)
else:
    print(f"len(config1) < len(config2), {len(config2) - len(config1)} lines added")
    for config in sorted(config2 - config1):
        print(config)
