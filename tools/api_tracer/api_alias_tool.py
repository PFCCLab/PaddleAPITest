from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

INPUT_DIR = "tools/api_tracer/trace_output_test_train/baidu/ERNIE-4.5-VL-28B-A3B-PT"


def parse_api(api):
    if ".Tensor." in api:
        return api
    parts = api.rsplit(".", 1)
    if len(parts) == 2 and re.match(r".*\.[A-Z][a-zA-Z0-9]*$", parts[0]):
        return parts[0]
    return api


@staticmethod
def get_alias_apis(input_path: str, yaml_path: str):
    """解析 api_apis.txt 文件, 生成 alias_api.txt 和 excluded_api.txt

    Args:
    - input_path (str): 输入文件路径
    - yaml_path (str): 目标 API 列表的 YAML 文件路径
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"[APIAlias] Not found yaml file: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        target_apis = yaml.safe_load(f)
    print(f"[APIAlias] Successfully loaded {len(target_apis)} target APIs.")

    input_file = Path(input_path) / "api_apis.txt"
    if not input_file.exists():
        raise FileNotFoundError(f"[APIAlias] Not found input file: {input_path}")

    output_path = input_file.parent / "apis.txt"
    output_excluded_path = input_file.parent / "apis_excluded.txt"

    apis = set()
    with input_file.open("r") as f:
        apis = {line.strip() for line in f if line.strip()}
    print(f"[APIAlias] Read {len(apis)} apis from {input_path}")

    alias_apis = set()
    excluded_apis = set()
    for api in apis:
        if not api.startswith("torch."):
            continue
        if api.startswith("torch.Tensor.__"):
            alias_apis.add(api)
            continue
        alias_api = parse_api(api)
        if alias_api not in target_apis:
            excluded_apis.add(alias_api)
            continue
        alias_apis.add(alias_api)

    with output_path.open("w") as f:
        f.writelines(f"{line}\n" for line in sorted(alias_apis))
    print(f"[APIAlias] Write {len(alias_apis)} alias apis to {output_path}")

    with output_excluded_path.open("w") as f:
        f.writelines(f"{line}\n" for line in sorted(excluded_apis))
    print(f"[APIAlias] Write {len(excluded_apis)} excluded apis to {output_excluded_path}")


if __name__ == "__main__":
    get_alias_apis(
        input_path=INPUT_DIR,
        yaml_path="tools/api_tracer/api_list/torch_api_list.yaml",
    )
