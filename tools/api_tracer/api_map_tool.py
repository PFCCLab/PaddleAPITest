from __future__ import annotations

import os

import pandas as pd
import yaml


def _load_api_data(filepath: str) -> list[str] | dict[str, list[str]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[APIMap] Not found API list file: '{filepath}'")

    _, extension = os.path.splitext(filepath.lower())

    if extension in [".yaml", ".yml"]:
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            elif isinstance(data, dict):
                for key, value in data.items():
                    if not value:
                        data[key] = []
                    elif not isinstance(value, list):
                        raise TypeError(
                            f"[APIMap] The value for key '{key}' in '{filepath}' is not a list."
                        )
                return data
            else:
                raise TypeError(f"[APIMap] YAML file '{filepath}' is not a list or a dictionary.")
    elif extension == ".txt":
        with open(filepath, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(
            f"[APIMap] Unsupported API list file format: '{extension}'. Please use .yaml or .txt."
        )


@staticmethod
def get_mapped_model_apis(
    torch_static_path: str,
    torch_dynamic_path: str,
    paddle_dynamic_path: str,
    mapping_table_path: str,
    category_priority: list[str],
    output_path: str,
):
    """根据给定的 API 列表和映射表文件, 生成详细的 API 映射报告

    Args:
        torch_static_path (str): Torch 静态API列表的文件路径 (.yaml 或 .txt)
        torch_dynamic_path (str): Torch 动态API列表的文件路径 (.yaml 或 .txt)
        paddle_dynamic_path (str): Paddle 动态API列表的文件路径 (.yaml 或 .txt)
        mapping_table_path (str): 包含API映射关系的全量表格路径 (.xlsx)
        output_path (str): 生成报告的输出文件路径, 根据后缀决定格式 (.xlsx, .csv, .txt)
    """
    print("[APIMap] Start mapping...")

    if not os.path.exists(mapping_table_path):
        raise FileNotFoundError(f"[APIMap] Not found mapping table: '{mapping_table_path}'")

    mapping_df = pd.read_excel(mapping_table_path)
    if "Pytorch" not in mapping_df.columns or "Paddle" not in mapping_df.columns:
        raise ValueError("[APIMap] 'Pytorch' and 'Paddle' columns must exist in the mapping table.")
    print(f"[APIMap] Successfully loaded Mapping table: '{mapping_table_path}'")

    # Load API lists
    torch_static_data = _load_api_data(torch_static_path)
    torch_dynamic_data = _load_api_data(torch_dynamic_path)

    if not isinstance(torch_static_data, dict):
        raise TypeError(f"'{torch_static_path}' is not in dictionary format.")
    if not isinstance(torch_dynamic_data, dict):
        raise TypeError(f"'{torch_dynamic_path}' is not in dictionary format.")

    priority_dict = {category: idx for idx, category in enumerate(category_priority)}
    max_priority = len(category_priority)

    torch_api_category_map = {}
    torch_static_list = []
    for category, apis in torch_static_data.items():
        torch_static_list.extend(apis)
        current_priority = priority_dict.get(category, max_priority)
        for api in apis:
            if api not in torch_api_category_map or current_priority < priority_dict.get(
                torch_api_category_map[api], max_priority
            ):
                torch_api_category_map[api] = category

    torch_dynamic_list = []
    for category, apis in torch_dynamic_data.items():
        torch_dynamic_list.extend(apis)
        current_priority = priority_dict.get(category, max_priority)
        for api in apis:
            if api not in torch_api_category_map or current_priority < priority_dict.get(
                torch_api_category_map[api], max_priority
            ):
                torch_api_category_map[api] = category

    torch_static_set = set(torch_static_list)
    print(
        f"[APIMap] Successfully loaded {len(torch_static_set)} Torch Static APIs from {len(torch_static_data)} categories."
    )
    torch_dynamic_set = set(torch_dynamic_list)
    print(
        f"[APIMap] Successfully loaded {len(torch_dynamic_set)} Torch Dynamic APIs from {len(torch_dynamic_data)} categories."
    )

    paddle_dynamic_list = _load_api_data(paddle_dynamic_path)
    paddle_dynamic_set = set(paddle_dynamic_list)
    print(f"[APIMap] Successfully loaded {len(paddle_dynamic_set)} Paddle Dynamic APIs.")

    source_torch_apis = torch_static_set | torch_dynamic_set
    source_paddle_apis = paddle_dynamic_set

    torch_to_paddle_map = (
        pd.Series(mapping_df["Paddle"].values, index=mapping_df["Pytorch"]).dropna().to_dict()
    )
    paddle_to_torch_map = (
        pd.Series(mapping_df["Pytorch"].values, index=mapping_df["Paddle"]).dropna().to_dict()
    )

    report_data = []
    processed_paddle_apis = set()

    # From Torch to Paddle
    for torch_api in sorted(source_torch_apis):
        paddle_api = torch_to_paddle_map.get(torch_api)
        category = torch_api_category_map.get(torch_api, "未分类")
        row = {
            "类别": category,
            "TorchAPI": torch_api,
            "PaddleAPI": paddle_api,
            "Torch": "有" if torch_api in source_torch_apis else "无",
            "Torch静": "是" if torch_api in torch_static_set else "否",
            "Torch动": "是" if torch_api in torch_dynamic_set else "否",
            "Paddle动": ("是" if paddle_api and paddle_api in source_paddle_apis else "否"),
        }
        report_data.append(row)
        if paddle_api and paddle_api in source_paddle_apis:
            processed_paddle_apis.add(paddle_api)

    # From Paddle to Torch
    remaining_paddle_apis = source_paddle_apis - processed_paddle_apis
    for paddle_api in sorted(remaining_paddle_apis):
        torch_api = paddle_to_torch_map.get(paddle_api)
        row = {
            "类别": "无",
            "TorchAPI": torch_api,
            "PaddleAPI": paddle_api,
            "Torch": "无",
            "Torch静": "否",
            "Torch动": "否",
            "Paddle动": "是",
        }
        report_data.append(row)

    result_df = pd.DataFrame(report_data).fillna("无")
    result_df.sort_values(by=["类别", "TorchAPI", "PaddleAPI"], inplace=True, ignore_index=True)

    def get_mapping_status(row):
        torch_in_source = row["Torch静"] == "是" or row["Torch动"] == "是"
        paddle_in_source = row["Paddle动"] == "是"
        torch_exists = row["TorchAPI"] != "无"
        paddle_exists = row["PaddleAPI"] != "无"

        if torch_in_source and paddle_in_source:
            return "已映射"
        elif (torch_in_source and paddle_exists and not paddle_in_source) or (
            paddle_in_source and torch_exists and not torch_in_source
        ):
            return "有映射但无采集"
        elif torch_in_source and not paddle_exists:
            return "无对应Paddle API"
        elif paddle_in_source and not torch_exists:
            return "无对应Torch API"
        else:
            return "未知状态"  # It shouldn't reach here

    result_df["映射状态"] = result_df.apply(get_mapping_status, axis=1)

    final_columns = [
        "类别",
        "TorchAPI",
        "PaddleAPI",
        "Torch",
        "Torch静",
        "Torch动",
        "Paddle动",
        "映射状态",
    ]
    result_df = result_df[final_columns]

    _, output_extension = os.path.splitext(output_path.lower())

    if output_extension == ".xlsx":
        result_df.to_excel(output_path, index=False)
    elif output_extension == ".csv":
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif output_extension == ".txt":
        result_df.to_csv(output_path, index=False, sep="\t", encoding="utf-8")
    else:
        raise ValueError(
            f"[APIMap] Unsupported output format: '{output_extension}'. Please use .xlsx, .csv, or .txt."
        )

    print(f"[APIMap] Done. API mapping report saved to: '{output_path}'")


# --- 主执行块 (用于演示) ---
if __name__ == "__main__":
    MAPPING_FILE = "tools/api_tracer/api_list/torch_paddle_mapping.xls"
    TORCH_STATIC_FILE = "tools/api_tracer/api_list/torch_api_static.yaml"
    TORCH_DYNAMIC_FILE = "tools/api_tracer/api_list/torch_api_dynamic.yaml"
    PADDLE_DYNAMIC_FILE_TXT = "tools/api_tracer/api_list/paddle_api_dynamic.yaml"
    category_priority = ["13个", "19个", "35个"]

    get_mapped_model_apis(
        torch_static_path=TORCH_STATIC_FILE,
        torch_dynamic_path=TORCH_DYNAMIC_FILE,
        paddle_dynamic_path=PADDLE_DYNAMIC_FILE_TXT,
        mapping_table_path=MAPPING_FILE,
        category_priority=category_priority,
        output_path="tools/api_tracer/api_report.xlsx",
    )
