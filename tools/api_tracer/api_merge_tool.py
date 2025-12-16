from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


@staticmethod
def get_merged_model_apis(
    input_path: str,
    output_path: str,
    sheet_name: str | None = None,
    model_groups: dict[str, list[str]] | None = None,
    yaml_paths: dict[str, str] | None = None,
):
    """从 XLSX 或 CSV 文件中读取 API 数据, 合并并分析 API

    Args:
    - excel_path (str): 输入的 XLSX 或 CSV 文件路径
    - output_path (str): 输出的 Excel 或 TXT 文件路径（根据后缀自动判断）
    - sheet_name (str, optional): 要读取的 Excel 工作表名称. 若为 None, 则读取第一个工作表
    - model_groups (Dict[str, List[str]], optional): 模型分组字典
        例如：{'核心模型': ['model_A'], '次要模型': ['model_B']}
        如果提供, 将按此字典顺序分组并排序模型列. 如果为 None, 则无 "首次出现分组" 列
    - yaml_path (Dict[str, str], optional): API 列表的 YAML 文件路径字典, 键用于指定列名称
        例如：{'是否在总表': 'tools/api_tracer/api_list/torch_api_list.yaml'}
    """
    # Read the Excel file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"[APIMerge] Not found input file: {input_path}")

    if input_file.suffix.lower() == ".xlsx":
        df = pd.read_excel(input_file, sheet_name=sheet_name, header=None)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file, header=None)  # No header, as it's multi-level
        if sheet_name:
            print(f"[APIMerge] `sheet_name={sheet_name}` will be ignored for csv")
    else:
        raise ValueError(
            f"[APIMerge] Unsupported input file type: {input_file.suffix}. Only .xlsx and .csv are supported"
        )

    # Load APIs from YAML
    named_apis_set = {}
    if yaml_paths is not None:
        for name, path in yaml_paths.items():
            try:
                with open(path, encoding="utf-8") as f:
                    apis = yaml.safe_load(f)
                if not isinstance(apis, list):
                    print(f"[APIMerge] {path} should be a list, but got {type(apis)}")
                    continue
                print(f"[APIMerge] Successfully loaded {len(apis)} APIs for '{name}' from {path}")
                named_apis_set[name] = set(apis)
            except Exception as e:
                print(f"[APIMerge] Error loading {path}: {e}")
                continue

    # Process the dataframe to extract models and their API sets
    # Assuming structure: model names in row 0, spanning 3 columns each (merged cells appear as value in first, NaN in others)
    # Row 1: 推理 训练 总和 repeated
    # Row 2+: APIs in columns

    # Find model names and column groups
    models = []
    col_groups = []
    current_model = None
    col_start = 0
    for col in range(df.shape[1]):  # type: ignore[reportGeneralTypeIssues]
        val = df.iloc[0, col]  # type: ignore[reportGeneralTypeIssues]
        if pd.notna(val) and str(val).strip():
            if current_model:
                col_groups.append((col_start, col))
            current_model = str(val).strip()
            models.append(current_model)
            col_start = col
    if current_model:
        col_groups.append((col_start, df.shape[1]))  # type: ignore[reportGeneralTypeIssues]

    dup_models = []
    for model in models:
        if models.count(model) > 1:
            dup_models.append(model)
    for model_name in dup_models:
        print(f"[APIMerge] Found duplicate model: {model_name}")

    # For each model, assume 3 columns: inference, train, total
    apis_per_model = {}  # model_name -> set of APIs (merged from all three columns)
    for i, model in enumerate(models):
        start_col, end_col = col_groups[i]
        if end_col - start_col != 3:
            raise ValueError(
                f"[APIMerge] Invalid number of columns for model {model} - should be 3, but is {end_col - start_col}."
                f"Please check the structure of your file."
            )

        model_api_slices = df.iloc[2:, start_col:end_col].values.flatten()  # type: ignore[reportGeneralTypeIssues]
        model_apis = {
            str(api).strip() for api in model_api_slices if pd.notna(api) and str(api).strip()
        }
        apis_per_model[model] = model_apis
    print(f"[APIMerge] Successfully read {len(models)} models and their APIs.")

    if model_groups:
        all_grouped_models = {
            m for models_in_group in model_groups.values() for m in models_in_group
        }
        all_data_models = set(apis_per_model.keys())

        missing_models = all_grouped_models - all_data_models
        for missing_model in missing_models:
            print(f"[APIMerge] Model '{missing_model}' is in groups but not found in data.")

        unassigned_models = all_data_models - all_grouped_models
        for unassigned_model in unassigned_models:
            print(
                f"[APIMerge] Model '{unassigned_model}' is not in any group but found in data.",
                "It will be displayed in the final table.",
            )

    all_apis = set().union(*apis_per_model.values())
    print(f"\n[APIMerge] Total API count: {len(all_apis)}")

    api_to_first_group = {}
    if model_groups:
        cumulative_apis = set()
        print("--- APIs in each group ---")
        for group_name, group_models in model_groups.items():
            current_group_apis = set().union(
                *(apis_per_model[model] for model in group_models if model in apis_per_model)
            )

            newly_introduced_apis = current_group_apis - cumulative_apis
            for api in newly_introduced_apis:
                api_to_first_group[api] = group_name

            cumulative_apis.update(current_group_apis)
            print(f"'{group_name}' (and previous groups) API count : {len(cumulative_apis)}")
        print("'未分组' API count : ", len(all_apis) - len(cumulative_apis), "\n")

    ordered_model_names = []
    if model_groups:
        for group_models_list in model_groups.values():
            ordered_model_names.extend([model for model in group_models_list if model in models])
        ordered_model_names.extend([m for m in models if m not in ordered_model_names])
    else:
        ordered_model_names = models

    sorted_apis = sorted(all_apis)
    data = []
    for api in sorted_apis:
        row = [api]
        if api_to_first_group:
            first_group = api_to_first_group.get(api, "未分组")
            row.append(first_group)
        for apis_set in named_apis_set.values():
            if api in apis_set:
                row.append("是")
            else:
                row.append("否")
        for model_name in ordered_model_names:
            row.append("是" if api in apis_per_model[model_name] else "否")
        data.append(row)

    columns = ["API"]
    if model_groups:
        columns.append("首次出现分组")
    columns.extend(named_apis_set.keys())
    columns.extend(ordered_model_names)
    result_df = pd.DataFrame(data, columns=columns)

    if model_groups:
        group_order = [*list(model_groups.keys()), "未分组"]
        result_df["首次出现分组"] = pd.Categorical(
            result_df["首次出现分组"], categories=group_order, ordered=True
        )
        result_df = result_df.sort_values(by=["首次出现分组", "API"], ignore_index=True)
    else:
        result_df = result_df.sort_values(by=["API"], ignore_index=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.suffix.lower() == ".txt":
        result_df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")
    elif output_file.suffix.lower() == ".xlsx":
        result_df.to_excel(output_file, index=False, engine="openpyxl")
    else:
        raise ValueError(
            f"[APIMerge] Unsupported output format: {output_file.suffix}. Please use .xlsx or .txt."
        )
    print(f"[APIMerge] Done. Merged API table saved to: {output_file}")


if __name__ == "__main__":
    MODEL_GROUPS = {
        "13个": [
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-57B-A14B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-30B-A3B",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-4-Maverick-17B-128E",
            "llava-hf/llava-1.5-7b-hf",
            "deepseek-ai/DeepSeek-V2-Lite",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "baidu/ERNIE-4.5-0.3B-PT",
            "baidu/ERNIE-4.5-21B-A3B-PT",
            "baidu/ERNIE-4.5-VL-28B-A3B-PT",
        ],
        "19个": [
            "zai-org/GLM-4.1V-9B-Thinking",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "black-forest-labs/FLUX.1-dev",
            "ByteDance/Dolphin",
            "echo840/MonkeyOCR",
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ],
        "35个": [
            "Salesforce/blip2-opt-2.7b",
            "OpenGVLab/InternVL3-1B",
            "moonshotai/Kimi-K2-Instruct",
            "moonshotai/Kimi-VL-A3B-Instruct",
            "zai-org/GLM-4.5",
            "mistralai/Magistral-Small-2507",
            "tencent/HunyuanWorld-1",
            "MiniMaxAI/MiniMax-M1-40k",
            "state-spaces/mamba2-2.7b",
            "RWKV/RWKV7-Goose-World3-2.9B-HF",
            "deepseek-ai/Janus-Pro-1B",
            "Kwai-Keye/Keye-VL-8B-Preview",
            "XiaomiMiMo/MiMo-VL-7B-SFT",
            "ByteDance-Seed/BAGEL-7B-MoT",
            "jieliu/SD3.5M-FlowGRPO-GenEval",
            "Skywork/UniPic2-Metaquery-9B",
            "VAPO",
        ],
    }

    yaml_paths = {
        "是否在总表中": "tools/api_tracer/api_list/torch_api_list.yaml",
        "是否在静态采集中": "tools/api_tracer/api_list/torch_api_static.yaml",
    }

    get_merged_model_apis(
        input_path="tools/api_tracer/Torch核心API统计(动态扫描).xlsx",
        output_path="tools/api_tracer/torch_merge_apis.xlsx",
        sheet_name="各模型API",
        model_groups=MODEL_GROUPS,
        yaml_paths=yaml_paths,
    )
