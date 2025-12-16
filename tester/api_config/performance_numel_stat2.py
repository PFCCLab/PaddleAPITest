from __future__ import annotations

from config_analyzer import TensorConfig, analyse_configs
from tqdm import tqdm


def is_0_size_tensor(tensor_config):
    return any(i == 0 for i in tensor_config.shape)


def is_0D_tensor(tensor_config):
    return len(tensor_config.shape) == 0


def tensor_numel(tensor_config):
    numel = 1
    for i in tensor_config.shape:
        numel = numel * i
    return numel


def get_tensor_configs(api_config):
    tensor_configs = []
    for arg_config in api_config.args:
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])

    for _key, arg_config in api_config.kwargs.items():
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, (list, tuple)):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
    return tensor_configs


file_list = [
    "tester/api_config/5_accuracy/accuracy_1.txt",
    "tester/api_config/5_accuracy/accuracy_2.txt",
    "tester/api_config/5_accuracy/accuracy_3.txt",
    "tester/api_config/5_accuracy/accuracy_4.txt",
    "tester/api_config/5_accuracy/accuracy_5.txt",
    "tester/api_config/5_accuracy/accuracy_6.txt",
    "tester/api_config/5_accuracy/accuracy_7.txt",
    "tester/api_config/5_accuracy/accuracy_8.txt",
    "tester/api_config/10_performance/EB5_config.txt",
    "tester/api_config/10_performance/EB45_config.txt",
    # "tester/api_config/8_big_tensor/big_tensor_1_8.txt",
]

apis_map = {}


class API_info2:
    def __init__(self):
        self.numel = 0
        self.config = ""


if __name__ == "__main__":
    for file in file_list:
        api_configs = analyse_configs(file)
        for api_config in tqdm(api_configs):
            if api_config.api_name not in apis_map:
                apis_map[api_config.api_name] = []
            tensor_configs = get_tensor_configs(api_config)
            numel = 0
            for tensor_config in tensor_configs:
                numel = numel + tensor_numel(tensor_config)
            api_info = API_info2()
            api_info.numel = numel
            api_info.config = api_config.config
            apis_map[api_config.api_name].append(api_info)

    with open("tester/api_config/10_performance/top_three_case.txt", "w") as top_three_case:
        for _api_name, api_infos in apis_map.items():

            def sort_func(x):
                return x.numel

            api_infos.sort(key=sort_func, reverse=True)
            cnt = 3 if len(api_infos) >= 3 else len(api_infos)
            for i in range(cnt):
                top_three_case.write(str(api_infos[i].config) + "\n")
        top_three_case.close()
