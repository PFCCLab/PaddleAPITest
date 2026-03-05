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
    "/host_home/wanghuan29/APItest/PaddleAPITest/EB45/EBRL.txt",
]

if __name__ == "__main__":
    for file in file_list:
        api_configs = analyse_configs(file)
        for api_config in tqdm(api_configs):
            tensor_configs = get_tensor_configs(api_config)
            for tensor_config in tensor_configs:
                if tensor_numel(tensor_config) >= 2147483647:
                    print(api_config.config)
                    break
