from __future__ import annotations

import random

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
    "tester/api_config/8_big_tensor/big_tensor_1_8.txt",
]

apis_map = {}


class API_info:
    def __init__(self):
        self.count = 0
        self.numel_100 = 0
        self.numel_1000 = 0
        self.numel_10000 = 0
        self.numel_100000 = 0
        self.numel_1000000 = 0
        self.numel_2147483647 = 0
        self.numel_other = 0
        self.numel_100_configs = []
        self.numel_1000_configs = []
        self.numel_10000_configs = []
        self.numel_100000_configs = []
        self.numel_1000000_configs = []
        self.numel_2147483647_configs = []
        self.numel_other_configs = []


if __name__ == "__main__":
    for file in file_list:
        api_configs = analyse_configs(file)
        for api_config in tqdm(api_configs):
            if api_config.api_name not in apis_map:
                apis_map[api_config.api_name] = API_info()
            tensor_configs = get_tensor_configs(api_config)
            numel = 0
            for tensor_config in tensor_configs:
                numel = numel + tensor_numel(tensor_config)
            apis_map[api_config.api_name].count += 1
            if numel < 100:
                apis_map[api_config.api_name].numel_100_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_100 += 1
            elif numel < 1000:
                apis_map[api_config.api_name].numel_1000_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_1000 += 1
            elif numel < 10000:
                apis_map[api_config.api_name].numel_10000_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_10000 += 1
            elif numel < 100000:
                apis_map[api_config.api_name].numel_100000_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_100000 += 1
            elif numel < 1000000:
                apis_map[api_config.api_name].numel_1000000_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_1000000 += 1
            elif numel < 2147483647:
                apis_map[api_config.api_name].numel_2147483647_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_2147483647 += 1
            else:
                apis_map[api_config.api_name].numel_other_configs.append(api_config.config)
                apis_map[api_config.api_name].numel_other += 1

    with open("tester/api_config/10_performance/numel_stat.txt", "w") as f:
        for api_name, api_info in apis_map.items():
            f.write(
                api_name
                + "\t"
                + str(api_info.count)
                + "\t"
                + str(api_info.numel_100)
                + "\t"
                + str(api_info.numel_1000)
                + "\t"
                + str(api_info.numel_10000)
                + "\t"
                + str(api_info.numel_100000)
                + "\t"
                + str(api_info.numel_1000000)
                + "\t"
                + str(api_info.numel_2147483647)
                + "\t"
                + str(api_info.numel_other)
                + "\n"
            )
        f.close()
    with open("tester/api_config/10_performance/case_little.txt", "w") as little:
        with open("tester/api_config/10_performance/case_middle.txt", "w") as middle:
            with open("tester/api_config/10_performance/case_big.txt", "w") as big:
                with open(
                    "tester/api_config/10_performance/numel_used_stat.txt", "w"
                ) as numel_used_stat:
                    for api_name, api_info in apis_map.items():
                        config_count = 500
                        # big
                        big_count = 100 if api_info.numel_other > 100 else api_info.numel_other
                        for config in random.sample(api_info.numel_other_configs, big_count):
                            big.write(str(config) + "\n")
                        config_count -= big_count
                        # middle
                        for config in random.sample(
                            api_info.numel_2147483647_configs,
                            min(config_count, api_info.numel_2147483647),
                        ):
                            middle.write(str(config) + "\n")
                        config_count -= min(config_count, api_info.numel_2147483647)
                        numel_2147483647 = min(config_count, api_info.numel_2147483647)

                        for config in random.sample(
                            api_info.numel_1000000_configs,
                            min(config_count, api_info.numel_1000000),
                        ):
                            middle.write(str(config) + "\n")
                        config_count -= min(config_count, api_info.numel_1000000)
                        numel_1000000 = min(config_count, api_info.numel_1000000)

                        for config in random.sample(
                            api_info.numel_100000_configs,
                            min(config_count, api_info.numel_100000),
                        ):
                            middle.write(str(config) + "\n")
                        config_count -= min(config_count, api_info.numel_100000)
                        numel_100000 = min(config_count, api_info.numel_100000)

                        for config in random.sample(
                            api_info.numel_10000_configs,
                            min(config_count, api_info.numel_10000),
                        ):
                            middle.write(str(config) + "\n")
                        config_count -= min(config_count, api_info.numel_10000)
                        numel_10000 = min(config_count, api_info.numel_10000)
                        # little
                        for config in random.sample(
                            api_info.numel_1000_configs,
                            min(config_count, api_info.numel_1000),
                        ):
                            little.write(str(config) + "\n")
                        config_count -= min(config_count, api_info.numel_1000)
                        numel_1000 = min(config_count, api_info.numel_1000)

                        for config in random.sample(
                            api_info.numel_100_configs,
                            min(config_count, api_info.numel_100),
                        ):
                            little.write(str(config) + "\n")
                        numel_100 = min(config_count, api_info.numel_100)
                        numel_used_stat.write(
                            api_name
                            + "\t"
                            + str(api_info.count)
                            + "\t"
                            + str(numel_100)
                            + "\t"
                            + str(numel_1000)
                            + "\t"
                            + str(numel_10000)
                            + "\t"
                            + str(numel_100000)
                            + "\t"
                            + str(numel_1000000)
                            + "\t"
                            + str(numel_2147483647)
                            + "\t"
                            + str(api_info.numel_other)
                            + "\n"
                        )
                    little.close()
                    middle.close()
                    big.close()
                    numel_used_stat.close()
