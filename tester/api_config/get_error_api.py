from __future__ import annotations


def read_file(file_path):
    with open(file_path, encoding="utf-8") as file:
        return {line.strip() for line in file}


def write_to_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for line in sorted(data):
            file.write(line + "\n")


def get_apiconfig_from_apiname(api_configs, api_names):
    res = set()
    for item in api_configs:
        if item.split("(", 1)[0] in api_names:
            res.add(item)
    return res


############################################################
# 挑选出total_config_file中config配置运行全错的api, 也就是说该api的所有配置中都运行失败
total_config_file = "test_gen.txt"  # 总的配置文件
pass_config_file = "test_gen_pass.txt"  # 通过的配置文件
############################################################

total_config_strings = read_file(total_config_file)
pass_config_strings = read_file(pass_config_file)

total_api_names = set()
for item in total_config_strings:
    api_name = item.split("(", 1)[0]
    total_api_names.add(api_name)
# write_to_file("total_api.txt", total_api_names)
print(
    "总共的congig数量：",
    len(total_config_strings),
    "总共的api数量：",
    len(total_api_names),
)


pass_api_names = set()
for item in pass_config_strings:
    api_name = item.split("(", 1)[0]
    pass_api_names.add(api_name)
# write_to_file("pass_api.txt", pass_api_names)
print(
    "通过的congig数量：",
    len(pass_config_strings),
    "通过的api数量：",
    len(pass_api_names),
)


error_api_names = total_api_names - pass_api_names
error_api_configs = get_apiconfig_from_apiname(total_config_strings, error_api_names)
write_to_file("not_pass_api_config.txt", error_api_configs)
write_to_file("not_pass_api.txt", error_api_names)
print(
    "全部case不通过的congig数量：",
    len(error_api_configs),
    "全部case不通过的api数量：",
    len(error_api_names),
)
