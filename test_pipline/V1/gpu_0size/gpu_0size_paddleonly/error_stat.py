from __future__ import annotations

error_configs = set()
error_apis = set()
with open("test_pipline/gpu_0size/gpu_0size_paddleonly/error_log.log", "w") as error_log:
    for i in range(24):
        test_log = f"test_pipline/gpu_0size/gpu_0size_paddleonly/0sizetensor_paddleonly_{i + 1}.log"
        with open(test_log) as file:
            lines = file.readlines()
            is_log_str = False
            log_str = ""
            config = ""
            for line in lines:
                if "test begin" in line:
                    if (
                        "cudaErrorIllegalAddress" in log_str
                        or "cudaErrorLaunchFailure" in log_str
                        or "cuda error" in log_str
                        or "CUDA error" in log_str
                        or "CUDNN error" in log_str
                        or "TID" in log_str
                    ):
                        error_log.write(log_str)
                        error_configs.add(config)
                        error_apis.add(api)
                    log_str = line
                    config = line[line.index("test begin: ") + 12 :]
                    api = config[0 : config.index("(")]
                else:
                    log_str += line

with open("test_pipline/gpu_0size/gpu_0size_paddleonly/error_config.txt", "w") as error_config_file:
    for config in error_configs:
        error_config_file.write(config)
with open("test_pipline/gpu_0size/gpu_0size_paddleonly/error_api.txt", "w") as error_api_file:
    for api in error_apis:
        error_api_file.write(api + "\n")
