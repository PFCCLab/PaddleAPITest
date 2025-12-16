#!/bin/env python3
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
from __future__ import annotations

import os

# 简洁目录
output_dir = "errors"
os.makedirs(output_dir, exist_ok=True)

error_configs = set()
error_apis = set()

# 保存错误日志
with open(os.path.join(output_dir, "error_log.log"), "w") as error_log:
    worker_logs_dir = "worker_logs"
    for filename in os.listdir(worker_logs_dir):
        if filename.startswith("worker_") and filename.endswith("_log.txt"):
            filepath = os.path.join(worker_logs_dir, filename)
            with open(filepath) as file:
                lines = file.readlines()
                log_str = ""
                config = ""
                api = ""
                inside_task = False

                for line in lines:
                    if "[Worker" in line and "Processing Task" in line:
                        # 遇到新的Task，先检查上一个Task是否是出错的
                        if log_str and (
                            "[torch error]" in log_str
                            or "[accuracy error]" in log_str
                            or "[cuda error]" in log_str
                            or "[paddle error]" in log_str
                            or "FatalError" in log_str
                            or "cudaErrorIllegalAddress" in log_str
                            or "cudaErrorLaunchFailure" in log_str
                            or "CUDA error" in log_str
                            or "CUDNN error" in log_str
                            or "TID" in log_str
                            or "Completed Task" not in log_str
                        ):
                            error_log.write(log_str + "\n")
                            error_configs.add(config)
                            error_apis.add(api)

                        # 开始新的Task
                        log_str = line
                        inside_task = True

                        # 提取config，比如 "paddle.nn.functional.rrelu(...)"
                        try:
                            config = line.split("Processing Task")[1].split(":")[1].strip()
                            api = config.split("(")[0].strip()
                        except Exception:
                            config = ""
                            api = ""
                    else:
                        if inside_task:
                            log_str += line

                # 文件结束，处理最后一个Task
                if log_str and (
                    "[torch error]" in log_str
                    or "[accuracy error]" in log_str
                    or "[cuda error]" in log_str
                    or "[paddle error]" in log_str
                    or "FatalError" in log_str
                    or "cudaErrorIllegalAddress" in log_str
                    or "cudaErrorLaunchFailure" in log_str
                    or "CUDA error" in log_str
                    or "CUDNN error" in log_str
                    or "TID" in log_str
                    or "Completed Task" not in log_str
                ):
                    error_log.write(log_str + "\n")
                    error_configs.add(config)
                    error_apis.add(api)

# 保存出错的config
with open(os.path.join(output_dir, "error_config.txt"), "w") as error_config_file:
    for config in error_configs:
        error_config_file.write(config + "\n")

# 保存出错的API
with open(os.path.join(output_dir, "error_api.txt"), "w") as error_api_file:
    for api in error_apis:
        error_api_file.write(api + "\n")
