from __future__ import annotations

import argparse


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="从日志中提取指定API的错误信息(出错的配置和日志信息)"
    )

    # 添加命令行参数
    parser.add_argument("--input-log", type=str, required=True, help="输入日志文件路径")
    parser.add_argument("--api", type=str, required=True, help="要查找的目标API")
    parser.add_argument(
        "--output-error-log",
        type=str,
        default="api_error_log.txt",
        help="目标API的错误日志输出路径（默认: %(default)s）",
    )

    parser.add_argument(
        "--output-error-config",
        type=str,
        default="api_error_config.txt",
        help="目标API的错误配置输出路径（默认: %(default)s）",
    )

    # 解析参数
    args = parser.parse_args()

    error_configs = set()
    target_api = args.api

    # 处理错误日志
    with open(args.output_error_log, "w") as error_log, open(args.input_log) as file:
        lines = file.readlines()
        log_str = ""
        config = ""
        for line in lines:
            if "test begin" in line:
                # 检测错误关键词
                if (
                    any(
                        keyword in log_str
                        for keyword in [
                            "accuracy error",
                            "paddle error",
                            "cudaErrorIllegalAddress",
                            "cudaErrorLaunchFailure",
                            "cuda error",
                            "CUDA error",
                            "CUDNN error",
                            "TID",
                            "PID",
                        ]
                    )
                    and target_api in log_str
                ):
                    error_log.write(log_str)
                    error_configs.add(config)
                # 提取配置和API信息
                log_str = line
                config = line[line.index("test begin: ") + 12 :]
                config[0 : config.index("(")] if "(" in config else config
            else:
                log_str += line

    # 写入错误配置
    with open(args.output_error_config, "w") as error_config_file:
        error_config_file.write("\n".join(error_configs))


if __name__ == "__main__":
    main()
