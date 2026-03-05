from __future__ import annotations

import re

error_configs = set()
error_apis = set()


def process_log_entry(log_str, config, api, error_log, error_configs, error_apis):
    """Helper function to process a single log entry."""
    if not log_str or not config or not api:
        return

    is_error = False
    error_keywords = [
        "accuracy error",
        "paddle error",
        "cudaErrorIllegalAddress",
        "cudaErrorLaunchFailure",
        "cuda error",
        "CUDA error",
        "CUDNN error",
        "TID",
    ]
    for keyword in error_keywords:
        if keyword in log_str:
            is_error = True
            break

    if is_error:
        if "paddle error" in log_str:
            log_lines = log_str.strip().splitlines()
            if len(log_lines) > 2 and "(NotFound) The kernel with key" in log_lines[2]:
                print(f"Skipping (Filter 1: Paddle NotFound) for API: {api}")
                return
        if "kernel_factory.cc:317" in log_str:
            print(f"Skipping (Filter 2: kernel_factory.cc) for API: {api}")
            return
        error_log.write(log_str + "\n" if not log_str.endswith("\n") else log_str)
        error_configs.add(config)
        error_apis.add(api)


with open("test_pipline/cpu_accuracy/error_log.log", "w") as error_log_file:
    for i in range(14):
        test_log = f"test_pipline/cpu_accuracy/cpu_accuracy_{i + 1}.log"
        print(f"Processing file: {test_log}")
        try:
            with open(test_log) as file:
                lines = file.readlines()
                current_log_str = ""
                current_config = ""
                current_api = ""
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    if "test begin:" in line:
                        process_log_entry(
                            current_log_str,
                            current_config,
                            current_api,
                            error_log_file,
                            error_configs,
                            error_apis,
                        )
                        current_log_str = line + "\n"
                        try:
                            config_part = line.split("test begin: ", 1)[1]
                            current_config = config_part.strip()
                            api_match = re.match(r"([^(]+)\(", config_part)
                            if api_match:
                                current_api = api_match.group(1).strip()
                            else:
                                current_api = "UnknownAPI"
                                print(f"Warning: Could not parse API from config: {current_config}")
                        except IndexError:
                            print(f"Warning: Could not parse config from line: {line}")
                            current_config = "UnknownConfig"
                            current_api = "UnknownAPI"
                            current_log_str = ""

                    elif current_log_str:
                        current_log_str += line + "\n"
                process_log_entry(
                    current_log_str,
                    current_config,
                    current_api,
                    error_log_file,
                    error_configs,
                    error_apis,
                )

        except FileNotFoundError:
            print(f"Warning: File not found {test_log}")
        except Exception as e:
            print(f"Error processing file {test_log}: {e}")


print("\nWriting error config and API files...")
with open("test_pipline/cpu_accuracy/error_config.txt", "w") as error_config_file:
    for config_entry in sorted(error_configs):
        error_config_file.write(config_entry + "\n")

with open("test_pipline/cpu_accuracy/error_api.txt", "w") as error_api_file:
    for api_entry in sorted(error_apis):
        error_api_file.write(api_entry + "\n")

print("Processing complete.")
