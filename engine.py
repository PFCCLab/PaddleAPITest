from __future__ import annotations

import os

os.environ["FLAGS_USE_SYSTEM_ALLOCATOR"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import argparse
from datetime import datetime

import paddle
import torch
from tester import (
    APIConfig,
    APITestAccuracy,
    APITestAccuracyStable,
    APITestCINNVSDygraph,
    APITestCustomDeviceVSCPU,
    APITestPaddleGPUPerformance,
    APITestPaddleOnly,
    APITestPaddleTorchGPUPerformance,
    APITestTorchGPUPerformance,
    set_cfg,
)
from tester.api_config.log_writer import close_process_files, read_log, write_to_log


def parse_bool(value):
    if isinstance(value, str):
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        elif value in ["false", "0", "no", "n"]:
            return False
    else:
        raise ValueError(f"Invalid boolean value: {value} parsed from command line")


def main():
    parser = argparse.ArgumentParser(description="API Test")
    parser.add_argument(
        "--api_config_file",
        default="",
    )
    parser.add_argument(
        "--api_config",
        default="",
    )
    parser.add_argument(
        "--paddle_only",
        default=False,
    )
    parser.add_argument(
        "--paddle_cinn",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        default=False,
    )
    parser.add_argument(
        "--paddle_gpu_performance",
        default=False,
    )
    parser.add_argument(
        "--torch_gpu_performance",
        default=False,
    )
    parser.add_argument(
        "--paddle_torch_gpu_performance",
        default=False,
    )
    parser.add_argument(
        "--accuracy_stable",
        default=False,
    )
    parser.add_argument(
        "--paddle_custom_device",
        default=False,
    )
    parser.add_argument(
        "--test_amp",
        default=False,
    )
    parser.add_argument(
        "--id",
        default="",
        type=str,
    )
    parser.add_argument(
        "--test_cpu",
        type=parse_bool,
        default=False,
        help="Whether to test CPU mode",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for accuracy tests",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for accuracy tests",
    )
    parser.add_argument(
        "--exit_on_error",
        type=parse_bool,
        default=False,
        help="Whether to exit the process when a paddle_error occurs.",
    )
    options = parser.parse_args()
    set_cfg(options)  # Set the command line arguments in the config module

    if options.test_cpu:
        paddle.device.set_device("cpu")

    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy
    elif options.paddle_gpu_performance:
        paddle.framework.set_flags({"FLAGS_use_system_allocator": False})
        paddle.framework.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})
        test_class = APITestPaddleGPUPerformance
    elif options.torch_gpu_performance:
        test_class = APITestTorchGPUPerformance
    elif options.paddle_torch_gpu_performance:
        paddle.set_flags({"FLAGS_use_system_allocator": False})
        paddle.framework.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})
        test_class = APITestPaddleTorchGPUPerformance
    elif options.accuracy_stable:
        test_class = APITestAccuracyStable
    elif options.paddle_custom_device:
        test_class = APITestCustomDeviceVSCPU

    if options.api_config != "":
        options.api_config = options.api_config.strip()
        print("test begin:", options.api_config, flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print("[config parse error]", options.api_config, str(err))
            return

        if options.accuracy:
            case = test_class(
                api_config,
                test_amp=options.test_amp,
                atol=options.atol,
                rtol=options.rtol,
                exit_on_error=options.exit_on_error,
            )
        else:
            case = test_class(api_config, test_amp=options.test_amp)
        case.test()
        case.clear_tensor()
        del case
        del api_config
        if (
            not options.paddle_gpu_performance
            and not options.torch_gpu_performance
            and not options.paddle_torch_gpu_performance
        ):
            torch.cuda.empty_cache()
            paddle.device.cuda.empty_cache()
    elif options.api_config_file != "":
        finish_configs = read_log("checkpoint")
        with open(options.api_config_file) as f:
            api_configs = {line.strip() for line in f if line.strip()}
        api_configs = api_configs - finish_configs
        api_configs = sorted(api_configs)
        for api_config_str in api_configs:
            write_to_log("checkpoint", api_config_str)

            print(datetime.now(), "test begin:", api_config_str, flush=True)
            try:
                api_config = APIConfig(api_config_str)
            except Exception as err:
                print("[config parse error]", api_config_str, str(err))
                continue

            if options.accuracy:
                case = test_class(
                    api_config,
                    test_amp=options.test_amp,
                    atol=options.atol,
                    rtol=options.rtol,
                )
            else:
                case = test_class(api_config, test_amp=options.test_amp)
            try:
                case.test()
            except Exception as err:
                if (
                    "CUDA error" in str(err)
                    or "memory corruption" in str(err)
                    or "CUDA out of memory" in str(err)
                    or "Out of memory error" in str(err)
                ):
                    exit(0)
            case.clear_tensor()
            del case
            del api_config
            if (
                not options.paddle_gpu_performance
                and not options.torch_gpu_performance
                and not options.paddle_torch_gpu_performance
            ):
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()

        # elif options.api_config_file != "":
        #     with open(options.api_config_file, "r") as f:
        #         configs = f.readlines()
        #         f.close()

        #     for config in configs:
        #         config = config.replace("\n", "")
        #         cmd = ["python", "engine.py", "--api_config=" + config]
        #         res = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #         # res.wait()
        #         print(str(res.stdout.read(), encoding="utf-8"), flush=True)
        #         print(str(res.stderr.read(), encoding="utf-8"), flush=True)
        #         # res.terminate()

    close_process_files()


if __name__ == "__main__":
    main()
