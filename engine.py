import argparse
from datetime import datetime

from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
                    APITestPaddleOnly, set_cfg)
from tester.api_config.log_writer import read_log, write_to_log
import torch
import paddle

def main():
    parser = argparse.ArgumentParser(
        description='API Test'
    )
    parser.add_argument(
        '--api_config_file',
        default="",
    )
    parser.add_argument(
        '--api_config',
        default="",
    )
    parser.add_argument(
        '--paddle_only',
        default=False,
    )
    parser.add_argument(
        '--paddle_cinn',
        default=False,
    )
    parser.add_argument(
        '--accuracy',
        default=False,
    )
    parser.add_argument(
        '--test_amp',
        default=False,
    )
    parser.add_argument(
        '--id',
        default="",
        type=str,
    )
    options = parser.parse_args()
    set_cfg(options)  # Set the command line arguments in the config module
    
    test_class = APITestAccuracy
    if options.paddle_only:
        test_class = APITestPaddleOnly
    elif options.paddle_cinn:
        test_class = APITestCINNVSDygraph
    elif options.accuracy:
        test_class = APITestAccuracy

    if options.api_config != "":
        print("test begin:", options.api_config, flush=True)
        try:
            api_config = APIConfig(options.api_config)
        except Exception as err:
            print("[config parse error]", options.api_config, str(err))
            return

        case = test_class(api_config, options.test_amp)
        case.test()
        case.clear_tensor()
        del case
        del api_config
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()
    elif options.api_config_file != "":
        finish_configs = read_log("checkpoint")
        with open(options.api_config_file, "r") as f:
            api_configs = set(line.strip() for line in f if line.strip())
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

            case = test_class(api_config, options.test_amp)
            try:
                case.test()
            except Exception as err:
                if "CUDA error" in str(err) or "memory corruption" in str(err) or "CUDA out of memory" in str(err):
                    exit(0)
            case.clear_tensor()
            del case
            del api_config
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

if __name__ == '__main__':
    main()
