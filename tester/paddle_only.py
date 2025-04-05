from .api_config import TensorConfig, APIConfig, analyse_configs
import re
import collections
import paddle
import numpy
import math
import json
import paddle
import inspect
from .base import APITestBase
import os
import time
from func_timeout import func_set_timeout

DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

api_config_accuracy_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_accuracy_error.txt", "a")
api_config_paddle_error = open(DIR_PATH+"/tester/api_config/test_log/api_config_paddle_error.txt", "a")
api_config_pass = open(DIR_PATH+"/tester/api_config/test_log/api_config_pass.txt", "a")

class APITestPaddleOnly(APITestBase):
    def __init__(self, api_config, test_amp):
        super().__init__(api_config)
        self.api_config = api_config
        self.test_amp = test_amp
    
    @func_set_timeout(600)
    def test(self):
        if self.need_skip():
            print("[Skip]")
            return

        if not self.ana_paddle_api_info():
            print("ana_paddle_api_info failed")
            return

        try:
            if not self.gen_paddle_input():
                print("gen_paddle_input failed")
                return  
            if self.test_amp:
                with paddle.amp.auto_cast():
                    paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)      
            else:
                paddle_output = self.paddle_api(*tuple(self.paddle_args), **self.paddle_kwargs)
            if self.need_check_grad():
                inputs_list = self.get_paddle_input_list()
                result_outputs, result_outputs_grads = self.gen_paddle_output_and_output_grad(paddle_output)
                if len(inputs_list) != 0 and len(result_outputs) != 0 and len(result_outputs_grads) != 0:
                    out_grads = paddle.grad(result_outputs, inputs_list, grad_outputs=result_outputs_grads,allow_unused=True)
        except Exception as err:
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            if "gradient_accumulator.cc" in str(err) or "Out of memory" in str(err):
                return
            print("[paddle error]", self.api_config.config, "\n", str(err))
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            if "CUDA error" in str(err) or "memory corruption" in str(err):
                raise Exception(err)
            return

        try:
            paddle.base.core.eager._for_test_check_cuda_error()
        except Exception as err:
            print("[cuda error]", self.api_config.config, "\n", str(err))
            paddle_output = None
            result_outputs = None
            result_outputs_grads = None
            out_grads = None
            api_config_paddle_error.write(self.api_config.config+"\n")
            api_config_paddle_error.flush()
            return

        paddle_output = None
        result_outputs = None
        result_outputs_grads = None
        out_grads = None
        print("[Pass]", self.api_config.config)
        api_config_pass.write(self.api_config.config+"\n")
        api_config_pass.flush()
  
