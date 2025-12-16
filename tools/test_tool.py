# logs = [
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/log4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_6.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp_7.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_1.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_2.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp3.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_4.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_5.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_6.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4_7.log",
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/logtmp4.log",]

# with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/oom.log", "w") as oom:
#     with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/oom_not.log", "w") as oom_not:
#         for log in logs:
#             with open(log, "r") as f:
#                 lines = f.readlines()
#                 is_log_str = False
#                 log_str = ""
#                 for line in lines:
#                     if "paddle error" in line or "cuda error" in line:
#                         is_log_str = True
#                         log_str = line
#                     elif "test begin" in line:
#                         is_log_str = False
#                         if "CUDAAllocator" in log_str:
#                             oom.write(log_str)
#                         else:
#                             oom_not.write(log_str)
#                     elif is_log_str:
#                         log_str += line

# with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/accuracy.log", "w") as accuracy:
#     for log in logs:
#         with open(log, "r") as f:
#             lines = f.readlines()
#             is_log_str = False
#             log_str = ""
#             for line in lines:
#                 if "accuracy error" in line:
#                     is_log_str = True
#                     log_str = line
#                 elif "test begin" in line:
#                     is_log_str = False
#                     if "Mismatched elements" in log_str:
#                         accuracy.write(log_str)
#                 elif is_log_str:
#                     log_str += line


# logs = [
# "/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/core_dump.log",]

# with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/cinn_core_dump.log", "w") as get_log:
#     with open("/host_home/wanghuan29/PaddleAPITest/tester/api_config/test_log/other.log", "w") as other:
#         for log in logs:
#             with open(log, "r") as f:
#                 lines = f.readlines()
#                 is_log_str = False
#                 log_str = ""
#                 for line in lines:
#                     if "test begin" in line:
#                         if "cinn_op." in log_str or "cinn::" in log_str:
#                             get_log.write(log_str)
#                         else:
#                             other.write(log_str)
#                         log_str = line
#                     else:
#                         log_str += line
from __future__ import annotations

logs = [
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log1.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log2.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log3.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log4.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log5.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log6.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log7.log",
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/log8.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/getset_item1.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/getset_item2.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/getset_item3.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/getset_item4.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/getset_item5.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp1.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp2.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp3.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp4.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp5.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp6.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp7.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp8.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp9.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp10.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp11.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp12.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp13.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp14.log",
    # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/tmp15.log",
]

# with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/error_acc_mismatch.log", "w") as error_acc_mismatch:
#     with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/error_acc_shape.log", "w") as error_acc_shape:
#         with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/error_acc_other.log", "w") as error_acc_other:
#             with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/error_type_diff.log", "w") as error_type_diff:
#                 with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/error_type_diff2.log", "w") as error_type_diff2:
#                     for log in logs:
#                         with open(log, "r") as f:
#                             lines = f.readlines()
#                             is_log_str = False
#                             log_str = ""
#                             config = ""
#                             for line in lines:
#                                 if "test begin" in line:
#                                     # if "[Pass]" not in log_str and "cudaErrorIllegalAddress" not in log_str and "scatter.cu" not in log_str and "Skip" not in log_str and "paddle.Tensor.__getitem__" not in log_str and "paddle.Tensor.__setitem__" not in log_str:
#                                     # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and "cudaErrorIllegalAddress" not in log_str and "scatter.cu" not in log_str:
#                                     # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and ("cudaErrorIllegalAddress" in log_str or "scatter.cu" in log_str):
#                                     # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and "__getitem__" not in log_str and "__setitem__" not in log_str and ".reshape" not in log_str and ".put_along_axis" not in log_str and "dense_tensor.cc:160" not in log_str:
#                                     # if "cudaErrorLaunchFailure" in log_str or "cudaErrorLaunchFailure" in log_str:
#                                     # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and "__getitem__" not in log_str:
#                                     # if config == log_str:
#                                     if "accuracy error" in log_str:
#                                         if "[output type diff error1]" in log_str:
#                                             error_type_diff.write(log_str)
#                                         else:
#                                             if "Mismatched" in log_str:
#                                                 error_acc_mismatch.write(log_str)
#                                             else:
#                                                 if "(shapes" in log_str:
#                                                     error_acc_shape.write(log_str)
#                                                 else:
#                                                     if "Unable to allocate" not in log_str and "cpu_allocator" not in log_str:
#                                                         if "[output type diff error2]" in log_str:
#                                                             error_type_diff2.write(log_str)
#                                                         else:
#                                                             error_acc_other.write(log_str)
#                                     log_str = line
#                                     config = line
#                                 else:
#                                     log_str += line

with open(
    "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/test_log/error.log",
    "w",
) as error:
    for log in logs:
        with open(log) as f:
            lines = f.readlines()
            is_log_str = False
            log_str = ""
            config = ""
            for line in lines:
                if "test begin" in line:
                    # if "[Pass]" not in log_str and "cudaErrorIllegalAddress" not in log_str and "scatter.cu" not in log_str and "Skip" not in log_str and "paddle.Tensor.__getitem__" not in log_str and "paddle.Tensor.__setitem__" not in log_str:
                    # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and "cudaErrorIllegalAddress" not in log_str and "scatter.cu" not in log_str:
                    # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and ("cudaErrorIllegalAddress" in log_str or "scatter.cu" in log_str):
                    # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and "__getitem__" not in log_str and "__setitem__" not in log_str and ".reshape" not in log_str and ".put_along_axis" not in log_str and "dense_tensor.cc:160" not in log_str:
                    # if "cudaErrorLaunchFailure" in log_str or "cudaErrorLaunchFailure" in log_str:
                    # if "[Pass]" not in log_str and "Skip" not in log_str and "error" in log_str and "__getitem__" not in log_str:
                    # if config == log_str:
                    if (
                        "cuda error" in log_str
                        or "CUDA error" in log_str
                        or "CUDNN error" in log_str
                        or "TID" in log_str
                    ):
                        error.write(log_str)
                    log_str = line
                    config = line
                else:
                    log_str += line
