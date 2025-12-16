# import os
# from config_analyzer import APIConfig,TensorConfig

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))[0:os.path.dirname(os.path.realpath(__file__)).index("PaddleAPITest")+13]

# def get_notsupport_config():
#     not_support_files = [
#         "tester/api_config/api_config_merged_not_support_amp.txt",
#         "tester/api_config/api_config_merged_not_support_arange.txt",
#         "tester/api_config/api_config_merged_not_support_empty.txt",
#         "tester/api_config/api_config_merged_not_support_flatten.txt",
#         "tester/api_config/api_config_merged_not_support_getset_item.txt",
#         "tester/api_config/api_config_merged_not_support_reshape.txt",
#         "tester/api_config/api_config_merged_not_support_slice.txt",
#         "tester/api_config/api_config_merged_not_support_topk.txt",
#         "tester/api_config/api_config_merged_not_support_zeros.txt",
#         "tester/api_config/api_config_merged_not_support.txt"
#     ]
#     configs = set()

#     for flie in not_support_files:
#         with open(DIR_PATH+"/"+flie, "r") as f:
#             origin_configs = f.readlines()
#             f.close()

#         for config in origin_configs:
#             configs.add(config)
#     return configs

# # logs = [
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_1.txt",
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_2.txt",
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_3.txt",
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_4.txt",
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_5.txt",
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_6.txt",
# # "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_7.txt"
# # ]

# # configs = set()

# # for log in logs:
# #     with open(log, "r") as f:
# #         origin_configs = f.readlines()
# #         f.close()

# #     for config in origin_configs:
# #         configs.add(config)

# # configs = configs - get_notsupport_config()

# # with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_all.txt", "w") as f:
# #     for config in sorted(configs):
# #         f.write(config)


# logs = [
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_amp.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_arange.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_empty.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_eye.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_flatten.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_full.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_getset_item.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_reshape.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_slice.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_sparse.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_tensor_init.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_topk.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support_zeros.txt",
# "/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_not_support.txt",
# ]

# configs = set()

# for log in logs:
#     with open(log, "r") as f:
#         origin_configs = f.readlines()
#         f.close()

#     for config in origin_configs:
#         configs.add(config)

# # configs = configs - get_notsupport_config()

# # with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/api_config_merged_all.txt", "w") as f:
# #     for config in sorted(configs):
# #         f.write(config)

# not_support_api = [
#  "paddle.gather",
#  "paddle.Tensor.gather",
#  "paddle.index_select",
#  "paddle.Tensor.index_select",
#  "paddle.Tensor.index_put",
#  "paddle.Tensor.index_sample",
#  "paddle.index_put",
#  "paddle.index_sample",
#  "paddle.gather_nd",
#  "paddle.Tensor.gather_nd",
#  "paddle.incubate.segment_max",
#  "paddle.incubate.segment_mean",
#  "paddle.incubate.segment_min",
#  "paddle.incubate.segment_sum",
#  "paddle.geometric.segment_max",
#  "paddle.geometric.segment_mean",
#  "paddle.geometric.segment_min",
#  "paddle.geometric.segment_sum",
#  "paddle.geometric.send_u_recv",
#  "paddle.geometric.send_ue_recv",
#  "paddle.geometric.send_uv",
#  "paddle.nn.functional.cross_entropy",
#  "paddle.nn.functional.one_hot",
#  "paddle.nn.functional.upsample",
#  "paddle.vision.ops.roi_align",
#  "paddle.vision.ops.roi_pool",
#  "paddle.nn.functional.binary_cross_entropy",
#  "paddle.multinomial",
#  "paddle.nn.functional.embedding",
#  "paddle.nn.functional.hsigmoid_loss",
#  "paddle.nn.functional.nll_loss",
#  "paddle.nn.functional.gather_tree",
#  "paddle.nn.functional.margin_cross_entropy",
#  "paddle.index_add",
#  "paddle.nn.functional.softmax_with_cross_entropy",
#  "paddle.put_along_axis",
#  "paddle.Tensor.put_along_axis",
#  "paddle.scatter",
#  "paddle.scatter_nd",
#  "paddle.scatter_nd_add",
#  "paddle.bernoulli",
#  "paddle.incubate.nn.functional.fused_multi_head_attention",
#  "paddle.geometric.sample_neighbors",
#  "paddle.incubate.nn.functional.block_multihead_attention"
#  ]
# apis = set()
# print(len(configs))
# with open("/host_home/wanghuan29/APItest3/PaddleAPITest/tester/api_config/lala.txt", "w") as f2:
#     for config in sorted(configs):
#         api = config[0:config.index("(")]
#         apis.add(api)
#     for api in apis:
#         f2.write(api+"\n")
from __future__ import annotations

logs = ["/host_home/wanghuan29/APItest/PaddleAPITest/tester/api_config/api_config.txt"]

configs = set()

for log in logs:
    with open(log) as f:
        origin_configs = f.readlines()
        f.close()

    for config in origin_configs:
        configs.add(config)

# configs = configs - get_notsupport_config()

with open(
    "/host_home/wanghuan29/APItest/PaddleAPITest/tester/api_config/api_config_CE_PaddleScience.txt",
    "w",
) as f:
    for config in sorted(configs):
        f.write(config)
