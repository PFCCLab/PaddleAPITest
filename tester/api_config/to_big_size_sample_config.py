from __future__ import annotations

import random


def extract_random_lines(input_path, output_path, patterns):
    # 初始化模式匹配容器
    pattern_buckets = {pattern: [] for pattern in patterns}

    # 第一次遍历：分类收集匹配行
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            for pattern in patterns:
                if pattern in stripped_line:
                    pattern_buckets[pattern].append(stripped_line)
                    break  # 确保一行只匹配一个模式（根据需求可调整）

    # 第二次遍历：随机抽样
    sampled_lines = []
    for pattern, required_count in patterns.items():
        bucket = pattern_buckets[pattern]
        if not bucket:
            print(f"警告：'{pattern}' 未找到匹配行")
            continue

        actual_count = min(required_count, len(bucket))
        sampled_lines.extend(random.sample(bucket, actual_count))
        if len(bucket) < required_count:
            print(f"注意：'{pattern}' 仅找到 {len(bucket)} 行，已全部提取")

    # 去重处理（根据需求可选）
    # sampled_lines = list(set(sampled_lines))

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sampled_lines))

    print(f"提取完成，共保存 {len(sampled_lines)} 行到 {output_path}")


if __name__ == "__main__":
    # 配置参数
    config = {
        "/host_home/wanghuan29/PaddleAPITest/tester/api_config/8_big_tensor/17/big_tensor_17_2.txt": {  # 输入文件路径
            "/host_home/wanghuan29/PaddleAPITest/tester/api_config/8_big_tensor/17/big_tensor_17_3.txt": {  # 输出文件路径
                "patterns": {  # 模式配置
                    "paddle.Tensor.reshape": 1000,
                    "paddle.reshape": 300,
                    "paddle.broadcast_to": 1000,
                    "paddle.Tensor.__rmul__": 1000,
                    "paddle.Tensor.__mul__": 1000,
                    "paddle.Tensor.tile": 1000,
                }
            }
        }
    }

    # 执行提取
    for input_file, outputs in config.items():
        for output_file, params in outputs.items():
            extract_random_lines(input_file, output_file, params["patterns"])
