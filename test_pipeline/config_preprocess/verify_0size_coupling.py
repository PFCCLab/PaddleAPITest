"""校验 moe_permute / moe_unpermute 的 0-size 配置是否满足 token 维与非 Tensor 参数的耦合约束。

约束（源自 kernel InferMeta，已用 paddle smoke test 实测确认）：
  moe_permute:   所有输入 Tensor 的 axis0（token 数 T）必须相等；tokens_per_expert 之和必须等于 T。
  moe_unpermute: permuted 组 {arg0, arg3} 的 axis0（P）必须相等；zipped 组 {arg1, arg2} 的 axis0（Z）
                 必须相等；total_zipped_tokens 必须等于 Z。

用法：
  python verify_0size_coupling.py <config_file> [<config_file> ...]
退出码非 0 表示存在违规，可用于流水线门禁。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tester.api_config.parser import APIConfig  # noqa: E402
from tester.input_generation.tensor_config import TensorConfig  # noqa: E402

PERMUTE = "paddle.nn.functional.moe_permute"
UNPERMUTE = "paddle.nn.functional.moe_unpermute"


def _tensor_args(api_config):
    return [arg for arg in api_config.args if isinstance(arg, TensorConfig)]


def _find_list_param(api_config, name):
    if name in api_config.kwargs and isinstance(api_config.kwargs[name], list):
        return api_config.kwargs[name]
    for arg in api_config.args:
        if isinstance(arg, list):
            return arg
    return None


def _find_int_param(api_config, name, positional_index):
    if name in api_config.kwargs:
        return api_config.kwargs[name]
    if positional_index < len(api_config.args) and isinstance(
        api_config.args[positional_index], int
    ):
        return api_config.args[positional_index]
    return None


def check_permute(api_config):
    """返回违规原因列表，空列表表示通过。约束取自 generation_rules 的输入规则。"""
    reasons = []
    tensors = _tensor_args(api_config)
    if not tensors:
        return ["无 Tensor 输入"]
    token_dims = {tensor.shape[0] for tensor in tensors}
    if len(token_dims) != 1:
        reasons.append(f"各输入 Tensor 的 axis0 不一致: {sorted(token_dims)}")
    token_num = tensors[0].shape[0]
    # probs 为最后一个 Tensor 参数，其 shape[1] 即 topk（与 routemap 共享）。
    topk = tensors[-1].shape[1] if len(tensors[-1].shape) >= 2 else None
    num_experts = _find_int_param(api_config, "num_experts", 4)
    tokens_per_expert = _find_list_param(api_config, "tokens_per_expert")
    if tokens_per_expert is None:
        reasons.append("缺少 tokens_per_expert")
        return reasons
    if num_experts is not None and len(tokens_per_expert) != num_experts:
        reasons.append(
            f"len(tokens_per_expert)={len(tokens_per_expert)} != num_experts {num_experts}"
        )
    if any(
        not isinstance(count, int) or count < 0 or count > token_num for count in tokens_per_expert
    ):
        reasons.append(
            f"tokens_per_expert 存在越界元素（需 0<=count<={token_num}）: {tokens_per_expert}"
        )
    if topk is not None and sum(tokens_per_expert) > token_num * topk:
        reasons.append(
            f"sum(tokens_per_expert)={sum(tokens_per_expert)} > T*topk={token_num * topk}"
        )
    return reasons


def check_unpermute(api_config):
    reasons = []
    tensors = api_config.args
    if len(tensors) < 4 or not all(isinstance(tensors[i], TensorConfig) for i in range(4)):
        return ["前 4 个位置参数不是 Tensor"]
    permuted_rows = tensors[0].shape[0]
    if tensors[3].shape[0] != permuted_rows:
        reasons.append(f"permuted 组 axis0 不一致: arg0={permuted_rows} arg3={tensors[3].shape[0]}")
    zipped_rows = tensors[1].shape[0]
    if tensors[2].shape[0] != zipped_rows:
        reasons.append(f"zipped 组 axis0 不一致: arg1={zipped_rows} arg2={tensors[2].shape[0]}")
    num_experts = _find_int_param(api_config, "num_experts", 5)
    if (
        num_experts is not None
        and len(tensors[1].shape) >= 2
        and tensors[1].shape[1] != num_experts
    ):
        reasons.append(f"rowmap.shape[1]={tensors[1].shape[1]} != num_experts {num_experts}")
    total_zipped_tokens = _find_int_param(api_config, "total_zipped_tokens", 4)
    if total_zipped_tokens is None:
        reasons.append("缺少 total_zipped_tokens")
    elif total_zipped_tokens != zipped_rows:
        reasons.append(
            f"total_zipped_tokens={total_zipped_tokens} != zipped 组 axis0 {zipped_rows}"
        )
    return reasons


CHECKERS = {PERMUTE: check_permute, UNPERMUTE: check_unpermute}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="待校验的配置文件")
    parser.add_argument(
        "--max-samples", type=int, default=5, help="每类违规打印的样例条数（默认 5）"
    )
    args = parser.parse_args()

    total = {PERMUTE: 0, UNPERMUTE: 0}
    violations = {PERMUTE: 0, UNPERMUTE: 0}
    samples = {PERMUTE: [], UNPERMUTE: []}

    for input_file in args.inputs:
        with open(input_file, encoding="utf-8") as config_file:
            for raw_line in config_file:
                line = raw_line.strip()
                if not line or not line.startswith("paddle."):
                    continue
                for api_name, checker in CHECKERS.items():
                    if not line.startswith(api_name + "("):
                        continue
                    total[api_name] += 1
                    try:
                        api_config = APIConfig(line)
                        reasons = checker(api_config)
                    except Exception as exc:  # 解析失败也算违规，便于暴露畸形配置
                        reasons = [f"解析异常: {type(exc).__name__}: {exc}"]
                    if reasons:
                        violations[api_name] += 1
                        if len(samples[api_name]) < args.max_samples:
                            samples[api_name].append((reasons, line))
                    break

    exit_code = 0
    for api_name in (PERMUTE, UNPERMUTE):
        print(f"=== {api_name} ===")
        print(f"  总数: {total[api_name]}，违规: {violations[api_name]}")
        for reasons, line in samples[api_name]:
            print(f"  [违规] {'; '.join(reasons)}")
            print(f"         {line[:160]}")
        if violations[api_name]:
            exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
