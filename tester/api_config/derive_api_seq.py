#!/usr/bin/env python3
"""
API config 推导脚本：由两个不同 seq 的配置推导任意目标 seq（含 MoE 结构重建）。

文件名：derive_api_seq.py
用法：
    python derive_api_seq.py 4096 --small api_config_1024.txt --large api_config_2048.txt
    python derive_api_seq.py 1048576 --small cfg_1024.txt --large cfg_2048.txt -o api_config_1M.txt
    python derive_api_seq.py 4096  # 使用默认（脚本目录下 api_config_1024/2048.txt）

────────────────────────────── 两类行的处理 ──────────────────────────────
配置里的行分两类：

A. 确定类（结构/维度随 seq 仿射变化，可精确推导）
   逐项比对 small/large 得到：任一会变的整数 v 关于 seq 仿射 v = α·seq + β。
   锚定在 small：mult = (TARGET-small)/small，v(T) = v_small + (v_large-v_small)·mult。
   按 (结构签名, 数字位置) 学习 {v_large -> v_target} 映射改写。
   常量（hidden、专家数、词表等模型维度）多重集两边相等，保持不变。

B. MoE dispatch 类（router 数据相关，逐 token 路由，不可照抄）
   这族行的数值（每专家 token 数、padded buffer 维度、dispatch 切片边界）
   是数据相关的，small 与 large 原始重叠极低，旧脚本「照抄 large 骨架」不正确。
   本脚本从输入配置本身推断 hidden、num_experts、top_k、padding_alignment 和
   moe_permute/moe_unpermute 调用结构；推导时按 large→target 的 seq 比例缩放
   input_N 与 tokens_per_expert，并由 tokens_per_expert 重新计算 padded buffer。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict

# ──────────────────────────── 配置（运行时由 parse_args 填充） ────────────────────────────

SEQ_SMALL = None
SEQ_LARGE = None
SEQ_TARGET = None
MULT = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────── 正则 ────────────────────────────

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
DTYPE_RE = re.compile(r'"[^"]*"|Dtype\([^)]*\)')  # 保护区：内部数字不提取
INT_RE = re.compile(r"-?\d+")


def signature(line):
    """结构签名：把所有数字替换为 #。"""
    return NUM_RE.sub("#", line)


def numbers(line):
    """提取行内数字（跳过 dtype 字符串区），返回 [(text, start, end), ...]。"""
    zones = [(m.start(), m.end()) for m in DTYPE_RE.finditer(line)]
    out = []
    for m in NUM_RE.finditer(line):
        if not any(s <= m.start() < e for s, e in zones):
            out.append((m.group(), m.start(), m.end()))
    return out


def is_int(text):
    return INT_RE.fullmatch(text) is not None


# ──────────────────────────── 学习仿射映射（确定类） ────────────────────────────


def collect_position_values(lines):
    """收集每个 (signature, position) 上出现过的所有整数。"""
    table = defaultdict(lambda: defaultdict(list))
    for line in lines:
        sig = signature(line)
        for pos, (text, _, _) in enumerate(numbers(line)):
            if is_int(text):
                table[sig][pos].append(int(text))
    return table


def build_value_map(lines_small, lines_large):
    """构建 (signature, position) -> {v_large: v_target} 的仿射映射。"""
    small = collect_position_values(lines_small)
    large = collect_position_values(lines_large)

    vmap = defaultdict(dict)
    for sig, positions in large.items():
        for pos, larges in positions.items():
            smalls = small.get(sig, {}).get(pos, [])
            cnt_s, cnt_l = Counter(smalls), Counter(larges)
            if cnt_s == cnt_l:
                continue  # 整列常量，跳过
            common = cnt_s & cnt_l
            rem_s = sorted((cnt_s - common).elements())
            rem_l = sorted((cnt_l - common).elements())
            if len(rem_s) != len(rem_l) or not rem_l:
                continue  # 个数随 seq 变（多为 MoE），跳过
            candidates = defaultdict(set)
            for v_small, v_large in zip(rem_s, rem_l):
                v_target = v_small + (v_large - v_small) * MULT
                candidates[v_large].add(v_target)
            for v_large, targets in candidates.items():
                if len(targets) == 1:  # 无歧义才采用
                    vmap[(sig, pos)][v_large] = next(iter(targets))
    return vmap


def derive_line(line, vmap):
    """用仿射映射改写一行（确定类，以 seq=2048 行为骨架）。"""
    sig = signature(line)
    nums = numbers(line)
    replacements = []
    for pos, (text, start, end) in enumerate(nums):
        if not is_int(text):
            continue
        value = int(text)
        if value == SEQ_LARGE:
            replacements.append((start, end, str(SEQ_TARGET)))
            continue
        mapping = vmap.get((sig, pos))
        if mapping and value in mapping and mapping[value] != value:
            new_value = mapping[value]
            if new_value < 0 and _looks_like_shape_context(line, start):
                continue
            replacements.append((start, end, str(new_value)))
    if not replacements:
        return _apply_common_shape_heuristics(line)
    result = line
    for start, end, new_text in reversed(replacements):
        result = result[:start] + new_text + result[end:]
    return _apply_common_shape_heuristics(result)


def _looks_like_shape_context(line, start):
    """形状/长度类 token 避免被映射成负数。"""
    prefix = line[max(0, start - 80) : start]
    return "Size([" in prefix or "list[" in prefix or line.startswith("paddle.arange(")


def _apply_common_shape_heuristics(line):
    """补充少量稳定的常规形状规则。"""
    new_line = line

    m = re.match(r'paddle\.empty\(shape=list\[(\d+),4,4,\], dtype="float32", \)', new_line)
    if m:
        expected = SEQ_TARGET * 40
        if int(m.group(1)) != expected:
            new_line = new_line.replace(f"list[{m.group(1)},4,4,]", f"list[{expected},4,4,]", 1)

    m = re.match(
        r'paddle\.Tensor\.cast\(Tensor\(paddle\.Size\(\[(\d+), 6\]\),"int64"\), '
        r"dtype=Dtype\(int32\), \)",
        new_line,
    )
    if m and int(m.group(1)) == SEQ_LARGE * 2:
        expected = SEQ_TARGET * 2
        new_line = new_line.replace(f"Size([{m.group(1)}, 6])", f"Size([{expected}, 6])", 1)

    if new_line.startswith("paddle.Tensor.reshape("):
        new_line = new_line.replace(
            f"list[1,{SEQ_LARGE // 128},128,-1,]",
            f"list[1,{SEQ_TARGET // 128},128,-1,]",
        )
        new_line = new_line.replace(
            f"list[1,{SEQ_LARGE // 4},4,-1,]",
            f"list[1,{SEQ_TARGET // 4},4,-1,]",
        )
        m = re.match(
            r'paddle\.Tensor\.reshape\(Tensor\(paddle\.Size\(\[(\d+)\]\),"bool"\), '
            r"list\[1,-1,1,1,\], \)",
            new_line,
        )
        if m:
            expected = SEQ_TARGET // 4 - 1
            if int(m.group(1)) != expected:
                new_line = new_line.replace(f"Size([{m.group(1)}])", f"Size([{expected}])", 1)

    if new_line.startswith(
        (
            "paddle.Tensor.__mul__",
            "paddle.Tensor.sum(",
            "paddle.Tensor.cast(",
            "paddle.nn.functional.rms_norm(",
            "paddle.Tensor.astype(",
            "paddle.Tensor.__add__",
            "paddle.nn.functional.softmax(",
        )
    ):
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 128}, 128, 512])",
            f"Size([1, {SEQ_TARGET // 128}, 128, 512])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4}, 8, 512])",
            f"Size([1, {SEQ_TARGET // 4}, 8, 512])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4}, 4, 1024])",
            f"Size([1, {SEQ_TARGET // 4}, 4, 1024])",
        )

    if new_line.startswith("paddle.Tensor.__setitem__(") or new_line.startswith(
        "paddle.Tensor.__getitem__("
    ):
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4}, 8, 512])",
            f"Size([1, {SEQ_TARGET // 4}, 8, 512])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4}, 4, 1024])",
            f"Size([1, {SEQ_TARGET // 4}, 4, 1024])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4 - 1}, 4, 512])",
            f"Size([1, {SEQ_TARGET // 4 - 1}, 4, 512])",
        )
        new_line = new_line.replace(
            f"slice(None,{SEQ_LARGE // 4},None)", f"slice(None,{SEQ_TARGET // 4},None)"
        )

    if new_line.startswith("paddle.concat("):
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 128}, 1, 64])",
            f"Size([1, {SEQ_TARGET // 128}, 1, 64])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4}, 1, 64])",
            f"Size([1, {SEQ_TARGET // 4}, 1, 64])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 128}, 512])",
            f"Size([1, {SEQ_TARGET // 128}, 512])",
        )
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 4}, 512])",
            f"Size([1, {SEQ_TARGET // 4}, 512])",
        )
        new_line = new_line.replace(
            f"Size([{SEQ_LARGE * 8}, 1024])",
            f"Size([{SEQ_TARGET * 2}, 1024])",
        )

    if new_line.startswith("paddle.zeros("):
        new_line = new_line.replace(f"list[{SEQ_LARGE // 4},]", f"list[{SEQ_TARGET // 4},]")
    if new_line.startswith("paddle.full("):
        new_line = new_line.replace(
            f"list[1,{SEQ_LARGE // 4},8,512,]", f"list[1,{SEQ_TARGET // 4},8,512,]"
        )
    if new_line.startswith("paddle.Tensor.__lt__("):
        new_line = new_line.replace(f", {SEQ_LARGE // 4}, )", f", {SEQ_TARGET // 4}, )")

    if new_line.startswith("paddle._C_ops.transpose("):
        new_line = new_line.replace(
            f"Size([{SEQ_LARGE * 8}, 1024])",
            f"Size([{SEQ_TARGET * 2}, 1024])",
        )
    if new_line.startswith("paddle.Tensor.matmul("):
        new_line = new_line.replace(
            f"Size([{SEQ_LARGE * 8}, {SEQ_LARGE * 2}])",
            f"Size([{SEQ_TARGET * 2}, {SEQ_TARGET}])",
        )
    if new_line.startswith("paddle.Tensor.__add__(") or new_line.startswith(
        "paddle.nn.functional.softmax("
    ):
        new_line = new_line.replace(
            f"Size([1, {SEQ_LARGE // 128}, 128, 512])",
            f"Size([1, {SEQ_TARGET // 128}, 128, 512])",
        )

    return new_line


# ──────────────────────────── 模型自适应 MoE 推导 ────────────────────────────


MOE_PERMUTE_FP8_RE = re.compile(
    r'moe_permute\(Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"float8_e4m3fn"\), '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"float32"\), '
    r"num_experts=(\d+), tokens_per_expert=list\[([^\]]*)\], "
    r"padding_alignment=(\d+), do_gather=True, using_ue8m0_scale=True, \)"
)

MOE_PERMUTE_BF16_RE = re.compile(
    r'moe_permute\(Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"bfloat16"\), None, '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"float32"\), '
    r"(\d+), list\[([^\]]*)\], padding_alignment=(\d+), do_gather=True, \)"
)

MOE_UNPERMUTE_GENERIC_RE = re.compile(
    r'moe_unpermute\(Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"bfloat16"\), '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[(\d+), (\d+)\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[(\d+)\]\),"float32"\), '
    r"(?:total_zipped_tokens=)?(\d+), (?:num_experts=)?(\d+), \)"
)


def _split_proportionally(total, weights):
    """把 total 按 weights 的比例拆成整数。"""
    if total <= 0:
        return [0 for _ in weights]
    raw = [max(w, 1e-9) for w in weights]
    s = sum(raw)
    scaled = [total * w / s for w in raw]
    floors = [int(math.floor(x)) for x in scaled]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda i: scaled[i] - floors[i], reverse=True)
    for k in range(remainder):
        floors[order[k % len(order)]] += 1
    return floors


def _scale_128(value, ratio, mode="ceil"):
    scaled = value * ratio / 128.0
    if mode == "floor":
        return int(math.floor(scaled)) * 128
    if mode == "round":
        return int(round(scaled)) * 128
    return int(math.ceil(scaled)) * 128


def parse_moe_permute(line):
    """解析 moe_permute 行，返回字典或 None。"""
    m = MOE_PERMUTE_FP8_RE.search(line)
    if m:
        tpe = [int(x) for x in m.group(10).split(",") if x.strip()]
        return {
            "variant": "fp8",
            "input_n": int(m.group(1)),
            "hidden": int(m.group(2)),
            "top_k": int(m.group(4)),
            "num_experts": int(m.group(9)),
            "tpe": tpe,
            "pad": int(m.group(11)),
            "padded_n": sum(math.ceil(t / int(m.group(11))) * int(m.group(11)) for t in tpe),
        }
    m = MOE_PERMUTE_BF16_RE.search(line)
    if m:
        tpe = [int(x) for x in m.group(8).split(",") if x.strip()]
        return {
            "variant": "bf16",
            "input_n": int(m.group(1)),
            "hidden": int(m.group(2)),
            "top_k": int(m.group(4)),
            "num_experts": int(m.group(7)),
            "tpe": tpe,
            "pad": int(m.group(9)),
            "padded_n": sum(math.ceil(t / int(m.group(9))) * int(m.group(9)) for t in tpe),
        }
    return None


def parse_moe_unpermute(line):
    """解析 moe_unpermute 行，返回字典或 None。"""
    m = MOE_UNPERMUTE_GENERIC_RE.search(line)
    if not m:
        return None
    return {
        "padded_n": int(m.group(1)),
        "hidden": int(m.group(2)),
        "input_n": int(m.group(3)),
        "top_k": int(m.group(6)),
        "num_experts": int(m.group(4)),
    }


def infer_moe_profile(lines):
    """从现有配置推断 MoE 的模型参数与调用统计。"""
    profile = {
        "hidden": None,
        "top_k": None,
        "num_experts": None,
        "fp8_count": 0,
        "bf16_count": 0,
        "permute_count": 0,
    }
    for line in lines:
        perm = parse_moe_permute(line)
        if perm:
            profile["hidden"] = profile["hidden"] or perm["hidden"]
            profile["top_k"] = profile["top_k"] or perm["top_k"]
            profile["num_experts"] = profile["num_experts"] or perm["num_experts"]
            profile["fp8_count" if perm["variant"] == "fp8" else "bf16_count"] += 1
            profile["permute_count"] += 1
            continue
        unperm = parse_moe_unpermute(line)
        if unperm:
            profile["hidden"] = profile["hidden"] or unperm["hidden"]
            profile["top_k"] = profile["top_k"] or unperm["top_k"]
            profile["num_experts"] = profile["num_experts"] or unperm["num_experts"]
    return profile


def rewrite_moe_permute_line(line, perm, new_input_n, new_tpe):
    """把 moe_permute 行按模型自适应规则重写。"""
    new_line = _rewrite_int_run(line, {perm["input_n"]: new_input_n})
    old_list = ",".join(str(t) for t in perm["tpe"]) + ","
    new_list = ",".join(str(t) for t in new_tpe) + ","
    return new_line.replace(old_list, new_list, 1)


def rewrite_moe_unpermute_line(line, unperm, new_input_n, new_padded_n):
    """把 moe_unpermute 行按模型自适应规则重写。"""
    return _rewrite_int_run(
        line,
        {
            unperm["padded_n"]: new_padded_n,
            unperm["input_n"]: new_input_n,
        },
    )


def rewrite_getitem_seq_chunks(line):
    """重写 verify 归为 MoE 的 __getitem__ 中可由 seq 推出的 chunk 维。"""
    if not line.startswith("paddle.Tensor.__getitem__"):
        return line
    ratio = SEQ_TARGET / SEQ_LARGE
    seq_values = {
        SEQ_LARGE // 128: int(round((SEQ_LARGE // 128) * ratio)),
        SEQ_LARGE // 4: int(round((SEQ_LARGE // 4) * ratio)),
    }
    new_line = line

    m = re.search(r"paddle\.Size\(\[1, (\d+), 1, 64\]\)", new_line)
    if m:
        old = int(m.group(1))
        new = seq_values.get(old)
        if new is not None:
            new_line = new_line[: m.start(1)] + str(new) + new_line[m.end(1) :]
            new_line = new_line.replace(f"slice(None,{old},None)", f"slice(None,{new},None)", 1)

    def repl_second_dim(match):
        old = int(match.group(2))
        new = seq_values.get(old)
        if new is None:
            return match.group(0)
        return match.group(1) + str(new) + match.group(3)

    new_line = re.sub(
        r"(paddle\.Size\(\[1, )(\d+)(, 4, 1024\]\))",
        repl_second_dim,
        new_line,
        count=1,
    )
    new_line = re.sub(
        r"(paddle\.Size\(\[1, )(\d+)(, 8, 512\]\))",
        repl_second_dim,
        new_line,
        count=1,
    )
    new_line = re.sub(
        r"(paddle\.Size\(\[)(\d+)(\]\),\"bool\"\))",
        repl_second_dim,
        new_line,
        count=1,
    )
    return new_line


def apply_model_adaptive_moe_reconstruction(derived, skeleton):
    """只重写 verify.py 视为 MoE 的行，且不依赖固定 hidden/num_experts/profile。"""
    ratio = SEQ_TARGET / SEQ_LARGE
    changed = 0
    input_map = {}
    padded_map = {}
    moe_markers = (
        "fp8_quant",
        "fused_act_dequant",
        "fused_linear",
        "_run_custom_op",
        "swiglu",
    )
    for i, base in enumerate(skeleton):
        perm = parse_moe_permute(base)
        if perm is not None:
            new_input_n = int(round(perm["input_n"] * ratio))
            new_tpe_total = int(round(sum(perm["tpe"]) * ratio))
            new_tpe = _split_proportionally(new_tpe_total, perm["tpe"])
            new_line = rewrite_moe_permute_line(base, perm, new_input_n, new_tpe)
            input_map[perm["input_n"]] = new_input_n
            padded_map[perm["padded_n"]] = sum(
                math.ceil(t / perm["pad"]) * perm["pad"] for t in new_tpe
            )
            if new_line != base:
                derived[i] = new_line
                changed += 1
            continue

        unperm = parse_moe_unpermute(base)
        if unperm is not None:
            new_input_n = int(round(unperm["input_n"] * ratio))
            new_padded_n = _scale_128(unperm["padded_n"], ratio, mode="ceil")
            input_map[unperm["input_n"]] = new_input_n
            padded_map[unperm["padded_n"]] = new_padded_n
            new_line = rewrite_moe_unpermute_line(base, unperm, new_input_n, new_padded_n)
            if new_line != base:
                derived[i] = new_line
                changed += 1
            continue

        if base.startswith("paddle.Tensor.__getitem__") or any(m in base for m in moe_markers):
            mp = {}
            for old, new in input_map.items():
                if old != new and str(old) in base:
                    mp[old] = new
            for old, new in padded_map.items():
                if old != new and str(old) in base:
                    mp[old] = new
            new_line = _rewrite_int_run(derived[i], mp) if mp else derived[i]
            if base.startswith("paddle.Tensor.__getitem__"):
                new_line = rewrite_getitem_seq_chunks(new_line)
            if new_line != derived[i]:
                derived[i] = new_line
                changed += 1
    return changed


def _rewrite_int_run(line, old_to_new):
    """把 line 中等于某个 old 的整数替换为对应 new（保护 dtype 区）。
    old_to_new 是 dict[int,int]；只替换恰好匹配的整数 token。"""
    nums = numbers(line)
    repl = []
    for text, start, end in nums:
        if is_int(text):
            v = int(text)
            if v in old_to_new and old_to_new[v] != v:
                repl.append((start, end, str(old_to_new[v])))
    if not repl:
        return line
    for start, end, nt in reversed(repl):
        line = line[:start] + nt + line[end:]
    return line


def parse_args():
    parser = argparse.ArgumentParser(
        description="API config 推导脚本：由两个不同 seq 的配置推导任意目标 seq（含 MoE 结构重建）。",
    )
    parser.add_argument(
        "target_seq", type=int, help="目标 sequence 长度（必须是 seq_small 的整数倍）"
    )
    parser.add_argument(
        "--small",
        default=None,
        help="小 seq 配置文件路径（默认：脚本目录下 api_config_<seq_small>.txt）",
    )
    parser.add_argument(
        "--large",
        default=None,
        help="大 seq 配置文件路径（默认：脚本目录下 api_config_<seq_large>.txt）",
    )
    parser.add_argument("--seq-small", type=int, default=1024, help="小配置的 seq 值（默认：1024）")
    parser.add_argument("--seq-large", type=int, default=2048, help="大配置的 seq 值（默认：2048）")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出文件路径（默认：源文件同目录下 api_config_derived_<target>.txt）",
    )
    return parser.parse_args()


def main():
    global SEQ_SMALL, SEQ_LARGE, SEQ_TARGET, MULT

    args = parse_args()

    SEQ_SMALL = args.seq_small
    SEQ_LARGE = args.seq_large
    SEQ_TARGET = args.target_seq

    assert (SEQ_TARGET - SEQ_SMALL) % SEQ_SMALL == 0, (
        f"TARGET ({SEQ_TARGET}) 必须满足 (TARGET - {SEQ_SMALL}) % {SEQ_SMALL} == 0"
    )
    MULT = (SEQ_TARGET - SEQ_SMALL) // SEQ_SMALL

    file_small = args.small or os.path.join(BASE_DIR, f"api_config_{SEQ_SMALL}.txt")
    file_large = args.large or os.path.join(BASE_DIR, f"api_config_{SEQ_LARGE}.txt")

    if args.output:
        output_path = args.output
    else:
        out_dir = os.path.dirname(file_large) or "."
        output_path = os.path.join(out_dir, f"api_config_derived_{SEQ_TARGET}.txt")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(file_small):
        print(f"错误：源文件缺失 {file_small}")
        sys.exit(1)
    if not os.path.exists(file_large):
        print(f"错误：源文件缺失 {file_large}")
        sys.exit(1)

    print(
        f"推导: seq {SEQ_SMALL}+{SEQ_LARGE} → {SEQ_TARGET}  "
        f"(确定类仿射 mult={MULT}; MoE 自适应重写)"
    )
    print("=" * 70)

    lines_small = [l.rstrip("\n") for l in open(file_small)]
    lines_large = [l.rstrip("\n") for l in open(file_large)]
    print(f"  seq={SEQ_SMALL}: {len(lines_small)} 行")
    print(f"  seq={SEQ_LARGE}: {len(lines_large)} 行 (作为输出骨架)")

    vmap = build_value_map(lines_small, lines_large)
    learned = sum(len(d) for d in vmap.values())
    print(f"  学到仿射映射: {len(vmap)} 个(签名,位置), 共 {learned} 条值映射")
    moe_profile = infer_moe_profile(lines_large)
    print(
        f"  MoE 结构: hidden={moe_profile['hidden']}, top_k={moe_profile['top_k']}, "
        f"num_experts={moe_profile['num_experts']}, permute={moe_profile['permute_count']}"
    )

    # 第一遍：确定类仿射改写（MoE block 内的行随后被结构重建覆盖）
    derived = [derive_line(l, vmap) for l in lines_large]

    # 第二遍：MoE 结构自适应重建。
    moe_changed = apply_model_adaptive_moe_reconstruction(derived, lines_large)

    affine_changed = sum(1 for a, b in zip(derived, lines_large) if a != b)

    with open(output_path, "w") as f:
        f.write("\n".join(derived) + "\n")

    print("=" * 70)
    print(f"完成: 共 {len(derived)} 行")
    print(f"  改写行数(含 MoE): {affine_changed}")
    print(f"  MoE 结构重建行数: {moe_changed}")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    main()
