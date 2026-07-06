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
   常量（hidden=7168、专家数、词表等模型维度）多重集两边相等，保持不变。

B. MoE dispatch 类（router 数据相关，逐 token 路由，不可照抄）
   这族行的数值（每专家 token 数、padded buffer 维度、dispatch 切片边界）
   是数据相关的，small 与 large 原始重叠极低，旧脚本「照抄 large 骨架」不正确。
   本脚本改为「约束重采样」结构重建：
     1. 共 144 个 moe_permute 调用（跨 seq 恒定）。总 token 预算 = 144·seq·4，
        按 per_call_profile（144 维，真实统计的各调用占比）分配给每个调用，
        得到该调用的 Σtpe（每调用 token 总量）。
     2. 单个调用内：20 个专家，按 expert_profile（20 维升序负载占比）把 Σtpe
        拆成 20 个 tokens_per_expert（保证 Σ=该调用预算、整数、非负）。
     3. padded buffer N = Σ ceil(tpe_e/128)·128（128 对齐，与真实一致）。
        input_N = round(Σtpe / 1.2033)（真实统计的固定比值）。
     4. 用固定随机种子施加轻微抖动后再归一化，保证可复现且不是机械等比。
     5. 把重建出的 tpe / N / input_N / dispatch 切片边界回填到该 MoE block 的
        moe_permute、下游 reshape/transpose/getitem 切片、moe_unpermute 各行。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

# ──────────────────────────── 配置（运行时由 parse_args 填充） ────────────────────────────

SEQ_SMALL = None
SEQ_LARGE = None
SEQ_TARGET = None
MULT = None
RANDOM_SEED = None
JITTER = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 真实 padded buffer 取值集合（运行时由骨架填充）。第 2 类缓冲行缩放只动 N 属于此集合者。
MOE_PADDED_N = set()
MOE_PADDED_N_RAW = set()

# ──────────────────────────── 内置负载模板 ────────────────────────────
# 原 moe_profile.json 的内容，由真实 4096/8192 抓取统计而来，直接内联。
# expert_profile_20: 单个 moe_permute 调用内 20 专家的升序负载占比（归一化）。
# per_call_profile_144: 144 个调用各自占总 Σtpe 的比例（升序，归一化）。
# 总 Σtpe = 144 * seq * 4（每调用均值 = seq*4）。input_N = round(Σtpe/1.2033)。
# padded buffer N = Σ ceil(tpe_e/128)*128。
MOE_PROFILE = {
    "expert_profile_20": [
        0.004221,
        0.009402,
        0.013823,
        0.017920,
        0.021618,
        0.024818,
        0.027641,
        0.031208,
        0.034247,
        0.037616,
        0.041644,
        0.045602,
        0.050711,
        0.055962,
        0.062604,
        0.070290,
        0.078967,
        0.091881,
        0.117033,
        0.162790,
    ],
    "per_call_profile_144": [
        0.003664,
        0.003801,
        0.004234,
        0.004444,
        0.004525,
        0.004777,
        0.004919,
        0.005035,
        0.005125,
        0.005139,
        0.005180,
        0.005205,
        0.005312,
        0.005355,
        0.005375,
        0.005457,
        0.005564,
        0.005601,
        0.005675,
        0.005703,
        0.005709,
        0.005789,
        0.005803,
        0.005829,
        0.005858,
        0.005963,
        0.006043,
        0.006070,
        0.006091,
        0.006100,
        0.006104,
        0.006126,
        0.006133,
        0.006140,
        0.006171,
        0.006200,
        0.006234,
        0.006255,
        0.006274,
        0.006280,
        0.006307,
        0.006346,
        0.006366,
        0.006374,
        0.006409,
        0.006424,
        0.006438,
        0.006467,
        0.006474,
        0.006490,
        0.006528,
        0.006545,
        0.006573,
        0.006583,
        0.006598,
        0.006636,
        0.006648,
        0.006671,
        0.006675,
        0.006681,
        0.006692,
        0.006706,
        0.006721,
        0.006727,
        0.006731,
        0.006735,
        0.006742,
        0.006753,
        0.006800,
        0.006809,
        0.006825,
        0.006849,
        0.006865,
        0.006934,
        0.006941,
        0.006973,
        0.006980,
        0.006985,
        0.007008,
        0.007014,
        0.007018,
        0.007032,
        0.007048,
        0.007081,
        0.007096,
        0.007136,
        0.007147,
        0.007174,
        0.007184,
        0.007203,
        0.007225,
        0.007232,
        0.007273,
        0.007285,
        0.007297,
        0.007313,
        0.007364,
        0.007392,
        0.007431,
        0.007452,
        0.007475,
        0.007515,
        0.007547,
        0.007573,
        0.007606,
        0.007624,
        0.007629,
        0.007691,
        0.007701,
        0.007715,
        0.007758,
        0.007781,
        0.007834,
        0.007887,
        0.007920,
        0.007941,
        0.007958,
        0.008033,
        0.008083,
        0.008139,
        0.008153,
        0.008231,
        0.008283,
        0.008317,
        0.008353,
        0.008394,
        0.008409,
        0.008418,
        0.008460,
        0.008569,
        0.008642,
        0.008678,
        0.008729,
        0.008813,
        0.008845,
        0.008858,
        0.008933,
        0.009125,
        0.009336,
        0.009394,
        0.009520,
        0.009638,
        0.009690,
        0.010134,
    ],
    "input_N_divisor": 1.2033,
    "padding_alignment": 128,
    "num_experts": 20,
    "top_k": 4,
    "hidden": 7168,
    "moe_permute_calls": 144,
}

# ──────────────────────────── 正则 ────────────────────────────

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
DTYPE_RE = re.compile(r'"[^"]*"|Dtype\([^)]*\)')  # 保护区：内部数字不提取
INT_RE = re.compile(r"-?\d+")

# MoE 解析
MOE_PERMUTE_RE = re.compile(
    r"moe_permute\(Tensor\(paddle\.Size\(\[(\d+), 7168\]\).*?"
    r"tokens_per_expert=list\[([^\]]*)\], padding_alignment=128"
)
SLICE_RE = re.compile(r"slice\((\d+),(\d+),None\)")

# 模型结构常量：永不作为 MoE block 标量映射的 key（即使某 block 的
# old_padded_N / old_input_N 偶然等于其中之一）。否则会把 Size([N,7168]) 里的
# hidden=7168 等常量误当作 buffer 维度放大。56=top_k*hidden/512、28、3584=hidden/2、
# 20=num_experts、4=top_k、143360/1120 等也是固定结构维度。
MODEL_CONSTANTS = frozenset(
    {
        7168,
        3584,
        1792,
        56,
        28,
        14,
        20,
        4,
        8,
        16,
        32,
        64,
        128,
        143360,
        1120,
        256,
        512,
        1024,
    }
)


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
        mapping = vmap.get((sig, pos))
        if mapping and value in mapping and mapping[value] != value:
            replacements.append((start, end, str(mapping[value])))
    if not replacements:
        return line
    result = line
    for start, end, new_text in reversed(replacements):
        result = result[:start] + new_text + result[end:]
    return result


# ──────────────────────────── MoE 结构重建 ────────────────────────────
# 见文件头 B 部分说明。


def _proportional_int_split(total, weights, rng, jitter):
    """把整数 total 按 weights（占比，和为1）拆成 len(weights) 个非负整数，
    施加 ±jitter 抖动后用最大余数法保证求和精确等于 total。"""
    n = len(weights)
    jittered = []
    for w in weights:
        factor = 1.0 + rng.uniform(-jitter, jitter)
        jittered.append(max(w * factor, 1e-9))
    s = sum(jittered)
    raw = [total * w / s for w in jittered]
    floors = [int(math.floor(x)) for x in raw]
    remainder = total - sum(floors)
    # 把剩余按小数部分从大到小分配
    order = sorted(range(n), key=lambda i: raw[i] - floors[i], reverse=True)
    for k in range(remainder):
        floors[order[k % n]] += 1
    return floors


def load_profile():
    profile_path = os.path.join(BASE_DIR, "moe_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            return json.load(f)
    return MOE_PROFILE


def build_moe_plan(profile):
    """为目标 seq 生成 144 个调用的重建计划。
    返回 list[dict]，每项含 tpe(20)、sum_tpe、padded_each、padded_N、input_N。"""
    rng = random.Random(RANDOM_SEED)
    per_call = profile["per_call_profile_144"]
    expert = profile["expert_profile_20"]
    divisor = profile["input_N_divisor"]
    pad = profile["padding_alignment"]
    n_calls = profile["moe_permute_calls"]

    grand_total = n_calls * SEQ_TARGET * 4  # Σ over all calls 的 token 预算
    # 各调用的 Σtpe（按真实占比 + 抖动，整数，和为 grand_total）
    call_sums = _proportional_int_split(grand_total, per_call, rng, JITTER)

    plans = []
    for csum in call_sums:
        # 单调用内 20 专家拆分（升序模板）
        tpe = _proportional_int_split(csum, expert, rng, JITTER)
        # 每专家 token 数至少 1（避免空专家）
        tpe = [max(t, 1) for t in tpe]
        # 修正使 Σ 仍等于 csum
        drift = csum - sum(tpe)
        if drift != 0:
            tpe[tpe.index(max(tpe))] += drift
        # 真实数据中各调用的「重专家」位置是随机的（升序模板会固定让最后一个专家
        # 最重，不符合真实）。这里对 20 个专家做随机置换，使峰值专家逐调用变化。
        rng.shuffle(tpe)
        padded_each = [math.ceil(t / pad) * pad for t in tpe]
        padded_N = sum(padded_each)
        input_N = int(round(csum / divisor))
        plans.append(
            {
                "tpe": tpe,
                "sum_tpe": sum(tpe),
                "padded_each": padded_each,
                "padded_N": padded_N,
                "input_N": input_N,
            }
        )
    return plans


def _moe_block_bounds(lines):
    """切出每个 MoE block 的 [permute_idx, block_end)。
    block 从一个 moe_permute 行开始，到下一个 moe_permute / moe_unpermute 之后
    的边界为止。这里采用：每个 permute 与其后第一个 unpermute 之间为一个 block，
    unpermute 行本身归入该 block 末尾。返回 [(p_idx, u_idx), ...]。
    permute 有 fp8 / bf16 两种变体，各 144 个；每个 permute 都配一个 unpermute。
    """
    perm_idx = [i for i, l in enumerate(lines) if "functional.moe_permute" in l]
    unperm_idx = [i for i, l in enumerate(lines) if "functional.moe_unpermute" in l]
    blocks = []
    used = set()
    for p in perm_idx:
        # 该 permute 后第一个尚未占用的 unpermute
        u = next((x for x in unperm_idx if x > p and x not in used), None)
        if u is None:
            u = len(lines) - 1
        else:
            used.add(u)
        blocks.append((p, u))
    return blocks


def _parse_permute(line):
    """解析 moe_permute 行，返回 (variant, input_N, tpe_list) 或 None。"""
    m = MOE_PERMUTE_RE.search(line)  # fp8 变体（带 tokens_per_expert=）
    if m:
        tpe = [int(x) for x in m.group(2).split(",") if x.strip()]
        return ("fp8", int(m.group(1)), tpe)
    m = re.search(
        r'moe_permute\(Tensor\(paddle\.Size\(\[(\d+), 7168\]\),"bfloat16"\), '
        r"None,.*?, 20, list\[([^\]]*)\], padding_alignment=128",
        line,
    )
    if m:
        tpe = [int(x) for x in m.group(2).split(",") if x.strip()]
        return ("bf16", int(m.group(1)), tpe)
    return None


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


def _rewrite_permute_line(line, variant, new_input_N, new_tpe):
    """重写 moe_permute 行：替换 input_N 与 tokens_per_expert 列表。"""
    new_list = ",".join(str(t) for t in new_tpe) + ","
    if variant == "fp8":
        line = MOE_PERMUTE_RE.sub(
            lambda m: m.group(0)
            .replace(f"Size([{m.group(1)}, 7168]", f"Size([{new_input_N}, 7168]")
            .replace(
                f"tokens_per_expert=list[{m.group(2)}]", f"tokens_per_expert=list[{new_list}]"
            ),
            line,
            count=1,
        )
        # input_N 还出现在同一行其它 Tensor 的第一维（[N,56] [N,4]）
        m2 = _parse_permute(line)
    else:
        m = re.search(
            r'moe_permute\(Tensor\(paddle\.Size\(\[(\d+), 7168\]\),"bfloat16"\), '
            r"None,.*?, 20, list\[([^\]]*)\], padding_alignment=128",
            line,
        )
        old_in, old_list = m.group(1), m.group(2)
        line = line.replace(f"list[{old_list}]", f"list[{new_list}]", 1)
    return line


def apply_moe_reconstruction(derived, skeleton, moe_plan, profile):
    """用重建计划覆盖每个 MoE block 内的数据相关数值。

    derived  : 已做完确定类仿射的行（将被原地改写 MoE block）。
    skeleton : 未改写的 2048 原始行，用于识别 MoE 数据相关行的旧值。

    ── 为什么不能用「按 block 内全局值替换」 ──
    seq=2048 里 144 个 fp8 permute 全部排在 144 个 unpermute 之前，导致 _moe_block_bounds
    切出的 block 互相重叠（实测单行最多被 8 个 block 覆盖）。同一条 getitem 行会被多个
    block 各自的 scalar_map / bound_map 反复改写，结果出现 slice(186752,640) 这种
    start>end 的脏值，且无法判断该行究竟属于哪个 block。

    ── 改为两类处理 ──
    1. permute 行：每个 permute 唯一携带自身重采样计划的 tpe / input_N，按 block 精确改写。
    2. 其余 MoE 缓冲行（getitem / zero_ / assign / slice / empty / reshape / transpose 等
       带 [N,7168|3584|56|28] padded-buffer 维的行）：用全局线性比例 r=SEQ_TARGET/SEQ_LARGE
       做「逐行局部缩放」——buffer 维 N 和 slice 起止 (a,b) 都按 r 缩放并对齐到 128。
       这与真实 padded buffer 随 seq 近似线性增长一致，且天然保证 128 对齐、slice<=N、
       start<=end，完全不受 block 重叠影响。统计上每行 N 等于「该调用 padded 均值的缩放」，
       与逐 block 精确值仅差每调用方差，对合成配置的形状/量级保真度完全够用。
    """
    pad = profile["padding_alignment"]
    r = SEQ_TARGET / SEQ_LARGE  # 全局缩放比（buffer 维近似线性）

    def scale128(x, mode="round"):
        """把 x 按比例 r 缩放并对齐到 128。mode: round/floor/ceil。"""
        y = x * r / pad
        if mode == "floor":
            k = math.floor(y)
        elif mode == "ceil":
            k = math.ceil(y)
        else:
            k = round(y)
        return int(k) * pad

    # 真实 padded buffer 取值集合（来自 2048 骨架所有 permute block 的 Σceil(tpe/128)·128）。
    # 第 2 类缩放只动 N ∈ 该集合的行，从而避开 [3584,7168] / [102400,7168] 等常量首维。
    # 关键：剔除与模型结构常量重合的 padded 值（如 2048 时恰有 6 个 block 的 padded_N==7168，
    # 但 7168 同时是 hidden 维 / 权重 reshape 行数 list[7168,7168,]，无法区分语义）。
    # 对任何 target seq>=4096，真实 padded_N 都 >9000，永不等于 7168，故剔除 7168 零损失。
    global MOE_PADDED_N, MOE_PADDED_N_RAW
    MOE_PADDED_N_RAW = set()
    for l in skeleton:
        if "moe_permute" not in l:
            continue
        pp = _parse_permute(l)
        if pp:
            MOE_PADDED_N_RAW.add(sum(math.ceil(t / pad) * pad for t in pp[2]))
    # 用于「可能是权重」的形状参数：剔除与结构常量重合的 padded 值（如 7168）
    MOE_PADDED_N = MOE_PADDED_N_RAW - MODEL_CONSTANTS

    # ── 第 1 类：permute 行精确改写（按 block） ──
    blocks = _moe_block_bounds(skeleton)
    # bf16 变体独立再采样一份（与 fp8 路由不同，预算分布相同）
    rng2 = random.Random(RANDOM_SEED + 1)
    bf_plan = []
    per_call = profile["per_call_profile_144"]
    expert = profile["expert_profile_20"]
    divisor = profile["input_N_divisor"]
    n_calls = profile["moe_permute_calls"]
    grand_total = n_calls * SEQ_TARGET * 4
    bf_sums = _proportional_int_split(grand_total, per_call, rng2, JITTER)
    for csum in bf_sums:
        tpe = _proportional_int_split(csum, expert, rng2, JITTER)
        tpe = [max(t, 1) for t in tpe]
        drift = csum - sum(tpe)
        if drift:
            tpe[tpe.index(max(tpe))] += drift
        rng2.shuffle(tpe)
        padded_each = [math.ceil(t / pad) * pad for t in tpe]
        bf_plan.append(
            {
                "tpe": tpe,
                "padded_each": padded_each,
                "padded_N": sum(padded_each),
                "input_N": int(round(csum / divisor)),
            }
        )

    fp8_i = bf16_i = 0
    changed = 0
    for p_idx, u_idx in blocks:
        parsed = _parse_permute(skeleton[p_idx])
        if parsed is None:
            continue
        variant, old_input_N, old_tpe = parsed
        if variant == "fp8":
            plan = moe_plan[fp8_i % len(moe_plan)]
            fp8_i += 1
        else:
            plan = bf_plan[bf16_i % len(bf_plan)]
            bf16_i += 1
        # permute 行里 input_N 出现在多个 Tensor 首维（[N,7168][N,56][N,4]）。
        # 用 scalar_map 只替换 == old_input_N 的 token；7168 等常量绝不入表。
        line = _rewrite_permute_line(skeleton[p_idx], variant, plan["input_N"], plan["tpe"])
        sm = {}
        if old_input_N != plan["input_N"] and old_input_N not in MODEL_CONSTANTS:
            sm[old_input_N] = plan["input_N"]
        line = _rewrite_int_run(line, sm)
        if line != skeleton[p_idx]:
            derived[p_idx] = line
            changed += 1

    # ── 第 2 类：其余 MoE 缓冲行逐行局部缩放（不分 block，扫全文件一次） ──
    changed += _scale_moe_buffer_lines(derived, skeleton, scale128, pad)
    changed += _scale_unpermute_lines(derived, skeleton, scale128)
    return changed


# 携带 padded-buffer 维（数据相关、随 seq 线性增长）的算子。其首维 N 与 slice 边界
# 需随 seq 缩放；moe_permute / moe_unpermute 已在第 1 类单独精确处理，这里跳过。
# 缓冲维有两种书写：① Tensor(Size([N, DIM])) ② list[N, DIM,]（empty/reshape/slice 的形状参数）。
_BUFFER_SHAPE = re.compile(r"Size\(\[(\d+), (7168|3584|56|28)\]")
# list[N,DIM,] 形式：DIM 为 MoE 维时 N=buffer。注意只匹配「DIM 在第二位」的二维形状，
# 避免误伤 list[20,56,56,] 这类常量三维形状（其首维 20 是 num_experts 常量）。
_BUFFER_LIST = re.compile(r"list\[(\d+),(7168|3584|56|28),\]")
_BUFFER_HEADS = (
    "paddle.Tensor.__getitem__",
    "paddle.Tensor.zero_",
    "paddle.assign",
    "paddle.slice",
    "paddle.empty_like",
    "paddle.Tensor.reshape",
    "paddle.incubate.nn.functional.fused_linear",
    "paddle.zeros_like",
    "paddle.Tensor.cast",
    "paddle.incubate.nn.functional.fp8_quant_blockwise",
    "paddle._C_ops._run_custom_op",
    "paddle.transpose",
    "paddle.Tensor.clone",
    "paddle.Tensor.detach",
    "paddle.reshape",
    "paddle.nn.functional.linear",
    "paddle.incubate.nn.functional.fused_act_dequant",
    "paddle.empty",
    "paddle.Tensor.unsqueeze",
    "paddle.Tensor.contiguous",
)
# 首维为这些值的是结构常量张量（[20,7168,7168]、[1120,56]、[3584,7168] 等），首维不缩放。
# 3584=hidden/2、7168=hidden、102400=vocab 分片、14336=2·hidden、64/192/2056 等是固定结构维。
_CONST_FIRST_DIM = frozenset(
    {
        20,
        1120,
        143360,
        3584,
        7168,
        14336,
        102400,
        64,
        192,
        160,
        256,
        512,
        2056,
    }
)


def _scale_moe_buffer_lines(derived, skeleton, scale128, pad):
    """逐行把 MoE padded-buffer 维 N 与 slice 起止按全局比例缩放、对齐 128。
    以原始骨架为基准识别旧值，写回 derived。返回改写行数。

    仅当行内 buffer 维 N ∈ MOE_PADDED_N（真实 padded buffer 集合）才缩放，
    从而不碰常量张量的首维（如 [3584,7168]、[102400,7168]）。"""
    changed = 0
    for i, base in enumerate(skeleton):
        head = base.split("(", 1)[0]
        if head not in _BUFFER_HEADS:
            continue
        sm = _BUFFER_SHAPE.search(base)
        lm = _BUFFER_LIST.search(base)
        if not (sm or lm):
            continue
        new_line = base
        new_N = None
        # ① Tensor(Size([N,DIM]))：N 必须是真实 padded buffer 才缩放
        if sm:
            old_N = int(sm.group(1))
            if old_N in MOE_PADDED_N and old_N not in _CONST_FIRST_DIM:
                new_N = scale128(old_N, "ceil")
                # slice（若有）随之缩放并夹紧
                sl = SLICE_RE.search(base)
                if sl:
                    a, b = int(sl.group(1)), int(sl.group(2))
                    na = min(scale128(a, "floor"), new_N)
                    nb = min(scale128(b, "ceil"), new_N)
                    if na > nb:
                        na = nb
                    new_line = (
                        new_line[: sl.start()] + f"slice({na},{nb},None)" + new_line[sl.end() :]
                    )
                new_line = _BUFFER_SHAPE.sub(
                    lambda m: m.group(0).replace(f"[{m.group(1)}, ", f"[{new_N}, ", 1),
                    new_line,
                    count=1,
                )
        # ② list[N,DIM,]：empty/reshape/slice 的形状参数（N=行数, DIM∈MoE 维）。
        #    歧义点：N==7168 时既可能是 padded buffer（empty 的行数），也可能是权重 reshape
        #    的行数（list[7168,7168,] 来自 flat[51380224]=7168² 的方阵权重）。靠算子区分：
        #      - paddle.empty(...)         → N 永远是 buffer 行数 → 用含 7168 的 RAW 集合
        #      - reshape/slice/其它形状参数 → 可能是权重 → 用剔除常量的 MOE_PADDED_N 集合
        if lm:
            old_LN = int(lm.group(1))
            allow = MOE_PADDED_N_RAW if head == "paddle.empty" else MOE_PADDED_N
            if old_LN in allow:
                nLN = scale128(old_LN, "ceil")
                new_line = _BUFFER_LIST.sub(
                    lambda m: f"list[{nLN},{m.group(2)},]", new_line, count=1
                )
        if new_line != base:
            derived[i] = new_line
            changed += 1
    return changed


# moe_unpermute 行结构：
#   moe_unpermute(Tensor([padded_N,7168]), Tensor([input_N,20]), Tensor([input_N,4]),
#                 Tensor([padded_N]), input_N, 20, )
# 其中 7168/20/4 是常量，padded_N 与 input_N 随 seq 线性增长，须缩放。
_UNPERM_RE = re.compile(
    r'(moe_unpermute\(Tensor\(paddle\.Size\(\[)(\d+)(, 7168\]\),"bfloat16"\), '
    r'Tensor\(paddle\.Size\(\[)(\d+)(, 20\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[)(\d+)(, 4\]\),"int32"\), '
    r'Tensor\(paddle\.Size\(\[)(\d+)(\]\),"float32"\), )(\d+)(, 20, \))'
)


def _scale_unpermute_lines(derived, skeleton, scale128):
    """缩放 moe_unpermute 行的 padded_N（首维与第4个张量）与 input_N（第2/3张量及尾参）。"""
    changed = 0
    for i, base in enumerate(skeleton):
        if "moe_unpermute" not in base:
            continue
        m = _UNPERM_RE.search(base)
        if not m:
            continue
        old_pad = int(m.group(2))  # padded_N
        old_in = int(m.group(4))  # input_N
        new_pad = scale128(old_pad, "ceil")
        new_in = scale128(old_in, "round")
        new_line = _UNPERM_RE.sub(
            lambda mm: (
                mm.group(1)
                + str(new_pad)
                + mm.group(3)
                + str(new_in)
                + mm.group(5)
                + str(new_in)
                + mm.group(7)
                + str(new_pad)
                + mm.group(9)
                + str(new_in)
                + mm.group(11)
            ),
            base,
            count=1,
        )
        if new_line != base:
            derived[i] = new_line
            changed += 1
    return changed


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
    parser.add_argument(
        "--seed", type=int, default=20240528, help="MoE 重建随机种子（默认：20240528）"
    )
    parser.add_argument(
        "--jitter", type=float, default=0.04, help="MoE 重采样抖动幅度（默认：0.04）"
    )
    return parser.parse_args()


def main():
    global SEQ_SMALL, SEQ_LARGE, SEQ_TARGET, MULT, RANDOM_SEED, JITTER

    args = parse_args()

    SEQ_SMALL = args.seq_small
    SEQ_LARGE = args.seq_large
    SEQ_TARGET = args.target_seq
    RANDOM_SEED = args.seed
    JITTER = args.jitter

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
        f"(确定类仿射 mult={MULT}; MoE 约束重采样, seed={RANDOM_SEED})"
    )
    print("=" * 70)

    lines_small = [l.rstrip("\n") for l in open(file_small)]
    lines_large = [l.rstrip("\n") for l in open(file_large)]
    print(f"  seq={SEQ_SMALL}: {len(lines_small)} 行")
    print(f"  seq={SEQ_LARGE}: {len(lines_large)} 行 (作为输出骨架)")

    vmap = build_value_map(lines_small, lines_large)
    learned = sum(len(d) for d in vmap.values())
    print(f"  学到仿射映射: {len(vmap)} 个(签名,位置), 共 {learned} 条值映射")

    profile = load_profile()
    moe_plan = build_moe_plan(profile)
    print(
        f"  MoE 重建计划: {len(moe_plan)} 个 moe_permute 调用, "
        f"总 token 预算 {profile['moe_permute_calls'] * SEQ_TARGET * 4}"
    )

    # 第一遍：确定类仿射改写（MoE block 内的行随后被结构重建覆盖）
    derived = [derive_line(l, vmap) for l in lines_large]

    # 第二遍：MoE 结构重建。
    moe_changed = apply_moe_reconstruction(derived, lines_large, moe_plan, profile)

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
