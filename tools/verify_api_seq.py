#!/usr/bin/env python3
"""
验证脚本：对比推导出的 seq=TARGET 配置 与 真实抓取的配置，量化推导准确率。

文件名：verify_api_seq.py
用法：
    python verify_api_seq.py 4096
        # 对比 api_config_derived_4096.txt 与 api_config_4096.txt
    python verify_api_seq.py 4096 my_derived.txt api_config_4096.txt
        # 自定义推导文件 / 真实文件

────────────────────────────── 为什么用多重集对比 ──────────────────────────────
配置里约 1/3 的行是 MoE dispatch（router 数据相关），其出现顺序在不同运行间
是随机的。因此「逐行按位置对比」会被顺序差异严重低估（实测仅 ~18%）。
真正有意义的指标是「多重集(Counter)重叠」：忽略顺序，看两边相同的行各有多少条。

脚本输出三档准确率：
  - 整体           ：全部行的多重集重叠
  - 确定类(可推导) ：剔除 MoE/数据相关行后的多重集重叠（这是推导脚本真正负责的部分）
  - MoE 类(不可推导)：仅 MoE 行的多重集重叠（作为对照，理论上限很低）

并打印 Top 不匹配行，便于定位推导脚本的薄弱点。
"""

from __future__ import annotations

import math
import os
import re
import statistics
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 判定「数据相关、不可确定性推导」的行：MoE 相关算子 + 张量取片
MOE_MARKERS = (
    "moe_permute",
    "moe_unpermute",
    "fp8_quant",
    "fused_act_dequant",
    "fused_linear",
    "_run_custom_op",
    "swiglu",
)

# MoE dispatch 缓冲带来的「数据相关」行特征（其首维 N=padded buffer，随路由变化）：
#   - 形如 [N,7168]/[N,3584]/[N,56]/[N,28] 的 fp8/bf16/float32 张量上的
#     transpose / reshape / unsqueeze / zero_ / empty
#   - empty_like(uint8) 这族 router 位图 scratch（dim 随路由小幅波动）
_DISPATCH_SHAPE = re.compile(r"Size\(\[\d+, (7168|3584|56|28)\]")


def is_dispatch_dependent(line):
    """识别由 MoE padded buffer 维度驱动、因而数据相关、不可精确推导的行。"""
    if line.startswith("paddle.empty_like") and "uint8" in line:
        # 小 uint8 位图（dim 5/24/116 等是常量，其余随路由波动）→ 数据相关
        m = re.search(r"Size\(\[(\d+)\]", line)
        if m and int(m.group(1)) not in (5, 24, 116):
            return True
    head = line.split("(", 1)[0]
    if head in (
        "paddle.transpose",
        "paddle.Tensor.reshape",
        "paddle.Tensor.unsqueeze",
        "paddle.Tensor.zero_",
        "paddle.empty",
        "paddle.Tensor.contiguous",
        "paddle.Tensor.clone",
        "paddle.assign",
    ):
        if _DISPATCH_SHAPE.search(line):
            return True
    return False


def is_deterministic(line):
    """确定类行 = 非 MoE 算子、非取片、且非 dispatch 缓冲驱动的行。"""
    if line.startswith("paddle.Tensor.__getitem__"):
        return False
    if any(m in line for m in MOE_MARKERS):
        return False
    if is_dispatch_dependent(line):
        return False
    return True


def load(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f]


def overlap(counter_a, counter_b):
    """两个多重集的交集元素总数。"""
    return sum((counter_a & counter_b).values())


def report(name, derived_lines, real_lines):
    cd, cr = Counter(derived_lines), Counter(real_lines)
    inter = overlap(cd, cr)
    denom = len(real_lines)
    pct = inter * 100 / denom if denom else 0.0
    print(
        f"  {name:<14}: {inter}/{denom} = {pct:.2f}%  "
        f"(推导 {len(derived_lines)} 行 / 真实 {denom} 行)"
    )
    return cd, cr


# ──────────────────────────── MoE 统计校验 ────────────────────────────
# 推导脚本对 MoE 行做结构重建（约束重采样），逐行/多重集对比无意义。
# 真正该验的是「重建是否满足 MoE 不变量」：
#   - moe_permute 调用数 == 288（fp8 144 + bf16 144）
#   - 每调用 num_experts==20、tokens_per_expert 长度==20
#   - 每调用 Σtpe == seq*4（精确）
#   - padded buffer N = Σceil(tpe/128)*128 且 128 对齐
#   - input_N ≈ Σtpe/1.2033
#   - 负载分布形状与真实一致（升序占比的逐档差异）

FP8_RE = re.compile(
    r'moe_permute\(Tensor\(paddle\.Size\(\[(\d+), 7168\]\),"float8_e4m3fn"\),'
    r".*?tokens_per_expert=list\[([^\]]*)\], padding_alignment=128"
)
BF_RE = re.compile(
    r'moe_permute\(Tensor\(paddle\.Size\(\[(\d+), 7168\]\),"bfloat16"\), '
    r"None,.*?, 20, list\[([^\]]*)\], padding_alignment=128"
)


def parse_permutes(lines):
    """返回 [(variant, input_N, tpe_list), ...]。"""
    out = []
    for l in lines:
        if "moe_permute" not in l:
            continue
        m = FP8_RE.search(l)
        if m:
            tpe = [int(x) for x in m.group(2).split(",") if x.strip()]
            out.append(("fp8", int(m.group(1)), tpe))
            continue
        m = BF_RE.search(l)
        if m:
            tpe = [int(x) for x in m.group(2).split(",") if x.strip()]
            out.append(("bf16", int(m.group(1)), tpe))
    return out


def avg_profile(permutes):
    """所有调用的升序归一化负载，逐档平均（20 维）。"""
    profs = []
    for _, _, tpe in permutes:
        s = sum(tpe)
        if s and len(tpe) == 20:
            profs.append([t / s for t in sorted(tpe)])
    if not profs:
        return []
    return [statistics.mean(col) for col in zip(*profs)]


def validate_moe(derived_lines, real_lines, seq):
    der = parse_permutes(derived_lines)
    real = parse_permutes(real_lines)
    print("\n[MoE 结构重建校验] (推导侧需满足不变量，真实侧作对照)")

    def stats(permutes, tag):
        n = len(permutes)
        if not n:
            print(f"  {tag}: 无 moe_permute 行")
            return
        # fp8 / bf16 各自一组，预算 = 各组调用数 * seq * 4
        for grp in ("fp8", "bf16"):
            sub = [p for p in permutes if p[0] == grp]
            if not sub:
                continue
            lens_ok = all(len(t) == 20 for _, _, t in sub)
            sums = [sum(t) for _, _, t in sub]
            total = sum(sums)
            budget = len(sub) * seq * 4
            mean = statistics.mean(sums)
            padded = [sum(math.ceil(x / 128) * 128 for x in t) for _, _, t in sub]
            pad_aligned = all(p % 128 == 0 for p in padded)
            ratios = [s / i for (_, i, _), s in zip(sub, sums) if i]
            print(f"  {tag}/{grp}: 调用数={len(sub)}  tpe长度全为20={lens_ok}")
            print(
                f"      Σtpe总预算={total} (基准 {len(sub)}*seq*4={budget}, "
                f"{'符合' if total == budget else '偏差%.2f%%' % ((total - budget) * 100 / budget)})"
            )
            print(f"      每调用Σtpe均值={mean:.0f} (基准 seq*4={seq * 4})")
            print(f"      padded 128对齐={pad_aligned}  padded均值={statistics.mean(padded):.0f}")
            print(f"      Σtpe/input_N 均值={statistics.mean(ratios):.4f} (基准1.2033)")

    stats(der, "推导")
    stats(real, "真实")

    # 负载分布形状对比
    pd, pr = avg_profile(der), avg_profile(real)
    if pd and pr:
        l1 = sum(abs(a - b) for a, b in zip(pd, pr))
        maxd = max(abs(a - b) for a, b in zip(pd, pr))
        print(f"  负载分布(升序20档归一)差异: L1={l1:.4f}  Max档差={maxd:.4f}")
        print(f"      推导末档(最重专家占比)={pd[-1]:.4f}  真实={pr[-1]:.4f}")


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_api_seq.py <TARGET_SEQ> [derived.txt] [real.txt]")
        sys.exit(1)

    target = int(sys.argv[1])
    derived_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(BASE_DIR, f"api_config_derived_{target}.txt")
    )
    real_path = (
        sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE_DIR, f"api_config_{target}.txt")
    )

    if not os.path.exists(derived_path):
        print(f"错误：推导文件不存在 {derived_path}\n  请先运行 python derive_api_seq.py {target}")
        sys.exit(1)
    if not os.path.exists(real_path):
        print(f"错误：真实文件不存在 {real_path}")
        sys.exit(1)

    derived = load(derived_path)
    real = load(real_path)

    print(f"验证 seq={target}")
    print(f"  推导文件: {os.path.basename(derived_path)}")
    print(f"  真实文件: {os.path.basename(real_path)}")
    print("=" * 70)

    # 三档统计
    print("[多重集重叠准确率]")
    report("整体", derived, real)

    det_d = [l for l in derived if is_deterministic(l)]
    det_r = [l for l in real if is_deterministic(l)]
    cd_det, cr_det = report("确定类(可推导)", det_d, det_r)

    moe_d = [l for l in derived if not is_deterministic(l)]
    moe_r = [l for l in real if not is_deterministic(l)]
    report("MoE类(不可推导)", moe_d, moe_r)

    # MoE 结构重建校验（不变量，而非逐行对比）
    validate_moe(derived, real, target)

    print("=" * 70)
    # 逐行位置对比（仅供参考，受 MoE 顺序随机影响）
    n = min(len(derived), len(real))
    exact = sum(1 for i in range(n) if derived[i] == real[i])
    print(f"[逐行位置精确匹配(仅参考)]: {exact}/{n} = {exact * 100 / n:.2f}%")

    # 确定类的薄弱点
    miss = cd_det - cr_det  # 推导出了、真实没有（推多/推错）
    extra = cr_det - cd_det  # 真实有、推导漏了（推少/推错）
    if miss or extra:
        print("\n[确定类未匹配 Top（定位推导薄弱点）]")
        if miss:
            print("  推导多出/推错的行:")
            for line, cnt in miss.most_common(8):
                print(f"    x{cnt}: {line[:95]}")
        if extra:
            print("  真实有但推导缺失的行:")
            for line, cnt in extra.most_common(8):
                print(f"    x{cnt}: {line[:95]}")


if __name__ == "__main__":
    main()
