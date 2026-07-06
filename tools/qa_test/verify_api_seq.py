#!/usr/bin/env python3
"""
验证脚本：对比推导出的配置 与 真实抓取的配置，量化推导准确率。

文件名：verify_api_seq.py
用法：
    python verify_api_seq.py -d api_config_derived_4096.txt -r api_config_4096.txt

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

import argparse
import os
import sys
from collections import Counter

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


def is_deterministic(line):
    """确定类行 = 非 MoE 算子 且 非 __getitem__ 取片。"""
    if line.startswith("paddle.Tensor.__getitem__"):
        return False
    return not any(m in line for m in MOE_MARKERS)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="对比推导配置与真实配置，量化推导准确率（多重集重叠）。",
    )
    parser.add_argument("-d", "--derived", required=True, help="推导的配置文件路径")
    parser.add_argument("-r", "--real", required=True, help="真实/基准配置文件路径")
    return parser.parse_args()


def main():
    args = parse_args()

    derived_path = args.derived
    real_path = args.real

    if not os.path.exists(derived_path):
        print(f"错误：推导文件不存在 {derived_path}")
        sys.exit(1)
    if not os.path.exists(real_path):
        print(f"错误：真实文件不存在 {real_path}")
        sys.exit(1)

    derived = load(derived_path)
    real = load(real_path)

    print(f"验证对比")
    print(f"  推导文件: {derived_path}")
    print(f"  真实文件: {real_path}")
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
