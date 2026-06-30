#!/usr/bin/env python3
"""
API trace 推导脚本：seq32 + seq64 → 任意目标 seq

文件名：derive_api_s32_s64.py
源数据：dpskv4_dist_seq32stp3（SEQ_SMALL=32）
        dpskv4_dist_seq64stp3 （SEQ_LARGE=64）
典型用法：
    python derive_api_s32_s64.py 4096    4096_s32x64   # → seq4096
    python derive_api_s32_s64.py 1048576 1M_s32x64     # → seq1M

算法：双指针 + 前瞻窗口对齐，MoE dispatch 上下文追踪
缩放策略：
  策略1：精确比值 SEQ_LARGE/SEQ_SMALL（seq_length 线性相关）
  策略2：差值 = SEQ_DIFF（v = seq + offset 模式）
  策略3：差值是 SEQ_DIFF 的整数倍（v = k*seq + offset 模式）
  策略4：MoE dispatch 上下文直接比例缩放（优先于策略1/2/3）
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

SEQ_64 = 64
SEQ_32 = 32
SEQ_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576
SCALE_FACTOR = SEQ_TARGET / SEQ_64

BASE_DIR_32 = "/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/nzs_tmp_new/outputs/dpskv4_dist_seq32stp2"
BASE_DIR_64 = "/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/nzs_tmp_new/outputs/dpskv4_dist_seq64stp2"
OUTPUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "1M_s32x64"
OUTPUT_DIR = os.path.join(
    "/root/paddlejob/share-storage/gpfs/system-public/lihaoyang08/workspace/0528_dsV4",
    OUTPUT_SUFFIX,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗口大小：最多支持 32 个缺失专家（每个专家 2 行），留裕量设为 80
ALIGN_WINDOW = 80

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
TENSOR_SHAPE_RE = re.compile(r"Tensor\(paddle\.Size\(\[[^\]]*\]\)")
DTYPE_ZONES_RE = re.compile(r'"[a-z]*\d+[^"]*"|Dtype\([^)]*\)')
LIST_RE = re.compile(r"list\[[^\]]*\]")

# MoE 行标志
MOE_PERMUTE_MARK = "moe_permute"  # moe_permute（不含 moe_unpermute）
MOE_UNPERMUTE_MARK = "moe_unpermute"  # 结束 MoE dispatch 上下文


# ──────────────────────────── 签名提取 ────────────────────────────


def get_smart_signature(line):
    """提取智能签名：替换 tensor shape 和 list 中的数值为 #，保留结构。"""
    if line.startswith("[API_TRACE] "):
        line = line[12:]

    idx = line.find("(")
    name = line[:idx] if idx != -1 else line
    params = line[len(name) :] if len(line) > len(name) else ""

    # 替换 Tensor shape 中的数字
    def replace_tensor_shape(m):
        return re.sub(r"\d+", "#", m.group(0))

    params = TENSOR_SHAPE_RE.sub(replace_tensor_shape, params)

    # 替换 list 中的数字
    def replace_list(m):
        return re.sub(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", "#", m.group(0))

    params = LIST_RE.sub(replace_list, params)

    # 保留 dtype 中的数字，替换其他数字
    dtype_zones = [(m.start(), m.end()) for m in DTYPE_ZONES_RE.finditer(params)]
    result = []
    last_end = 0
    for m in NUM_RE.finditer(params):
        in_dtype = any(s <= m.start() < e for s, e in dtype_zones)
        if in_dtype:
            continue
        result.append(params[last_end : m.start()])
        result.append("#")
        last_end = m.end()
    result.append(params[last_end:])

    return name + "".join(result)


# ──────────────────────────── 数值推导 ────────────────────────────


def extract_numbers(line):
    dtype_zones = [(m.start(), m.end()) for m in DTYPE_ZONES_RE.finditer(line)]
    results = []
    for m in NUM_RE.finditer(line):
        in_dtype = any(s <= m.start() < e for s, e in dtype_zones)
        if not in_dtype:
            results.append((m.group(), m.start(), m.end()))
    return results


def is_integer(s):
    try:
        int(s)
        return "." not in s and "e" not in s.lower()
    except ValueError:
        return False


def derive_value(v64, v32, line_num, pos_num, anomalies, force_scale=False):
    """
    推导单个整数值。

    force_scale=True 时启用策略4（MoE dispatch buffer 专用）：
      直接按 × SCALE_FACTOR 比例缩放，优先于其他策略执行。
      触发条件：
        - moe_permute 行的所有数值位置
        - MoE dispatch 上下文中每个 Tensor 的第一维
    """
    if v64 == v32:
        return str(v64), False
    if v32 == 0:
        return str(v64), False

    # 策略4：MoE dispatch 上下文，直接按比例缩放（优先于策略1/2/3）
    # 放在最前以防 strategy3 因 diff=k×SEQ_DIFF 错误覆盖
    if force_scale and v64 > 0:
        new_val = int(round(v64 * SCALE_FACTOR))
        return str(new_val), True

    ratio = v64 / v32
    SEQ_DIFF = SEQ_64 - SEQ_32

    # 策略1：精确 2.0 比值（seq_length 线性相关）
    if abs(ratio - 2.0) < 1e-6:
        if v64 % SEQ_32 == 0:
            return str(int(v64 * SCALE_FACTOR)), True
        if v64 == SEQ_64 + 1 and v32 == SEQ_32 + 1:
            return str(SEQ_TARGET + 1), True
        return str(v64), False

    # 策略2：差值 = SEQ_DIFF（v = seq + offset 模式）
    if v64 - v32 == SEQ_DIFF:
        offset = v64 - SEQ_64
        return str(SEQ_TARGET + offset), True

    # 策略3：差值是 SEQ_DIFF 的整数倍（v = k*seq + offset 模式）
    if (v64 - v32) % SEQ_DIFF == 0 and (v64 - v32) != 0:
        k = (v64 - v32) // SEQ_DIFF
        offset = v64 - k * SEQ_64
        if offset == v32 - k * SEQ_32:
            new_val = k * SEQ_TARGET + offset
            if new_val >= 0:
                return str(new_val), True

    anomalies.append(f"L{line_num}: pos{pos_num} val64={v64}, val32={v32}, ratio={ratio:.4f}")
    return str(v64), False


def _find_tensor_first_dim_starts(line):
    """
    返回 line 中每个 Tensor(paddle.Size([...])) 内第一个数字的起始字符位置集合。
    用于在 MoE dispatch 上下文中识别 dispatch buffer 大小所在位置。
    """
    positions = set()
    for tm in TENSOR_SHAPE_RE.finditer(line):
        for m in NUM_RE.finditer(line, tm.start(), tm.end()):
            positions.add(m.start())
            break  # 每个 Tensor shape 只取第一个数字（第一维）
    return positions


def derive_line(line_64, line_32, line_num, anomalies, moe_dispatch_context=False):
    """
    对匹配的两行进行数值推导，返回 (derived_line, was_scaled)。

    moe_dispatch_context=True：当前行处于 MoE dispatch 上下文中
    （moe_permute 之后、moe_unpermute 处理完之前），
    对每个 Tensor 的第一维启用 force_scale（dispatch buffer 大小）。
    """
    nums_64 = extract_numbers(line_64)
    nums_32 = extract_numbers(line_32)

    if len(nums_64) != len(nums_32):
        return line_64, False

    # moe_permute 行：全部位置启用 force_scale
    is_moe_permute = MOE_PERMUTE_MARK in line_64

    # MoE dispatch 上下文（非 moe_permute 行）：仅 Tensor 第一维启用 force_scale
    tensor_first_dim_starts = set()
    if moe_dispatch_context and not is_moe_permute:
        tensor_first_dim_starts = _find_tensor_first_dim_starts(line_64)

    result = line_64
    replacements = []

    for i, ((v64_str, s64, e64), (v32_str, s32, e32)) in enumerate(zip(nums_64, nums_32)):
        if v64_str == v32_str:
            continue
        if not is_integer(v64_str) or not is_integer(v32_str):
            continue

        v64, v32 = int(v64_str), int(v32_str)
        if v64 == 0 and v32 == 0:
            continue

        # force_scale 条件：
        #   1. moe_permute 行 → 全部位置
        #   2. MoE dispatch 上下文 + 当前位置是 Tensor 第一维
        force_scale = is_moe_permute or (s64 in tensor_first_dim_starts)

        new_val, should_replace = derive_value(
            v64, v32, line_num, i, anomalies, force_scale=force_scale
        )
        if should_replace:
            replacements.append((s64, e64, new_val))

    for start, end, new_val in reversed(replacements):
        result = result[:start] + new_val + result[end:]

    return result, len(replacements) > 0


# ──────────────────────────── 核心对齐算法 ────────────────────────────


def align_sequences(sigs_64, sigs_32, window=ALIGN_WINDOW):
    """
    双指针 + 前瞻窗口对齐。

    返回：[(idx64, idx32_or_None), ...]
      - (i64, i32)  : 两行匹配，进行数值推导
      - (i64, None) : seq64 独有行，原样输出

    seq64 独有行说明：
      seq=32 训练时某些 MoE 专家收到 0 个 token（稀疏路由），
      该专家的 __getitem__ + linear 调用被跳过。
      seq=64 训练时更多 token，所有专家都有 token，因此
      每个 MoE 层 seq=64 比 seq=32 多出若干调用（~2 行/跳过专家）。
      对于目标大 seq，token 量极大，每个专家必然激活，
      seq=64 的调用结构正是目标 seq 的正确结构。
    """
    n64, n32 = len(sigs_64), len(sigs_32)
    i64, i32 = 0, 0
    result = []

    while i64 < n64 and i32 < n32:
        if sigs_64[i64] == sigs_32[i32]:
            result.append((i64, i32))
            i64 += 1
            i32 += 1
            continue

        # 在 seq64 前方找 seq32[i32]：seq64 多出 skip64 行
        skip64 = None
        for k in range(1, window + 1):
            if i64 + k < n64 and sigs_64[i64 + k] == sigs_32[i32]:
                skip64 = k
                break

        # 在 seq32 前方找 seq64[i64]：seq32 多出 skip32 行
        skip32 = None
        for k in range(1, window + 1):
            if i32 + k < n32 and sigs_64[i64] == sigs_32[i32 + k]:
                skip32 = k
                break

        if skip64 is not None and (skip32 is None or skip64 <= skip32):
            # seq64 有 skip64 行多余，直接原样输出
            for _ in range(skip64):
                result.append((i64, None))
                i64 += 1
        elif skip32 is not None:
            # seq32 有 skip32 行多余，跳过
            i32 += skip32
        else:
            # 窗口内无法重对齐，双方各前进一步
            result.append((i64, None))
            i64 += 1
            i32 += 1

    # seq64 剩余行
    while i64 < n64:
        result.append((i64, None))
        i64 += 1

    return result


# ──────────────────────────── 文件处理 ────────────────────────────


def process_file(worker_idx):
    filename = f"workerlog.{worker_idx}"
    path_64 = os.path.join(BASE_DIR_64, "output_0", filename)
    path_32 = os.path.join(BASE_DIR_32, "output_0", filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path_64) or not os.path.exists(path_32):
        return None, [f"{filename}: 文件缺失"]

    print(f"\n处理 {filename}...")

    api_lines_64, api_sigs_64 = [], []
    api_lines_32, api_sigs_32 = [], []

    with open(path_64) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[API_TRACE]"):
                api_lines_64.append(line)
                api_sigs_64.append(get_smart_signature(line))

    with open(path_32) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[API_TRACE]"):
                api_lines_32.append(line)
                api_sigs_32.append(get_smart_signature(line))

    print(f"  seq=64: {len(api_lines_64)} 行")
    print(f"  seq=32: {len(api_lines_32)} 行")
    print(f"  正在对齐 API 序列（窗口={ALIGN_WINDOW}）...")

    alignment = align_sequences(api_sigs_64, api_sigs_32)

    matched_count = sum(1 for _, i32 in alignment if i32 is not None)
    unmatched_count = len(alignment) - matched_count
    print(f"  对齐完成: 配对 {matched_count} 行, seq64独有 {unmatched_count} 行")

    output_lines = []
    anomalies = []
    scaled = 0

    # MoE dispatch 上下文追踪：
    #   moe_permute  → 处理该行后置 True（该行本身由 is_moe_permute 检测处理）
    #   moe_unpermute → 处理该行时 context=True，处理完后置 False
    in_moe_dispatch = False

    for idx64, idx32 in alignment:
        line64 = api_lines_64[idx64]

        # 当前行使用的上下文（处理完后再更新标志）
        current_moe_context = in_moe_dispatch

        if idx32 is None:
            # seq64 独有行（额外专家调用）：原样输出，保留正确结构
            output_lines.append(line64)
        else:
            line32 = api_lines_32[idx32]
            derived, was_scaled = derive_line(
                line64, line32, idx64 + 1, anomalies, moe_dispatch_context=current_moe_context
            )
            output_lines.append(derived)
            if was_scaled:
                scaled += 1

        # 处理完后更新 MoE dispatch 上下文标志
        if MOE_PERMUTE_MARK in line64:
            in_moe_dispatch = True
        elif MOE_UNPERMUTE_MARK in line64:
            in_moe_dispatch = False

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    total = len(api_lines_64)
    match_rate = matched_count * 100 // total if total > 0 else 0

    return {
        "total": total,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "scaled": scaled,
        "anomalies": len(anomalies),
        "match_rate": match_rate,
    }, anomalies


# ──────────────────────────── 主函数 ────────────────────────────


def main():
    worker_indices = set()
    for d in os.listdir(BASE_DIR_64):
        if d.startswith("output_"):
            for f in os.listdir(os.path.join(BASE_DIR_64, d)):
                if f.startswith("workerlog."):
                    worker_indices.add(int(f.split(".")[1]))

    if not worker_indices:
        print("错误：没有找到 workerlog 文件")
        return

    print(f"待处理文件: {sorted(worker_indices)} 个")
    print(f"缩放因子: seq_length {SEQ_64} → {SEQ_TARGET} (×{SCALE_FACTOR:.0f})")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"对齐窗口: {ALIGN_WINDOW}")
    print("=" * 70)

    total = defaultdict(int)
    all_anomalies = []

    for idx in sorted(worker_indices):
        stats, anomalies = process_file(idx)
        if stats is None:
            print(f"  SKIP: {anomalies[0]}")
            continue

        for k, v in stats.items():
            total[k] += v
        for a in anomalies:
            all_anomalies.append(f"[workerlog.{idx}] {a}")

        status = f"⚠ {stats['anomalies']} anomalies" if stats["anomalies"] else "OK"
        print(
            f"  结果: 匹配率 {stats['match_rate']}%, "
            f"配对 {stats['matched']} 行, "
            f"缩放 {stats['scaled']} 行, "
            f"seq64独有 {stats['unmatched']} 行, {status}"
        )

    print("=" * 70)
    total_lines = total["total"]
    total_matched = total["matched"]
    overall_rate = total_matched * 100 // total_lines if total_lines > 0 else 0
    print(
        f"总计: {total_lines} 行, 匹配率 {overall_rate}%, "
        f"配对 {total_matched} 行, "
        f"缩放 {total['scaled']} 行, "
        f"seq64独有 {total['unmatched']} 行, "
        f"{total['anomalies']} 处异常"
    )
    print(f"输出目录: {OUTPUT_DIR}")

    if all_anomalies:
        report_path = os.path.join(OUTPUT_DIR, "anomaly_report.txt")
        with open(report_path, "w") as f:
            f.write(f"异常报告\n总异常数: {len(all_anomalies)}\n{'=' * 70}\n\n")
            for a in all_anomalies:
                f.write(a + "\n")
        print(f"\n异常报告: {report_path}")
        print("前 10 条异常:")
        for a in all_anomalies[:10]:
            print(f"  {a}")
    else:
        print("\n无异常。")


if __name__ == "__main__":
    main()
