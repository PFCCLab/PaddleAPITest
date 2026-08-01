#!/usr/bin/env python3
"""
API trace 推导脚本：seq64 + seq4096 → 任意目标 seq

文件名：derive_api_s64_s4096.py
源数据：dpskv4_dist_seq64stp3  （SEQ_SMALL=64）
        dpskv4_dist_seq4096stp3（SEQ_LARGE=4096）
典型用法：
    python derive_api_s64_s4096.py 1048576 1M_s64x4096   # → seq1M

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

SEQ_SMALL = 64
SEQ_LARGE = 4096
SEQ_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1048576
SCALE_FACTOR = SEQ_TARGET / SEQ_LARGE  # 1M/4096 = 256

BASE_DIR_SMALL = "/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/nzs_tmp_new/outputs/dpskv4_dist_seq64stp2"
BASE_DIR_LARGE = "/root/paddlejob/share-storage/gpfs/system-public/ningzhengsheng/nzs_tmp_new/outputs/dpskv4_dist_seq4096stp2"
OUTPUT_SUFFIX = sys.argv[2] if len(sys.argv) > 2 else "1M_s64x4096"
OUTPUT_DIR = os.path.join(
    os.environ.get("API_DERIVE_OUTPUT_ROOT", "generated/api_traces"),
    OUTPUT_SUFFIX,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 窗口大小：最多支持 32 个缺失专家（每个专家 2 行），留裕量设为 80
ALIGN_WINDOW = 80

SEQ_RATIO = SEQ_LARGE / SEQ_SMALL  # 64.0
SEQ_DIFF = SEQ_LARGE - SEQ_SMALL  # 4032

NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
TENSOR_SHAPE_RE = re.compile(r"Tensor\(paddle\.Size\(\[[^\]]*\]\)")
DTYPE_ZONES_RE = re.compile(r'"[a-z]*\d+[^"]*"|Dtype\([^)]*\)')
LIST_RE = re.compile(r"list\[[^\]]*\]")

# MoE 行标志
MOE_PERMUTE_MARK = "moe_permute"
MOE_UNPERMUTE_MARK = "moe_unpermute"


# ──────────────────────────── 签名提取 ────────────────────────────


def get_smart_signature(line):
    """提取智能签名：替换 tensor shape 和 list 中的数值为 #，保留结构。"""
    if line.startswith("[API_TRACE] "):
        line = line[12:]

    idx = line.find("(")
    name = line[:idx] if idx != -1 else line
    params = line[len(name) :] if len(line) > len(name) else ""

    def replace_tensor_shape(m):
        return re.sub(r"\d+", "#", m.group(0))

    params = TENSOR_SHAPE_RE.sub(replace_tensor_shape, params)

    def replace_list(m):
        return re.sub(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", "#", m.group(0))

    params = LIST_RE.sub(replace_list, params)

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


def derive_value(v_large, v_small, line_num, pos_num, anomalies, force_scale=False):
    """
    推导单个整数值（从 SEQ_LARGE 侧缩放至 SEQ_TARGET）。

    force_scale=True：MoE dispatch buffer 专用，直接 ×SCALE_FACTOR，优先于策略1/2/3。
    """
    if v_large == v_small:
        return str(v_large), False
    if v_small == 0:
        return str(v_large), False

    # 策略4：MoE dispatch 上下文直接比例缩放（优先于其他策略）
    if force_scale and v_large > 0:
        new_val = int(round(v_large * SCALE_FACTOR))
        return str(new_val), True

    ratio = v_large / v_small

    # 策略1：精确比值 SEQ_LARGE/SEQ_SMALL（seq_length 线性相关）
    if abs(ratio - SEQ_RATIO) < 1e-6:
        if v_large % SEQ_SMALL == 0:
            return str(int(v_large * SCALE_FACTOR)), True
        if v_large == SEQ_LARGE + 1 and v_small == SEQ_SMALL + 1:
            return str(SEQ_TARGET + 1), True
        return str(v_large), False

    # 策略2：差值 = SEQ_DIFF（v = seq + offset 模式）
    if v_large - v_small == SEQ_DIFF:
        offset = v_large - SEQ_LARGE
        return str(SEQ_TARGET + offset), True

    # 策略3：差值是 SEQ_DIFF 的整数倍（v = k*seq + offset 模式）
    if (v_large - v_small) % SEQ_DIFF == 0 and (v_large - v_small) != 0:
        k = (v_large - v_small) // SEQ_DIFF
        offset = v_large - k * SEQ_LARGE
        if offset == v_small - k * SEQ_SMALL:
            new_val = k * SEQ_TARGET + offset
            if new_val >= 0:
                return str(new_val), True

    anomalies.append(
        f"L{line_num}: pos{pos_num} v_large={v_large}, v_small={v_small}, ratio={ratio:.4f}"
    )
    return str(v_large), False


def _find_tensor_first_dim_starts(line):
    """返回每个 Tensor shape 内第一个数字的字符起始位置集合（dispatch buffer 第一维）。"""
    positions = set()
    for tm in TENSOR_SHAPE_RE.finditer(line):
        for m in NUM_RE.finditer(line, tm.start(), tm.end()):
            positions.add(m.start())
            break
    return positions


def derive_line(line_large, line_small, line_num, anomalies, moe_dispatch_context=False):
    """对匹配的两行进行数值推导，返回 (derived_line, was_scaled)。"""
    nums_large = extract_numbers(line_large)
    nums_small = extract_numbers(line_small)

    if len(nums_large) != len(nums_small):
        return line_large, False

    is_moe_permute = MOE_PERMUTE_MARK in line_large

    tensor_first_dim_starts = set()
    if moe_dispatch_context and not is_moe_permute:
        tensor_first_dim_starts = _find_tensor_first_dim_starts(line_large)

    result = line_large
    replacements = []

    for i, ((vl_str, sl, el), (vs_str, ss, es)) in enumerate(zip(nums_large, nums_small)):
        if vl_str == vs_str:
            continue
        if not is_integer(vl_str) or not is_integer(vs_str):
            continue

        v_large, v_small = int(vl_str), int(vs_str)
        if v_large == 0 and v_small == 0:
            continue

        force_scale = is_moe_permute or (sl in tensor_first_dim_starts)

        new_val, should_replace = derive_value(
            v_large, v_small, line_num, i, anomalies, force_scale=force_scale
        )
        if should_replace:
            replacements.append((sl, el, new_val))

    for start, end, new_val in reversed(replacements):
        result = result[:start] + new_val + result[end:]

    return result, len(replacements) > 0


# ──────────────────────────── 核心对齐算法 ────────────────────────────


def align_sequences(sigs_large, sigs_small, window=ALIGN_WINDOW):
    """
    双指针 + 前瞻窗口对齐。
    返回：[(idx_large, idx_small_or_None), ...]
      - (il, is_) : 两行匹配，进行数值推导
      - (il, None): SEQ_LARGE 独有行，原样输出
    """
    nl, ns, il, is_, result = len(sigs_large), len(sigs_small), 0, 0, []

    while il < nl and is_ < ns:
        if sigs_large[il] == sigs_small[is_]:
            result.append((il, is_))
            il += 1
            is_ += 1
            continue

        skip_l = next(
            (
                k
                for k in range(1, window + 1)
                if il + k < nl and sigs_large[il + k] == sigs_small[is_]
            ),
            None,
        )
        skip_s = next(
            (
                k
                for k in range(1, window + 1)
                if is_ + k < ns and sigs_large[il] == sigs_small[is_ + k]
            ),
            None,
        )

        if skip_l is not None and (skip_s is None or skip_l <= skip_s):
            for _ in range(skip_l):
                result.append((il, None))
                il += 1
        elif skip_s is not None:
            is_ += skip_s
        else:
            result.append((il, None))
            il += 1
            is_ += 1

    while il < nl:
        result.append((il, None))
        il += 1

    return result


# ──────────────────────────── 文件处理 ────────────────────────────


def process_file(worker_idx):
    filename = f"workerlog.{worker_idx}"
    path_large = os.path.join(BASE_DIR_LARGE, "output_0", filename)
    path_small = os.path.join(BASE_DIR_SMALL, "output_0", filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path_large) or not os.path.exists(path_small):
        return None, [f"{filename}: 文件缺失"]

    print(f"\n处理 {filename}...")

    api_lines_large, api_sigs_large = [], []
    api_lines_small, api_sigs_small = [], []

    with open(path_large) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[API_TRACE]"):
                api_lines_large.append(line)
                api_sigs_large.append(get_smart_signature(line))

    with open(path_small) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[API_TRACE]"):
                api_lines_small.append(line)
                api_sigs_small.append(get_smart_signature(line))

    print(f"  seq={SEQ_LARGE}: {len(api_lines_large)} 行")
    print(f"  seq={SEQ_SMALL}: {len(api_lines_small)} 行")
    print(f"  正在对齐 API 序列（窗口={ALIGN_WINDOW}）...")

    alignment = align_sequences(api_sigs_large, api_sigs_small)

    matched_count = sum(1 for _, is_ in alignment if is_ is not None)
    unmatched_count = len(alignment) - matched_count
    print(f"  对齐完成: 配对 {matched_count} 行, seq{SEQ_LARGE}独有 {unmatched_count} 行")

    output_lines = []
    anomalies = []
    scaled = 0
    in_moe_dispatch = False

    for idx_l, idx_s in alignment:
        line_large = api_lines_large[idx_l]
        current_moe_context = in_moe_dispatch

        if idx_s is None:
            output_lines.append(line_large)
        else:
            line_small = api_lines_small[idx_s]
            derived, was_scaled = derive_line(
                line_large,
                line_small,
                idx_l + 1,
                anomalies,
                moe_dispatch_context=current_moe_context,
            )
            output_lines.append(derived)
            if was_scaled:
                scaled += 1

        if MOE_PERMUTE_MARK in line_large:
            in_moe_dispatch = True
        elif MOE_UNPERMUTE_MARK in line_large:
            in_moe_dispatch = False

    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    total = len(api_lines_large)
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
    for d in os.listdir(BASE_DIR_LARGE):
        if d.startswith("output_"):
            for f in os.listdir(os.path.join(BASE_DIR_LARGE, d)):
                if f.startswith("workerlog."):
                    worker_indices.add(int(f.split(".")[1]))

    if not worker_indices:
        print("错误：没有找到 workerlog 文件")
        return

    print(f"待处理文件: {sorted(worker_indices)} 个")
    print(f"缩放因子: seq_length {SEQ_LARGE} → {SEQ_TARGET} (×{SCALE_FACTOR:.0f})")
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
            f"seq{SEQ_LARGE}独有 {stats['unmatched']} 行, {status}"
        )

    print("=" * 70)
    total_lines = total["total"]
    total_matched = total["matched"]
    overall_rate = total_matched * 100 // total_lines if total_lines > 0 else 0
    print(
        f"总计: {total_lines} 行, 匹配率 {overall_rate}%, "
        f"配对 {total_matched} 行, "
        f"缩放 {total['scaled']} 行, "
        f"seq{SEQ_LARGE}独有 {total['unmatched']} 行, "
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
