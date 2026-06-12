# 缩小大 Tensor 配置小工具
# @author: cangtianhuang
# @date: 2026-06-11

"""
用法:
    python shrink_large_configs.py \
        --error-logs  <log_dir1> [<log_dir2> ...] \
        --source-configs <config1.txt> [<config2.txt> ...] \
        --output <output.txt> \
        --factor <N>           # 将元素数量缩小到原来的 1/N（如 4、8、16）
        [--threshold <M>]      # 只缩小元素数达到此阈值的 Tensor（默认 1048576，即 1M）
        [--error-types crash oom timeout numpy_error]  # 默认全部四种

说明:
    脚本从 error-logs 目录中读取 api_config_crash.txt / api_config_oom.txt /
    api_config_timeout.txt / api_config_numpy_error.txt，收集出错的配置行。
    再从 source-configs 中找到对应行，对其中 Tensor 的 shape 进行等比缩小：
      - 每个维度除以 factor^(1/ndim)，保持各维度比例尽量一致
      - 若某维度为 0 或 1，则不参与缩放
      - list[...] / tuple(...) 中与 Tensor shape 同步缩放的整数也同步处理
      - strides=[...] 按相同倍率缩放（跳过 0 和 1）
      - -1（动态维度）保留不变

示例:
    python shrink_large_configs.py \
        --error-logs workspace/0601_dsV4/test_log_1_dsv4_1M \
                     workspace/0601_dsV4/test_log_5_v2_1M_fix_tofix \
        --source-configs workspace/0601_dsV4/dsv4_1M_tofix.txt \
                         workspace/0601_dsV4/v2_1M_fix_tofix.txt \
        --output workspace/0601_dsV4/shrunk_4x.txt \
        --factor 4
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Match a full Tensor(...) token including optional strides/is_contiguous fields.
# Captures:
#   group 1: paddle.Size([...]) content  — the comma-sep integers
#   group 2: dtype string (without quotes)
#   group 3: extra fields like ,is_contiguous=False,strides=[1,2048]   (may be empty)
_TENSOR_RE = re.compile(r'Tensor\(paddle\.Size\(\[([^\]]*)\]\),"([^"]+)"((?:,[^)]*)?)\)')

# Match strides=[...] inside the extra-fields group
_STRIDES_RE = re.compile(r"(strides=\[)([^\]]*?)(\])")

# Match list[...] or tuple(...) — shape-like integer sequences used as args
# e.g. list[1,1048576,262144,] or tuple(1,1048576,262144,)
_LIST_ARG_RE = re.compile(r"(list\[|tuple\()([^\]\)]*)([\]\)])")

# Match a bare integer (possibly negative for -1 sentinel)
_INT_RE = re.compile(r"-?\d+")


# ---------------------------------------------------------------------------
# Shape / value scaling helpers
# ---------------------------------------------------------------------------


def _numel(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


def _scale_dims(dims: list[int], factor: float, threshold: int) -> list[int]:
    """
    Scale a shape so its element count is reduced by approximately `factor`.

    Strategy:
      1. Compute which dims are "large" (> 1 and > threshold ^ (1/ndim)).
      2. Distribute the reduction evenly across large dims.
      3. -1 (dynamic) is kept as-is; 0 and 1 are kept as-is.
    """
    if not dims:
        return dims
    numel = _numel(dims)
    if numel < threshold:
        return dims  # already small enough

    new_dims = list(dims)
    # Identify scalable indices (not 0, not 1, not -1)
    scalable = [i for i, d in enumerate(dims) if d > 1 and d != -1]
    if not scalable:
        return new_dims

    # We want product(new[i] for i in scalable) ≈ product(old[i]) / factor
    # Distribute reduction: each dim gets divided by factor^(1/len(scalable))
    per_dim_ratio = factor ** (1.0 / len(scalable))
    for i in scalable:
        new_val = max(1, round(dims[i] / per_dim_ratio))
        new_dims[i] = new_val

    return new_dims


def _scale_single_value(v: int, scale: float) -> int:
    """Scale a single integer value, keeping 0, 1, -1 unchanged."""
    if v in (0, 1, -1):
        return v
    return max(1, round(v / scale))


def _dims_from_str(s: str) -> list[int]:
    """Parse comma-separated integer string into list[int], ignoring empty."""
    result = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            result.append(int(tok))
        except ValueError:
            pass  # skip non-integer tokens (shouldn't happen in shape)
    return result


def _dims_to_str(dims: list[int]) -> str:
    return ", ".join(str(d) for d in dims)


# ---------------------------------------------------------------------------
# Per-line transform
# ---------------------------------------------------------------------------


def _compute_scale_for_line(line: str, factor: float, threshold: int) -> float | None:
    """
    Determine the actual scale factor to apply to this config line.
    We look at the *largest* tensor numel; if it exceeds threshold, scale by factor.
    Returns None if no tensor exceeds threshold (line needs no change).
    """
    max_numel = 0
    for m in _TENSOR_RE.finditer(line):
        dims = _dims_from_str(m.group(1))
        n = _numel(dims)
        if n > max_numel:
            max_numel = n
    if max_numel < threshold:
        return None
    return factor


def _replace_strides(extra: str, old_dims: list[int], new_dims: list[int]) -> str:
    """
    Scale strides= values inside the extra-fields string.
    We compute a per-element scale based on old vs new tensor numel.
    """
    # Build a mapping: old dim value → new dim value
    # (for the simple proportional case used in actual strides)
    # actual scale = old_numel / new_numel
    old_numel = _numel([d for d in old_dims if d > 0])
    new_numel = _numel([d for d in new_dims if d > 0])
    if old_numel == 0 or new_numel == 0:
        return extra
    scale = old_numel / new_numel  # > 1 (we are shrinking)

    def replace_strides_match(m: re.Match) -> str:
        prefix, content, suffix = m.group(1), m.group(2), m.group(3)
        vals = _dims_from_str(content)
        new_vals = [_scale_single_value(v, scale) for v in vals]
        return prefix + ", ".join(str(v) for v in new_vals) + suffix

    return _STRIDES_RE.sub(replace_strides_match, extra)


def _transform_line(line: str, factor: float, threshold: int) -> str:
    """
    Return a new config line with all large Tensor shapes scaled down by `factor`.
    Also scales list[...]/tuple(...) args and strides proportionally.
    """
    line = line.rstrip("\n")

    actual_scale = _compute_scale_for_line(line, factor, threshold)
    if actual_scale is None:
        return line  # nothing to do

    # --- Step 1: collect old→new dim mappings per Tensor occurrence ----------
    # We process Tensor(...) tokens and record (old_dims, new_dims)
    tensor_replacements: list[tuple[str, str]] = []  # (old_token, new_token)

    for m in _TENSOR_RE.finditer(line):
        old_token = m.group(0)
        old_dims = _dims_from_str(m.group(1))
        dtype = m.group(2)
        extra = m.group(3)  # e.g. ",is_contiguous=False,strides=[1,2048]"

        new_dims = _scale_dims(old_dims, actual_scale, threshold)
        new_shape_str = ", ".join(str(d) for d in new_dims)

        # Scale strides in extra field
        new_extra = _replace_strides(extra, old_dims, new_dims)

        new_token = f'Tensor(paddle.Size([{new_shape_str}]),"{dtype}"{new_extra})'
        if old_token != new_token:
            tensor_replacements.append((old_token, new_token))

    # Apply Tensor replacements (use plain string replace to preserve order)
    new_line = line
    for old_tok, new_tok in tensor_replacements:
        new_line = new_line.replace(old_tok, new_tok, 1)

    # --- Step 2: scale list[...] / tuple(...) args that contain large ints ---
    # We only scale values that look like shape dimensions (positive integers > 1)
    # and are larger than a conservative threshold.
    # Use a heuristic: scale any integer > sqrt(threshold) in list/tuple args.
    list_threshold = max(4, int(math.sqrt(threshold)))

    def replace_list_arg(m: re.Match) -> str:
        prefix, content, suffix = m.group(1), m.group(2), m.group(3)
        tokens = content.split(",")
        new_tokens = []
        for tok in tokens:
            stripped = tok.strip()
            if stripped == "" or stripped == "-1":
                new_tokens.append(tok)
                continue
            try:
                v = int(stripped)
                if v > list_threshold:
                    new_v = _scale_single_value(v, actual_scale)
                    new_tokens.append(tok.replace(stripped, str(new_v), 1))
                else:
                    new_tokens.append(tok)
            except ValueError:
                new_tokens.append(tok)
        return prefix + ",".join(new_tokens) + suffix

    new_line = _LIST_ARG_RE.sub(replace_list_arg, new_line)

    return new_line


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

ERROR_FILE_NAMES = {
    "crash": "api_config_crash.txt",
    "oom": "api_config_oom.txt",
    "timeout": "api_config_timeout.txt",
    "numpy_error": "api_config_numpy_error.txt",
}


def collect_error_lines(log_dirs: list[Path], error_types: list[str]) -> set[str]:
    """Collect all lines from the specified error files in each log directory."""
    error_lines: set[str] = set()
    for log_dir in log_dirs:
        for etype in error_types:
            fname = ERROR_FILE_NAMES[etype]
            fpath = log_dir / fname
            if not fpath.exists():
                print(f"[skip] {fpath} not found", file=sys.stderr)
                continue
            count = 0
            with open(fpath) as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line:
                        error_lines.add(line)
                        count += 1
            print(f"[info] {fpath}: {count} lines", file=sys.stderr)
    print(f"[info] total unique error lines: {len(error_lines)}", file=sys.stderr)
    return error_lines


def load_source_lines(source_configs: list[Path]) -> list[str]:
    """Load all lines from source config files (preserving order, dedup by line content)."""
    seen: set[str] = set()
    lines: list[str] = []
    for cfg in source_configs:
        if not cfg.exists():
            print(f"[warn] source config not found: {cfg}", file=sys.stderr)
            continue
        with open(cfg) as f:
            for raw_line in f:
                line = raw_line.strip()
                if line and line not in seen:
                    seen.add(line)
                    lines.append(line)
        print(f"[info] loaded {cfg}: {len(lines)} unique lines so far", file=sys.stderr)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shrink large Tensor shapes in API configs that caused crash/oom/timeout/numpy_error."
    )
    parser.add_argument(
        "--error-logs",
        nargs="+",
        required=True,
        metavar="LOG_DIR",
        help="One or more test_log_* directories containing api_config_*.txt error files.",
    )
    parser.add_argument(
        "--source-configs",
        nargs="+",
        required=True,
        metavar="CONFIG_FILE",
        help="Source config .txt files (e.g. dsv4_1M_tofix.txt) to read original configs from.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="OUTPUT_FILE",
        help="Output .txt file with shrunken configs.",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=8.0,
        help="Reduce element count by this factor (default: 8). "
        "Use 4, 8, 16, etc. for 4×/8×/16× shrinkage.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1048576,
        help="Only shrink Tensors whose element count reaches this value (default: 1048576 = 1M).",
    )
    parser.add_argument(
        "--error-types",
        nargs="+",
        default=["crash", "oom", "timeout", "numpy_error"],
        choices=["crash", "oom", "timeout", "numpy_error"],
        help="Which error types to include (default: all four).",
    )
    parser.add_argument(
        "--keep-unchanged",
        action="store_true",
        help="Also write lines from source that matched error set but had no large tensors.",
    )
    args = parser.parse_args()

    log_dirs = [Path(p) for p in args.error_logs]
    source_cfgs = [Path(p) for p in args.source_configs]
    output_path = Path(args.output)

    # 1. Collect error lines
    error_lines = collect_error_lines(log_dirs, args.error_types)
    if not error_lines:
        print("[warn] No error lines found — output will be empty.", file=sys.stderr)

    # 2. Load source configs
    source_lines = load_source_lines(source_cfgs)
    if not source_lines:
        print("[error] No source config lines loaded.", file=sys.stderr)
        sys.exit(1)

    # 3. Filter source lines to those in error set
    matched = [l for l in source_lines if l in error_lines]
    print(
        f"[info] source lines matching error set: {len(matched)} / {len(source_lines)}",
        file=sys.stderr,
    )

    not_in_source = error_lines - set(source_lines)
    if not_in_source:
        print(
            f"[warn] {len(not_in_source)} error lines not found in source configs "
            f"(from other run / already shrunk?)",
            file=sys.stderr,
        )

    # 4. Transform matched lines
    output_lines: list[str] = []
    changed = 0
    unchanged = 0
    for line in matched:
        new_line = _transform_line(line, args.factor, args.threshold)
        if new_line != line:
            output_lines.append(new_line)
            changed += 1
        else:
            unchanged += 1
            if args.keep_unchanged:
                output_lines.append(line)

    print(f"[info] transformed (shape changed): {changed}", file=sys.stderr)
    print(f"[info] no large tensors (skipped): {unchanged}", file=sys.stderr)

    # 5. Deduplicate while preserving order
    seen_out: set[str] = set()
    deduped: list[str] = []
    for l in output_lines:
        if l not in seen_out:
            seen_out.add(l)
            deduped.append(l)
    print(f"[info] output lines after dedup: {len(deduped)}", file=sys.stderr)

    # 6. Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for l in deduped:
            f.write(l + "\n")
    print(f"[done] written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
