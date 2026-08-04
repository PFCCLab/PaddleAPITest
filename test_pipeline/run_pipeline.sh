#!/bin/bash
# ============================================================================
# API Config 全流程处理脚本
#
# 从原始 api_config_*.txt 出发，推导1M、验证、合并去重、生成0size、提取API名。
#
# 用法：
#   bash run_pipeline.sh -i <输入目录> -o <输出目录>
#
# 示例：
#   bash run_pipeline.sh -i api_config_0703 -o api_config_dedup_0703
#
# 最终输出：
#   paddleonly_1M/1M.txt           - 1M 配置
#   paddleonly_1M/1M_api.txt       - 1M API 集合
#   paddleonly_4096/4096.txt       - 合并配置
#   paddleonly_4096/4096_api.txt   - 合并配置的 API 集合
#   paddleonly_0size/0size.txt     - 0size 配置
#   paddleonly_0size/0size_api.txt - 0size API 集合
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── 参数解析 ───
INPUT_DIR=""
OUTPUT_DIR=""

while getopts "i:o:h" opt; do
    case $opt in
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h)
            echo "用法: bash $0 -i <输入目录> -o <输出目录>"
            echo "  -i  输入目录（包含 api_config_*.txt）"
            echo "  -o  输出目录"
            exit 0
            ;;
        \?) echo "无效选项: -$OPTARG" >&2; exit 1 ;;
    esac
done

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "错误：必须指定 -i（输入目录）和 -o（输出目录）"
    echo "用法: bash $0 -i <输入目录> -o <输出目录>"
    exit 1
fi

INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
PADDLEONLY_1M_DIR="$OUTPUT_DIR/paddleonly_1M"
PADDLEONLY_0SIZE_DIR="$OUTPUT_DIR/paddleonly_0size"
PADDLEONLY_4096_DIR="$OUTPUT_DIR/paddleonly_4096"
mkdir -p "$PADDLEONLY_1M_DIR" "$PADDLEONLY_0SIZE_DIR" "$PADDLEONLY_4096_DIR"

echo "======================================================================"
echo "API Config 全流程处理"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_DIR"
echo "======================================================================"
echo ""
echo "[输入物料]"
ls -la "$INPUT_DIR"

REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
# 入口位于 test_pipeline 根目录，具体处理器统一收敛到该子目录，避免配置目录承担流水线职责。
PROCESSOR_DIR="$SCRIPT_DIR/config_preprocess"

# ─── 检查输入 ───
# 推导 4096/1M 依赖 1024 和 2048 两个基准文件，缺一不可。
if [ ! -f "$INPUT_DIR/api_config_1024.txt" ] || [ ! -f "$INPUT_DIR/api_config_2048.txt" ]; then
    echo "错误：输入目录需要至少包含 api_config_1024.txt 和 api_config_2048.txt（用于推导 4096/1M）"
    exit 1
fi

# ============================================================================
# Step 1: 推导虚假 4096 并验证（如果有真实 4096）
# ============================================================================
echo ""
echo "[Step 1] 推导 4096 并验证..."

# 所有处理器使用绝对脚本目录调用，调用方可从任意工作目录启动入口。
DERIVED_4096="$OUTPUT_DIR/.derived_4096.txt"
DERIVED_1M="$OUTPUT_DIR/.derived_1M.txt"

python "$PROCESSOR_DIR/derive_api_seq.py" 4096 \
    --small "$INPUT_DIR/api_config_1024.txt" \
    --large "$INPUT_DIR/api_config_2048.txt" \
    -o "$DERIVED_4096"

if [ -f "$INPUT_DIR/api_config_4096.txt" ]; then
    echo ""
    python "$PROCESSOR_DIR/verify_api_seq.py" \
        -d "$DERIVED_4096" \
        -r "$INPUT_DIR/api_config_4096.txt"
else
    echo "  [跳过验证] 未找到真实 api_config_4096.txt"
fi

# ============================================================================
# Step 2: 推导 1M
# ============================================================================
echo ""
echo "[Step 2] 推导 1M (seq=1048576)..."

python "$PROCESSOR_DIR/derive_api_seq.py" 1048576 \
    --small "$INPUT_DIR/api_config_1024.txt" \
    --large "$INPUT_DIR/api_config_2048.txt" \
    -o "$DERIVED_1M"

# ============================================================================
# Step 3: 1M 去重（仅 1M，不与原始配置合并）
# ============================================================================
echo ""
echo "[Step 3] 去重 → paddleonly_1M/1M.txt..."

python "$PROCESSOR_DIR/dedup_config.py" \
    -i "$DERIVED_1M" \
    -o "$PADDLEONLY_1M_DIR/1M.txt"

# ============================================================================
# Step 4: 合并原始配置并生成 0size
# ============================================================================
echo ""
echo "[Step 4] 合并原始配置(按实际存在的 seq) + 去重，并生成 0-size..."

# 合并原始 seq 配置。
ORIG_INPUTS=""
ORIG_SEQS=""
for seq in 1024 2048 4096 8192; do
    if [ -f "$INPUT_DIR/api_config_${seq}.txt" ]; then
        ORIG_INPUTS="$ORIG_INPUTS $INPUT_DIR/api_config_${seq}.txt"
        ORIG_SEQS="${ORIG_SEQS}${ORIG_SEQS:+_}${seq}"
    else
        echo "  [提示] 未找到 api_config_${seq}.txt，跳过"
    fi
done

if [ -z "$ORIG_INPUTS" ]; then
    echo "错误：未找到任何 api_config_{1024,2048,4096,8192}.txt，无法生成合并配置"
    exit 1
fi

ORIG_MERGED_NAME="4096.txt"
echo "  实际参与合并的 seq: $ORIG_SEQS  →  paddleonly_4096/$ORIG_MERGED_NAME"

python "$PROCESSOR_DIR/merge_configs.py" \
    -i $ORIG_INPUTS \
    -o "$OUTPUT_DIR/_tmp_orig_merged.txt"

python "$PROCESSOR_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_orig_merged.txt" \
    -o "$PADDLEONLY_4096_DIR/$ORIG_MERGED_NAME"

# 转 0size
python "$PROCESSOR_DIR/to_0_size_config.py" \
    -i "$PADDLEONLY_4096_DIR/$ORIG_MERGED_NAME" \
    -o "$OUTPUT_DIR/_tmp_0size.txt"

# 去重 0size
python "$PROCESSOR_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_0size.txt" \
    -o "$PADDLEONLY_0SIZE_DIR/0size.txt"

rm -f "$OUTPUT_DIR/_tmp_orig_merged.txt" "$OUTPUT_DIR/_tmp_0size.txt"

# ============================================================================
# Step 5: 提取 API 名集合
# ============================================================================
echo ""
echo "[Step 5] 提取 API 名称集合..."

python "$PROCESSOR_DIR/extract_api_set.py" \
    -i "$PADDLEONLY_1M_DIR/1M.txt" \
    -o "$PADDLEONLY_1M_DIR/1M_api.txt"

python "$PROCESSOR_DIR/extract_api_set.py" \
    -i "$PADDLEONLY_4096_DIR/$ORIG_MERGED_NAME" \
    -o "$PADDLEONLY_4096_DIR/4096_api.txt"

python "$PROCESSOR_DIR/extract_api_set.py" \
    -i "$PADDLEONLY_0SIZE_DIR/0size.txt" \
    -o "$PADDLEONLY_0SIZE_DIR/0size_api.txt"

# ============================================================================
# 清理中间文件，只保留最终结果
# ============================================================================
rm -f "$DERIVED_4096" "$DERIVED_1M" "$OUTPUT_DIR/_tmp_orig_merged.txt" "$OUTPUT_DIR/_tmp_0size.txt"

echo ""
echo "======================================================================"
echo "完成！输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""
ls -la "$OUTPUT_DIR"
