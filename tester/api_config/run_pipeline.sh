#!/bin/bash
# ============================================================================
# API Config 全流程处理脚本
#
# 从原始 api_config_*.txt 出发，推导1M、验证、生成0size、合并去重、提取API名。
#
# 用法：
#   bash run_pipeline.sh -i <输入目录> -o <输出目录>
#
# 示例：
#   bash run_pipeline.sh -i api_config_0703 -o api_config_dedup_0703
#
# 最终输出：
#   paddleonly/1M_preprocessed.txt                          - 推导 1M，去重（仅 1M）
#   paddleonly_2048_4096_8192/2048_4096_8192_preprocessed.txt - 原始配置合并去重，日志会显示实际参与合并的 seq
#   paddleonly_0size/0size_preprocessed.txt                 - 0size 配置去重
#   paddleonly/1M_api_extracted.txt                         - 从上面提取的 API 名集合
#   paddleonly_2048_4096_8192/2048_4096_8192_api_extracted.txt - 从上面提取的 API 名集合
#   paddleonly_0size/0size_api_extracted.txt                - 从上面提取的 API 名集合
#
# 注：1024/2048 为必需（用于推导 4096/1M），缺失会直接报错退出；
#     4096 缺失仅跳过验证与该 seq 的合并，输出文件名会自动去掉对应部分。
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
PADDLEONLY_DIR="$OUTPUT_DIR/paddleonly"
PADDLEONLY_0SIZE_DIR="$OUTPUT_DIR/paddleonly_0size"
PADDLEONLY_2048_4096_8192_DIR="$OUTPUT_DIR/paddleonly_2048_4096_8192"
mkdir -p "$PADDLEONLY_DIR" "$PADDLEONLY_0SIZE_DIR" "$PADDLEONLY_2048_4096_8192_DIR"

echo "======================================================================"
echo "API Config 全流程处理"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_DIR"
echo "======================================================================"
echo ""
echo "[输入物料]"
ls -la "$INPUT_DIR"

# ─── 同步 config_analyzer.py 和 api.yaml ───
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="$REPO_DIR/tester/api_config"

# echo ""
# echo "[准备] 同步 config_analyzer.py 和 api.yaml..."
# cp "$SOURCE_DIR/config_analyzer.py" "$SCRIPT_DIR/config_analyzer.py"
# cp "$SOURCE_DIR/api.yaml" "$SCRIPT_DIR/api.yaml"
# echo "  已从 $SOURCE_DIR 复制到 $SCRIPT_DIR"

# ─── 检查输入 ───
# 推导 4096/1M 依赖 1024 和 2048 两个基准文件，缺一不可
if [ ! -f "$INPUT_DIR/api_config_1024.txt" ] || [ ! -f "$INPUT_DIR/api_config_2048.txt" ]; then
    echo "错误：输入目录需要至少包含 api_config_1024.txt 和 api_config_2048.txt（用于推导 4096/1M）"
    exit 1
fi

# ============================================================================
# Step 1: 推导虚假 4096 并验证（如果有真实 4096）
# ============================================================================
echo ""
echo "[Step 1] 推导 4096 并验证..."

python "$SCRIPT_DIR/derive_api_seq.py" 4096 \
    --small "$INPUT_DIR/api_config_1024.txt" \
    --large "$INPUT_DIR/api_config_2048.txt" \
    -o "$OUTPUT_DIR/api_config_derived_4096.txt"

if [ -f "$INPUT_DIR/api_config_4096.txt" ]; then
    echo ""
    python "$SCRIPT_DIR/verify_api_seq.py" \
        -d "$OUTPUT_DIR/api_config_derived_4096.txt" \
        -r "$INPUT_DIR/api_config_4096.txt"
else
    echo "  [跳过验证] 未找到真实 api_config_4096.txt"
fi

# ============================================================================
# Step 2: 推导 1M
# ============================================================================
echo ""
echo "[Step 2] 推导 1M (seq=1048576)..."

python "$SCRIPT_DIR/derive_api_seq.py" 1048576 \
    --small "$INPUT_DIR/api_config_1024.txt" \
    --large "$INPUT_DIR/api_config_2048.txt" \
    -o "$OUTPUT_DIR/api_config_1M.txt"

# ============================================================================
# Step 3: 1M 去重 → paddleonly/1M_preprocessed.txt（仅 1M，不与原始配置合并）
# ============================================================================
echo ""
echo "[Step 3] 去重 → paddleonly/1M_preprocessed.txt..."

python "$SCRIPT_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/api_config_1M.txt" \
    -o "$PADDLEONLY_DIR/1M_preprocessed.txt"

# ============================================================================
# Step 4: 合并原始配置(1024+2048+4096，按实际存在的文件)去重
#         再由该去重结果生成 0size → paddleonly_0size/0size_preprocessed.txt
# ============================================================================
echo ""
echo "[Step 4] 合并原始配置(按实际存在的 seq) + 去重，并生成 0-size..."

# 合并原始 seq 配置（不含 1M、8192），产物固定命名，日志保留实际存在的 seq
ORIG_INPUTS=""
ORIG_SEQS=""
for seq in 1024 2048 4096; do
    if [ -f "$INPUT_DIR/api_config_${seq}.txt" ]; then
        ORIG_INPUTS="$ORIG_INPUTS $INPUT_DIR/api_config_${seq}.txt"
        ORIG_SEQS="${ORIG_SEQS}${ORIG_SEQS:+_}${seq}"
    else
        echo "  [提示] 未找到 api_config_${seq}.txt，跳过"
    fi
done

if [ -z "$ORIG_INPUTS" ]; then
    echo "错误：未找到任何 api_config_{1024,2048,4096}.txt，无法生成合并配置"
    exit 1
fi

ORIG_MERGED_NAME="2048_4096_8192_preprocessed.txt"
echo "  实际参与合并的 seq: $ORIG_SEQS  →  paddleonly_2048_4096_8192/$ORIG_MERGED_NAME"

python "$SCRIPT_DIR/merge_configs.py" \
    -i $ORIG_INPUTS \
    -o "$OUTPUT_DIR/_tmp_orig_merged.txt"

python "$SCRIPT_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_orig_merged.txt" \
    -o "$PADDLEONLY_2048_4096_8192_DIR/$ORIG_MERGED_NAME"

# 转 0size
python "$SCRIPT_DIR/to_0_size_config.py" \
    -i "$PADDLEONLY_2048_4096_8192_DIR/$ORIG_MERGED_NAME" \
    -o "$OUTPUT_DIR/_tmp_0size.txt"

# 去重 0size
python "$SCRIPT_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_0size.txt" \
    -o "$PADDLEONLY_0SIZE_DIR/0size_preprocessed.txt"

rm -f "$OUTPUT_DIR/_tmp_orig_merged.txt" "$OUTPUT_DIR/_tmp_0size.txt"

# ============================================================================
# Step 5: 提取 API 名集合
# ============================================================================
echo ""
echo "[Step 5] 提取 API 名称集合..."

python "$SCRIPT_DIR/extract_api_set.py" \
    -i "$PADDLEONLY_DIR/1M_preprocessed.txt" \
    -o "$PADDLEONLY_DIR/1M_api_extracted.txt"

python "$SCRIPT_DIR/extract_api_set.py" \
    -i "$PADDLEONLY_2048_4096_8192_DIR/$ORIG_MERGED_NAME" \
    -o "$PADDLEONLY_2048_4096_8192_DIR/2048_4096_8192_api_extracted.txt"

python "$SCRIPT_DIR/extract_api_set.py" \
    -i "$PADDLEONLY_0SIZE_DIR/0size_preprocessed.txt" \
    -o "$PADDLEONLY_0SIZE_DIR/0size_api_extracted.txt"

# ============================================================================
# 清理中间文件，只保留最终结果
# ============================================================================
rm -f "$OUTPUT_DIR/api_config_derived_4096.txt"
rm -f "$OUTPUT_DIR/api_config_1M.txt"

echo ""
echo "======================================================================"
echo "完成！输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""
ls -la "$OUTPUT_DIR"
