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
#   api_config_1M_paddleonly.txt   - 原始配置 + 推导1M，合并去重
#   api_config_0size_paddleonly.txt - 0size 配置去重
#   1M_api_extracted.txt           - 从上面提取的 API 名集合
#   0size_api_extracted.txt        - 从上面提取的 API 名集合
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

echo "======================================================================"
echo "API Config 全流程处理"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_DIR"
echo "======================================================================"

# ─── 同步 config_analyzer.py 和 api.yaml ───
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_DIR="$REPO_DIR/tester/api_config"

echo ""
echo "[准备] 同步 config_analyzer.py 和 api.yaml..."
cp "$SOURCE_DIR/config_analyzer.py" "$SCRIPT_DIR/config_analyzer.py"
cp "$SOURCE_DIR/api.yaml" "$SCRIPT_DIR/api.yaml"
echo "  已从 $SOURCE_DIR 复制到 $SCRIPT_DIR"

# ─── 检查输入 ───
if [ ! -f "$INPUT_DIR/api_config_1024.txt" ] || [ ! -f "$INPUT_DIR/api_config_2048.txt" ]; then
    echo "错误：输入目录需要至少包含 api_config_1024.txt 和 api_config_2048.txt"
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
# Step 3: 合并原始配置 + 1M，去重 → api_config_1M_paddleonly.txt
# ============================================================================
echo ""
echo "[Step 3] 合并 + 去重 → api_config_1M_paddleonly.txt..."

# 合并：原始所有 txt + 推导的 1M
python "$SCRIPT_DIR/merge_configs.py" \
    -i "$INPUT_DIR" "$OUTPUT_DIR/api_config_1M.txt" \
    -o "$OUTPUT_DIR/_tmp_merged.txt"

# 去重
python "$SCRIPT_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_merged.txt" \
    -o "$OUTPUT_DIR/api_config_1M_paddleonly.txt"

rm -f "$OUTPUT_DIR/_tmp_merged.txt"

# ============================================================================
# Step 4: 合并原始配置(1024+2048+4096+8192)去重，再生成 0size → api_config_0size_paddleonly.txt
# ============================================================================
echo ""
echo "[Step 4] 合并原始配置(1024+2048+4096+8192) + 去重 + 生成 0-size → api_config_0size_paddleonly.txt..."

# 合并原始 seq 配置（不含 1M）
ORIG_INPUTS=""
for seq in 1024 2048 4096 8192; do
    if [ -f "$INPUT_DIR/api_config_${seq}.txt" ]; then
        ORIG_INPUTS="$ORIG_INPUTS $INPUT_DIR/api_config_${seq}.txt"
    fi
done

python "$SCRIPT_DIR/merge_configs.py" \
    -i $ORIG_INPUTS \
    -o "$OUTPUT_DIR/_tmp_orig_merged.txt"

python "$SCRIPT_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_orig_merged.txt" \
    -o "$OUTPUT_DIR/_tmp_orig_dedup.txt"

# 转 0size
python "$SCRIPT_DIR/to_0_size_config.py" \
    -i "$OUTPUT_DIR/_tmp_orig_dedup.txt" \
    -o "$OUTPUT_DIR/_tmp_0size.txt"

# 去重 0size
python "$SCRIPT_DIR/dedup_config.py" \
    -i "$OUTPUT_DIR/_tmp_0size.txt" \
    -o "$OUTPUT_DIR/api_config_0size_paddleonly.txt"

rm -f "$OUTPUT_DIR/_tmp_orig_merged.txt" "$OUTPUT_DIR/_tmp_orig_dedup.txt" "$OUTPUT_DIR/_tmp_0size.txt"

# ============================================================================
# Step 5: 提取 API 名集合
# ============================================================================
echo ""
echo "[Step 5] 提取 API 名称集合..."

python "$SCRIPT_DIR/extract_api_set.py" \
    -i "$OUTPUT_DIR/api_config_1M_paddleonly.txt" \
    -o "$OUTPUT_DIR/1M_api_extracted.txt"

python "$SCRIPT_DIR/extract_api_set.py" \
    -i "$OUTPUT_DIR/api_config_0size_paddleonly.txt" \
    -o "$OUTPUT_DIR/0size_api_extracted.txt"

# ============================================================================
# 清理中间文件，只保留最终 4 个
# ============================================================================
rm -f "$OUTPUT_DIR/api_config_derived_4096.txt"
rm -f "$OUTPUT_DIR/api_config_1M.txt"

echo ""
echo "======================================================================"
echo "完成！输出目录: $OUTPUT_DIR"
echo "======================================================================"
echo ""
ls -la "$OUTPUT_DIR"
