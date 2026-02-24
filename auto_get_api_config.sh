#!/bin/bash
set -euo pipefail

# --- 配置区 ---
REPO="PFCCLab/PaddleAPITest"        # 仓库所有者和名称
TARGET_TYPE="${1:-}"                # 第一个参数：指定类型（如 precision），留空则处理所有类型
# -------------

echo "🔍 查询仓库: $REPO"

# 1. 获取所有 Release 的标签名
release_data=$(gh release list --repo "$REPO" --limit 1000 --json tagName)
if [ -z "$release_data" ]; then
    echo "❌ 未获取到任何 Release 信息"
    exit 1
fi

# 2. 过滤出符合 {type}-v{数字} 或 {type}-v{数字}patch{数字} 格式的标签
filtered_tags=$(echo "$release_data" | jq -r '.[] | .tagName' | grep -E '^[A-Za-z0-9_-]+-v[0-9]+(patch[0-9]+)?$')
if [ -z "$filtered_tags" ]; then
    echo "❌ 没有找到符合 {type}-v{数字} 或 {type}-v{数字}patch{数字} 格式的 Release"
    exit 1
fi

# 3. 解析所有标签，按类型存储
declare -A tags_by_type        # 类型 -> 所有标签（空格分隔）
declare -A majors_by_type      # 类型 -> 所有主版本号（v0、v1 ... 空格分隔）
while IFS= read -r tag; do
    type_part="${tag%%-v*}"                     # 提取类型（第一个 -v 之前）
    version_part="${tag#*-v}"                   # 提取 v 之后的部分
    major_version="v$(echo "$version_part" | grep -oE '^[0-9]+')"  # 主版本号（v0, v1...）

    tags_by_type["$type_part"]="${tags_by_type[$type_part]:-} $tag"
    majors_by_type["$type_part"]="${majors_by_type[$type_part]:-} $major_version"
done <<< "$filtered_tags"

# 4. 根据用户输入决定要处理的类型列表
if [ -n "$TARGET_TYPE" ]; then
    if [ -n "${tags_by_type[$TARGET_TYPE]:-}" ]; then
        types_to_process=("$TARGET_TYPE")
        echo "🎯 指定类型: $TARGET_TYPE"
    else
        echo "❌ 类型 '$TARGET_TYPE' 不存在或没有符合格式的 Release"
        exit 1
    fi
else
    types_to_process=("${!tags_by_type[@]}")
    echo "📦 将处理所有类型"
fi

# 5. 为每个类型找出最新的主版本号，然后收集该主版本下的所有标签
tags_to_download=()
for type in "${types_to_process[@]}"; do
    echo ""
    echo "🔖 处理类型: $type"

    IFS=' ' read -r -a majors <<< "${majors_by_type[$type]}"
    unique_majors=$(printf "%s\n" "${majors[@]}" | sort -u -V)
    latest_major=$(echo "$unique_majors" | tail -n1)
    echo "   最新主版本: $latest_major"

    IFS=' ' read -r -a tags <<< "${tags_by_type[$type]}"
    for tag in "${tags[@]}"; do
        version_part="${tag#*-v}"
        major_candidate="v$(echo "$version_part" | grep -oE '^[0-9]+')"
        if [ "$major_candidate" = "$latest_major" ]; then
            tags_to_download+=("$tag")
        fi
    done
done

# 6. 去重并排序
unique_tags=($(printf "%s\n" "${tags_to_download[@]}" | sort -u -V))

# 7. 检查 & 下载
if [ ${#unique_tags[@]} -eq 0 ]; then
    echo "❌ 没有找到任何需要下载的标签"
    exit 0
fi

echo ""
echo "⬇️  将下载以下标签:"
printf '   %s\n' "${unique_tags[@]}"

# ── 辅助函数 ────────────────────────────────────────────────

# 计算文件 MD5（兼容 Linux md5sum / macOS md5）
calc_md5() {
    local file="$1"
    if command -v md5sum &>/dev/null; then
        md5sum "$file" | awk '{print $1}'
    else
        md5 -q "$file"
    fi
}

# 对目录内所有 .tar.gz 进行 MD5 校验
#   - 若同名 .md5 文件存在 → 与其比对
#   - 若不存在            → 生成 .md5 文件供后续增量校验
verify_md5() {
    local dir="$1"
    local all_ok=true

    while IFS= read -r archive; do
        local md5_file="${archive}.md5"
        local actual
        actual=$(calc_md5 "$archive")

        if [ -f "$md5_file" ]; then
            # 读取期望值（支持「仅 hash」或「hash  filename」两种格式）
            local expected
            expected=$(awk '{print $1}' "$md5_file")
            if [ "$actual" = "$expected" ]; then
                echo "   ✅ MD5 校验通过: $(basename "$archive")"
            else
                echo "   ❌ MD5 不匹配: $(basename "$archive")"
                echo "      期望: $expected"
                echo "      实际: $actual"
                all_ok=false
            fi
        else
            # 无参考 .md5，生成一份留存
            echo "$actual  $(basename "$archive")" > "$md5_file"
            echo "   📝 已生成 MD5 记录: $(basename "$md5_file")"
        fi
    done < <(find "$dir" -maxdepth 1 -name "*.tar.gz")

    $all_ok
}

# ── 主下载循环 ───────────────────────────────────────────────

for tag in "${unique_tags[@]}"; do
    echo ""
    echo "────────────────────────────────"
    echo "🏷️  标签: $tag"

    dir=".api_config/$tag"

    # ── 短路检测：目录存在且非空时跳过下载 ──────────────────
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "⏭️  目录已存在且非空，跳过下载: $dir"

        # 对已有 .tar.gz 补跑 MD5 校验
        if find "$dir" -maxdepth 1 -name "*.tar.gz" | grep -q .; then
            echo "🔎 对已有文件进行 MD5 校验..."
            if ! verify_md5 "$dir"; then
                echo "⚠️  MD5 校验失败，建议删除 $dir 后重新下载" >&2
            fi
        fi
        continue
    fi

    # ── 下载 ────────────────────────────────────────────────
    echo "⬇️  正在下载: $tag"
    if ! gh release download "$tag" --dir "$dir" --repo "$REPO" --skip-existing; then
        echo "⚠️  下载失败或取消: $tag" >&2
        continue
    fi
    echo "✅ 下载完成: $tag"

    # ── MD5 校验（下载后立即校验）───────────────────────────
    if find "$dir" -maxdepth 1 -name "*.tar.gz" | grep -q .; then
        echo "🔎 校验 MD5..."
        if ! verify_md5 "$dir"; then
            echo "⚠️  MD5 校验失败，跳过解压: $tag" >&2
            continue
        fi
    fi

    # ── 解压 ────────────────────────────────────────────────
    find "$dir" -maxdepth 1 -name "*.tar.gz" | while read -r file; do
        echo "📦 解压: $file"
        tar -xzf "$file" -C "$dir" --one-top-level
    done
    echo "✅ 解压完成: $tag"
done

echo ""
echo "🎉 全部任务结束"
