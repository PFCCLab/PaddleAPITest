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
    
    # 存入关联数组
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
    
    # 5.1 获取该类型的所有主版本号，去重后排序，取最新
    IFS=' ' read -r -a majors <<< "${majors_by_type[$type]}"
    unique_majors=$(printf "%s\n" "${majors[@]}" | sort -u -V)  # -V 自然排序 v0 < v1
    latest_major=$(echo "$unique_majors" | tail -n1)
    echo "   最新主版本: $latest_major"
    
    # 5.2 遍历该类型的所有标签，筛选出主版本等于 latest_major 的
    IFS=' ' read -r -a tags <<< "${tags_by_type[$type]}"
    for tag in "${tags[@]}"; do
        version_part="${tag#*-v}"
        major_candidate="v$(echo "$version_part" | grep -oE '^[0-9]+')"
        if [ "$major_candidate" = "$latest_major" ]; then
            tags_to_download+=("$tag")
        fi
    done
done

# 6. 去重并排序（按标签名，方便查看）
unique_tags=($(printf "%s\n" "${tags_to_download[@]}" | sort -u -V))

# 7. 下载
if [ ${#unique_tags[@]} -eq 0 ]; then
    echo "❌ 没有找到任何需要下载的标签"
    exit 0
fi

echo ""
echo "⬇️  将下载以下标签:"
printf '   %s\n' "${unique_tags[@]}"

for tag in "${unique_tags[@]}"; do
    echo ""
    echo "⬇️  正在下载: $tag"
    dir=".api_config/$tag"
    if gh release download "$tag" --dir $dir --repo "$REPO" --skip-existing; then
        echo "✅ 下载完成: $tag"
        find "$dir" -maxdepth 1 -name "*.tar.gz" | while read -r file; do
            echo "📦 解压: $file"
            tar -xzf "$file" -C "$dir" --one-top-level
        done
        echo "✅ 解压完成: $tag"
    else
        echo "⚠️  下载失败或取消: $tag" >&2
    fi
done

echo ""
echo "🎉 全部任务结束"
