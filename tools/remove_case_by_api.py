from __future__ import annotations

import glob
import re


def delete_lines_with_keywords(file_pattern, keyword_set, case_sensitive=True):
    """删除匹配模式文件中包含关键字的行
    :param file_pattern: 文件匹配模式（如"A*"）
    :param keyword_set: 关键字集合
    :param case_sensitive: 是否区分大小写
    """
    # 获取匹配文件列表
    target_files = glob.glob(file_pattern)
    if not target_files:
        print(f"警告：未找到匹配 {file_pattern} 的文件")
        return

    # 预先编译正则表达式
    flags = 0 if case_sensitive else re.IGNORECASE
    patterns = [re.compile(re.escape(kw), flags) for kw in keyword_set]
    total_removed = 0

    for file_path in target_files:
        try:
            # 读取文件内容
            with open(file_path) as f:
                lines = f.readlines()

            # 过滤包含关键字的行
            original_count = len(lines)
            new_lines = [
                line for line in lines if not any(pattern.search(line) for pattern in patterns)
            ]
            removed_count = original_count - len(new_lines)
            total_removed += removed_count

            # 写回文件
            with open(file_path, "w") as f:
                f.writelines(new_lines)
            print(
                f"处理 {file_path}: 原始行数 {original_count}, 删除 {removed_count} 行, 保留 {len(new_lines)} 行"
            )

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e!s}")

    print(f"\n处理完成！共处理 {len(target_files)} 个文件, 总计删除 {total_removed} 行")


def load_keywords(keyword_file):
    """从文件加载关键字集合"""
    try:
        with open(keyword_file) as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"错误：关键字文件 {keyword_file} 不存在")
        exit(1)


if __name__ == "__main__":
    # 配置参数
    FILE_PATTERN = "tester/api_config/monitor_config/accuracy/GPU/monitoring_configs_*.txt"  # 匹配所有以A开头的文件
    KEYWORD_FILE = "kw.txt"  # 关键字文件名
    CASE_SENSITIVE = True  # 区分大小写（设为False关闭）

    # 执行删除操作
    keywords = load_keywords(KEYWORD_FILE)
    if keywords:
        print(
            f"加载 {len(keywords)} 个关键字: {', '.join(sorted(keywords)[:5])}"
            + ("..." if len(keywords) > 5 else "")
        )
        delete_lines_with_keywords(FILE_PATTERN, keywords, CASE_SENSITIVE)
    else:
        print("警告：关键字集为空，未执行任何操作")
