"""Paddle API 配置行的共享解析工具。"""

# 每个逻辑调用必须以 paddle. 开头并具有完整外层括号。
# 同一物理行允许连续记录多个调用，调用之间可以没有空白分隔符。
# 字符串和嵌套参数中的 paddle. 只属于当前调用，不能成为切分边界。
# 不能无损归属的残余文本必须报错，由调用方决定拒绝或终止。

from __future__ import annotations

import string


# 采集配置中的 API 名只接受 ASCII 标识符，避免 Unicode 或运算符混入边界。
API_NAME_CHARS = frozenset(string.ascii_letters + string.digits + "_.")


def matching_close(text, start, opening, closing):
    """返回与 start 处括号匹配的结束位置。"""
    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _api_name_end(text, start):
    """返回 API 名结束位置，并阻止粘连的第二个 paddle 前缀混入名称。"""
    index = start
    while index < len(text) and text[index] in API_NAME_CHARS:
        # 名称内部再次出现 paddle. 表示前一个调用缺少左括号，而非更长 API 名。
        if index > start and text.startswith("paddle.", index):
            break
        index += 1
    return index


def _is_valid_api_name(api_name):
    """验证 paddle 后的每级名称均为 Python 标识符。"""
    parts = api_name.split(".")
    return parts[0] == "paddle" and len(parts) > 1 and all(
        part
        and (part[0] in string.ascii_letters or part[0] == "_")
        and all(char in string.ascii_letters + string.digits + "_" for char in part[1:])
        for part in parts[1:]
    )


def split_top_level_calls(text):
    """将一行中连续的顶层 paddle.* 调用无损拆分。"""
    calls = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if not text.startswith("paddle.", cursor):
            # 非 paddle 残余文本不能静默丢弃，否则会造成测试覆盖缺失。
            raise ValueError(
                "顶层调用结束后存在无法识别的内容，期望下一个 paddle.* 调用"
            )

        name_end = _api_name_end(text, cursor)
        if not _is_valid_api_name(text[cursor:name_end]):
            raise ValueError("顶层 paddle.* API 名不合法")
        if name_end >= len(text) or text[name_end] != "(":
            raise ValueError("顶层 paddle.* 调用缺少左括号")
        closing = matching_close(text, name_end, "(", ")")
        if closing is None:
            raise ValueError("顶层 paddle.* 调用括号不匹配")
        # 调用文本原样保留，规范化阶段不重排参数或改变空白。
        calls.append(text[cursor : closing + 1])
        cursor = closing + 1
    return calls
