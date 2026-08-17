"""Resolve API configuration file inputs shared by both engine entrypoints."""

from __future__ import annotations

import glob
import os
from pathlib import Path


def resolve_config_files(inputs: str | list[str]) -> list[str]:
    """Expand files, directories, and glob patterns into unique paths."""
    resolved: list[str] = []
    seen: set[str] = set()
    # 逗号只用于命令行输入集合，单个路径仍可通过显式文件路径传入。
    raw_inputs = [inputs] if isinstance(inputs, str) else inputs
    expanded_inputs = [part.strip() for item in raw_inputs for part in str(item).split(",")]
    for raw_input in expanded_inputs:
        value = str(raw_input).strip()
        if not value:
            raise ValueError("--api_config_file 不允许为空输入")
        path = Path(value)
        if path.is_dir():
            # 目录只展开当前层 txt，递归范围必须由 ** glob 明确表达。
            candidates = sorted(path.glob("*.txt"))
            if not candidates:
                raise FileNotFoundError(f"目录中没有配置文件: {value}")
        elif path.is_file():
            candidates = [path]
        elif glob.has_magic(value):
            candidates = [Path(item) for item in glob.glob(value, recursive=True)]
            candidates = [item for item in sorted(candidates) if item.is_file()]
            if not candidates:
                raise FileNotFoundError(f"配置输入没有匹配文件: {value}")
        else:
            raise FileNotFoundError(f"No config file found: {value}")
        for candidate in candidates:
            normalized = os.path.abspath(os.fspath(candidate))
            if normalized not in seen:
                seen.add(normalized)
                resolved.append(normalized)
    if not resolved:
        raise FileNotFoundError("--api_config_file 没有解析出配置文件")
    return resolved
