from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from tester.api_config.config_analyzer import APIConfig

from .spec import CompareCase


def case_from_config_line(config_line: str, case_id: str = "case_0") -> CompareCase:
    raw_config = config_line.strip()
    api_config = APIConfig(raw_config)
    return CompareCase(
        id=case_id,
        shape=(),
        tensors={"api_config": api_config},
        metadata={
            "api_name": api_config.api_name,
            "raw_config": raw_config,
        },
    )


def cases_from_config_lines(config_lines: Iterable[str]) -> list[CompareCase]:
    cases = []
    for index, line in enumerate(config_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cases.append(case_from_config_line(stripped, case_id=f"case_{len(cases)}"))
    return cases


def cases_from_config_file(config_file: str | Path) -> list[CompareCase]:
    path = Path(config_file)
    return cases_from_config_lines(path.read_text().splitlines())
