#!/usr/bin/env python3
"""Enforce the minimum comment ratio for newly added source lines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

# The check deliberately uses an allowlist so data and configuration files stay excluded.
SOURCE_SUFFIXES = frozenset({".py", ".sh"})
# One comment line is required for every ten newly added, non-blank source lines.
MIN_COMMENT_RATIO = 0.10


@dataclass(frozen=True)
class LineCounts:
    """Counts used to calculate the ratio for a source diff."""

    comments: int = 0
    code: int = 0

    @property
    def relevant_lines(self) -> int:
        return self.comments + self.code

    @property
    def ratio(self) -> float:
        return self.comments / self.relevant_lines if self.relevant_lines else 1.0


def is_source_file(path: str | None) -> bool:
    """Return whether a diff path belongs to a source type covered by this policy."""

    return path is not None and PurePosixPath(path).suffix.lower() in SOURCE_SUFFIXES


def is_comment_line(path: str, content: str) -> bool:
    """Recognize line comments and the delimiters of Python documentation strings."""

    if content.startswith("#"):
        return True
    return PurePosixPath(path).suffix == ".py" and content.startswith(('"""', "'''"))


def added_line_counts(diff: str) -> LineCounts:
    """Count added comment and code lines from a zero-context Git diff."""

    comments = 0
    code = 0
    current_path: str | None = None

    for line in diff.splitlines():
        # Git emits this header before the content of each file's hunks.
        if line.startswith("+++ "):
            current_path = line.removeprefix("+++ b/")
            if current_path == "/dev/null":
                current_path = None
            continue
        # File headers and removed lines must not affect an added-line policy.
        if not line.startswith("+") or not is_source_file(current_path):
            continue

        content = line[1:].strip()
        # Blank lines make neither code nor comments more difficult to maintain.
        if not content:
            continue
        # A shebang selects an interpreter; it is not explanatory documentation.
        if content.startswith("#!"):
            continue
        if is_comment_line(current_path, content):
            comments += 1
        else:
            code += 1

    return LineCounts(comments=comments, code=code)


def git_diff(base_ref: str | None) -> str:
    """Read staged changes locally or the completed change range in CI."""

    command = ["git", "diff", "--no-ext-diff", "--unified=0"]
    if base_ref:
        command.append(f"{base_ref}...HEAD")
    else:
        command.append("--cached")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        help="compare this Git revision with HEAD instead of inspecting the staged diff",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        counts = added_line_counts(git_diff(args.base_ref))
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr, end="")
        return error.returncode

    ratio = counts.ratio * 100
    minimum = MIN_COMMENT_RATIO * 100
    print(
        f"Added source lines: {counts.relevant_lines} "
        f"({counts.comments} comments, {counts.code} code); comment ratio: {ratio:.2f}%"
    )
    if counts.relevant_lines and counts.ratio < MIN_COMMENT_RATIO:
        print(
            f"Comment ratio must be at least {minimum:.0f}% for newly added .py and .sh lines. "
            "Add explanatory comments before committing.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
