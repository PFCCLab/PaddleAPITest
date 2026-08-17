"""Parse compute-sanitizer output and remove known non-actionable diagnostics."""

from __future__ import annotations

import re
from dataclasses import dataclass

SANITIZER_PREFIX = "========="
SANITIZER_CUDA_API_ERROR = "CUDA API Error:"
SANITIZER_PROGRAM_HIT = "Program hit"
SANITIZER_ERROR_SUMMARY = "ERROR SUMMARY:"
CUDA_VERSION_ERROR_RE = re.compile(
    r"cudaVersion argument [(][0-9]+[)] exceeds the driver version [(][0-9]+[)]"
)


@dataclass(frozen=True)
class SanitizerAnalysis:
    """Filtered output and whether the sanitizer failure is entirely ignorable."""

    output: str
    only_ignored_diagnostics: bool


def _is_diagnostic_header(line):
    return line.startswith(SANITIZER_PREFIX) and (
        SANITIZER_CUDA_API_ERROR in line or SANITIZER_PROGRAM_HIT in line
    )


def _is_summary(line):
    return line.startswith(SANITIZER_PREFIX) and SANITIZER_ERROR_SUMMARY in line


def _is_ignored_diagnostic(block):
    text = "\n".join(block)
    first_line = block[0]
    if SANITIZER_CUDA_API_ERROR in first_line and CUDA_VERSION_ERROR_RE.search(text):
        return True
    return "CUDA_ERROR_INVALID_VALUE" in first_line and "cuGetProcAddress_v2" in text


def _diagnostic_end(lines, start):
    """Return the end of one prefixed sanitizer diagnostic block."""
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if _is_diagnostic_header(line) or _is_summary(line):
            break
        if not line.startswith(SANITIZER_PREFIX):
            break
        end += 1
    return end


def analyze_sanitizer_output(output, returncode, sanitizer_error_exitcode):
    """Remove complete known-noise blocks and classify the sanitizer exit status."""
    if returncode != sanitizer_error_exitcode:
        return SanitizerAnalysis(output, False)

    lines = output.splitlines()
    kept_lines = []
    summaries = []
    ignored_any = False
    actionable_error = False
    index = 0

    while index < len(lines):
        line = lines[index]
        if _is_diagnostic_header(line):
            end = _diagnostic_end(lines, index)
            block = lines[index:end]
            if _is_ignored_diagnostic(block):
                ignored_any = True
            else:
                actionable_error = True
                kept_lines.extend(block)
            index = end
            continue
        if _is_summary(line):
            summaries.append(line)
        else:
            kept_lines.append(line)
        index += 1

    only_ignored = ignored_any and not actionable_error
    if actionable_error:
        kept_lines.extend(summaries)
    return SanitizerAnalysis("\n".join(kept_lines), only_ignored)
