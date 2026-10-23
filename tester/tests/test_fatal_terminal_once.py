"""fatal 终态只写一条，sanitizer 不把合并终态还原成 paddle_cuda。

运行：``python -m pytest tester/tests/test_fatal_terminal_once.py -q``
"""

from __future__ import annotations

import engineV4
from tester.reporting import log_worker


def _count_lines(path):
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _parent_log_dir(tmp_path, monkeypatch):
    engineV4.log_runtime.close_process_files()
    monkeypatch.setattr(engineV4.log_runtime, "RESULT_LOG_PATH", tmp_path)
    monkeypatch.setattr(engineV4.log_runtime, "RESULT_LOG_SUFFIX", "")


def _settle_session_crash(config, exitcode, output=""):
    """复现父进程对 sanitizer session crash 的补写决策。"""
    engineV4._settle_crash(config, exitcode, "session", output)


def test_parent_keeps_recorded_session_fatal(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _parent_log_dir(parent_dir, monkeypatch)
    config = "paddle.example()"
    output = f"[oom] | {config}\n"

    _settle_session_crash(config, log_worker.fatal_exit_code("oom", True), output)

    assert _count_lines(parent_dir / "api_config_oom.txt") == 0
    assert _count_lines(parent_dir / "api_config_paddle_cuda.txt") == 0


def test_parent_writes_unrecorded_session_fatal(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    _parent_log_dir(parent_dir, monkeypatch)
    config = "paddle.example()"

    _settle_session_crash(config, log_worker.fatal_exit_code("oom", False))

    written = list(parent_dir.glob("*.txt"))
    # 子进程没写过终态时，父进程仍要补上这一条。
    assert written == [parent_dir / "api_config_oom.txt"]
    assert _count_lines(parent_dir / "api_config_oom.txt") == 1


def test_sanitizer_keeps_merged_terminal_with_cuda_text():
    output = (
        "[both_error] Torch forward | paddle.example()\n"
        "[torch] Torch forward torch_error: boom\n"
        "[paddle] Paddle forward paddle_error: CUDA error: an illegal memory access was encountered\n"
    )
    code = engineV4._normalize_sanitizer_exitcode(
        output,
        returncode=86,
        sanitizer_error_exitcode=86,
    )
    # 合并终态已经落盘；正文里的 CUDA 文本不能再把它改成 paddle_cuda。
    assert code == 86


def test_sanitizer_still_maps_unrecorded_cuda():
    output = "RuntimeError: CUDA error: an illegal memory access was encountered\n"
    code = engineV4._normalize_sanitizer_exitcode(
        output,
        returncode=86,
        sanitizer_error_exitcode=86,
    )
    # 没有已落盘终态时，CUDA 文本仍按 paddle_cuda 还原。
    assert code == log_worker.fatal_exit_code("paddle_cuda", False)
