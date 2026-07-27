"""复测清单管理和陈旧结果清理。"""

from __future__ import annotations

import csv
import os

from . import log_runtime as runtime
from .log_schema import LOG_PREFIXES, MAX_CSV_CONFIG_LENGTH, RESULT_LOG_PREFIXES

RETEST_PENDING_FILENAME = ".retest_pending.txt"
RETEST_TYPES_FILENAME = ".retest_types.txt"


def parse_retest_types(value):
    """解析逗号分隔的复测分类，并保持用户给定顺序。"""
    if not value:
        return ()
    valid_types = set(LOG_PREFIXES) - {"checkpoint"}
    retest_types = []
    for raw_type in value.split(","):
        log_type = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        if not log_type:
            raise ValueError("--retest contains an empty classification")
        if log_type not in valid_types:
            choices = ", ".join(sorted(valid_types))
            raise ValueError(
                f"invalid --retest classification '{raw_type.strip()}'; choose: {choices}"
            )
        if log_type not in retest_types:
            retest_types.append(log_type)
    return tuple(retest_types)


def _filter_text_file(file_path, keep_line):
    if not file_path.exists():
        return 0
    temp_file = file_path.with_name(f".{file_path.name}.retest.tmp")
    try:
        removed = 0
        kept = 0
        with file_path.open() as source, temp_file.open("w") as target:
            for line in source:
                if not keep_line(line.strip()):
                    removed += 1
                    continue
                target.write(line)
                kept += 1
        if not removed:
            return 0
        if kept:
            os.replace(temp_file, file_path)
        else:
            file_path.unlink()
        return removed
    finally:
        temp_file.unlink(missing_ok=True)


def _rewrite_text_excluding(file_path, excluded_lines):
    return _filter_text_file(file_path, lambda line: line not in excluded_lines)


def _filter_csv_file(file_path, keep_config):
    if not file_path.exists():
        return 0
    temp_file = file_path.with_name(f".{file_path.name}.retest.tmp")
    try:
        with (
            file_path.open(newline="") as source,
            temp_file.open("w", newline="") as target,
        ):
            reader = csv.DictReader(source)
            if not reader.fieldnames or "config" not in reader.fieldnames:
                raise ValueError(f"CSV file has no config column: {file_path}")
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            removed = 0
            for row in reader:
                if not keep_config(row.get("config", "")):
                    removed += 1
                    continue
                writer.writerow(row)
        if not removed:
            return 0
        os.replace(temp_file, file_path)
        return removed
    finally:
        temp_file.unlink(missing_ok=True)


def _rewrite_csv_excluding(file_path, excluded_configs):
    return _filter_csv_file(file_path, lambda config: config not in excluded_configs)


def _restore_truncated_retest_configs(configs, checkpoints):
    truncated = {config for config in configs if len(config) == MAX_CSV_CONFIG_LENGTH}
    if not truncated:
        return set(configs)

    matches_by_prefix = {config: [] for config in truncated}
    for checkpoint in checkpoints:
        if len(checkpoint) < MAX_CSV_CONFIG_LENGTH:
            continue
        prefix = checkpoint[:MAX_CSV_CONFIG_LENGTH]
        if prefix in matches_by_prefix:
            matches_by_prefix[prefix].append(checkpoint)

    restored = set(configs) - truncated
    for config, matches in matches_by_prefix.items():
        if len(matches) > 1:
            raise ValueError(
                "cannot restore truncated retest config: multiple checkpoint entries "
                f"share its {MAX_CSV_CONFIG_LENGTH}-character prefix"
            )
        restored.add(matches[0] if matches else config)
    return restored


def _resume_retest(retest_types, pending_file, types_file):
    stored_types_text = types_file.read_text().strip() if types_file.exists() else ""
    try:
        stored_types = parse_retest_types(stored_types_text)
    except ValueError:
        stored_types = ()
    if set(stored_types) != set(retest_types):
        expected_types = ",".join(retest_types)
        raise ValueError(
            f"unfinished retest uses '{stored_types_text or 'unknown'}'; "
            f"resume it before starting '{expected_types}'"
        )

    with pending_file.open() as source:
        retest_configs = {line.strip() for line in source if line.strip()}
    retest_configs -= runtime.read_log("checkpoint")
    if not retest_configs:
        finish_retest()
    return retest_configs, retest_configs


def _start_retest(retest_types, pending_file, types_file):
    raw_configs = set()
    for log_type in retest_types:
        raw_configs.update(runtime.read_log(log_type))
    retest_configs = _restore_truncated_retest_configs(raw_configs, runtime.read_log("checkpoint"))
    if retest_configs:
        runtime.write_lines_atomic(types_file, (",".join(retest_types) + "\n",))
        runtime.write_lines_atomic(
            pending_file,
            (f"{config}\n" for config in sorted(retest_configs)),
        )
    return retest_configs, raw_configs | retest_configs


def prepare_retest(retest_types):
    """读取复测分类，并清除这些配置的当前结构化结果。"""
    pending_file = runtime.TEST_LOG_PATH / RETEST_PENDING_FILENAME
    types_file = runtime.TEST_LOG_PATH / RETEST_TYPES_FILENAME
    if pending_file.exists():
        retest_configs, cleanup_configs = _resume_retest(retest_types, pending_file, types_file)
    else:
        retest_configs, cleanup_configs = _start_retest(retest_types, pending_file, types_file)
    if not retest_configs:
        return set()

    runtime.close_process_files()
    for prefix in LOG_PREFIXES.values():
        _rewrite_text_excluding(runtime.TEST_LOG_PATH / f"{prefix}.txt", cleanup_configs)

    comp_dir = runtime.TEST_LOG_PATH / "comp"
    if comp_dir.exists():
        for dimension_dir in comp_dir.iterdir():
            if not dimension_dir.is_dir():
                continue
            for prefix in LOG_PREFIXES.values():
                _rewrite_text_excluding(
                    dimension_dir / f"{prefix}.txt",
                    cleanup_configs,
                )

    _rewrite_text_excluding(runtime.TEST_LOG_PATH / "api_config_incomplete.txt", cleanup_configs)
    csv_configs = cleanup_configs | {config[:MAX_CSV_CONFIG_LENGTH] for config in cleanup_configs}
    _rewrite_csv_excluding(runtime.TEST_LOG_PATH / "tol.csv", csv_configs)
    _rewrite_csv_excluding(runtime.TEST_LOG_PATH / "stable.csv", csv_configs)
    return retest_configs


def finish_retest():
    """清除已完成复测的恢复 manifest。"""
    (runtime.TEST_LOG_PATH / RETEST_PENDING_FILENAME).unlink(missing_ok=True)
    (runtime.TEST_LOG_PATH / RETEST_TYPES_FILENAME).unlink(missing_ok=True)


def cleanup_uncheckpointed_result_logs():
    """删除中断运行中先于 checkpoint 写入的结果记录。"""
    checkpoint_file = runtime.TEST_LOG_PATH / "checkpoint.txt"
    checkpoints = runtime.read_log_lines(checkpoint_file)

    removed = 0
    result_dirs = [runtime.TEST_LOG_PATH]
    comp_dir = runtime.TEST_LOG_PATH / "comp"
    if comp_dir.exists():
        result_dirs.extend(path for path in comp_dir.iterdir() if path.is_dir())
    for result_dir in result_dirs:
        for prefix in RESULT_LOG_PREFIXES.values():
            log_file = result_dir / f"{prefix}.txt"
            removed += _filter_text_file(log_file, checkpoints.__contains__)
        if result_dir != runtime.TEST_LOG_PATH and not any(result_dir.iterdir()):
            result_dir.rmdir()
    if comp_dir.exists() and not any(comp_dir.iterdir()):
        comp_dir.rmdir()

    csv_checkpoints = checkpoints | {config[:MAX_CSV_CONFIG_LENGTH] for config in checkpoints}
    for csv_name in ("tol.csv", "stable.csv"):
        _filter_csv_file(runtime.TEST_LOG_PATH / csv_name, csv_checkpoints.__contains__)
    return removed
