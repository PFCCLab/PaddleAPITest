"""主进程的增量聚合、最终收尾和完整性检查。"""

from __future__ import annotations

import csv
import io
import json
import os

from . import log_runtime as runtime
from .log_schema import (
    FINAL_RESULT_PRIORITY,
    LOG_PREFIXES,
    RESULT_LOG_PREFIXES,
    STABLE_HEADER,
    TOL_HEADER,
)

_result_offsets = {}
_inorder_offsets = {}
_inorder_completed_offsets = {}
_INORDER_STATE_FILENAME = ".inorder_state.json"
_INORDER_BUILD_FILENAME = ".log_inorder.build"
_COPY_BUFFER_SIZE = 4 * 1024 * 1024
_CSV_SPECS = (
    ("tol", TOL_HEADER, ("API", "dtype", "config", "mode")),
    ("stable", STABLE_HEADER, ("API", "dtype", "config", "comp")),
)


def _reset_aggregation_state():
    _result_offsets.clear()
    _inorder_offsets.clear()
    _inorder_completed_offsets.clear()
    _load_inorder_state()


def _inorder_state_path():
    return runtime.TEST_LOG_PATH / _INORDER_STATE_FILENAME


def _inorder_build_path():
    return runtime.TEST_LOG_PATH / _INORDER_BUILD_FILENAME


def _write_inorder_state():
    """持久化 source offset，保证删除 .tmp 前可以恢复聚合进度。"""
    # state 必须在 source 清理前落盘，主进程重启才能避免重复或丢失。
    state_path = _inorder_state_path()
    temp_path = state_path.with_name(f".{state_path.name}.tmp")
    state = {
        "build_offset": _inorder_build_path().stat().st_size,
        "sources": {file_path.name: offset for file_path, offset in _inorder_offsets.items()},
    }
    try:
        with temp_path.open("w", encoding="utf-8") as state_file:
            json.dump(state, state_file, sort_keys=True)
            state_file.write("\n")
            runtime.sync_file(state_file)
        os.replace(temp_path, state_path)
        runtime.sync_directory(state_path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _copy_file_prefix(source_path, target_path, size):
    """复制指定长度并持久化，供 build 恢复和最终发布共用。"""
    remaining = size
    with source_path.open("rb") as source, target_path.open("wb") as target:
        while remaining:
            chunk = source.read(min(_COPY_BUFFER_SIZE, remaining))
            if not chunk:
                raise ValueError(f"{source_path} ended before {size} bytes")
            target.write(chunk)
            remaining -= len(chunk)
        runtime.sync_file(target)
    runtime.sync_directory(target_path.parent)


def _prepare_inorder_build(expected_size):
    """将 build 恢复到 state 指定的安全长度。"""
    build_path = _inorder_build_path()
    if not build_path.exists():
        out_path = runtime.TEST_LOG_PATH / "log_inorder.log"
        if expected_size:
            if not out_path.exists():
                raise FileNotFoundError(f"missing recoverable {build_path}")
            _copy_file_prefix(out_path, build_path, expected_size)
        else:
            with build_path.open("wb") as target:
                runtime.sync_file(target)
            runtime.sync_directory(build_path.parent)

    actual_size = build_path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(f"inorder build is shorter than state: {actual_size} < {expected_size}")
    if actual_size > expected_size:
        # build 写入后、state 更新前退出时，超出安全点的尾部必须回退。
        with build_path.open("r+b") as build_file:
            build_file.truncate(expected_size)
            runtime.sync_file(build_file)
    return build_path


def _load_inorder_state():
    """加载上次聚合提交点，并截断未提交的 build 尾部。"""
    state_path = _inorder_state_path()
    try:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            expected_size = int(state["build_offset"])
            for name, offset in state.get("sources", {}).items():
                _inorder_offsets[runtime.TMP_LOG_PATH / name] = int(offset)
        else:
            # 无 state 时，已发布日志长度是唯一可信的 build 基线。
            out_path = runtime.TEST_LOG_PATH / "log_inorder.log"
            expected_size = out_path.stat().st_size if out_path.exists() else 0
        _prepare_inorder_build(expected_size)
        if not state_path.exists():
            # 首次追加前建立基线，崩溃后才能按旧 offset 回退。
            _write_inorder_state()
    except (OSError, TypeError, ValueError) as err:
        raise RuntimeError(f"failed to recover inorder aggregation state: {err}") from err


def _read_pending_result_bytes(file_path):
    """读取上次聚合后的新增完整行。"""
    offset = _result_offsets.get(file_path, 0)
    file_size = file_path.stat().st_size
    if file_size < offset:
        offset = 0
    if file_size == offset:
        return b"", offset, offset
    with file_path.open("rb") as f:
        f.seek(offset)
        data = f.read(file_size - offset)
    if data and not data.endswith(b"\n"):
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            return b"", offset, offset
        data = data[: last_newline + 1]
        file_size = offset + last_newline + 1
    return data, offset, file_size


def _save_result_offsets(pending_offsets):
    """保存读取偏移；source 清理由整轮聚合成功后统一执行。"""
    _result_offsets.update(pending_offsets)


def _copy_inorder_range(file_path, out_f, start_offset, end_offset):
    """复制已完成字节区间，并限制每个物理行最多 200000 字节。"""
    if end_offset > start_offset:
        with file_path.open("rb") as in_f:
            in_f.seek(start_offset)
            remaining = end_offset - start_offset
            while remaining:
                line = in_f.readline(min(200001, remaining))
                if not line:
                    break
                remaining -= len(line)
                filtered = b"gpu_resources.cc:" in line and b"Please NOTE: device:" in line
                too_long = len(line) > 200000
                if not filtered:
                    out_f.write(line[:200000] + (b"\n" if too_long else b""))
                while remaining and not line.endswith(b"\n"):
                    line = in_f.readline(min(4 * 1024 * 1024, remaining))
                    if not line:
                        break
                    remaining -= len(line)


def mark_inorder_case_complete(pid, completed_offset):
    """记录 worker 最后一个已完成 case 的安全读取上界。"""
    if completed_offset is None:
        return
    file_path = runtime.TMP_LOG_PATH / f"log_{pid}.log"
    _inorder_completed_offsets[file_path] = max(
        completed_offset, _inorder_completed_offsets.get(file_path, 0)
    )


def _flush_completed_inorder_logs():
    """按 worker 增量复制完整 case，并持久化 source offset。"""
    if not _inorder_completed_offsets:
        return True
    out_file = _inorder_build_path()
    try:
        with out_file.open("ab") as out_f:
            for file_path, end_offset in list(_inorder_completed_offsets.items()):
                start_offset = _inorder_offsets.get(file_path, 0)
                if end_offset < start_offset:
                    start_offset = 0
                end_offset = min(end_offset, file_path.stat().st_size)
                _copy_inorder_range(file_path, out_f, start_offset, end_offset)
                runtime.sync_file(out_f)
                _inorder_offsets[file_path] = end_offset
                _inorder_completed_offsets.pop(file_path, None)
        _write_inorder_state()
        return True
    except Exception as err:
        print(f"Error flushing case blocks to {out_file}: {err}", flush=True)
        return False


def _aggregate_text_logs(log_files, out_file, cleanup):
    """合并、去重并排序 worker 文本日志。"""
    if not log_files and not (cleanup and out_file.exists()):
        return True
    all_lines = set()
    pending_offsets = {}
    try:
        for file_path in log_files:
            data, _, end_offset = _read_pending_result_bytes(file_path)
            pending_offsets[file_path] = end_offset
            all_lines.update(line.strip() for line in data.decode().splitlines() if line.strip())
        if cleanup and out_file.exists():
            all_lines.update(runtime.read_log_lines(out_file))
        if cleanup:
            runtime.write_sorted_lines(out_file, all_lines)
        elif all_lines:
            with out_file.open("a") as f:
                f.writelines(f"{line}\n" for line in sorted(all_lines))
                runtime.sync_file(f)
        _save_result_offsets(pending_offsets)
    except Exception as err:
        print(f"Error writing to {out_file}: {err}", flush=True)
        return False
    return True


def _aggregate_result_logs(cleanup, tmp_exists):
    """聚合所有主结果日志类型。"""
    results = []
    for prefix in LOG_PREFIXES.values():
        log_files = list(runtime.TMP_LOG_PATH.glob(f"{prefix}_*.txt")) if tmp_exists else []
        out_file = runtime.TEST_LOG_PATH / f"{prefix}.txt"
        results.append(_aggregate_text_logs(log_files, out_file, cleanup))
    return all(results)


def _aggregate_inorder_logs(cleanup, tmp_exists):
    """恢复完整 case，发布 build 后再清理 worker 临时日志。"""
    if not tmp_exists:
        return _publish_inorder_build() if cleanup else True
    if not _flush_completed_inorder_logs():
        return False
    log_files = sorted(runtime.TMP_LOG_PATH.glob("log_*.log"))
    if not log_files:
        return _publish_inorder_build() if cleanup else True
    if not cleanup:
        return True

    # 主进程重启时，完整 end 之后的内容可以恢复；未闭合尾部不进入 build。
    for file_path in log_files:
        start_offset = _inorder_offsets.get(file_path, 0)
        end_offset = _find_last_complete_case_end(file_path, start_offset)
        if end_offset > start_offset:
            _inorder_completed_offsets[file_path] = end_offset
    if not _flush_completed_inorder_logs():
        return False
    return _publish_inorder_build()


def _find_last_complete_case_end(file_path, start_offset):
    """扫描 source 尾部，只把完整结束标记之前的字节视为可恢复。"""
    # 正常增量路径不扫描历史文件，避免每批次退化为全量解析。
    last_complete = start_offset
    with file_path.open("rb") as source:
        source.seek(start_offset)
        while line := source.readline():
            if line.startswith(b"<<< CASE "):
                last_complete = source.tell()
    return last_complete


def _publish_inorder_build():
    """将已 fsync 的 build 原子发布为最终日志。"""
    # build 保留到 source 清理成功，cleanup 重试始终复用同一份内容。
    build_path = _inorder_build_path()
    out_path = runtime.TEST_LOG_PATH / "log_inorder.log"
    publish_path = out_path.with_name(f".{out_path.name}.publish.tmp")
    try:
        # 发布副本独立于 build，原子替换后仍可继续 cleanup 重试。
        _copy_file_prefix(build_path, publish_path, build_path.stat().st_size)
        os.replace(publish_path, out_path)
        runtime.sync_directory(out_path.parent)
        return True
    except Exception as err:
        print(f"Error publishing {build_path} to {out_path}: {err}", flush=True)
        return False
    finally:
        publish_path.unlink(missing_ok=True)


def _cleanup_inorder_sources():
    """最终日志发布成功后删除 worker source 和聚合状态。"""
    # source 删除是最后一步；之前任意失败都必须保留它用于恢复。
    for file_path in sorted(runtime.TMP_LOG_PATH.glob("log_*.log")):
        try:
            file_path.unlink(missing_ok=True)
            _inorder_offsets.pop(file_path, None)
        except FileNotFoundError:
            _inorder_offsets.pop(file_path, None)
        except Exception as err:
            print(f"Error cleaning {file_path}: {err}", flush=True)
            return False
    _inorder_completed_offsets.clear()
    _inorder_state_path().unlink(missing_ok=True)
    _inorder_build_path().unlink(missing_ok=True)
    runtime.sync_directory(runtime.TEST_LOG_PATH)
    _inorder_offsets.clear()
    return True


def _aggregate_csv_logs(log_files, out_file, header, cleanup):
    """将 worker CSV 分片中的完整行聚合到一个文件。"""
    if not log_files:
        return True

    pending_offsets = {}
    pending_rows = []
    try:
        for file_path in log_files:
            data, start_offset, end_offset = _read_pending_result_bytes(file_path)
            pending_offsets[file_path] = end_offset
            if not data:
                continue
            reader = csv.reader(io.StringIO(data.decode()))
            if start_offset == 0:
                shard_header = next(reader, None)
                if shard_header != header:
                    raise ValueError(f"unexpected CSV header in {file_path}: {shard_header}")
            for row in reader:
                if row and len(row) != len(header):
                    raise ValueError(
                        f"unexpected CSV column count in {file_path}: "
                        f"expected {len(header)}, got {len(row)}"
                    )
                if row:
                    pending_rows.append(row)

        if cleanup:
            # cleanup 可能重复执行，已有 CSV 行必须幂等保留而不是再次追加。
            existing_rows = []
            if out_file.exists():
                with out_file.open(newline="") as source:
                    reader = csv.reader(source)
                    existing_header = next(reader, None)
                    if existing_header not in (None, header):
                        raise ValueError(f"unexpected CSV header in {out_file}: {existing_header}")
                    existing_rows = [row for row in reader if row]
            rows = list(dict.fromkeys(tuple(row) for row in existing_rows + pending_rows))
            temp_file = out_file.with_name(f".{out_file.name}.aggregate.tmp")
            with temp_file.open("w", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(header)
                writer.writerows(rows)
                runtime.sync_file(target)
            os.replace(temp_file, out_file)
            runtime.sync_directory(out_file.parent)
            temp_file.unlink(missing_ok=True)
        elif pending_rows:
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer)
            if not out_file.exists() or out_file.stat().st_size == 0:
                writer.writerow(header)
            writer.writerows(pending_rows)
            with out_file.open("a", newline="") as out_f:
                out_f.write(buffer.getvalue())
                runtime.sync_file(out_f)
        _save_result_offsets(pending_offsets)
    except Exception as err:
        print(f"Error aggregating CSV into {out_file}: {err}", flush=True)
        return False
    return True


def _sort_csv(file_path, columns):
    """按稳定的报告字段排序聚合 CSV。"""
    if not file_path.exists():
        return True
    temp_file = file_path.with_name(f".{file_path.name}.sort.tmp")
    try:
        with file_path.open(newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames:
                raise ValueError("missing CSV header")
            missing = [column for column in columns if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"missing CSV columns: {', '.join(missing)}")
            rows = list(reader)
        rows.sort(key=lambda row: tuple(row[column] for column in columns))
        with temp_file.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_file, file_path)
        return True
    except Exception as err:
        print(f"Error arranging {file_path}: {err}", flush=True)
        return False
    finally:
        temp_file.unlink(missing_ok=True)


def _count_result_logs():
    """统计结果分类并写出未归类配置列表。"""
    log_counts = {}
    errors = []
    checkpoint_file = runtime.TEST_LOG_PATH / "checkpoint.txt"
    api_configs = set()
    try:
        api_configs = runtime.read_log_lines(checkpoint_file)
        if checkpoint_file.exists():
            log_counts["checkpoint"] = len(api_configs)
    except Exception as err:
        print(f"Error reading {checkpoint_file}: {err}", flush=True)
        errors.append(f"read {checkpoint_file}: {err}")

    for log_type, prefix in RESULT_LOG_PREFIXES.items():
        log_file = runtime.TEST_LOG_PATH / f"{prefix}.txt"
        if not log_file.exists():
            continue
        try:
            lines = runtime.read_log_lines(log_file)
            api_configs -= lines
            log_counts[log_type] = len(lines)
        except Exception as err:
            print(f"Error reading {log_file}: {err}", flush=True)
            errors.append(f"read {log_file}: {err}")

    unclassified_file = runtime.TEST_LOG_PATH / "api_config_unclassified.txt"
    legacy_unclassified_file = runtime.TEST_LOG_PATH / "api_config_incomplete.txt"
    if not errors:
        try:
            runtime.write_sorted_lines(unclassified_file, api_configs)
            legacy_unclassified_file.unlink(missing_ok=True)
            if api_configs:
                log_counts["unclassified"] = len(api_configs)
        except Exception as err:
            print(f"Error writing to {unclassified_file}: {err}", flush=True)
            errors.append(f"write {unclassified_file}: {err}")
    if errors:
        log_counts["_aggregation_errors"] = errors
    return log_counts


def _aggregate_comp_logs(cleanup, tmp_exists):
    """分别聚合每个 comp 维度的结果日志。"""
    comp_tmp_dir = runtime.TMP_LOG_PATH / "comp" if tmp_exists else None
    comp_out_dir = runtime.TEST_LOG_PATH / "comp"
    has_comp_tmp = comp_tmp_dir is not None and comp_tmp_dir.exists()
    has_comp = has_comp_tmp or comp_out_dir.exists()
    if not has_comp_tmp:
        return has_comp, True

    results = []
    for dim_dir in sorted(comp_tmp_dir.iterdir()):
        if not dim_dir.is_dir():
            continue
        out_dim_dir = comp_out_dir / dim_dir.name
        out_dim_dir.mkdir(parents=True, exist_ok=True)
        for prefix in LOG_PREFIXES.values():
            log_files = list(dim_dir.glob(f"{prefix}_*.txt"))
            results.append(_aggregate_text_logs(log_files, out_dim_dir / f"{prefix}.txt", cleanup))
    return has_comp, all(results)


def _cleanup_result_sources():
    """所有派生结果发布成功后，统一清除 worker 结果 source。"""
    # 文本、CSV、comp 输出全部成功后，才允许清理任何结果 source。
    source_files = []
    for prefix in LOG_PREFIXES.values():
        source_files.extend(runtime.TMP_LOG_PATH.glob(f"{prefix}_*.txt"))
    source_files.extend(runtime.TMP_LOG_PATH.glob("tol_*.csv"))
    source_files.extend(runtime.TMP_LOG_PATH.glob("stable_*.csv"))
    source_files.extend(runtime.TMP_LOG_PATH.glob("comp/**/*.txt"))
    source_files.extend(runtime.TMP_LOG_PATH.glob("comp/**/*.csv"))
    try:
        for file_path in source_files:
            file_path.unlink(missing_ok=True)
        for directory in (
            sorted(
                (path for path in (runtime.TMP_LOG_PATH / "comp").rglob("*") if path.is_dir()),
                reverse=True,
            )
            if (runtime.TMP_LOG_PATH / "comp").exists()
            else ()
        ):
            if not any(directory.iterdir()):
                directory.rmdir()
        for file_path in source_files:
            _result_offsets.pop(file_path, None)
        return True
    except Exception as err:
        print(f"Error cleaning result sources: {err}", flush=True)
        return False


def _find_duplicate_classifications(log_dir):
    """查找同一目录中被多个结果类型分类的配置。"""
    config_to_types: dict[str, list[str]] = {}
    for log_type, prefix in RESULT_LOG_PREFIXES.items():
        log_file = log_dir / f"{prefix}.txt"
        if not log_file.exists():
            continue
        with log_file.open("r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    config_to_types.setdefault(line, []).append(log_type)
    return {config: types for config, types in config_to_types.items() if len(types) > 1}


def _sync_comp_main_summary():
    """将各 comp 维度分类收敛为互斥的主结果摘要。"""
    comp_out_dir = runtime.TEST_LOG_PATH / "comp"
    if not comp_out_dir.exists():
        return True

    try:
        main_lines_by_type = {
            log_type: runtime.read_log_lines(runtime.TEST_LOG_PATH / f"{prefix}.txt")
            for log_type, prefix in RESULT_LOG_PREFIXES.items()
        }
        for dim_dir in sorted(comp_out_dir.iterdir()):
            if not dim_dir.is_dir():
                continue
            for log_type, prefix in RESULT_LOG_PREFIXES.items():
                main_lines_by_type[log_type].update(
                    runtime.read_log_lines(dim_dir / f"{prefix}.txt")
                )
    except Exception as err:
        print(f"Error reading comp summary logs: {err}", flush=True)
        return False

    selected_configs = set()
    for log_type in FINAL_RESULT_PRIORITY:
        lines = main_lines_by_type[log_type]
        lines.difference_update(selected_configs)
        selected_configs.update(lines)

    try:
        for log_type, lines in main_lines_by_type.items():
            log_file = runtime.TEST_LOG_PATH / f"{LOG_PREFIXES[log_type]}.txt"
            runtime.write_sorted_lines(log_file, lines)
        return True
    except Exception as err:
        print(f"Error writing comp summary logs: {err}", flush=True)
        return False


def _check_log_integrity(log_counts):
    """检查重复分类并将诊断信息附加到统计结果。"""
    comp_out_dir = runtime.TEST_LOG_PATH / "comp"
    try:
        duplicates = _find_duplicate_classifications(runtime.TEST_LOG_PATH)
        if duplicates:
            log_counts.setdefault("_integrity_errors", []).append(
                {"scope": "main log directory", "duplicates": duplicates}
            )
        for dim_dir in sorted(comp_out_dir.iterdir()) if comp_out_dir.exists() else []:
            if not dim_dir.is_dir():
                continue
            duplicates = _find_duplicate_classifications(dim_dir)
            if duplicates:
                log_counts.setdefault("_comp_integrity_errors", []).append(
                    {"scope": f"comp/{dim_dir.name}", "duplicates": duplicates}
                )
    except Exception as err:
        print(f"Error checking log integrity: {err}", flush=True)
        log_counts.setdefault("_aggregation_errors", []).append(f"integrity check: {err}")


def _remove_empty_tmp_dir():
    if runtime.TMP_LOG_PATH.exists() and not any(runtime.TMP_LOG_PATH.iterdir()):
        runtime.TMP_LOG_PATH.rmdir()


def aggregate_logs():
    """增量聚合仍在运行的 worker 日志。"""
    return _aggregate_logs(final=False, cleanup=False)


def recover_logs():
    """聚合并清理上次中断遗留的 worker 分片。"""
    return _aggregate_logs(final=False, cleanup=True)


def finalize_logs():
    """完成最终聚合、排序、统计和完整性检查。"""
    return _aggregate_logs(final=True, cleanup=True)


def _aggregate_logs(*, final, cleanup):
    """聚合 worker 日志，并按需完成统计和完整性检查。"""
    cleanup_tmp = final or cleanup
    tmp_exists = runtime.TMP_LOG_PATH.exists()
    if not tmp_exists and not cleanup_tmp:
        runtime.TMP_LOG_PATH.mkdir(exist_ok=True)
        return True

    results = [_aggregate_result_logs(cleanup_tmp, tmp_exists)]
    inorder_result = (
        _aggregate_inorder_logs(cleanup_tmp, tmp_exists)
        if cleanup_tmp
        else _flush_completed_inorder_logs()
    )
    results.append(inorder_result)
    for name, header, _ in _CSV_SPECS:
        shards = sorted(runtime.TMP_LOG_PATH.glob(f"{name}_*.csv")) if tmp_exists else []
        results.append(
            _aggregate_csv_logs(
                shards,
                runtime.TEST_LOG_PATH / f"{name}.csv",
                header,
                cleanup_tmp,
            )
        )

    has_comp = (runtime.TEST_LOG_PATH / "comp").exists()
    if cleanup_tmp:
        has_comp, comp_success = _aggregate_comp_logs(cleanup_tmp, tmp_exists)
        results.append(comp_success)
    all_success = all(results)
    if final:
        final_results = [
            _sort_csv(runtime.TEST_LOG_PATH / f"{name}.csv", columns)
            for name, _, columns in _CSV_SPECS
        ]
        if has_comp:
            final_results.append(_sync_comp_main_summary())
        all_success = all(final_results) and all_success
    if cleanup_tmp and all_success:
        # 只有所有派生输出都成功后，source 才允许被删除。
        all_success = _cleanup_result_sources() and _cleanup_inorder_sources()
    if not final:
        if cleanup_tmp and all_success:
            _remove_empty_tmp_dir()
        return all_success

    if all_success:
        _remove_empty_tmp_dir()
    log_counts = _count_result_logs()
    if not all_success:
        log_counts.setdefault("_aggregation_errors", []).append(
            "one or more worker log aggregation steps failed"
        )
    _check_log_integrity(log_counts)
    return log_counts
