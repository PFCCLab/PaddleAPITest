from __future__ import annotations

import csv
import json
import pathlib
import shutil
import sqlite3
import subprocess
import sys
from typing import Any

from .spec import CompareSuite


def write_json(path: pathlib.Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def tail_text(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-limit:]
    return value[-limit:]


def summarize_kernel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["kernel_name"]), []).append(int(row.get("duration_ns") or 0))
    summary = []
    for name, durations in sorted(grouped.items(), key=lambda item: (-sum(item[1]), item[0])):
        total = sum(durations)
        count = len(durations)
        summary.append(
            {
                "kernel_name": name,
                "count": count,
                "total_time_ns": total,
                "mean_time_ns": total / count if count else 0,
                "min_time_ns": min(durations) if durations else 0,
                "max_time_ns": max(durations) if durations else 0,
            }
        )
    return summary


def string_id_map(conn: sqlite3.Connection) -> dict[int, str]:
    tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
    if "StringIds" not in tables:
        return {}
    columns = [row[1] for row in conn.execute("pragma table_info(StringIds)")]
    id_col = "id" if "id" in columns else columns[0]
    value_col = "value" if "value" in columns else columns[-1]
    result = {}
    for key, value in conn.execute(f"select {id_col}, {value_col} from StringIds"):
        try:
            result[int(key)] = str(value)
        except Exception:
            continue
    return result


def parse_nsys_sqlite(sqlite_path: pathlib.Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = sorted(
            row[0] for row in conn.execute("select name from sqlite_master where type='table'")
        )
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            return {"status": "parsed_no_kernel_table", "tables": tables, "kernels": []}

        id_to_string = string_id_map(conn)
        columns = [row[1] for row in conn.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
        name_columns = ["demangledName", "shortName", "name", "mangledName"]
        direct_name_col = next((name for name in name_columns if name in columns), None)
        id_name_col = next(
            (
                name
                for name in ["demangledName", "shortName", "name", "mangledName"]
                if name in columns
            ),
            None,
        )
        start_col = "start" if "start" in columns else None
        end_col = "end" if "end" in columns else None
        rows = []
        for row in conn.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL"):
            if direct_name_col and isinstance(row[direct_name_col], str):
                kernel_name = row[direct_name_col]
            elif id_name_col and row[id_name_col] is not None:
                name_id = row[id_name_col]
                try:
                    kernel_name = id_to_string.get(int(name_id), str(name_id))
                except (TypeError, ValueError):
                    kernel_name = str(name_id)
            else:
                kernel_name = "unknown"
            duration = int(row[end_col] - row[start_col]) if start_col and end_col else 0
            rows.append({"kernel_name": kernel_name, "duration_ns": duration})
        return {"status": "parsed", "tables": tables, "kernels": summarize_kernel_rows(rows)}
    finally:
        conn.close()


def profile_worker_command(
    worker: pathlib.Path, case_line: str, implementation: str, repeat: int
) -> list[str]:
    return [
        sys.executable,
        str(worker),
        "--case",
        case_line,
        "--implementation",
        implementation,
        "--repeat",
        str(repeat),
    ]


def write_kernel_summary_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "implementation",
        "kernel_name",
        "count",
        "total_time_ns",
        "mean_time_ns",
        "min_time_ns",
        "max_time_ns",
        "source_sqlite",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def run_kernel_profiles(
    out_dir: pathlib.Path,
    suite: CompareSuite,
    case_line: str,
    repo_root: pathlib.Path,
    implementations: list[str],
    repeat: int,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    profile_dir = out_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    worker = repo_root / "tools" / "operator_compare_profile_worker.py"
    nsys = shutil.which("nsys")
    status_path = profile_dir / "profile_status.json"
    if nsys is None:
        example_implementation = implementations[0] if implementations else suite.standard_id
        data = {
            "status": "skipped",
            "reason": "nsys not found in PATH",
            "manual_command_template": " ".join(
                profile_worker_command(worker, case_line, example_implementation, repeat)
            ),
        }
        write_json(status_path, data)
        return data

    all_kernel_rows: list[dict[str, Any]] = []
    runs = []
    for implementation in implementations:
        base = profile_dir / f"nsys_{implementation.replace('|', '_')}"
        rep_path = pathlib.Path(f"{base}.nsys-rep")
        sqlite_path = pathlib.Path(f"{base}.sqlite")
        worker_cmd = profile_worker_command(worker, case_line, implementation, repeat)
        profile_cmd = [
            nsys,
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,cublas,cudnn",
            "--sample=none",
            "--output",
            str(base),
            *worker_cmd,
        ]
        run_info: dict[str, Any] = {
            "implementation": implementation,
            "profile_command": profile_cmd,
        }
        try:
            completed = subprocess.run(
                profile_cmd,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as err:
            run_info.update(
                {
                    "status": "profile_timeout",
                    "timeout_seconds": timeout_seconds,
                    "stdout": tail_text(err.stdout),
                    "stderr": tail_text(err.stderr),
                }
            )
            runs.append(run_info)
            continue
        run_info.update(
            {
                "profile_returncode": completed.returncode,
                "stdout": tail_text(completed.stdout),
                "stderr": tail_text(completed.stderr),
            }
        )
        if completed.returncode != 0:
            run_info["status"] = "profile_failed"
            runs.append(run_info)
            continue
        export_cmd = [
            nsys,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(rep_path),
        ]
        try:
            exported = subprocess.run(
                export_cmd,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as err:
            run_info.update(
                {
                    "status": "export_timeout",
                    "timeout_seconds": timeout_seconds,
                    "export_stdout": tail_text(err.stdout),
                    "export_stderr": tail_text(err.stderr),
                }
            )
            runs.append(run_info)
            continue
        run_info.update(
            {
                "export_command": export_cmd,
                "export_returncode": exported.returncode,
                "export_stdout": tail_text(exported.stdout),
                "export_stderr": tail_text(exported.stderr),
            }
        )
        if exported.returncode != 0 or not sqlite_path.exists():
            run_info["status"] = "export_failed"
            runs.append(run_info)
            continue
        parsed = parse_nsys_sqlite(sqlite_path)
        run_info.update(
            {
                "status": parsed.get("status"),
                "sqlite": str(sqlite_path),
                "nsys_rep": str(rep_path),
                "tables": parsed.get("tables", []),
            }
        )
        for row in parsed.get("kernels", []):
            all_kernel_rows.append(
                {**row, "implementation": implementation, "source_sqlite": str(sqlite_path)}
            )
        runs.append(run_info)

    successful_statuses = {"parsed", "parsed_no_kernel_table"}
    successful_runs = [run for run in runs if run.get("status") in successful_statuses]
    if not runs:
        status = "skipped"
    elif len(successful_runs) == len(runs):
        status = "completed"
    elif successful_runs:
        status = "partial"
    else:
        status = "failed"

    write_kernel_summary_csv(profile_dir / "kernel_summary.csv", all_kernel_rows)
    data = {"status": status, "runs": runs, "kernel_summary": all_kernel_rows}
    write_json(profile_dir / "kernel_summary.json", all_kernel_rows)
    write_json(status_path, data)
    return data
