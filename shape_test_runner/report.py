#!/usr/bin/env python3
"""进度汇总、失败问题整理与复现包生成。

三个入口都只依赖 lib.py 与磁盘产物：
- build_status : 读 checkpoint / api_config_*.txt，输出 status.md/json
- collect      : 导出失败 config 对应的单 case .log，并按错误签名聚合
- build_repro  : 生成可直接执行的复现脚本与失败配置清单
"""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import time
from datetime import datetime
from pathlib import Path

from lib import (
    CAT_ORDER,
    FAIL_CATS,
    REF_CATS,
    atomic_write_json,
    case_signature,
    classify_job,
    job_log_dir,
    job_runtime_state,
    load_json,
    now_str,
    parse_case_blocks,
    pick_case_block,
)

# 参考分类（REF_CATS）不进入真实失败清单：它们要么是环境/规则导致的预期结果，
# 要么已在阈值内复核通过，混入失败统计会让真实缺陷被噪声淹没。
REF_NOTE = {
    "oom": "显存不足，通常没有单 case 日志（进程被打挂），默认不重跑",
    "skip": "按规则跳过，不是失败",
    "paddle_bitwise": "阈值内 known 差异（manual_threshold 复核通过）",
    "paddle_bitwise_knows": "known bitwise 差异，默认不重跑",
}

# 自定义算子配置形如 _run_custom_op("name", ...)，优先按它归因；
# 否则取第一个 paddle.xxx( 调用；两者都没有时返回 (unknown)，保证分组键非空。
_CUSTOM_OP = re.compile(r"_run_custom_op\(\s*[\"']([^\"']+)")
_PADDLE_API = re.compile(r"\bpaddle\.([A-Za-z_][A-Za-z0-9_.]*)\(")


def config_api(config: str) -> str:
    """从一条 API config 里提取 API/算子名，用于问题按 API 聚合。"""
    custom = _CUSTOM_OP.search(config)
    if custom:
        return f"_run_custom_op/{custom.group(1)}"
    match = _PADDLE_API.search(config)
    if match:
        return f"paddle.{match.group(1)}"
    return "(unknown)"


# 只对 int/float 加千分位；占位符（如 ? 或 -）原样返回，避免格式化异常打断整份报表。
def _fmt_int(value) -> str:
    return f"{int(value):,}" if isinstance(value, (int, float)) else str(value)


def _elapsed_seconds(started: str, finished: str | None) -> float:
    """解析 state 里的时间字符串；running 时传 None 表示算到当前。"""
    if not started:
        return 0.0
    try:
        start_ts = datetime.strptime(started, "%Y-%m-%d %H:%M:%S").timestamp()
        end_ts = (
            datetime.strptime(finished, "%Y-%m-%d %H:%M:%S").timestamp()
            if finished
            else time.time()
        )
    except (ValueError, TypeError):
        return 0.0
    # 时间字符串损坏或时钟回拨时返回 0：报表宁可显示 "-"，也不要出现负数或巨型耗时。
    return max(0.0, end_ts - start_ts)


# 耗时只保留到一个量级（秒/分/时），避免表格列宽随数值位数不断膨胀。
def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "-"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _load_run(run_id: str, run_dir: Path) -> dict:
    run = load_json(run_dir / "run.json", {})
    if not run:
        raise SystemExit(f"未找到运行记录: {run_dir}/run.json")
    # 旧版 run.json 可能缺 run_id，用调用方传入的补上，保证下游字段始终存在。
    run["run_id"] = run.get("run_id", run_id)
    return run


def build_status(run_id: str, run_dir: Path) -> dict:
    """汇总所有 job 的进度与分类计数，落盘 status.md / status.json。"""
    run = _load_run(run_id, run_dir)
    # state.json 只提供进程与心跳信息，进度本身从日志目录推导，缺失也不影响汇总。
    state = load_json(run_dir / "state.json", {})
    return build_status_view(
        run_id,
        run_dir,
        run.get("jobs", []),
        state,
        inputs=run.get("inputs", {}),
        modes=run.get("modes", []),
        supervisor_alive=bool(state.get("supervisor_pid")) and _alive(state.get("supervisor_pid")),
    )


def build_status_view(
    run_id: str,
    run_dir: Path,
    jobs: list[dict],
    state: dict,
    inputs: dict | None = None,
    modes: list | None = None,
    parts: list[dict] | None = None,
    campaign: bool = False,
    supervisor_alive: bool | None = None,
) -> dict:
    """通用进度汇总：单机 run 和 campaign 都复用这一套口径。"""
    jobs_out = []
    # totals 先逐 job 累加标量，百分比统一在循环外计算，避免重复取整导致总和偏差。
    totals = {"total": 0, "done": 0, "pass": 0, "fail": 0, "ref": 0, "skip": 0}
    for job in jobs:
        log_dir = job_log_dir(run_dir, job)
        info = classify_job(job, log_dir)
        runtime = job_runtime_state(run_dir, job, state)
        runtime.update(
            {
                "progress": round(info["progress"], 1),
                "done": info["done"],
                "total": info["total"],
                "pass": info["pass"],
                "fail": info["fail"],
                "ref": info["ref"],
                "skip": info["skip"],
                "pass_rate": round(info["pass_rate"], 1),
                "counts": info["counts"],
            }
        )
        elapsed = _elapsed_seconds(runtime.get("started_at"), runtime.get("finished_at"))
        # 只有真正跑完过 case 才估算速率与剩余时间；刚启动或已中断的 job 不给误导性的 ETA。
        # 速率按「已完成数 / 已耗时」线性外推，仅供参考，不参与任何调度或超时决策。
        if elapsed > 0 and info["done"] > 0:
            rate = info["done"] / elapsed
            runtime["elapsed_sec"] = elapsed
            runtime["rate_per_min"] = rate * 60.0
            runtime["eta_sec"] = max(0.0, (info["total"] - info["done"]) / rate) if rate else 0.0
        jobs_out.append(runtime)
        for key in totals:
            totals[key] += info[key]
    totals["progress"] = (
        round(totals["done"] / totals["total"] * 100.0, 1) if totals["total"] else 0.0
    )
    # pass 率分母与 job 级口径一致：skip 是主动跳过不计入，OOM 等参考分类要计入，否则通过率会被高估。
    judged = totals["pass"] + totals["fail"] + (totals["ref"] - totals["skip"])
    totals["pass_rate"] = round(totals["pass"] / judged * 100.0, 1) if judged else 0.0

    # 调用方（campaign）已按 part 维度判断存活时不再探测本机 supervisor，避免跨机器误判。
    if supervisor_alive is None:
        supervisor_alive = bool(state.get("supervisor_pid")) and _alive(state.get("supervisor_pid"))
    payload = {
        "run_id": run_id,
        "generated_at": now_str(),
        "campaign": campaign,
        "inputs": inputs or {},
        "modes": modes or [],
        "parts": parts or [],
        "totals": totals,
        "jobs": jobs_out,
        "supervisor_alive": supervisor_alive,
    }
    atomic_write_json(run_dir / "status.json", payload)
    (run_dir / "status.md").write_text(_render_status_md(payload), encoding="utf-8")
    return payload


def _alive(pid) -> bool:
    # 延迟导入：pid_alive 只在判断 supervisor 存活时使用，保证本模块可脱离进程探测单独渲染报表。
    from lib import pid_alive

    return pid_alive(pid)


def _render_status_md(payload: dict) -> str:
    # 单机与 campaign 共用同一渲染器，只有标题、part 表和结尾命令不同。
    # 表头与数据行的列数、列序必须严格一致，否则 Markdown 表格在平台上会错位。
    campaign = payload.get("campaign")
    title = "多机 campaign 进度" if campaign else "进度"
    alive_text = "存在运行中的 part" if campaign else "运行中"
    dead_text = "全部 part 已退出" if campaign else "已退出"
    lines = [
        f"# shape_test_runner {title}：{payload['run_id']}",
        "",
        f"- 更新时间：{payload['generated_at']}",
        f"- supervisor：{alive_text if payload['supervisor_alive'] else dead_text}",
        f"- 总体进度：{payload['totals']['done']}/{payload['totals']['total']} "
        f"({payload['totals']['progress']}%)",
        f"- pass {payload['totals']['pass']}，真实失败 {payload['totals']['fail']}，"
        f"参考分类(oom/skip/bitwise) {payload['totals']['ref']}",
        "",
    ]
    # parts 只有 campaign 才有；单机 run 不输出这一段，保持报表精简。
    if payload.get("parts"):
        lines += [
            "## 分 part 状态",
            "",
            "| part | host | 状态 | 进度 | pass | 真实失败 | 参考分类 | 最近心跳 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for part in payload["parts"]:
            lines.append(
                "| {pid} | {host} | {status} | {done}/{total} ({progress}%) | {p} | {f} | {r} | {hb} |".format(
                    pid=part["id"],
                    host=part.get("host") or "-",
                    status=part["status"],
                    done=_fmt_int(part["done"]),
                    total=_fmt_int(part["total"]),
                    progress=part["progress"],
                    p=_fmt_int(part["pass"]),
                    f=_fmt_int(part["fail"]),
                    r=_fmt_int(part["ref"]),
                    hb=part.get("heartbeat") or "-",
                )
            )
        lines.append("")
    lines += [
        "| job | 状态 | 进度 | pass | 真实失败 | 参考分类 | pass率 | 最近活动 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for job in payload["jobs"]:
        lines.append(
            "| {key} | {status} | {done}/{total} ({progress}%) | {p} | {f} | {r} | {rate}% | {last} |".format(
                key=job["key"],
                status=job["status"],
                done=_fmt_int(job["done"]),
                total=_fmt_int(job["total"]),
                progress=job["progress"],
                p=_fmt_int(job["pass"]),
                f=_fmt_int(job["fail"]),
                r=_fmt_int(job["ref"]),
                rate=job["pass_rate"],
                last=job.get("last_activity") or "-",
            )
        )
    # 明细按 CAT_ORDER 固定顺序输出，便于跨 job、跨次运行直接对比。
    # 只列非 pass 分类：pass 已在上一列展示，重复出现会让人误以为两处统计口径不同。
    detail = []
    for job in payload["jobs"]:
        items = [
            f"{cat}={_fmt_int(job['counts'][cat])}"
            for cat in CAT_ORDER
            if cat in job["counts"] and cat != "pass"
        ]
        rate = job.get("rate_per_min")
        # 既没有失败分类也没有速率信息的 job 直接跳过，避免输出大段无内容条目。
        if not items and not rate:
            continue
        line = f"- `{job['key']}`：pass={_fmt_int(job['pass'])}"
        if items:
            line += "，" + "，".join(items)
        if rate:
            line += (
                f"；已运行 {_fmt_duration(job.get('elapsed_sec', 0))}，"
                f"{rate:.1f} case/min，预计剩余 {_fmt_duration(job.get('eta_sec', 0))}"
            )
        detail.append(line)
    if detail:
        lines += ["", "## 失败分类与速率", ""] + detail
    # 结尾命令按模式区分：campaign 需要先 collect 再 debug，单机 run 直接走 repro。
    if campaign:
        lines += [
            "",
            f"> 汇总问题：`st.py collect --run-id {payload['run_id']}`；"
            f"合并复现包：`st.py debug --run-id {payload['run_id']} --card H`。",
        ]
    else:
        lines += [
            "",
            "> 失败问题与单 case 日志：运行 `python shape_test_runner/st.py collect --run-id "
            f"{payload['run_id']}`；复现包：`st.py repro --run-id {payload['run_id']}`。",
        ]
    return "\n".join(lines) + "\n"


def collect_issues(run_id: str, run_dir: Path) -> dict:
    """导出失败配置的单 case 日志，并按错误签名聚合问题。"""
    run = _load_run(run_id, run_dir)
    # 单机入口只负责读 run.json，实际聚合逻辑与 campaign 共用 collect_issues_view。
    return collect_issues_view(
        run_id,
        run_dir,
        run.get("jobs", []),
        inputs=run.get("inputs", {}),
        modes=run.get("modes", []),
    )


def collect_issues_view(
    run_id: str,
    run_dir: Path,
    jobs: list[dict],
    inputs: dict | None = None,
    modes: list | None = None,
    campaign: bool = False,
) -> dict:
    """通用问题整理：单机 run 和 campaign 共用。"""
    issues_dir = run_dir / "issues"
    cases_root = issues_dir / "cases"
    # 重新整理时先清掉旧派生文件，避免已通过的 case 残留旧日志。
    shutil.rmtree(cases_root, ignore_errors=True)
    jobs_out = []
    # 两套聚合维度互不覆盖：签名回答「是不是同一种错误」，API 回答「哪个算子最可疑」。
    # 涉及 job 用 set 去重、最后排序输出，保证同一批产物多次整理的结果稳定可比。
    signatures: dict[tuple, dict] = {}
    api_counts: dict[tuple, dict] = {}

    for job in jobs:
        log_dir = job_log_dir(run_dir, job)
        info = classify_job(job, log_dir)
        classes = info["classes"]
        # 没有任何失败分类时不做日志解析：OOM/crash 打挂进程的 job 往往根本没有可切的 case 日志。
        blocks = parse_case_blocks(log_dir) if classes else {}
        job_summary = {
            "key": job["key"],
            "log_dir": str(log_dir),
            "total": info["total"],
            "done": info["done"],
            "pass": info["pass"],
            "fail": info["fail"],
            "classes": {},
        }
        for cat in CAT_ORDER:
            configs = classes.get(cat)
            if not configs or cat == "pass":
                continue
            # 目录按 <job>/<分类> 分层，与 pipeline_errors 的形态保持一致，方便两边对照排查。
            dest = cases_root / job["key"] / cat
            rows = []
            missing = []
            for config in configs:
                found = pick_case_block(blocks.get(config, []))
                if not found:
                    missing.append(config)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                name = f"{cat}__{found['case_id']}.log"
                (dest / name).write_text(found["block"], encoding="utf-8")
                rows.append((found["case_id"], name, config, found["block"]))
                # 只有真实失败参与签名与 API 聚合；参考分类单独成清单，不进入问题排序。
                if cat in FAIL_CATS:
                    sig = case_signature(found["block"])
                    entry = signatures.setdefault(
                        (cat, sig),
                        {
                            "category": cat,
                            "signature": sig,
                            "count": 0,
                            "jobs": set(),
                            "examples": [],
                        },
                    )
                    entry["count"] += 1
                    entry["jobs"].add(job["key"])
                    if len(entry["examples"]) < 3:
                        entry["examples"].append(f"{job['key']}/{name}")
                    # 同一个 API 可能有很多 shape/参数变体，单列一张 API 维度表。
                    api = config_api(config)
                    api_entry = api_counts.setdefault(
                        (cat, api),
                        {"category": cat, "api": api, "count": 0, "jobs": set(), "examples": []},
                    )
                    api_entry["count"] += 1
                    api_entry["jobs"].add(job["key"])
                    if len(api_entry["examples"]) < 3:
                        api_entry["examples"].append(f"{job['key']}/{name}")
            # 有日志才写合并日志和索引，避免留下只有表头的空目录。
            if rows:
                (dest / f"{cat}.all.log").write_text("".join(r[3] for r in rows), encoding="utf-8")
                idx = [
                    f"# {job['key']} / {cat} 单 case 日志",
                    "",
                    "| case-id | 日志文件 | 配置（截断） |",
                    "|---|---|---|",
                ]
                for case_id, name, config, _ in rows:
                    short = config if len(config) <= 160 else config[:157] + "..."
                    short = short.replace("|", "\\|")
                    idx.append(f"| {case_id} | `{name}` | `{short}` |")
                (dest / "INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
            # 缺日志的配置必须单独落盘：静默丢弃会让 OOM/crash 看起来像「没有失败」。
            if missing:
                # OOM/crash 把进程打挂时平台不产出 case 日志，必须显式列出，不能静默缺失。
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "missing_log.txt").write_text(
                    f"# {job['key']} / {cat}：以下 {len(missing)} 条配置没有 case 日志（通常为 OOM/crash）\n"
                    + "\n".join(missing)
                    + "\n",
                    encoding="utf-8",
                )
            job_summary["classes"][cat] = {
                "count": len(configs),
                "with_log": len(rows),
                "missing_log": len(missing),
            }
        jobs_out.append(job_summary)

    # 排序键固定为 数量降序 → 分类 → 签名，保证同一输入生成的 INDEX.md 可复现。
    sig_list = []
    for entry in signatures.values():
        sig_list.append(
            {
                "category": entry["category"],
                "signature": entry["signature"],
                "count": entry["count"],
                "jobs": sorted(entry["jobs"]),
                "examples": entry["examples"],
            }
        )
    sig_list.sort(key=lambda item: (-item["count"], item["category"], item["signature"]))
    api_list = [
        {
            "category": entry["category"],
            "api": entry["api"],
            "count": entry["count"],
            "jobs": sorted(entry["jobs"]),
            "examples": entry["examples"],
        }
        for entry in api_counts.values()
    ]
    api_list.sort(key=lambda item: (-item["count"], item["category"], item["api"]))

    payload = {
        "run_id": run_id,
        "generated_at": now_str(),
        "campaign": campaign,
        "inputs": inputs or {},
        "modes": modes or [],
        "jobs": jobs_out,
        "signatures": sig_list,
        "apis": api_list,
    }
    issues_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(issues_dir / "issues.json", payload)
    (issues_dir / "INDEX.md").write_text(_render_issues_md(payload), encoding="utf-8")
    return payload


def _render_issues_md(payload: dict) -> str:
    inputs = payload.get("inputs", {})
    # 输入行同时兼容 source/path 两种字段名：老 run.json 没有 source 时退回路径展示。
    # 表格内的竖线统一转义，否则配置里的 | 会把 Markdown 列切断。
    input_line = "，".join(
        f"{shape}={item.get('source', item.get('path', ''))}（{item.get('total', '?')} 条）"
        for shape, item in inputs.items()
    )
    lines = [
        f"# 问题整理：{payload['run_id']}",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 输入：{input_line or '-'}",
        "- 单 case 日志目录：`issues/cases/<job>/<分类>/`，同分类合并日志：`<分类>.all.log`",
        "- 没有日志的配置（通常是 OOM/crash）：`issues/cases/<job>/<分类>/missing_log.txt`",
        "",
        "## 失败分类与日志覆盖",
        "",
        "| job | 分类 | 配置数 | 有日志 | 无日志 |",
        "|---|---|---|---|---|",
    ]
    for job in payload["jobs"]:
        for cat, stat in job["classes"].items():
            lines.append(
                f"| {job['key']} | {cat} | {stat['count']} | {stat['with_log']} | {stat['missing_log']} |"
            )
    lines += ["", "## 问题签名聚合（真实失败）", ""]
    if payload["signatures"]:
        lines += ["| 分类 | 数量 | 签名 | 涉及 job | 示例日志 |", "|---|---|---|---|---|"]
        for entry in payload["signatures"]:
            sig = entry["signature"].replace("|", "\\|")
            lines.append(
                f"| {entry['category']} | {entry['count']} | `{sig}` | "
                f"{', '.join(entry['jobs'])} | {', '.join(entry['examples'])} |"
            )
    else:
        lines.append("本次没有真实失败。")
    lines += ["", "## 按 API 聚合（真实失败 Top 30）", ""]
    if payload.get("apis"):
        lines += ["| 分类 | 数量 | API | 涉及 job | 示例日志 |", "|---|---|---|---|---|"]
        for entry in payload["apis"][:30]:
            api = entry["api"].replace("|", "\\|")
            lines.append(
                f"| {entry['category']} | {entry['count']} | `{api}` | "
                f"{', '.join(entry['jobs'])} | {', '.join(entry['examples'])} |"
            )
    # 参考分类说明只输出本次实际出现的分类，避免固定模板里的无关条目造成干扰。
    # 配置与签名统一截断到 160 字符并转义竖线，超长 shape 列表在表格里不可读。
    refs = [
        f"- `{cat}`：{REF_NOTE.get(cat, '参考分类')}"
        for cat in REF_CATS
        if any(cat in job["classes"] for job in payload["jobs"])
    ]
    if refs:
        lines += ["", "## 参考分类说明", ""] + refs
    # 明确指出完整 traceback 的位置：crash / CUDA error 只有 run 级日志才是完整的。
    lines += [
        "",
        "## 完整运行日志",
        "",
        "每个 job 的完整日志在 `logs/<job>/log_*.log`（含环境信息与完整 traceback），"
        "crash / CUDA error 类优先看它。",
    ]
    return "\n".join(lines) + "\n"


def _write_shell(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    # 生成后立即加可执行位，避免使用者忘记 chmod 后遇到 Permission denied。
    path.chmod(0o755)


def _export_lines(env: dict) -> str:
    # 环境变量逐个 shlex.quote，值里带空格或括号时也不会破坏 source 的解析结果。
    return "".join(
        f"export {key}={shlex.quote(str(value))}\n" for key, value in (env or {}).items()
    )


# 复现脚本不假设自己放在哪一层：从脚本目录向上找 engineV4.py。
_ROOT_SNIPPET = """ROOT="${PADDLEAPITEST_ROOT:-}"
if [ -z "${ROOT}" ]; then
  probe="${DIR}"
  while [ "${probe}" != "/" ] && [ ! -f "${probe}/engineV4.py" ]; do probe="$(dirname "${probe}")"; done
  ROOT="${probe}"
fi
[ -f "${ROOT}/engineV4.py" ] || { echo "找不到 PaddleAPITest 根目录，请设置 PADDLEAPITEST_ROOT"; exit 2; }
"""


def build_repro(run_id: str, run_dir: Path) -> dict:
    """生成复现包：真实失败 config、参考分类、批量脚本与单 case 脚本。"""
    run = _load_run(run_id, run_dir)
    repro = run_dir / "repro"
    # 复现包每次整体重建，避免上一轮已修复的配置残留。
    shutil.rmtree(repro, ignore_errors=True)
    # 三个子目录即使为空也先建好，保证复现包结构与 README/INDEX 的描述一致。
    configs_dir = repro / "configs"
    refs_dir = repro / "refs"
    logs_dir = repro / "logs"
    for path in (configs_dir, refs_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    # 每个 job 独立产出配置、参数、环境变量与脚本，job 之间不共享任何状态。
    rows = []
    for job in run.get("jobs", []):
        log_dir = job_log_dir(run_dir, job)
        info = classify_job(job, log_dir)
        classes = info["classes"]
        # 失败清单一律按 FAIL_CATS 顺序去重，与流水线 debug bundle 的清单口径保持一致。
        fail_lines = []
        for cat in FAIL_CATS:
            for line in classes.get(cat, []):
                if line not in fail_lines:
                    fail_lines.append(line)
        # 参考分类按 job 单独成文件，不与真实失败混在同一份清单里，避免误跑。
        ref_files = []
        for cat in REF_CATS:
            lines = classes.get(cat, [])
            if not lines:
                continue
            path = refs_dir / f"{job['key']}_{cat}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            ref_files.append((cat, path.name, len(lines)))
        # engine 参数和环境变量各存一份，批量脚本与单 case 脚本共用，保证复现口径一致。
        (configs_dir / f"{job['key']}_args.txt").write_text(
            "".join(arg + "\n" for arg in job.get("engine_args", [])), encoding="utf-8"
        )
        (configs_dir / f"{job['key']}_env.sh").write_text(
            _export_lines(job.get("env")), encoding="utf-8"
        )
        cfg_path = configs_dir / f"{job['key']}_failures.txt"
        # 没有真实失败的 job 不生成批量脚本，避免产出永远 skip 的空脚本。
        if fail_lines:
            cfg_path.write_text("\n".join(fail_lines) + "\n", encoding="utf-8")
            _write_shell(repro / f"run_{job['key']}.sh", _batch_script(run, job))
        rows.append(
            {
                "job": job["key"],
                "failures": len(fail_lines),
                "config": cfg_path.name if fail_lines else "",
                "refs": ref_files,
                "log_dir": str(log_dir),
            }
        )

    # 单 case 脚本始终生成：即使本次没有失败，也便于后续手工指定行号验证。
    _write_shell(repro / "run_case.sh", _single_case_script(run))
    (repro / "README.md").write_text(_render_repro_readme(run, rows), encoding="utf-8")
    (repro / "INDEX.md").write_text(_render_repro_index(rows), encoding="utf-8")
    return {"run_id": run_id, "generated_at": now_str(), "jobs": rows}


def _batch_script(run: dict, job: dict) -> str:
    """批量复现脚本：参数与环境变量与本次运行完全一致。"""
    # 注释里写明来源 yaml/输入，复现时能快速判断这个 job 是怎么组合出来的。
    yaml_note = "，".join(job.get("merged_from", [])) or job["key"]
    return f"""#!/bin/bash
# 复现 job {job["key"]}（来源: {yaml_note}）
# 用法: bash run_{job["key"]}.sh
set -uo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
{_ROOT_SNIPPET}CFG="${{DIR}}/configs/{job["key"]}_failures.txt"
LOG="${{DIR}}/logs/{job["key"]}"
if [ ! -s "${{CFG}}" ]; then echo "[skip] {job["key"]}: 无真实失败配置"; exit 0; fi
cd "${{ROOT}}" || exit 1
mkdir -p "${{LOG}}"
mapfile -t ARGS < "${{DIR}}/configs/{job["key"]}_args.txt"
source "${{DIR}}/configs/{job["key"]}_env.sh"
echo "[run] {job["key"]} | $(wc -l < "${{CFG}}") configs | log=${{LOG}}"
${{PYTHON:-python}} engineV4.py --api_config_file="${{CFG}}" --log_dir="${{LOG}}" "${{ARGS[@]}}"
"""


def _single_case_script(run: dict) -> str:
    """单 case 复现：按失败清单行号执行，便于最小化定位。"""
    # 行号基于 <job>_failures.txt：sed 取行后以 --api_config 传入单条配置，其余参数复用 args 文件。
    return f"""#!/bin/bash
# 单 case 复现。用法: bash run_case.sh <job> <行号>
# 例: bash run_case.sh 1M_accuracy 3
set -uo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
{_ROOT_SNIPPET}JOB="${{1:-}}"; N="${{2:-}}"
if [ -z "${{JOB}}" ] || [ -z "${{N}}" ]; then echo "用法: bash $0 <job> <行号>"; exit 2; fi
CFG="${{DIR}}/configs/${{JOB}}_failures.txt"
ARGS_FILE="${{DIR}}/configs/${{JOB}}_args.txt"
if [ ! -s "${{CFG}}" ] || [ ! -f "${{ARGS_FILE}}" ]; then echo "找不到 ${{JOB}} 的失败清单或参数文件"; exit 2; fi
LINE=$(sed -n "${{N}}p" "${{CFG}}")
if [ -z "${{LINE}}" ]; then echo "行号 ${{N}} 超出 ${{CFG}} 范围"; exit 2; fi
cd "${{ROOT}}" || exit 1
LOG="${{DIR}}/logs/${{JOB}}_case_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${{LOG}}"
mapfile -t ARGS < "${{ARGS_FILE}}"
source "${{DIR}}/configs/${{JOB}}_env.sh"
echo "[run] ${{JOB}} 第 ${{N}} 行 -> ${{LOG}}"
echo "[case] ${{LINE}}"
${{PYTHON:-python}} engineV4.py --api_config="${{LINE}}" --log_dir="${{LOG}}" "${{ARGS[@]}}"
"""


def _render_repro_readme(run: dict, rows: list[dict]) -> str:
    total_fail = sum(row["failures"] for row in rows)
    # README 只列真正有失败配置的 job，命令可直接复制执行；无失败时给出一行说明。
    lines = [
        f"# 复现包：{run.get('run_id')}",
        "",
        f"- 生成时间：{now_str()}",
        f"- 真实失败配置合计：{total_fail} 条",
        "- 运行位置：把本目录放在 `PaddleAPITest/` 下，或在任意位置设置 "
        "`PADDLEAPITEST_ROOT=/path/to/PaddleAPITest`。",
        "",
        "## 批量复现",
        "",
        "```bash",
        "cd repro",
    ]
    for row in rows:
        if row["failures"]:
            lines.append(f"bash run_{row['job']}.sh")
    if not any(row["failures"] for row in rows):
        lines.append("# 本次没有真实失败")
    lines += [
        "```",
        "",
        "## 单 case 复现",
        "",
        "```bash",
        "cd repro",
        "bash run_case.sh <job> <行号>",
        "# 例：bash run_case.sh 1M_accuracy 3",
        "```",
        "",
        # 口径段落解释「哪些分类会重跑、参数与环境变量从哪来」，避免复现者改错输入。
        "## 口径",
        "",
        "- `configs/<job>_failures.txt` 只收真实失败（paddle_accuracy / paddle_error / "
        "paddle_crash / paddle_cuda / config_input / config_parse / config_convert / "
        "both_error / torch_error / torch_only_error）。",
        "- `configs/<job>_args.txt` / `<job>_env.sh` 保存本次运行的 engine 参数和环境变量，"
        "复现脚本直接复用，避免手抄参数走样。",
        "- `refs/` 下是 OOM / skip / bitwise 参考清单，默认不跑；OOM 往往没有单 case 日志。",
        "- 原始失败日志见 `../issues/cases/`，完整 traceback 见 `../logs/<job>/log_*.log`。",
        "- 复现脚本会从自身目录向上自动寻找 `engineV4.py`；也可显式设置 "
        "`PADDLEAPITEST_ROOT=/path/to/PaddleAPITest`。",
    ]
    return "\n".join(lines) + "\n"


def _render_repro_index(rows: list[dict]) -> str:
    lines = [
        "# 复现包索引",
        "",
        "| job | 真实失败 | 失败清单 | 参考清单 | 原始日志目录 |",
        "|---|---|---|---|---|",
    ]
    # 参考清单按分类聚合显示数量；没有失败清单的 job 用 - 占位，保持列对齐。
    for row in rows:
        refs = ", ".join(f"{cat}({n})" for cat, _name, n in row["refs"]) or "-"
        cfg = f"`configs/{row['config']}`" if row["config"] else "-"
        lines.append(f"| {row['job']} | {row['failures']} | {cfg} | {refs} | `{row['log_dir']}` |")
    return "\n".join(lines) + "\n"


def config_fingerprint(path: Path) -> str:
    """输入文件指纹，用于 status 里区分两次同名配置。"""
    # 分块读取避免大配置文件一次性读入内存；取 12 位十六进制已足够区分同名输入。
    # 指纹只用于展示与区分两次输入，不参与任何缓存命中或跳过判断。
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]
