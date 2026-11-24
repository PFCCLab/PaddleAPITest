#!/usr/bin/env python3
"""shape_test_runner 公共逻辑：运行目录、输入校验、engine 日志解析。

本模块刻意不 import paddle / torch / engineV4，保证在"环境已就绪"和
"环境缺失需要体检"两种状态下都能运行；所有判断只读文件。
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent
# 工具包的上一层就是仓库根：engineV4.py 与 inputs/ 都相对它解析，不能依赖 cwd。
PROJECT_ROOT = KIT_DIR.parent
# 多机共享盘时，各机器挂载点可能不同；用 SHAPE_TEST_RUNS_DIR 指向同一个 runs 目录。
RUNS_DIR = Path(os.environ.get("SHAPE_TEST_RUNS_DIR") or (KIT_DIR / "runs")).expanduser()

# 心跳间隔与失联阈值：跨机器无法用 PID 判断存活，只能看共享盘上的心跳时间。
HEARTBEAT_INTERVAL = 15
# 阈值取间隔的 8 倍：共享盘写盘抖动或 NFS 属性缓存会延迟可见性，
# 卡太紧会把正常运行的 part 误判成失联。
STALE_SECONDS = 120

# 真实失败：进入复现 config。与 api_test/runtime/utils/debug_bundle.py 保持同一口径。
FAIL_CATS = (
    "paddle_accuracy",
    "paddle_error",
    "paddle_crash",
    "paddle_cuda",
    "config_input",
    "config_parse",
    "config_convert",
    "both_error",
    "torch_error",
    "torch_only_error",
)
# timeout 既不算真实失败也不进参考清单：超时多为机器负载导致，纳入复现会掩盖真实根因。
# 参考分类：只整理成清单和日志，不进入默认复现批次。
REF_CATS = ("oom", "skip", "paddle_bitwise", "paddle_bitwise_knows")

# 展示顺序：配置类问题在前、崩溃与报错居中、参考分类和 pass 收尾。
# 该顺序同时决定状态表与 INDEX 的行序，调整会直接改变报告可读性。
CAT_ORDER = (
    "config_input",
    "config_parse",
    "config_convert",
    "both_error",
    "paddle_crash",
    "paddle_cuda",
    "paddle_error",
    "paddle_accuracy",
    "timeout",
    "oom",
    "torch_error",
    "torch_only_error",
    "paddle_bitwise",
    "paddle_bitwise_knows",
    "skip",
    "pass",
)
# 这些分类不会独立出现在 api_config_*.txt 里，但取值顺序用于展示。
CLASS_FILE_PREFIX = "api_config_"
CHECKPOINT_FILE = "checkpoint.txt"

# 单 case 日志协议：engine 用 >>> / <<< 成对包裹一个配置，尾行第三段是 PASS/FAIL/SKIP/COMPLETED。
# 两个正则都锚定行首，避免 traceback 里引用的日志片段被当成新 case 的开头。
CASE_HEAD = re.compile(r"^>>> CASE ([0-9a-f]+)")
CASE_TAIL = re.compile(r"^<<< CASE ([0-9a-f]+) \| (\w+)")
SIZE_DIM = re.compile(r"Size\(\[([0-9,\s]*)\]\)")

# 错误签名按优先级匹配；越靠前越能说明根因。
# "Error:" 放在元组末尾：它最宽松，只有前面所有具体异常名都没命中时才作为兜底。
ERROR_MARKERS = (
    "CUDA error",
    "illegal memory access",
    "device-side assert",
    "misaligned address",
    "an illegal memory access",
    "Segmentation fault",
    "core dumped",
    "Fatal Python error",
    "Precondition failed",
    "RuntimeError:",
    "AssertionError:",
    "ValueError:",
    "TypeError:",
    "AttributeError:",
    "NotImplementedError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "OSError:",
    "MemoryError:",
    "Error:",
)


def now_str() -> str:
    """本地时间字符串，状态文件里统一使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_epoch() -> float:
    return time.time()


def atomic_write_json(path: Path, payload) -> None:
    """先写临时文件再替换，避免状态文件被中断写坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default if default is not None else {}


def read_config_lines(path: Path) -> list[str]:
    """读取 API config 文件，跳过空行和整行注释。"""
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def validate_config_file(path: Path) -> dict:
    """静态校验配置：存在、非空、每行是可解析的 Python 表达式、重复行统计。

    只做语法级检查，不 import Paddle，避免把"环境缺包"误判成"配置错误"。
    """
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "total": 0,
        "unique": 0,
        "duplicates": [],
        "duplicate_count": 0,
        "parse_errors": [],
        "zero_dim_cases": 0,
    }
    if not result["exists"]:
        return result
    lines = read_config_lines(path)
    result["total"] = len(lines)
    seen = set()
    dup = []
    zero = 0
    # 解析错误和重复行号都只保留前 20 条：配置文件可能上万行，报告里给样本即可。
    for lineno, line in enumerate(lines, 1):
        if line in seen:
            dup.append(lineno)
        else:
            seen.add(line)
        try:
            ast.parse(line, mode="eval")
        except SyntaxError as err:
            if len(result["parse_errors"]) < 20:
                result["parse_errors"].append(f"line {lineno}: {err.msg}")
        dims = SIZE_DIM.findall(line)
        # 只认真正的 0 维，不能把 1024 里的字符 "0" 算进去。
        # 逐个 token 精确比较，才能把 Size([0, 4]) 与 Size([1024, 1024]) 区分开。
        if any(token.strip() == "0" for dim in dims for token in dim.split(",")):
            zero += 1
    result["unique"] = len(seen)
    result["duplicates"] = dup[:20]
    result["duplicate_count"] = len(dup)
    result["zero_dim_cases"] = zero
    return result


def find_run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def job_key(shape: str, mode: str) -> str:
    return f"{shape}_{mode}"


def job_log_dir(run_dir: Path, job: dict) -> Path:
    # log_dir 存的是相对 run 根目录的路径；老 run.json 缺该字段时回退到 logs/<key>。
    return run_dir / job.get("log_dir", f"logs/{job['key']}")


def pid_alive(pid) -> bool:
    """判断 PID 是否还活着；权限不足时按存活处理，避免误杀/误判。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown-host"


def heartbeat_fresh(entry: dict, stale_seconds: int = STALE_SECONDS) -> bool:
    """共享盘上的心跳是否仍新鲜；没有心跳字段时按不新鲜处理。"""
    # 心跳缺失或不是数字一律按失联处理：宁可让用户手动确认，也不要误报成运行中。
    try:
        ts = float(entry.get("heartbeat_ts"))
    except (TypeError, ValueError):
        return False
    return (time.time() - ts) <= stale_seconds


def read_unique_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    seen = set()
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def classify_counts(log_dir: Path) -> dict:
    """读取一个 engine log_dir 的分类结果。"""
    counts = {}
    # 每类结果写成 api_config_<cat>.txt；文件不存在表示该类本次没命中，
    # 文件存在但为空则不计入 counts，两者语义不同。
    for cat in CAT_ORDER:
        path = log_dir / f"{CLASS_FILE_PREFIX}{cat}.txt"
        if path.exists():
            lines = read_unique_lines(path)
            if lines:
                counts[cat] = lines
    checkpoint = read_unique_lines(log_dir / CHECKPOINT_FILE)
    return {"classes": counts, "checkpoint": checkpoint}


def _case_log_sources(log_dir: Path) -> list[Path]:
    """优先用 engine 合并好的 log_inorder.log，缺失时回退到 worker 临时日志。"""
    inorder = log_dir / "log_inorder.log"
    if inorder.exists() and inorder.stat().st_size > 0:
        return [inorder]
    tmp = log_dir / ".tmp"
    if not tmp.is_dir():
        return []
    # 回退路径只有 .tmp/log_<pid>.log（每个 worker 一份）；按 mtime 排序还原写入顺序，
    # 同一配置重跑时后写入的块才会排在后面。
    workers = [p for p in tmp.glob("log_*.log") if p.is_file()]
    return sorted(workers, key=lambda p: p.stat().st_mtime)


def parse_case_blocks(log_dir: Path) -> dict[str, list[dict]]:
    """把 engine 日志切成单 case 块。

    返回 config -> [{"case_id", "block", "status"}]。同一 config 可能因重跑
    出现多次，调用方按"最后一次非 PASS/SKIP"优先取。
    """
    blocks: dict[str, list[dict]] = {}
    # 多个来源共享同一个 blocks 字典，因此跨文件也能按 config 聚合候选块。
    for source in _case_log_sources(log_dir):
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pending: list[str] = []
        case_id = None
        for line in text.splitlines(keepends=True):
            head = CASE_HEAD.match(line)
            if head:
                # 上一条没有 <<< CASE（crash/被杀）也保留，便于定位。
                if case_id and pending:
                    _append_block(blocks, pending, case_id, "INCOMPLETE")
                pending = [line]
                case_id = head.group(1)
                continue
            if case_id is None:
                continue
            pending.append(line)
            tail = CASE_TAIL.match(line)
            if tail:
                _append_block(blocks, pending, case_id, tail.group(2))
                pending = []
                case_id = None
        if case_id and pending:
            _append_block(blocks, pending, case_id, "INCOMPLETE")
    return blocks


def _append_block(store: dict, block: list[str], case_id: str, status: str) -> None:
    # 约定：CASE 头之后第一行就是原始 config。
    config = block[1].strip() if len(block) > 1 else ""
    if not config:
        return
    store.setdefault(config, []).append(
        {"case_id": case_id, "block": "".join(block), "status": status.upper()}
    )


def pick_case_block(candidates: list[dict]) -> dict | None:
    """优先取最后一次 FAIL/INCOMPLETE 块；只有 PASS/SKIP 时取最后一个。"""
    if not candidates:
        return None
    bad = [c for c in candidates if c["status"] not in ("PASS", "SKIP")]
    return bad[-1] if bad else candidates[-1]


def normalize_signature(text: str) -> str:
    """抹掉地址和具体数值，让同类错误能聚合。"""
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\b\d{3,}\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 截断到 200 字符：签名只用于分组展示，保留完整 traceback 只会撑坏报告表格。
    return text[:200]


def case_signature(block: str) -> str:
    """从单 case 日志里提取最像根因的一行。"""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    for marker in ERROR_MARKERS:
        for line in lines:
            if marker in line and not line.startswith(">>> CASE"):
                return normalize_signature(line)
    # accuracy 失败通常没有异常栈，用比较结果兜底。
    # 三者都缺失时返回固定串，保证调用方拿到的永远是非空且可哈希的签名。
    for line in lines:
        if "FINAL RESULT" in line or "max_abs_diff" in line or "max_rel_diff" in line:
            return normalize_signature(line)
    return "(no error signature)"


def classify_job(job: dict, log_dir: Path) -> dict:
    """汇总单个 job：进度、分类计数、失败率。"""
    total = int(job.get("total") or 0)
    data = classify_counts(log_dir)
    classes = data["classes"]
    done = len(data["checkpoint"])
    counts = {cat: len(lines) for cat, lines in classes.items()}
    passed = counts.get("pass", 0)
    skipped = counts.get("skip", 0)
    ref = sum(counts.get(cat, 0) for cat in REF_CATS)
    failed = sum(counts.get(cat, 0) for cat in FAIL_CATS)
    # 分母沿用流水线口径：pass / (pass + 真实失败 + oom 等非 skip 参考分类)。
    # skip 是主动跳过，不计入分母；OOM 计入，否则 pass 率会被高估。
    judged = passed + failed + (ref - skipped)
    rate = (passed / judged * 100.0) if judged else 0.0
    # total 来自 run.json 声明的输入行数；缺失时进度记 0 而不是 100，避免"没跑完却显示已完成"。
    progress = (done / total * 100.0) if total else 0.0
    return {
        "total": total,
        "done": done,
        "progress": progress,
        "pass": passed,
        "fail": failed,
        "ref": ref,
        "skip": skipped,
        "pass_rate": rate,
        "counts": counts,
        "classes": classes,
    }


def job_runtime_state(
    run_dir: Path, job: dict, state: dict, stale_seconds: int = STALE_SECONDS
) -> dict:
    """结合 state.json 与共享盘心跳判断 job 所处阶段。

    跨机器时 PID 只在本机有意义；远端 part 只能靠 heartbeat_ts 判断存活。
    """
    key = job["key"]
    entry = (state.get("jobs") or {}).get(key, {})
    log_dir = job_log_dir(run_dir, job)
    host = str(entry.get("host") or "")
    local = host in ("", hostname())
    alive = local and pid_alive(entry.get("pid"))
    fresh = heartbeat_fresh(entry, stale_seconds)
    hb_status = entry.get("hb_status") or entry.get("status")
    # 判定优先级：本机 PID 存活或远端心跳新鲜 > 心跳停在 running（失联）> 已落盘退出码 >
    # 只按日志目录推断 > 完全没有启动痕迹。
    if alive or (hb_status == "running" and fresh):
        status = "运行中"
    elif hb_status == "running":
        # 心跳停了但退出码还没落盘：机器掉线/进程被杀。
        status = "失联"
    elif entry.get("exit_code") is not None:
        status = "已完成" if entry["exit_code"] == 0 else f"结束(exit={entry['exit_code']})"
    elif log_dir.exists():
        info = classify_job(job, log_dir)
        status = "已完成" if info["total"] and info["done"] >= info["total"] else "中断"
    else:
        status = "未启动"
    # 最近活动时间取 log_dir 下最新文件的 mtime，用于识别"还在写日志但没退出"的任务。
    last_mtime = 0.0
    if log_dir.exists():
        for path in log_dir.rglob("*"):
            try:
                if path.is_file():
                    last_mtime = max(last_mtime, path.stat().st_mtime)
            except OSError:
                continue
    return {
        "key": key,
        "shape": job.get("shape"),
        "mode": job.get("mode"),
        "status": status,
        "host": host or hostname(),
        "heartbeat": (
            datetime.fromtimestamp(float(entry["heartbeat_ts"])).strftime("%Y-%m-%d %H:%M:%S")
            if entry.get("heartbeat_ts")
            else ""
        ),
        "pid": entry.get("pid"),
        "exit_code": entry.get("exit_code"),
        "started_at": entry.get("started_at"),
        "finished_at": entry.get("finished_at"),
        "last_activity": (
            datetime.fromtimestamp(last_mtime).strftime("%Y-%m-%d %H:%M:%S") if last_mtime else ""
        ),
        "log_dir": str(log_dir),
    }
