#!/usr/bin/env python3
"""多机共享盘 campaign：切分配置、认领 part、汇总进度与问题。

约定：
- 一个 campaign 就是 ``runs/<run_id>/``，里面只放 ``campaign.json``、``shards/``、
  ``parts/`` 以及汇总产物；每台机器只写自己的 ``parts/<part>/``，互不覆盖。
- 跨机器不能用 PID 判断存活，改用共享盘上的 ``parts/<part>/heartbeat.json``。
- 同一个 part 用 ``.run.lock`` 抢占，避免两台机器重复跑。
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from pathlib import Path

import lib
from lib import (
    atomic_write_json,
    classify_job,
    heartbeat_fresh,
    hostname,
    load_json,
    now_epoch,
    now_str,
    pid_alive,
    read_config_lines,
    read_unique_lines,
    validate_config_file,
)

# 盘上布局的常量集中在这里定义，避免各处手写字符串时拼歪，导致读写落到不同的路径上。
# shards/ 只存放切分产物，不含运行期可变数据，因此 verify 可以反复读取而不与 worker 争用。
SHARDS_DIR = "shards"
PARTS_DIR = "parts"
# campaign.json 是切分结果的唯一权威来源；part 下的 run.json 只描述该 part 自己的运行态。
CAMPAIGN_FILE = "campaign.json"
# 锁文件名以点开头，既不会和 shape 名撞车，也不会被 *.txt、api_config_* 之类的扫描带上。
LOCK_FILE = ".run.lock"
# 心跳文件是跨机器判断存活的唯一依据：别的机器看不到对方 PID，只能读这个文件的时间戳。
HEARTBEAT_FILE = "heartbeat.json"
# 默认 round-robin：让单台机器拿到的配置在空间或模式上更分散，失败不会集中在同一种 shape。
DEFAULT_STRATEGY = "round-robin"


# 以下路径工具只做「路径推导 + 存在性判断」，不读盘内容，因此可以被单机流程安全调用。
def campaign_file(run_dir: Path) -> Path:
    return run_dir / CAMPAIGN_FILE


def is_campaign(run_dir: Path) -> bool:
    return campaign_file(run_dir).is_file()


# 缺少 campaign.json 说明 run_id 给错了，直接退出，而不是回退成单机 run 的语义。
def load_campaign(run_dir: Path) -> dict:
    campaign = load_json(campaign_file(run_dir), {})
    if not campaign:
        raise SystemExit(f"未找到 campaign 记录: {campaign_file(run_dir)}")
    return campaign


# part 名直接来自 campaign.json 里的 id（part-XX），这里不做合法性校验，调用方需保证它存在。
def part_dir(campaign_dir: Path, part: str) -> Path:
    return campaign_dir / PARTS_DIR / part


def split_lines(lines: list[str], shards: int, strategy: str = DEFAULT_STRATEGY) -> list[list[str]]:
    """把一个配置文件切成 shards 份；不足 shards 行时自动减少份数。"""
    # 空输入返回空列表而不是 [[]]，让上层能区分「没有分片」和「有一份空分片」。
    if not lines:
        return []
    # 份数收敛到实际行数：否则 contiguous 分支会切出若干空分片，part 编号也被白白虚增。
    shards = max(1, min(int(shards), len(lines)))
    if strategy == "contiguous":
        # 先均分再补余数，保证一定产出 shards 份。
        # 余数分给前 rem 份，各份行数最多差 1；verify 的切分守恒校验依赖这里的总量不变。
        base, rem = divmod(len(lines), shards)
        chunks, start = [], 0
        for index in range(shards):
            size = base + (1 if index < rem else 0)
            chunks.append(lines[start : start + size])
            start += size
        return chunks
    # 未知策略直接失败：拼错 --strategy 时若静默回退，用户会误以为自己用的是 contiguous 切分。
    if strategy != "round-robin":
        raise SystemExit(f"未知切分策略: {strategy}（可选 round-robin / contiguous）")
    # round-robin 按行号取模分发，相邻配置落到不同机器，单条坏配置不会拖垮整段分片。
    buckets: list[list[str]] = [[] for _ in range(shards)]
    for index, line in enumerate(lines):
        buckets[index % shards].append(line)
    return buckets


def create_campaign(
    run_id: str,
    inputs: dict[str, Path],
    shards: int,
    *,
    strategy: str = DEFAULT_STRATEGY,
    dedup: bool = False,
    modes: list[str],
    model: str,
    card: str,
    python: str,
    gpus: str,
    sanitizer: str,
    accuracy_mode: str,
    workers: int | None,
    timeout: int | None,
    engine_extra: list[str],
    force: bool = False,
    stale_seconds: int = lib.STALE_SECONDS,
) -> dict:
    """校验输入、切分并写 campaign.json，返回 campaign 内容。"""
    # 本函数只做校验与落盘，不启动任何进程；真正跑 part 由 st.py 的 run-part 分支负责。
    # 副作用：创建 runs/<run_id>/{campaign.json,shards/}，--force 时整目录重建。
    run_dir = lib.RUNS_DIR / run_id
    # 已存在时默认拒绝，避免两台机器用同一个 run_id 互相覆盖对方刚写好的分片。
    if run_dir.exists():
        if not force:
            raise SystemExit(f"campaign 目录已存在：{run_dir}。换 --run-id，或加 --force 覆盖。")
        # --force 只允许覆盖 campaign，避免误删同名的单机 run。
        # 判据是 campaign.json 是否存在：单机 run 只有 run.json，不会命中这条覆盖路径。
        if not campaign_file(run_dir).is_file():
            raise SystemExit(f"{run_dir} 不是 campaign 目录，拒绝 --force 覆盖")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    normalized: dict[str, list[str]] = {}
    input_meta = {}
    # 先校验再用：total 为 0 或存在解析错误时整批失败，避免切完分片再回头追查残缺输入。
    for shape, path in inputs.items():
        info = validate_config_file(path)
        if not info["exists"] or info["total"] == 0:
            raise SystemExit(f"{shape} 输入不可用：{path}")
        if info["parse_errors"]:
            raise SystemExit(f"{shape} 存在无法解析的配置：{info['parse_errors'][:3]}")
        lines = read_config_lines(path)
        # 去重按行文本精确匹配并保留首次出现顺序，相同输入因此总能切出可复现的分片。
        if dedup:
            seen, unique = set(), []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique.append(line)
            lines = unique
        normalized[shape] = lines
        # 记录 total/unique/sha1：verify 用它做守恒校验，复现时也能确认输入没有被替换过。
        input_meta[shape] = {
            "source": str(path),
            "total": info["total"],
            "unique": len(lines),
            "dedup": bool(dedup),
            "duplicates": info.get("duplicate_count", 0),
            "sha1": _sha1_lines(lines),
        }

    # 每个 shape 单独切；part 数量以 --shards 为准，较小的文件可能出现“该 shape 没有分片”。
    buckets = {shape: split_lines(lines, shards, strategy) for shape, lines in normalized.items()}
    # 各 shape 行数可能相差很大，part 总数取最长的那份，行数少的 shape 在后几个 part 里可以缺席。
    part_count = max((len(value) for value in buckets.values()), default=0)
    parts = []
    for index in range(part_count):
        part_id = f"part-{index:02d}"
        part_path = run_dir / SHARDS_DIR / part_id
        # exist_ok 让 --force 之后的重切不会因为残留目录而失败。
        part_path.mkdir(parents=True, exist_ok=True)
        shapes, totals = {}, {}
        for shape, shape_buckets in buckets.items():
            if index >= len(shape_buckets) or not shape_buckets[index]:
                continue
            file_path = part_path / f"{shape}.txt"
            file_path.write_text("\n".join(shape_buckets[index]) + "\n", encoding="utf-8")
            shapes[shape] = str(file_path.relative_to(run_dir))
            totals[shape] = len(shape_buckets[index])
        # 所有 shape 在该下标都没有内容时不登记 part，避免汇总里出现 0/0 的空 job。
        if shapes:
            parts.append({"id": part_id, "shapes": shapes, "totals": totals})

    # campaign.json 用原子写一次成型：其他机器可能已经在轮询它，以此判断切分是否就绪。
    # 这里的 shards 记录实际产出的 part 数，可能小于请求值（行数不足时 split_lines 已收敛）。
    campaign = {
        "run_id": run_id,
        "created_at": now_str(),
        "kind": "campaign",
        "strategy": strategy,
        "shards": len(parts),
        "dedup": bool(dedup),
        "modes": modes,
        "model": model or "local",
        "card": card,
        "python": python,
        "gpus": gpus,
        "sanitizer": sanitizer,
        "accuracy_mode": accuracy_mode,
        "workers": workers,
        "timeout": timeout,
        "engine_extra": engine_extra,
        "heartbeat_interval": lib.HEARTBEAT_INTERVAL,
        "stale_seconds": stale_seconds,
        "inputs": input_meta,
        "parts": parts,
    }
    atomic_write_json(campaign_file(run_dir), campaign)
    return campaign


def _sha1_lines(lines: list[str]) -> str:
    # 函数内 import：hashlib 只有创建 campaign 时才会用到，没必要在模块导入期就加载。
    import hashlib

    digest = hashlib.sha1()
    # 只取前 12 位供人工比对，不承担安全职责，因此不需要更长的摘要。
    digest.update("\n".join(lines).encode("utf-8"))
    return digest.hexdigest()[:12]


def part_inputs(campaign_dir: Path, campaign: dict, part: str) -> dict[str, Path]:
    # 只按 campaign.json 登记的相对路径解析，不在这里检查文件存在：缺文件留到 run-part 阶段暴露。
    for item in campaign.get("parts", []):
        if item["id"] == part:
            return {shape: campaign_dir / rel for shape, rel in item["shapes"].items()}
    # 走到这里说明 part 名拼错或来自别的 campaign，直接报错而不是返回空 dict 让上层以为没输入。
    raise SystemExit(f"campaign 中没有 part: {part}")


def part_hint(campaign: dict) -> str:
    # 只打印前 6 个 part：机器再多也没必要把全部 id 塞进一行提示里。
    ids = [p["id"] for p in campaign.get("parts", [])]
    return ", ".join(ids[:6]) + (" ..." if len(ids) > 6 else "")


def campaign_jobs(campaign_dir: Path, campaign: dict, stale_seconds: int | None = None):
    """把各 part 的 run.json/state/心跳归一成一套 jobs + state，供汇总复用。"""
    # stale 阈值优先级：显式参数 > campaign 固化的值 > 库默认；固化是为了让汇总不受当台机器环境影响。
    stale = int(stale_seconds or campaign.get("stale_seconds") or lib.STALE_SECONDS)
    # 返回三元组：jobs 给 report/debug 复用，state 复刻单机 state.json 的结构，part_rows 供表格渲染。
    jobs, state = [], {"jobs": {}}
    part_rows = []
    for item in campaign.get("parts", []):
        part = item["id"]
        # 三份文件都可能缺失或只写了一部分（例如正在被拷贝），load_json 统一回退默认值。
        pdir = part_dir(campaign_dir, part)
        run = load_json(pdir / "run.json", {})
        hb = load_json(pdir / HEARTBEAT_FILE, {})
        pstate = load_json(pdir / "state.json", {})
        part_jobs = []
        for job in run.get("jobs", []):
            # key 加上 part 前缀后全局唯一，同一 shape/mode 出现在不同 part 时不会被合并成一条。
            key = f"{part}/{job['key']}"
            entry = dict((pstate.get("jobs") or {}).get(job["key"], {}))
            # 心跳里的 host 优先于 state 里的 host：part 被别的机器接管后要显示当前实际执行者。
            entry["host"] = hb.get("host") or entry.get("host")
            entry["heartbeat_ts"] = hb.get("heartbeat_ts")
            entry["hb_status"] = hb.get("status")
            state["jobs"][key] = entry
            view = dict(job)
            view["key"] = key
            view["part"] = part
            # 汇总视图统一用 parts/<part>/... 的相对路径，整目录搬走后仍能定位到日志。
            view["log_dir"] = str(Path(PARTS_DIR) / part / job["log_dir"])
            view["input"] = str(Path(PARTS_DIR) / part / job["input"])
            jobs.append(view)
            part_jobs.append(view)

        if not run:
            # 未启动的 part 也生成占位 job，这样总体进度包含“还没跑的量”。
            # 占位 key 与真实 run 的命名保持一致，part 启动前后进度视图不会突然跳变。
            for mode in campaign.get("modes", []):
                for shape, total in item.get("totals", {}).items():
                    view = {
                        "key": f"{part}/{shape}_{mode}",
                        "part": part,
                        "shape": shape,
                        "mode": mode,
                        "input": str(Path(PARTS_DIR) / part / "inputs" / f"{shape}.txt"),
                        "log_dir": str(Path(PARTS_DIR) / part / "logs" / f"{shape}_{mode}"),
                        "total": int(total),
                        "engine_args": [],
                        "env": {},
                    }
                    jobs.append(view)
                    part_jobs.append(view)

        # 统计口径与单机一致，统一交给 classify_job 解析日志，避免两套计数逻辑对不上账。
        totals = {"done": 0, "total": 0, "pass": 0, "fail": 0, "ref": 0, "skip": 0}
        for job in part_jobs:
            info = classify_job(job, lib.job_log_dir(campaign_dir, job))
            for name in totals:
                totals[name] += info[name]
        progress = round(totals["done"] / totals["total"] * 100.0, 1) if totals["total"] else 0.0

        # 状态判定顺序即优先级：未启动 > 运行中 > 失联 > 全 0 退出 > 有非 0 退出 > 中断。
        fresh = heartbeat_fresh(hb, stale)
        hb_status = hb.get("status")
        exits = [entry.get("exit_code") for entry in (pstate.get("jobs") or {}).values()]
        if not run:
            status = "未启动"
        elif hb_status == "running" and fresh:
            status = "运行中"
        # 心跳还写着 running 但 ts 已过期：机器掉线或进程被 kill，来不及写收尾状态。
        elif hb_status == "running":
            status = "失联"
        # 全部 exit_code 为 0 才算完成；混入 None 说明还有 job 没结束，会落到下面的中断分支。
        elif exits and all(code == 0 for code in exits):
            status = "已完成"
        elif any(code not in (None, 0) for code in exits):
            # 取第一个非 0 退出码做代表，只是给运维一个入口，完整失败清单在 collect 里。
            bad = next(code for code in exits if code not in (None, 0))
            status = f"结束(exit={bad})"
        else:
            status = "中断"
        part_rows.append(
            {
                "id": part,
                "host": hb.get("host") or "",
                "status": status,
                "heartbeat": _fmt_ts(hb.get("heartbeat_ts")),
                **totals,
                "progress": progress,
            }
        )
    return jobs, state, part_rows


def _fmt_ts(ts) -> str:
    # ts 可能是 float 字符串、None 或损坏值，任何一种都只降级成空串，不让单个坏字段拖垮整张表。
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return ""


def parts_table(campaign_dir: Path, campaign: dict, stale_seconds: int | None = None) -> list[dict]:
    """part 视角的状态表：给 parts / worker / status 共用。"""
    # 这张表是 parts/worker/status 三个子命令的共同数据源，字段改名会同时影响三处输出。
    _jobs, _state, rows = campaign_jobs(campaign_dir, campaign, stale_seconds)
    # 以 campaign.json 的 part 列表为准遍历，避免某个 part 目录缺失时整行从表里消失。
    by_id = {row["id"]: row for row in rows}
    out = []
    for item in campaign.get("parts", []):
        row = dict(by_id.get(item["id"], {"id": item["id"], "status": "未启动"}))
        row["shapes"] = item.get("shapes", {})
        row["configs"] = sum(item.get("totals", {}).values())
        # available 同时要求「没有新鲜心跳」和「状态属于可接管集合」，防止运行中的 part 被 worker 抢走。
        busy = part_busy(
            part_dir(campaign_dir, item["id"]),
            int(campaign.get("stale_seconds") or lib.STALE_SECONDS),
        )
        # 已完成或正常退出的 part 默认不再分配；要重跑失败 part 由 next_available_part 的 rerun_failed 决定。
        row["available"] = bool(busy) is False and row["status"] in ("未启动", "中断", "失联")
        out.append(row)
    return out


def next_available_part(
    campaign_dir: Path,
    campaign: dict,
    stale_seconds: int | None = None,
    rerun_failed: bool = False,
    exclude: set[str] | None = None,
) -> str | None:
    """worker 用：挑第一个可跑的 part；默认不重跑已经结束的 part。"""
    # exclude 让单次 worker 循环记住刚失败的 part，避免同一个进程反复挑中同一个坏分片。
    exclude = exclude or set()
    # 按表格顺序挑第一个可用 part：顺序稳定，多机同时启动时会自然分散到不同 part。
    for row in parts_table(campaign_dir, campaign, stale_seconds):
        if row["id"] in exclude:
            continue
        if row["available"]:
            return row["id"]
        # 只有显式 rerun_failed 才回收「结束(exit!=0)」的 part；中断与已完成不会被自动重跑。
        if rerun_failed and row["status"].startswith("结束"):
            return row["id"]
    return None


@contextlib.contextmanager
def aggregate_lock(campaign_dir: Path, stale_seconds: int = 600):
    """collect/debug/repro/verify 的跨机器互斥，避免并发写坏汇总产物。"""
    # 聚合锁与 part 的 .run.lock 分开命名：汇总进程不应该阻塞 worker 抢占 part。
    lock = campaign_dir / ".aggregate.lock"
    # 先建目录再拿锁：锁文件就在 campaign 目录下，目录不存在时 O_CREAT 会直接失败。
    campaign_dir.mkdir(parents=True, exist_ok=True)
    # 最多重试 3 次：重试只发生在接管陈旧锁之后，足以覆盖「刚清理又被抢」的竞争窗口。
    for _ in range(3):
        try:
            # O_EXCL 让「创建」和「占用」成为原子操作，是共享盘（含 NFS）上可靠的互斥原语。
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            # 写入 host/pid/ts 三元组，接管方据此判断锁主是否还活着、锁是否已经过期。
            info = {"host": hostname(), "pid": os.getpid(), "ts": now_epoch()}
            os.write(fd, json.dumps(info, ensure_ascii=False).encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            # 拿不到锁就读锁内容判断：同机看 PID 是否存活，跨机只能看 ts，因为 PID 空间彼此不可见。
            info = load_json(lock, {})
            try:
                age = time.time() - float(info.get("ts"))
            except (TypeError, ValueError):
                # ts 解析失败按极大值处理，等价于「锁已失效」，避免残留坏锁永久阻塞聚合。
                age = 1e9
            owner_local = info.get("host") == hostname() and pid_alive(info.get("pid"))
            # 同时满足「超时」和「锁主不是本机活进程」才接管，防止误删同机上仍在运行的聚合。
            if age > stale_seconds and not owner_local:
                lock.unlink(missing_ok=True)
                continue
            # 不降级成无锁执行：并发写 verify/collect 产物会互相覆盖，宁可让用户稍后重试。
            raise SystemExit(
                f"campaign 聚合正在 {info.get('host')} 执行（{int(age)}s 前）；稍后重试"
            )
    else:
        # for-else：三次都被别人持有（而不是接管成功）才会走到这里，说明锁竞争异常激烈。
        raise SystemExit(f"无法获取聚合锁: {lock}")
    try:
        yield
    finally:
        # 正常路径在这里释放；进程被 kill 时锁会残留，靠上面的 stale 判定由下一次聚合接管。
        lock.unlink(missing_ok=True)


def reset_part(
    campaign_dir: Path, campaign: dict, part: str, *, clean: bool = False, force: bool = False
) -> str:
    """清理失联 part 的锁/心跳；--clean 连日志和状态一起删，从头跑。"""
    # 只动本 part 目录，绝不触碰 shards/ 与其它 part，因此可以在任意一台机器上安全执行。
    pdir = part_dir(campaign_dir, part)
    if not pdir.exists():
        raise SystemExit(f"part 不存在: {pdir}")
    hb = load_json(pdir / HEARTBEAT_FILE, {})
    # fresh 的 running 说明可能真有一台机器在跑，必须 --force 才允许重置，避免两个进程写同一 part。
    if hb.get("status") == "running" and heartbeat_fresh(hb) and not force:
        raise SystemExit(f"{part} 仍在 {hb.get('host')} 运行；确认停止后加 --force")
    if clean:
        # --clean 删除整个 part 目录（含 checkpoint），语义是「当作从未跑过」，代价是丢掉已完成进度。
        shutil.rmtree(pdir)
        return f"{part} 已清空（下次 run-part 从头跑）"
    # 非 clean 分支刻意保留日志与 checkpoint：run-part 会按 checkpoint 续跑，只补缺失的配置。
    (pdir / LOCK_FILE).unlink(missing_ok=True)
    if hb:
        hb.update({"status": "reset", "heartbeat_ts": now_epoch(), "heartbeat_at": now_str()})
        atomic_write_json(pdir / HEARTBEAT_FILE, hb)
    state = load_json(pdir / "state.json", {})
    # 清空 supervisor_pid 是必要的：否则同机旧 PID 被复用时，可能被误判为仍有 supervisor 在跑。
    state["supervisor_pid"] = None
    state["updated_at"] = now_str()
    atomic_write_json(pdir / "state.json", state)
    return f"{part} 已重置锁和心跳（日志保留，run-part 会按 checkpoint 续跑）"


def verify_campaign(campaign_dir: Path, campaign: dict, stale_seconds: int | None = None) -> dict:
    """校验切分守恒与每个 part 的 case 覆盖，输出 verify.md/json。"""
    # 每次校验先删旧产物再重建：残留的 missing/*.txt 会让人误以为这一轮仍然缺配置。
    verify_dir = campaign_dir / "verify"
    missing_dir = verify_dir / "missing"
    shutil.rmtree(verify_dir, ignore_errors=True)
    missing_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    split_rows = []
    # 切分守恒：各 part 行数之和应等于去重后的输入行数。
    for shape, meta in campaign.get("inputs", {}).items():
        # dedup 时等式右边取 unique，否则取原始 total，这里决定「期望值」到底用哪个数。
        expected = int(meta.get("unique") if campaign.get("dedup") else meta.get("total") or 0)
        actual = sum(part.get("totals", {}).get(shape, 0) for part in campaign.get("parts", []))
        split_rows.append(
            {
                "shape": shape,
                "expected": expected,
                "actual": actual,
                "ok": expected == actual,
            }
        )

    for item in campaign.get("parts", []):
        part = item["id"]
        pdir = part_dir(campaign_dir, part)
        run = load_json(pdir / "run.json", {})
        # 以 run.json 的 modes 为准：part 启动后可能被手动加跑过其它模式，与 campaign 记录不一致。
        modes = run.get("modes") or campaign.get("modes", [])
        jobs = {(job["shape"], job["mode"]): job for job in run.get("jobs", [])}
        expected_by_shape = {}
        for shape, rel in item.get("shapes", {}).items():
            path = campaign_dir / rel
            # 用 set 做包含判断：分片文件里的重复行不影响缺失统计，missing 只关心「有没有」。
            expected_by_shape[shape] = set(read_unique_lines(path))
        if not run:
            for shape in item.get("totals", {}):
                # run.json 缺失时仍保留该 shape 的期望集合（空集），让汇总报告「未启动」而不是直接跳过。
                expected_by_shape.setdefault(shape, set())
        for mode in modes:
            for shape, expected in expected_by_shape.items():
                job = jobs.get((shape, mode))
                # 找不到 job 说明这个模式在该 part 没被调度过，日志目录按命名约定回退推断。
                log_dir = (pdir / job["log_dir"]) if job else (pdir / "logs" / f"{shape}_{mode}")
                # covered = checkpoint ∪ 历史 api_config 文件：前者是续跑断点，后者是已执行批次的记录。
                checkpoint = set(read_unique_lines(log_dir / "checkpoint.txt"))
                classes = set()
                for path in log_dir.glob("api_config_*.txt"):
                    classes.update(read_unique_lines(path))
                covered = checkpoint | classes
                missing = sorted(expected - covered)
                # extra 非空通常意味着分片被重新生成过，旧日志里的配置已不属于当前输入。
                extra = sorted(covered - expected)
                if missing:
                    # 缺失清单按 part__shape__mode 命名，双下划线分隔避免 shape 里的连字符造成解析歧义。
                    (missing_dir / f"{part}__{shape}__{mode}.txt").write_text(
                        "\n".join(missing) + "\n", encoding="utf-8"
                    )
                rows.append(
                    {
                        "part": part,
                        "shape": shape,
                        "mode": mode,
                        "started": bool(run),
                        "expected": len(expected),
                        "checkpoint": len(checkpoint),
                        "covered": len(covered),
                        "missing": len(missing),
                        "extra": len(extra),
                        "missing_file": (
                            f"verify/missing/{part}__{shape}__{mode}.txt" if missing else ""
                        ),
                    }
                )

    # complete 要求守恒通过、每个 (shape, mode) 都启动过且无缺失；存在未启动 job 也会判为不完整。
    complete = all(row["ok"] for row in split_rows) and all(
        row["started"] and row["missing"] == 0 for row in rows
    )
    payload = {
        "run_id": campaign["run_id"],
        "generated_at": now_str(),
        "complete": complete,
        "split": split_rows,
        "rows": rows,
        "missing_total": sum(row["missing"] for row in rows),
        "unstarted_jobs": sum(1 for row in rows if not row["started"]),
    }
    verify_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(verify_dir / "verify.json", payload)
    (verify_dir / "verify.md").write_text(_render_verify_md(payload), encoding="utf-8")
    return payload


def _render_verify_md(payload: dict) -> str:
    # 直接拼 Markdown 文本，不引入表格库：校验要能在任意机器上跑，不给仓库加额外依赖。
    lines = [
        f"# campaign 完整性校验：{payload['run_id']}",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 结论：{'完整' if payload['complete'] else '不完整'}",
        f"- 缺失配置合计：{payload['missing_total']}，未启动 job：{payload['unstarted_jobs']}",
        "",
        "## 切分守恒",
        "",
        "| shape | 期望行数 | 分片行数 | 结果 |",
        "|---|---|---|---|",
    ]
    for row in payload["split"]:
        lines.append(
            f"| {row['shape']} | {row['expected']} | {row['actual']} | "
            f"{'OK' if row['ok'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 覆盖率",
        "",
        "| part | shape | mode | 期望 | checkpoint | 已覆盖 | 缺失 | 额外 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['part']} | {row['shape']} | {row['mode']} | {row['expected']} | "
            f"{row['checkpoint']} | {row['covered']} | {row['missing']} | {row['extra']} |"
        )
    if payload["missing_total"]:
        lines += [
            "",
            "缺失配置清单：`verify/missing/<part>__<shape>__<mode>.txt`。",
            "常见原因：part 未启动、机器掉线、或 checkpoint 未落盘。",
        ]
    return "\n".join(lines) + "\n"


def campaign_status(campaign_dir: Path, campaign: dict, stale_seconds: int | None = None) -> dict:
    # 延迟 import：report 依赖较重，只有真正做汇总的子命令才需要加载它。
    import report

    jobs, state, parts = campaign_jobs(campaign_dir, campaign, stale_seconds)
    return report.build_status_view(
        campaign["run_id"],
        campaign_dir,
        jobs,
        state,
        inputs=campaign.get("inputs", {}),
        modes=campaign.get("modes", []),
        parts=parts,
        campaign=True,
        # supervisor_alive 用「任一 part 运行中」近似：campaign 没有单一 supervisor，只能按 part 汇总。
        supervisor_alive=any(row["status"] == "运行中" for row in parts),
    )


def campaign_collect(campaign_dir: Path, campaign: dict) -> dict:
    # collect 只关心失败归类，不需要 part 存活信息，所以丢弃 state/parts 两个返回值。
    import report

    jobs, _state, _parts = campaign_jobs(campaign_dir, campaign)
    return report.collect_issues_view(
        campaign["run_id"],
        campaign_dir,
        jobs,
        inputs=campaign.get("inputs", {}),
        modes=campaign.get("modes", []),
        campaign=True,
    )


def campaign_debug(
    campaign_dir: Path,
    campaign: dict,
    *,
    card: str | None = None,
    model: str | None = None,
    out_dir: Path | None = None,
):
    # debug 复用单机的 jobs 视图，campaign=True 让产物带上 part 维度。
    import debug_bundle

    jobs, _state, _parts = campaign_jobs(campaign_dir, campaign)
    return debug_bundle.build_debug_from_jobs(
        campaign["run_id"],
        campaign_dir,
        jobs,
        run=campaign,
        card=card,
        model=model,
        out_dir=out_dir,
        campaign=True,
    )


def campaign_repro(campaign_dir: Path, campaign: dict) -> dict:
    """逐 part 复用单机 repro；顶层只放索引，避免把不同机器的脚本混在一起。"""
    import report

    # 只处理真正启动过的 part：没有 run.json 的 part 也没有日志，生成复现包只会得到空目录。
    rows = []
    for item in campaign.get("parts", []):
        part = item["id"]
        pdir = part_dir(campaign_dir, part)
        if not (pdir / "run.json").is_file():
            continue
        # 每个 part 的复现脚本写在自己的 repro/ 下，路径带 part 前缀，同名 job 不会互相覆盖。
        payload = report.build_repro(f"{campaign['run_id']}/{part}", pdir)
        rows.append({"part": part, "path": str(pdir / "repro"), "jobs": payload["jobs"]})
    target = campaign_dir / "repro"
    # 顶层 repro 目录整体重建，保证索引与实际 part 列表一致，不残留上一轮的条目。
    shutil.rmtree(target, ignore_errors=True)
    # 顶层只放 README 索引，不复制脚本：脚本必须留在产生日志的那套目录结构里才可复跑。
    target.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# campaign 复现包索引：{campaign['run_id']}",
        "",
        "每个 part 的复现脚本在本 part 的 `repro/` 下；失败配置按 part 隔离，避免跨机器混跑。",
        "",
        "| part | repro 目录 | 有失败配置的 job |",
        "|---|---|---|",
    ]
    for row in rows:
        jobs = ", ".join(job["job"] for job in row["jobs"] if job["failures"]) or "-"
        lines.append(f"| {row['part']} | `{row['path']}` | {jobs} |")
    (target / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_id": campaign["run_id"], "parts": rows}


def part_busy(pdir: Path, stale_seconds: int = lib.STALE_SECONDS) -> dict | None:
    """返回仍占用该 part 的心跳信息；没有则 None。"""
    # 只认 running + 未过期的心跳；done/reset 等收尾状态视为已释放，可立即被接管。
    hb = load_json(pdir / HEARTBEAT_FILE, {})
    if hb.get("status") == "running" and heartbeat_fresh(hb, stale_seconds):
        return hb
    return None


def acquire_part_lock(
    pdir: Path, force: bool = False, stale_seconds: int = lib.STALE_SECONDS
) -> Path:
    """用 O_EXCL 抢占 part；锁里的 owner 失联时允许接管。"""
    # 先确保 part 目录存在，否则共享盘上 O_CREAT 会因父目录缺失而失败。
    pdir.mkdir(parents=True, exist_ok=True)
    lock = pdir / LOCK_FILE
    # 与聚合锁同构的重试次数：3 次足以覆盖「清理陈旧锁后被别人抢先」的窗口。
    for _ in range(3):
        try:
            # 锁内容在创建后立刻写入，读锁方可能在 write 完成前看到空文件，故下面按「解析失败即失效」处理。
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            payload = {"host": hostname(), "pid": os.getpid(), "acquired_at": now_str()}
            os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            os.close(fd)
            return lock
        except FileExistsError:
            # 竞争失败不做 sleep 退避：抢占窗口很短，立即判断持有者状态比固定等待更贴近真实可用性。
            owner = load_json(lock, {})
            hb = load_json(pdir / HEARTBEAT_FILE, {})
            owner_local = owner.get("host") == hostname() and pid_alive(owner.get("pid"))
            # busy 用心跳而不是锁文件判断：锁可能因进程崩溃残留，心跳则会随进程停止而陈旧。
            busy = hb.get("status") == "running" and heartbeat_fresh(hb, stale_seconds)
            # 接管条件：显式 --force，或「对方既无新鲜心跳、也不是本机活进程」两条同时成立。
            if force or (not busy and not owner_local):
                lock.unlink(missing_ok=True)
                continue
            # 报错带上 host 与 PID，方便用户去那台机器确认进程是否真的还在跑。
            raise SystemExit(
                f"{pdir} 已被 {owner.get('host')} PID={owner.get('pid')} 占用；"
                "确认对方已停止后加 --force 接管"
            )
    # 三次都是被别人持有而非接管成功，说明有并发 worker 在抢同一个 part。
    raise SystemExit(f"无法获取 part 锁: {lock}")


def release_part_lock(lock: Path) -> None:
    # 释放失败可以忽略：锁删不掉最多让下一次靠 stale 判定接管，不该在收尾阶段抛异常。
    try:
        lock.unlink(missing_ok=True)
    except OSError:
        pass


class Heartbeat:
    """后台刷新 shared-disk 心跳；跨机器状态只认这个文件。"""

    # 构造时只准备线程，不启动：start() 才写第一次 running，避免对象创建即产生盘上副作用。
    def __init__(self, pdir: Path, run_id: str, part: str, interval: int = lib.HEARTBEAT_INTERVAL):
        self.pdir = pdir
        self.run_id = run_id
        self.part = part
        # interval 至少 1 秒，防止误传 0 变成忙等，把共享盘写满小文件。
        self.interval = max(1, int(interval))
        self.current_job = ""
        # 用 Event.wait 而不是 sleep：finish() 能立刻唤醒后台线程，不必等满一个心跳周期。
        self._stop = threading.Event()
        # daemon=True：主进程异常退出时不让心跳线程拖住解释器退出。
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        # 先同步写一次 running 再起线程，保证 run-part 返回前盘上就能看到「有人在跑」。
        self._write("running")
        self._thread.start()

    def set_job(self, key: str) -> None:
        # 只改内存字段，新 job 要等下一次心跳才落盘，因此看板最多有一个心跳周期的延迟。
        self.current_job = key

    def finish(self, status: str = "done") -> None:
        # 先停线程再写终态，避免后台循环在终态之后又把它覆盖回 running。
        self._stop.set()
        if self._thread.is_alive():
            # join 带 2 秒上限：即使线程卡在共享盘 IO，收尾也不能无限等待。
            self._thread.join(timeout=2)
        self._write(status)

    def _loop(self) -> None:
        # 后台循环只负责续期 ts；current_job 由 set_job 更新，跨线程读单个字符串引用是安全的。
        while not self._stop.wait(self.interval):
            self._write("running")

    def _write(self, status: str) -> None:
        # 每次整体原子替换文件，读方不会看到半截 JSON，也就不会把正常心跳误判成失联。
        atomic_write_json(
            self.pdir / HEARTBEAT_FILE,
            {
                "run_id": self.run_id,
                "part": self.part,
                "host": hostname(),
                "pid": os.getpid(),
                "status": status,
                "current_job": self.current_job,
                "heartbeat_ts": now_epoch(),
                "heartbeat_at": now_str(),
            },
        )
