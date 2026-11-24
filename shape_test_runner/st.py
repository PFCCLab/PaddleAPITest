#!/usr/bin/env python3
"""给定 0size.txt / 1M.txt，启动并在本地已就绪的环境里运行 PaddleAPITest。

不安装任何依赖、不管理镜像/Paddle/PaddleFleet 版本；只负责：
启动测试 -> 汇总进度 -> 整理失败问题与单 case 日志 -> 生成复现包。

常用命令::

    python shape_test_runner/st.py doctor
    python shape_test_runner/st.py start --0size /path/0size.txt --1m /path/1M.txt
    python shape_test_runner/st.py start --1m /path/1M.txt --mode accuracy
    python shape_test_runner/st.py status --run-id <id>
    python shape_test_runner/st.py collect --run-id <id>
    python shape_test_runner/st.py repro --run-id <id>
    python shape_test_runner/st.py debug --run-id <id> --card H
    python shape_test_runner/st.py stop --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# 以脚本方式直接运行时没有包上下文，把本目录放进 sys.path 才能 import 同级的 campaign/lib。
sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign  # noqa: E402
import debug_bundle  # noqa: E402
import lib  # noqa: E402
import report  # noqa: E402

# 两种配置集定位不同：0size 覆盖含 0 维的边界 shape，1M 覆盖大 shape 的显存与耗时。
SHAPE_CHOICES = ("0size", "1M")
# 主模式互斥：engineV4 一次只接受一个主开关，这里列出取值仅用于命令行校验。
MODE_CHOICES = ("accuracy", "paddleonly", "stable")
# 与 test_pipeline/generic_configs 的默认值保持一致。
# 0size 单例快、case 多，4 个 worker 才能填满流水；1M 单例吃显存，1 个更稳。
DEFAULT_WORKERS = {"0size": 4, "1M": 1}
# 超时只兜住卡死的 case：1M 的编译与显存分配更慢，因此放宽到 20 分钟。
DEFAULT_TIMEOUT = {"0size": 600, "1M": 1200}
# 仅在 paddleonly 下注入：打开分配器填充与 NaN/Inf 检查，让越界和未初始化读更容易暴露。
PADDLE_FLAGS = {
    # 系统分配器不带缓存池，compute-sanitizer 才能准确报告越界地址。
    "FLAGS_use_system_allocator": "true",
    "FLAGS_check_cuda_error": "true",
    # 用 255 填充未初始化显存，避免脏数据让 kernel 侥幸跑出看似正确的结果。
    "FLAGS_alloc_fill_value": "255",
    "FLAGS_check_nan_inf": "true",
}


def _log(message: str) -> None:
    # supervisor 的 stdout 被重定向进日志文件，必须 flush 才能保证日志时序与现场一致。
    print(f"[{lib.now_str()}] {message}", flush=True)


def _expand_gpu_ids(spec: str) -> list[str]:
    """把 0 / 0,2 / 0-3 展开成 id 列表，仅用于计算 num_gpus。"""
    # 只做字符串展开，不校验 id 是否存在：真正的设备可见性由 engineV4 的 --gpu_ids 决定。
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        # 区间写法 0-3 展开成逐个 id；单值写法保持原样，交给下游统一处理。
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(str(i) for i in range(int(start), int(end) + 1))
        else:
            ids.append(part)
    return ids


def build_engine_args(
    shape: str,
    mode: str,
    *,
    workers: int,
    timeout: int,
    gpus: str,
    sanitizer: bool,
    extra: list[str],
    accuracy_mode: str = "auto",
) -> list[str]:
    """按 pipeline 口径组装 engineV4 参数（不含输入和 log_dir）。"""
    # 输入文件与 log_dir 由调用方拼接，这里只决定怎么跑，便于 dry-run 直接复用。
    args: list[str] = []
    if mode == "accuracy":
        # auto 与 pipeline 对齐：1M 数据量大，双卡分别跑 paddle/torch 可以摊薄单卡显存。
        dual = accuracy_mode == "dual" or (accuracy_mode == "auto" and shape == "1M")
        args.append("--accuracy_dual_gpu=True" if dual else "--accuracy=True")
        # atol/rtol 归零表示只接受严格一致，避免放宽阈值后精度问题在汇总里被吃掉。
        args += ["--atol=0.0", "--rtol=0.0", "--use_gpu_mode=True"]
    elif mode == "paddleonly":
        args += ["--paddle_only=True", "--use_gpu_mode=True"]
        if sanitizer:
            # sanitizer 只在 paddleonly 下有收益：没有 torch 对照时，越界只能靠它定位。
            args.append("--use_compute_sanitizer=True")
    elif mode == "stable":
        # stable 复用上次缓存的 numpy 输入，排除随机数差异带来的偶发不一致。
        args += ["--accuracy_stable=True", "--use_cached_numpy=True", "--use_gpu_mode=True"]
    else:
        # 显式兜底：模式拼错时宁可退出，也不要静默按某个默认模式跑完。
        raise SystemExit(f"未知模式: {mode}")

    args += [
        f"--num_workers_per_gpu={workers}",
        f"--timeout={timeout}",
        "--show_runtime_status=True",
    ]
    # num_gpus 必须与 gpu_ids 个数对齐，engineV4 用它来决定 worker 池规模。
    if gpus and gpus != "-1":
        count = len(_expand_gpu_ids(gpus))
        # 把展开后的卡数与原始 gpu_ids 一起传给 engineV4，它据此把 worker 绑到对应设备。
        args += [f"--num_gpus={count}", f"--gpu_ids={gpus}"]
    else:
        # -1 表示使用全部可见 GPU，个数由 engineV4 自行探测，这里不做重复计数。
        args.append("--num_gpus=-1")
    # 透传参数放最后：argparse 以最后一次出现为准，调用方可以覆盖前面拼好的同名参数。
    args.extend(extra)
    return args


def check_environment(python: str) -> list[dict]:
    """只体检不安装：解释器、paddle、torch、engine 依赖、GPU。"""
    # 只读探测：每一步失败都只记录不抛异常，便于一次把缺失项列全。
    checks = []

    def record(name: str, ok: bool, detail: str, fatal: bool = False) -> None:
        # fatal 区分缺了必然跑不了与缺了只影响部分功能，决定 doctor 的退出码。
        checks.append({"name": name, "ok": ok, "detail": detail, "fatal": fatal})

    # 入口脚本与 tester 目录是硬依赖，缺任意一个都无法开跑。
    record(
        "engineV4.py",
        (lib.PROJECT_ROOT / "engineV4.py").is_file(),
        str(lib.PROJECT_ROOT / "engineV4.py"),
        fatal=True,
    )
    record(
        "tester/",
        (lib.PROJECT_ROOT / "tester").is_dir(),
        str(lib.PROJECT_ROOT / "tester"),
        fatal=True,
    )

    def run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
        # 体检命令必须限时：驱动异常时 nvidia-smi 会挂住，不能让 doctor 一起卡死。
        try:
            proc = subprocess.run(
                cmd, cwd=lib.PROJECT_ROOT, capture_output=True, text=True, timeout=timeout
            )
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except (FileNotFoundError, subprocess.TimeoutExpired) as err:
            # 127 沿用 shell 的命令不存在约定；超时也归到这里，调用方只看是否非 0。
            return 127, str(err)

    # 先确认解释器可用，后面的 import 探测才有意义。
    rc, out = run([python, "--version"])
    record("python", rc == 0, out.splitlines()[-1] if out else f"rc={rc}", fatal=True)

    # paddle/torch/pynvml 缺任意一个都会让 accuracy 或显存预估失效，因此全部设为 fatal。
    for module, fatal in (("paddle", True), ("torch", True), ("pynvml", True)):
        rc, out = run(
            [python, "-c", f"import {module}; print(getattr({module}, '__version__', 'ok'))"]
        )
        # 只取最后一行：import 失败时 stderr 可能有多行，末尾那行才是真正的报错。
        detail = out.splitlines()[-1] if out else f"rc={rc}"
        record(f"import {module}", rc == 0, detail, fatal=fatal)

    # GPU 信息是提示性的：拿不到也不影响纯 CPU 排查，因此不标记 fatal。
    rc, out = run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"])
    record("nvidia-smi", rc == 0, out.replace("\n", "; ") if out else f"rc={rc}", fatal=False)

    # compute-sanitizer 属于可选工具，缺失只影响 paddleonly 的越界定位能力。
    rc, out = run(["compute-sanitizer", "--version"], timeout=20)
    record("compute-sanitizer", rc == 0, (out.splitlines()[0] if out else f"rc={rc}"), fatal=False)
    return checks


def cmd_doctor(args) -> int:
    # 体检只依赖文件系统与子进程，不需要 paddle 环境本身可用。
    checks = check_environment(args.python)
    print(f"# 环境体检（{lib.PROJECT_ROOT}）\n")
    for item in checks:
        # 三态展示：OK 正常，FAIL 阻断，WARN 缺失但可以继续跑。
        flag = "OK  " if item["ok"] else ("FAIL" if item["fatal"] else "WARN")
        print(f"[{flag}] {item['name']}: {item['detail']}")
    fatal = [item["name"] for item in checks if item["fatal"] and not item["ok"]]
    if fatal:
        print(f"\n阻断项：{', '.join(fatal)}。本项目不负责安装，请先准备好运行环境。")
        # 用 2 区别于普通失败 1，便于外层脚本识别环境不合格而不是命令执行失败。
        return 2
    print("\n环境检查通过。")
    return 0


def _selected_shapes(args) -> dict[str, Path]:
    # 统一转成绝对路径：run.json 里存的是快照路径，相对路径在别处重跑会指错位置。
    mapping = {}
    if args.size0:
        mapping["0size"] = Path(args.size0).expanduser().resolve()
    if args.size1m:
        mapping["1M"] = Path(args.size1m).expanduser().resolve()
    if args.shapes:
        # --shapes 是过滤器而不是新来源，必须先由 --0size / --1m 给出对应文件。
        wanted = [s.strip() for s in args.shapes.split(",") if s.strip()]
        # 只接受预定义取值，拼错时立刻报错，避免跑出一份空的汇总。
        unknown = [s for s in wanted if s not in SHAPE_CHOICES]
        if unknown:
            raise SystemExit(f"未知 shape: {unknown}")
        if not mapping:
            raise SystemExit("指定了 --shapes 但没有给 --0size / --1m 输入文件")
        mapping = {shape: path for shape, path in mapping.items() if shape in wanted}
    return mapping


def _selected_modes(args) -> list[str]:
    # 允许逗号分隔并容忍空格；真正的合法性校验在下面统一做。
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODE_CHOICES]
    if unknown:
        raise SystemExit(f"未知 mode: {unknown}，可选 {MODE_CHOICES}")
    return modes


def _validate_inputs(inputs: dict[str, Path], dedup: bool) -> dict[str, list[str] | None]:
    """校验输入；返回每个 shape 的最终有效行（None 表示无重复、按原文件走）。"""
    # None 表示没有重复行、调用方直接拷贝原文件，只有 --dedup 且确有重复时才落具体内容。
    final_lines: dict[str, list[str] | None] = {}
    failed = False
    print("# 输入校验\n")
    for shape, path in inputs.items():
        # 只做语法级检查，不 import paddle，避免把环境缺失误判成配置错误。
        info = lib.validate_config_file(path)
        if not info["exists"]:
            print(f"[FAIL] {shape}: 文件不存在 {path}")
            failed = True
            continue
        if info["total"] == 0:
            print(f"[FAIL] {shape}: 文件为空 {path}")
            failed = True
            continue
        print(f"[OK]   {shape}: {path}")
        print(
            f"       配置 {info['total']} 条，去重后 {info['unique']} 条，"
            f"含 0 维的 case {info['zero_dim_cases']} 条"
        )
        if info["duplicates"]:
            print(
                f"[WARN] {shape}: 发现重复行（行号示例 {info['duplicates']}）"
                + ("，将使用去重后的副本" if dedup else "，建议加 --dedup 或自行去重")
            )
        if info["parse_errors"]:
            # 只回显前若干条，避免一个坏文件刷屏掩盖真正的原因。
            print(f"[FAIL] {shape}: 存在无法解析的配置：")
            for err in info["parse_errors"]:
                print(f"       {err}")
            failed = True
        # 0size 文件的语义就是含 0 维，一条都检测不到通常意味着给错了文件。
        if shape == "0size" and info["zero_dim_cases"] == 0:
            print(f"[WARN] 0size: 没有检测到含 0 维的 Size，请确认文件没给错")
        final_lines[shape] = lib.read_config_lines(path) if (dedup and info["duplicates"]) else None
    if failed:
        # 有硬错误时整体退出：半套输入跑出来的汇总没有统计意义。
        raise SystemExit("输入校验失败，未启动测试。")
    return final_lines


def _preflight(
    inputs: dict[str, Path], python: str, skip_doctor: bool, sanitizer_choice: str
) -> bool:
    """环境体检；返回是否启用 compute-sanitizer。"""
    if skip_doctor:
        # 跳过体检就无法探测工具是否存在，auto 只能退化成 off，除非用户显式指定 on。
        return sanitizer_choice == "on"
    checks = check_environment(python)
    fatal = [item for item in checks if item["fatal"] and not item["ok"]]
    if fatal:
        for item in fatal:
            print(f"[FAIL] {item['name']}: {item['detail']}")
        # 体检失败即中止：这里不做安装也不降级，保持只跑已就绪环境的边界。
        raise SystemExit("环境不完整（本项目不负责安装）。修复后重试，或加 --skip-doctor。")
    # auto 的含义是有就用，只有本机确实装了 compute-sanitizer 才启用，避免启动即失败。
    return (sanitizer_choice == "on") or (
        sanitizer_choice == "auto"
        and any(c["name"] == "compute-sanitizer" and c["ok"] for c in checks)
    )


def _build_jobs(
    inputs: dict[str, Path],
    final_lines: dict,
    modes: list[str],
    *,
    workers: int | None,
    timeout: int | None,
    gpus: str,
    sanitizer: bool,
    engine_extra: list[str],
    accuracy_mode: str,
) -> list[dict]:
    """按 shape × mode 生成 job；单机 start 和多机 run-part 共用。"""
    # job 顺序固定跟着 SHAPE_CHOICES 与 modes，单机和多机才能逐字对比日志目录。
    jobs = []
    # 只遍历选中的 shape：没有输入文件的配置集不生成 job，避免产出恒为 0 进度的空记录。
    for shape in SHAPE_CHOICES:
        if shape not in inputs:
            continue
        # 未显式指定时按 shape 取默认值：worker 数与超时都随配置集规模变化。
        shape_workers = workers or DEFAULT_WORKERS[shape]
        shape_timeout = timeout or DEFAULT_TIMEOUT[shape]
        for mode in modes:
            jobs.append(
                {
                    # key 同时决定日志目录名，续跑靠它定位已有 checkpoint，改名会丢进度。
                    "key": lib.job_key(shape, mode),
                    "shape": shape,
                    "mode": mode,
                    "input": f"inputs/{shape}.txt",
                    "log_dir": f"logs/{lib.job_key(shape, mode)}",
                    # total 优先用去重后的行数，保证进度分母与实际执行条数一致。
                    "total": len(final_lines[shape] or lib.read_config_lines(inputs[shape])),
                    "engine_args": build_engine_args(
                        shape,
                        mode,
                        workers=shape_workers,
                        timeout=shape_timeout,
                        gpus=gpus,
                        sanitizer=sanitizer,
                        extra=engine_extra,
                        accuracy_mode=accuracy_mode,
                    ),
                    # PADDLE_FLAGS 只对 paddleonly 生效；accuracy 要与 torch 对齐，注入会改变数值行为。
                    "env": dict(PADDLE_FLAGS) if mode == "paddleonly" else {},
                    # 记录参数来源，便于回溯这条 job 等价于哪个 pipeline 配置。
                    "merged_from": [f"generic_configs/run_{shape}_{mode}.yaml"],
                }
            )
    return jobs


def _write_and_launch(
    run_id: str,
    run_dir: Path,
    inputs: dict[str, Path],
    final_lines: dict,
    jobs: list[dict],
    *,
    modes: list[str],
    python: str,
    gpus: str,
    sanitizer: bool,
    model: str,
    card: str,
    auto_report: bool,
    foreground: bool,
    dry_run: bool,
    extra: dict | None = None,
) -> int:
    """落 run.json 并启动 supervisor；单机 start 与 run-part 共用。"""
    if dry_run:
        # dry-run 只打印：不建目录也不写 run.json，避免留下半成品运行记录。
        print("\n# dry-run：将执行以下命令\n")
        for job in jobs:
            envs = " ".join(f"{k}={v}" for k, v in job["env"].items())
            print(f"# {job['key']}  ({job['total']} 条)")
            prefix = f"{envs} " if envs else ""
            # 用 shlex.quote 包裹路径，带空格或特殊字符时打印结果仍可直接复制执行。
            print(
                f"cd {shlex.quote(str(lib.PROJECT_ROOT))} && {prefix}{python} engineV4.py "
                f"--api_config_file={shlex.quote(str(inputs[job['shape']]))} "
                f"--log_dir={shlex.quote(str(run_dir / job['log_dir']))} "
                f"{' '.join(job['engine_args'])}\n"
            )
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs").mkdir(exist_ok=True)
    for shape, path in inputs.items():
        # 去重副本写进运行目录，保证事后复现用的是当时真正跑过的行集合。
        dest = run_dir / "inputs" / f"{shape}.txt"
        if final_lines[shape] is not None:
            dest.write_text("\n".join(final_lines[shape]) + "\n", encoding="utf-8")
        else:
            shutil.copyfile(path, dest)

    # run.json 是一次运行的不可变快照，status/repro 只读消费，不再回推参数。
    run_payload = {
        "run_id": run_id,
        "created_at": lib.now_str(),
        "updated_at": lib.now_str(),
        "project_root": str(lib.PROJECT_ROOT),
        "python": python,
        "gpus": gpus,
        "sanitizer": sanitizer,
        "model": model or "local",
        "card": card,
        "shapes": [shape for shape in SHAPE_CHOICES if shape in inputs],
        "modes": modes,
        "inputs": {
            shape: {
                "path": f"inputs/{shape}.txt",
                "source": str(path),
                "total": len(final_lines[shape] or lib.read_config_lines(path)),
                # sha1 用于事后判断报告是不是基于同一份输入生成。
                "sha1": report.config_fingerprint(run_dir / "inputs" / f"{shape}.txt"),
            }
            for shape, path in inputs.items()
        },
        "auto_report": auto_report,
        "jobs": jobs,
    }
    # part 运行才需要 campaign 归属与心跳参数，单机 start 不写这些字段。
    if extra:
        run_payload.update(extra)
    # 原子写：并发机器读 run.json 时不会读到写了一半的内容。
    lib.atomic_write_json(run_dir / "run.json", run_payload)
    state_path = run_dir / "state.json"
    # 复用既有 state.json，让 --resume 或 run-part 重入时能接着上次的进度。
    state = lib.load_json(state_path, {"jobs": {}})
    state.setdefault("jobs", {})
    state["host"] = lib.hostname()
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(state_path, state)

    print(f"\n运行目录：{run_dir}")
    print(
        f"jobs：{', '.join(job['key'] for job in jobs)}"
        f"（{'、'.join(str(job['total']) for job in jobs)} 条配置）"
    )
    # 前台模式直接在当前进程 supervise，调试时 Ctrl-C 能立刻拿到完整栈。
    if foreground:
        return _supervise(run_id, run_dir, auto_report=auto_report)

    log_path = run_dir / "supervisor.log"
    # 后台 supervisor 用独立会话启动，命令行退出（含 Ctrl-C）不会连带杀掉测试进程。
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_supervise",
                "--run-id",
                run_id,
                "--run-dir",
                str(run_dir),
            ],
            cwd=str(lib.PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    # 记录 PID 只为 status/resume 判断存活；跨机器时这个值不具参考价值。
    state["supervisor_pid"] = proc.pid
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(state_path, state)
    print(f"已在后台启动 supervisor PID={proc.pid}，日志：{log_path}")
    print(f"查看进度：python shape_test_runner/st.py status --run-id {run_id}")
    return 0


def cmd_start(args) -> int:
    inputs = _selected_shapes(args)
    if not inputs:
        raise SystemExit("至少提供一个输入：--0size <file> 或 --1m <file>")
    # 去重同时保留首次出现顺序：同一个 mode 重复给出时只跑一次。
    modes = list(dict.fromkeys(_selected_modes(args)))
    # run_id 决定运行目录名，默认用秒级时间戳，足够人工区分多次启动。
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = lib.find_run_dir(run_id)
    # 目录已存在说明同名运行还在或已结束，必须显式 --resume 才允许复用。
    if run_dir.exists() and not args.resume:
        raise SystemExit(f"运行目录已存在：{run_dir}。换 --run-id，或加 --resume 续跑。")

    # 顺序固定为校验输入、环境体检、组装 job：前两步失败时不会留下任何运行目录。
    final_lines = _validate_inputs(inputs, args.dedup)
    sanitizer = _preflight(inputs, args.python, args.skip_doctor, args.sanitizer)
    jobs = _build_jobs(
        inputs,
        final_lines,
        modes,
        workers=args.workers,
        timeout=args.timeout,
        gpus=args.gpus,
        sanitizer=sanitizer,
        engine_extra=args.engine_arg,
        accuracy_mode=args.accuracy_mode,
    )
    return _write_and_launch(
        run_id,
        run_dir,
        inputs,
        final_lines,
        jobs,
        modes=modes,
        python=args.python,
        gpus=args.gpus,
        sanitizer=sanitizer,
        model=args.model or "local",
        card=args.card or os.environ.get("APITEST_CARD", "").strip(),
        auto_report=args.auto_report,
        foreground=args.foreground,
        dry_run=args.dry_run,
    )


def cmd_split(args) -> int:
    """把一个 txt 切成多个 part，写共享盘 campaign。"""
    # 切分只写 shards/ 与 campaign.json，不启动任何测试进程，可以随时重复执行。
    inputs = _selected_shapes(args)
    if not inputs:
        raise SystemExit("至少提供一个输入：--0size <file> 或 --1m <file>")
    # 0 或负数的份数会产出空 campaign，这里提前拒绝而不是静默成功。
    if args.shards <= 0:
        raise SystemExit("--shards 必须大于 0")
    modes = list(dict.fromkeys(_selected_modes(args)))
    run_id = args.run_id or time.strftime("campaign_%Y%m%d_%H%M%S")
    # 卡型可以来自命令行或 APITEST_CARD，后者便于在容器里统一注入。
    card = args.card or os.environ.get("APITEST_CARD", "").strip()
    info = campaign.create_campaign(
        run_id,
        inputs,
        args.shards,
        strategy=args.strategy,
        dedup=args.dedup,
        modes=modes,
        model=args.model or "local",
        card=card,
        python=args.python or "python",
        gpus=args.gpus or "-1",
        sanitizer=args.sanitizer,
        accuracy_mode=args.accuracy_mode,
        workers=args.workers,
        timeout=args.timeout,
        engine_extra=args.engine_arg,
        force=args.force,
        stale_seconds=args.stale,
    )
    run_dir = lib.find_run_dir(run_id)
    print(f"campaign：{run_dir}")
    print(
        f"切分策略：{info['strategy']}，part 数：{len(info['parts'])}，"
        f"模式：{', '.join(info['modes'])}"
    )
    print("\n| part | shapes | 配置数 |")
    print("|---|---|---|")
    for part in info["parts"]:
        shapes = ", ".join(f"{shape}({n})" for shape, n in part["shapes"].items())
        print(f"| {part['id']} | {shapes} | {sum(part['totals'].values())} |")
    # 行数少于请求份数时 campaign 会自动减少 part，必须提示，否则容易被误认为漏跑。
    if len(info["parts"]) < args.shards:
        print(f"\n[WARN] 请求 {args.shards} 份，但行数不足，实际只生成 {len(info['parts'])} 份。")
    # 切分默认不去重：重复行会真的被执行两次，是否合并交给用户决定。
    for shape, meta in info.get("inputs", {}).items():
        if meta.get("duplicates"):
            print(
                f"[WARN] {shape} 有 {meta['duplicates']} 行重复；需要去重请重新 split 并加 --dedup"
            )
    print("\n下一步：在每台机器上执行")
    print(f"  python shape_test_runner/st.py run-part --run-id {run_id} --part <part>")
    print(f"任一机器查看汇总：python shape_test_runner/st.py status --run-id {run_id}")
    return 0


def _run_part_impl(campaign_dir: Path, info: dict, part: str, opts) -> int:
    """真正执行一个 part；run-part 和 worker 共用。"""
    inputs = campaign.part_inputs(campaign_dir, info, part)
    pdir = campaign.part_dir(campaign_dir, part)
    # 失联阈值以 campaign 创建时的记录优先，保证所有机器用同一口径判断。
    stale = int(info.get("stale_seconds") or lib.STALE_SECONDS)
    force = bool(getattr(opts, "force", False))
    busy = campaign.part_busy(pdir, stale)
    # 远端占用只能靠心跳判断（PID 跨机器无意义），因此这里不看 state.json。
    if busy and not force:
        raise SystemExit(
            f"{part} 正在 {busy.get('host')} 上运行（心跳 {busy.get('heartbeat_at')}）；"
            "确认对方停止后加 --force 接管"
        )
    # 本机残留锁（supervisor 被 kill -9 等）再用 PID 复核，避免把僵尸锁当成占用。
    owner = lib.load_json(pdir / campaign.LOCK_FILE, {})
    if (
        owner
        and owner.get("host") == lib.hostname()
        and lib.pid_alive(owner.get("pid"))
        and not force
    ):
        raise SystemExit(
            f"{part} 已被本机 PID={owner.get('pid')} 占用；确认进程已停止后加 --force 接管"
        )

    mode = getattr(opts, "mode", None)
    # 命令行给了 --mode 就以命令行为准，否则沿用 campaign 创建时的模式组合。
    if mode:
        modes = list(dict.fromkeys(_selected_modes(opts)))
    else:
        modes = list(info.get("modes") or ["accuracy", "paddleonly"])
    final_lines = _validate_inputs(inputs, False)
    # 以下参数统一遵循命令行优先、其次 campaign 记录、最后内置默认的取值顺序。
    python = getattr(opts, "python", None) or info.get("python") or "python"
    sanitizer = _preflight(
        inputs,
        python,
        bool(getattr(opts, "skip_doctor", False)),
        getattr(opts, "sanitizer", None) or info.get("sanitizer") or "auto",
    )
    gpus = getattr(opts, "gpus", None)
    if gpus is None:
        gpus = info.get("gpus") or "-1"
    workers = getattr(opts, "workers", None)
    if workers is None:
        workers = info.get("workers")
    timeout = getattr(opts, "timeout", None)
    if timeout is None:
        timeout = info.get("timeout")
    # campaign 级透传在前、命令行追加在后，同名参数以命令行（后出现）为准。
    extra = list(info.get("engine_extra") or []) + list(getattr(opts, "engine_arg", None) or [])
    jobs = _build_jobs(
        inputs,
        final_lines,
        modes,
        workers=workers,
        timeout=timeout,
        gpus=gpus,
        sanitizer=sanitizer,
        engine_extra=extra,
        accuracy_mode=getattr(opts, "accuracy_mode", None) or info.get("accuracy_mode") or "auto",
    )
    return _write_and_launch(
        f"{info['run_id']}/{part}",
        pdir,
        inputs,
        final_lines,
        jobs,
        modes=modes,
        python=python,
        gpus=gpus,
        sanitizer=sanitizer,
        model=getattr(opts, "model", None) or info.get("model") or "local",
        card=getattr(opts, "card", None)
        or info.get("card")
        or os.environ.get("APITEST_CARD", "").strip(),
        auto_report=True,
        foreground=bool(getattr(opts, "foreground", False)),
        dry_run=bool(getattr(opts, "dry_run", False)),
        extra={
            "campaign_run_id": info["run_id"],
            "part": part,
            "heartbeat_interval": info.get("heartbeat_interval") or lib.HEARTBEAT_INTERVAL,
            "stale_seconds": stale,
            "force_lock": force,
        },
    )


def cmd_run_part(args) -> int:
    """在共享盘 campaign 上认领并运行一个 part。"""
    campaign_dir = lib.find_run_dir(args.run_id)
    # load_campaign 在 campaign.json 缺失时直接退出，避免把单机 run 当成 campaign 跑。
    info = campaign.load_campaign(campaign_dir)
    campaign.part_inputs(campaign_dir, info, args.part)  # 提前校验 part 是否存在
    return _run_part_impl(campaign_dir, info, args.part, args)


def _worker_argv(args) -> list[str]:
    # 只转发显式给出的选项，未给出的一律留空，让子进程继续沿用 campaign 的默认值。
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--run-id",
        args.run_id,
        "--max-parts",
        str(args.max_parts),
    ]
    for name in ("mode", "model", "card", "python", "gpus", "accuracy_mode", "sanitizer"):
        value = getattr(args, name, None)
        if value:
            cmd += [f"--{name.replace('_', '-')}", str(value)]
    # workers/timeout 用 is not None 判断，避免合法的 0 被当成未设置。
    for name in ("workers", "timeout"):
        value = getattr(args, name, None)
        if value is not None:
            cmd += [f"--{name}", str(value)]
    for extra in getattr(args, "engine_arg", None) or []:
        cmd += ["--engine-arg", extra]
    # 布尔开关用 store_true，出现即为真，未出现时不传以免覆盖 campaign 配置。
    for flag in ("force", "skip_doctor", "rerun_failed"):
        if getattr(args, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    return cmd


def cmd_worker(args) -> int:
    """一台机器上持续认领空闲 part，直到没有可跑的 part。"""
    campaign_dir = lib.find_run_dir(args.run_id)
    campaign.load_campaign(campaign_dir)  # 提前报错
    stop_flag = campaign_dir / "STOP"
    # 只删标记不碰进程，这一步不需要互斥锁，多机同时启动 worker 也不会互相破坏。
    stop_flag.unlink(missing_ok=True)  # 重新启动 worker 视为继续
    if args.foreground:
        return _worker_loop(args)
    # 日志名带主机名和时间戳，多机共享同一目录时不会互相覆盖。
    log_path = campaign_dir / f"worker_{lib.hostname()}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            _worker_argv(args),
            cwd=str(lib.PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"worker 已启动 PID={proc.pid}，日志：{log_path}")
    print(f"停止：st.py stop --run-id {args.run_id}（会写 STOP 标记，worker 跑完当前 part 退出）")
    return 0


def _worker_loop(args) -> int:
    campaign_dir = lib.find_run_dir(args.run_id)
    info = campaign.load_campaign(campaign_dir)
    # worker 必须在本进程里顺序跑 part，才能真正持有 part 锁和心跳。
    args.foreground = True
    tried: set[str] = set()
    ran = 0
    while ran < args.max_parts:
        # 只检查不删除 STOP：多机共享同一标记，谁删掉都会让其他机器继续认领。
        if (campaign_dir / "STOP").exists():
            print("[worker] 检测到 STOP 标记，退出")
            break
        # exclude 传入本进程试过的 part，避免抢锁失败后在同一 part 上死循环。
        part = campaign.next_available_part(
            campaign_dir, info, rerun_failed=args.rerun_failed, exclude=tried
        )
        # 没有可认领的 part 说明其余都在跑或已完成，退出把资源让给其他机器。
        if not part:
            print("[worker] 没有可跑的 part，退出")
            break
        # dry-run 只走一轮打印就退出，否则会反复打印同一个 part。
        if args.dry_run:
            _run_part_impl(campaign_dir, info, part, args)
            break
        print(f"[worker] 认领 {part}")
        # 抢锁失败等系统级退出不算跑完一个 part：记入 tried 后换下一个，避免死循环。
        try:
            rc = _run_part_impl(campaign_dir, info, part, args)
        except SystemExit as err:
            print(f"[worker] {part} 跳过：{err}")
            tried.add(part)
            continue
        tried.add(part)
        ran += 1
        print(f"[worker] {part} 结束 rc={rc}")
    return 0


def _update_job_state(run_dir: Path, key: str, **fields) -> None:
    # 读改写全程走原子替换：status 随时可能在另一台机器上读取这份 state.json。
    path = run_dir / "state.json"
    state = lib.load_json(path, {"jobs": {}})
    state.setdefault("jobs", {}).setdefault(key, {}).update(fields)
    # 记录最后一次写入的主机：汇总时据此区分本机 PID 可查与只能看心跳。
    state["host"] = lib.hostname()
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(path, state)


def _clear_supervisor(run_dir: Path) -> None:
    """supervisor 退出前清掉 PID，避免 status 把复用后的 PID 误判成运行中。"""
    # PID 会被系统回收复用，退出时置空比留下旧值更安全。
    path = run_dir / "state.json"
    state = lib.load_json(path, {"jobs": {}})
    state["supervisor_pid"] = None
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(path, state)


def _supervise(run_id: str, run_dir: Path, auto_report: bool = True) -> int:
    """顺序执行 jobs；engineV4 自带 checkpoint，重跑同一 log_dir 即续跑。"""
    run = lib.load_json(run_dir / "run.json", {})
    # run.json 是唯一权威参数来源，缺失时无法推断该跑什么，直接失败。
    if not run:
        raise SystemExit(f"未找到 run.json: {run_dir}")
    part = str(run.get("part") or "")
    # 单机 run 没有 campaign_run_id，此时 campaign_dir 为 None，收尾跳过 campaign 汇总。
    campaign_run_id = str(run.get("campaign_run_id") or "")
    campaign_dir = lib.find_run_dir(campaign_run_id) if campaign_run_id else None
    lock = None
    heartbeat = None
    if part:
        # 真正跑测试的进程持有锁和心跳；父进程退出不影响。
        lock = campaign.acquire_part_lock(
            run_dir,
            force=bool(run.get("force_lock")),
            stale_seconds=int(run.get("stale_seconds") or lib.STALE_SECONDS),
        )
        # 心跳间隔取自 run.json，保证同一 campaign 下所有 part 的心跳节奏一致。
        heartbeat = campaign.Heartbeat(
            run_dir,
            campaign_run_id or run_id,
            part,
            interval=int(run.get("heartbeat_interval") or lib.HEARTBEAT_INTERVAL),
        )
        heartbeat.start()

    state_path = run_dir / "state.json"
    state = lib.load_json(state_path, {"jobs": {}})
    # 覆盖成当前进程 PID：resume 或接管之后，旧 PID 已经不代表这次的 supervisor。
    state["supervisor_pid"] = os.getpid()
    state["host"] = lib.hostname()
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(state_path, state)

    # 汇总所有 job 的退出码：任何一个非 0 就整体非 0，方便 CI 直接判断。
    rc = 0
    try:
        for job in run.get("jobs", []):
            key = job["key"]
            entry = (state.get("jobs") or {}).get(key, {})
            # 只有明确成功才跳过；中断/失败/未开始的 job 一律重跑，已完成的 case 由 checkpoint 跳过。
            if entry.get("exit_code") == 0:
                _log(f"{key} 已完成，跳过")
                continue
            # 让心跳带上当前 job，远端看 status 时能知道卡在哪一步。
            if heartbeat:
                heartbeat.set_job(key)
            input_path = run_dir / job["input"]
            log_dir = run_dir / job["log_dir"]
            # 输入缺失通常意味着 run 目录被手工清理过，记 rc=1 但继续跑其他 job。
            if not input_path.exists():
                _log(f"{key} 缺少输入 {input_path}，跳过")
                rc = 1
                continue
            log_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                run.get("python", "python"),
                "engineV4.py",
                f"--api_config_file={input_path}",
                f"--log_dir={log_dir}",
                *job.get("engine_args", []),
            ]
            env = os.environ.copy()
            env.update(job.get("env", {}))
            # 进程级 stdout 单独落文件：engine 的 log_dir 只放 case 粒度日志。
            stdout_path = run_dir / "logs" / f"{key}.stdout.log"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            _log(f"启动 {key}：{shlex.join(cmd)}")
            # 独立会话启动，stop 时按进程组发信号可以连 engine 的子 worker 一起收掉。
            with stdout_path.open("a", encoding="utf-8") as out:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(lib.PROJECT_ROOT),
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                # 先落 running 状态再 wait，status 立刻就能看到 PID 与开始时间。
                _update_job_state(
                    run_dir,
                    key,
                    status="running",
                    pid=proc.pid,
                    started_at=lib.now_str(),
                    exit_code=None,
                    cmd=cmd,
                )
                # 前台模式下 Ctrl-C 只打断 supervisor，必须显式收掉 engine 进程组。
                try:
                    code = proc.wait()
                except KeyboardInterrupt:
                    _terminate_group(proc.pid)
                    proc.wait()
                    _update_job_state(
                        run_dir, key, status="中断", finished_at=lib.now_str(), exit_code=-1
                    )
                    raise
            _log(f"{key} 结束，exit={code}")
            # 退出码写进 state 之后对 status 才有意义，它据此区分结束与中断。
            _update_job_state(
                run_dir,
                key,
                status="已完成" if code == 0 else "结束",
                finished_at=lib.now_str(),
                exit_code=code,
            )
            report.build_status(run_id, run_dir)
            # 继续跑后续 job：一个配置集失败不应阻塞另一个的进度。
            if code != 0:
                rc = code

        # 报告要在 supervisor PID 清空之后生成，否则 status 会显示"运行中"。
        # 清空 supervisor PID 是幂等的：正常退出与 finally 兜底都会走到这里。
        _clear_supervisor(run_dir)
        if auto_report:
            _log("生成进度汇总与问题索引 ...")
            # 三个产物有依赖顺序：issues 依赖分类文件，repro 依赖 issues 的结论。
            report.build_status(run_id, run_dir)
            report.collect_issues(run_id, run_dir)
            report.build_repro(run_id, run_dir)
            _log(
                f"报告目录：{run_dir}/status.md、{run_dir}/issues/INDEX.md、{run_dir}/repro/README.md"
            )
            # 知道卡型时顺手生成 debugH/debugB；不知道就留给 debug 子命令补。
            card = (str(run.get("card") or "") or os.environ.get("APITEST_CARD", "")).strip()
            try:
                bundle = (
                    debug_bundle.build_debug_bundle(run_id, run_dir, card=card) if card else None
                )
            except Exception as err:  # 收尾报告不能影响主流程退出码
                _log(f"debug 复现包生成失败：{err}")
                bundle = None
            # bundle 为 None 表示没有真实失败，属于正常情况，不当错误处理。
            if bundle:
                _log(f"debug 复现包：{bundle['path']}/README.md")
            # campaign 下 debug 需要整个 campaign 的 run_id，提示里给出正确的续跑命令。
            else:
                hint = campaign_run_id or run_id
                _log(
                    "未生成 debugH/debugB（无失败配置或未指定卡型）；可运行 "
                    f"`st.py debug --run-id {hint} --card H` 补生成"
                )
    finally:
        # 收尾顺序固定：先停心跳、再释放锁、最后清 PID，避免别人抢到锁却看到旧心跳。
        if heartbeat:
            heartbeat.finish("done" if rc == 0 else "failed")
        if lock:
            campaign.release_part_lock(lock)
        _clear_supervisor(run_dir)
        # part 收尾后刷新一次 campaign 视图，便于其他机器立即看到汇总。
        if campaign_dir and campaign.is_campaign(campaign_dir):
            try:
                campaign.campaign_status(campaign_dir, campaign.load_campaign(campaign_dir))
            except Exception as err:
                _log(f"campaign 汇总刷新失败：{err}")
    return rc


def _terminate_group(pid: int, sig=signal.SIGTERM) -> None:
    # 先按进程组发信号，让 engine 派生的 worker 一起退出；组是否存在交给异常分支判断。
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        # 组不存在时退化为单 PID，覆盖 getpgid 失败但进程本身仍存活的情况。
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            # 进程已退出或属主不同：再试也没有意义，最终由调用方的存活检查决定是否补 SIGKILL。
            pass


def cmd_resume(args) -> int:
    run_dir = lib.find_run_dir(args.run_id)
    # campaign 的续跑单位是 part，必须走 run-part 才会重新抢锁并写心跳。
    if campaign.is_campaign(run_dir):
        raise SystemExit(
            f"{args.run_id} 是 campaign，请用 run-part 续跑："
            f"st.py run-part --run-id {args.run_id} --part <part>"
        )
    run = lib.load_json(run_dir / "run.json", {})
    if not run:
        raise SystemExit(f"未找到运行记录: {run_dir}")
    state = lib.load_json(run_dir / "state.json", {})
    # PID 仍存活时不重复启动，否则两个 supervisor 会写同一批 log_dir。
    if lib.pid_alive(state.get("supervisor_pid")):
        raise SystemExit(f"supervisor 仍在运行 PID={state['supervisor_pid']}，不能重复启动")
    if args.foreground:
        return _supervise(args.run_id, run_dir, auto_report=run.get("auto_report", True))
    log_path = run_dir / "supervisor.log"
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_supervise", "--run-id", args.run_id],
            cwd=str(lib.PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["supervisor_pid"] = proc.pid
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(run_dir / "state.json", state)
    print(f"已续跑 supervisor PID={proc.pid}")
    return 0


def cmd_parts(args) -> int:
    """列出 campaign 的所有 part：状态、host、心跳、是否可认领。"""
    run_dir = lib.find_run_dir(args.run_id)
    info = campaign.load_campaign(run_dir)
    # 未指定 --stale 时由 parts_table 回退到创建时的阈值，保持跨机器判断一致。
    rows = campaign.parts_table(run_dir, info, getattr(args, "stale", None))
    payload = {"run_id": info["run_id"], "parts": rows, "stop": (run_dir / "STOP").exists()}
    # JSON 输出给脚本消费，不掺任何提示文案。
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"# campaign {info['run_id']} 的 part")
    if payload["stop"]:
        print(
            "\n[WARN] STOP 标记存在；worker 会在当前 part 结束后退出。重新启动 worker 会清除该标记。"
        )
    print("\n| part | shapes | 配置数 | 状态 | host | 心跳 | 可认领 |")
    print("|---|---|---|---|---|---|---|")
    available = []
    for row in rows:
        shapes = ", ".join(row.get("shapes", {}).keys())
        print(
            f"| {row['id']} | {shapes} | {row.get('configs', 0)} | {row['status']} | "
            f"{row.get('host') or '-'} | {row.get('heartbeat') or '-'} | "
            f"{'是' if row.get('available') else '否'} |"
        )
        if row.get("available"):
            available.append(row["id"])
    if available:
        print(f"\n可认领：{', '.join(available)}")
        print(f"单 part：st.py run-part --run-id {info['run_id']} --part {available[0]}")
        print(f"自动连续跑：st.py worker --run-id {info['run_id']}")
    # 没有可认领 part 也可能是全部在跑，据此提示而不是报错。
    else:
        print("\n当前没有可认领的 part。")
    return 0


def cmd_verify(args) -> int:
    """校验 campaign 的切分守恒与 case 覆盖率。"""
    run_dir = lib.find_run_dir(args.run_id)
    info = campaign.load_campaign(run_dir)
    # 加聚合锁：verify 会重建 verify/ 目录，两机同时执行会删掉对方的中间产物。
    with campaign.aggregate_lock(run_dir):
        payload = campaign.verify_campaign(run_dir, info, getattr(args, "stale", None))
    print((run_dir / "verify" / "verify.md").read_text(encoding="utf-8"))
    if not payload["complete"]:
        print(
            f"不完整：缺失 {payload['missing_total']} 条，未启动 job {payload['unstarted_jobs']} 个"
        )
        # 默认非 0 让 CI 能卡住；确认缺失可接受时用 --allow-incomplete 放行。
        return 0 if args.allow_incomplete else 1
    print("完整性校验通过。")
    return 0


def cmd_finalize(args) -> int:
    """所有 part 跑完后一次性生成 status / issues / repro / debugH。"""
    run_dir = lib.find_run_dir(args.run_id)
    info = campaign.load_campaign(run_dir)
    # finalize 会重写 status/issues/debug 产物，必须独占整个流程。
    with campaign.aggregate_lock(run_dir):
        verify = campaign.verify_campaign(run_dir, info, getattr(args, "stale", None))
        # 先打印 verify.md 再退出，用户能直接看到缺在哪个 part/shape。
        if not verify["complete"] and not args.allow_incomplete:
            print((run_dir / "verify" / "verify.md").read_text(encoding="utf-8"))
            print("存在缺失配置，未生成最终报告；确认可接受后加 --allow-incomplete。")
            return 1
        status = campaign.campaign_status(run_dir, info, getattr(args, "stale", None))
        campaign.campaign_collect(run_dir, info)
        # debug 包跨 part 合并，必须走 campaign 级入口而不是逐 part 的 debug_bundle。
        bundle = campaign.campaign_debug(
            run_dir,
            info,
            card=args.card,
            model=args.model,
            out_dir=Path(args.out).expanduser() if args.out else None,
        )
        # 逐 part 复现包体积大且只在排查时需要，默认不生成。
        if args.repro:
            campaign.campaign_repro(run_dir, info)
    print((run_dir / "status.md").read_text(encoding="utf-8"))
    print(f"\n问题索引：{run_dir}/issues/INDEX.md")
    if bundle:
        print(f"debug 复现包：{bundle['path']}/README.md")
    else:
        print("debug 复现包：本次没有失败配置")
    verify_text = "通过" if verify["complete"] else f"缺失 {verify['missing_total']} 条"
    print(f"完整性：{verify_text}")
    return 0


def cmd_reset_part(args) -> int:
    """清理失联 part 的锁/心跳；--clean 连日志一起删。"""
    run_dir = lib.find_run_dir(args.run_id)
    info = campaign.load_campaign(run_dir)
    # reset 只动锁/心跳（或整个 part 目录），用于掉线机回收，不会自动重跑。
    print(
        campaign.reset_part(
            run_dir,
            info,
            args.part,
            clean=args.clean,
            force=args.force,
        )
    )
    return 0


def cmd_status(args) -> int:
    run_dir = lib.find_run_dir(args.run_id)
    # 目录不存在通常是 run_id 打错，直接报出来比后续 KeyError 更明确。
    if not run_dir.is_dir():
        raise SystemExit(f"未找到运行目录: {run_dir}")
    is_camp = campaign.is_campaign(run_dir)

    def once() -> dict:
        if is_camp:
            payload = campaign.campaign_status(
                run_dir, campaign.load_campaign(run_dir), getattr(args, "stale", None)
            )
        else:
            # 单机 run 与 campaign 共用同一套 status.md 渲染，只是数据来源不同。
            payload = report.build_status(args.run_id, run_dir)
        print((run_dir / "status.md").read_text(encoding="utf-8"))
        return payload

    if args.watch <= 0:
        payload = once()
    else:
        while True:
            # 无 TERM（重定向到文件或 CI）时不清屏，避免输出里混入 ANSI 控制字符。
            os.system("clear" if os.environ.get("TERM") else "")
            payload = once()
            # supervisor / 所有 part 都停下后就不会再推进，避免 watch 卡死。
            if not payload["supervisor_alive"]:
                break
            time.sleep(args.watch)
    # strict 只统计真实失败，oom/skip 等参考分类不算失败。
    if args.strict and payload["totals"]["fail"] > 0:
        return 1
    return 0


def cmd_collect(args) -> int:
    run_dir = lib.find_run_dir(args.run_id)
    if campaign.is_campaign(run_dir):
        # campaign 的 issues 是跨 part 合并产物，需要聚合锁串行化。
        with campaign.aggregate_lock(run_dir):
            payload = campaign.campaign_collect(run_dir, campaign.load_campaign(run_dir))
    else:
        payload = report.collect_issues(args.run_id, run_dir)
    print(f"问题索引：{run_dir}/issues/INDEX.md")
    # 逐 job 逐分类打印条数与日志覆盖：缺日志意味着失败现场已经丢失。
    for job in payload["jobs"]:
        for cat, stat in job["classes"].items():
            print(
                f"  {job['key']}/{cat}: {stat['count']} 条，"
                f"有日志 {stat['with_log']}，缺日志 {stat['missing_log']}"
            )
    print(f"错误签名 {len(payload['signatures'])} 组")
    return 0


def cmd_repro(args) -> int:
    run_dir = lib.find_run_dir(args.run_id)
    if campaign.is_campaign(run_dir):
        # repro 会先删后建 repro/ 目录，加锁避免两台机器同时生成时互相覆盖。
        with campaign.aggregate_lock(run_dir):
            payload = campaign.campaign_repro(run_dir, campaign.load_campaign(run_dir))
        # 逐 part 脚本留在各自 part 目录，顶层只放索引，避免跨机器混跑。
        print(f"campaign 复现索引：{run_dir}/repro/README.md")
        for row in payload["parts"]:
            print(f"  {row['part']}: {row['path']}")
        return 0
    payload = report.build_repro(args.run_id, run_dir)
    print(f"复现包：{run_dir}/repro/")
    for job in payload["jobs"]:
        print(
            f"  {job['job']}: 真实失败 {job['failures']} 条"
            + (f"，批量脚本 run_{job['job']}.sh" if job["config"] else "（无失败）")
        )
    print(f"说明：{run_dir}/repro/README.md")
    return 0


def cmd_debug(args) -> int:
    """生成 debugH / debugB 风格的复现包。"""
    run_dir = lib.find_run_dir(args.run_id)
    if campaign.is_campaign(run_dir):
        with campaign.aggregate_lock(run_dir):
            # 卡型决定 debugH/debugB 的目录名与命令模板，未给时由 debug_bundle 回退到 run.json。
            bundle = campaign.campaign_debug(
                run_dir,
                campaign.load_campaign(run_dir),
                card=args.card,
                model=args.model,
                out_dir=Path(args.out).expanduser() if args.out else None,
            )
    else:
        bundle = debug_bundle.build_debug_bundle(
            args.run_id,
            run_dir,
            card=args.card,
            model=args.model,
            out_dir=Path(args.out).expanduser() if args.out else None,
        )
    # 没有真实失败时 bundle 为 None，这不是错误，返回 0 便于脚本串联。
    if not bundle:
        print("本次运行没有可整理的真实失败/参考分类配置，未生成 debug 包。")
        return 0
    print(f"debug 复现包：{bundle['path']}/")
    print(
        f"  卡型={bundle['card']}，模型={bundle['model']}"
        + ("（跨 part 合并）" if bundle.get("campaign") else "")
    )
    for item in bundle["summaries"]:
        print(
            f"  {item['slot']}: 真实失败 {item['fail']} 条，"
            f"有日志 {item['with_log']}，无日志 {item['missing_log']}"
        )
    print(f"说明：{bundle['path']}/README.md")
    return 0


def _stop_part(pdir: Path) -> int:
    """停掉一个本机 part 的进程组；返回杀掉的 PID 数。"""
    state = lib.load_json(pdir / "state.json", {})
    jobs = state.get("jobs") or {}
    # 只处理此刻仍存活的 PID：state 里可能残留早已退出的记录。
    pids = [entry.get("pid") for entry in jobs.values() if lib.pid_alive(entry.get("pid"))]
    supervisor = state.get("supervisor_pid")
    for pid in pids:
        _terminate_group(int(pid))
    if lib.pid_alive(supervisor):
        _terminate_group(int(supervisor))
    # 先 SIGTERM 再等 10 秒：给 engine 落 checkpoint 的机会，超时才升级为 SIGKILL。
    deadline = time.time() + 10
    while time.time() < deadline and any(lib.pid_alive(p) for p in pids + [supervisor]):
        time.sleep(0.2)
    for pid in pids + [supervisor]:
        if lib.pid_alive(pid):
            _terminate_group(int(pid), signal.SIGKILL)
    # 只把真正被杀掉的 job 标成已终止，未运行的 job 保持原状态供下次续跑。
    for entry in jobs.values():
        if entry.get("pid") in pids:
            entry["status"] = "已终止"
            entry["finished_at"] = lib.now_str()
    state["supervisor_pid"] = None
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(pdir / "state.json", state)
    # 心跳改成 stopped，其他机器立刻能看出该 part 不再运行。
    hb_path = pdir / campaign.HEARTBEAT_FILE
    hb = lib.load_json(hb_path, {})
    if hb:
        # 保留 host 等字段只改状态与时间戳：远端 status 依赖 host 决定该找谁停。
        hb.update(
            {"status": "stopped", "heartbeat_ts": lib.now_epoch(), "heartbeat_at": lib.now_str()}
        )
        lib.atomic_write_json(hb_path, hb)
    return len(pids)


def cmd_stop(args) -> int:
    run_dir = lib.find_run_dir(args.run_id)
    if campaign.is_campaign(run_dir):
        info = campaign.load_campaign(run_dir)
        part = getattr(args, "part", None)
        if part:
            pdir = campaign.part_dir(run_dir, part)
            # part id 打错时返回 2（用法错误），与没有东西可停的 0 区分。
            if not pdir.exists():
                print(f"part 不存在: {part}")
                return 2
            hb = lib.load_json(pdir / campaign.HEARTBEAT_FILE, {})
            # 跨机器的进程杀不掉：只提示需要去哪台机器执行，不假装成功。
            if hb.get("host") and hb.get("host") != lib.hostname():
                print(f"{part} 在 {hb.get('host')} 上运行，需在对应机器停止。")
                return 2
            n = _stop_part(pdir)
            print(f"已停止 {part} 的本机 job {n} 个。")
            return 0
        # 整 campaign 停止：写 STOP 让 worker 跑完当前 part 后退出。
        (run_dir / "STOP").write_text(lib.now_str() + "\n", encoding="utf-8")
        local, remote = 0, 0
        for part in info.get("parts", []):
            pdir = campaign.part_dir(run_dir, part["id"])
            hb = lib.load_json(pdir / campaign.HEARTBEAT_FILE, {})
            # 远端 part 无法在本机终止，计数后在提示里说明需要手工处理。
            if hb and hb.get("host") != lib.hostname():
                remote += 1
                continue
            local += _stop_part(pdir)
        print(
            f"已停止本机 part 的 job {local} 个；{remote} 个 part 在别的机器上，需在对应机器执行 stop。"
        )
        return 0
    state = lib.load_json(run_dir / "state.json", {})
    jobs = state.get("jobs") or {}
    # 先记录操作发生时仍存活的 job；kill 之后就查不到它们了。
    live_keys = [key for key, entry in jobs.items() if lib.pid_alive(entry.get("pid"))]
    pids = [jobs[key].get("pid") for key in live_keys]
    supervisor = state.get("supervisor_pid")
    for pid in pids:
        _terminate_group(int(pid))
    if lib.pid_alive(supervisor):
        _terminate_group(int(supervisor))
    deadline = time.time() + 10
    while time.time() < deadline and any(lib.pid_alive(p) for p in pids + [supervisor]):
        time.sleep(0.2)
    # 宽限期后仍存活说明进程忽略了 SIGTERM（多半卡在 CUDA 调用里），直接 SIGKILL。
    for pid in pids + [supervisor]:
        if lib.pid_alive(pid):
            _terminate_group(int(pid), signal.SIGKILL)
    for key in live_keys:
        jobs[key]["status"] = "已终止"
        jobs[key]["finished_at"] = lib.now_str()
    state["supervisor_pid"] = None
    state["updated_at"] = lib.now_str()
    lib.atomic_write_json(run_dir / "state.json", state)
    print(f"已停止运行 {args.run_id}（job {len(pids)} 个，supervisor PID={supervisor}）")
    return 0


def _cmd_supervise(args) -> int:
    # --run-dir 用于 campaign part：run_id 形如 campaign/part，不能直接拼到 runs 下。
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else lib.find_run_dir(args.run_id)
    # 后台重启（resume/run-part）沿用 run.json 里的 auto_report 设置。
    auto = lib.load_json(run_dir / "run.json", {}).get("auto_report", True)
    return _supervise(args.run_id, run_dir, auto_report=auto)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PaddleAPITest 0size/1M 配置运行器：启动、进度、问题与复现包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
        "  python shape_test_runner/st.py start --0size 0size.txt --1m 1M.txt\n"
        "  python shape_test_runner/st.py start --1m 1M.txt --mode accuracy\n"
        "  python shape_test_runner/st.py status --run-id run_20261120_101010 --watch 30\n",
    )
    # 子命令即协议：每个命令只做一件可独立重试的事，失败不留下半成品状态。
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor：只读体检；退出码 2 表示环境阻断，便于包装脚本区分。
    p = sub.add_parser("doctor", help="只检查环境，不安装任何东西")
    p.add_argument("--python", default="python")
    p.set_defaults(func=cmd_doctor)

    # start：单机一次跑完 shape 与 mode 的组合，与 run-part 共享 job 组装逻辑。
    p = sub.add_parser("start", help="校验输入并启动测试")
    p.add_argument("--0size", dest="size0", help="0size 配置 txt")
    p.add_argument("--1m", dest="size1m", help="1M 配置 txt")
    p.add_argument("--shapes", help="只测指定 shape：0size,1M")
    p.add_argument(
        "--mode",
        default="accuracy,paddleonly",
        help="accuracy,paddleonly,stable 的逗号组合（默认 accuracy,paddleonly）",
    )
    p.add_argument("--run-id", help="运行标识，默认按时间戳生成")
    p.add_argument("--python", default="python", help="运行测试的 Python（默认 PATH 里的 python）")
    p.add_argument("--model", help="模型名，用于 debugH/debugB 的模型目录（默认 local）")
    p.add_argument("--card", help="卡型 H/B/GB 等，用于生成 debugH/debugB（默认读 APITEST_CARD）")
    # 缺省 -1 表示使用 engineV4 探测到的全部可见 GPU。
    p.add_argument("--gpus", default="-1", help="GPU id，如 0,1 或 0-3；默认全部")
    p.add_argument("--workers", type=int, help="每卡 worker 数（默认 0size=4，1M=1）")
    p.add_argument("--timeout", type=int, help="单 case 超时秒数（默认 0size=600，1M=1200）")
    # accuracy 主模式只在 --mode 含 accuracy 时生效；dual 需要两张卡分别跑 paddle 与 torch。
    p.add_argument(
        "--accuracy-mode",
        choices=("auto", "accuracy", "dual"),
        default="auto",
        help="accuracy 主模式：auto=0size 用 accuracy、1M 用 dual（与 pipeline 一致）",
    )
    p.add_argument(
        "--sanitizer",
        choices=("auto", "on", "off"),
        default="auto",
        help="paddleonly 是否启用 compute-sanitizer；auto=存在则启用",
    )
    p.add_argument("--engine-arg", action="append", default=[], help="额外透传参数，可重复")
    # 去重只影响写进运行目录的副本，不改动源文件。
    p.add_argument("--dedup", action="store_true", help="输入有重复行时使用去重副本")
    p.add_argument("--skip-doctor", action="store_true", help="跳过环境体检")
    p.add_argument("--foreground", action="store_true", help="前台运行（调试用）")
    p.add_argument("--resume", action="store_true", help="复用已存在的 run-id 并续跑")
    p.add_argument(
        "--auto-report",
        dest="auto_report",
        action="store_true",
        default=True,
        help="结束后自动生成 status/issues/repro（默认开启）",
    )
    p.add_argument("--no-auto-report", dest="auto_report", action="store_false")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    p.set_defaults(func=cmd_start)

    # _supervise：内部命令，由 start/resume 以独立会话拉起，参数刻意保持精简。
    p = sub.add_parser("_supervise")
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-dir", help="实际运行目录（campaign part 用）")
    p.set_defaults(func=_cmd_supervise)

    # split：只切片不执行，产出的 campaign 可被多台机器共享挂载。
    p = sub.add_parser("split", help="把 txt 切成多个 part，生成多机 campaign")
    p.add_argument("--0size", dest="size0", help="0size 配置 txt")
    p.add_argument("--1m", dest="size1m", help="1M 配置 txt")
    p.add_argument("--shapes", help="只切指定 shape：0size,1M")
    p.add_argument("--shards", type=int, required=True, help="切成的 part 数")
    p.add_argument(
        "--strategy",
        choices=("round-robin", "contiguous"),
        default="round-robin",
        help="切分策略（默认 round-robin，负载更均匀）",
    )
    p.add_argument("--run-id", help="campaign 标识，默认按时间戳生成")
    p.add_argument(
        "--mode", default="accuracy,paddleonly", help="accuracy,paddleonly,stable 的逗号组合"
    )
    p.add_argument("--model", help="模型名，用于 debugH/debugB 的模型目录（默认 local）")
    p.add_argument("--card", help="卡型 H/B/GB 等（默认读 APITEST_CARD）")
    p.add_argument("--python", help="各机器运行测试的 Python（默认 python）")
    p.add_argument("--gpus", help="默认 GPU id（默认全部）")
    p.add_argument("--workers", type=int, help="默认每卡 worker 数")
    p.add_argument("--timeout", type=int, help="默认单 case 超时秒数")
    p.add_argument("--accuracy-mode", choices=("auto", "accuracy", "dual"), default="auto")
    p.add_argument("--sanitizer", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--engine-arg", action="append", default=[], help="额外透传参数，可重复")
    p.add_argument(
        "--stale", type=int, default=lib.STALE_SECONDS, help="心跳失联阈值秒数（默认 120）"
    )
    p.add_argument("--dedup", action="store_true", help="切分前对输入去重")
    p.add_argument("--force", action="store_true", help="覆盖已存在的 campaign 目录")
    p.set_defaults(func=cmd_split)

    # run-part：单次认领；这里的覆盖项只影响本次执行，不回写 campaign.json。
    p = sub.add_parser("run-part", help="在共享盘 campaign 上认领并运行一个 part")
    p.add_argument("--run-id", required=True, help="campaign 标识")
    p.add_argument("--part", required=True, help="part id，如 part-00")
    p.add_argument("--mode", help="覆盖 campaign 的 mode 组合")
    p.add_argument("--model", help="覆盖模型目录名")
    p.add_argument("--card", help="覆盖卡型")
    p.add_argument("--python", help="覆盖 Python")
    p.add_argument("--gpus", help="覆盖 GPU id")
    p.add_argument("--workers", type=int, help="覆盖每卡 worker 数")
    p.add_argument("--timeout", type=int, help="覆盖单 case 超时")
    p.add_argument(
        "--accuracy-mode", choices=("auto", "accuracy", "dual"), help="覆盖 accuracy 主模式"
    )
    p.add_argument("--sanitizer", choices=("auto", "on", "off"), help="覆盖 sanitizer 策略")
    p.add_argument("--engine-arg", action="append", default=[], help="追加透传参数，可重复")
    p.add_argument("--force", action="store_true", help="接管失联 part 或忽略运行中的告警")
    p.add_argument("--skip-doctor", action="store_true", help="跳过环境体检")
    p.add_argument("--foreground", action="store_true", help="前台运行（调试用）")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    p.set_defaults(func=cmd_run_part)

    # parts：同一份数据同时提供表格与 --json，供人看和供脚本消费。
    p = sub.add_parser("parts", help="列出 campaign 的 part 状态与可认领项")
    p.add_argument("--run-id", required=True)
    p.add_argument("--stale", type=int, help="心跳失联阈值秒数")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.set_defaults(func=cmd_parts)

    # worker：本机循环认领；_worker 是同一逻辑的后台子进程入口。
    p = sub.add_parser("worker", help="自动认领并连续运行空闲 part")
    p.add_argument("--run-id", required=True)
    # 默认上限取远大于实际 part 数的值，语义是直到没有空闲 part 为止。
    p.add_argument(
        "--max-parts",
        type=int,
        default=1000000,
        help="最多连续跑多少个 part（默认直到没有空闲 part）",
    )
    p.add_argument("--mode", help="覆盖 campaign 的 mode 组合")
    p.add_argument("--model", help="覆盖模型目录名")
    p.add_argument("--card", help="覆盖卡型")
    p.add_argument("--python", help="覆盖 Python")
    p.add_argument("--gpus", help="覆盖 GPU id")
    p.add_argument("--workers", type=int, help="覆盖每卡 worker 数")
    p.add_argument("--timeout", type=int, help="覆盖单 case 超时")
    p.add_argument("--accuracy-mode", choices=("auto", "accuracy", "dual"))
    p.add_argument("--sanitizer", choices=("auto", "on", "off"))
    p.add_argument("--engine-arg", action="append", default=[], help="追加透传参数")
    p.add_argument("--force", action="store_true", help="接管失联/运行中的 part")
    p.add_argument("--skip-doctor", action="store_true")
    p.add_argument("--rerun-failed", action="store_true", help="连已结束但非 0 的 part 也重跑")
    p.add_argument("--foreground", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_worker)

    # _worker：由 worker 拉起，参数是父进程参数的转写结果，不额外解析。
    p = sub.add_parser("_worker")
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--max-parts",
        type=int,
        default=1000000,
        help="最多连续跑多少个 part（默认直到没有空闲 part）",
    )
    p.add_argument("--mode", help="覆盖 campaign 的 mode 组合")
    p.add_argument("--model", help="覆盖模型目录名")
    p.add_argument("--card", help="覆盖卡型")
    p.add_argument("--python", help="覆盖 Python")
    p.add_argument("--gpus", help="覆盖 GPU id")
    p.add_argument("--workers", type=int, help="覆盖每卡 worker 数")
    p.add_argument("--timeout", type=int, help="覆盖单 case 超时")
    p.add_argument("--accuracy-mode", choices=("auto", "accuracy", "dual"))
    p.add_argument("--sanitizer", choices=("auto", "on", "off"))
    p.add_argument("--engine-arg", action="append", default=[], help="追加透传参数")
    p.add_argument("--force", action="store_true", help="接管失联/运行中的 part")
    p.add_argument("--skip-doctor", action="store_true")
    p.add_argument("--rerun-failed", action="store_true", help="连已结束但非 0 的 part 也重跑")
    p.add_argument("--foreground", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_worker_loop)

    # verify：只读校验，不写测试状态，可以随时重复执行。
    p = sub.add_parser("verify", help="校验切分守恒与 case 覆盖率")
    p.add_argument("--run-id", required=True)
    p.add_argument("--stale", type=int)
    p.add_argument("--allow-incomplete", action="store_true", help="不完整也返回 0")
    p.set_defaults(func=cmd_verify)

    # finalize：跨 part 汇总，比 verify 多做一步产物生成。
    p = sub.add_parser("finalize", help="全齐后一次性生成 status/issues/debug")
    p.add_argument("--run-id", required=True)
    p.add_argument("--card", help="卡型 H/B/GB 等")
    p.add_argument("--model", help="模型目录名")
    p.add_argument("--out", help="debug 包输出父目录")
    p.add_argument("--stale", type=int)
    p.add_argument("--allow-incomplete", action="store_true", help="有缺失也生成报告")
    p.add_argument("--repro", action="store_true", help="同时生成逐 part 复现包")
    p.set_defaults(func=cmd_finalize)

    # reset-part：掉线机回收，不启动测试；--clean 会删日志且不可恢复。
    p = sub.add_parser("reset-part", help="清理失联 part 的锁和心跳")
    p.add_argument("--run-id", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--clean", action="store_true", help="连日志/状态一起删，从头跑")
    p.add_argument("--force", action="store_true", help="part 仍在运行也强制清理")
    p.set_defaults(func=cmd_reset_part)

    # resume：复用已有 run 目录，跳过 exit_code 为 0 的 job。
    p = sub.add_parser("resume", help="续跑未完成的 job")
    p.add_argument("--run-id", required=True)
    p.add_argument("--foreground", action="store_true")
    p.set_defaults(func=cmd_resume)

    # status：默认打印一次；--watch 时循环刷新直到没有 supervisor 存活。
    p = sub.add_parser("status", help="查看进度")
    p.add_argument("--run-id", required=True)
    p.add_argument("--watch", type=int, default=0, help="每 N 秒刷新，0=只打印一次")
    p.add_argument("--stale", type=int, help="心跳失联阈值秒数；campaign 默认读创建时的值")
    p.add_argument("--strict", action="store_true", help="有真实失败时返回 1")
    p.set_defaults(func=cmd_status)

    # collect：只整理 issues，不改 state.json。
    p = sub.add_parser("collect", help="导出失败问题与单 case 日志")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_collect)

    # repro：基于 issues 结论生成逐 job 的批量复现脚本。
    p = sub.add_parser("repro", help="生成复现包")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_repro)

    # debug：可单独补生成 debugH/debugB，不依赖 auto_report。
    p = sub.add_parser("debug", help="生成 debugH/debugB 风格的失败复现包")
    p.add_argument("--run-id", required=True)
    p.add_argument("--card", help="卡型 H/B/GB 等；默认读 run.json / APITEST_CARD，否则 local")
    p.add_argument("--model", help="模型目录名；默认读 run.json，否则 local")
    p.add_argument("--out", help="输出父目录；默认 runs/<run_id>")
    p.set_defaults(func=cmd_debug)

    # stop：campaign 下写 STOP 标记；单机 run 直接杀进程组。
    p = sub.add_parser("stop", help="停止运行")
    p.add_argument("--run-id", required=True)
    p.add_argument("--part", help="campaign 下只停指定 part（仅本机）")
    p.set_defaults(func=cmd_stop)
    return parser


def main(argv=None) -> int:
    # argv 保留给测试注入，正常调用走 sys.argv。
    args = build_parser().parse_args(argv)
    # 退出码由子命令决定，main 只做透传。
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
