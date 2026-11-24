#!/usr/bin/env python3
"""生成 debugH / debugB 风格的失败复现包（本地版）。

与流水线里的 ``api_test/runtime/utils/debug_bundle.py`` 保持同一套口径和目录形态，
但输入换成 shape_test_runner 的 ``runs/<run_id>/``，不依赖 BOS、模型目录约定或
流水线环境变量。

产物::

    debug<卡型>/
    ├── README.md / INDEX.md
    ├── run_accuracy.sh / run_paddleonly.sh / run_stable.sh
    ├── run_case.sh
    └── <model>/
        ├── accuracy_0size.txt / accuracy_1M.txt ...
        ├── oom_accuracy_0size.txt / skip_*.txt / bitwise_*.txt
        └── pipeline_errors/<mode>_<shape>/
            ├── INDEX.md
            ├── <分类>__<case-id>.log
            ├── <分类>.all.log
            ├── missing_log.txt
            └── log_<timestamp>.log      # 该 job 的原始运行日志
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from lib import (
    CAT_ORDER,
    FAIL_CATS,
    REF_CATS,
    classify_job,
    job_log_dir,
    load_json,
    now_str,
    parse_case_blocks,
    pick_case_block,
)

# 键必须取自 lib.REF_CATS；值里的 {slot} 在写文件前才展开，同一 pattern 会被多个分类复用。
# 参考分类到顶层清单文件名的映射；bitwise 的两类合并成一个文件。
REF_FILES = {
    "oom": "oom_{slot}.txt",
    "skip": "skip_{slot}.txt",
    "paddle_bitwise": "bitwise_{slot}.txt",
    "paddle_bitwise_knows": "bitwise_{slot}.txt",
}

# 复现脚本里从自身目录向上寻找 engineV4.py，复制到别处也能直接跑。
# 只有探到含 engineV4.py 的目录才认作根；找不到时以退出码 2 报错，而不是在错误目录里跑。
# PADDLEAPITEST_ROOT 优先，便于复现包被拷到挂载路径不同的机器上执行。
_ROOT_SNIPPET = """ROOT="${PADDLEAPITEST_ROOT:-}"
if [ -z "${ROOT}" ]; then
  probe="${DIR}"
  while [ "${probe}" != "/" ] && [ ! -f "${probe}/engineV4.py" ]; do probe="$(dirname "${probe}")"; done
  ROOT="${probe}"
fi
[ -f "${ROOT}/engineV4.py" ] || { echo "找不到 PaddleAPITest 根目录，请设置 PADDLEAPITEST_ROOT"; exit 2; }
"""


def _write_shell(path: Path, text: str) -> None:
    # 复现脚本要求能直接 bash run_xxx.sh 执行，因此写完立即补上 0755。
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _slot(mode: str, shape: str) -> str:
    """配置/日志目录统一使用 accuracy_0size 这种顺序。"""
    # 顺序固定 <mode>_<shape>：README 里用 rsplit("_", 1) 反解 shape，颠倒会解析错。
    return f"{mode}_{shape}"


def _read_unique(path: Path) -> list[str]:
    # 读已有清单时去重并保序，重跑追加不会让参考清单无限膨胀。
    if not path.exists():
        return []
    seen, out = set(), []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _copy_run_logs(log_dir: Path, dest: Path, prefix: str = "") -> None:
    """复制 run 级原始日志（含环境信息和完整 traceback），跳过合并日志。

    多 part 合并时加前缀，避免不同机器的同名日志互相覆盖。
    """
    # 只匹配 log_*.log：.tmp/ 下的 worker 片段和合并日志 log_inorder.log 都不算 run 级日志，
    # 合并日志已在单 case 拆分时用过，重复打包只会让包体积翻倍。
    for path in sorted(log_dir.glob("log_*.log")):
        if path.name == "log_inorder.log":
            continue
        target = dest / (f"{prefix}__{path.name}" if prefix else path.name)
        shutil.copyfile(path, target)


def _write_slot(model_dir: Path, slot: str, sources: list[tuple[str, Path]], classes: dict) -> dict:
    """写单个 mode_shape 的失败配置、参考清单和 pipeline_errors。

    :param sources: [(label, log_dir)]；单机一个，campaign 是该 slot 跨所有 part 的日志。
    """
    # 失败清单按 FAIL_CATS 顺序拼接并去重：同一配置可能同时命中多个分类，
    # 复现批次里只应出现一次；顺序稳定也便于对比两次运行的清单差异。
    fail_configs = []
    for cat in FAIL_CATS:
        for line in classes.get(cat, []):
            if line not in fail_configs:
                fail_configs.append(line)

    # 参考清单是累积写入：跨 part 或跨批次重跑时保留磁盘上已有行，只追加本次新出现的配置。
    ref_counts = {}
    for cat, pattern in REF_FILES.items():
        lines = classes.get(cat, [])
        if not lines:
            continue
        path = model_dir / pattern.format(slot=slot)
        existing = _read_unique(path)
        merged = existing + [line for line in lines if line not in existing]
        path.write_text("\n".join(merged) + "\n", encoding="utf-8")
        ref_counts[cat] = len(lines)

    if fail_configs:
        (model_dir / f"{slot}.txt").write_text("\n".join(fail_configs) + "\n", encoding="utf-8")

    # 单机只有一个 source；campaign 时同一 slot 的所有 part 日志在此合并，
    # 后续 pick_case_block 才能取到"最后一次非 PASS"的那份块。
    blocks: dict[str, list[dict]] = {}
    for _label, log_dir in sources:
        for config, candidates in parse_case_blocks(log_dir).items():
            blocks.setdefault(config, []).extend(candidates)

    dest = model_dir / "pipeline_errors" / slot
    rows, missing = [], []
    for cat in CAT_ORDER:
        configs = classes.get(cat) or []
        if cat == "pass" or not configs:
            continue
        for config in configs:
            found = pick_case_block(blocks.get(config, []))
            if not found:
                missing.append((cat, config))
                continue
            dest.mkdir(parents=True, exist_ok=True)
            name = f"{cat}__{found['case_id']}.log"
            (dest / name).write_text(found["block"], encoding="utf-8")
            rows.append((cat, found["case_id"], name, config, found["block"]))
    if rows:
        by_cat = {}
        for cat, _cid, name, _cfg, block in rows:
            by_cat.setdefault(cat, []).append(block)
        for cat, chunks in by_cat.items():
            (dest / f"{cat}.all.log").write_text("".join(chunks), encoding="utf-8")
        # 表格里的配置列截断到 160 字符并转义竖线，否则超长配置会把 Markdown 表格撑坏。
        idx = [
            f"# 流水线失败日志（{model_dir.name}/{slot}）",
            "",
            "| 分类 | case-id | 日志文件 | 配置（截断） |",
            "|---|---|---|---|",
        ]
        for cat, case_id, name, config, _block in rows:
            short = config if len(config) <= 160 else config[:157] + "..."
            short = short.replace("|", "\\|")
            idx.append(f"| {cat} | {case_id} | `{name}` | `{short}` |")
        idx += [
            "",
            "> 单 case 日志格式：`>>> CASE <id>` / 配置行 / `[标记]` / 报错详情 / "
            "`<<< CASE <id> | FAIL/SKIP/COMPLETED`。",
        ]
        (dest / "INDEX.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    if missing:
        # OOM/crash 打挂进程时没有 case 日志，必须显式列出。
        dest.mkdir(parents=True, exist_ok=True)
        # 文件头写明 slot 和条数，直接 cat 也能看出是哪个 mode_shape 缺日志。
        (dest / "missing_log.txt").write_text(
            f"# {model_dir.name}/{slot}：以下 {len(missing)} 条配置没有 case 日志（通常为 OOM/crash）\n"
            + "\n".join(config for _cat, config in missing)
            + "\n",
            encoding="utf-8",
        )
    # 只有真正产出过 pipeline_errors 目录时才复制 run 日志，避免全部分类都缺日志时留下空目录。
    if dest.exists():
        multi = len(sources) > 1
        # 单机不加前缀，保持 log_<ts>.log 原名；campaign 多 part 时加 <part>__ 前缀区分来源机器。
        for label, log_dir in sources:
            _copy_run_logs(log_dir, dest, prefix=label if multi else "")
    return {
        "slot": slot,
        "fail": len(fail_configs),
        "refs": ref_counts,
        "with_log": len(rows),
        "missing_log": len(missing),
        "has_dir": dest.exists(),
    }


def _args_text(job: dict) -> str:
    # 每个参数单独 quote：engine_args 里可能带 = 号或空格的取值，拼错会让复现参数悄悄漂移。
    return " ".join(shlex.quote(arg) for arg in job.get("engine_args", []))


def _mode_script(mode: str, jobs: dict, model: str, env: dict) -> str | None:
    """生成 run_<mode>.sh：按 shape 分支复用本次运行的 engine 参数。

    :param jobs: shape -> job 的映射，只包含该 mode 下实际运行过的 shape。
    """
    # shape 顺序固定 0size -> 1M；jobs 里没有的 shape 不生成分支，避免复现时跑空配置。
    shapes = [shape for shape in ("0size", "1M") if shape in jobs]
    if not shapes:
        return None
    branches = []
    for shape in shapes:
        job = jobs[shape]
        branches.append(
            f'    {shape}) ${{PYTHON:-python}} engineV4.py --api_config_file="${{cfg}}" '
            f'--log_dir="${{log}}" {_args_text(job)} ;;'
        )
    case_block = "\n".join(branches)
    # env 原样 export 到脚本：sanitizer、精度开关等通过环境变量传给 engineV4，不塞进命令行。
    exports = "".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in env.items())
    return f"""#!/bin/bash
# 用法: bash run_{mode}.sh [model] [0size|1M|all]
# 默认 model={model}；命令参数与本次运行完全一致。
set -uo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
{_ROOT_SNIPPET}MODEL="${{1:-{model}}}"
SHAPE="${{2:-all}}"
{exports}rc=0
run_one() {{
  local s="$1"
  local cfg="${{DIR}}/${{MODEL}}/{mode}_${{s}}.txt"
  local log="${{DIR}}/logs/${{MODEL}}/${{s}}_{mode}"
  if [ ! -s "${{cfg}}" ]; then echo "[skip] ${{MODEL}} ${{s}} {mode}: 无失败配置"; return 0; fi
  mkdir -p "${{log}}"
  echo "[run] ${{MODEL}} ${{s}} {mode} | $(wc -l < "${{cfg}}") configs | log=${{log}}"
  case "${{s}}" in
{case_block}
    *) echo "未知 shape: ${{s}}"; return 2 ;;
  esac
}}
for s in 0size 1M; do
  [ "${{SHAPE}}" = "all" ] || [ "${{SHAPE}}" = "${{s}}" ] || continue
  run_one "${{s}}" || rc=$?
done
exit ${{rc}}
"""


def _case_script(jobs: dict, model: str) -> str:
    """单 case 复现：按 debug<卡型>/<model>/<mode>_<shape>.txt 的行号执行。"""
    # 行号按 <mode>_<shape>.txt 的物理行计算，与脚本里 sed -n '<N>p' 的语义严格一致。
    cases = []
    for (mode, shape), job in sorted(jobs.items()):
        envs = "".join(
            f"      export {key}={shlex.quote(str(value))}\n"
            for key, value in (job.get("env") or {}).items()
        )
        # 每个 mode:shape 分支各自 export 环境，避免上一个分支的变量泄漏到下一个分支。
        cases.append(f"  {mode}:{shape})\n{envs}      ARGS=({_args_text(job)}) ;;")
    case_block = "\n".join(cases)
    return f"""#!/bin/bash
# 单 case 复现。用法: bash run_case.sh <mode> <shape> <行号> [model]
# 例: bash run_case.sh accuracy 1M 3
set -uo pipefail
DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
{_ROOT_SNIPPET}MODE="${{1:-}}"; SHAPE="${{2:-}}"; N="${{3:-}}"; MODEL="${{4:-{model}}}"
if [ -z "${{MODE}}" ] || [ -z "${{SHAPE}}" ] || [ -z "${{N}}" ]; then
  echo "用法: bash $0 <mode> <shape> <行号> [model]"; exit 2
fi
CFG="${{DIR}}/${{MODEL}}/${{MODE}}_${{SHAPE}}.txt"
if [ ! -s "${{CFG}}" ]; then echo "找不到失败清单: ${{CFG}}"; exit 2; fi
LINE=$(sed -n "${{N}}p" "${{CFG}}")
if [ -z "${{LINE}}" ]; then echo "行号 ${{N}} 超出 ${{CFG}} 范围"; exit 2; fi
cd "${{ROOT}}" || exit 1
LOG="${{DIR}}/logs/${{MODEL}}/${{MODE}}_${{SHAPE}}_case_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${{LOG}}"
case "${{MODE}}:${{SHAPE}}" in
{case_block}
  *) echo "没有该 mode/shape 的运行参数: ${{MODE}}:${{SHAPE}}"; exit 2 ;;
esac
echo "[run] ${{MODEL}} ${{MODE}} ${{SHAPE}} 第 ${{N}} 行 -> ${{LOG}}"
echo "[case] ${{LINE}}"
cd "${{ROOT}}" || exit 1
"${{PYTHON:-python}}" engineV4.py --api_config="${{LINE}}" --log_dir="${{LOG}}" "${{ARGS[@]}}"
"""


def _render_readme(
    card: str,
    model: str,
    run_id: str,
    summaries: list[dict],
    campaign: bool = False,
    parts: int = 0,
) -> str:
    # 汇总口径是"真实失败条数之和"，oom/skip/bitwise 等参考清单不计入。
    total_fail = sum(item["fail"] for item in summaries)
    lines = [
        f"# debug{card} 失败复现包（{run_id}）",
        "",
        f"- 生成时间：{now_str()}",
        f"- 卡型：{card}，模型目录：{model}",
        f"- 真实失败配置合计：{total_fail} 条",
    ]
    # campaign 模式额外标注合并了多少个 part；单机模式不写这一行以免误导。
    if campaign:
        lines.append(f"- 汇总范围：{parts} 个 part（失败配置和日志已跨机器合并）")
    lines += [
        "- 放置位置：可在任意目录执行；脚本会从自身向上寻找 `engineV4.py`，"
        "也可设置 `PADDLEAPITEST_ROOT=/path/to/PaddleAPITest`。",
        "",
        "## 批量复现",
        "",
        "```bash",
    ]
    for mode in ("accuracy", "paddleonly", "stable"):
        if any(item["slot"].startswith(f"{mode}_") for item in summaries):
            lines.append(f"bash run_{mode}.sh {model} all")
            for shape in ("0size", "1M"):
                if any(item["slot"] == f"{mode}_{shape}" for item in summaries):
                    lines.append(f"bash run_{mode}.sh {model} {shape}")
    lines += [
        "```",
        "",
        "## 单 case 复现",
        "",
        "```bash",
    ]
    if summaries:
        first_mode, first_shape = summaries[0]["slot"].rsplit("_", 1)
        lines.append(f"bash run_case.sh {first_mode} {first_shape} 1")
    lines += [
        "```",
        "",
        "## 包内容",
        "",
        "| slot | 真实失败 | OOM | skip | bitwise | 有日志 | 无日志 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in summaries:
        refs = item.get("refs", {})
        lines.append(
            f"| {item['slot']} | {item['fail']} | {refs.get('oom', 0)} | {refs.get('skip', 0)} | "
            f"{refs.get('paddle_bitwise', 0) + refs.get('paddle_bitwise_knows', 0)} | "
            f"{item['with_log']} | {item['missing_log']} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- `<slot>.txt` 只收真实失败（paddle_accuracy / paddle_error / paddle_crash / "
        "paddle_cuda / config_input / config_parse / config_convert / both_error / "
        "torch_error / torch_only_error）。",
        "- `oom_*.txt` / `skip_*.txt` / `bitwise_*.txt` 是参考清单，默认不重跑。",
        "- `pipeline_errors/<slot>/` 是对应失败的原始单 case 日志、合并日志和 run 日志；"
        "OOM/crash 没有日志时看 `missing_log.txt`。",
        "- 批量参数与本次运行完全一致；单 case 使用同一套参数，仅把输入换成单条配置。",
    ]
    return "\n".join(lines) + "\n"


def _render_index(card: str, run_id: str, model: str, summaries: list[dict]) -> str:
    lines = [
        f"# debug{card} 索引（{run_id}）",
        "",
        f"- 模型目录：`{model}`",
        f"- 生成时间：{now_str()}",
        "",
        "| slot | 真实失败 | 参考分类 | pipeline_errors |",
        "|---|---|---|---|",
    ]
    for item in summaries:
        # 参考分类按实际命中项渲染，一个都没有时写 -，避免表格列错位。
        refs = ", ".join(f"{key}({value})" for key, value in item.get("refs", {}).items()) or "-"
        path = f"`{model}/pipeline_errors/{item['slot']}/`" if item["has_dir"] else "-"
        lines.append(f"| {item['slot']} | {item['fail']} | {refs} | {path} |")
    return "\n".join(lines) + "\n"


def detect_card(explicit: str | None, run: dict) -> str:
    """卡型优先级：显式参数 > run.json > APITEST_CARD > local。"""
    # 三级回退保证任何环境都能得到非空卡型；local 只是兜底占位，不是真实卡型。
    card = (explicit or "").strip()
    if not card:
        card = str(run.get("card") or "").strip()
    if not card:
        card = os.environ.get("APITEST_CARD", "").strip()
    return card or "local"


def build_debug_bundle(
    run_id: str,
    run_dir: Path,
    card: str | None = None,
    model: str | None = None,
    out_dir: Path | None = None,
) -> dict | None:
    """读取一个单机 run 的产物，生成 debug<卡型>/；没有失败配置时返回 None。"""
    # run.json 是唯一必需输入：缺失时直接 SystemExit，避免生成一个内容为空的复现包。
    run = load_json(run_dir / "run.json", {})
    if not run:
        raise SystemExit(f"未找到运行记录: {run_dir}/run.json")
    return build_debug_from_jobs(
        run_id,
        run_dir,
        run.get("jobs", []),
        run,
        card=card,
        model=model,
        out_dir=out_dir,
    )


def build_debug_from_jobs(
    run_id: str,
    run_dir: Path,
    jobs: list[dict],
    run: dict,
    card: str | None = None,
    model: str | None = None,
    out_dir: Path | None = None,
    campaign: bool = False,
) -> dict | None:
    """把若干 job（单机或跨 part）的失败汇总成一个 debug<卡型>/。"""
    card = detect_card(card, run)
    model = (model or "").strip() or str(run.get("model") or "").strip() or "local"
    parent = Path(out_dir).resolve() if out_dir else run_dir
    debug_dir = parent / f"debug{card}"
    # 每次整目录重建：上一次运行残留的文件（已修好的 case）不应留在新包里。
    shutil.rmtree(debug_dir, ignore_errors=True)
    model_dir = debug_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    # 同一个 mode_shape 跨 part 合并：config 去重、日志来源全部保留。
    slots: dict[str, dict] = {}
    first_jobs: dict[tuple[str, str], dict] = {}
    for job in jobs:
        log_dir = job_log_dir(run_dir, job)
        info = classify_job(job, log_dir)
        classes = info["classes"]
        # 只有 pass 的 slot 不建目录，否则每个跑过的 mode_shape 都会留下空包结构。
        if not any(classes.get(cat) for cat in CAT_ORDER if cat != "pass"):
            continue
        slot = _slot(job["mode"], job["shape"])
        entry = slots.setdefault(slot, {"classes": {}, "sources": []})
        entry["sources"].append((job.get("part") or "", log_dir))
        # 跨 part 去重按配置行原文，不做归一化：配置文本本身就是 engine 的输入主键。
        for cat, lines in classes.items():
            bucket = entry["classes"].setdefault(cat, [])
            for line in lines:
                if line not in bucket:
                    bucket.append(line)
        # 同一 mode 的复现脚本参数只取第一个 part：campaign 要求各 part 启动参数一致。
        first_jobs.setdefault((job["mode"], job["shape"]), job)

    summaries = [
        _write_slot(model_dir, slot, entry["sources"], entry["classes"])
        for slot, entry in slots.items()
    ]
    # 一个真实失败都没有就删掉 debug<卡型>/，调用方据 None 判断是否提示用户。
    if not summaries:
        shutil.rmtree(debug_dir, ignore_errors=True)
        return None

    # 同一 mode 的脚本只生成一次，参数取该 mode_shape 的任一 part（campaign 默认参数一致）。
    for mode in ("accuracy", "paddleonly", "stable"):
        mode_jobs = {
            shape: first_jobs[(mode, shape)]
            for shape in ("0size", "1M")
            if (mode, shape) in first_jobs
        }
        # 该 mode 一个 shape 都没跑过时跳过，避免生成会立刻以 exit 2 结束的空脚本。
        if not mode_jobs:
            continue
        env = {}
        for job in mode_jobs.values():
            env.update(job.get("env") or {})
        script = _mode_script(mode, mode_jobs, model, env)
        if script:
            _write_shell(debug_dir / f"run_{mode}.sh", script)
    _write_shell(debug_dir / "run_case.sh", _case_script(first_jobs, model))

    # parts 只按带 part 字段的 job 去重统计，且仅 campaign 场景展示，单机恒为 0。
    parts = len({job.get("part") for job in jobs if job.get("part")}) if campaign else 0
    (debug_dir / "README.md").write_text(
        _render_readme(card, model, run_id, summaries, campaign=campaign, parts=parts),
        encoding="utf-8",
    )
    (debug_dir / "INDEX.md").write_text(
        _render_index(card, run_id, model, summaries), encoding="utf-8"
    )
    return {
        "path": str(debug_dir),
        "card": card,
        "model": model,
        "summaries": summaries,
        "campaign": campaign,
    }
