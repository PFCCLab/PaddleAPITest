# shape_test_runner

给定 `0size.txt` / `1M.txt`（可以只给其中一个），在**已经就绪**的 PaddleAPITest
环境里启动测试，并负责后续三件事：**整理进度、整理问题、生成复现方法**。
整理结果可以按流水线同名口径落成 **debugH / debugB** 复现包。

同时支持**多机共享盘**：把一个 txt 切成多个 part，每台机器跑一个 part，
任一机器都能从共享盘汇总所有 part 的进度和问题。

## 边界

- 不安装任何依赖，不管理镜像、Paddle、PaddleFleet、PaddleAPITest 版本。
- 不修改 engineV4 / tester 的行为，只在外面调用。
- 启动前做环境体检；缺 paddle / torch / pynvml 时直接报阻断项，不伪造结果。
- 输入文件每行是一条 API config（与 `tester/api_config` 下的 txt 同格式）。

## 快速开始

```bash
# 1. 只体检，不安装
python shape_test_runner/st.py doctor

# 2. 同时测 0size 和 1M（默认 accuracy + paddleonly 两种模式）
python shape_test_runner/st.py start --0size /path/0size.txt --1m /path/1M.txt \
  --model eb5_1 --card H

# 只测 1M、只测 accuracy
python shape_test_runner/st.py start --1m /path/1M.txt --mode accuracy

# 3. 看进度
python shape_test_runner/st.py status --run-id run_20261120_101010
python shape_test_runner/st.py status --run-id run_20261120_101010 --watch 30

# 4. 补生成 debugH/debugB（start 时给了 --card 会自动生成）
python shape_test_runner/st.py debug --run-id run_20261120_101010 --card H
```

多机共享盘：

```bash
# 主控机：切成 8 个 part（0size/1M 各自 round-robin 切分）
python shape_test_runner/st.py split --run-id m1m --0size 0size.txt --1m 1M.txt \
  --shards 8 --model eb5_1 --card H

# 每台机器：自动认领并连续跑空闲 part
python shape_test_runner/st.py worker --run-id m1m
# 或只跑指定 part
python shape_test_runner/st.py run-part --run-id m1m --part part-03

# 任意机器：看汇总、汇总问题、合并复现包
python shape_test_runner/st.py parts   --run-id m1m
python shape_test_runner/st.py status  --run-id m1m --watch 30
python shape_test_runner/st.py verify  --run-id m1m
python shape_test_runner/st.py finalize --run-id m1m --card H
```

共享盘挂载点不同时，各机器统一设置 `SHAPE_TEST_RUNS_DIR=/path/to/shared/runs`。

启动后默认在后台顺序执行，结束后自动生成 `status.md`、`issues/INDEX.md` 和
`repro/`。前台调试用 `--foreground`，只打印命令用 `--dry-run`。

## 命令

| 命令 | 作用 |
|---|---|
| `doctor` | 检查 python / paddle / torch / pynvml / GPU / compute-sanitizer，只读 |
| `start` | 单机校验输入并启动；`--mode`、`--shapes`、`--gpus`、`--workers`、`--timeout` 可覆盖默认 |
| `split` | 把 txt 切成多个 part，写 `campaign.json`；`--shards`、`--strategy`、`--dedup` |
| `run-part` | 在共享盘 campaign 上认领并运行一个 part；有锁和心跳，支持 `--force` 接管 |
| `parts` | 列出所有 part 的状态、host、心跳、是否可认领；`--json` 供脚本使用 |
| `worker` | 自动认领并连续运行空闲 part；`--max-parts`、`--rerun-failed`、`--foreground` |
| `verify` | 校验切分守恒与每个 part 的 case 覆盖率，缺失配置写到 `verify/missing/` |
| `finalize` | 全齐后一次性生成 `status/issues/debug`；`--repro` 同时生成逐 part 复现包 |
| `reset-part` | 清理失联 part 的锁和心跳；`--clean` 连日志一起删，从头跑 |
| `status` | 读 checkpoint/分类文件/心跳，输出 `status.md` / `status.json`；`--watch N` 持续刷新 |
| `collect` | 导出失败分类、单 case `.log`、错误签名聚合，输出 `issues/`（campaign 跨 part 汇总） |
| `repro` | 生成失败配置清单、批量脚本、单 case 脚本，输出 `repro/`（campaign 按 part 分目录） |
| `debug` | 生成 `debugH` / `debugB` 风格复现包；campaign 会跨 part 合并失败配置与日志 |
| `resume` | 对未完成 job 续跑（engineV4 用 checkpoint 跳过已完成 case；campaign 用 `run-part`） |
| `stop` | 杀掉本机 supervisor 与 job 进程组；campaign 写 `STOP` 让 worker 退出；`--part` 只停一个 part |

`start` 的主要参数：

- `--shapes 0size,1M`：只跑指定 shape（也可只给一个输入文件）。
- `--mode accuracy,paddleonly,stable`：逗号组合，默认 `accuracy,paddleonly`。
- `--accuracy-mode auto|accuracy|dual`：默认与 pipeline 一致，0size 用 `--accuracy`，
  1M 用 `--accuracy_dual_gpu`。
- `--sanitizer auto|on|off`：paddleonly 默认在机器有 `compute-sanitizer` 时启用。
- `--model <name>`：写进 `run.json`，作为 `debug<卡型>/<model>/` 的目录名（默认 `local`）。
- `--card H|B|GB|...`：写进 `run.json`，结束后自动生成 `debugH` / `debugB`；
  不传时读 `APITEST_CARD`，都没有则只生成 `repro`，可事后用 `debug --card` 补。
- `--dedup`：输入有重复行时使用去重后的副本（不改原文件）。
- `--engine-arg KEY=VALUE`：额外透传，可重复。

默认值与 `test_pipeline/generic_configs` 对齐：0size 每卡 4 worker、超时 600s；
1M 每卡 1 worker、超时 1200s；accuracy 阈值 `atol=rtol=0.0`。

## 产物

```text
shape_test_runner/runs/<run_id>/
├── run.json              # 输入快照、jobs、engine 参数
├── state.json            # supervisor / job 的 PID 与退出码
├── supervisor.log
├── inputs/               # 运行时实际使用的 0size.txt / 1M.txt 副本
├── logs/<shape>_<mode>/  # engineV4 --log_dir
├── status.md / status.json
├── issues/
│   ├── INDEX.md          # 失败分类、日志覆盖、错误签名聚合
│   ├── issues.json
│   └── cases/<job>/<分类>/<case-id>.log
└── repro/
    ├── README.md / INDEX.md
    ├── configs/<job>_failures.txt
    ├── configs/<job>_args.txt / <job>_env.sh
    ├── refs/<job>_<oom|skip|bitwise>.txt
    ├── run_<job>.sh      # 批量复现
    └── run_case.sh       # 单 case：bash run_case.sh <job> <行号>

# 指定了 --card 时额外生成（结构对齐流水线 debugH/debugB）
debugH/
├── README.md / INDEX.md
├── run_accuracy.sh / run_paddleonly.sh / run_stable.sh
├── run_case.sh
└── <model>/
    ├── accuracy_0size.txt / accuracy_1M.txt ...
    ├── oom_*.txt / skip_*.txt / bitwise_*.txt
    └── pipeline_errors/<mode>_<shape>/
        ├── INDEX.md
        ├── <分类>__<case-id>.log / <分类>.all.log
        ├── missing_log.txt
        └── log_<timestamp>.log
```

## debugH / debugB

`debug` 子命令复刻流水线里 `api_test/runtime/utils/debug_bundle.py` 的产物形态，
但输入是本地 `runs/<run_id>/`，不需要 BOS：

```bash
python shape_test_runner/st.py debug --run-id <id> --card H --model eb5_1
# 默认输出到 runs/<id>/debugH/；--out /tmp 可改父目录
```

- 每个 `mode_shape` 一个 slot（如 `accuracy_0size`、`paddleonly_1M`）：
  slot 文件只收真实失败；`oom_*.txt`、`skip_*.txt`、`bitwise_*.txt` 是参考清单。
- `pipeline_errors/<slot>/` 放单 case 日志、分类合并日志、`missing_log.txt`、
  该 job 的原始 `log_*.log`。
- 顶层脚本与流水线同名：`run_accuracy.sh <model> [0size|1M|all]`、
  `run_paddleonly.sh`、`run_stable.sh`（有 stable job 时），外加 `run_case.sh <mode> <shape> <行号>`。
- 脚本内嵌本次运行的实际 engine 参数；复制到别的机器后，设置 `PADDLEAPITEST_ROOT`
  或保持目录在 PaddleAPITest 下即可直接跑。

> 流水线侧仍由 `api_test/runtime/utils/debug_bundle.py` 在 job 收尾时生成并上传 BOS；
> 这里的 `debug` 是同一口径的本地版，产物名和目录结构保持一致，便于对照。

## 多机共享盘（campaign）

```text
runs/<campaign>/
├── campaign.json          # 切分清单、默认参数、模型/卡型、stale 阈值
├── shards/part-00/0size.txt / 1M.txt ...
├── parts/part-00/         # 一台机器一个 part，内部就是普通 run 目录
│   ├── run.json / state.json / heartbeat.json / .run.lock
│   ├── inputs/ logs/ status.md issues/ repro/ debugH/
├── status.md / status.json   # 跨 part 汇总
├── issues/                   # 跨 part 汇总的问题与单 case 日志
├── repro/                    # 各 part 的复现索引
└── debugH/                   # 跨 part 合并的 debugH/debugB
```

- `split` 按 shape 分别切分；`--strategy round-robin`（默认，负载更均匀）或 `contiguous`。
  行数不足 `--shards` 时自动减少份数并告警；重复行建议加 `--dedup`。
- `run-part` 只写自己的 `parts/<part>/`；心跳每 15s 刷新 `heartbeat.json`。
  超过 `--stale`（默认 120s）没有心跳即判“失联”；`.run.lock` 防止两台机器跑同一个 part。
- `status` 汇总所有 part：未启动 part 也计入总进度；跨机器不看 PID，只看心跳。
- `collect` / `debug` 跨 part 合并：失败配置去重，单 case 日志按 `<part>/<job>/<分类>/` 归档，
  debug 包里的同名 run 日志会加 `part-XX__` 前缀，避免互相覆盖。
- `stop` 只能停本机 part；别的机器上的 part 需在对应机器执行。
- 共享盘挂载点不同：所有机器设置同一个 `SHAPE_TEST_RUNS_DIR=/path/to/shared/runs`。

## 自动认领与收尾

- `worker` 让每台机器一条命令吃空闲 part；跑到没有可认领 part 后自动退出。
  `--max-parts N` 限制单进程最多跑几个 part，`--rerun-failed` 连非 0 退出的 part 也重跑。
- `st.py stop --run-id X` 会写 `STOP` 标记并停本机 part；worker 跑完当前 part 后退出。
  重新执行 `worker` 会清除 `STOP`。只想停一个 part 用 `st.py stop --part part-03`。
- 机器掉线后用 `st.py reset-part --run-id X --part part-03` 清理锁和心跳，part 会重新可认领；
  `--clean` 会删掉该 part 的日志/状态，从头跑。
- `verify` 检查两部分：
  - 切分守恒：各 part 行数之和是否等于输入（含 `--dedup` 后的口径）；
  - case 覆盖：`输入配置 - (checkpoint ∪ api_config_*.txt)`，缺失清单在 `verify/missing/`。
- `finalize` 先跑 `verify`；完整时自动生成 `status.md`、`issues/`、`debugH`，
  加 `--repro` 再生成逐 part 复现包；不完整默认拒绝出报告，`--allow-incomplete` 可强制。
- `collect/debug/repro/verify/finalize` 在 campaign 上会拿 `.aggregate.lock`，
  多台机器同时执行时不会互相写坏产物。

问题汇总除了错误签名，还会按 API 聚合（`paddle.xxx` / `_run_custom_op/op`）；
`status.md` 的每个 job 会显示已运行时长、case/min 和预计剩余时间。

## 问题口径

- 真实失败进入复现清单：`paddle_accuracy`、`paddle_error`、`paddle_crash`、
  `paddle_cuda`、`config_input`、`config_parse`、`config_convert`、`both_error`、
  `torch_error`、`torch_only_error`。
- `oom` / `skip` / `paddle_bitwise` / `paddle_bitwise_knows` 只作为参考清单，默认不重跑。
- OOM / crash 常把进程打挂，平台没有单 case 日志；这类配置会写进
  `issues/cases/<job>/<分类>/missing_log.txt` 并标注数量，不会静默消失。
- `issues/INDEX.md` 会按错误签名聚合（地址和数字已归一化），先看同一签名的数量，
  再看单 case 日志。

## 续跑与并发

- `resume --run-id X` 复用原 log_dir，engineV4 依据 checkpoint 跳过已完成 case。
- 同一 run 内 job 串行执行，避免多 shape 抢 GPU；需要并行时用不同 `--run-id`
  配合不同 `--gpus`。

## 自测

不依赖 GPU / paddle / torch：

```bash
python3 shape_test_runner/tests/test_tools.py
```

覆盖：0 维识别、配置语法校验、进度统计、单 case 日志拆分、OOM 缺日志标注、
错误签名聚合、复现脚本生成与 `bash -n` 校验。
