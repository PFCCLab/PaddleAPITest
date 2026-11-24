---
name: shape-test-runner
description: 在已就绪的 PaddleAPITest 环境中，用 0size.txt / 1M.txt 启动 engineV4 测试；支持单机，也支持一个 txt 切多份、多机共享盘自动认领（split/worker），并汇总进度、校验覆盖率、整理失败问题与复现方法、生成 debugH/debugB。当用户要跑 0size/1M、多机分片、查看/汇总进度、检查漏跑、整理失败原因或生成复现包时使用。
---

# Shape Test Runner

工具位置：`PaddleAPITest/shape_test_runner/`，入口 `st.py`。
只负责执行与整理，不负责环境：**不安装依赖、不管理镜像或 Paddle/PaddleFleet/PaddleAPITest 版本**。

## 工作流

1. **体检**：`python shape_test_runner/st.py doctor`
   - 缺 `paddle` / `torch` / `pynvml` 时停止，明确告知用户环境未就绪；不要尝试 pip install。
2. **确认输入**：用户给的 `0size.txt` / `1M.txt` 每行一条 API config。
   - 只给一个文件就只跑该 shape；`--shapes` 可进一步限制。
   - `start` 会做语法校验、重复行提示；重复行多时建议 `--dedup`（只写副本，不改原文件）。
3. **启动**：
   ```bash
   python shape_test_runner/st.py start --0size <path> --1m <path> --model eb5_1 --card H
   python shape_test_runner/st.py start --1m <path> --mode accuracy
   ```
   默认 `accuracy,paddleonly`，后台串行；`--foreground` 调试，`--dry-run` 只看命令。
   给了 `--card H|B|GB` 时，结束会自动生成 `debugH` / `debugB`；也可事后补：
   ```bash
   python shape_test_runner/st.py debug --run-id <id> --card H --model eb5_1
   ```
4. **进度**：`python shape_test_runner/st.py status --run-id <id>`
   - 进度来自 `checkpoint.txt`，分类来自 `api_config_*.txt`，不依赖进程日志猜测。
   - `--watch 30` 持续刷新；`--strict` 有真实失败时返回 1。
5. **问题与复现**：结束后默认自动生成；手动重跑用
   ```bash
   python shape_test_runner/st.py collect --run-id <id>
   python shape_test_runner/st.py repro  --run-id <id>
   python shape_test_runner/st.py debug  --run-id <id> --card H
   ```
6. **汇报**：向用户给出
   - 总进度与每个 job 的 pass / 真实失败 / oom/skip/bitwise 数量；
   - `issues/INDEX.md` 的错误签名 Top N 和示例 case；
   - 失败配置与单 case 日志路径；
   - `repro/README.md` 或 `debug<卡型>/README.md` 的复现命令。

## 多机共享盘

一个 txt 切多份、多台机器共享同一 runs 目录时：

```bash
# 主控机
python shape_test_runner/st.py split --run-id m1m --0size 0size.txt --1m 1M.txt \
  --shards 8 --model eb5_1 --card H

# 每台机器（共享盘同一路径）：自动认领并连续跑空闲 part
python shape_test_runner/st.py worker --run-id m1m
# 或只跑指定 part
python shape_test_runner/st.py run-part --run-id m1m --part part-03

# 任意机器
python shape_test_runner/st.py parts    --run-id m1m
python shape_test_runner/st.py status   --run-id m1m --watch 30
python shape_test_runner/st.py verify   --run-id m1m
python shape_test_runner/st.py finalize --run-id m1m --card H
```

- `split` 按 shape 切分，默认 `round-robin`；行数不足自动减少 part 并告警。
- `run-part` 只写 `parts/<part>/`，用 `.run.lock` 防重复、`heartbeat.json` 报活；
  超过默认 120s 无心跳判“失联”，`--force` 可接管。
- 跨机器不要看 PID，只看心跳；`status` 会把未启动 part 也计入总进度。
- `collect`/`debug` 自动跨 part 合并；同名 run 日志加 `part-XX__` 前缀。
- 共享盘挂载点不同：统一设置 `SHAPE_TEST_RUNS_DIR=/path/to/shared/runs`。
- `stop` 只停本机 part；campaign 会写 `STOP` 让 worker 退出；`--part` 只停一个。
- 机器掉线：`reset-part --run-id X --part part-03` 清锁和心跳；`--clean` 从头跑。
- `verify` 校验切分守恒与 case 覆盖（输入 - checkpoint/分类），缺失清单在 `verify/missing/`。
- `finalize` 在完整时一次性生成 status/issues/debug；`--repro` 同时生成逐 part 复现包。
- `collect/debug/repro/verify/finalize` 有 `.aggregate.lock`，多机并发不会写坏汇总。
- `issues/INDEX.md` 额外按 API 聚合；`status.md` 带已运行时长、case/min 和 ETA。

## debugH / debugB

与流水线 `api_test/runtime/utils/debug_bundle.py` 同口径，但输入是本地 run：

- 目录：`runs/<id>/debug<卡型>/<model>/`；每个 `mode_shape` 一个 slot
  （`accuracy_0size`、`paddleonly_1M` 等）。
- slot 文件只收真实失败；`oom_*.txt` / `skip_*.txt` / `bitwise_*.txt` 是参考清单。
- `pipeline_errors/<slot>/`：单 case 日志、`<分类>.all.log`、`missing_log.txt`、原始 run 日志。
- 脚本：`run_accuracy.sh <model> [0size|1M|all]`、`run_paddleonly.sh`、
  `run_stable.sh`（有 stable job 时）、`run_case.sh <mode> <shape> <行号>`。
- 脚本内嵌本次运行的真实参数；复制到其他机器后设置 `PADDLEAPITEST_ROOT` 即可执行。
- 流水线侧仍由 `api_test/runtime/utils/debug_bundle.py` 生成并上传 BOS；本技能是同一口径的本地版。

## 失败口径（必须遵守）

- 真实失败：`paddle_accuracy`、`paddle_error`、`paddle_crash`、`paddle_cuda`、
  `config_input`、`config_parse`、`config_convert`、`both_error`、`torch_error`、
  `torch_only_error`。
- 参考分类：`oom`、`skip`、`paddle_bitwise`、`paddle_bitwise_knows`，单独归档，默认不重跑。
- 整理失败时**配置和 `.log` 必须同时给出**：`issues/cases/<job>/<分类>/<case-id>.log`。
- OOM/crash 没有单 case 日志是正常现象，必须保留 `missing_log.txt` 并说明数量。
- 不要把"没跑完"说成"通过"；环境缺失时直接说明未运行原因和可复现命令。

## 常用运维

- 中断后续跑：`python shape_test_runner/st.py resume --run-id <id>`
  （engineV4 靠 checkpoint 跳过已完成 case）。
- 停止：`python shape_test_runner/st.py stop --run-id <id>`。
- 同一 run 内 job 串行；要并行请用不同 `--run-id` + 不同 `--gpus`，避免抢卡。
- 自测工具本身：`python3 shape_test_runner/tests/test_tools.py`（不需要 GPU）。

## 产物位置

`shape_test_runner/runs/<run_id>/` 下：
`status.md`、`issues/INDEX.md`、`issues/cases/`、`repro/`、`debug<卡型>/`、`logs/<shape>_<mode>/`；
多机 campaign 另有 `campaign.json`、`shards/`、`parts/<part>/`、`heartbeat.json`。
细节见 `shape_test_runner/README.md`。
