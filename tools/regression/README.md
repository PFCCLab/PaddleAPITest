# 项目回归测试流水线

本目录用于维护项目级固定回归配置集，不归属于某个单独模块。
当前固定集合为 `tools/regression/regression_configs.txt`，覆盖 114 个 API key、473 条配置。

## 配置来源

- `/root/paddlejob/share-storage/gpfs/system-public/lihaoyang08/PaddleAPITest-worktrees/opt/tester/api_config/monitor_config/accuracy/GPU`
- `/root/paddlejob/share-storage/gpfs/system-public/lihaoyang08/baidu/paddle/jelly/case/scripts/api_test/apitest_config`

`collect_configs.py` 会从以上目录收集真实 APIConfig，并为每个 API key 保留最多 5 条不同配置。
如果某个 API key 源配置不足 5 条，也会按实际数量进入回归集合。
收集时会排除 `1M` 与 `0size` 配置目录，避免把大规模或零维专项集合混入常规回归。
`paddle.empty` 与 `paddle.empty_like` 不进入固定回归集合，因为它们返回未初始化内存，
不适合参与稳定的 paddle only 与 accuracy 门禁。
`paddle._C_ops._run_custom_op` 按第一个参数 `op_name` 细分 API key，例如：

```text
paddle._C_ops._run_custom_op:fused_swiglu_scale_clamp
```

## 运行方式

```bash
source /root/.venv-lihaoyang08/bin/activate
python tools/regression/collect_configs.py
tools/regression/regression_runner.sh
```

流水线使用同一份 `regression_configs.txt` 分别运行：

- `engineV4.py --paddle_only=True`
- `engineV4.py --accuracy=True`

默认执行参数为 `--gpu_ids=-1`、`--num_gpus=-1` 与 `--num_workers_per_gpu=4`，
即使用可见 GPU 最大数量，并在每张 GPU 上启动 4 个 worker。可通过 `GPU_IDS`、
`REGRESSION_NUM_GPUS` 和 `REGRESSION_WORKERS_PER_GPU` 临时覆盖。

执行结束后会调用 `tools/error_stat/error_stat.py --split-errors` 解析结果。门禁规则是：

- 允许：`pass`、`skip`、`paddle_bitwise`
- 不允许：`paddle_error`、`paddle_accuracy`、`paddle_cuda`、`paddle_crash`、`oom`、`timeout`、`torch_error`、`config_input`、`config_parse`、`config_convert`

最近一次通过记录：

- 日志目录：`/tmp/project_regression_1785489586`
- 并发参数：8 张 GPU、每张 GPU 4 个 worker
- `paddle_only`：473 pass，0 fail
- `accuracy`：473 pass，0 fail
- `error_stat`：无 `paddle_bitwise`，无其他错误分类

如候选集首次运行产生非允许分类，可用 `refine_configs.py` 基于 `error_stat` 结果剔除这些配置，
再重新运行流水线：

```bash
python tools/regression/refine_configs.py \
  --input tools/regression/regression_configs.txt \
  --output tools/regression/regression_configs.txt \
  --log-dir /path/to/log/paddle_only \
  --log-dir /path/to/log/accuracy
```
