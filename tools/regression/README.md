# 项目回归测试

`tools/regression` 维护项目级固定回归配置集，用于在修改后验证 `paddle_only` 和 `accuracy` 两类门禁。

当前固定集合为 `tools/regression/regression_configs.txt`，覆盖 114 个 API key、473 条配置。

## 运行回归

```bash
tools/regression/regression_runner.sh
```

流水线会使用同一份 `regression_configs.txt` 依次运行：

- `engineV4.py --paddle_only=True`
- `engineV4.py --accuracy=True`

默认执行参数：

- `--gpu_ids=-1`
- `--num_gpus=-1`
- `--num_workers_per_gpu=4`
- `--timeout=180`

可通过环境变量覆盖：

```bash
GPU_IDS=0 REGRESSION_NUM_GPUS=1 REGRESSION_WORKERS_PER_GPU=2 \
  tools/regression/regression_runner.sh
```

常用环境变量：

- `REGRESSION_CONFIG_FILE`：配置集合路径，默认 `tools/regression/regression_configs.txt`
- `REGRESSION_LOG_DIR`：日志输出目录，默认临时目录 `/tmp/paddleapitest_project_regression.XXXXXX`
- `PYTHON`：Python 解释器，默认 `python`
- `GPU_IDS`：传给 `engineV4.py --gpu_ids`
- `REGRESSION_NUM_GPUS`：传给 `engineV4.py --num_gpus`
- `REGRESSION_WORKERS_PER_GPU`：传给 `engineV4.py --num_workers_per_gpu`
- `REGRESSION_TIMEOUT`：单配置超时时间

执行结束后，脚本会调用 `tools/error_stat/error_stat.py --split-errors` 解析结果，并用
`tools/regression/check_error_stat.py` 检查门禁。

## 门禁规则

允许分类：

- `pass`
- `skip`
- `paddle_bitwise`

不允许分类：

- `paddle_error`
- `paddle_accuracy`
- `paddle_cuda`
- `paddle_crash`
- `oom`
- `timeout`
- `torch_error`
- `config_input`
- `config_parse`
- `config_convert`

## 维护配置集合

如需从外部 APIConfig 文件重新收集固定集合，显式传入来源目录或文件：

```bash
python tools/regression/collect_configs.py \
  --source /path/to/api_config_dir \
  --source /path/to/api_config_file.txt \
  --output tools/regression/regression_configs.txt \
  --summary tools/regression/regression_summary.txt
```

`collect_configs.py` 会为每个 API key 保留最多 5 条不同配置。源配置不足 5 条时，按实际数量进入回归集合。

收集策略：

- 排除路径中包含 `needfix`、`need_fix`、`not_monitor`、`1m`、`0size` 的配置。
- 排除配置文本中包含 `float8_` 的配置。
- 排除 `paddle.empty` 与 `paddle.empty_like`，因为它们返回未初始化内存，不适合作为稳定门禁。
- 仅保留项目当前支持输入生成和 accuracy 转换的 API。
- `paddle._C_ops._run_custom_op` 按第一个参数 `op_name` 细分 API key，例如 `paddle._C_ops._run_custom_op:fused_swiglu_scale_clamp`。

如候选集运行产生非允许分类，可根据 `error_stat` 结果剔除失败配置：

```bash
python tools/regression/refine_configs.py \
  --input tools/regression/regression_configs.txt \
  --output tools/regression/regression_configs.txt \
  --log-dir /path/to/log/paddle_only \
  --log-dir /path/to/log/accuracy
```
