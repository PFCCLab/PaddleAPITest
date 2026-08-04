# 项目回归测试

`tools/regression` 维护项目级固定回归配置集，用于在修改后验证 `paddle_only` 和 `accuracy` 两类门禁。

当前固定集合为 `tools/regression/regression_configs.txt`，覆盖 146 个 API key、438 条配置；每个
API key 保留 3 条不同的小型配置。

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
- `REGRESSION_LOG_DIR`：日志输出目录，默认 `tools/regression/logs/run.XXXXXX`
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

固定配置集直接维护在 `regression_configs.txt`。修改单条配置时，先以
`engineV4.py --api_config=...` 最小复现，再运行本目录的回归入口验证。
