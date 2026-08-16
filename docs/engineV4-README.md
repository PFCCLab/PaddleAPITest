# engineV4 运行指南

`engineV4.py` 是 PaddleAPITest 的推荐执行入口。它读取一条或多条 `api config`，在
Paddle、Torch、CINN 或自定义设备上执行测试，并将结果写入结构化日志。engineV4 适合
单条配置定位问题，也适合多 GPU 批量巡检；完整参数定义见 [CLI 参考](CLI_REFERENCE.md)。

## 运行前提

- Linux、Python 3.10 或更高版本。
- 可用的 PaddlePaddle（GPU 模式需 Paddle GPU 版本）和 CUDA 运行时。
- `accuracy`、`accuracy_stable` 及 Torch 性能模式需要 PyTorch；CINN 模式需要对应的
  CINN 运行环境。
- 在仓库根目录运行，确保 `tester/`、`sanitizer_session.py` 等本地模块可导入。

项目依赖安装示例见根目录 [README](../README.md)。先确认设备和运行时可用：

```bash
python -c "import paddle; print(paddle.__version__)"
nvidia-smi
python engineV4.py --help
```

## 基本约定

每次运行必须同时指定：

1. 一个输入源：`--api_config`、`--api_config_file`、`--api_config_file_pattern` 或
   `--retest`。
2. 一个主模式：例如 `--paddle_only=True` 或 `--accuracy=True`。

输入源互斥，主模式也互斥。布尔参数接受 `True`/`False`、`1`/`0`、`yes`/`no`、`y`/`n`。
配置文件按行读取，只保留以 `paddle.` 开头的非空行；重复配置会去重。

## 快速开始

### 单条配置

单条模式用于最快速的复现。普通模式默认使用 GPU 0；双卡模式默认使用 GPU 0 和 1。
配置中包含双引号时，外层使用单引号：

```bash
python engineV4.py \
  --paddle_only=True \
  --api_config='paddle.abs(Tensor([1, 100],"float32"), )' \
  --num_gpus=1
```

Paddle/Torch 精度比较：

```bash
python engineV4.py \
  --accuracy=True \
  --api_config='paddle.abs(Tensor([1, 100],"float32"), )' \
  --gpu_ids=0 --num_gpus=1
```

普通单条模式只能使用一张 GPU；`--accuracy_dual_gpu=True` 和
`--accuracy_stable_dual_gpu=True` 必须使用一对 GPU。

### 批量配置

```bash
python engineV4.py \
  --accuracy=True \
  --api_config_file=tester/api_config/7_0_size/0_size_tensor_1_8_1.txt \
  --gpu_ids=0-3 \
  --num_gpus=4 \
  --num_workers_per_gpu=1 \
  --log_dir=tester/api_config/test_log
```

多个 glob 用逗号分隔：

```bash
python engineV4.py \
  --paddle_only=True \
  --api_config_file_pattern='tester/api_config/7_0_size/*.txt,tester/api_config/8_big_tensor/*.txt'
```

未指定 `--log_dir` 时，批量模式默认写入 `logs/test_log_<timestamp>`，单条模式默认写入
`logs/test_log_single_<timestamp>`。

## 测试模式

| 参数 | 含义 |
| --- | --- |
| `--paddle_only=True` | 只执行 Paddle，检查配置解析和 Paddle API 支持情况。 |
| `--accuracy=True` | 比较 Paddle 与对应 Torch API 的前向结果和梯度。 |
| `--accuracy_stable=True` | 重复执行 Paddle/Torch，检查跨框架精度和框架内稳定性。 |
| `--accuracy_dual_gpu=True` | accuracy 的双卡版本；每个 worker 使用计算卡和完整结果比较卡。 |
| `--accuracy_stable_dual_gpu=True` | accuracy-stable 的双卡版本。 |
| `--paddle_cinn=True` | 比较 Paddle 动态图与 CINN；反向检查另加 `--test_backward=True`。 |
| `--paddle_gpu_performance=True` | 测量 Paddle GPU 性能。 |
| `--torch_gpu_performance=True` | 测量 Torch GPU 性能。 |
| `--paddle_torch_gpu_performance=True` | 对比 Paddle 与 Torch GPU 性能。 |
| `--paddle_custom_device=True` | 比较自定义设备与 CPU。 |
| `--custom_device_vs_gpu=True` | 比较自定义设备与 GPU，方向由 `--custom_device_vs_gpu_mode` 选择。 |

不要在一次命令中混用多个主模式；例如 `--paddle_only` 和 `--accuracy` 必须拆成两次运行。

## GPU 与并发

`--gpu_ids` 支持 `0`、`0,2`、`0-3` 和 `-1`；`-1` 表示由引擎选择全部可见 GPU。
`--num_gpus=-1` 表示使用全部选定 GPU，默认值为 `-1`。`--num_workers_per_gpu` 控制
每卡 worker 上限，默认 `1`；在 `--use_gpu_mode=True` 下，`-1` 表示每卡一个 worker。

批量调度会先按 NVML 的真实空闲显存估算 case 准入预算，再创建或复用 worker。worker
启动从较低并发开始，连续成功后逐步扩容；显存不足的 case 会延迟重试而不是立即把同一张卡
超卖。worker 超时、崩溃或外部终止后，case 会回到待派发队列，日志中的终态只结算一次。

双卡模式把 GPU 按规范化后的 `--gpu_ids` 顺序两两配对，例如 `0,2,5,7` 形成 `(0,2)`
和 `(5,7)`。它要求 GPU 数量为偶数、至少两张，并且 `--num_workers_per_gpu=1`：

```bash
python engineV4.py \
  --accuracy_dual_gpu=True \
  --api_config_file=tester/api_config/8_big_tensor/big_tensor_merged.txt \
  --gpu_ids=0,2,5,7 --num_gpus=4 --num_workers_per_gpu=1
```

双卡只缓解跨阶段结果驻留造成的显存压力；单个 kernel 或 workspace 本身超过计算卡容量时，
仍然会失败。

### `test_cpu` 与 `use_gpu_mode`

`--test_cpu=True` 只把 Paddle kernel 的前向/反向切到 CPU；accuracy 模式的 Torch reference
仍在 GPU。`--use_gpu_mode=True` 把输入生成和结果比较放到 GPU，并复用 CUDA allocator，
算子执行设备由 `--test_cpu` 决定。两者可以组合：

| `test_cpu` | `use_gpu_mode` | Paddle kernel | Torch reference | 输入/比较 |
| --- | --- | --- | --- | --- |
| `False` | `False` | GPU | GPU | CPU |
| `False` | `True` | GPU | GPU | GPU |
| `True` | `False` | CPU | GPU | CPU |
| `True` | `True` | CPU | GPU | GPU |

`--use_cached_numpy=True` 在非 GPU mode 启用 NumPy 输入 backend；GPU mode 使用模式默认 backend
并打印 warning。输入 backend 也可用环境变量 `PADDLEAPITEST_INPUT_BACKEND`
选择 `numpy`、`torch` 或 `paddle`。

## 续跑与复测

批量运行会在 `--log_dir/checkpoint.txt` 记录已完成配置。使用同一日志目录重新运行时，
已 checkpoint 的配置会跳过，适合中断后续跑。不要让多个进程同时写同一个日志目录。

按日志分类复测，不需要重新指定配置文件：

```bash
python engineV4.py \
  --accuracy_stable=True \
  --retest=config_input,timeout \
  --log_dir=tester/api_config/test_log \
  --gpu_ids=0-3 --num_gpus=4
```

常用分类包括 `pass`、`skip`、`paddle_error`、`paddle_accuracy`、`paddle_cuda`、
`paddle_crash`、`oom`、`timeout`、`torch_error`、`config_input`、`config_parse` 和
`config_convert`。复测开始前，引擎会清理所选 case 的旧结构化结果；`log_inorder.log` 会
保留已完成 case 记录。

## 输出与调试

日志目录通常包含：

- `checkpoint.txt`：已完成配置，用于续跑。
- `log_inorder.log`：按完成顺序聚合的 case 日志。
- `api_config_*.txt`：按终态分类的配置列表。
- `comp/`、`stable*.csv`：accuracy 和稳定性比较结果。

单条配置 dump 只支持 `--api_config` 与 `--paddle_only`/`--accuracy`：

```bash
python engineV4.py \
  --accuracy=True \
  --api_config='paddle.abs(Tensor([1, 100],"float32"), )' \
  --use_dump=True --dump_dir=tester/api_config/test_log/dump_case \
  --num_gpus=1
```

CUDA 非法访存、race 或同步问题可使用常驻 compute-sanitizer session：

```bash
python engineV4.py \
  --paddle_only=True \
  --api_config_file=configs.txt \
  --use_compute_sanitizer=True \
  --sanitizer_command='compute-sanitizer --target-processes all --error-exitcode=86'
```

该 session 入口由 engineV4 内部管理，不要手工传入隐藏参数 `--_sanitizer_session`。
`PADDLEAPITEST_GPU_PRESSURE_TIMEOUT_SECONDS` 可调整批量显存压力下无进展的保护超时，默认
`600` 秒；必须是有限的非负数。

## 常见问题

### `no accelerator devices were found`

当前模式需要 GPU，但 CUDA/Paddle 未发现设备。检查 `nvidia-smi`、CUDA 环境和
`CUDA_VISIBLE_DEVICES`；纯 CPU 模式只适用于不需要 GPU runtime 的测试路径。

### 双卡参数校验失败

确认 GPU 数为偶数、至少两张，`--num_workers_per_gpu=1`，且 `--num_gpus` 与
`--gpu_ids` 的数量一致。GPU ID 不能重复，也不能把 `-1` 与显式 ID 混用。

### 结果显示 `oom` 或 `timeout`

先用单条 `--api_config` 复现，再降低 `--num_workers_per_gpu`、缩小配置或启用
`--use_gpu_mode=True`。批量任务可直接用 `--retest=oom,timeout` 重跑失败分类。

### 精度差异

先确认 `--atol`/`--rtol` 和 `--random_seed`，需要严格比较时使用
`--bitwise_alignment=True`。针对已知 API 的差异，可用 `--accuracy_manual_file` 做严格失败
后的按 API 二次阈值复核。

## 回归验证

修改 engineV4 或运行时逻辑后，按仓库约定执行：

```bash
tools/regression/regression_runner.sh
```

环境缺少 GPU、Paddle、Torch 或其他依赖时无法运行回归；此时至少执行
`python engineV4.py --help` 检查参数解析，并在报告中说明缺失依赖。项目回归配置和覆盖方式
见 [回归测试文档](../tools/regression/README.md)。
