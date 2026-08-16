# PaddleAPITest 命令行参考

本文档说明 `engineV4.py`、`run.py`、Shell 模板和回归脚本的用户接口。直接运行 API config 使用 `engineV4.py`；需要 YAML、后台管理或多轮复测时使用 `run.py`。V2 的差异见 [engineV2 运行指南](engineV2-README.md)。

```bash
python engineV4.py --help
python run.py --help
```

## `engineV4.py`

```text
python engineV4.py INPUT MODE [OPTIONS]
```

`INPUT` 和 `MODE` 表示必须选择的输入方式和主模式，并非字面参数。每次运行必须选择一个输入和一个主模式。布尔参数接受 `True`/`False`、`1`/`0`、`yes`/`no`、`y`/`n`。

#### `-h`, `--help`

打印参数摘要并退出。

### 输入

#### `--api_config=CONFIG`

直接运行一条 API config。配置含双引号时用单引号包裹。普通单条模式使用一张 GPU，双卡模式使用一对 GPU。默认：空。

#### `--api_config_file=PATH`

从文件逐行读取 API config。默认：空。

#### `--api_config_file_pattern=GLOB[,GLOB...]`

读取一个或多个逗号分隔 glob 匹配的文件。默认：空。

#### `--retest=CLASS[,CLASS...]`

重跑 `--log_dir` 中已有的分类，例如 `config_input,timeout`；开始前会清除这些 case 的旧结构化结果。默认：空。

`--api_config`、`--api_config_file`、`--api_config_file_pattern`、`--retest` 互斥，必须且只能选择一个。

#### `--log_dir=PATH`

日志、checkpoint、分类配置和比较结果目录。批量默认 `logs/test_log_<timestamp>`；`--api_config` 默认 `logs/test_log_single_<timestamp>`。

### 主模式

本节必须且只能选择一个选项。

#### `--paddle_only=True`

执行 Paddle API，检查配置能否解析和运行。

#### `--accuracy=True`

比较 Paddle 与对应 Torch API 的结果和可用梯度。

#### `--accuracy_dual_gpu=True`

每个 worker 使用计算卡和完整结果比较卡。隐式启用 accuracy 和 GPU mode；要求至少两张且为偶数张 GPU，并要求 `--num_workers_per_gpu=1`。

#### `--accuracy_stable=True`

Paddle 和 Torch 各执行两次，检查跨框架精度和框架内稳定性。

#### `--accuracy_stable_dual_gpu=True`

稳定性双卡模式。隐式启用 stable accuracy 和 GPU mode；GPU 与 worker 限制同 `accuracy_dual_gpu`。

#### `--paddle_cinn=True`

比较 Paddle 动态图和 CINN；配合 `--test_backward=True` 可检查反向。

#### `--paddle_gpu_performance=True`、`--torch_gpu_performance=True`、`--paddle_torch_gpu_performance=True`

分别测量 Paddle、Torch，或比较两者 GPU 性能。

#### `--paddle_custom_device=True`、`--custom_device_vs_gpu=True`

分别比较自定义设备与 CPU、自定义设备与 GPU。后者用 `--custom_device_vs_gpu_mode` 选择传输方向。

### 执行与比较

#### `--num_gpus=N`

使用的 GPU 数；`-1` 使用全部选中 GPU。默认：`-1`。

#### `--gpu_ids=IDS`

选择 GPU，支持 `0`、`0,2`、`0-3`、`-1`。默认：空。

#### `--num_workers_per_gpu=N`

每卡最大 worker 数。GPU mode 下 `-1` 表示每卡一个 worker；双卡模式必须为 `1`。默认：`1`。

#### `--test_cpu=True`

Paddle 前向和反向在 CPU 执行，Torch reference 仍在 GPU。输入生成和比较设备仍由 `--use_gpu_mode` 决定。默认：`False`。

#### `--use_gpu_mode=True`

在 GPU 生成输入和比较结果，并复用 CUDA allocator；Paddle 算子和 Torch reference 的执行设备由 `--test_cpu` 决定。显式 NumPy backend 会保留 CPU logical value，运行头会显示最终 backend/device。默认：`False`。

#### `--use_cached_numpy=True`、`--test_amp=True`

`--use_cached_numpy=True` 复用 NumPy output grad；非 GPU mode 选择 NumPy input backend，GPU mode
选择模式默认 backend 并打印 warning。
`--test_amp=True` 启用自动混合精度检查。默认：`False`。

#### `--atol=FLOAT`、`--rtol=FLOAT`

绝对和相对精度阈值。默认均为 `0.01`。

#### `--accuracy_manual_threshold_config=PATH`

先执行 `atol=rtol=0.0` 的严格比较；若失败且 API 在该 YAML 的
`manual_threshold_config` 中，则按其 `[atol, rtol]` 复核。复核通过的配置写入
`api_config_paddle_bitwise_knows.txt` 并按已知不完全对齐跳过；默认：空。

#### `--record_accuracy_tolerance=True`、`--test_backward=True`

前者启用 accuracy 容差诊断，比较设备沿用 `--use_gpu_mode` 配置；后者只在 `paddle_cinn` 中启用反向检查。默认：`False`。

#### `--random_seed=N`

设置 NumPy 随机种子。默认：`0`。

#### `--custom_device_vs_gpu_mode={upload,download}`

设置自定义设备与 GPU 比较的数据传输方向。默认：`upload`。

#### `--bitwise_alignment=True`

将比较阈值设为零，进行 bitwise 对齐。默认：`False`。

#### `--timeout=SECONDS`、`--show_runtime_status=True`

单 case 最大执行时间，默认 `1800` 秒；后者控制实时进度输出，设为 `False` 时只打印失败 case，默认 `True`。

### 诊断

#### `--use_dump=True|False`、`--dump_dir=PATH`

启用单条配置 dump 并设置输出目录。命令行优先于 `USE_DUMP`、`DUMP_DIR`；未设置时 dump 关闭，空目录值使用 `tester/api_config/test_log/dump_case`。

#### `--use_compute_sanitizer=True`

所有 case 通过一个常驻 compute-sanitizer session 顺序运行；同一 worker 的正常 case 复用 Python/Paddle/CUDA runtime，session crash 或 timeout 后才重建。仅 engineV4 支持。默认：`False`。

#### `--sanitizer_command=COMMAND`、`--sanitizer_error_exitcode=N`

Sanitizer 命令前缀和报错退出码。默认分别为 `compute-sanitizer --target-processes all --error-exitcode=86` 和 `86`。

session 入口为内部参数，禁止手工设置。

## `run.py`

```text
python run.py [-c CONFIG] [OVERRIDES] [ENGINE_OPTIONS]
```

默认配置为 `test_pipeline/run_config.yaml`。`run.py` 校验 YAML、展开环境变量、构造引擎命令，并支持前后台运行。

### 任务与进程

#### `-c FILE`, `--config FILE`

选择 YAML 任务文件。

#### `--stop`、`--status`、`--dry-run`

终止后台任务、查询记录的进程状态，或只打印最终命令和环境。

#### `--foreground`、`--background`、`--engine {engineV2,engineV4}`

覆盖 YAML 的 `runner.foreground` 或 `runner.engine`。

### 输入和引擎覆盖

#### `--api-config VALUE`、`-i FILE`、`--input FILE`、`--api-config-file FILE`、`--api-config-file-pattern GLOB`

分别覆盖 `input.api_config`、`input.api_config_file`、`input.api_config_file_pattern`。

#### `-o DIR`、`--output DIR`、`--log-dir DIR`

覆盖 `output.log_dir`。

#### `--timeout N`、`--num-gpus N`、`--num-workers-per-gpu N`、`--gpu-ids IDS`

覆盖对应的 `engine_args`。

#### `--accuracy_manual_threshold_config FILE`

覆盖 `engine_args.accuracy_manual_threshold_config`。

#### `--set-env KEY=VALUE`

写入或覆盖 YAML `env`，可重复。只设置下文列出的用户可配置环境变量。

#### `--engine-arg KEY=VALUE`

写入或覆盖已声明的 `engine_args`，可重复。值会解析为布尔、整数、浮点数或字符串。

未知顶层参数及 `--` 之后的参数会原样透传给引擎。

### YAML

顶层键为 `name`、`runner`、`env`、`input`、`output`、`retest`、`engine_args`。

- `runner`：`engine`、`foreground`、`dry_run`、`pid_file`。
- `input`：`api_config`、`api_config_file`、`api_config_file_pattern` 三选一。
- `output`：`log_dir`。
- `retest`：`enabled`、`rounds`、`error_configs`、`log_dir_template`、`skip_unavailable`。
- `engine_args`：去掉 `--` 的 engineV4 参数名。

所有字符串都经 `os.path.expandvars` 展开，支持 `${VAR}` 和 `$VAR`；未设置的变量保持原文。

## 环境变量

### 用户可配置

#### `PADDLEAPITEST_IMPL`

选择 Paddle-to-Torch 参考实现：`torch`、`te`、`apex`。未设置时每个 Rule 使用自身默认值；Rule 只会采用其支持的实现。

#### `PADDLEAPITEST_INPUT_BACKEND`

选择输入生成 backend：`numpy`、`torch`、`paddle`。未指定时由测试模式选择默认值；普通直接调用默认 NumPy，Paddle-only/CINN/Paddle performance 默认 Paddle，accuracy/Torch performance 默认 Torch。非法值会在参数阶段失败。GPU mode 下显式 NumPy 会显示为 CPU logical value。

#### `CUDA_VISIBLE_DEVICES`、`CUDA_HOME`、`CUDA_PATH`

前者限制 CUDA 可见设备，但 worker 绑定时会由引擎重设；后两者用于 dump 元数据定位 CUDA。

#### Paddle Flags

`FLAGS_use_system_allocator`、`FLAGS_check_cuda_error`、`FLAGS_alloc_fill_value`、`FLAGS_check_nan_inf`、`FLAGS_use_accuracy_compatible_kernel` 必须在 Paddle 启动前设置。模板和流水线 YAML 提供常用取值，应以当前 Paddle 版本支持的语义为准。

#### `APITEST_MODEL`

generic configs 使用的模型目录变量。`run.py` 支持任意 `${VAR}` 或 `$VAR`，不限于该名称。

#### 回归脚本

`tools/regression/regression_runner.sh` 支持：`REGRESSION_CONFIG_FILE`（默认 `tools/regression/regression_configs.txt`）、`REGRESSION_LOG_DIR`（默认时间戳目录）、`PYTHON`（默认 `python`）、`GPU_IDS`（默认 `-1`）、`REGRESSION_NUM_GPUS`（默认 `-1`）、`REGRESSION_WORKERS_PER_GPU`（默认 `4`）、`REGRESSION_TIMEOUT`（默认 `180` 秒）。

#### `API_DERIVE_OUTPUT_ROOT`

`tools/derive_api_*` 的输出根目录，默认 `generated/api_traces`。

### 内部变量：禁止配置

以下变量用于 worker 协议或临时工具路径，不是稳定用户接口，禁止在 Shell 或 `run.py` YAML 中设置：

```text
PADDLEAPITEST_WORKER_SLOT
PADDLEAPITEST_WORKERS_ON_GPU
PADDLEAPITEST_SUPPRESS_CASE_TAGS
PADDLEAPITEST_NP_FALLBACK
TEST_NON_CONTIGUOUS
USE_GPU_INPUT_CACHE
GPU_INPUT_CACHE_MIN_NUMEL
GPU_INPUT_CACHE_APIS
CACHED_NUMPY_MAX_ENTRIES
LOG_CACHED_NUMPY
RECORD_CACHE_EVENTS
SKIP_GPU_CLEANUP
```

引擎还会向 worker 注入 `USE_DUMP`、`DUMP_DIR`、`USE_CACHED_NUMPY`、`USE_GPU_MODE`、`CUDA_VISIBLE_DEVICES` 以传递 CLI/YAML 值；这些变量是内部状态，不要手动伪造 worker 状态。

### 第三方运行时变量

`PYTHONPATH`、`PYTORCH_CUDA_ALLOC_CONF`、`HF_HOME`、`LD_PRELOAD`、`TE_CUBLASLT_PRELOAD`、`NVIDIA_TF32_OVERRIDE` 属于 Python、CUDA、PyTorch、Transformers 或 Transformer Engine；仅在对应运行时或专项集成明确需要时配置。

## Shell 模板

`run-example.sh`、`test_pipeline/V4/*.sh` 顶部的 `ENGINE`、`FOREGROUND`、`DRY_RUN`、`FILE_INPUT`、`FILE_PATTERN`、`LOG_DIR`、`NUM_GPUS`、`NUM_WORKERS_PER_GPU`、`GPU_IDS`、`TIME_OUT` 是脚本变量，不是引擎环境变量。模板会将它们转换为 CLI 参数。

模板支持 `--stop`、`--status`、`--help`。设置 `FOREGROUND=true` 前台运行，设置 `DRY_RUN=true` 仅输出最终命令。
