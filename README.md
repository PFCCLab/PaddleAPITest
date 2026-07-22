# PaddleAPITest

PaddleAPITest 是面向 PaddlePaddle API 的配置驱动测试框架。它将真实业务、Paddle CI/CE 和专项场景中的 API 调用序列化为 `api config`，统一执行 API 可用性、Paddle/Torch 精度、重复稳定性、CINN、性能、大 Tensor、0-size Tensor 和自定义设备测试。

一条配置包含 API、参数、Tensor shape、dtype、place 等执行信息，例如：

```text
paddle.concat(tuple(Tensor([31376, 768],"float32"),Tensor([1, 768],"float32"),), axis=0, )
```

配置可由 Paddle Trace API 或 `tools/api_tracer/` 采集；`tester/paddle_to_torch/` 提供 Paddle 到 Torch 的等价转换。

## 快速开始

### 环境要求

- Linux、Python 3.10+、CUDA 13.0
- PaddlePaddle develop；精度、稳定性和 Torch 性能测试使用 PyTorch 2.12.0

使用 [uv](https://docs.astral.sh/uv/) 创建虚拟环境，随后依次安装 CUDA 13.0 的 Paddle develop、PyTorch 和其余依赖：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu130/
uv pip install torch==2.12.0 torchvision torchaudio -i https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
```

Transformer Engine（TE）仅在相关 FP8/MoE 配置中需要，可选安装：

```bash
uv pip install --no-build-isolation "transformer_engine[pytorch]"
```

### 运行单条配置

推荐使用 `engineV4.py`。Paddle-only 单配置：

```bash
python engineV4.py \
  --paddle_only=True \
  --api_config='paddle.abs(Tensor([1, 100],"float64"), )' \
  --num_gpus=1
```

Paddle/Torch 精度单配置：

```bash
python engineV4.py \
  --accuracy=True \
  --api_config='paddle.abs(Tensor([1, 100],"float64"), )' \
  --num_gpus=1
```

配置包含双引号时用单引号包裹 `--api_config`；单配置模式最多使用一块 GPU。

### 批量运行

```bash
python engineV4.py \
  --accuracy=True \
  --api_config_file=tester/api_config/7_0_size/0_size_tensor_1_8_1.txt \
  --log_dir=tester/api_config/test_log \
  --num_gpus=4 \
  --num_workers_per_gpu=1 \
  --gpu_ids=0-3
```

多个 glob 用逗号分隔：

```bash
python engineV4.py \
  --paddle_only=True \
  --api_config_file_pattern='tester/api_config/7_0_size/*.txt,tester/api_config/8_big_tensor/*.txt' \
  --log_dir=tester/api_config/test_log
```

`--api_config`、`--api_config_file` 和 `--api_config_file_pattern` 必须且只能选择一个。完整参数以 `python engineV4.py --help` 为准。

## 测试模式

每次运行必须且只能启用一种主模式：

| 参数 | 用途 |
| --- | --- |
| `--paddle_only=True` | 执行 Paddle API，检查配置解析和 Paddle 支持情况 |
| `--accuracy=True` | 比较 Paddle 与等价 Torch API 的前向输出和梯度 |
| `--accuracy_stable=True` | Paddle/Torch 分别执行两轮，同时检查跨框架精度与框架内稳定性 |
| `--paddle_cinn=True` | 比较 Paddle 动态图与 CINN；可配合 `--test_backward=True` |
| `--paddle_gpu_performance=True` | 测量 Paddle GPU 性能 |
| `--torch_gpu_performance=True` | 测量 Torch GPU 性能 |
| `--paddle_torch_gpu_performance=True` | 对比 Paddle 与 Torch GPU 性能 |
| `--paddle_custom_device=True` | 比较自定义设备与 CPU |
| `--custom_device_vs_gpu=True` | 通过 upload/download 流程比较自定义设备与 GPU |

常用附加参数包括 `--test_amp`、`--test_cpu`、`--atol`、`--rtol`、`--manual_threshold_config_file`、`--bitwise_alignment`、`--timeout`、`--random_seed`、`--generate_failed_tests` 和 `--exit_on_error`。

## 引擎与运行入口

### engineV4

`engineV4.py` 是推荐入口，提供多 GPU worker slot、异常恢复、结构化日志和 compute-sanitizer。

### engineV2

`engineV2.py` 使用 Pebble `ProcessPool`。除调度方式和 engineV4 专属 compute-sanitizer 外，其测试模式、GPU mode、显存策略、dump 和主要参数与 engineV4 对齐。详见 [engineV2 文档](engineV2-README.md)。

### 其他入口

- `engine.py`、`engineV3.py`：历史或专项兼容入口，新任务优先使用 engineV4。
- `run-v4.sh`：可编辑的 shell 模板，支持前后台启动、状态查询和停止。
- `run.py`：YAML runner，负责环境变量、命令行参数、后台进程和多轮失败重测编排。
- `test_pipeline/V4/`：0-size、1M、big tensor 等标准流水线脚本。

```bash
python run.py -c test_pipeline/run_config.yaml --dry-run
python run.py -c test_pipeline/run_config.yaml
```

模型配置集可以使用 `${APITEST_MODEL}` 占位，示例见 [generic configs 文档](test_pipeline/generic_configs/README.md)。

## GPU Mode 与显存策略

`--use_gpu_mode=True` 在 GPU 上生成 Tensor 并进行比较，复用 CUDA allocator，适用于大规模 `accuracy_stable` 测试。此模式会忽略 `--use_cached_numpy=True`。

进程级环境变量 `PADDLEAPITEST_GPU_MEMORY_POLICY` 控制显存使用的激进程度：

| 值 | 行为 | 建议场景 |
| --- | --- | --- |
| `conservative` | 默认值；首轮输出和梯度转移到 CPU，并在显存压力下释放缓存 | 大 Tensor、未知 shape、优先避免 OOM |
| `aggressive` | 首轮输出和梯度继续驻留 GPU，减少同步和 D2H 开销 | 已知的小 shape 配置集、优先吞吐 |

策略在启动时固定，不按 case 切换或 OOM 自动重试。大 Tensor 使用 `conservative`：

```bash
PADDLEAPITEST_GPU_MEMORY_POLICY=conservative \
python engineV4.py \
  --accuracy_stable=True \
  --use_gpu_mode=True \
  --api_config_file=tester/api_config/8_big_tensor/big_tensor_merged.txt \
  --num_gpus=1 \
  --num_workers_per_gpu=1 \
  --log_dir=tester/api_config/test_log_big_tensor
```

小 shape 配置可用 `aggressive`：

```bash
PADDLEAPITEST_GPU_MEMORY_POLICY=aggressive \
python engineV4.py \
  --accuracy_stable=True \
  --use_gpu_mode=True \
  --api_config_file=tester/api_config/7_0_size/0_size_tensor_1_8_1.txt \
  --log_dir=tester/api_config/test_log_0size
```

两种策略都保留 CPU 输入快照和大结果分块比较。

## 并行、日志与恢复

- `--num_gpus=-1` 使用全部选定 GPU；也可指定明确数量。
- `--gpu_ids` 支持 `0`、`0,2`、`0-3` 和 `-1`。
- `--num_workers_per_gpu` 控制每张 GPU 的 worker 数；GPU mode 和大 Tensor 通常从 1 开始。
- `--timeout` 是单 case 超时时间，单位为秒。
- `--show_runtime_status=True` 输出实时进度和运行状态。

`--log_dir` 中会保存：

- `checkpoint.txt`：已完成配置，用于续跑时跳过。
- `log_inorder.log`：按完成顺序聚合的 case 日志。
- `api_config_*.txt`：按 pass、Paddle error、accuracy error、OOM、timeout 等终态分类的配置。
- `comp/`、`stable*.csv` 等：精度稳定性各比较维度的结果。

并发任务必须使用不同日志目录。失败重测可用 `run.py` 的 `retest`，或使用 [工具文档](tools/README.md) 中的 checkpoint/错误配置工具。

## 调试能力

### 单 API Dump

Dump 保留单条配置的阶段、环境、日志和 Tensor；仅支持 `--api_config` 与 `accuracy`/`paddle_only`：

```bash
python engineV4.py \
  --accuracy=True \
  --api_config='paddle.abs(Tensor([1, 100],"float32"), )' \
  --use_dump=True \
  --dump_dir=tester/api_config/test_log/dump_case \
  --num_gpus=1
```

也可设置 `USE_DUMP=True` 和 `DUMP_DIR=<path>`；优先级为命令行、环境变量、默认值。只设置目录不会启用 dump。

### Compute Sanitizer

engineV4 可为 case 启动 compute-sanitizer，定位 CUDA 非法访存、race 和同步错误：

```bash
python engineV4.py \
  --paddle_only=True \
  --api_config_file=configs.txt \
  --use_compute_sanitizer=True \
  --sanitizer_command='compute-sanitizer --target-processes all --error-exitcode=86'
```

该能力仅由 engineV4 提供。`--_sanitizer_child` 是内部参数，不应手工设置。

## 配置集

`tester/api_config/` 保存配置集、配置处理脚本和默认日志目录。主要分类包括：

| 目录 | 内容 |
| --- | --- |
| `1_not_support/` | 当前不支持的配置 |
| `2_paddle_only_random/` | 具有随机创建或随机计算行为的 Paddle-only 配置 |
| `3_paddle_only/` | 可由 Paddle 执行但尚不支持 Paddle/Torch 精度转换的配置 |
| `4_paddle_only_amp/`、`6_accuracy_amp/` | AMP 专项配置 |
| `monitor_config/accuracy/` | Paddle/Torch 精度巡检配置 |
| `7_0_size/` | 含 0 维 shape 的配置 |
| `8_big_tensor/` | 派生的大 Tensor 配置 |
| `9_getset_item/` | Tensor getitem/setitem 专项配置 |
| `10_performance/` | 性能测试配置 |
| `CI_CE_config/` | 从 Paddle CI/CE 采集的配置 |
| `big_and_0size/` | 大 Tensor 与 0-size 综合配置 |

配置文件每行一个 api config；派生、合并、去重和筛选脚本位于 `tester/api_config/` 与 `tools/`。

## 项目结构

```text
PaddleAPITest/
├── engineV4.py                 # 推荐测试引擎
├── engineV2.py                 # Pebble ProcessPool 引擎
├── run.py                      # YAML runner
├── run-v4.sh                   # shell 运行模板
├── test_pipeline/              # 标准流水线、YAML 配置和脚本
├── tester/
│   ├── api_config/             # 配置集、解析、日志和 dump
│   ├── paddle_to_torch/        # Paddle API 到 Torch 的转换规则
│   ├── accuracy.py             # Paddle/Torch 精度测试
│   ├── accuracy_stable.py      # 跨框架精度和重复执行稳定性
│   ├── base.py                 # 测试基类、输入生成与比较
│   ├── runtime_config.py       # worker 运行配置和 GPU 显存策略
│   └── *_performance.py        # 性能测试实现
└── tools/                      # 配置集、日志和错误分析工具
```

## 开发与扩展

- 新增 Paddle/Torch 映射或 Rule：参见 [Paddle2Torch 文档](tester/paddle_to_torch/README.md)。
- 采集 API 调用配置：参见 [API Tracer 文档](tools/api_tracer/README.md)。
- 整理配置、checkpoint 或错误日志：参见 [Tools 文档](tools/README.md)。
- CINN 专项流水线：参见 [CINN 测试文档](test_pipeline/CINN/README_test_cinn.md)。
