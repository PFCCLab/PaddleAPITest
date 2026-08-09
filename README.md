# PaddleAPITest

PaddleAPITest 是面向 PaddlePaddle API 的配置驱动测试框架。它将真实业务、Paddle CI/CE 和专项场景中的 API 调用序列化为 `api config`，统一执行 API 可用性、Paddle/Torch 精度、重复稳定性、CINN、性能、大 Tensor、0-size Tensor 和自定义设备测试。

完整的命令行参数、YAML 配置键、用户可配置环境变量和内部变量边界见 [CLI 参考](docs/CLI_REFERENCE.md)。

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

配置包含双引号时用单引号包裹 `--api_config`。普通单配置模式最多使用一块 GPU；
`accuracy_dual_gpu` 和 `accuracy_stable_dual_gpu` 单配置使用一对 GPU。未指定
`--gpu_ids` 和 `--num_gpus` 时，两类模式分别默认使用 GPU 0 和 GPU 0/1。

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

`--api_config`、`--api_config_file`、`--api_config_file_pattern` 和下文的 `--retest` 必须且只能选择一个。完整参数以 `python engineV4.py --help` 为准。

### 快速复测分类

已有日志目录可以直接按分类复测，无需修改 checkpoint，也无需再次指定原始配置文件。例如复测全部 `config_input`：

```bash
python engineV4.py \
  --accuracy_stable=True \
  --retest=config_input \
  --log_dir=tester/api_config/test_log \
  --num_gpus=4 \
  --gpu_ids=0-3
```

多个分类用逗号分隔，例如 `--retest=config_input,timeout`。可用分类与 `api_config_*.txt` 对应，包括 `pass`、`skip`、`paddle_error`、`paddle_accuracy`、`paddle_bitwise`、`paddle_cuda`、`paddle_crash`、`oom`、`timeout`、`torch_error`、`config_input`、`config_parse` 和 `config_convert`。

复测开始时，引擎会从 checkpoint、主分类、`comp/` 分类和 stable/tolerance CSV 中移除所选配置的旧结构化结果；`log_inorder.log` 保留历史 case。复测中断后，重新执行相同命令只运行尚未 checkpoint 的配置；全部完成后恢复文件自动删除。`engineV2.py` 支持相同参数。不要让多个进程同时复测同一日志目录。

## 测试模式

每次运行必须且只能启用一种主模式：

| 参数 | 用途 |
| --- | --- |
| `--paddle_only=True` | 执行 Paddle API，检查配置解析和 Paddle 支持情况 |
| `--accuracy=True` | 比较 Paddle 与等价 Torch API 的前向输出和梯度 |
| `--accuracy_dual_gpu=True` | 与 accuracy 等价，每个 worker 使用一张输入/计算卡和一张全量比较卡 |
| `--accuracy_stable=True` | Paddle/Torch 分别执行两轮，同时检查跨框架精度与框架内稳定性 |
| `--accuracy_stable_dual_gpu=True` | 与 accuracy-stable 等价，每个 worker 使用一张计算卡和一张全量比较卡 |
| `--paddle_cinn=True` | 比较 Paddle 动态图与 CINN；可配合 `--test_backward=True` |
| `--paddle_gpu_performance=True` | 测量 Paddle GPU 性能 |
| `--torch_gpu_performance=True` | 测量 Torch GPU 性能 |
| `--paddle_torch_gpu_performance=True` | 对比 Paddle 与 Torch GPU 性能 |
| `--paddle_custom_device=True` | 比较自定义设备与 CPU |
| `--custom_device_vs_gpu=True` | 通过 upload/download 流程比较自定义设备与 GPU |

常用附加参数包括 `--test_amp`、`--test_cpu`、`--atol`、`--rtol`、`--manual_threshold_config_file`、`--bitwise_alignment`、`--timeout`、`--random_seed`、`--generate_failed_tests` 和 `--exit_on_error`。

`--test_cpu=True` 只将 Paddle 框架输入及前向、反向切到 CPU。Torch reference 始终在
GPU 上执行；输入逻辑值生成设备和结果比较设备只由 `--use_gpu_mode` 控制。

## 引擎与运行入口

### engineV4

`engineV4.py` 是推荐入口，提供多 GPU worker slot、异常恢复、结构化日志和 compute-sanitizer。

### engineV2

`engineV2.py` 使用 Pebble `ProcessPool`。除调度方式、engineV4 专属 compute-sanitizer
和 `accuracy_dual_gpu` 外，其测试模式、双卡 accuracy-stable、GPU mode、动态显存管理、
dump 和主要参数与 engineV4 对齐。详见 [engineV2 文档](docs/engineV2-README.md)。

### 其他入口

- `engine.py`、`engineV3.py`：历史或专项兼容入口，新任务优先使用 engineV4。
- `run-v4.sh`：可编辑的 shell 模板，支持前后台启动、状态查询和停止。
- `run.py`：YAML runner，负责环境变量、命令行参数、后台进程和多轮失败重测编排。
- `test_pipeline/V4/`：0-size、1M、big tensor 等标准流水线脚本。

```bash
python run.py -c test_pipeline/run_config.yaml --dry-run
python run.py -c test_pipeline/run_config.yaml
```

模型配置集可以使用 `${APITEST_MODEL}` 占位，示例见 [generic configs 文档](test_pipeline/generic_configs/README.md)。`run.py` 会对 YAML 中所有字符串展开 `${VAR}` 和 `$VAR`，`APITEST_MODEL` 只是项目约定的模型目录变量。

## GPU Mode 与动态显存管理

`--use_gpu_mode=True` 在 GPU 上生成 Tensor 逻辑值并进行结果比较，同时复用 CUDA allocator，
适用于大规模 `accuracy_stable` 测试。它不改变算子设备；此模式会忽略
`--use_cached_numpy=True`。

两个开关正交，四种组合的语义如下：

| `test_cpu` | `use_gpu_mode` | Paddle kernel | Torch reference | 输入生成与比较 |
| --- | --- | --- | --- | --- |
| `False` | `False` | GPU | GPU | CPU |
| `False` | `True` | GPU | GPU | GPU |
| `True` | `False` | CPU | GPU | CPU |
| `True` | `True` | CPU | GPU | GPU |

组合模式 `test_cpu=True,use_gpu_mode=True` 会先在 GPU 生成逻辑输入，再分别物化为
Paddle CPU 输入和 Torch GPU 输入，最后把结果送到 GPU 比较。accuracy/accuracy-stable
始终需要 GPU 运行 Torch reference；`accuracy_stable_dual_gpu` 支持 `test_cpu=True`。

GPU mode 会在输入生成前根据 TensorConfig 元数据和测试模式估算阶段存活集合。单 worker
独占 GPU 时使用整卡容量，多 worker 共享时按 worker 数均分；只有通用下界已经超过容量的
配置才进入 `skip`。输出、output grad 和算子 workspace 不做 API 专项推导，仍由运行时治理兜底。

GPU mode 不需要选择固定显存策略。框架会在 Torch/Paddle 阶段边界查询整卡空闲显存，
按下一阶段输入、已观测输出/梯度和 reference workspace 估算 headroom；有压力时先释放两个
框架的 allocator cache 并重新查询，只有 headroom 仍不足时才将第一轮结果逐棵转移到 CPU。
小 shape 在显存充足时不会执行不必要的 D2H。

`test_tol` 只将容差置零并记录误差诊断，不改变上述比较设备：启用 GPU mode 时仍在
GPU 完成 Tensor 比较。

```bash
python engineV4.py \
  --accuracy_stable=True \
  --use_gpu_mode=True \
  --api_config_file=tester/api_config/8_big_tensor/big_tensor_merged.txt \
  --num_gpus=1 \
  --num_workers_per_gpu=1 \
  --log_dir=tester/api_config/test_log_big_tensor
```

该流程始终保留不可变 CPU 输入快照、四次真实执行、全部稳定性比较和大结果分块比较。

### Accuracy 双卡模式

`--accuracy_dual_gpu=True` 为每个 worker 原子分配一对 GPU，并隐式启用 accuracy 和
GPU mode。逻辑 `gpu:0` 负责 GPU 输入生成；GPU 算子模式下也负责 Torch/Paddle 前后向，
逻辑 `gpu:1` 保存完整输出和输入梯度并执行全量比较。

```bash
python engineV4.py \
  --accuracy_dual_gpu=True \
  --api_config='paddle.add(Tensor([2, 3],"float32"), Tensor([2, 3],"float32"), )' \
  --gpu_ids=0,1 \
  --num_gpus=2 \
  --num_workers_per_gpu=1
```

`test_cpu=True` 时 Paddle 在 CPU 执行，Torch reference 仍在 GPU 0 执行；GPU 0 同时按
GPU mode 生成逻辑输入，Paddle CPU 结果再搬到 GPU 1 比较。每侧 API 只执行一次；前向
比较通过后才执行 Paddle backward。前向结果比较结束后立即释放，避免与后续双侧梯度
长期重叠。该模式不进行 CPU spill、采样或跨卡 autograd，也不能拆分单次 GPU kernel
自身的 workspace 峰值。

GPU 按规范化后的 `--gpu_ids` 顺序两两配对。模式要求至少两张且 GPU 总数为偶数，并要求
`--num_workers_per_gpu=1`；单条配置默认使用 GPU 0/1。

### Accuracy Stable 双卡模式

`--accuracy_stable_dual_gpu=True` 为每个 worker 原子分配一对 GPU。单个进程同时看到两张卡：逻辑 `gpu:0` 负责输入生成、T1/P1/T2/P2 前向与反向，逻辑 `gpu:1` 保存每轮完整输出和输入梯度并执行原有全量比较。

```bash
python engineV4.py \
  --accuracy_stable_dual_gpu=True \
  --use_gpu_mode=True \
  --api_config_file=tester/api_config/8_big_tensor/big_tensor_merged.txt \
  --gpu_ids=0-7 \
  --num_gpus=8 \
  --num_workers_per_gpu=1 \
  --log_dir=tester/api_config/test_log_big_tensor_dual_gpu
```

`--accuracy_stable_dual_gpu=True` 本身就是一种 accuracy-stable 测试模式，并隐式启用 `--use_gpu_mode=True`。如果没有显式传入 GPU mode，引擎会打印参数 warning 后继续执行。GPU 按规范化后的 `--gpu_ids` 顺序两两配对，例如 `--gpu_ids=0,2,5,7` 产生 `(0,2)`、`(5,7)` 两个 worker。该模式要求至少两张且 GPU 总数为偶数，并要求 `--num_workers_per_gpu=1`；单条 `--api_config` 默认使用 GPU 0/1，也可以显式指定任意两张卡。

每次 Torch/Paddle backward 都在计算卡完成，随后将 detach 后的完整 output 和 input grad 搬到比较卡。dual 模式不进行 CPU spill、NumPy CPU fallback、采样、shape 裁剪、分布式 shard 或跨卡 autograd；所有元素仍在比较卡上参与比较。小 Tensor 直接调用 `torch.testing.assert_close`，大 Tensor 在比较卡上使用有界分块工作区完成等价的全量比较。

双卡模式只能释放跨阶段驻留结果造成的计算卡压力；如果任意一次完整 forward/backward 自身已经超过单张计算卡显存，该模式无法将单个算子的 workspace 透明拆到两张卡。

## 并行、日志与恢复

- `--num_gpus=-1` 使用全部选定 GPU；也可指定明确数量。
- `--gpu_ids` 支持 `0`、`0,2`、`0-3` 和 `-1`。
- `--num_workers_per_gpu` 控制每张 GPU 的 worker 上限；实际 worker 总数不会超过 pending case 数，0 pending 时不会启动 worker。
- `--timeout` 是单 case 超时时间，单位为秒。
- `--show_runtime_status=True` 输出实时进度和运行状态。

未指定 `--log_dir` 时，批量运行默认写入 `logs/test_log_<timestamp>`，单条 `--api_config` 默认写入
`logs/test_log_single_<timestamp>`。目录中会保存：

- `checkpoint.txt`：已完成配置，用于续跑时跳过。
- `log_inorder.log`：按完成顺序聚合的 case 日志。
- `api_config_*.txt`：按 pass、Paddle error、accuracy error、OOM、timeout 等终态分类的配置。
- `comp/`、`stable*.csv` 等：精度稳定性各比较维度的结果。

并发任务必须使用不同日志目录。单次分类复测优先使用 `--retest`；多轮分支复测可用 `run.py` 的 `retest`。

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
│   ├── runtime_config.py       # worker 运行配置和 GPU 显存预算
│   └── *_performance.py        # 性能测试实现
└── tools/                      # 配置集、日志和错误分析工具
```

## 开发与扩展

### 开发检查

安装 `pre-commit` 后，每次提交会检查新增的 Python 和 Shell 源码行。新增的非空行中，注释行占比必须至少为 10%；`.txt`、`.yaml`、`.yml`、文档和其他非源码文件不参与统计。检查只计算本次提交新增的行，因此不会被历史代码的注释比例阻塞。

```bash
pre-commit install
pre-commit run check-added-comment-ratio
```

- 新增 Paddle/Torch 映射或 Rule：参见 [Paddle2Torch 文档](tester/paddle_to_torch/README.md)。
- 采集 API 调用配置：参见 [API Tracer 文档](tools/api_tracer/README.md)。
- 整理配置、checkpoint 或错误日志：参见 [Tools 文档](tools/README.md)。
- CINN 专项流水线：参见 [CINN 测试文档](test_pipeline/CINN/README_test_cinn.md)。
