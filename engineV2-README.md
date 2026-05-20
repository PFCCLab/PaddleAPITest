# engineV2：高性能多进程测试框架

`engineV2.py` 是为 PaddleAPITest 项目设计的高性能测试框架，支持多 GPU 并行执行，具备负载均衡、超时处理和崩溃恢复能力。相比原始的 `engine.py` 实现，它能显著提升 Paddle API 配置测试效率，加速比约为 5-10 倍。

## 功能特性

- **多 GPU 并行**：拥有灵活的 *gpus 相关参数*，可配置多 GPU 并行测试，支持任务跨 GPU 动态分发
- **进程级并行**：基于 Pebble 库的 ProcessPool 进程池实现，支持进程的高效并行，每张 GPU 可拥有多个 worker
- **动态负载均衡**：新的子进程自动分配至负载最轻 GPU，计算资源最优利用
- **超时/崩溃恢复**：由张量大小推断执行时限（梯度阈值），主动杀死 coredump 进程，自动检测并重启死亡进程

## 其他优化

- **进程安全日志**：`log_writer.py` 采用无争用日志，进程日志可靠聚合
- **优雅关闭**：绑定中断信号，安全终止所有子进程
- **延迟导入**：采用类型提示和模块惰性加载，隔离子进程初始化环境
- **自动化执行**：`run.sh` 脚本一键执行，支持 nohup 后台运行和显存监控

## 安装说明

1. 安装 PaddleAPITest 项目依赖项
    ```bash
    pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install func_timeout pandas pebble pynvml pyyaml
    ```
2. 克隆 PaddleAPITest 仓库并进入项目目录
   ```bash
   git clone https://github.com/PFCCLab/PaddleAPITest.git
   cd PaddleAPITest
   ```
3. 确保 `engineV2.py` 、 `log_writer.py` 和 `run.sh` 路径正确

> [!CAUTION] CAUTION 1
> 目前 engineV2 仅支持 ***python>=3.10***，如报错 *`NameError: name 'torch' is not defined`*，请在 `run_test_case()` 函数首行手动添加导入语句：
> ```python
> import torch
> import paddle
> from tester import (APIConfig, APITestAccuracy, APITestCINNVSDygraph,
>                     APITestPaddleOnly)
> ```
> 
> 问题的原因在于：多进程的 **spawn** 启动方法（start method）与旧版 Python 中函数序列化 (pickling) 存在机制缺陷
> 
> 在 Python 3.10 之前的版本中，当子进程的执行函数（如 `run_test_case`）在主进程中被定义并通过序列化传递到子进程时，该函数会试图在其原始定义时的全局命名空间（即主进程的全局命名空间）中寻找依赖项（如 `torch`），而不是在当前执行时的全局命名空间（即已经通过 `init_worker_gpu` 初始化过的子进程命名空间）中寻找
> 
> 由于 gpu 隔离的需求，主进程并没有导入 `torch` 和 `paddle`，所以当 `run_test_case` 在子进程中被反序列化并准备执行时，它找不到这些库，从而引发 `NameError`

> [!CAUTION] CAUTION 2
> 在更高 CUDA 版本下，比如 CUDA Version: 12.9，可能会出现以下报错：
> ```bash
> ImportError: /usr/local/cuda/lib64/libcusparse.so.12: undefined symbol: __nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12
> ```
> 
> 解决方案是在 `init_worker_gpu()` 函数中交换 `torch` 和 `paddle` 的导入顺序，将 `torch` 置于 `paddle` 之前，具体原因未知

## 使用指南

### 命令行参数

| 参数                             | 类型  | 说明                                                                                   |
| -------------------------------- | ----- | -------------------------------------------------------------------------------------- |
| `--api_config`                   | str   | API 配置字符串（单条测试）                                                             |
| `--api_config_file`              | str   | API 配置文件路径（如`tester/api_config/5_accuracy/accuracy_1.txt`）                    |
| `--api_config_file_pattern`      | str   | API 配置文件 glob 模式，以逗号分隔（如 `tester/api_config/5_accuracy/accuracy_*.txt`） |
| `--paddle_only`                  | bool  | 运行 Paddle 测试（默认 False）                                                         |
| `--accuracy`                     | bool  | 运行 Paddle vs Torch 精度测试（默认 False）                                            |
| `--paddle_cinn`                  | bool  | 运行 CINN vs Dygraph 对比测试（默认 False）                                            |
| `--paddle_gpu_performance`       | bool  | 运行 Paddle 性能测试（默认 False）                                                     |
| `--torch_gpu_performance`        | bool  | 运行 Torch 性能测试（默认 False）                                                      |
| `--paddle_torch_gpu_performance` | bool  | 运行 Paddle vs Torch 性能测试（默认 False）                                            |
| `--accuracy_stable`              | bool  | 启用稳定性测试（默认 False）                                                           |
| `--paddle_custom_device`         | bool  | 运行Custom Devic e或者XPU的API与CPU的精度对比（默认 False）                              |
| `--num_gpus`                     | int   | 使用的 GPU 数量（默认 -1，-1 动态最大）                                                |
| `--num_workers_per_gpu`          | int   | 每 GPU 的 worker 进程数（默认 1，-1 动态最大）                                         |
| `--gpu_ids`                      | str   | 使用的 GPU 序号，以逗号分隔或横线范围（默认 ""，"-1" 动态最大）                        |
| `--required_memory`              | float | 每 worker 进程预估使用显存 GB（默认 10.0）                                             |
| `--test_amp`                     | bool  | 启用自动混合精度测试（默认 False）                                                     |
| `--test_cpu`                     | bool  | 启用 Paddle CPU 模式测试（默认 False）                                                 |
| `--use_cached_numpy`             | bool  | 启用 Numpy 缓存（默认 False）                                                          |
| `--log_dir`                      | str   | 日志输出路径（默认 "tester/api_config/test_log"）                                      |
| `--atol`                         | float | 精度测试的绝对误差容忍度，仅在启用 `--accuracy` 时有效（默认 1e-2）                    |
| `--rtol`                         | float | 精度测试的相对误差容忍度，仅在启用 `--accuracy` 时有效（默认 1e-2）                    |
| `--test_tol`                     | bool  | 启用精度误差容忍度范围测试，仅在启用 `--accuracy` 时有效（默认 False）                 |
| `--test_backward`                | bool  | 启用反向测试，仅在启用 `--paddle_cinn` 时有效（默认 False）                            |
| `--timeout`                      | int   | 单个测试用例执行超时秒数（默认 1800）                                                  |
| `--show_runtime_status`          | bool  | 是否实时显示当前的测试进度（默认 True）                                               |
| `--random_seed`                  | int   | numpy random的随机种子(默认为0，此时不会显式设置numpy random的seed)                   |
| `--custom_device_vs_gpu`        | bool  | 启用自定义设备与GPU的精度对比测试模式（默认 False）                                   |
| `--custom_device_vs_gpu_mode`   | str   | 自定义设备与GPU对比的模式：`upload`、`download` 或 `http`（默认 `upload`）             |
| `--bitwise_alignment`            | bool  | 是否进行诸位对齐对比，开启后所有的api的精度对比都按照atol=0.0,rtol = 0.0的精度对比结果(默认False)|
| `--generate_failed_tests`        | bool  | 是否为失败的测试用例生成可复现的测试文件。开启后，当测试失败时，会在`failed_tests`目录下生成独立的Python测试文件，便于后续复现和调试（默认False）|
| `--exit_on_error`                | bool  | 是否在精度测试出现`paddle_error`或者 `accuracy_error`  错误时立即退出测试进程(exit code 为1)。默认为False，测试进程会继续执行 |

### 示例命令

以精度测试为例，配置文件路径为 `tester/api_config/api_config_tmp.txt`，输出日志路径为 `tester/api_config/test_log`：

**多 GPU 多进程模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" >> "tester/api_config/test_log/log.log" 2>&1
```

```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" --num_gpus=4 --num_workers_per_gpu=2 --gpu_ids="4,5,6,7" >> "tester/api_config/test_log/log.log" 2>&1
```

**多 GPU 单进程模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" --num_gpus=2 >> "tester/api_config/test_log/log.log" 2>&1
```

```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" --gpu_ids="0,1" >> "tester/api_config/test_log/log.log" 2>&1
```

**单 GPU 单进程模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_tmp.txt" --num_gpus=1 --gpu_ids="7" >> "tester/api_config/test_log/log.log" 2>&1
```

**使用 run.sh 脚本**：
```bash
# chmod +x run.sh
./run.sh
```
该脚本使用参数：`NUM_GPUS=-1, NUM_WORKERS_PER_GPU=-1, GPU_IDS="4,5,6,7"`，在后台运行程序，可在修改 `run.sh` 参数后使用

### 自定义设备与 GPU 精度对比测试

该功能支持跨设备的精度对比测试，提供两种数据传输方式：**BOS 云存储中转**和 **HTTP 直连**。

#### 方式一：BOS 云存储中转（upload/download 模式）

分为两步操作：先在一台机器上 upload，再在另一台机器上 download 对比。

**工作流程**：

1. **Upload 模式（GPU 侧）**：在 GPU 上执行 Paddle API，保存 Forward 输出和 Backward 梯度为 PDTensor 文件，上传到 BOS
2. **Download 模式（XPU/其他设备侧）**：在目标设备上执行相同 API，从 BOS 下载 GPU 参考数据，进行精度对比

**配置文件**：编辑 `tester/bos_config.yaml`

```yaml
bos_path: "xly-devops/liujingzong/"
bos_conf_path: "./conf"
bcecmd_path: "./bcecmd"
```

**命令示例**：

```bash
# GPU 侧：执行并上传
python engineV2.py --custom_device_vs_gpu=True \
  --custom_device_vs_gpu_mode=upload \
  --random_seed=1210 \
  --api_config_file="./test1.txt" \
  --gpu_ids=7

# XPU 侧：下载并对比
python engineV2.py --custom_device_vs_gpu=True \
  --custom_device_vs_gpu_mode=download \
  --random_seed=1210 \
  --api_config_file="./test1.txt" \
  --gpu_ids=7
```

#### 方式二：HTTP 直连（http 模式）

只需在本地执行一条命令，自动触发远端执行并拉取结果对比，无需 BOS 服务。

**工作流程**：

1. 在远端机器启动 HTTP 服务器（带多 GPU 进程池）
2. 本地发送 API 配置到远端，远端执行后返回 PDTensor 结果
3. 本地执行同一 API，与远端结果进行精度对比

**第一步：在远端机器启动服务器**

```bash
cd /path/to/PaddleAPITest
python -m tester.http_server --host 0.0.0.0 --port 8089 --num_gpus=-1
```

服务器参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8089` | 监听端口 |
| `--num_gpus` | `-1` | GPU 数量，`-1` 表示全部 |
| `--num_workers_per_gpu` | `1` | 每张 GPU 的 worker 数 |
| `--required_memory` | `10.0` | 每个 worker 最低显存（GB） |
| `--gpu_ids` | `""` | 指定 GPU，如 `6,7` 或 `0-3` |
| `--timeout` | `1800` | 单个 API 执行超时（秒） |
| `--admin_token` | `""` | 若非空，启用 `/admin/*` 管理接口（见下方"远程代码同步"）|

可通过健康检查确认服务状态：

```bash
curl http://<远端IP>:8089/health
# {"status": "ok", "device_type": "gpu", "paddle_version": "3.0.0"}
```

**第二步：在本地配置远端地址**

编辑 `tester/http_config.yaml`：

```yaml
remote_host: "10.78.119.13"   # 远端机器 IP
remote_port: 8089               # 远端服务端口
timeout: 300                    # 单次请求超时（秒）
```

**第三步：在本地执行对比**

```bash
# 单个 API 测试
python engineV2.py --custom_device_vs_gpu=True \
  --custom_device_vs_gpu_mode=http \
  --random_seed=42 \
  --api_config='paddle.abs(Tensor([2, 3], "float32"))'

# 批量测试
python engineV2.py --custom_device_vs_gpu=True \
  --custom_device_vs_gpu_mode=http \
  --random_seed=42 \
  --api_config_file_pattern="tester/api_config/5_accuracy/*.txt" \
  --num_gpus=-1
```

**AMP（自动混合精度）模式**：

加入 `--test_amp=True` 后，本地设备侧和远端 GPU server 侧会**同步**在 `paddle.amp.auto_cast()` 上下文下执行，确保两端精度对比处于相同的混合精度环境中：

```bash
python engineV2.py --custom_device_vs_gpu=True \
  --custom_device_vs_gpu_mode=http \
  --test_amp=True \
  --random_seed=42 \
  --api_config_file="tester/api_config/6_accuracy_amp/accuracy_amp.txt"
```

`test_amp` 标志会随请求 payload 一并发送到 server，因此无需在 server 启动命令中做任何额外配置。

**并发机制**：

服务端通过三层机制处理并发请求：
- **ThreadingHTTPServer**：每个请求一个线程，并行接收
- **Semaphore**：限制同时排队+执行的请求数为 `worker数 × 2`，超出则阻塞等待
- **ProcessPool**：实际执行进程数由 GPU 数和 workers_per_gpu 决定

当客户端 worker 多于服务端 worker 时，多余的请求会排队等待，客户端的 `http_timeout` 作为最终兜底——超时后写入 `timeout` 日志，确保不会出现"没跑也没日志"的情况。

> 详细的设计原理和异常处理说明见 [docs/http_cross_device_comparison.md](docs/http_cross_device_comparison.md)。

#### 远程代码同步（sync_watch）

在两台机器间 SSH 不通的情况下，可通过 `scripts/sync_watch.py` 将本地代码变更**自动同步**到远端服务器，并触发服务器重启，无需手动操作。

**前提：远端启动时带上 `--admin_token`**

```bash
python -m tester.http_server --host 0.0.0.0 --port 8089 --gpu_ids=6 --admin_token=your_token
```

**本地安装依赖（一次性）**

```bash
pip install watchdog
```

**本地启动监听**

```bash
python scripts/sync_watch.py \
  --host <远端IP> --port 8089 --token your_token
```

启动后，每当本地 `.py` 文件被保存，脚本会在约 1.5 秒防抖后：
1. 将变更文件通过 `POST /admin/upload_file` 推送到远端
2. 调用 `POST /admin/restart` 触发服务器原地重启
3. 轮询 `/health` 直至服务器重新就绪

可选参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--host` | 必填 | 远端服务器 IP |
| `--port` | `8089` | 远端服务端口 |
| `--token` | 必填 | 与 `--admin_token` 一致 |
| `--watch_dir` | 仓库根目录 | 本地监听目录 |
| `--debounce` | `1.5` | 防抖时间（秒）|

## 监控方法

执行 `run.sh` 后可通过以下方式监控：

- **GPU使用情况**：`watch -n 1 nvidia-smi`
- **日志目录**：`ls -lh tester/api_config/test_log`
- **详细日志**：`tail -f tester/api_config/test_log/log.log`
- **终止进程**：`kill <PYTHON_PID>`（PID 由 run.sh 显示）
- **进程情况**：`watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv`

## 核心组件

### engineV2.py

*主工作流*：
- 解析命令行参数配置执行模式
- 批量模式下读取 `--api_config_file` 配置，跳过已完成测试
- 支持 `--api_config` 执行单条配置测试

*多 GPU 执行*：
- 初始化 ProcessPool 进程池，根据参数动态分配 GPU 和 worker 数量
- 通过 `init_worker_gpu()` 将进程绑定到GPU，使用共享的 manager.dict 跟踪分配状态
- 以 20000 个配置为批次处理任务，优化内存和 I/O 效率

*任务执行*：
- `run_test_case()` 执行单个测试用例，根据参数选择测试类：*APITestAccuracy*、*APITestPaddleOnly*、*APITestCINNVSDygraph*
- 使用 `log_writer.write_to_log()` 记录检查点和错误

*超时处理*：
- `estimate_timeout()` 根据张量元素数量设置超时（≤1M 元素 90 秒，>100M 元素 3600 秒的梯度阈值）
- 记录并统计失败任务（超时/崩溃），自动重启进程

*清理机制*：
- `cleanup()` 终止进程池，5 秒超时等待 worker 结束
- 信号处理器（SIGINT、SIGTERM）会清理 GPU 显存（调用 torch 和 paddle 的 empty_cache）后优雅退出

### log_writer.py

- 自动创建 test_log 目录，支持安全删除
- 提供无争用的 `write_to_log()` 和 `read_log()` 方法
- 通过 `aggregate_logs()` 将各进程临时日志合并为单一文件，保持追加写入特性
- 记录已完成配置，避免重复测试

### run.sh

- 配置参数：NUM_GPUS=-1, NUM_WORKERS_PER_GPU=-1, LOG_DIR=tester/api_config/test_log
- 使用 nohup 后台运行 engineV2.py，输出重定向至 log.log
- 显示 GPU 监控、日志查看和进程终止命令

## 测试结果

- 在数十万个配置上验证正确性，精度测试速度达原始引擎的 5-10 倍
- 可处理大张量（平均每 GPU 约 20GB，峰值 70GB），在多进程多 GPU、单进程多 GPU、单 GPU 模式下均表现稳定

## 注意事项

1. `estimate_timeout()` 梯度阈值粒度较粗，可进一步调整 TIMEOUT_STEPS
2. 安装 PaddlePaddle（develop） 与 PyTorch（2.6） 需确保兼容性，必须首先安装 PaddlePaddle 再安装 PyTorch

## 许可协议

遵循 PaddleAPITest 仓库的许可条款
