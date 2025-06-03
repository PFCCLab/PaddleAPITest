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
    pip install func_timeout pebble pynvml
    ```
2. 克隆 PaddleAPITest 仓库并进入项目目录
   ```bash
   git clone https://github.com/PFCCLab/PaddleAPITest.git
   cd PaddleAPITest
   ```
3. 确保 `engineV2.py` 、 `log_writer.py` 和 `run.sh` 路径正确

## 使用指南

### 命令行参数

| 参数                        | 类型  | 说明                                                                 |
| --------------------------- | ----- | -------------------------------------------------------------------- |
| `--api_config`              | str   | API 配置字符串（单条测试）                                           |
| `--api_config_file`         | str   | API 配置文件路径（如`tester/api_config/5_accuracy/accuracy_1.txt`）  |
| `--api_config_file_pattern` | str   | API 配置文件模式（如 `tester/api_config/5_accuracy/accuracy_*.txt`） |
| `--paddle_only`             | bool  | 仅运行 Paddle 测试（默认 False）                                     |
| `--accuracy`                | bool  | 启用精度测试（默认 False）                                           |
| `--paddle_cinn`             | bool  | 运行 CINN vs Dygraph 对比测试（默认 False）                          |
| `--num_gpus`                | int   | 使用的 GPU 数量（默认 0，-1 动态最大）                               |
| `--num_workers_per_gpu`     | int   | 每 GPU 的 worker 进程数（默认 1，-1 动态最大）                       |
| `--gpu_ids`                 | str   | 使用的 GPU 序号，以逗号分隔（默认 ""，"-1" 动态最大）                 |
| `--required_memory`         | float | 每 worker 进程预估使用显存（默认 10.0）                              |
| `--test_amp`                | bool  | 启用自动混合精度测试（默认 False）                                   |
| `--test_cpu`                | bool  | 启用 Paddle CPU 模式测试（默认 False）                               |
| `--use_cached_numpy`        | bool  | 启用 Numpy 缓存（默认 False）                                        |
| `--log_dir`                 | str   | 日志输出路径（默认 "tester/api_config/test_log"）                    |

### 示例命令

以精度测试为例，配置文件路径为 `tester/api_config/api_config_temp.txt`，输出日志路径为 `tester/api_config/test_log`：

**多进程多 GPU 模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt" --num_gpus=8 --num_workers_per_gpu=1 >> "tester/api_config/test_log/log.log" 2>&1
```

**单进程多 GPU 模式**：
```bash
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt" --num_gpus=0 >> "tester/api_config/test_log/log.log" 2>&1
```

**单进程单 GPU 模式**：
```bash
export CUDA_VISIBLE_DEVICES=""
python engineV2.py --accuracy=True --api_config_file="tester/api_config/api_config_temp.txt" --num_gpus=0 >> "tester/api_config/test_log/log.log" 2>&1
```

**使用 run.sh 脚本**：
```bash
# chmod +x run.sh
./run.sh
```
该脚本使用参数：NUM_GPUS=-1, NUM_WORKERS_PER_GPU=-1，在后台运行程序，可在修改 `run.sh` 参数后使用

## 监控方法

执行 `run.sh` 后可通过以下方式监控：

- **GPU使用情况**：`watch -n 1 nvidia-smi`
- **日志文件**：`ls -lh tester/api_config/test_log`
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
