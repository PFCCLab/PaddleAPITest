# gpu_accuracy_tolerance 测试流程

### 1. 准备测试环境

1. 建议在虚拟环境或 docker 中进行开发，并正确安装 python 与 nvidia 驱动，engineV2 建议使用 **python>=3.10**

2. PaddlePaddle 与 PyTorch 的部分依赖项可能发生冲突，请先安装 **paddlepaddle-gpu** 再安装 **torch**，重新安装请添加 `--force-reinstall` 参数

3. 安装 paddlepaddle-gpu

   - [使用 pip 快速安装 paddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
   ```
   - 若需要本地编译 Paddle，可参考：[Linux 下使用 ninja 从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile-by-ninja.html)

4. 安装 torch

   - [使用 pip 快速安装 torch](https://pytorch.org/get-started/locally/)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
5. 安装第三方库

   ```bash
   pip install pebble nvidia-ml-py pandas
   ```

### 2. 准备测试集

accuracy 的配置集位于目录：`tester/api_config/5_accuracy`

```bash
tester/api_config/5_accuracy
├── accuracy_1.txt
├── accuracy_2.txt
├── accuracy_3.txt
├── accuracy_4.txt
├── accuracy_5.txt
├── accuracy_6.txt
├── accuracy_7.txt
├── accuracy_8.txt
├── accuracy_cpu_error.txt
├── accuracy_cpu_kernel.txt
├── accuracy_gpu_error_dtype_diff.txt
├── accuracy_gpu_error_grads_diff.txt
├── accuracy_gpu_error.txt
└── accuracy_gpu_error_uncertain.txt

### 3. 准备测试脚本

`run-example.sh` 是与 engineV2 配套的执行脚本，可以非常方便地修改测试参数并执行测试

复制 `run-example.sh`，重命名为 `run_tol.sh`
```bash
cp run-example.sh run_tol.sh
```

以测试 gpu accuracy tolerance 为例，文件内容可修改为：
```bash
#!/bin/bash

# Script to run engineV2.py
# Usage: ./run.sh

# export PYTHONPATH=/root/paddlejob/Paddle/build/python

# 配置参数
# NUM_GPUS!=0 时，engineV2 不受外部 "CUDA_VISIBLE_DEVICES" 影响
# FILE_INPUT="tester/api_config/5_accuracy/accuracy_1.txt"
FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
LOG_DIR="tester/api_config/test_log_tol"
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=-1
GPU_IDS="-1"

TEST_MODE_ARGS=(
    --accuracy=True
    # --paddle_only=True
    # --paddle_cinn=True
    # --test_amp=True
    # --test_cpu=True
    # --use_cached_numpy=True
    # --atol=1e-2
    # --rtol=1e-2
    --record_accuracy_tolerance=True
)

IN_OUT_ARGS=(
    # --api_config_file="$FILE_INPUT"
    --api_config_file_pattern="$FILE_PATTERN"
    --log_dir="$LOG_DIR"
)

PARALLEL_ARGS=(
    --num_gpus="$NUM_GPUS"
    --num_workers_per_gpu="$NUM_WORKERS_PER_GPU"
    --gpu_ids="$GPU_IDS"
)

mkdir -p "$LOG_DIR" || {
    echo "错误：无法创建日志目录 '$LOG_DIR'"
    exit 1
}

# 执行程序
LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"
nohup python engineV2.py \
        "${TEST_MODE_ARGS[@]}" \
        "${IN_OUT_ARGS[@]}" \
        "${PARALLEL_ARGS[@]}" \
        >> "$LOG_FILE" 2>&1 &

PYTHON_PID=$!

sleep 1
if ! ps -p "$PYTHON_PID" > /dev/null; then
    echo "错误：engineV2 启动失败，请检查 $LOG_FILE"
    exit 1
fi

echo -e "\n\033[32m执行中... 另开终端运行监控:\033[0m"
echo -e "1. GPU使用:   watch -n 1 nvidia-smi"
echo -e "2. 日志目录:  ls -lh $LOG_DIR"
echo -e "3. 详细日志:  tail -f $LOG_FILE"
echo -e "4. 终止任务:  kill $PYTHON_PID"
echo -e "\n进程已在后台运行，关闭终端不会影响进程执行"

exit 0

# watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
```

### 4. 执行测试

若不使用 `run_tol.sh`，以测试 gpu accuracy tolerance 为例，可直接执行以下命令：（建议使用 nohup 避免终端终止时停止主进程）
```bash
python engineV2.py --api_config_file_pattern="tester/api_config/5_accuracy/accuracy_*.txt" --accuracy=True --record_accuracy_tolerance=True --num_gpus=-1 --num_workers_per_gpu=-1 --log_dir="tester/api_config/test_log_tol" >> "tester/api_config/test_log_tol/log.log" 2>&1
```

或者直接运行 run_tol.sh：
```bash
# chmod +x run_tol.sh
./run_tol.sh
```

最终的所有测试结果会保存在 `tester/api_config/test_log_tol` 目录下，包括：
- 检查点文件 checkpoint.txt
- 以 api_config_ 开头的配置集文件
- 测试日志文件 log.log
- 详细测试结果文件 log_inorder.log
- 精度数据文件 tol.csv

### 4. 继续测试

apitest 拥有检查点 checkpoint 机制，保存了所有已经测试过的配置。若希望继续测试，可直接运行脚本，无需重新测试已经测过的配置，**切勿删除测试结果目录**

若存在 skip、oom、crash、timeout 等异常配置，且希望重新测试它们，可使用 `tools/retest_remover.py` 小工具：
- 修改 `TEST_LOG_PATH` 为 `tester/api_config/test_log_tol`
- （可选）决定 `LOG_PREFIXES` 中需要重测的配置集
- 直接运行

即可删去 checkpoint.txt 中的配置，然后继续测试

### 5. 整理测试结果

若需要整理出具精度报告，可使用 `tools/tol_csv_stat.py` 小工具：
- 修改 `TEST_LOG_PATH` 为 `tester/api_config/test_log_tol`
- 直接运行

即可在原测试结果目录中生成以下文件：
- `tol_full.csv`, 包含所有精度测试结果（如果仅有一个 tol*.csv 则去重排序，多个则合并）
- `tol_stat.csv`，包含所有 API, dtype, mode（前反向）级别的精度范围统计
- `tol_stat_api.csv`，包含所有 API, dtype, mode（前反向）级别的精度测试计数
