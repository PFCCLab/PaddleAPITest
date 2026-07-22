# CINN big tensor 测试流程

### 1. 准备测试环境

1. 建议在虚拟环境或 docker 中进行开发，并正确安装 python 与 nvidia 驱动，engineV2 建议使用 **python>=3.10**

2. 克隆本仓库

    ```bash
    git clone https://github.com/PFCCLab/PaddleAPITest.git
    cd PaddleAPITest
    ```

3. 安装 paddlepaddle-gpu 与 torch

   - [使用 pip 快速安装 paddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
   ```

    - [使用 pip 快速安装 torch](https://pytorch.org/get-started/locally/)，或者运行命令 (cuda>=11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   - 若需要本地编译 Paddle，可参考：[Linux 下使用 ninja 从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile-by-ninja.html)
   - PaddlePaddle 与 PyTorch 的部分依赖项可能发生冲突，请先安装 **paddlepaddle-gpu** 再安装 **torch**，重新安装请添加 `--force-reinstall` 参数

4. 安装第三方库

   ```bash
   pip install pandas pebble nvidia-ml-py pyyaml
   ```

### 2. 准备测试集

big tensor 的配置集位于：`tester/api_config/8_big_tensor/big_tensor_merged.txt`

该文件为同文件夹下其他 **有效**配置 的 **合并去重**

### 3. 准备测试脚本

`run-example.sh` 是与 engineV2 配套的执行脚本，可以方便地修改测试参数并执行测试

复制 `run-example.sh`，重命名为 `run_cinn.sh` (可直接使用 `test_pipeline/CINN/run_cinn.sh` )
```bash
cp run-example.sh run_cinn.sh
```

文件内容修改为：
```bash
#!/bin/bash

# Script to run engineV2.py
# Usage: ./run.sh

# 配置参数
# NUM_GPUS!=0 时，engineV2 不受外部 "CUDA_VISIBLE_DEVICES" 影响
FILE_INPUT="tester/api_config/8_big_tensor/big_tensor_merged.txt"
# FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
LOG_DIR="tester/api_config/test_log_cinn"
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=1 # 建议单卡单进程
GPU_IDS="-1"

TEST_MODE_ARGS=(
    # --accuracy=True
    # --paddle_only=True
    --paddle_cinn=True # CINN 测试
    # --paddle_gpu_performance=True
    # --torch_gpu_performance=True
    # --paddle_torch_gpu_performance=True
    # --accuracy_stable=True
    # --test_amp=True
    # --test_cpu=True
    --use_cached_numpy=True # 开启 numpy 缓存
    # --atol=1e-2
    # --rtol=1e-2
    # --test_tol=True
    # --test_backward=True # 需要测试反向时，取消此注释
)

IN_OUT_ARGS=(
    --api_config_file="$FILE_INPUT"
    # --api_config_file_pattern="$FILE_PATTERN"
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

> [!NOTE]
> `--paddle_cinn` 模式默认不测试反向，如果要开启反向测试，请打开 `--test_backward` 参数

### 4. 执行测试

直接运行 `run_cinn.sh`：
```bash
# chmod +x run_cinn.sh
./run_cinn.sh
# test_pipeline/CINN/run_cinn.sh
```

或者直接执行以下命令（建议使用 nohup 避免终端终止时停止主进程）：
```bash
python engineV2.py --api_config_file="tester/api_config/8_big_tensor/big_tensor_merged.txt" --paddle_cinn=True --num_gpus=-1 --log_dir="tester/api_config/test_log_cinn" >> "tester/api_config/test_log_cinn/log.log" 2>&1
```

最终的所有测试结果会保存在 `tester/api_config/test_log_cinn` 目录下，包括：
- 检查点文件 checkpoint.txt
- 以 api_config_ 开头的配置集文件
- 测试日志文件 log.log
- 详细测试结果文件 log_inorder.log

### 5. 继续测试

PaddleAPITest 拥有检查点 checkpoint 机制，保存了所有已经测试过的配置。若希望中止测试，可直接杀死主进程；若希望继续测试，可直接重新运行脚本，无需重新测试已经测过的配置，**切勿删除测试结果目录**

若存在 skip、oom、crash、timeout 等异常配置，且希望重新测试它们，可使用 `tools/retest_remover.py` 小工具：
```bash
python tools/retest_remover.py -p tester/api_config/test_log_cinn -r skip oom
```

即可删去 checkpoint.txt 中的配置，然后继续测试

### 6. 整理测试结果

若需要整理出具 error 报告，可使用 `tools/error_stat/error_stat.py` 小工具：
```bash
python tools/error_stat/error_stat.py -i tester/api_config/test_log_cinn -o tester/api_config/test_log_cinn_filtered -s
```

即可在输出目录中生成以下文件（* 代表配置集前缀，如 pass、paddle_error、crash）
- 有效配置
  - `*_api.txt`
  - `*_config.txt`
  - `*_log.log`
- 无效配置
  - `invalid_*_api.txt`
  - `invalid_*_config.txt`
  - `invalid_*_log.txt`
