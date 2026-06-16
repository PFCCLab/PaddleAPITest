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
# REQUIRED_MEMORY=10

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
    # --required_memory="$REQUIRED_MEMORY"
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
