#!/bin/bash

# Script to run engineV2.py
# Usage: ./run.sh

# 配置参数
# NUM_GPUS!=0 时，engineV2 不受外部 "CUDA_VISIBLE_DEVICES" 影响
FILE_INPUT="report/0size_tensor_gpu/error_config.txt"
# FILE_PATTERN="report/0size_tensor_gpu/error_config.txt"
LOG_DIR="tester/api_config/test_log_gpu_0size_regr"
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=15
GPU_IDS="-1"
REQUIRED_MEMORY=5
TIME_OUT=600

TEST_MODE_ARGS=(
	--accuracy=True
	# --paddle_only=True
    # --paddle_cinn=True
	# --test_amp=True
	# --test_cpu=True
	# --use_cached_numpy=True
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
    --required_memory="$REQUIRED_MEMORY"
)
TIME_OUT_ARGS=(
    --timeout="$TIME_OUT"
)

mkdir -p "$LOG_DIR" || {
    echo "错误：无法创建日志目录 '$LOG_DIR'"
    exit 1
}

# 执行程序
LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"
if [ "$backprocess" -eq 1 ]; then
    nohup python engineV2.py \
            "${TEST_MODE_ARGS[@]}" \
            "${IN_OUT_ARGS[@]}" \
            "${PARALLEL_ARGS[@]}" \
            "${TIME_OUT_ARGS[@]}" \
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
else
    python engineV2.py \
            "${TEST_MODE_ARGS[@]}" \
            "${IN_OUT_ARGS[@]}" \
            "${PARALLEL_ARGS[@]}" \
            "${TIME_OUT_ARGS[@]}" \
            2>&1 | tee -a "$LOG_FILE"

    PYTHON_PID=$!

    sleep 1
    if ! ps -p "$PYTHON_PID" > /dev/null; then
        echo "错误：engineV2 执行失败，请检查 $LOG_FILE"
        exit 1
    fi
fi

exit 0

# watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv