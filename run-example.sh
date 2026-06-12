#!/bin/bash

# Script to run engineV2.py
# Usage: ./run.sh

# 配置参数
FILE_INPUT="tester/api_config/5_accuracy/accuracy_1.txt"
# FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
LOG_DIR="tester/api_config/test_log"

# NUM_GPUS!=0 时，engineV2 不受外部 "CUDA_VISIBLE_DEVICES" 影响
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=-1
GPU_IDS="4-7"
# REQUIRED_MEMORY=10
TIME_OUT=600

# 测试模式
TEST_MODE_ARGS=(
    # Paddle vs Torch 正确性对比
    --accuracy=True
    # Paddle 单框架执行
    # --paddle_only=True
    # Paddle 动态图 vs CINN；test_backward 仅此模式生效
    # --paddle_cinn=True
    # 性能测试
    # --paddle_gpu_performance=True
    # --torch_gpu_performance=True
    # --paddle_torch_gpu_performance=True
    # 稳定性测试
    # --accuracy_stable=True
)

# 测试参数
TEST_PARAM_ARGS=(
    # 混合精度
    # --test_amp=True
    # CPU 路径
    # --test_cpu=True
    # CPU numpy 缓存；gpu_cache_mode 下自动关闭
    # --use_cached_numpy=True
    # GPU tensor 缓存 + GPU compare；增加显存驻留
    # --use_gpu_cache_mode=True
    # 对比阈值；bitwise_alignment 会将阈值置 0
    # --atol=1e-2
    # --rtol=1e-2
    # --bitwise_alignment=True
    # accuracy 容差诊断，保留 CPU compare
    # --test_tol=True
    # 仅 paddle_cinn 生效
    # --test_backward=True
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
TIME_OUT_ARGS=(
    --timeout="$TIME_OUT"
)
mkdir -p "$LOG_DIR" || {
    echo "错误：无法创建日志目录 '$LOG_DIR'"
    exit 1
}

# 执行程序
LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"
nohup python engineV2.py \
        "${TEST_MODE_ARGS[@]}" \
        "${TEST_PARAM_ARGS[@]}" \
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

exit 0

# watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv