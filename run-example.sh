#!/bin/bash

# Script to run engineV2.py
# Usage: ./<当前脚本名>

SCRIPT_FILE="${BASH_SOURCE[0]##*/}"
RUN_COMMAND="./${SCRIPT_FILE}"

if [[ ! -f "engineV2.py" || ! -d "tester" ]]; then
    echo "错误: 请在 PaddleAPITest 项目根目录执行此脚本"
    exit 1
fi

# ── Paddle Flags ──────────────────────────────────────────────
# 这些环境变量在启动 Paddle 前生效，用于控制 Paddle 运行时行为。
# - FLAGS_use_system_allocator: 使用系统 allocator，便于释放内存和定位问题。
# - FLAGS_check_cuda_error: 更积极检查 CUDA 错误。
# - FLAGS_alloc_fill_value / FLAGS_check_nan_inf: 用于发现未初始化值、NaN/Inf 等数值问题。
# - FLAGS_use_accuracy_compatible_kernel: 使用更偏精度兼容的 kernel；默认关闭，避免改变常规性能/行为基线。
export FLAGS_use_system_allocator=true
export FLAGS_check_cuda_error=true
export FLAGS_alloc_fill_value=255
export FLAGS_check_nan_inf=true
# export FLAGS_use_accuracy_compatible_kernel=true

# ── 输入输出 ──────────────────────────────────────────────────
# input 三选一：--api_config / --api_config_file / --api_config_file_pattern
# API_CONFIG=""
FILE_INPUT="tester/api_config/5_accuracy/accuracy_1.txt"
# FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
LOG_DIR="tester/api_config/test_log"

# ── GPU / worker 调度 ─────────────────────────────────────────
# NUM_GPUS!=0 时，engineV2 不受外部 "CUDA_VISIBLE_DEVICES" 影响。
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=1
GPU_IDS="-1"
# REQUIRED_MEMORY=10
TIME_OUT=600

# ── 测试模式：必须且只能启用一种 ───────────────────────────────
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
    # 自定义设备对比
    # --paddle_custom_device=True
    # --custom_device_vs_gpu=True
)

# ── 测试参数 ──────────────────────────────────────────────────
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
    # --manual_threshold_config_file="tester/api_config/manual_threshold.yaml"
    # --bitwise_alignment=True
    # accuracy 容差诊断，保留 CPU compare
    # --test_tol=True
    # 仅 paddle_cinn 生效
    # --test_backward=True
    # 随机种子；非默认值时会设置 numpy seed
    # --random_seed=0
    # custom_device_vs_gpu 上传/下载模式
    # --custom_device_vs_gpu_mode=upload
    # 生成失败 case 的可复现测试文件
    # --generate_failed_tests=True
    # paddle_error 时立即退出
    # --exit_on_error=True
    # 控制运行时进度输出
    # --show_runtime_status=True
)

IN_OUT_ARGS=(
    # --api_config="$API_CONFIG"
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

# ── engineV4 only 参数说明 ─────────────────────────────────────
# 以下参数仅 engineV4.py 支持，engineV2.py 不要传入：
# - --use_compute_sanitizer=True: 为每个 case 通过 compute-sanitizer 子进程运行，定位 CUDA 访存/同步等问题。
# - --sanitizer_command="compute-sanitizer --target-processes all --error-exitcode=86"
# - --sanitizer_error_exitcode=86
# - --_sanitizer_child=True: engineV4 内部子进程参数，普通运行不要配置。

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
echo -e "4. 终止任务:  kill $PYTHON_PID  # ${RUN_COMMAND} 没有 --stop 管理逻辑"
echo -e "\n进程已在后台运行，关闭终端不会影响进程执行"

exit 0

# watch -n 1 nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
