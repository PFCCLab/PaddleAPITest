#!/bin/bash
set -euo pipefail

# ============================================================
# PaddleAPITest 运行脚本
#
# 使用方式:
#   ./<当前脚本名>              正常启动（后台）
#   ./<当前脚本名> --stop       终止上次启动的后台进程
#   ./<当前脚本名> --status     查看运行状态
#
# 配置方法: 修改下方变量 / 注释切换即可
# ============================================================

# ── 引擎选择 ──────────────────────────────────────────────────
ENGINE=engineV4           # engineV2 | engineV4

# ── 运行模式开关 ──────────────────────────────────────────────
FOREGROUND=false          # true=前台运行(调试用，Ctrl+C终止)
DRY_RUN=false             # true=只打印最终命令，不执行

# ── compute-sanitizer（engineV4 only）──────────────────────────
# compute-sanitizer 用于定位 CUDA kernel 的非法访存、race、同步错误等问题。
# engineV4 会为每个 worker slot 启动独立 sanitizer 子进程，保留多 GPU/多 worker 并发。
# engineV2 不支持以下 sanitizer 参数；ENGINE=engineV2 时请保持 USE_COMPUTE_SANITIZER=false，并不要传入 SANITIZER_ARGS。
USE_COMPUTE_SANITIZER=false
SANITIZER_COMMAND="compute-sanitizer --target-processes all --error-exitcode=86"
SANITIZER_ERROR_EXITCODE=86
# --_sanitizer_child 是 engineV4 内部参数，普通运行不要配置。

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
# NUM_GPUS!=0 时，引擎不受外部 "CUDA_VISIBLE_DEVICES" 影响
API_CONFIG=""
FILE_INPUT="tester/api_config/5_accuracy/accuracy_1.txt"
# FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
LOG_DIR="tester/api_config/test_log"

# ── GPU / worker 调度 ─────────────────────────────────────────
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=-1
GPU_IDS="4-7"
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

# ============================================================
# ========== 以下为运行逻辑，通常不需要修改 ====================
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$ENGINE.py" || ! -d "tester" ]]; then
    echo "错误: 请在 PaddleAPITest 项目根目录执行此脚本"
    exit 1
fi
SCRIPT_FILE="${BASH_SOURCE[0]##*/}"
SCRIPT_NAME="${SCRIPT_FILE%.sh}"
RUN_COMMAND="./${SCRIPT_FILE}"
PID_FILE="${SCRIPT_DIR}/.${SCRIPT_NAME}.pid"

# ── 运维命令处理 ──
case "${1:-}" in
    --stop)
        if [[ -f "$PID_FILE" ]]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                # Kill entire process group (main + all workers)
                kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null
                echo "已终止进程组 PGID=$pid"
            else
                echo "进程 PID=$pid 已不存在"
            fi
            rm -f "$PID_FILE"
        else
            echo "未找到 PID 文件，没有正在运行的任务"
        fi
        exit 0
        ;;
    --status)
        if [[ -f "$PID_FILE" ]]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "\033[32m运行中\033[0m  PID=$pid  引擎=$ENGINE"
                # 显示子进程（worker）
                children=$(pgrep -P "$pid" 2>/dev/null | wc -l)
                echo "  Worker 进程数: $children"
                # 显示运行时长
                elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
                echo "  已运行: ${elapsed:-unknown}"
                # 显示日志文件
                log=$(ls -t "$LOG_DIR"/log_*.log 2>/dev/null | head -1)
                [[ -n "${log:-}" ]] && echo "  最新日志: $log"
            else
                echo -e "\033[31m已结束\033[0m  PID=$pid (进程不存在)"
                rm -f "$PID_FILE"
            fi
        else
            echo "无运行记录（PID 文件不存在）"
        fi
        exit 0
        ;;
    --help|-h)
        echo "Usage: ${RUN_COMMAND} [--stop|--status|--help]"
        echo ""
        echo "  (无参数)   启动测试任务"
        echo "  --stop     终止后台任务"
        echo "  --status   查看运行状态"
        echo ""
        echo "配置方法: 编辑脚本顶部变量，注释/取消注释切换参数"
        exit 0
        ;;
    "") ;; # 正常启动
    *)
        echo "未知参数: $1 (使用 --help 查看帮助)"
        exit 1
        ;;
esac

# ── 防重复启动 ──
if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo -e "\033[33m警告: 已有运行中的任务 PID=$old_pid\033[0m"
        echo "使用 ${RUN_COMMAND} --stop 终止后再启动，或删除 $PID_FILE 强制启动"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# ── 组装参数 ──
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

SANITIZER_ARGS=()
if [[ "$ENGINE" == "engineV4" ]]; then
    SANITIZER_ARGS=(
        --use_compute_sanitizer="$USE_COMPUTE_SANITIZER"
        --sanitizer_command="$SANITIZER_COMMAND"
        --sanitizer_error_exitcode="$SANITIZER_ERROR_EXITCODE"
    )
elif [[ "$USE_COMPUTE_SANITIZER" == "true" ]]; then
    echo "错误: compute-sanitizer 参数仅 engineV4 支持"
    exit 1
fi

ALL_ARGS=(
    "${TEST_MODE_ARGS[@]}"
    "${TEST_PARAM_ARGS[@]}"
    "${IN_OUT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${TIME_OUT_ARGS[@]}"
    "${SANITIZER_ARGS[@]}"
)

# ── 打印有效配置 ──
echo "── PaddleAPITest ──────────────────────────"
echo "  引擎:    $ENGINE.py"
echo "  输入:    $FILE_INPUT"
echo "  日志:    $LOG_DIR"
echo "  GPU:     ids=$GPU_IDS  workers/gpu=$NUM_WORKERS_PER_GPU"
echo "  超时:    ${TIME_OUT}s"
echo "  模式:    ${TEST_MODE_ARGS[*]:-<无>}"
echo "  参数:    ${TEST_PARAM_ARGS[*]:-<无>}"
echo "  Sanitizer: enabled=$USE_COMPUTE_SANITIZER exitcode=$SANITIZER_ERROR_EXITCODE command='$SANITIZER_COMMAND'"
echo "────────────────────────────────────────────"

# ── Dry-run 模式 ──
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY-RUN] 最终命令:"
    echo "  python $ENGINE.py ${ALL_ARGS[*]}"
    exit 0
fi

# ── 创建日志目录 ──
mkdir -p "$LOG_DIR" || {
    echo "错误: 无法创建日志目录 '$LOG_DIR'"
    exit 1
}

LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"

# ── 启动 ──
if [[ "$FOREGROUND" == "true" ]]; then
    echo -e "\n\033[36m[前台模式] Ctrl+C 终止\033[0m"
    echo "日志同时写入: $LOG_FILE"
    echo ""
    python "$ENGINE.py" "${ALL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
else
    nohup setsid python "$ENGINE.py" "${ALL_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    PYTHON_PID=$!
    echo "$PYTHON_PID" > "$PID_FILE"

    # 任务自然结束时清理 PID 文件；若已启动新任务，则不误删新 PID。
    (
        while kill -0 "$PYTHON_PID" 2>/dev/null; do
            sleep 5
        done

        recorded_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ "$recorded_pid" == "$PYTHON_PID" ]]; then
            rm -f "$PID_FILE"
        fi
    ) >/dev/null 2>&1 &

    sleep 1
    if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo -e "\033[31m错误: $ENGINE 启动失败\033[0m"
        echo "查看日志: tail -50 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi

    echo ""
    echo -e "\033[32m启动成功! PID=$PYTHON_PID\033[0m"
    echo ""
    echo "常用操作:"
    echo "  查看状态:  ${RUN_COMMAND} --status"
    echo "  终止任务:  ${RUN_COMMAND} --stop"
    echo "  跟踪日志:  tail -f $LOG_FILE"
    echo "  GPU监控:   watch -n 1 nvidia-smi"
    echo ""
    echo "进程已在后台运行，关闭终端不影响执行"
fi
