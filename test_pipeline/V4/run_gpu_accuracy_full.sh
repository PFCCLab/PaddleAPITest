#!/bin/bash
set -euo pipefail

# ============================================================
# PaddleAPITest V4 运行脚本
#
# 使用方式:
#   ./run_gpu_accuracy_full.sh              正常启动（后台）
#   ./run_gpu_accuracy_full.sh --stop       终止上次启动的后台进程
#   ./run_gpu_accuracy_full.sh --status     查看运行状态
# ============================================================

# ── 引擎选择 ──────────────────────────────────────────────────
ENGINE=engineV4

# ── 运行模式开关 ──────────────────────────────────────────────
FOREGROUND=false
DRY_RUN=false

# ── compute-sanitizer ─────────────────────────────────────────
USE_COMPUTE_SANITIZER=false
SANITIZER_COMMAND="compute-sanitizer --target-processes all --error-exitcode=86"
SANITIZER_ERROR_EXITCODE=86

# ── Paddle Flags ──────────────────────────────────────────────
export FLAGS_use_system_allocator=true
export FLAGS_check_cuda_error=true
export FLAGS_alloc_fill_value=255
export FLAGS_check_nan_inf=true

# ── 输入输出 ──────────────────────────────────────────────────
# NUM_GPUS!=0 时，engineV4 不受外部 "CUDA_VISIBLE_DEVICES" 影响
# FILE_INPUT="tester/api_config/5_accuracy/accuracy_1.txt"
# FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
FILE_PATTERN="tester/api_config/monitor_config/accuracy/GPU/monitoring_configs*.txt"
LOG_DIR="tester/api_config/test_log_gpu_accuracy_full"

# ── GPU 调度 ──────────────────────────────────────────────────
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=-1
GPU_IDS="-1"
# REQUIRED_MEMORY=10

# ── 测试模式 ──────────────────────────────────────────────────
TEST_MODE_ARGS=(
    --accuracy=True
    --atol=0.0
    --rtol=0.0
    --bitwise_alignment=True
    # --paddle_only=True
    # --paddle_cinn=True
    # --test_amp=True
    # --test_cpu=True
    # --use_cached_numpy=True
)

# ============================================================
# ========== 以下为运行逻辑，通常不需要修改 ====================
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "$ENGINE.py" || ! -d "tester" ]]; then
    echo "错误: 请在 PaddleAPITest 项目根目录执行此脚本"
    exit 1
fi

SCRIPT_NAME="${BASH_SOURCE[0]##*/}"
SCRIPT_NAME="${SCRIPT_NAME%.sh}"
PID_FILE="${SCRIPT_DIR}/.${SCRIPT_NAME}.pid"

case "${1:-}" in
    --stop)
        if [[ -f "$PID_FILE" ]]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
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
                children=$(pgrep -P "$pid" 2>/dev/null | wc -l)
                echo "  Worker 进程数: $children"
                elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
                echo "  已运行: ${elapsed:-unknown}"
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
        echo "Usage: ./${BASH_SOURCE[0]##*/} [--stop|--status|--help]"
        echo ""
        echo "  (无参数)   启动测试任务"
        echo "  --stop     终止后台任务"
        echo "  --status   查看运行状态"
        exit 0
        ;;
    "") ;;
    *)
        echo "未知参数: $1 (使用 --help 查看帮助)"
        exit 1
        ;;
esac

if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo -e "\033[33m警告: 已有运行中的任务 PID=$old_pid\033[0m"
        echo "使用 ./${BASH_SOURCE[0]##*/} --stop 终止后再启动，或删除 $PID_FILE 强制启动"
        exit 1
    fi
    rm -f "$PID_FILE"
fi

IN_OUT_ARGS=(
    # --api_config_file="$FILE_INPUT"
    --api_config_file_pattern="$FILE_PATTERN"
    --log_dir="$LOG_DIR"
)

PARALLEL_ARGS=(
    --num_gpus="$NUM_GPUS"
    --num_workers_per_gpu="$NUM_WORKERS_PER_GPU"
    --gpu_ids="$GPU_IDS"
    # --required_memory="$REQUIRED_MEMORY"
)

SHOW_RUNTIME_STATUS_ARGS=(
    --show_runtime_status=False
)

SANITIZER_ARGS=(
    --use_compute_sanitizer="$USE_COMPUTE_SANITIZER"
    --sanitizer_command="$SANITIZER_COMMAND"
    --sanitizer_error_exitcode="$SANITIZER_ERROR_EXITCODE"
)

ALL_ARGS=(
    "${TEST_MODE_ARGS[@]}"
    "${IN_OUT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${SHOW_RUNTIME_STATUS_ARGS[@]}"
    "${SANITIZER_ARGS[@]}"
)

echo "── PaddleAPITest ──────────────────────────"
echo "  引擎:    $ENGINE.py"
echo "  输入:    $FILE_PATTERN"
echo "  日志:    $LOG_DIR"
echo "  GPU:     ids=$GPU_IDS  workers/gpu=$NUM_WORKERS_PER_GPU"
echo "  模式:    ${TEST_MODE_ARGS[*]:-<无>}"
echo "  Sanitizer: enabled=$USE_COMPUTE_SANITIZER exitcode=$SANITIZER_ERROR_EXITCODE command='$SANITIZER_COMMAND'"
echo "────────────────────────────────────────────"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY-RUN] 最终命令:"
    echo "  python $ENGINE.py ${ALL_ARGS[*]}"
    exit 0
fi

mkdir -p "$LOG_DIR" || {
    echo "错误: 无法创建日志目录 '$LOG_DIR'"
    exit 1
}

LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"

if [[ "$FOREGROUND" == "true" ]]; then
    echo -e "\n\033[36m[前台模式] Ctrl+C 终止\033[0m"
    echo "日志同时写入: $LOG_FILE"
    echo ""
    python "$ENGINE.py" "${ALL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
else
    nohup setsid python "$ENGINE.py" "${ALL_ARGS[@]}" >> "$LOG_FILE" 2>&1 &
    PYTHON_PID=$!
    echo "$PYTHON_PID" > "$PID_FILE"

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
    echo "  查看状态:  ./${BASH_SOURCE[0]##*/} --status"
    echo "  终止任务:  ./${BASH_SOURCE[0]##*/} --stop"
    echo "  跟踪日志:  tail -f $LOG_FILE"
    echo "  GPU监控:   watch -n 1 nvidia-smi"
    echo ""
    echo "进程已在后台运行，关闭终端不影响执行"
fi
