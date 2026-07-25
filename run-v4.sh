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

# ── PaddleAPITest 运行策略 ────────────────────────────────────
# PADDLEAPITEST_IMPL: FP8 blockwise 参考实现，torch（默认）| te。
# PADDLEAPITEST_GPU_MEMORY_POLICY: GPU mode 显存策略，conservative（默认）| aggressive。
# export PADDLEAPITEST_IMPL=torch
# export PADDLEAPITEST_GPU_MEMORY_POLICY=conservative

# ── 输入输出 ──────────────────────────────────────────────────
# input 三选一：--api_config / --api_config_file / --api_config_file_pattern
# NUM_GPUS!=0 时，引擎不受外部 "CUDA_VISIBLE_DEVICES" 影响
# API_CONFIG=""
FILE_INPUT="tester/api_config/5_accuracy/accuracy_1.txt"
# FILE_PATTERN="tester/api_config/5_accuracy/accuracy_*.txt"
LOG_DIR="tester/api_config/test_log"

# ── GPU / worker 调度 ─────────────────────────────────────────
# dual GPU 模式要求 NUM_WORKERS_PER_GPU=1，且选中 GPU 数量至少为 2 且为偶数。
# GPU_IDS 可不连续，引擎按规范化后的顺序两两配对。
NUM_GPUS=-1
NUM_WORKERS_PER_GPU=1
GPU_IDS="-1"
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
    # 双卡稳定性测试；自身等价于 accuracy_stable，并隐式启用 use_gpu_mode
    # --accuracy_stable_dual_gpu=True
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
    # CPU numpy 缓存；gpu_mode 下自动关闭
    # --use_cached_numpy=True
    # GPU tensor 缓存 + GPU compare；增加显存驻留
    # --use_gpu_mode=True
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

if [[ ! -f "$ENGINE.py" || ! -d "tester" ]]; then
    echo "[错误] 请在 PaddleAPITest 项目根目录执行 | 缺少 $ENGINE.py 或 tester/"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
                echo ">>> 终止任务 | 已终止 | PGID $pid"
            else
                echo ">>> 终止任务 | 已结束 | PID $pid"
            fi
            rm -f "$PID_FILE"
        else
            echo ">>> 终止任务 | 无记录"
        fi
        exit 0
        ;;
    --status)
        if [[ -f "$PID_FILE" ]]; then
            pid=$(cat "$PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                # 显示子进程（worker）
                children=$(pgrep -P "$pid" 2>/dev/null | wc -l || true)
                # 显示运行时长
                elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | xargs)
                # 显示日志文件
                log=$(ls -t "$LOG_DIR"/log_*.log 2>/dev/null | head -1 || true)
                echo ">>> 运行状态 | 运行中"
                echo "进程    PID $pid | $ENGINE.py | Worker $children | 已运行 ${elapsed:-未知}"
                [[ -n "${log:-}" ]] && echo "日志    $log"
            else
                echo ">>> 运行状态 | 已结束"
                echo "进程    PID $pid"
                rm -f "$PID_FILE"
            fi
        else
            echo ">>> 运行状态 | 无记录"
            echo "PID文件  $PID_FILE"
        fi
        exit 0
        ;;
    --help|-h)
        echo ">>> 使用帮助 | ${RUN_COMMAND}"
        echo "命令    无参数启动 | --status 查看状态 | --stop 终止 | --help 查看帮助"
        echo "配置    编辑脚本顶部变量，注释或取消注释切换参数"
        exit 0
        ;;
    "") ;; # 正常启动
    *)
        echo "[错误] 未知参数 | 参数 $1 | 帮助 ${RUN_COMMAND} --help"
        exit 1
        ;;
esac

# ── 防重复启动 ──
if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "[警告] 任务已在运行 | PID $old_pid | 终止 ${RUN_COMMAND} --stop"
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
    echo "[错误] 参数不支持 | compute-sanitizer 仅支持 engineV4.py"
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

format_option() {
    local option="$1"
    case "$option" in
        *=True) option="${option%=True}=true" ;;
        *=False) option="${option%=False}=false" ;;
    esac
    printf '%s' "$option"
}

COMPACT_OPTIONS=()
for option in "${TEST_MODE_ARGS[@]}" "${TEST_PARAM_ARGS[@]}"; do
    COMPACT_OPTIONS+=("$(format_option "$option")")
done
COMPACT_OPTIONS+=(
    "--gpu_ids=$GPU_IDS"
    "--num_workers_per_gpu=$NUM_WORKERS_PER_GPU"
    "--timeout=$TIME_OUT"
)
if [[ "$USE_COMPUTE_SANITIZER" == "true" ]]; then
    COMPACT_OPTIONS+=(
        "--use_compute_sanitizer=true"
        "--sanitizer_error_exitcode=$SANITIZER_ERROR_EXITCODE"
        "--sanitizer_command='$SANITIZER_COMMAND'"
    )
fi
printf -v OPTIONS_TEXT ' | %s' "${COMPACT_OPTIONS[@]}"
OPTIONS_TEXT="${OPTIONS_TEXT:3}"

# ── Dry-run 模式 ──
if [[ "$DRY_RUN" == "true" ]]; then
    printf -v COMMAND_TEXT ' %q' python "$ENGINE.py" "${ALL_ARGS[@]}"
    echo ">>> 模拟运行 | $ENGINE.py"
    echo "命令    ${COMMAND_TEXT:1}"
    exit 0
fi

RUN_MODE_LABEL="后台"
[[ "$FOREGROUND" == "true" ]] && RUN_MODE_LABEL="前台"
echo ">>> 启动测试 | $ENGINE.py | $RUN_MODE_LABEL"
echo "输入    $FILE_INPUT"
echo "日志    $LOG_DIR"
echo "参数    $OPTIONS_TEXT"

# ── 创建日志目录 ──
mkdir -p "$LOG_DIR" || {
    echo "[错误] 无法创建日志目录 | 日志 $LOG_DIR"
    exit 1
}

LOG_FILE="$LOG_DIR/log_$(date +%Y%m%d_%H%M%S).log"

# ── 启动 ──
if [[ "$FOREGROUND" == "true" ]]; then
    echo "开始    日志 $LOG_FILE | Ctrl+C 终止"
    # 忽略 shell 自身的 SIGINT，让 Ctrl+C 只作用于 python 子进程
    trap '' INT
    python "$ENGINE.py" "${ALL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    trap - INT
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
        echo "[错误] 启动失败 | $ENGINE.py | 日志 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi

    echo "已启动  PID $PYTHON_PID | 日志 $LOG_FILE"
    echo "管理    状态: ${RUN_COMMAND} --status | 终止: ${RUN_COMMAND} --stop"
    echo "跟踪    tail -f $LOG_FILE"
fi
