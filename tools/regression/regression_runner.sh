#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_FILE="${REGRESSION_CONFIG_FILE:-tools/regression/regression_configs.txt}"
LOG_ROOT="${REGRESSION_LOG_DIR:-/tmp/paddleapitest_project_regression_$(date +%s)}"
PYTHON_BIN="${PYTHON:-python}"

cd "${ROOT_DIR}"

COMMON_ARGS=(
  --api_config_file="${CONFIG_FILE}"
  --use_gpu_mode=True
  --gpu_ids="${GPU_IDS:--1}"
  --num_gpus="${REGRESSION_NUM_GPUS:--1}"
  --num_workers_per_gpu="${REGRESSION_WORKERS_PER_GPU:-4}"
  --timeout="${REGRESSION_TIMEOUT:-180}"
  --show_runtime_status=False
  --random_seed="${REGRESSION_SEED:-20260731}"
)

"${PYTHON_BIN}" engineV4.py \
  --paddle_only=True \
  --log_dir="${LOG_ROOT}/paddle_only" \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" engineV4.py \
  --accuracy=True \
  --log_dir="${LOG_ROOT}/accuracy" \
  "${COMMON_ARGS[@]}"

"${PYTHON_BIN}" tools/error_stat/error_stat.py \
  --input "${LOG_ROOT}/paddle_only" \
  --output "${LOG_ROOT}/paddle_only" \
  --split-errors
"${PYTHON_BIN}" tools/error_stat/error_stat.py \
  --input "${LOG_ROOT}/accuracy" \
  --output "${LOG_ROOT}/accuracy" \
  --split-errors

"${PYTHON_BIN}" tools/regression/check_error_stat.py --log-dir "${LOG_ROOT}/paddle_only"
"${PYTHON_BIN}" tools/regression/check_error_stat.py --log-dir "${LOG_ROOT}/accuracy"

echo "Regression logs: ${LOG_ROOT}"
