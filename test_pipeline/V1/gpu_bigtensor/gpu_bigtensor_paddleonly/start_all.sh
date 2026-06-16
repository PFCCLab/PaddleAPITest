#!/bin/bash

# Assuming this script and run*.sh scripts are in the same directory (PaddleAPITest/)

echo "Starting gpu_bigtensor_paddleonly_run1.sh to gpu_bigtensor_paddleonly_run8.sh in the background..."
echo ""

for i in {1..8}; do
  SCRIPT_NAME="gpu_bigtensor_paddleonly_run${i}.sh"
  if [ -f "$SCRIPT_NAME" ]; then
    bash "./${SCRIPT_NAME}" &
    echo "Started ${SCRIPT_NAME} (PID: $!)"
  else
    echo "Warning: Script ${SCRIPT_NAME} not found, skipping."
  fi
done

echo ""
echo "Attempted to start all found scripts in the background."
echo "Use stop_all.sh to terminate them."