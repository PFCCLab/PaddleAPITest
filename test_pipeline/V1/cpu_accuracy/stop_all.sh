#!/bin/bash

# Assuming this script and run*.sh scripts are in the same directory (e.g., PaddleAPITest/)

PYTHON_CMD_PATTERN="python engine.py --api_config_file=tester/api_config/api_config_merged.*\.txt --accuracy=True"

SCRIPT_CMD_PATTERN="bash \./cpu_accuracy_run(1[0-9]|[1-9]|20)\.sh" 

# --- Script Logic ---

echo "Attempting to stop all related run script processes (cpu_accuracy_run1.sh to cpu_accuracy_run20.sh)..."
echo ""

# --- Step 1: Send SIGTERM (Graceful Shutdown) ---
echo "[Phase 1] Sending SIGTERM (graceful shutdown signal)..."

echo "Stopping python engine.py processes matching pattern (SIGTERM):"
echo "Pattern: $PYTHON_CMD_PATTERN"
if pgrep -f "$PYTHON_CMD_PATTERN" > /dev/null; then
    pkill -f "$PYTHON_CMD_PATTERN"
    PYTHON_TERM_STATUS=$?
    echo "SIGTERM sent to Python processes."
else
    PYTHON_TERM_STATUS=1 
    echo "No matching Python processes found to send SIGTERM."
fi

echo "Stopping run scripts matching pattern (SIGTERM):"
echo "Pattern: $SCRIPT_CMD_PATTERN"
if pgrep -f "$SCRIPT_CMD_PATTERN" > /dev/null; then
    pkill -f "$SCRIPT_CMD_PATTERN"
    SCRIPT_TERM_STATUS=$?
    echo "SIGTERM sent to run scripts."
else
    SCRIPT_TERM_STATUS=1 
    echo "No matching run scripts found to send SIGTERM."
fi

if [ $PYTHON_TERM_STATUS -eq 0 ] || [ $SCRIPT_TERM_STATUS -eq 0 ]; then
    echo "Waiting 5 seconds for processes to exit gracefully..."
    sleep 5
fi

# --- Step 2: Check for Remaining Processes ---
echo "[Phase 2] Checking for remaining processes after waiting..."
PYTHON_PIDS=$(pgrep -f "$PYTHON_CMD_PATTERN")
SCRIPT_PIDS=$(pgrep -f "$SCRIPT_CMD_PATTERN")

# --- Step 3: Send SIGKILL (Force Kill) if Necessary ---
if [ -n "$PYTHON_PIDS" ] || [ -n "$SCRIPT_PIDS" ]; then
  echo "[Phase 3] Found remaining processes. Sending SIGKILL (force kill signal)..."

  if [ -n "$PYTHON_PIDS" ]; then
    echo "Force killing remaining python processes (SIGKILL)..."
    pkill -9 -f "$PYTHON_CMD_PATTERN" =
    if [ $? -eq 0 ]; then
        echo "SIGKILL sent to Python processes."
    else
        if ! pgrep -f "$PYTHON_CMD_PATTERN" > /dev/null; then
             echo "Python processes seem to have exited before SIGKILL was needed or completed."
        else
             echo "Warning: Failed to send SIGKILL to Python processes or they are unkillable."
        fi
    fi
  fi

  if [ -n "$SCRIPT_PIDS" ]; then
    echo "Force killing remaining run scripts (SIGKILL)..."
    pkill -9 -f "$SCRIPT_CMD_PATTERN" 
     if [ $? -eq 0 ]; then
        echo "SIGKILL sent to run scripts."
     else
        if ! pgrep -f "$SCRIPT_CMD_PATTERN" > /dev/null; then
            echo "Run scripts seem to have exited before SIGKILL was needed or completed."
        else
            echo "Warning: Failed to send SIGKILL to run scripts or they are unkillable."
        fi
     fi
  fi
  sleep 1 
else
  echo "[Phase 3] No remaining processes found after graceful shutdown attempt."
fi

# --- Step 4: Final Verification ---
echo "[Phase 4] Final verification..."
FINAL_PYTHON_PIDS=$(pgrep -f "$PYTHON_CMD_PATTERN")
FINAL_SCRIPT_PIDS=$(pgrep -f "$SCRIPT_CMD_PATTERN")

if [ -n "$FINAL_PYTHON_PIDS" ] || [ -n "$FINAL_SCRIPT_PIDS" ]; then
  echo "Error: Some processes might still be running even after SIGKILL! Manual check required:"
  if [ -n "$FINAL_PYTHON_PIDS" ]; then
      echo "--- Remaining Python Processes (Post-SIGKILL Check) ---"
      pgrep -af "$PYTHON_CMD_PATTERN"
      echo "----------------------------------------------------"
  fi
   if [ -n "$FINAL_SCRIPT_PIDS" ]; then
      echo "--- Remaining Script Processes (Post-SIGKILL Check) ---"
      pgrep -af "$SCRIPT_CMD_PATTERN"
      echo "---------------------------------------------------"
  fi
else
  echo "Successfully stopped all targeted processes (verified after potential SIGKILL)."
fi

echo "Stop process completed."