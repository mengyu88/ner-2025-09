#!/usr/bin/env bash
# Independently wait for the overnight controller and produce the final ranking.
set -uo pipefail

RUN_ROOT="/root/shared-nvme/projects/SPSR-Net/runs/genia_bhpc_replacement_20260818_overnight"
WAIT_PID_FILE="$RUN_ROOT/followup_waiter.pid"

while true; do
  if grep -q 'Follow-up ended exit_code=0' "$RUN_ROOT/followup_status.log" 2>/dev/null; then
    cd /root/shared-nvme/projects/SPSR-Net
    /root/.venvs/spsr-net/bin/python scripts/summarize_genia_bhpc_overnight.py \
      > "$RUN_ROOT/final_ranking.log" 2>&1
    exit $?
  fi
  if [[ -f "$WAIT_PID_FILE" ]]; then
    waiter_pid=$(tr -d '[:space:]' < "$WAIT_PID_FILE")
    if ! kill -0 "$waiter_pid" 2>/dev/null; then
      echo "Follow-up waiter exited without a successful completion." > "$RUN_ROOT/final_ranking.log"
      exit 1
    fi
  fi
  sleep 60
done
