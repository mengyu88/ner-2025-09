#!/usr/bin/env bash
# Wait for the initial three-stage study and then continue the scheduled search.
set -uo pipefail

RUN_ROOT="/root/shared-nvme/projects/SPSR-Net/runs/genia_bhpc_replacement_20260818_overnight"
INITIAL_PID_FILE="$RUN_ROOT/controller.pid"

if [[ ! -f "$INITIAL_PID_FILE" ]]; then
  echo "Initial controller PID is missing: $INITIAL_PID_FILE" >&2
  exit 1
fi

initial_pid=$(tr -d '[:space:]' < "$INITIAL_PID_FILE")
while kill -0 "$initial_pid" 2>/dev/null; do
  sleep 60
done

if ! grep -q 'GENIA BHPC replacement sweep completed' "$RUN_ROOT/controller.log" 2>/dev/null; then
  echo "Initial sweep did not complete successfully; follow-up was not started." | tee "$RUN_ROOT/followup_status.log"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] Starting scheduled follow-up stages" | tee "$RUN_ROOT/followup_status.log"
cd /root/shared-nvme/projects/SPSR-Net
bash scripts/run_genia_bhpc_followup.sh >> "$RUN_ROOT/followup_launcher.log" 2>&1
code=$?
if [[ "$code" -eq 0 ]]; then
  /root/.venvs/spsr-net/bin/python scripts/summarize_genia_bhpc_overnight.py \
    > "$RUN_ROOT/final_ranking.log" 2>&1
fi
echo "[$(date -u +%FT%TZ)] Follow-up ended exit_code=$code" | tee -a "$RUN_ROOT/followup_status.log"
exit "$code"
