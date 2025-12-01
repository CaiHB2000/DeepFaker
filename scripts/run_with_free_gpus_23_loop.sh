#!/usr/bin/env bash
# Loop runner for GPU2/3. Keeps rerunning the given joblist.
set -euo pipefail
JOB_LIST=$1
THRESHOLD=${2:-600}
while true; do
  if [[ ! -s "$JOB_LIST" ]]; then
    echo "[loop] joblist empty, sleep 60" >&2
    sleep 60
    continue
  fi
  echo "[loop] launch batch from $JOB_LIST" >&2
  scripts/run_with_free_gpus_23.sh "$JOB_LIST" "$THRESHOLD"
  echo "[loop] batch completed, sleep 60 then re-check" >&2
  sleep 60
done
