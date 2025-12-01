#!/usr/bin/env bash
# Usage: scripts/run_with_free_gpus.sh job_list.txt [memory_threshold_mb]
# Each line in job_list: <config_path> <seed> <log_path>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 JOB_LIST [THRESHOLD_MB]" >&2
  exit 1
fi

JOB_LIST=$1
THRESHOLD=${2:-600}
GPU_IDS=(4 5)
declare -A GPU_PIDS=()

if [[ ! -f "$JOB_LIST" ]]; then
  echo "Job list $JOB_LIST not found" >&2
  exit 1
fi

launch_job() {
  local gpu=$1
  local config=$2
  local seed=$3
  local log=$4
  local extra=${5:-}
  mkdir -p "$(dirname "$log")"
  echo "[launch] GPU$gpu <- $config seed=$seed" >&2
  CUDA_VISIBLE_DEVICES=$gpu python dynamic_distill/scripts/train_mvp.py --config "$config" --seed "$seed" ${extra:+$extra} >"$log" 2>&1 &
  local pid=$!
  GPU_PIDS[$gpu]=$pid
  pids+=($pid)
}

gpu_available() {
  local gpu=$1
  local used pid
  pid=${GPU_PIDS[$gpu]:-}
  if [[ -n "${pid:-}" ]]; then
    if kill -0 "$pid" 2>/dev/null; then
      return 1
    else
      unset GPU_PIDS[$gpu]
    fi
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 99999)
  if [[ $used =~ ^[0-9]+$ ]] && (( used < THRESHOLD )); then
    return 0
  fi
  return 1
}

declare -a pids=()

while read -r config seed log rest; do
  [[ -z "${config:-}" ]] && continue
  while true; do
    for gpu in "${GPU_IDS[@]}"; do
      if gpu_available "$gpu"; then
        launch_job "$gpu" "$config" "$seed" "$log" "$rest"
        sleep 5
        continue 3
      fi
    done
    sleep 20
  done
done <"$JOB_LIST"

for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "All jobs finished."
