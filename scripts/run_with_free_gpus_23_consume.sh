#!/usr/bin/env bash
# Consume a joblist: each launched line is removed from the file.
set -euo pipefail
JOB_LIST=$1
THRESHOLD=${2:-600}
GPU_IDS=(2 3)
mkdir -p logs/rerun_distill
while true; do
  # read next job
  line=$(sed -n '1p' "$JOB_LIST" || true)
  if [[ -z "$line" ]]; then
    echo "[consume] joblist empty, exit" >&2
    exit 0
  fi
  config=$(echo "$line" | awk '{print $1}')
  seed=$(echo "$line" | awk '{print $2}')
  log=$(echo "$line" | awk '{print $3}')
  # wait for free gpu
  while true; do
    for gpu in "${GPU_IDS[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 99999)
      pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=$gpu python dynamic_distill/scripts/train_mvp.py" || true)
      if [[ $used =~ ^[0-9]+$ ]] && (( used < THRESHOLD )) && [[ -z "$pid" ]]; then
        echo "[consume] launch GPU$gpu <- $config seed=$seed" >&2
        mkdir -p "$(dirname "$log")"
        CUDA_VISIBLE_DEVICES=$gpu python dynamic_distill/scripts/train_mvp.py --config "$config" --seed "$seed" >"$log" 2>&1 &
        # remove first line from job list
        tail -n +2 "$JOB_LIST" > "$JOB_LIST.tmp" && mv "$JOB_LIST.tmp" "$JOB_LIST"
        sleep 5
        continue 3
      fi
    done
    sleep 20
  done
done
