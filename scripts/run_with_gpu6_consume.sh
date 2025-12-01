#!/usr/bin/env bash
# consume joblist on GPU6 sequentially
set -euo pipefail
JOB_LIST=$1
THRESHOLD=${2:-600}
while true; do
  line=$(sed -n '1p' "$JOB_LIST" || true)
  if [[ -z "$line" ]]; then
    echo "[gpu6] joblist empty, exit" >&2
    exit 0
  fi
  config=$(echo "$line" | awk '{print $1}')
  seed=$(echo "$line" | awk '{print $2}')
  log=$(echo "$line" | awk '{print $3}')
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 2>/dev/null || echo 99999)
    pid=$(pgrep -f "CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py" || true)
    if [[ $used =~ ^[0-9]+$ ]] && (( used < THRESHOLD )) && [[ -z "$pid" ]]; then
      echo "[gpu6] launch <- $config seed=$seed" >&2
      mkdir -p "$(dirname "$log")"
      CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config "$config" --seed "$seed" >"$log" 2>&1 &
      tail -n +2 "$JOB_LIST" > "$JOB_LIST.tmp" && mv "$JOB_LIST.tmp" "$JOB_LIST"
      sleep 5
      break
    fi
    sleep 20
  done
done
