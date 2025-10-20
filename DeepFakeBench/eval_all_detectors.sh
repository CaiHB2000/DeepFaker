#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ========= paths =========
# 脚本放在 DeepFakeBench/ 下，这里就指向 DeepFakeBench
ROOT="$(cd "$(dirname "$0")" && pwd)"
DET_DIR="$ROOT/training/config/detector"
W_DIR="$ROOT/training/pretrained"

# 评测数据集（可多个）
DATASETS=("Celeb-DF-v2")

# 日志
RUN_TAG="$(date +%F_%H-%M-%S)"
LOG_DIR="$ROOT/logs/testing_bench/$RUN_TAG"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/eval_all_$RUN_TAG.log"

echo "===> Start evaluating datasets: ${DATASETS[*]}" | tee -a "$MASTER_LOG"
echo "[INFO] ROOT=$ROOT" | tee -a "$MASTER_LOG"
echo "[INFO] DET_DIR=$DET_DIR" | tee -a "$MASTER_LOG"
echo "[INFO] W_DIR=$W_DIR" | tee -a "$MASTER_LOG"

# ========= map: detector -> weight filename =========
declare -A MAP=(
  ["capsule_net"]="capsule_best.pth"
  ["core"]="core_best.pth"
  ["efficientnetb4"]="effnb4_best.pth"
  ["f3net"]="f3net_best.pth"
  ["ffd"]="ffd_best.pth"
  ["meso4"]="meso4_best.pth"
  ["meso4Inception"]="meso4Incep_best.pth"
  ["recce"]="recce_best.pth"
  ["spsl"]="spsl_best.pth"
  ["srm"]="srm_best.pth"
  ["ucf"]="ucf_best.pth"
  ["xception"]="xception_best.pth"
)

# ========= sanity checks =========
if [[ ! -d "$DET_DIR" ]]; then
  echo "[ERROR] Detector config dir not found: $DET_DIR" | tee -a "$MASTER_LOG"
  exit 1
fi
if [[ ! -d "$W_DIR" ]]; then
  echo "[ERROR] Pretrained dir not found: $W_DIR" | tee -a "$MASTER_LOG"
  exit 1
fi

shopt -s nullglob
YAMLS=("$DET_DIR"/*.yaml)
echo "[INFO] Found ${#YAMLS[@]} detector yamls" | tee -a "$MASTER_LOG"
if (( ${#YAMLS[@]} == 0 )); then
  echo "[WARN] No yaml found under $DET_DIR" | tee -a "$MASTER_LOG"
fi

SUMMARY_TSV="$LOG_DIR/summary.tsv"
echo -e "detector\tdataset\tAUC\tEER\tACC\tAP" > "$SUMMARY_TSV"

# ========= loop =========
for YAML in "${YAMLS[@]}"; do
  DET="$(basename "$YAML" .yaml)"
  WEIGHT="${MAP[$DET]:-}"

  if [[ -z "${WEIGHT:-}" ]]; then
    echo "[SKIP] $DET: no mapped weight" | tee -a "$MASTER_LOG"
    continue
  fi
  if [[ ! -f "$W_DIR/$WEIGHT" ]]; then
    echo "[SKIP] $DET: weight not found ($W_DIR/$WEIGHT)" | tee -a "$MASTER_LOG"
    continue
  fi

  echo -e "\n==== Evaluating $DET ====" | tee -a "$MASTER_LOG"
  DET_LOG="$LOG_DIR/${DET}.log"

  python3 "$ROOT/training/test.py" \
    --detector_path "$YAML" \
    --test_dataset "${DATASETS[@]}" \
    --weights_path "$W_DIR/$WEIGHT" \
    2>&1 | tee "$DET_LOG" | tee -a "$MASTER_LOG"

  # 抓指标（按你 test.py 的打印格式）
  awk -v det="$DET" '
    /dataset: /{ds=$2}
    /^[[:space:]]*AUC:/ {auc=$2}
    /^[[:space:]]*eer:/ {eer=$2}
    /^[[:space:]]*acc:/ {acc=$2}
    /^[[:space:]]*ap:/  {ap=$2; printf("%s\t%s\t%s\t%s\t%s\t%s\n", det, ds, auc, eer, acc, ap) }
  ' "$DET_LOG" >> "$SUMMARY_TSV"
done

echo -e "\n===> All done."
echo "Logs: $LOG_DIR"
echo "Summary: $SUMMARY_TSV"
