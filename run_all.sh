#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# Default paths (can be overridden by env/args)
# -------------------------------
IN_CSV_DEFAULT="./DeepFakeBench/results/celebdfv2_xception_scores.csv"
LMDB_ROOT_DEFAULT="./DeepFakeBench/datasets/lmdb/Celeb-DF-v2_lmdb"
OUT_DIR_DEFAULT="./results/celebdfv2"

# 允许通过环境变量覆盖
IN_CSV="${IN_CSV:-$IN_CSV_DEFAULT}"
LMDB_ROOT="${LMDB_ROOT:-$LMDB_ROOT_DEFAULT}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

# 允许通过命令行覆盖：run_all.sh <in_csv> <lmdb_root> <out_dir>
if [[ $# -ge 1 ]]; then IN_CSV="$1"; fi
if [[ $# -ge 2 ]]; then LMDB_ROOT="$2"; fi
if [[ $# -ge 3 ]]; then OUT_DIR="$3"; fi

# 可调参数（也可用环境变量覆盖）
N_NODES="${N_NODES:-300}"
N_STEPS="${N_STEPS:-6}"
ALPHA="${ALPHA:-0.33}"
BETA="${BETA:-0.33}"
GAMMA="${GAMMA:-0.34}"
SIGK="${SIGK:-5.0}"
JPEG_Q="${JPEG_Q:-70}"

POL_HI="${POL_HI:-0.8}"
POL_MID="${POL_MID:-0.6}"
POL_HI_F="${POL_HI_F:-0.3}"
POL_MID_F="${POL_MID_F:-0.6}"

PROBE="${PROBE:-phash_stability_enhanced}"

# 新增：聚合重标策略
RESCALE="${RESCALE:-quantile}"            # none | linear | quantile

# linear 模式可选
EXPAND_LO="${EXPAND_LO:-0.7}"
EXPAND_HI="${EXPAND_HI:-0.9}"

# quantile 模式可选
QLOW="${QLOW:-0.20}"
QHIGH="${QHIGH:-0.80}"
TGT_LOW_LO="${TGT_LOW_LO:-0.05}"
TGT_LOW_HI="${TGT_LOW_HI:-0.35}"
TGT_MID_LO="${TGT_MID_LO:-0.35}"
TGT_MID_HI="${TGT_MID_HI:-0.75}"
TGT_HIGH_LO="${TGT_HIGH_LO:-0.75}"
TGT_HIGH_HI="${TGT_HIGH_HI:-0.98}"

# 分析阶段开关（1=执行，0=跳过）
DO_VALIDATE="${DO_VALIDATE:-1}"
DO_VIZ="${DO_VIZ:-1}"

# 路径准备
mkdir -p "${OUT_DIR}"
FIG_DIR="${OUT_DIR}/figs"
mkdir -p "${FIG_DIR}"

CONTENT_CSV="${OUT_DIR}/content_capsule.csv"
PROV_CSV="${OUT_DIR}/cred_phash.csv"
PROP_CSV="${OUT_DIR}/early_ba.csv"
# 注意：文件名里包含超参数便于区分
P2P_CSV="${OUT_DIR}/p2p_a${ALPHA}_b${BETA}_g${GAMMA}.csv"
POLICY_CSV="${OUT_DIR}/policy_simple.csv"
VALIDATE_JSON="${OUT_DIR}/validate_report.json"

echo "================ P2P Pipeline ================"
echo "[Config]"
echo "IN_CSV   = ${IN_CSV}"
echo "LMDB_ROOT= ${LMDB_ROOT}"
echo "OUT_DIR  = ${OUT_DIR}"
echo "N_NODES  = ${N_NODES}, N_STEPS = ${N_STEPS}"
echo "ALPHA/BETA/GAMMA = ${ALPHA}/${BETA}/${GAMMA}, SIGK=${SIGK}"
echo "JPEG_Q   = ${JPEG_Q}"
echo "Policy thresholds: hi=${POL_HI}, mid=${POL_MID}, hi_f=${POL_HI_F}, mid_f=${POL_MID_F}"
echo "DO_VALIDATE=${DO_VALIDATE}, DO_VIZ=${DO_VIZ}"
echo "================================================"

# 简单依赖提示（不强制安装）
missing=0
for pkg in lmdb imagehash Pillow opencv-python networkx pyyaml; do
  python -c "import ${pkg//-/_}" 2>/dev/null || { echo "[Warn] Python package missing: $pkg"; missing=1; }
done
if [[ $missing -eq 1 ]]; then
  echo "[Tip] Install missing deps: pip install lmdb imagehash pillow opencv-python networkx pyyaml"
fi

# 保证可以作为包运行
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 1) content：把 DeepFakeBench 输出转为统一 schema
echo "[1/7] Content → ${CONTENT_CSV}"
python -m p2p.runners.run_content_from_csv \
  --in_csv  "${IN_CSV}" \
  --out_csv "${CONTENT_CSV}"

# 2) provenance：基于 LMDB 的 pHash 稳定性
echo "[2/7] Provenance → ${PROV_CSV}"
python -m p2p.runners.run_provenance \
  --items_csv "${CONTENT_CSV}" \
  --lmdb_root "${LMDB_ROOT}" \
  --out_csv   "${PROV_CSV}" \
  --jpeg_q    "${JPEG_Q}" \
  --probe     "${PROBE}"

# 3) propagation：BA 模拟早期扩散风险
echo "[3/7] Propagation → ${PROP_CSV}"
python -m p2p.runners.run_propagation \
  --content_csv "${CONTENT_CSV}" \
  --prov_csv    "${PROV_CSV}" \
  --out_csv     "${PROP_CSV}" \
  --n           "${N_NODES}" \
  --steps       "${N_STEPS}"

# 4) aggregate：统一 P2P 风险分
echo "[4/7] Aggregate → ${P2P_CSV}"
python -m p2p.runners.run_aggregate \
  --content_csv "${CONTENT_CSV}" \
  --prov_csv    "${PROV_CSV}" \
  --prop_csv    "${PROP_CSV}" \
  --out_csv     "${P2P_CSV}" \
  --alpha       "${ALPHA}" \
  --beta        "${BETA}" \
  --gamma       "${GAMMA}" \
  --k           "${SIGK}" \
  --rescale     "${RESCALE}" \
  --expand_lo   "${EXPAND_LO}" \
  --expand_hi   "${EXPAND_HI}" \
  --q_low       "${QLOW}" \
  --q_high      "${QHIGH}" \
  --tgt_low_lo  "${TGT_LOW_LO}" \
  --tgt_low_hi  "${TGT_LOW_HI}" \
  --tgt_mid_lo  "${TGT_MID_LO}" \
  --tgt_mid_hi  "${TGT_MID_HI}" \
  --tgt_high_lo "${TGT_HIGH_LO}" \
  --tgt_high_hi "${TGT_HIGH_HI}"

# 5) policy（可选：默认执行）
echo "[5/7] Policy → ${POLICY_CSV}"
python -m p2p.runners.run_policy \
  --p2p_csv "${P2P_CSV}" \
  --out_csv "${POLICY_CSV}" \
  --hi      "${POL_HI}" \
  --mid     "${POL_MID}" \
  --hi_f    "${POL_HI_F}" \
  --mid_f   "${POL_MID_F}"

# 6) validate（可开关）
if [[ "${DO_VALIDATE}" == "1" ]]; then
  echo "[6/7] Validate → ${VALIDATE_JSON}"
  python p2p/analysis/validate.py \
    --content_csv "${CONTENT_CSV}" \
    --prov_csv    "${PROV_CSV}" \
    --prop_csv    "${PROP_CSV}" \
    --p2p_csv     "${P2P_CSV}" \
    --report_json "${VALIDATE_JSON}"
else
  echo "[6/7] Validate skipped (DO_VALIDATE=0)"
fi

# 7) visualize（可开关）
if [[ "${DO_VIZ}" == "1" ]]; then
  echo "[7/7] Visualize → ${FIG_DIR}"
  python p2p/analysis/viz.py \
    --content_csv "${CONTENT_CSV}" \
    --prov_csv    "${PROV_CSV}" \
    --prop_csv    "${PROP_CSV}" \
    --p2p_csv     "${P2P_CSV}" \
    --policy_csv  "${POLICY_CSV}" \
    --out_dir     "${FIG_DIR}"
else
  echo "[7/7] Visualize skipped (DO_VIZ=0)"
fi

echo "================ DONE ================"
echo "Content CSV     : ${CONTENT_CSV}"
echo "Provenance CSV  : ${PROV_CSV}"
echo "Propagation CSV : ${PROP_CSV}"
echo "P2P CSV         : ${P2P_CSV}"
echo "Policy CSV      : ${POLICY_CSV}"
echo "Validate Report : ${VALIDATE_JSON} (if enabled)"
echo "Figures         : ${FIG_DIR} (if enabled)"
