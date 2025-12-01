#!/usr/bin/env bash
set -euo pipefail
CFG=dynamic_distill/configs/wefend_pseudolabel_consensus/pseudolabel_dynamic_softquota.yaml
OVR="distillation.soft_quota.min_frac=0.05,distillation.soft_quota.max_frac=0.12,distillation.positive_event_gate.teacher_conf=0.85,distillation.force_distill.min_frac=0.09,distillation.force_distill.max_frac=0.18,data.train.csv_file=pseudolabels/wefend_train_with_pseudo.csv,notes.strategy_name=pseudo_thr985_force15"
CUDA_VISIBLE_DEVICES=$1 python dynamic_distill/scripts/train_mvp.py --config $CFG --seed $2 --override_config $OVR
