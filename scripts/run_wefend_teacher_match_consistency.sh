#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_consistency
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_consistency/wefend_consistency_low.yaml --seed 0 --progress > logs/wefend_teacher_match_consistency/wefend_consistency_low_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_consistency/wefend_consistency_dual.yaml --seed 0 --progress > logs/wefend_teacher_match_consistency/wefend_consistency_dual_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_consistency/wefend_consistency_studentfocus.yaml --seed 0 --progress > logs/wefend_teacher_match_consistency/wefend_consistency_studentfocus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_consistency/wefend_consistency_midrelax.yaml --seed 0 --progress > logs/wefend_teacher_match_consistency/wefend_consistency_midrelax_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_consistency/wefend_consistency_fallback.yaml --seed 0 --progress > logs/wefend_teacher_match_consistency/wefend_consistency_fallback_seed0.log 2>&1 &
wait
echo 'All consistency jobs finished.'
