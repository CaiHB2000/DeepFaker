#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_weakteacher
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher/wefend_weak_teacher_standard.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher/wefend_weak_teacher_standard_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher/wefend_weak_teacher_dual.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher/wefend_weak_teacher_dual_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher/wefend_weak_teacher_focus.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher/wefend_weak_teacher_focus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher/wefend_weak_teacher_fallback.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher/wefend_weak_teacher_fallback_seed0.log 2>&1 &
wait
echo 'All weak-teacher jobs finished.'
