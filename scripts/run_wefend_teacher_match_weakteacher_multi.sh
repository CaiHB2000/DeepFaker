#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_weakteacher_multi
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_multi/wefend_weak_teacher_standard.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher_multi/wefend_weak_teacher_standard_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_multi/wefend_weak_teacher_standard.yaml --seed 1 --progress > logs/wefend_teacher_match_weakteacher_multi/wefend_weak_teacher_standard_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_multi/wefend_weak_teacher_standard.yaml --seed 2 --progress > logs/wefend_teacher_match_weakteacher_multi/wefend_weak_teacher_standard_seed2.log 2>&1 &
wait
echo 'All weak-teacher multi jobs finished.'
