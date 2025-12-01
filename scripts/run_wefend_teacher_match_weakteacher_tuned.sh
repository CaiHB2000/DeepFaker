#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_weakteacher_tuned
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_high.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_high_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_high.yaml --seed 1 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_high_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_mid.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_mid_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_mid.yaml --seed 1 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_mid_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_conservative.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_conservative_seed0.log 2>&1 &
wait
echo '--- weakteacher tuned slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_conservative.yaml --seed 1 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_conservative_seed1.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_softquota.yaml --seed 0 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_softquota_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_softquota.yaml --seed 1 --progress > logs/wefend_teacher_match_weakteacher_tuned/wefend_weakteacher_tuned_softquota_seed1.log 2>&1 &
wait
echo 'All weakteacher tuned jobs finished.'
