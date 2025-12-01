#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_layered
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_layered/wefend_layered_standard.yaml --seed 0 --progress > logs/wefend_teacher_match_layered/wefend_layered_standard_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_layered/wefend_layered_confidence.yaml --seed 0 --progress > logs/wefend_teacher_match_layered/wefend_layered_confidence_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_layered/wefend_layered_avgtemp.yaml --seed 0 --progress > logs/wefend_teacher_match_layered/wefend_layered_avgtemp_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_layered/wefend_layered_posboost.yaml --seed 0 --progress > logs/wefend_teacher_match_layered/wefend_layered_posboost_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_layered/wefend_layered_midrelax.yaml --seed 0 --progress > logs/wefend_teacher_match_layered/wefend_layered_midrelax_seed0.log 2>&1 &
wait
echo '--- layered batch slice complete ---'
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_layered/wefend_layered_studentfocus.yaml --seed 0 --progress > logs/wefend_teacher_match_layered/wefend_layered_studentfocus_seed0.log 2>&1 &
wait
echo 'All layered jobs finished.'
