#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_multistage
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_multistage/wefend_pos_balanced_uncertainty.yaml --seed 0 --progress > logs/wefend_teacher_match_multistage/wefend_pos_balanced_uncertainty_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_multistage/wefend_pos_balanced_fallback_stage.yaml --seed 0 --progress > logs/wefend_teacher_match_multistage/wefend_pos_balanced_fallback_stage_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_multistage/wefend_pos_balanced_two_stage.yaml --seed 0 --progress > logs/wefend_teacher_match_multistage/wefend_pos_balanced_two_stage_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_multistage/wefend_pos_balanced_reweight_light.yaml --seed 0 --progress > logs/wefend_teacher_match_multistage/wefend_pos_balanced_reweight_light_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_multistage/wefend_pos_balanced_student_temp.yaml --seed 0 --progress > logs/wefend_teacher_match_multistage/wefend_pos_balanced_student_temp_seed0.log 2>&1 &
wait
echo '--- multistage batch slice complete ---'
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_multistage/wefend_pos_balanced_gate_dual.yaml --seed 0 --progress > logs/wefend_teacher_match_multistage/wefend_pos_balanced_gate_dual_seed0.log 2>&1 &
wait
echo 'All multistage jobs finished.'
