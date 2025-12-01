#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_enhanced
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_eventreweight.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_eventreweight_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_delta_tight.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_delta_tight_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_temp_mix.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_temp_mix_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_gate_relaxed.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_gate_relaxed_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_student_focus.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_student_focus_seed0.log 2>&1 &
wait
echo '--- enhanced batch slice complete ---'
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_soft_fallback.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_soft_fallback_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_enhanced/wefend_pos_balanced_eventreweight_temp.yaml --seed 0 --progress > logs/wefend_teacher_match_enhanced/wefend_pos_balanced_eventreweight_temp_seed0.log 2>&1 &
wait
echo 'All enhanced jobs finished.'
