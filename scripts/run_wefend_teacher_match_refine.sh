#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_refine
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_delta_compact.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_delta_compact_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_delta_soft.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_delta_soft_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_mild.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_pos_mild_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_pos_balanced_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_event_reweight.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_event_reweight_seed0.log 2>&1 &
wait
echo '--- refine batch slice complete ---'
CUDA_VISIBLE_DEVICES=0 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_temp_lowkl.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_temp_lowkl_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_feat_mix.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_feat_mix_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_dual_mix.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_dual_mix_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_refine/wefend_longtrain_student_focus.yaml --seed 0 --progress > logs/wefend_teacher_match_refine/wefend_longtrain_student_focus_seed0.log 2>&1 &
wait
echo 'All refine jobs finished.'
