#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_batch
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_delta.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_delta_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_posfocus.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_posfocus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_cost.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_cost_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_uncertainty.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_uncertainty_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_curriculum.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_curriculum_seed0.log 2>&1 &
wait
echo '--- batch complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_studentfocus.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_studentfocus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_eventreweight.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_eventreweight_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_longtrain.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_longtrain_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_late.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_late_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_tempplus.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_tempplus_seed0.log 2>&1 &
wait
echo '--- batch complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_tempminus.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_tempminus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_dualstage.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_dualstage_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_batch/wefend_teacher_match_nomistake.yaml --seed 0 --progress > logs/wefend_teacher_match_batch/wefend_teacher_match_nomistake_seed0.log 2>&1 &
wait
echo 'All teacher-match batch jobs finished.'
