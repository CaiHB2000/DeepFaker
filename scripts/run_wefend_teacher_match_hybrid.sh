#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_teacher_match_hybrid
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_hybrid/wefend_hybrid_layered_aux1.yaml --seed 0 --progress > logs/wefend_teacher_match_hybrid/wefend_hybrid_layered_aux1_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_hybrid/wefend_hybrid_layered_aux2.yaml --seed 0 --progress > logs/wefend_teacher_match_hybrid/wefend_hybrid_layered_aux2_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_hybrid/wefend_hybrid_layered_student.yaml --seed 0 --progress > logs/wefend_teacher_match_hybrid/wefend_hybrid_layered_student_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_hybrid/wefend_hybrid_layered_lowcap.yaml --seed 0 --progress > logs/wefend_teacher_match_hybrid/wefend_hybrid_layered_lowcap_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_teacher_match_hybrid/wefend_hybrid_layered_dualtemp.yaml --seed 0 --progress > logs/wefend_teacher_match_hybrid/wefend_hybrid_layered_dualtemp_seed0.log 2>&1 &
wait
echo 'All hybrid jobs finished.'
