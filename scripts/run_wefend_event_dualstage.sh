#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_event_dualstage
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_balanced_stage1.yaml --seed 0 --progress > logs/wefend_event_dualstage/dualstage_balanced_stage1_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_balanced_stage1.yaml --seed 1 --progress > logs/wefend_event_dualstage/dualstage_balanced_stage1_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_dualgate_stage1.yaml --seed 0 --progress > logs/wefend_event_dualstage/dualstage_dualgate_stage1_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_dualgate_stage1.yaml --seed 1 --progress > logs/wefend_event_dualstage/dualstage_dualgate_stage1_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_posfocus_stage1.yaml --seed 0 --progress > logs/wefend_event_dualstage/dualstage_posfocus_stage1_seed00.log 2>&1 &
wait
echo '--- dualstage stage1 slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_posfocus_stage1.yaml --seed 1 --progress > logs/wefend_event_dualstage/dualstage_posfocus_stage1_seed01.log 2>&1 &
wait
echo '--- dualstage stage1 complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_balanced_stage2_seed00.yaml --seed 0 --progress > logs/wefend_event_dualstage/dualstage_balanced_stage2_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_balanced_stage2_seed01.yaml --seed 1 --progress > logs/wefend_event_dualstage/dualstage_balanced_stage2_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_dualgate_stage2_seed00.yaml --seed 0 --progress > logs/wefend_event_dualstage/dualstage_dualgate_stage2_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_dualgate_stage2_seed01.yaml --seed 1 --progress > logs/wefend_event_dualstage/dualstage_dualgate_stage2_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_posfocus_stage2_seed00.yaml --seed 0 --progress > logs/wefend_event_dualstage/dualstage_posfocus_stage2_seed00.log 2>&1 &
wait
echo '--- dualstage stage2 slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dualstage/dualstage_posfocus_stage2_seed01.yaml --seed 1 --progress > logs/wefend_event_dualstage/dualstage_posfocus_stage2_seed01.log 2>&1 &
wait
echo 'All dual-stage jobs finished.'
