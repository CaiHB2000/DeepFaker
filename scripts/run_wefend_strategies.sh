#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_strategies
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_pos_boost_strict.yaml --seed 0 --progress > logs/wefend_strategies/wefend_pos_boost_strict_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_pos_boost_soft.yaml --seed 0 --progress > logs/wefend_strategies/wefend_pos_boost_soft_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_cost_sensitive.yaml --seed 0 --progress > logs/wefend_strategies/wefend_cost_sensitive_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_fusion_emphasis.yaml --seed 0 --progress > logs/wefend_strategies/wefend_fusion_emphasis_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_teacher_match.yaml --seed 0 --progress > logs/wefend_strategies/wefend_teacher_match_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_uncertainty_focus.yaml --seed 0 --progress > logs/wefend_strategies/wefend_uncertainty_focus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_event_relaxed.yaml --seed 0 --progress > logs/wefend_strategies/wefend_event_relaxed_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_event_strict.yaml --seed 0 --progress > logs/wefend_strategies/wefend_event_strict_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_early_stage.yaml --seed 0 --progress > logs/wefend_strategies/wefend_early_stage_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_late_stage.yaml --seed 0 --progress > logs/wefend_strategies/wefend_late_stage_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_delta_schedule.yaml --seed 0 --progress > logs/wefend_strategies/wefend_delta_schedule_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_fallback_light.yaml --seed 0 --progress > logs/wefend_strategies/wefend_fallback_light_seed0.log 2>&1 &
wait
echo 'All strategy jobs finished.'
