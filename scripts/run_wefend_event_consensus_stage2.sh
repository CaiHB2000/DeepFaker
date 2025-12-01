#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_event_consensus_stage2
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_balanced_ms.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_balanced_ms_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_balanced_ms.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_balanced_ms_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_dualgate_v2.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_dualgate_v2_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_dualgate_v2.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_dualgate_v2_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_calibrated_fix.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_calibrated_fix_seed00.log 2>&1 &
wait
echo '--- stage2 slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_calibrated_fix.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_calibrated_fix_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_budget_positive.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_budget_positive_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_budget_positive.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_budget_positive_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_lambda_high.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_lambda_high_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_lambda_high.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_lambda_high_seed01.log 2>&1 &
wait
echo '--- stage2 slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_consistency_mix.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_consistency_mix_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_consistency_mix.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_consistency_mix_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_ema_mixed.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_ema_mixed_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_ema_mixed.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_ema_mixed_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_posgate_strong.yaml --seed 0 --progress > logs/wefend_event_consensus_stage2/event_consensus_posgate_strong_seed00.log 2>&1 &
wait
echo '--- stage2 slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus_stage2/event_consensus_posgate_strong.yaml --seed 1 --progress > logs/wefend_event_consensus_stage2/event_consensus_posgate_strong_seed01.log 2>&1 &
wait
echo 'All stage-2 event-consensus jobs finished.'
