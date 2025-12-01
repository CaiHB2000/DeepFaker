#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_event_dynamic_lambda
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_balanced.yaml --seed 0 --progress > logs/wefend_event_dynamic_lambda/dynlambda_balanced_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_balanced.yaml --seed 1 --progress > logs/wefend_event_dynamic_lambda/dynlambda_balanced_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_balanced_stage.yaml --seed 0 --progress > logs/wefend_event_dynamic_lambda/dynlambda_balanced_stage_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_balanced_stage.yaml --seed 1 --progress > logs/wefend_event_dynamic_lambda/dynlambda_balanced_stage_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_dualgate.yaml --seed 0 --progress > logs/wefend_event_dynamic_lambda/dynlambda_dualgate_seed00.log 2>&1 &
wait
echo '--- dynamic lambda slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_dualgate.yaml --seed 1 --progress > logs/wefend_event_dynamic_lambda/dynlambda_dualgate_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_posfocus.yaml --seed 0 --progress > logs/wefend_event_dynamic_lambda/dynlambda_posfocus_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_posfocus.yaml --seed 1 --progress > logs/wefend_event_dynamic_lambda/dynlambda_posfocus_seed01.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_budget.yaml --seed 0 --progress > logs/wefend_event_dynamic_lambda/dynlambda_budget_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_budget.yaml --seed 1 --progress > logs/wefend_event_dynamic_lambda/dynlambda_budget_seed01.log 2>&1 &
wait
echo '--- dynamic lambda slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_ema_mix.yaml --seed 0 --progress > logs/wefend_event_dynamic_lambda/dynlambda_ema_mix_seed00.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_dynamic_lambda/dynlambda_ema_mix.yaml --seed 1 --progress > logs/wefend_event_dynamic_lambda/dynlambda_ema_mix_seed01.log 2>&1 &
wait
echo 'All dynamic-lambda jobs finished.'
