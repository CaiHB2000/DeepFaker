#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_event_consensus
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_highmix.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_highmix_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_lowtemp.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_lowtemp_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_posfocus.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_posfocus_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_dualgate.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_dualgate_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_calibrated.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_calibrated_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_softquota.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_softquota_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_negassist.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_negassist_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_costschedule.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_costschedule_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_curriculum.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_curriculum_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_dualview.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_dualview_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_emahard.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_emahard_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_event_consensus/event_consensus_balanced.yaml --seed 0 --progress > logs/wefend_event_consensus/event_consensus_balanced_seed0.log 2>&1 &
wait
echo 'All event-consensus jobs finished.'
