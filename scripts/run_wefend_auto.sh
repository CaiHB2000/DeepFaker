#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_batch
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_00.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_00_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_01.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_01_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_02.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_02_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_03.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_03_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_04.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_04_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_05.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_05_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_06.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_06_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_07.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_07_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_08.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_08_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_09.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_09_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_10.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_10_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/auto_wefend/wefend_auto_20251106-012545_11.yaml --seed 0 --progress > logs/wefend_batch/wefend_auto_20251106-012545_11_seed0.log 2>&1 &
wait
echo 'All jobs finished.'
