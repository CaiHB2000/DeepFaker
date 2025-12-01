#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_layered_random
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_00.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_00_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_01.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_01_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_02.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_02_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_03.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_03_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_04.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_04_seed0.log 2>&1 &
wait
echo '--- layered random slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_05.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_05_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_06.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_06_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_07.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_07_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_08.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_08_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_09.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_09_seed0.log 2>&1 &
wait
echo '--- layered random slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_10.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_10_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_layered_random/wefend_layered_rand_20251106-191120_11.yaml --seed 0 --progress > logs/wefend_layered_random/wefend_layered_rand_20251106-191120_11_seed0.log 2>&1 &
wait
echo 'All layered random jobs finished.'
