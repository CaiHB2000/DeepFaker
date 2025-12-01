#!/usr/bin/env bash
set -e
mkdir -p logs/wefend_weakteacher_random
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_00.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_00_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_01.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_01_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_02.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_02_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_03.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_03_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_04.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_04_seed0.log 2>&1 &
wait
echo '--- weakteacher random slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_05.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_05_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_06.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_06_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_07.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_07_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_08.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_08_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_09.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_09_seed0.log 2>&1 &
wait
echo '--- weakteacher random slice complete ---'
CUDA_VISIBLE_DEVICES=2 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_10.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_10_seed0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python dynamic_distill/scripts/train_mvp.py --config dynamic_distill/configs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_11.yaml --seed 0 --progress > logs/wefend_weakteacher_random/wefend_weakteacher_rand_20251107-001608_11_seed0.log 2>&1 &
wait
echo 'All weakteacher random jobs finished.'
