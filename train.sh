#!/bin/bash
CUDA_VISIBLE_DEVICES=5 nohup /data/ljl/envs/bloom/bin/python main.py train \
   --device 0 \
   --seed 20248888 \
   --train-path "data/UD2UC/en.train" \
   --dev-path "data/UD2UC/en.dev" \
   --test-path "data/UD2UC/en.test" \
   --model-path-base log/saved_parsers/en.pt > log/train/en.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 nohup /data/ljl/envs/bloom/bin/python main.py train \
#    --device 0 \
#    --seed 20248888 \
#    --train-path "data/UD2UC/de.train" \
#    --dev-path "data/UD2UC/de.dev" \
#    --test-path "data/UD2UC/de.test" \
#    --model-path-base log/saved_parsers/de.pt > log/train/de.log 2>&1 &

# CUDA_VISIBLE_DEVICES=5 nohup /data/ljl/envs/bloom/bin/python main.py train \
#    --device 1 \
#    --seed 20248888 \
#    --train-path "data/UD2UC/de.train" \
#    --dev-path "data/UD2UC/de.dev" \
#    --test-path "data/UD2UC/de.test" \
#    --model-path-base log/saved_parsers/de.pt > log/train/de.log 2>&1 &
