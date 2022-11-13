#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate parserL

nohup python main.py train \
   --device 1 \
   --train-path "data/universal/en/En.u1.train" \
   --dev-path "data/universal/en/En.u1.dev" \
   --test-path "data/universal/en/En.u1.test" \
   --model-path-base log/models/en_adapter_1e-2 > log/train_log/en_adapter_1e-2.trainlog 2>&1 &

nohup python main.py train \
   --device 2 \
   --train-path "data/universal/zh/Zh.u1.train" \
   --dev-path "data/universal/zh/Zh.u1.dev" \
   --test-path "data/universal/zh/Zh.u1.test" \
   --model-path-base log/models/zh_adapter_1e-2 > log/train_log/zh_adapter_1e-2.trainlog 2>&1 &