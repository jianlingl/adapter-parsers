#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate parserL

# nohup python main.py train \
#    --device 0 \
#    --train-path "data/universal/en/En.u1.train" \
#    --dev-path "data/universal/en/En.u1.dev" \
#    --test-path "data/universal/en/En.u1.test" \
#    --model-path-base log/models/en_our_span_partitioned_no_adapter5 > log/train_log/en_our_span_partitioned_no_adapter5.trainlog 2>&1 &

nohup python main.py train \
   --device 0 \
   --train-path "data/universal/zh/Zh.u1.train" \
   --dev-path "data/universal/zh/Zh.u1.dev" \
   --test-path "data/universal/zh/Zh.u1.test" \
   --model-path-base log/models/zh_our_span_partitioned_no_adapter5---- > log/train_log/zh_our_span_partitioned_no_adapter5----.trainlog 2>&1 &