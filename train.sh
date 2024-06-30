# #!/bin/bash
# CUDA_VISIBLE_DEVICES=6 nohup /data/ljl/envs/bloom/bin/python main.py train \
#    --device 0 \
#    --seed 20248888 \
#    --train-path "data/UD2UC/fr.train" \
#    --dev-path "data/UD2UC/fr.dev" \
#    --test-path "data/UD2UC/fr.test" \
#    --model-path-base log/saved_parsers/fr.pt > log/train/fr.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup /data/ljl/envs/bloom/bin/python main.py train \
   --device 0 \
   --seed 20248888 \
   --train-path "data/UD2UC" \
   --dev-path "data/UD2UC" \
   --test-path "data/UD2UC" \
   --model-path-base log/saved_parsers/all_mul_mbert.pt > log/train/multi/mbert.log 2>&1 &
