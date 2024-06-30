#!/bin/bash

CUDA_VISIBLE_DEVICES=7 nohup /data/ljl/envs/bloom/bin/python main.py test \
   --device 6 \
   --model-path "log/saved_parsers/en_predTag.pt" \
   --cross-test --cross-folder "data/UD2UC" > log/test/en_mbert.test 2>&1 &
