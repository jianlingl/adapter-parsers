#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate parserL

nohup python main.py test \
   --device 1 \
   --test-path "data/universal/en/En.u1.test" \
   --model-path "log/models/en_optimizerAda_cat_span_Tag_Position_fine_tune_partitioned_transformer1.pt" > log/test_log/en_optimizerAda_cat_span_Tag_Position_fine_tune_partitioned_transformer1.testlog 2>&1 &
sleep 5

nohup python main.py test \
   --device 1 \
   --test-path "data/universal/zh/Zh.u1.test" \
   --model-path "log/models/zh_optimizerAda_cat_span_Tag_Position_fine_tune_partitioned_transformer1.pt" > log/test_log/zh_optimizerAda_cat_span_Tag_Position_fine_tune_partitioned_transformer1.testlog 2>&1 &