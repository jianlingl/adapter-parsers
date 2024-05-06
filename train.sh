#!/bin/bash
#  gold UC: ja, all, our ud2uc: all, en,

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 0 \
#    --seed 6688 \
#    --train-path "data/universal/all.train" \
#    --dev-path "data/universal/all.dev" \
#    --test-path "data/universal/all.test" \
#    --model-path-base log/saved_parsers/all.pt > log/train/all.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 7 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/all_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/all_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/all_test.txt" \
#    --model-path-base log/saved_parsers/all_our1.pt > log/train/all_our1.log 2>&1 &

# kk kmr （train=dev） cy
nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 2 \
   --seed 6688 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/kk_train.txt" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/kk_train.txt" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/kk_test.txt" \
   --model-path-base log/saved_parsers/kk_our1.pt > log/train/kk_our1.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 3 \
   --seed 6688 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/kmr_train.txt" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/kmr_train.txt" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/kmr_test.txt" \
   --model-path-base log/saved_parsers/kmr_our1.pt > log/train/kmr_our1.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 4 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/cy_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/cy_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/cy_test.txt" \
#    --model-path-base log/saved_parsers/cy_our1.pt > log/train/cy_our1.log 2>&1 &
