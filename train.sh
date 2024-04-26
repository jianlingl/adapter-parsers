#!/bin/bash
#  our ud2uc: en de fr hu ko sv
# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 7 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/zh.train" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/zh.dev" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/zh.test" \
#    --model-path-base log/saved_parsers/zh.pt > log/train/zh.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 6 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/de_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/de_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/de_test.txt" \
#    --model-path-base log/saved_parsers/de_our.pt > log/train/de_our.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 5 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/fr_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/fr_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/fr_test.txt" \
#    --model-path-base log/saved_parsers/fr_our.pt > log/train/fr_our.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 4 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_test.txt" \
#    --model-path-base log/saved_parsers/en_our.pt > log/train/en_our.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 3 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/all_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/all_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/all_test.txt" \
#    --model-path-base log/saved_parsers/all_our.pt > log/train/all_our.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 2 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/ko_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/ko_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/ko_test.txt" \
#    --model-path-base log/saved_parsers/ko_our.pt > log/train/ko_our.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 1 \
#    --seed 6688 \
#    --train-path "data/universal/all.train" \
#    --dev-path "data/universal/all.dev" \
#    --test-path "data/universal/all.test" \
#    --model-path-base log/saved_parsers/all.pt > log/train/all.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 0 \
#    --seed 6688 \
#    --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/sv_train.txt" \
#    --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/sv_dev.txt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/sv_test.txt" \
#    --model-path-base log/saved_parsers/sv_our.pt > log/train/sv_our.log 2>&1 &
