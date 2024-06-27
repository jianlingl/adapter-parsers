#!/bin/bash
nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 0 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/en.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/en.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/en.test" \
   --model-path-base log/saved_parsers/chatGPT_all/en.pt > log/train/chatGPT_all/en.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 1 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/de.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/de.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/de.test" \
   --model-path-base log/saved_parsers/chatGPT_all/de.pt > log/train/chatGPT_all/de.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 2 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/fr.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/fr.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/fr.test" \
   --model-path-base log/saved_parsers/chatGPT_all/fr.pt > log/train/chatGPT_all/fr.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 3 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/he.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/he.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/he.test" \
   --model-path-base log/saved_parsers/chatGPT_all/he.pt > log/train/chatGPT_all/he.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 4 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/ko.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/ko.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/ko.test" \
   --model-path-base log/saved_parsers/chatGPT_all/ko.pt > log/train/chatGPT_all/ko.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 5 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/sv.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/sv.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/sv.test" \
   --model-path-base log/saved_parsers/chatGPT_all/sv.pt > log/train/chatGPT_all/sv.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 6 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/zh.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/zh.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/chatGPT_all/zh.test" \
   --model-path-base log/saved_parsers/chatGPT_all/zh.pt > log/train/chatGPT_all/zh.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
#    --device 7 \
#    --seed 20248888 \
#    --train-path "data/universal/size_as_UD/ko.train" \
#    --dev-path "data/universal/size_as_UD/ko.dev" \
#    --test-path "data/universal/size_as_UD/ko.test" \
#    --model-path-base log/saved_parsers/size_as_UD/ko__.pt > log/train/size_as_UD/ko__.log 2>&1 &