#!/bin/bash
nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 0 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/en.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/en.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/en.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/en.pt > log/train/reconstruct_simp/en.log 2>&1 &


nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 1 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/de.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/de.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/de.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/de.pt > log/train/reconstruct_simp/de.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 2 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/fr.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/fr.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/fr.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/fr.pt > log/train/reconstruct_simp/fr.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 3 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/he.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/he.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/he.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/he.pt > log/train/reconstruct_simp/he.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 4 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/ko.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/ko.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/ko.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/ko.pt > log/train/reconstruct_simp/ko.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 5 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/sv.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/sv.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/sv.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/sv.pt > log/train/reconstruct_simp/sv.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 6 \
   --seed 20248888 \
   --train-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/zh.train" \
   --dev-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/zh.dev" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/reconstruct_simp/zh.test" \
   --model-path-base log/saved_parsers/reconstruct_simp/zh.pt > log/train/reconstruct_simp/zh.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 7 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/ko.train" \
   --dev-path "data/universal/size_as_UD/ko.dev" \
   --test-path "data/universal/size_as_UD/ko.test" \
   --model-path-base log/saved_parsers/size_as_UD/ko__.pt > log/train/size_as_UD/ko__.log 2>&1 &