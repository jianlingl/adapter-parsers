#!/bin/bash
#  gold UC: ja, all, our ud2uc: all, en,



nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 0 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/en.train" \
   --dev-path "data/universal/size_as_UD/en.dev" \
   --test-path "data/universal/size_as_UD/en.test" \
   --model-path-base log/saved_parsers/size_as_UD/en.pt > log/train/size_as_UD/en.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 1 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/de.train" \
   --dev-path "data/universal/size_as_UD/de.dev" \
   --test-path "data/universal/size_as_UD/de.test" \
   --model-path-base log/saved_parsers/size_as_UD/de.pt > log/train/size_as_UD/de.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 2 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/he.train" \
   --dev-path "data/universal/size_as_UD/he.dev" \
   --test-path "data/universal/size_as_UD/he.test" \
   --model-path-base log/saved_parsers/size_as_UD/he.pt > log/train/size_as_UD/he.log 2>&1 &


nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 3 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/hu.train" \
   --dev-path "data/universal/size_as_UD/hu.dev" \
   --test-path "data/universal/size_as_UD/hu.test" \
   --model-path-base log/saved_parsers/size_as_UD/hu.pt > log/train/size_as_UD/hu.log 2>&1 &


nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 4 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/ko.train" \
   --dev-path "data/universal/size_as_UD/ko.dev" \
   --test-path "data/universal/size_as_UD/ko.test" \
   --model-path-base log/saved_parsers/size_as_UD/ko.pt > log/train/size_as_UD/ko.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 5 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/ja.train" \
   --dev-path "data/universal/size_as_UD/ja.dev" \
   --test-path "data/universal/size_as_UD/ja.test" \
   --model-path-base log/saved_parsers/size_as_UD/ja.pt > log/train/size_as_UD/ja.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 6 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/sv.train" \
   --dev-path "data/universal/size_as_UD/sv.dev" \
   --test-path "data/universal/size_as_UD/sv.test" \
   --model-path-base log/saved_parsers/size_as_UD/sv.pt > log/train/size_as_UD/sv.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py train \
   --device 7 \
   --seed 20248888 \
   --train-path "data/universal/size_as_UD/zh.train" \
   --dev-path "data/universal/size_as_UD/zh.dev" \
   --test-path "data/universal/size_as_UD/zh.test" \
   --model-path-base log/saved_parsers/size_as_UD/zh.pt > log/train/size_as_UD/zh.log 2>&1 &