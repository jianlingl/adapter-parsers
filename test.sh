#!/bin/bash
   # --test-path "data/universal/en.test" \

nohup /home/ljl/.conda/envs/parserL/bin/python main.py test \
   --device 1 \
   --model-path "log/saved_parsers/en_our.pt" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_test.txt" \
   --cross-test --cross-folder "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/" > log/test/our-en-cross.test.log 2>&1 &

