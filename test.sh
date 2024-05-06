#!/bin/bash
   # --test-path "data/universal/en.test" \
   # --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_test.txt" \
   # --cross-test --cross-folder "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/" 

nohup /home/ljl/.conda/envs/parserL/bin/python main.py test \
   --device 2 \
   --model-path "log/saved_parsers/en_our.pt" \
   --test-path "data/universal/en.test" \
   --cross-test --cross-folder "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/" > log/test/our-en-corss.test.log 2>&1 &

nohup /home/ljl/.conda/envs/parserL/bin/python main.py test \
   --device 3 \
   --model-path "log/saved_parsers/en.pt" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_test.txt" \
   --cross-test --cross-folder "data/universal/" > log/test/gold-en-corss.test.log 2>&1 &
