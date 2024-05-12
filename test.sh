#!/bin/bash
   # --test-path "data/universal/en.test" \
   # --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/en_test.txt" \
   # --cross-test --cross-folder "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/before/" 

nohup /home/ljl/.conda/envs/parserL/bin/python main.py test \
   --device 6 \
   --model-path "log/saved_parsers/most_simp_no_mullabel/zh_our.pt" \
   --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/most_simp_no_mul-labels/zh.test" > log/test/most_simp_no_mullabels_zh.test.log 2>&1 &

# nohup /home/ljl/.conda/envs/parserL/bin/python main.py test \
#    --device 3 \
#    --model-path "log/saved_parsers/size_as_UD_4checkepo/en_our.pt" \
#    --test-path "/home/ljl/rules_UD2UC_trees/UD2UC/UC.24.4.21/more_simp/en_test.txt" > log/test/more_simp_en.test.log 2>&1 &
