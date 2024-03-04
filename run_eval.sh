#!/bin/bash

# 遍历从0到22的整数
for i in {0..??}
do
   # 将每个整数作为参数传递给eval.py脚本
#    python eval.py $i
   CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_gsm8k.py --model /data/siqizhu/llama-2-7b-lora-math --target_model lora_$i --data_file ./data/test/GSM8K_test.jsonl
#    echo "lora_{$i} done"
   # 或者如果你的环境中是python3
   # python3 eval.py $i
done
