#!/bin/bash
# SFT 训练后模型评测脚本

cd /root/alignsql

python scripts/evaluate_vllm.py \
    --model_path /root/autodl-tmp/models/sft/merged \
    --spider_dir /root/alignsql/dataset \
    --stage sft \
    --split dev \
    --max_samples -1 \
    --temperature 0 \
    --max_new_tokens 512 \
    --etype all