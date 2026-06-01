#!/bin/bash
# Zero-shot 基座模型评测脚本

cd /root/alignsql

python scripts/evaluate_vllm.py \
    --model_path /root/autodl-tmp/models/qwen3-8b \
    --spider_dir /root/alignsql/dataset/spider \
    --stage zeroshot \
    --split dev \
    --max_samples -1 \
    --temperature 0 \
    --max_new_tokens 512 \
    --etype all