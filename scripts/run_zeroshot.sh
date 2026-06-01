#!/bin/bash
# Zero-shot 基座模型评测脚本
# 用法:
#   bash scripts/run_zeroshot.sh              # Spider
#   bash scripts/run_zeroshot.sh bird         # BIRD

DATASET="${1:-spider}"

cd /root/alignsql

if [ "$DATASET" = "bird" ]; then
    SPIDER_DIR="/root/alignsql/dataset/bird/minidev/MINIDEV"
    DATASET_ARG="--dataset bird"
else
    SPIDER_DIR="/root/alignsql/dataset/spider"
    DATASET_ARG=""
fi

python scripts/evaluate_vllm.py \
    --model_path /root/autodl-tmp/models/qwen3-8b \
    --spider_dir "${SPIDER_DIR}" \
    --stage zeroshot \
    ${DATASET_ARG} \
    --split dev \
    --max_samples -1 \
    --temperature 0 \
    --max_new_tokens 512 \
    --etype all