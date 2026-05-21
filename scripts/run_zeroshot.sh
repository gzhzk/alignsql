#!/bin/bash
# AlignSQL Phase 1: Zero-shot Baseline Evaluation
# 测试 Qwen3-8B 基座模型的零样本 NL2SQL 能力

set -e

# 配置路径（根据 setup_record.md）
MODEL_PATH="/root/autodl-tmp/models/qwen3-8b"
SPIDER_DIR="/root/autodl-tmp/data/spider_data"
OUTPUT_DIR="experiments"

echo "=========================================="
echo "AlignSQL Phase 1: Zero-shot Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $SPIDER_DIR"
echo ""

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please download the model first:"
    echo "  pip install modelscope"
    echo "  modelscope download Qwen/Qwen3-8B --local_dir $MODEL_PATH"
    exit 1
fi

# 检查数据集路径
if [ ! -d "$SPIDER_DIR" ]; then
    echo "Error: Spider dataset not found at $SPIDER_DIR"
    echo "Please download the dataset first:"
    echo "  pip install datasets"
    echo "  python scripts/download_spider.py"
    exit 1
fi

# 执行评估（使用 vLLM 加速）
python scripts/evaluate_vllm.py \
    --model_path "$MODEL_PATH" \
    --spider_dir "$SPIDER_DIR" \
    --stage zeroshot \
    --split dev \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens 512 \
    --temperature 0.0 \

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/zeroshot/results.json"
echo "=========================================="