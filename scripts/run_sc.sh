#!/bin/bash
# Self-Consistency 推理评测 & N 值消融
# 用法: bash scripts/run_sc.sh [n_candidates]
#
# 不传参数时依次跑 greedy baseline + N=3,5,8 消融对比

MODEL_PATH=/root/autodl-tmp/models/sft/merged
SPIDER_DIR=/root/alignsql/dataset
STAGE=sft
TEMPERATURE=0.8
MAX_TOKENS=512

if [ -n "$1" ]; then
    # 单次运行指定 N 值
    N=$1
    echo "========================================"
    echo "  Self-Consistency N=${N}"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path ${MODEL_PATH} \
        --spider_dir ${SPIDER_DIR} \
        --stage ${STAGE} \
        --temperature ${TEMPERATURE} \
        --max_new_tokens ${MAX_TOKENS} \
        --self_consistency \
        --n_candidates ${N}
else
    # 先跑 greedy baseline
    echo "========================================"
    echo "  Greedy Baseline"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path ${MODEL_PATH} \
        --spider_dir ${SPIDER_DIR} \
        --stage ${STAGE} \
        --temperature 0 \
        --max_new_tokens ${MAX_TOKENS}

    # 再跑不同 N 值的 SC
    for N in 3 5 8; do
        echo "========================================"
        echo "  Self-Consistency N=${N}"
        echo "========================================"
        python scripts/evaluate_vllm.py \
            --model_path ${MODEL_PATH} \
            --spider_dir ${SPIDER_DIR} \
            --stage ${STAGE} \
            --temperature ${TEMPERATURE} \
            --max_new_tokens ${MAX_TOKENS} \
            --self_consistency \
            --n_candidates ${N}
    done

    echo ""
    echo "=========== 结果汇总 ==========="
    echo "Greedy:  $(grep -oP '"exec": \K[0-9.]+' outputs/sft/results.json | tail -1)"
    for N in 3 5 8; do
        echo "SC N=${N}: $(grep -oP '"exec": \K[0-9.]+' outputs/sc_n${N}/results.json | tail -1)"
    done
fi
