#!/bin/bash
# 训练后模型评测脚本（适用于 SFT / DPO / RL 等 stage）
# 默认并行跑 greedy baseline + N=5,8,12 Self-Consistency 消融
#
# 用法:
#   bash scripts/run_eval.sh                         # 默认 sft，完整消融
#   bash scripts/run_eval.sh --stage dpo             # dpo 模型，完整消融
#   bash scripts/run_eval.sh --stage dpo 5           # 仅跑 SC N=5
#   bash scripts/run_eval.sh --model_path /path 5    # 自定义模型路径
#
# 注: 基座模型 (zeroshot) 评测走 scripts/run_zeroshot.sh

STAGE=sft
MODEL_PATH=""
SPIDER_DIR=/root/alignsql/dataset
TEMPERATURE=0.8
MAX_TOKENS=512

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --spider_dir)
            SPIDER_DIR="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# 如果未指定 model_path，按 stage 约定路径
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/root/autodl-tmp/models/${STAGE}/merged"
fi

do_greedy() {
    echo "========================================"
    echo "  Greedy Baseline (${STAGE})"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path "${MODEL_PATH}" \
        --spider_dir "${SPIDER_DIR}" \
        --stage "${STAGE}" \
        --temperature 0 \
        --max_new_tokens "${MAX_TOKENS}"
}

do_sc() {
    local N=$1
    echo "========================================"
    echo "  Self-Consistency N=${N} (${STAGE})"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path "${MODEL_PATH}" \
        --spider_dir "${SPIDER_DIR}" \
        --stage "${STAGE}" \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --self_consistency \
        --n_candidates "${N}"
}

if [ -n "$1" ]; then
    # 单次运行指定 N 值
    do_sc "$1"
else
    # 先跑 greedy baseline
    do_greedy

    # 再跑不同 N 值的 SC
    for N in 5 8 12; do
        do_sc "$N"
    done

    echo ""
    echo "=========== 结果汇总 ==========="
    echo "Greedy:  $(grep -oP '"exec": \K[0-9.]+' "outputs/${STAGE}/results.json" | tail -1)"
    for N in 5 8 12; do
        echo "SC N=${N}: $(grep -oP '"exec": \K[0-9.]+' "outputs/${STAGE}/sc_n${N}/results.json" | tail -1)"
    done
fi
