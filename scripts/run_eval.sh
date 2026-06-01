#!/bin/bash
# 训练后模型评测脚本（适用于 SFT / DPO / RL 等 stage）
# 默认并行跑 greedy baseline + N=5,8,12 Self-Consistency 消融
#
# 用法:
#   bash scripts/run_eval.sh                              # 默认 sft，Spider 完整消融
#   bash scripts/run_eval.sh --stage dpo                  # DPO 模型，Spider
#   bash scripts/run_eval.sh --dataset bird               # SFT 模型，BIRD 完整消融
#   bash scripts/run_eval.sh --dataset bird --greedy      # BIRD 仅跑 greedy
#   bash scripts/run_eval.sh --stage dpo --dataset bird 5 # DPO，BIRD，仅 SC N=5
#   bash scripts/run_eval.sh --model_path /path 5         # 自定义模型路径
#
# 注: 基座模型 (zeroshot) 评测走 scripts/run_zeroshot.sh

STAGE=sft
DATASET=spider
MODEL_PATH=""
SPIDER_DIR=/root/alignsql/dataset/spider
BIRD_DIR=/root/alignsql/dataset/bird/minidev
TEMPERATURE=0.8
MAX_TOKENS=512
GREEDY_ONLY=false
NO_EVIDENCE=""

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --no_evidence)
            NO_EVIDENCE="--no_evidence"
            shift
            ;;
        --greedy)
            GREEDY_ONLY=true
            shift
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

# BIRD 默认路径
if [ "$DATASET" = "bird" ]; then
    SPIDER_DIR="${BIRD_DIR}/MINIDEV"
fi

# 如果未指定 model_path，按 stage 约定路径
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/root/autodl-tmp/models/${STAGE}/merged"
fi

do_greedy() {
    echo "========================================"
    echo "  Greedy Baseline (${STAGE}, ${DATASET})"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path "${MODEL_PATH}" \
        --spider_dir "${SPIDER_DIR}" \
        --stage "${STAGE}" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --max_new_tokens "${MAX_TOKENS}" \
        ${NO_EVIDENCE:+$NO_EVIDENCE}
}

do_sc() {
    local N=$1
    echo "========================================"
    echo "  Self-Consistency N=${N} (${STAGE}, ${DATASET})"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path "${MODEL_PATH}" \
        --spider_dir "${SPIDER_DIR}" \
        --stage "${STAGE}" \
        --dataset "${DATASET}" \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --self_consistency \
        --n_candidates "${N}" \
        ${NO_EVIDENCE:+$NO_EVIDENCE}
}

if [ "$GREEDY_ONLY" = true ]; then
    # 仅跑 greedy baseline
    do_greedy
elif [ -n "$1" ]; then
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
    echo "Dataset: ${DATASET}, Stage: ${STAGE}"
    OUTDIR="outputs/${DATASET}/${STAGE}"
    echo "Greedy:  $(grep -oP '"exec": \K[0-9.]+' "${OUTDIR}/results.json" | tail -1)"
    for N in 5 8 12; do
        echo "SC N=${N}: $(grep -oP '"exec": \K[0-9.]+' "${OUTDIR}/sc_n${N}/results.json" | tail -1)"
    done
fi
