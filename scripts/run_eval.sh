#!/bin/bash
# 评测入口：greedy baseline + Self-Consistency 消融
#
# 用法:
#   bash scripts/run_eval.sh                        # 默认 sft，全量消融（greedy + SC-5/8/12）
#   bash scripts/run_eval.sh --model dpo            # DPO 模型（全量消融）
#   bash scripts/run_eval.sh --model dpo 5          # DPO 模型，仅 SC-5
#   bash scripts/run_eval.sh --model dpo --model_path /path/to/model # 自定义模型路径
#   bash scripts/run_eval.sh --model dpo --greedy   # DPO 模型，仅 greedy
#
# 注: 基座模型 (zeroshot) 评测走 scripts/run_zeroshot.sh

MODEL="sft"
MODEL_PATH=""
SPIDER_DIR="/root/alignsql/dataset/spider"
TEMPERATURE=0.8
MAX_TOKENS=512
GREEDY_ONLY=false

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
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
        --greedy)
            GREEDY_ONLY=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# 未指定 model_path 时按 model 名拼接
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/root/autodl-tmp/models/${MODEL}/merged"
fi

do_greedy() {
    echo "========================================"
    echo "  Greedy Baseline (${MODEL})"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path "${MODEL_PATH}" \
        --spider_dir "${SPIDER_DIR}" \
        --stage "${MODEL}" \
        --dataset spider \
        --temperature 0 \
        --max_new_tokens "${MAX_TOKENS}"
}

do_sc() {
    local N=$1
    echo "========================================"
    echo "  Self-Consistency N=${N} (${MODEL})"
    echo "========================================"
    python scripts/evaluate_vllm.py \
        --model_path "${MODEL_PATH}" \
        --spider_dir "${SPIDER_DIR}" \
        --stage "${MODEL}" \
        --dataset spider \
        --temperature "${TEMPERATURE}" \
        --max_new_tokens "${MAX_TOKENS}" \
        --self_consistency \
        --n_candidates "${N}"
}

if [ "$GREEDY_ONLY" = true ]; then
    do_greedy
elif [ -n "$1" ]; then
    do_sc "$1"
else
    do_greedy
    for N in 5 8 12; do
        do_sc "$N"
    done

    echo ""
    echo "=========== 结果汇总 ==========="
    echo "Model: ${MODEL}"
    OUTDIR="outputs/spider/${MODEL}"
    echo "Greedy:  $(grep -oP '"exec": \K[0-9.]+' "${OUTDIR}/results.json" | tail -1)"
    for N in 5 8 12; do
        echo "SC N=${N}: $(grep -oP '"exec": \K[0-9.]+' "${OUTDIR}/sc_n${N}/results.json" | tail -1)"
    done
fi
