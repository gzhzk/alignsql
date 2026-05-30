# Self-Consistency 推理方案

## 概述

Self-Consistency（自一致性）是一种提升 Text-to-SQL 准确率的推理策略。核心思想是：对同一个问题生成多个候选 SQL，选择执行结果一致的作为最终答案。

```
输入问题
    ↓ 采样 N 次
[SQL₁, SQL₂, ..., SQLₙ]  (temperature > 0)
    ↓ 逐一执行
[结果₁, 结果₂, ..., 结果ₙ]
    ↓ 结果聚类投票
选择票数最多的结果对应的 SQL
```

## 原理详解

### 1. 采样多样性

- 使用 `temperature > 0`（如 0.8）采样，让模型在不同 decoding 路径上生成多样化 SQL
- 设置 `n_candidates` 控制候选数量（如 12 次）
- vLLM 原生支持 `n` 参数，复用一次 prefill 阶段，仅重复 decode，降低计算开销

### 2. 执行结果投票

对每个候选 SQL 执行两步：

**Step 1: 执行并聚类**

```
SQL₁ → 结果 A (4条记录)
SQL₂ → 结果 A (4条记录)  →  聚类 A，票数 3
SQL₃ → 结果 B (2条记录)  →  聚类 B，票数 1
SQL₄ → 结果 A (4条记录)
SQL₅ → 结果 A (4条记录)
SQL₆ → 执行错误 → 丢弃
```

**Step 2: 选择获胜者**

- 票数最多的聚类获胜
- 平票时选择平均 SQL 长度最短的（更简洁）

### 3. 为什么有效

| 场景 | 无 SC | 有 SC |
|------|-------|-------|
| 模型偶尔生成错误 SQL | 错误答案 | 通过投票过滤 |
| 正确 SQL 执行结果一致 | 一致通过 | 自然胜出 |
| 正确但有多种等效写法 | 可能不一致 | 投票可统一 |

## 使用方法

```bash
python scripts/evaluate_vllm.py \
    --model_path <模型路径> \
    --spider_dir <数据集路径> \
    --stage sft \
    --self_consistency \
    --n_candidates 12 \
    --temperature 0.8 \
    --etype all
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--self_consistency` | 启用 SC 模式 | false |
| `--n_candidates` | 每个问题采样的候选数 | 5 |
| `--temperature` | 采样温度，需 > 0 | 0.1 |

## 实验结果

### SC-N vs Greedy 对比

| 难度 | 样本数 | Greedy | SC-5 | SC-8 | SC-12 |
|------|--------|--------|------|------|-------|
| easy | 248 | 88.71% | 89.11% | 89.52% | 89.92% |
| medium | 446 | 73.77% | 75.34% | 76.46% | 76.91% |
| hard | 174 | 66.67% | 64.94% | 67.82% | 67.24% |
| extra | 166 | 46.39% | 51.20% | 52.41% | 50.60% |
| **all** | **1034** | **71.76%** | **73.02%** | **74.27%** | **74.18%** |

### SC-N 全量对比（Spider dev）

| 配置 | Execution | Exact Match |
|------|-----------|-------------|
| Greedy (T=0) | 71.76% | 67.02% |
| SC-5 (T=0.8) | 73.02% | 68.38% |
| SC-8 (T=0.8) | 74.27% | 68.57% |
| SC-12 (T=0.8) | 74.18% | 68.96% |

### 难度分析

| 难度 | SC-5 提升 | SC-8 提升 | SC-12 提升 |
|------|-----------|-----------|------------|
| easy | +0.40% | +0.81% | +1.21% |
| medium | +1.57% | +2.69% | +3.14% |
| hard | -1.73% | +1.15% | +0.57% |
| extra | +4.81% | +6.02% | +4.21% |
| **all** | **+1.26%** | **+2.51%** | **+2.42%** |

### 分析结论

- **SC-8 效果最好**：Execution 74.27%，比 Greedy 提升 2.51%
- **SC-5 性价比高**：仅用 5 个候选，提升 1.26%，适合快速实验
- **extra 难度提升最显著**：SC-N 在复杂 SQL（UNION/INTERSECT）上效果最好
- **hard 难度不稳定**：SC-5 在 hard 上反而下降，说明候选数过少时投票不可靠

## 输出文件

运行 SC 模式后，会生成两个文件：

```
outputs/sft/sc_n12/
├── results.json    # 最终评测结果
└── candidates.json # 每个问题的候选 SQL（可用于 DPO 训练）
```

### candidates.json 结构

```json
{
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "gold_sql": "SELECT count(*) FROM singer",
  "candidates": [
    "SELECT count(*) FROM singer",
    "SELECT count(*) FROM singer",
    "SELECT count(*) FROM singer",
    ...  // 共 n_candidates 个
  ]
}
```

## 实现细节

### 核心函数

```python
# alignsql/models/inference.py

def sample_candidates(llm, tokenizer, prompts, *, n=5, temperature=0.8, ...):
    """采样 N 个候选 SQL"""
    params = SamplingParams(temperature=temperature, n=n, ...)
    outputs = llm.generate(formatted, params)
    return [[extract_sql_fn(out.text) for out in o.outputs] for o in outputs]

def execute_and_vote(candidates, db_paths):
    """执行投票，返回最终 SQL"""
    # 1. 执行每个候选
    # 2. 按结果聚类（空结果不聚类）
    # 3. 票数最多者获胜
    # 4. 平票取最短 SQL
```

### 空结果处理

空结果集（0 行）不参与聚类，防止错误 SQL 形成虚假多数：

```python
if not rows:
    key = f"_EMPTY_{empty_idx}"  # 每条空结果独立分组
    empty_idx += 1
```

## 注意事项

1. **温度必须 > 0**：temperature=0 会导致每次采样相同，无法生成多样化候选
2. **候选数影响**：n 越大投票越可靠，但推理时间线性增长
3. **平票处理**：多条候选结果完全一致时，选择 SQL 最短的（更简洁）
4. **执行错误**：执行失败的候选直接丢弃，不参与投票