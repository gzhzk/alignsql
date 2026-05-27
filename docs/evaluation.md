# Spider 评测系统

## 概述

基于官方 Spider 评测逻辑，支持三种评测模式：执行精度（exec）、精确匹配（match）、全部（all）。

## 使用方法

### 命令行参数

```bash
python scripts/evaluate_vllm.py \
    --model_path <模型路径> \
    --spider_dir <数据集路径> \
    --stage <zeroshot|sft|dpo> \
    --split <dev|train> \
    --etype <all|exec|match> \
    --max_new_tokens 512 \
    --temperature 0 \
    --max_samples -1 \
    --output_dir outputs
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 模型路径 |
| `--spider_dir` | 必填 | Spider 数据集路径 |
| `--stage` | zeroshot | 评测阶段 |
| `--split` | dev | 评测数据集划分 |
| `--etype` | all | 评测类型 |
| `--max_new_tokens` | 384 | 最大生成 token 数 |
| `--temperature` | 0.1 | 生成温度 |
| `--max_samples` | -1 | 最大样本数，-1 为全部 |
| `--output_dir` | outputs | 输出目录 |

### 评测模式

| 模式 | 说明 |
|------|------|
| `--etype all` | 同时报告 exec 和 match |
| `--etype exec` | 仅执行精度 |
| `--etype match` | 仅精确匹配 |

## 官方评测逻辑

### 1. 执行精度（Execution Accuracy）

将预测 SQL 和标准 SQL 都在真实数据库上执行，比较结果是否一致。

```python
execute_sql(db, pred_sql) == execute_sql(db, gold_sql)
```

### 2. 精确匹配（Exact Match）

将 SQL 解析成语义结构，比较 10 个组件：

| 组件 | 评测内容 |
|------|----------|
| select | SELECT 列名、聚合函数、distinct |
| select(no AGG) | SELECT 不含聚合的部分 |
| where | WHERE 条件 |
| where(no OP) | WHERE 不含操作符的部分 |
| group(no Having) | GROUP BY 不含 HAVING |
| group | 完整的 GROUP BY + HAVING |
| order | ORDER BY 和 LIMIT |
| and/or | AND/OR 连接词 |
| IUEN | INTERSECT/UNION/EXCEPT |
| keywords | WHERE/GROUP/ORDER 等关键词 |

所有组件 F1 都为 1 才算精确匹配成功。

### 3. 难度分级

按 SQL 复杂度分 4 级：

- **easy**: component1 ≤ 1, others = 0, component2 = 0
- **medium**: (others ≤ 2, comp1 ≤ 1, comp2 = 0) 或 (comp1 ≤ 2, others < 2)
- **hard**: (others > 2, comp1 ≤ 2) 或 (2 < comp1 ≤ 3, others ≤ 2)
- **extra**: 其他情况

其中：
- component1: where、group、order、limit、join、or、like
- component2: union、except、intersect

## 输出示例

```
================================================================================
                    easy              medium             hard              extra             all
count               500               300                200               34                1034
================================================================================
                    EXECUTION ACCURACY
================================================================================
execution          0.850             0.720              0.550             0.350             0.720

================================================================================
                    EXACT MATCHING ACCURACY
================================================================================
exact match        0.820             0.680              0.500             0.300             0.680

--------------------------------------------------------------------------------
                  PARTIAL MATCHING ACCURACY
--------------------------------------------------------------------------------
select             0.900             0.850              0.750             0.600             0.820
where              0.880             0.800              0.650             0.450             0.760
group              0.920             0.850              0.700             0.500             0.810
...
```

## 文件结构

```
alignsql/
├── scripts/
│   ├── evaluate_vllm.py      # 主评测脚本
│   ├── process_sql.py         # 官方 SQL 解析
│   ├── evaluation.py          # 官方评测逻辑
│   ├── run_zeroshot.sh        # zeroshot 运行脚本
│   └── run_sft.sh             # sft 运行脚本
├── dataset/                   # Spider 数据集
│   ├── dev.json
│   ├── train.json
│   ├── tables.json
│   └── database/              # SQLite 数据库
└── outputs/                  # 评测结果
    ├── zeroshot/
    │   └── results.json
    └── sft/
        └── results.json
```

## 实验结果汇总

### Qwen3-8B 模型对比

| 模型 | Execution | Exact Match | 说明 |
|------|-----------|-------------|------|
| **Zero-shot** | 43.91% | 35.69% | 基座模型无微调 |
| **SFT** | 72.24% | 67.41% | LoRA 微调后 |

### 按难度分级对比

| 难度 | 样本数 | Zero-shot EX | SFT EX | 提升 |
|------|--------|--------------|--------|------|
| easy | 248 | 72.18% | 89.11% | +16.93% |
| medium | 446 | 45.96% | 74.44% | +28.48% |
| hard | 174 | 25.86% | 65.52% | +39.66% |
| extra | 166 | 15.06% | 48.19% | +33.13% |
| **all** | **1034** | **43.91%** | **72.24%** | **+28.33%** |

### 实验结论

1. **SFT 微调效果显著**：整体提升 28.33%，从 43.91% 提升到 72.24%
2. **难度越高提升越大**：hard 级别提升 39.66%，extra 级别提升 33.13%
3. **exec vs exact 差约 5%**：模型生成的 SQL 语义正确性较高，与标准答案结构差异较小

### 结果文件位置

| 实验 | 结果文件 |
|------|----------|
| Zero-shot | `outputs/zeroshot/results.json` |
| SFT | `outputs/sft/results.json` |
| DPO | `outputs/dpo/results.json` |

## 依赖

- vLLM（批量推理加速）
- nltk（SQL tokenize）
- tqdm（进度条）
- sqlite3（内置）