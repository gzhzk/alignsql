# Spider 评测系统文档

## 概述

本目录包含 Spider 数据集 SQL 生成能力的评估脚本，支持 zero-shot / SFT / DPO 各阶段模型的效果评测。

## 目录结构

```
scripts/
├── evaluate_vllm.py       # vLLM 加速版（推荐，用于正式评测）
├── evaluate_transformer.py # Transformers 版本（调试用）
└── run_zeroshot.sh        # 快速运行脚本
```

## 脚本说明

### 1. evaluate_vllm.py（推荐）

使用 vLLM 进行批量推理，速度快（~1000 samples/s）。

**优势**：
- 批量推理，GPU 利用率高
- PagedAttention 显存管理
- 前缀缓存加速

**使用方法**：
```bash
python scripts/evaluate_vllm.py \
    --model_path /path/to/model \
    --spider_dir /path/to/spider \
    --stage zeroshot \
    --split dev \
    --max_samples 200 \
    --temperature 0.1 \
    --max_new_tokens 384
```

### 2. evaluate_transformer.py

使用 HuggingFace Transformers，适合调试或 vLLM 不可用时。

```bash
python scripts/evaluate_transformer.py \
    --model_path /path/to/model \
    --spider_dir /path/to/spider \
    --stage sft \
    --batch_size 8
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径 | 必填 |
| `--spider_dir` | Spider 数据集路径 | 必填 |
| `--stage` | 实验阶段 | `zeroshot` |
| `--split` | 数据集划分 | `dev` |
| `--max_samples` | 最大样本数 (-1 全部) | -1 |
| `--temperature` | 采样温度 | 0.1 |
| `--max_new_tokens` | 最大生成长度 | 384 |
| `--output_dir` | 结果输出目录 | `experiments` |

## 输出格式

结果保存到 `experiments/{stage}/results.json`：

```json
{
  "accuracy": 41.8,
  "stage": "zeroshot",
  "results": [
    {
      "index": 0,
      "db_id": "concert_singer",
      "question": "How many singers do we have?",
      "gold_sql": "SELECT count(*) FROM singer",
      "pred_sql": "SELECT COUNT(*) FROM singer;",
      "correct": true
    },
    ...
  ]
}
```

## 评测结果基准

| 模型 | 准确率 | 说明 |
|------|--------|------|
| Qwen3-8B (zero-shot) | 41.8% | Baseline |
| Qwen3-8B (SFT) | TBD | 待训练后评测 |
| Qwen3-8B (DPO) | TBD | 待训练后评测 |

## 工作流程

1. **Zero-shot 基准测试**
   ```bash
   python scripts/evaluate_vllm.py \
       --model_path /path/to/qwen3-8b \
       --spider_dir /path/to/spider \
       --stage zeroshot
   ```

2. **SFT 模型评测**
   ```bash
   python scripts/evaluate_vllm.py \
       --model_path /path/to/sft-model \
       --spider_dir /path/to/spider \
       --stage sft
   ```

3. **DPO 模型评测**
   ```bash
   python scripts/evaluate_vllm.py \
       --model_path /path/to/dpo-model \
       --spider_dir /path/to/spider \
       --stage dpo
   ```

## 核心模块

### load_schema
从 Spider `tables.json` 加载数据库 schema，生成易读文本格式。

### extract_sql
从模型输出中提取纯 SQL 语句，支持多行 SQL 和代码块处理。

### compare_results
比较预测结果与标准结果，支持浮点数容差和行顺序不敏感比较。

### batch_generate_sql
vLLM 批量生成 SQL，充分利用 GPU 并行能力。

## 注意事项

1. **Schema 格式**: 简化格式包含表名、列名、类型、主键和外键关系
2. **SQL 提取**: 使用最后一个分号作为 SQL 结束标志
3. **结果比较**: 排序后比较，忽略行顺序
4. **错误处理**: schema 加载失败会跳过该样本

## 相关文档

- [Zero-shot 方案](zeroshot.md) - 零样本 SQL 生成技术细节
- [项目报告](project_report.md) - 整体项目介绍

## 更新日志

- 2026-05-21: 初始版本，vLLM 批量推理优化