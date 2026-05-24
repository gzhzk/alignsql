# AlignSQL Zero-shot 评测方案

> 基于 Spider 数据集的 NL2SQL 基座模型评估

## 概述

Zero-shot 是指在不进行任何微调的情况下，直接使用预训练模型（如 Qwen3-8B）生成 SQL。这是评估模型原始能力的基准方案。

```
Qwen3-8B (基座)
    ↓ 直接推理
EX 准确率 43.91%
```

## 快速开始

```bash
cd /root/alignsql

# zeroshot 评测
python scripts/evaluate_vllm.py \
    --model_path /root/autodl-tmp/models/Qwen3-8B \
    --spider_dir /root/alignsql/dataset \
    --stage zeroshot \
    --max_new_tokens 512 \
    --temperature 0 \
    --etype all

# 或者用现成脚本
bash scripts/run_zeroshot.sh
```

## 评测指标

支持官方 Spider 三种评测模式：

| 模式 | 说明 |
|------|------|
| `--etype exec` | 执行精度 - SQL 执行结果是否一致 |
| `--etype match` | 精确匹配 - SQL 结构各组件是否完全匹配 |
| `--etype all` | 两者都报告（默认） |

### 难度分级

| 难度 | 说明 |
|------|------|
| easy | 基础查询 |
| medium | 一般复杂（带聚合、多条件） |
| hard | 较复杂（多表 JOIN、嵌套查询） |
| extra | 非常复杂（多重嵌套 UNION/INTERSECT） |

### 10 个评测组件（Exact Match）

select、select(no AGG)、where、where(no OP)、group(no Having)、group、order、and/or、IUEN、keywords

## 评测结果

### Qwen3-8B Zero-shot（Spider dev 全量 1034 样本）

| 指标 | 准确率 | 说明 |
|------|--------|------|
| **Execution Accuracy** | **43.91%** | SQL 执行结果一致 |
| **Exact Match** | **35.69%** | SQL 结构完全匹配 |

### 按难度分级

| 难度 | 样本数 | Execution | Exact Match |
|------|--------|-----------|-------------|
| easy | 248 | 72.18% | 70.16% |
| medium | 446 | 45.96% | 38.79% |
| hard | 174 | 25.86% | 10.92% |
| extra | 166 | 15.06% | 1.81% |
| **all** | **1034** | **43.91%** | **35.69%** |

### exec vs exact 的区别

| 指标 | 说明 |
|------|------|
| **exec** | 把预测 SQL 和标准 SQL 都在真实数据库上执行，结果一致就对 |
| **exact** | 把 SQL 解析成结构，逐 clause 比对（select/where/group by 等），结构完全一致才算对 |

- `exec` 通常 >= `exact`（差值说明语义对但写法不完全一致）
- 两者差约 **8%**，说明模型部分 SQL 语义正确但结构与标准答案有差异

## System Prompt

```python
SYSTEM_PROMPT = """You are an expert SQLite SQL generator.

CRITICAL RULES:
1. Output ONLY the raw SQL query
2. Do NOT use markdown code blocks
3. Do NOT include any explanation
4. Start directly with SELECT, WITH, INSERT, UPDATE, or DELETE
5. End with a semicolon

CORRECT example:
SELECT name FROM users WHERE age > 18;

Now generate the SQL query."""
```

## 用户 Prompt

```python
def build_prompt(schema_str: str, question: str) -> str:
    return f"""Database Schema:
{schema_str}

Question: {question}

SQL Query:"""
```

## Schema 格式

```text
Table: singer
Columns:
    singer_id INTEGER PRIMARY KEY
    name TEXT
    country TEXT
    age INTEGER

Table: concert
Columns:
    concert_id INTEGER PRIMARY KEY
    stadium_id INTEGER
    singer_id INTEGER

Foreign Key Relationships:
  concert.singer_id = singer.singer_id
  concert.stadium_id = stadium.stadium_id
```

## 关键修复记录

### 1. Schema Bug（致命）

**问题**: `col[0]` 是 table_id 不是 table name

```python
# 错误
for table in table_names:
    if col[0] == table:  # col[0] 是整数 table_id
        ...

# 正确
for table_idx, table in enumerate(table_names):
    if col[0] == table_idx:  # 用 index 匹配
        ...
```

### 2. 结果比较 Bug

**问题**: 顺序敏感比较，语义相同但顺序不同的结果会判错

```python
# 正确 - 排序后比较
def compare_results(r1, r2):
    rows1 = sorted([norm_row(r) for r in r1], key=safe_key)
    rows2 = sorted([norm_row(r) for r in r2], key=safe_key)
    return rows1 == rows2
```

### 3. SQL 提取 Bug

多行 SQL 累积提取，不截断

## 性能优化

使用 vLLM 批量推理，约 **10-15 分钟** 完成 1034 样本全量评测。

## 输出格式

结果保存在 `experiments/{stage}/results.json`：

```json
{
  "etype": "all",
  "stage": "zeroshot",
  "scores": {
    "easy":   {"count": 500, "exec": 0.85, "exact": 0.82},
    "medium": {"count": 300, "exec": 0.72, "exact": 0.68},
    "hard":   {"count": 200, "exec": 0.55, "exact": 0.50},
    "extra":  {"count": 34,  "exec": 0.35, "exact": 0.30},
    "all":    {"count": 1034,"exec": 0.72, "exact": 0.68}
  },
  "results": [...]
}
```

## 相关文档

- [评测系统文档](evaluation.md) - 脚本使用说明