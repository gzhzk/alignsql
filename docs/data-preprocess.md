# SFT 数据预处理

> 将 Spider 原始数据集转换为 LLaMA-Factory 兼容的 Alpaca 格式。

## 流程概览

```
Spider 原始数据 (JSON/Parquet)
    ↓ classify_difficulty()    按 SQL 复杂度分类
    ↓ build_schema_description()  构建数据库 Schema 描述
    ↓ build_prompt()           组装 Prompt
    ↓ 输出 Alpaca 格式 JSON
```

## 输入数据

### Spider 数据集

| 文件 | 说明 |
|------|------|
| `train_spider.json` | 主训练集，7,000 条 |
| `train_others.json` | 训练补充，1,659 条 |
| `tables.json` | 数据库 Schema 定义（表名、列名、主键、外键） |

### 数据格式

**train_spider.json** 示例：

```json
{
  "db_id": "department_management",
  "query": "SELECT count(*) FROM head WHERE age > 56",
  "question": "How many heads of the departments are older than 56 ?"
}
```

**tables.json** 结构：

```json
{
  "db_id": "department_management",
  "table_names_original": ["department", "head", "management"],
  "column_names_original": [[0, "Department_ID"], [0, "Name"], ...],
  "column_types": ["text", "text", "number", ...],
  "primary_keys": [0],
  "foreign_keys": [[5, 3]]
}
```

## 处理步骤

### Step 1: 难度分类

根据 SQL 结构划分 4 个难度级别（Spider 官方标准）：

| 难度 | 规则 | 示例 |
|------|------|------|
| **easy** | 简单 SELECT + WHERE | `SELECT * FROM table WHERE id = 1` |
| **medium** | 含 GROUP BY / ORDER BY / 聚合 | `SELECT COUNT(*) FROM table GROUP BY col` |
| **hard** | 多表 JOIN 或子查询 | `SELECT * FROM a JOIN b ON ... WHERE ... IN (SELECT ...)` |
| **extra** | UNION / EXCEPT / INTERSECT | `SELECT ... UNION SELECT ...` |

### Step 2: Schema 构建

将数据库结构转换为自然语言描述：

```
department_management contains tables such as department, head, management.
Table department has columns such as Department_ID, Name, Creation, Ranking.
department.Department_ID is the primary key.
The head_ID of management is the foreign key of head_ID of head.
```

包含信息：
- 表名列表
- 每张表的列名
- 主键声明
- 外键关系

### Step 3: Prompt 组装

使用 DB-GPT-Hub 风格的角色扮演 Prompt：

```json
{
  "system": "I want you to act as a SQL terminal in front of an example database...",
  "instruction": "Convert the following question to a SQL query based on the database schema.",
  "input": "###Input:\n{schema}\n\nQuestion: {question}\n\n###Response:",
  "output": "{sql_query}"
}
```

## 输出格式

### Alpaca 格式

**Alpaca 数据格式** 是大语言模型（LLM）微调（尤其是指令微调 Instruction Tuning）中最经典、最常用的数据格式之一。它最初由[斯坦福大学的 Alpaca 项目](https://crfm.stanford.edu/2023/03/13/alpaca.html)提出，主要用于训练模型理解并执行人类的指令。

它的核心是用一个 JSON 数组来组织数据，每条数据（即一个 JSON 对象）通常包含三个核心字段：instruction、input 和 output。

```json
{
  "db_id": "department_management",
  "difficulty": "medium",
  "system": "I want you to act as a SQL terminal in front of an example database...",
  "instruction": "Convert the following question to a SQL query based on the database schema.",
  "input": "###Input:\n{schema_description}\n\nQuestion: {question}\n\n###Response:",
  "output": "SELECT count(*) FROM head WHERE age > 56"
}
```

## 使用方法

```bash
# 处理训练集
python scripts/prepare_sft.py \
    --dataset_dir dataset \
    --input dataset/train-00000-of-00001.parquet \
    --output data_processed/sft_train.json

# 处理补充数据
python scripts/prepare_sft.py \
    --dataset_dir dataset \
    --input dataset/train_others.json \
    --output data_processed/sft_others.json

# 合并训练集
python -c "
import json
with open('data_processed/sft_train.json') as f:
    train = json.load(f)
with open('data_processed/sft_others.json') as f:
    others = json.load(f)
with open('data_processed/sft_data.json', 'w') as f:
    json.dump(train + others, f, ensure_ascii=False, indent=2)
print(f'合并后共 {len(train + others)} 条数据')
"
```

## 输出统计

处理完成后会输出各难度级别的分布：

```
输出数据统计 (8659 条):
  Easy:    2743 (31.7%)
  Medium:  4649 (53.7%)
  Hard:    607 (7.0%)
  Extra:   650 (7.5%)
```

## 难度分布对比

| 难度 | Spider 官方 | 预处理后 |
|------|:-----------:|:--------:|
| Easy | 31.7% | 31.7% |
| Medium | 53.7% | 53.7% |
| Hard | 7.0% | 7.0% |
| Extra | 7.5% | 7.5% |

预处理保持了 Spider 官方的难度分布比例。


## 相关配置

在 `config/dataset_info.json` 中注册数据集：

```json
{
  "spider_sft": {
    "file_name": "data_processed/sft_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
}
```


## 参考

Schema 构建方式参考自 [DB-GPT-Hub](https://github.com/eosphoros-ai/DB-GPT-Hub)