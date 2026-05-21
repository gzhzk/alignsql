# Zero-shot SQL 生成方案

## 概述

Zero-shot 是指在不进行任何微调的情况下，直接使用预训练模型（如 Qwen3-8B）生成 SQL。这是评估模型原始能力的基准方案。

## 评测结果

| 模型 | 准确率 | 评测条件 |
|------|--------|----------|
| Qwen3-8B | **41.8%** | Spider dev 全量 1034 样本 |

## 技术方案

### System Prompt

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

### User Prompt

```python
def build_prompt(schema_str: str, question: str) -> str:
    return f"""Database Schema:
{schema_str}

Question: {question}

SQL Query:"""
```

### Schema 格式

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
# 错误写法
for table in table_names:
    if col[0] == table:  # col[0] 是整数 table_id
        ...

# 正确写法
for table_idx, table in enumerate(table_names):
    if col[0] == table_idx:  # 用 index 匹配
        ...
```

**影响**: 导致列全丢或错位，准确率从 40%+ 暴跌到 20%-

### 2. Primary Key Bug

**问题**: 假设第一列是主键

```python
# 错误
Primary key: {col_names[0]}

# 正确
pk_indices = db_schema["primary_keys"]
pk_map = {table_idx: [col_name, ...], ...}
```

### 3. Foreign Keys 完全没加

**问题**: 模型无法知道 JOIN 关系

**修复**: 添加 Foreign Key Relationships 段

### 4. 结果比较 Bug

**问题**: 顺序敏感比较，语义相同但顺序不同的结果会判错

```python
# 错误
for r1, r2 in zip(result1, result2):
    if r1 != r2: return False

# 正确
rows1 = sorted(rows1, key=str)
rows2 = sorted(rows2, key=str)
return rows1 == rows2
```

### 5. SQL 提取 Bug

**问题**: 只取第一行，复杂 SQL 被截断

```python
# 错误
if line.startswith("SELECT"):
    return line.rstrip(";") + ";"

# 正确 - 多行累积
if not in_sql:
    if line.startswith("SELECT"):
        in_sql = True
        sql_lines = [line]
else:
    sql_lines.append(line)
    if ';' in line:
        return ' '.join(sql_lines)
```

## 性能优化

### vLLM 批量推理

```python
def batch_generate_sql(llm, tokenizer, prompts, ...):
    formatted = [tokenizer.apply_chat_template(p) for p in prompts]
    outputs = llm.generate(formatted, SamplingParams(...))
    return [extract_sql(o.outputs[0].text) for o in outputs]
```

**效果**: 200 样本从 ~30 秒降到 ~0.2 秒（1000+ samples/s）

### 配置

```python
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    enforce_eager=True,
    enable_prefix_caching=True,
)
```

## 错误类型分析

### FAILED（SQL 提取失败）

- 模型输出思考过程而非 SQL
- SQL 被截断未到分号
- 输出格式异常

### 结果错误

- GROUP BY 漏写
- JOIN 条件缺失
- 列名/表名拼写错误
- 大小写敏感问题

## 后续改进方向

1. **Few-shot 示例**: 加入 2-3 个样例可能提升 3-5%
2. **Schema 优化**: 更清晰的表关系表示
3. **Prompt 调整**: 根据错误类型针对性优化

## 相关文档

- [评测系统文档](evaluation.md) - 脚本使用说明
- [项目报告](project_report.md) - 整体项目介绍