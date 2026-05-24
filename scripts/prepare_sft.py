"""
Spider 数据集 SFT 数据预处理脚本

参考 DB-GPT-Hub 的数据处理方式：
1. 加载 Spider 数据集（parquet 或 JSON 格式）
2. 解析数据库 Schema（DB-GPT-Hub 风格的自然语言描述）
3. 难度分类（Easy/Medium/Hard/Extra）
4. 生成 LLaMA-Factory 兼容的训练数据

DB-GPT-Hub 的核心设计：
- 角色扮演式 instruction
- 详细的 Schema 描述（表名、列名、类型、主键、外键）
- ###Input/###Response 分隔符
"""
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


SYSTEM_PROMPT = """I want you to act as a SQL terminal in front of an example database,
you need only to return the sql command to me.
Below is an instruction that describes a task, Write a response that appropriately completes the request."""


TASK_INSTRUCTION = "Convert the following question to a SQL query based on the database schema."


def classify_difficulty(query: str) -> str:
    """
    基于 SQL 关键词的难度分类

    分类规则（Spider 官方标准）:
    - Easy: 简单 SELECT + WHERE，无聚合、无 GROUP BY
    - Medium: GROUP BY / ORDER BY / HAVING / 聚合函数
    - Hard: 多表 JOIN / 子查询
    - Extra: UNION / EXCEPT / INTERSECT / 深度嵌套子查询
    """
    query_upper = query.upper()

    if any(op in query_upper for op in ['UNION', 'EXCEPT', 'INTERSECT']):
        return 'extra'

    from_count = query_upper.count(' FROM ')
    if from_count > 1:
        return 'hard'

    if query_upper.count('SELECT') > 1:
        return 'hard'

    if any(kw in query_upper for kw in ['GROUP BY', 'ORDER BY', 'HAVING']):
        return 'medium'

    if any(f in query_upper for f in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
        return 'medium'

    return 'easy'


def build_schema_description(tables_path: Path, db_id: str) -> Optional[str]:
    """
    构建 DB-GPT-Hub 风格的 Schema 描述

    DB-GPT-Hub 格式示例:
    "department_management contains tables such as department, head, management.
     Table department has columns such as Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees.
     Department_ID is the primary key.
     Table head has columns such as head_ID, name, born_state, age.
     head_ID is the primary key.
     The head_ID of management is the foreign key of head_ID of head."

    返回: 格式化后的 Schema 字符串
    """
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)

    db_schema = None
    for db in tables_data:
        if db['db_id'] == db_id:
            db_schema = db
            break

    if db_schema is None:
        return None

    table_names = db_schema['table_names_original']
    column_names = db_schema['column_names_original']
    column_types = db_schema['column_types']
    primary_keys = set(db_schema.get('primary_keys', []))
    foreign_keys = db_schema.get('foreign_keys', [])

    schema_parts = []

    for table_idx, table_name in enumerate(table_names):
        cols = []
        for i in range(len(column_names)):
            col = column_names[i]
            if col[0] == table_idx:
                cols.append(col[1])
        schema_parts.append(f"Table {table_name} has columns such as {', '.join(cols)}.")

    schema_str = f"{db_id} contains tables such as {', '.join(table_names)}.\n"
    schema_str += " ".join(schema_parts)

    pk_parts = []
    for i, pk_idx in enumerate(primary_keys):
        if pk_idx < len(column_names):
            col = column_names[pk_idx]
            if col[0] < len(table_names):
                pk_parts.append(f"{table_names[col[0]]}.{col[1]} is the primary key")
    if pk_parts:
        schema_str += "\n" + ". ".join(pk_parts) + "."

    fk_parts = []
    for fk in foreign_keys:
        if len(fk) == 2:
            i1, i2 = fk
            if i1 < len(column_names) and i2 < len(column_names):
                c1, c2 = column_names[i1], column_names[i2]
                t1, t2 = c1[0], c2[0]
                if t1 < len(table_names) and t2 < len(table_names):
                    fk_parts.append(f"The {c1[1]} of {table_names[t1]} is the foreign key of {c2[1]} of {table_names[t2]}")
    if fk_parts:
        schema_str += "\n" + ". ".join(fk_parts) + "."

    return schema_str


def build_prompt(schema_str: str, question: str) -> str:
    """
    构建 DB-GPT-Hub 风格的输入 Prompt

    格式:
    ###Input:
    {schema_description}
    {question}
    ###Response:
    """
    return f"###Input:\n{schema_str}\n\nQuestion: {question}\n\n###Response:"


def process_parquet(parquet_path: Path, tables_path: Path,
                   output_path: Path, max_samples: int = -1):
    """
    处理 parquet 格式数据

    输入: Spider train-00000-of-00001.parquet
    输出: LLaMA-Factory 兼容的 JSON 数据
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    total = table.num_rows if max_samples < 0 else min(max_samples, table.num_rows)

    results = []
    skipped = 0

    for i in tqdm(range(total), desc='Processing'):
        db_id = table.column('db_id')[i].as_py()
        query = table.column('query')[i].as_py()
        question = table.column('question')[i].as_py()

        schema_str = build_schema_description(tables_path, db_id)
        if schema_str is None:
            skipped += 1
            continue

        difficulty = classify_difficulty(query)
        input_text = build_prompt(schema_str, question)

        item = {
            'db_id': db_id,
            'difficulty': difficulty,
            'system': SYSTEM_PROMPT,
            'instruction': TASK_INSTRUCTION,
            'input': input_text,
            'output': query
        }
        results.append(item)

    print(f'Processed: {len(results)} samples, Skipped: {skipped}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def process_json(json_path: Path, tables_path: Path,
                output_path: Path, max_samples: int = -1):
    """
    处理 JSON 格式数据

    输入: Spider train_spider.json
    输出: LLaMA-Factory 兼容的 JSON 数据
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data) if max_samples < 0 else min(max_samples, len(data))

    results = []
    skipped = 0

    for item in tqdm(data[:total], desc='Processing'):
        db_id = item['db_id']
        query = item['query']
        question = item['question']

        schema_str = build_schema_description(tables_path, db_id)
        if schema_str is None:
            skipped += 1
            continue

        difficulty = classify_difficulty(query)
        input_text = build_prompt(schema_str, question)

        result = {
            'db_id': db_id,
            'difficulty': difficulty,
            'system': SYSTEM_PROMPT,
            'instruction': TASK_INSTRUCTION,
            'input': input_text,
            'output': query
        }
        results.append(result)

    print(f'Processed: {len(results)} samples, Skipped: {skipped}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def analyze_output(output_path: Path):
    """分析输出数据的难度分布"""
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    counts = {'easy': 0, 'medium': 0, 'hard': 0, 'extra': 0}
    for item in data:
        difficulty = item.get('difficulty', 'easy')
        counts[difficulty] = counts.get(difficulty, 0) + 1

    total = len(data)
    print(f'\n输出数据统计 ({total} 条):')
    print(f"  Easy:   {counts['easy']:4d} ({100*counts['easy']/total:.1f}%)")
    print(f"  Medium: {counts['medium']:4d} ({100*counts['medium']/total:.1f}%)")
    print(f"  Hard:   {counts['hard']:4d} ({100*counts['hard']/total:.1f}%)")
    print(f"  Extra:  {counts['extra']:4d} ({100*counts['extra']/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Spider SFT 数据预处理 (DB-GPT-Hub 风格)')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='数据集目录')
    parser.add_argument('--input', type=str, default=None,
                        help='输入文件路径（默认使用 train-00000-of-00001.parquet）')
    parser.add_argument('--output', type=str, default='data/sft_data.json',
                        help='输出文件路径')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='最大样本数（-1 表示全部）')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tables_path = dataset_dir / 'tables.json'

    if not tables_path.exists():
        print(f'Error: tables.json not found at {tables_path}')
        return

    if args.input:
        input_path = Path(args.input)
        process_func = process_parquet if input_path.suffix == '.parquet' else process_json
    else:
        parquet_path = dataset_dir / 'train-00000-of-00001.parquet'
        if parquet_path.exists():
            input_path = parquet_path
            process_func = process_parquet
        else:
            input_path = dataset_dir / 'train_spider.json'
            process_func = process_json

    if not input_path.exists():
        print(f'Error: input file not found')
        return

    output_path = Path(args.output)

    print(f'Input: {input_path}')
    print(f'Output: {output_path}')

    process_func(input_path, tables_path, output_path, args.max_samples)
    analyze_output(output_path)

    print(f'\n数据已保存到: {output_path}')


if __name__ == '__main__':
    main()