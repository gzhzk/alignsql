"""
Spider 数据集评测脚本 (vLLM 加速版)
"""
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

os.environ["VLLM_USE_V1"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

from tqdm import tqdm
from vllm import LLM, SamplingParams


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

SQL_KEYWORDS = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")


def build_prompt(schema_str: str, question: str) -> str:
    return f"""Database Schema:
{schema_str}

Question: {question}

SQL Query:"""


def load_schema(spider_dir: Path, db_id: str) -> str:
    tables_file = spider_dir / "tables.json"
    with open(tables_file, "r", encoding="utf-8") as f:
        tables_data = json.load(f)
    
    db_schema = None
    for db in tables_data:
        if db["db_id"] == db_id:
            db_schema = db
            break
    
    if db_schema is None:
        raise ValueError(f"Database '{db_id}' not found")
    
    table_names = db_schema["table_names_original"]
    column_names = db_schema["column_names_original"]
    column_types = db_schema["column_types"]
    primary_keys = set(db_schema.get("primary_keys", []))
    foreign_keys = db_schema.get("foreign_keys", [])
    
    schema_parts = []
    
    for table_idx, table_name in enumerate(table_names):
        cols = []
        for i in range(len(column_names)):
            col = column_names[i]
            if col[0] == table_idx:
                col_name = col[1]
                col_type = column_types[i] if i < len(column_types) else "TEXT"
                pk_mark = " PRIMARY KEY" if i in primary_keys else ""
                cols.append(f"    {col_name} {col_type}{pk_mark}")
        
        schema_parts.append(f"Table: {table_name}\nColumns:\n" + "\n".join(cols))
    
    schema_str = "\n\n".join(schema_parts)
    
    if foreign_keys:
        schema_str += "\n\nForeign Key Relationships:"
        for fk in foreign_keys:
            if len(fk) == 2:
                i1, i2 = fk
                if i1 < len(column_names) and i2 < len(column_names):
                    c1, c2 = column_names[i1], column_names[i2]
                    t1, t2 = c1[0], c2[0]
                    if t1 < len(table_names) and t2 < len(table_names):
                        n1, n2 = table_names[t1], table_names[t2]
                        schema_str += f"\n  {n1}.{c1[1]} = {n2}.{c2[1]}"
    
    return schema_str


def extract_sql(text: str) -> str:
    if not text:
        return ""
    
    text = text.strip()
    lines = text.split('\n')
    sql_lines = []
    in_sql = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(('```', '#', '//', '--')):
            continue
        
        if not in_sql:
            if any(stripped.upper().startswith(kw) for kw in SQL_KEYWORDS):
                in_sql = True
                sql_lines = [stripped]
        else:
            sql_lines.append(stripped)
    
    if sql_lines:
        full_sql = ' '.join(sql_lines)
        last_semicolon = full_sql.rfind(';')
        if last_semicolon != -1:
            return full_sql[:last_semicolon + 1]
        return full_sql
    
    for pattern in [r'```sql\s*(.*?)```', r'```\s*(SELECT.*?)```', r'```\s*(WITH.*?)```']:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            if sql.endswith(';'):
                return sql
            return sql + ';'
    
    match = re.search(r'(SELECT\s+[\s\S]+?;)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return ""


def execute_sql(db_path: str, sql: str) -> Tuple[bool, Any]:
    try:
        if not db_path or not Path(db_path).exists():
            return False, "DB not found"
        conn = sqlite3.connect(db_path)
        conn.text_factory = str
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return True, result
    except Exception as e:
        return False, str(e)


def compare_results(r1: List, r2: List) -> bool:
    if len(r1) != len(r2):
        return False
    if len(r1) == 0:
        return True
    
    def norm(v):
        if v is None:
            return None
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, str):
            return v.strip()
        return v
    
    def norm_row(row):
        return tuple(norm(x) for x in (row if isinstance(row, (list, tuple)) else [row]))
    
    def safe_key(row):
        return tuple(str(x) if x is not None else "" for x in row)
    
    rows1 = sorted([norm_row(r) for r in r1], key=safe_key)
    rows2 = sorted([norm_row(r) for r in r2], key=safe_key)
    return rows1 == rows2


def batch_generate_sql(llm: LLM, tokenizer, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
    formatted = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True
        )
        for tok in ['<|think|>', '</think|>', '<|reserved_2066|>']:
            text = text.replace(tok, '')
        formatted.append(text)
    
    outputs = llm.generate(formatted, SamplingParams(temperature=temperature, max_tokens=max_new_tokens), use_tqdm=False)
    return [extract_sql(o.outputs[0].text) for o in outputs]


def load_model(model_path: str) -> Tuple[LLM, Any]:
    print(f"Loading model from {model_path}...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded")
    return llm, tokenizer


def evaluate(model_path: str, spider_dir: str, split: str = "dev", stage: str = "zeroshot",
            max_samples: int = -1, temperature: float = 0.1, max_new_tokens: int = 384) -> Tuple[List, float]:
    spider_path = Path(spider_dir)
    with open(spider_path / f"{split}.json", "r") as f:
        data = json.load(f)
    
    if max_samples > 0:
        data = data[:max_samples]
    
    llm, tokenizer = load_model(model_path)
    
    schemas, questions, gold_sqls, db_paths = [], [], [], []
    for item in data:
        db_id = item["db_id"]
        db_path = spider_path / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            db_path = spider_path / "database" / f"{db_id}.sqlite"
        
        try:
            schemas.append(load_schema(spider_path, db_id))
            questions.append(item["question"])
            gold_sqls.append(item["query"])
            db_paths.append(str(db_path))
        except Exception as e:
            print(f"Warning: {db_id} - {e}")
            schemas.append(None)
    
    valid_idx = [i for i, s in enumerate(schemas) if s is not None]
    if len(valid_idx) == 0:
        return [], 0.0
    
    print(f"Evaluating {len(valid_idx)} samples...")
    prompts = [build_prompt(schemas[i], questions[i]) for i in valid_idx]
    pred_sqls = batch_generate_sql(llm, tokenizer, prompts, max_new_tokens, temperature)
    
    correct = 0
    results = []
    pbar = tqdm(total=len(pred_sqls), desc="Evaluating")
    
    for i, pred_sql in enumerate(pred_sqls):
        gold = gold_sqls[valid_idx[i]]
        db = db_paths[valid_idx[i]]
        question = questions[valid_idx[i]]
        
        is_correct = False
        if pred_sql:
            _, pred_res = execute_sql(db, pred_sql)
            _, gold_res = execute_sql(db, gold)
            if isinstance(pred_res, list) and isinstance(gold_res, list):
                is_correct = compare_results(pred_res, gold_res)
        
        if is_correct:
            correct += 1
        
        results.append({
            "index": valid_idx[i],
            "db_id": data[valid_idx[i]]["db_id"],
            "question": question,
            "gold_sql": gold,
            "pred_sql": pred_sql if pred_sql else "FAILED",
            "correct": is_correct,
        })
        
        pbar.set_postfix({"Acc": f"{100*correct/(i+1):.1f}%"})
        pbar.update(1)
    
    pbar.close()
    acc = correct / len(pred_sqls) * 100
    print(f"\nAccuracy: {correct}/{len(pred_sqls)} = {acc:.2f}%")
    return results, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--spider_dir", type=str, required=True)
    parser.add_argument("--stage", type=str, default="zeroshot", choices=["zeroshot", "sft", "dpo"])
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--output_dir", type=str, default="experiments")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results, accuracy = evaluate(args.model_path, args.spider_dir, args.split, args.stage,
                                  args.max_samples, args.temperature, args.max_new_tokens)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump({"accuracy": accuracy, "stage": args.stage, "results": results}, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()