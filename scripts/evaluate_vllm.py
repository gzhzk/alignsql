"""
Spider 数据集评测脚本 (vLLM 加速版)
支持官方三种评测模式：exec / match / all
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

os.environ["VLLM_USE_V1"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from tqdm import tqdm
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor"))
from process_sql import get_schema, Schema, get_sql
from evaluation import Evaluator

from alignsql.models.inference import sample_candidates, execute_and_vote

SYSTEM_PROMPTS = {
    "zeroshot": """You are an expert SQLite SQL generator.

CRITICAL RULES:
1. Output ONLY the raw SQL query
2. Do NOT use markdown code blocks
3. Do NOT include any explanation
4. Start directly with SELECT, WITH, INSERT, UPDATE, or DELETE
5. End with a semicolon

CORRECT example:
SELECT name FROM users WHERE age > 18;

Now generate the SQL query.""",

    "sft": """I want you to act as a SQL terminal in front of an example database,
you need only to return the sql command to me.
Below is an instruction that describes a task, Write a response that appropriately completes the request.""",

    "dpo": """Given the database schema and question, generate the correct SQL query.

Reminder:
- Output only raw SQL, no markdown or explanation
- Start with SELECT/WITH/INSERT/UPDATE/DELETE
- End with semicolon
- Use table_name.column_name for disambiguation when needed""",
}

SQL_KEYWORDS = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")


TASK_INSTRUCTION = "Convert the following question to a SQL query based on the database schema."


def build_prompt(schema_str: str, question: str, stage: str) -> str:
    if stage == "sft":
        return f"""{TASK_INSTRUCTION}

###Input:
{schema_str}

Question: {question}

###Response:"""

    base = f"""Database Schema:
{schema_str}

Question: {question}

"""
    if stage == "dpo":
        return base + "SQL:"
    else:
        return base + "SQL Query:"


def load_schema(spider_dir: Path, db_id: str, stage: str = "zeroshot") -> str:
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
    primary_keys = db_schema.get("primary_keys", [])
    foreign_keys = db_schema.get("foreign_keys", [])

    if stage == "sft":
        # Natural language format matching training data (DB-GPT-Hub style)
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
        for pk in primary_keys:
            if isinstance(pk, int) and pk < len(column_names):
                col = column_names[pk]
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
                        fk_parts.append(
                            f"The {c1[1]} of {table_names[t1]} is the foreign key of {c2[1]} of {table_names[t2]}"
                        )
        if fk_parts:
            schema_str += "\n" + ". ".join(fk_parts) + "."

        return schema_str

    # Structured format with column types (zeroshot / dpo)
    column_types = db_schema["column_types"]
    primary_keys_set = set(primary_keys)

    schema_parts = []
    for table_idx, table_name in enumerate(table_names):
        cols = []
        for i in range(len(column_names)):
            col = column_names[i]
            if col[0] == table_idx:
                col_name = col[1]
                col_type = column_types[i] if i < len(column_types) else "TEXT"
                pk_mark = " PRIMARY KEY" if i in primary_keys_set else ""
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


def batch_generate_sql(llm: LLM, tokenizer, prompts: List[str], max_new_tokens: int, temperature: float, stage: str) -> List[str]:
    system_prompt = SYSTEM_PROMPTS[stage]
    formatted = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True
        )
        for tok in ['<|think|>', '</think|>', '<|reserved_2066|>']:
            text = text.replace(tok, '')
        formatted.append(text)
    
    outputs = llm.generate(formatted, SamplingParams(temperature=temperature, max_tokens=max_new_tokens, seed=42), use_tqdm=False)
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


def init_scores():
    return {
        "easy": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "medium": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "hard": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "extra": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "all": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
    }


def print_scores(scores, etype):
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    
    print("\n" + "=" * 80)
    print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    
    counts = [scores[level]['count'] for level in levels]
    print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))
    
    if etype in ["all", "exec"]:
        print('=' * 80)
        print(' ' * 20 + 'EXECUTION ACCURACY')
        print('=' * 80)
        this_scores = [scores[level]['exec'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("execution", *this_scores))
    
    if etype in ["all", "match"]:
        print('\n' + '=' * 80)
        print(' ' * 18 + 'EXACT MATCHING ACCURACY')
        print('=' * 80)
        exact_scores = [scores[level]['exact'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exact match", *exact_scores))
        
        print('\n' + '-' * 80)
        print('-' * 18 + 'PARTIAL MATCHING ACCURACY')
        print('-' * 80)
        for type_ in partial_types:
            this_scores = [scores[level]['partial'].get(type_, {}).get('acc', 0) for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))
        
        print('\n' + '-' * 80)
        print('-' * 18 + 'PARTIAL MATCHING RECALL')
        print('-' * 80)
        for type_ in partial_types:
            this_scores = [scores[level]['partial'].get(type_, {}).get('rec', 0) for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))
        
        print('\n' + '-' * 80)
        print('-' * 18 + 'PARTIAL MATCHING F1')
        print('-' * 80)
        for type_ in partial_types:
            this_scores = [scores[level]['partial'].get(type_, {}).get('f1', 0) for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))


def evaluate(model_path: str, spider_dir: str, split: str = "dev", stage: str = "zeroshot",
             max_samples: int = -1, temperature: float = 0.1, max_new_tokens: int = 384,
             etype: str = "all",
             self_consistency: bool = False, n_candidates: int = 5,
             output_dir: str | Path | None = None) -> Tuple[List, Dict]:
    spider_path = Path(spider_dir)
    tables_file = spider_path / "tables.json"
    
    with open(spider_path / f"{split}.json", "r") as f:
        data = json.load(f)
    
    with open(tables_file, "r") as f:
        tables_data = json.load(f)
    
    db_schemas = {db["db_id"]: db for db in tables_data}
    db_schema_objs = {}
    found_dbs = 0
    for db_id in db_schemas:
        db_path = spider_path / "database" / db_id / f"{db_id}.sqlite"
        
        if db_path.exists():
            try:
                schema = get_schema(str(db_path))
                db_schema_objs[db_id] = Schema(schema)
                found_dbs += 1
            except Exception as e:
                print(f"Warning: Failed to load schema for {db_id}: {e}")
    
    print(f"Found {found_dbs} databases out of {len(db_schemas)}")
    if found_dbs == 0:
        print("ERROR: No databases found! Check dataset/database/ directory")
    
    if max_samples > 0:
        data = data[:max_samples]
    
    llm, tokenizer = load_model(model_path)
    
    items = []
    for item in data:
        db_id = item["db_id"]
        db_path = spider_path / "database" / db_id / f"{db_id}.sqlite"
        
        if db_id not in db_schemas:
            continue
        
        has_db = db_path.exists()
        items.append({
            "db_id": db_id,
            "question": item["question"],
            "gold_sql": item["query"],
            "db_path": str(db_path) if has_db else "",
            "schema": load_schema(spider_path, db_id, stage),
            "has_db": has_db,
        })
    
    evaluator = Evaluator()
    scores = init_scores()
    results = []

    print(f"Evaluating {len(items)} samples (etype={etype})...")
    print(f"db_schema_objs loaded: {len(db_schema_objs)} databases")
    
    # 检查一个样本的解析情况
    if items:
        sample = items[0]
        sample_db_id = sample["db_id"]
        print(f"\nDebug - First sample:")
        print(f"  db_id: {sample_db_id}")
        print(f"  has_db: {sample.get('has_db', False)}")
        print(f"  in db_schema_objs: {sample_db_id in db_schema_objs}")
        print(f"  gold_sql: {sample['gold_sql']}")
        if sample_db_id in db_schema_objs:
            try:
                test_parsed = get_sql(db_schema_objs[sample_db_id], sample['gold_sql'])
                print(f"  gold_parsed success: {test_parsed is not None}")
                print(f"  hardness: {evaluator.eval_hardness(test_parsed)}")
            except Exception as e:
                print(f"  gold_parsed FAILED: {e}")
    
    prompts = [build_prompt(item["schema"], item["question"], stage) for item in items]

    # ── Self-Consistency or greedy ─────────────────────────
    if self_consistency:
        print(f"Mode: Self-Consistency (n={n_candidates}, T={temperature})")
        candidates = sample_candidates(
            llm, tokenizer, prompts,
            n=n_candidates,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens,
            system_prompt=SYSTEM_PROMPTS[stage],
            extract_sql_fn=extract_sql,
            strip_tokens=['<|think|>', '</think|>', '<|reserved_2066|>'],
        )
        # Save candidates for DPO reuse
        cand_records = []
        for item, cands in zip(items, candidates):
            cand_records.append({
                "db_id": item["db_id"],
                "question": item["question"],
                "gold_sql": item["gold_sql"],
                "candidates": cands,
            })
        tag = f"sc_n{n_candidates}"
        cand_path = Path(output_dir) / "candidates.json"
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cand_path, "w") as f:
            json.dump(cand_records, f, ensure_ascii=False, indent=2)
        print(f"Candidates saved to {cand_path}")

        db_paths = [item["db_path"] for item in items]
        pred_sqls = execute_and_vote(candidates, db_paths)
    else:
        print(f"Mode: greedy (temperature=0)")
        pred_sqls = batch_generate_sql(llm, tokenizer, prompts, max_new_tokens, temperature, stage)
    
    pbar = tqdm(total=len(items), desc="Evaluating")
    
    for idx, (item, pred_sql) in enumerate(zip(items, pred_sqls)):
        gold_sql = item["gold_sql"]
        db_path = item["db_path"]
        db_id = item["db_id"]
        has_db = item.get("has_db", False)
        
        result = {
            "index": idx,
            "db_id": db_id,
            "question": item["question"],
            "gold_sql": gold_sql,
            "pred_sql": pred_sql if pred_sql else "FAILED",
            "hardness": "unknown",
            "exec_correct": False,
            "exact_match": False,
        }
        
        pred_parsed = None
        gold_parsed = None
        
        if pred_sql and db_id in db_schema_objs:
            try:
                pred_parsed = get_sql(db_schema_objs[db_id], pred_sql)
            except Exception as e:
                pass
        
        if db_id in db_schema_objs:
            try:
                gold_parsed = get_sql(db_schema_objs[db_id], gold_sql)
            except Exception as e:
                pass
        
        hardness = "unknown"
        if gold_parsed is not None:
            hardness = evaluator.eval_hardness(gold_parsed)
            result["hardness"] = hardness
            scores[hardness]["count"] += 1
        
        exec_correct = False
        if pred_sql and gold_parsed is not None and has_db and db_path:
            try:
                _, pred_res = execute_sql(db_path, pred_sql)
                _, gold_res = execute_sql(db_path, gold_sql)
                if isinstance(pred_res, list) and isinstance(gold_res, list):
                    exec_correct = compare_results(pred_res, gold_res)
            except:
                pass
        
        result["exec_correct"] = exec_correct
        
        exact_match = False
        if pred_parsed is not None and gold_parsed is not None:
            exact_match = evaluator.eval_exact_match(pred_parsed, gold_parsed) == 1
            if exact_match:
                partial_scores = evaluator.partial_scores
                for ptype, pscore in partial_scores.items():
                    if hardness != "unknown":
                        if ptype not in scores[hardness]['partial']:
                            scores[hardness]['partial'][ptype] = {'acc': 0, 'rec': 0, 'f1': 0}
                        scores[hardness]['partial'][ptype]['acc'] += pscore['acc']
                        scores[hardness]['partial'][ptype]['rec'] += pscore['rec']
                        scores[hardness]['partial'][ptype]['f1'] += pscore['f1']
        
        result["exact_match"] = exact_match
        
        if hardness != "unknown":
            if exec_correct:
                scores[hardness]["exec"] += 1
            if exact_match:
                scores[hardness]["exact"] += 1
        
        results.append(result)
        pbar.set_postfix({
            "Exec": f"{sum(scores[h]['exec'] for h in ['easy','medium','hard','extra'])/max(1,sum(scores[h]['count'] for h in ['easy','medium','hard','extra']))*100:.1f}%",
            "Exact": f"{sum(scores[h]['exact'] for h in ['easy','medium','hard','extra'])/max(1,sum(scores[h]['count'] for h in ['easy','medium','hard','extra']))*100:.1f}%"
        })
        pbar.update(1)
    
    pbar.close()
    
    for hardness in ['easy', 'medium', 'hard', 'extra']:
        if scores[hardness]["count"] > 0:
            total_count = scores[hardness]["count"]
            scores[hardness]["exec"] /= total_count
            scores[hardness]["exact"] /= total_count
            for ptype in scores[hardness]['partial']:
                scores[hardness]['partial'][ptype]['count'] = total_count
    
    total_count = sum(scores[h]["count"] for h in ['easy', 'medium', 'hard', 'extra'])
    scores["all"]["count"] = total_count
    all_partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                        'group', 'order', 'and/or', 'IUEN', 'keywords']
    if total_count > 0:
        scores["all"]["exec"] = sum(scores[h]["exec"] * scores[h]["count"] for h in ['easy', 'medium', 'hard', 'extra']) / total_count
        scores["all"]["exact"] = sum(scores[h]["exact"] * scores[h]["count"] for h in ['easy', 'medium', 'hard', 'extra']) / total_count
        for ptype in all_partial_types:
            all_acc = sum(scores[h]['partial'].get(ptype, {}).get('acc', 0) * scores[h]['count'] for h in ['easy', 'medium', 'hard', 'extra'])
            all_rec = sum(scores[h]['partial'].get(ptype, {}).get('rec', 0) * scores[h]['count'] for h in ['easy', 'medium', 'hard', 'extra'])
            all_f1 = sum(scores[h]['partial'].get(ptype, {}).get('f1', 0) * scores[h]['count'] for h in ['easy', 'medium', 'hard', 'extra'])
            scores["all"]['partial'][ptype] = {
                'acc': all_acc / total_count,
                'rec': all_rec / total_count,
                'f1': all_f1 / total_count,
                'count': total_count
            }
    
    print_scores(scores, etype)
    
    return results, scores


def main():
    parser = argparse.ArgumentParser(description="Spider Evaluation with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--spider_dir", type=str, required=True, help="Path to Spider dataset")
    parser.add_argument("--stage", type=str, default="zeroshot", choices=["zeroshot", "sft", "dpo"])
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train"])
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to evaluate, -1 for all")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--etype", type=str, default="all", choices=["all", "exec", "match"],
                        help="Evaluation type: all, exec, match")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--self_consistency", action="store_true",
                        help="Enable Self-Consistency inference")
    parser.add_argument("--n_candidates", type=int, default=5,
                        help="Number of candidates per question (SC mode)")
    args = parser.parse_args()

    # Output dir: outputs/{stage}/ (greedy) or outputs/{stage}/sc_n{N}/ (SC)
    output_dir = Path(args.output_dir) / args.stage
    if args.self_consistency:
        output_dir = output_dir / f"sc_n{args.n_candidates}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results, scores = evaluate(
        args.model_path, args.spider_dir, args.split, args.stage,
        args.max_samples, args.temperature, args.max_new_tokens, args.etype,
        self_consistency=args.self_consistency, n_candidates=args.n_candidates,
        output_dir=output_dir,
    )
    
    output_data = {
        "etype": args.etype,
        "stage": args.stage,
        "split": args.split,
        "scores": {
            level: {
                "count": int(scores[level]["count"]),
                "exec": float(scores[level]["exec"]),
                "exact": float(scores[level]["exact"]),
            } for level in ['easy', 'medium', 'hard', 'extra', 'all']
        },
        "results": results,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
