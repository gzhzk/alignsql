#!/usr/bin/env python3
"""DPO 偏好数据准备 —— 采样候选 + 执行对比 + 生成 preference pairs.

Usage:
    python scripts/prepare_dpo.py --model_path /root/autodl-tmp/models/sft/merged \\
        --spider_dir dataset --output data_processed/dpo_pairs.json

    # 利用已有的 SC 候选（跳过模型加载，只做执行对比）
    python scripts/prepare_dpo.py --candidates_file outputs/sft/sc_n8/candidates.json \\
        --spider_dir dataset --output data_processed/dpo_pairs.json

    # 快速测试
    python scripts/prepare_dpo.py --model_path <path> --max_samples 100

参数:
  --model_path      模型路径（与 --candidates_file 二选一）
  --candidates_file 复用 SC 候选（跳过采样，与 --model_path 二选一）
  --n_candidates    候选数（默认 8）
  --temperature     采样温度（默认 1.2）
  --max_samples     限制处理条数（调试用）
  --no_others       只使用 train_spider.json，排除 train_others.json
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, List, Tuple

os.environ["VLLM_USE_V1"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignsql.utils.io import save_json

# ── 常量（与 evaluate_vllm.py 保持一致）───────────────────────────

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

# ── 辅助函数（与 evaluate_vllm.py 保持一致）──────────────────────


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


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql or "", re.IGNORECASE))


def compare_results(r1: List, r2: List, order_sensitive: bool = False) -> bool:
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

    rows1 = [norm_row(r) for r in r1]
    rows2 = [norm_row(r) for r in r2]
    if not order_sensitive:
        rows1 = sorted(rows1, key=safe_key)
        rows2 = sorted(rows2, key=safe_key)
    return rows1 == rows2


def load_model(model_path: str):
    from vllm import LLM

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


# ── DPO 数据准备核心逻辑 ──────────────────────────────────────────


def load_train_data(
    spider_dir: Path, split: str = "train", include_others: bool = True
) -> list[dict]:
    """加载 Spider 训练或开发数据。

    返回 list[dict]，每个元素:
      {db_id, query, question, ...}
    """
    if split == "dev":
        path = spider_dir / "dev.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        with open(path) as f:
            data = json.load(f)
        for item in data:
            item["_source"] = "dev"
        print(f"  Loaded {len(data)} questions from dev.json")
        return data

    # split == "train"
    spider_path = spider_dir / "train_spider.json"
    if not spider_path.exists():
        raise FileNotFoundError(f"{spider_path} not found")

    with open(spider_path) as f:
        spider_data = json.load(f)
    for item in spider_data:
        item["_source"] = "spider"
    total = len(spider_data)
    print(f"  Loaded {total} questions from train_spider.json")

    if include_others:
        others_path = spider_dir / "train_others.json"
        if others_path.exists():
            with open(others_path) as f:
                others_data = json.load(f)
            for item in others_data:
                item["_source"] = "others"
            spider_data.extend(others_data)
            print(f"  Loaded {len(others_data)} questions from train_others.json")
            total += len(others_data)

    print(f"  Total training questions: {total}")
    return spider_data


def dedupe_sqls(candidates: list[str]) -> list[str]:
    return list(dict.fromkeys([sql for sql in candidates if sql and sql.strip()]))


def generate_mutation_negatives(gold_sql: str) -> list[str]:
    sql = gold_sql.strip()
    suffix = ";" if sql.endswith(";") else ""
    core = sql[:-1].strip() if suffix else sql
    mutations = []

    patterns = [
        (r"\border\s+by\s+[\s\S]+?(?=\blimit\b|$)", ""),
        (r"\blimit\s+\d+\b", ""),
        (r"\bdistinct\b", ""),
        (r"\bASC\b", "DESC"),
        (r"\bDESC\b", "ASC"),
        (r"\bcount\s*\(", "sum("),
        (r"\bmax\s*\(", "min("),
        (r"\bmin\s*\(", "max("),
        (r"\s+>=\s+", " < "),
        (r"\s+<=\s+", " > "),
        (r"\s+>\s+", " < "),
        (r"\s+<\s+", " > "),
    ]

    for pattern, repl in patterns:
        mutated = re.sub(pattern, repl, core, count=1, flags=re.IGNORECASE).strip()
        if mutated and mutated != core:
            mutations.append(mutated + suffix)

    return list(dict.fromkeys(mutations))


def create_pair(
    item,
    schema_str,
    candidates,
    gold_result,
    db_path,
    stats,
    max_pairs=4,
    mutation_negatives=0,
):
    """对一条问题构造 1～max_pairs 条 preference pairs。

    策略:
      1. gold SQL 始终作为 chosen 之一（保证每道题至少一条正样本）
      2. chosen 集合为 gold SQL + 执行正确的候选
      3. 优先用 gold SQL 配不同 wrong candidate（最高质量），再轮转 correct candidate
    """
    chosen = []
    rejected = []
    empty_count = 0
    exec_error_count = 0
    wrong_result_count = 0
    synthetic_rejected = 0
    gold_sql = item["query"]
    order_sensitive = has_order_by(item["query"])
    stats["questions_processed"] += 1

    for sql in candidates:
        if not sql or not sql.strip():
            empty_count += 1
            continue
        success, result = execute_sql(db_path, sql)
        if success and compare_results(result, gold_result, order_sensitive):
            chosen.append(sql)
        else:
            rejected.append(sql)
            if success:
                wrong_result_count += 1
            else:
                exec_error_count += 1

    stats["total_candidates"] += len(candidates)
    stats["candidates_empty"] += empty_count
    stats["candidate_exec_errors"] += exec_error_count
    stats["candidate_wrong_results"] += wrong_result_count
    stats["candidate_correct"] += len(chosen)
    stats["unique_candidates"] += len(dedupe_sqls(candidates))

    if mutation_negatives > 0:
        existing = set(dedupe_sqls(candidates))
        for sql in generate_mutation_negatives(gold_sql):
            if synthetic_rejected >= mutation_negatives or sql in existing:
                continue
            success, result = execute_sql(db_path, sql)
            if not success or not compare_results(result, gold_result, order_sensitive):
                rejected.append(sql)
                synthetic_rejected += 1
                existing.add(sql)

    # chosen = gold SQL + correct candidates（去重）
    chosen_set = [gold_sql] + [s for s in chosen if s != gold_sql]
    chosen_set = list(dict.fromkeys(chosen_set))

    rejected = list(dict.fromkeys(rejected))
    stats["unique_rejected"] += len(rejected)
    stats["synthetic_rejected"] += synthetic_rejected

    if not rejected:
        stats["drop_no_rejected"] += 1
        return []
    stats["questions_with_rejected"] += 1
    # gold_sql always in chosen_set, so drop_all_rejected is impossible now
    # (unless gold itself fails, but that's filtered earlier)

    # 构造多 pair，优先 gold SQL × rejected
    pairs = []
    for c in chosen_set:
        for r in rejected:
            if len(pairs) >= max_pairs:
                break
            pairs.append({"chosen": c, "rejected": r})
        if len(pairs) >= max_pairs:
            break

    stats["pairs_created"] += len(pairs)
    stats["questions_with_pairs"] += 1
    if len(chosen_set) * len(rejected) > max_pairs:
        stats["questions_capped_by_max_pairs"] += 1

    base_msg = [
        {"role": "system", "content": SYSTEM_PROMPTS["sft"]},
        {"role": "user", "content": build_prompt(schema_str, item["question"], "sft")},
    ]
    return [
        {
            "messages": base_msg,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": gold_sql,
        }
        for p in pairs
    ]


def prepare_dpo(
    model_path: str = "",
    spider_dir: str = "dataset",
    output_path: str = "data_processed/dpo_pairs.json",
    split: str = "train",
    n_candidates: int = 5,
    temperature: float = 1.2,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    max_samples: int = -1,
    batch_size: int = 100,
    include_others: bool = True,
    candidates_file: str | None = None,
    max_pairs_per_question: int = 4,
    candidate_rounds: int = 1,
    seed: int = 42,
    extra_temperatures: list[float] | None = None,
    mutation_negatives: int = 0,
) -> list[dict]:
    """DPO 偏好数据准备主流程。"""
    spider_path = Path(spider_dir)
    output_path = str(output_path)
    if candidate_rounds < 1:
        raise ValueError("candidate_rounds must be >= 1")
    if max_pairs_per_question < 1:
        raise ValueError("max_pairs_per_question must be >= 1")
    if mutation_negatives < 0:
        raise ValueError("mutation_negatives must be >= 0")

    stats = {
        "total_loaded": 0,
        "no_schema": 0,
        "no_db": 0,
        "gold_failed": 0,
        "pairs_created": 0,
        "drop_no_rejected": 0,
        "total_candidates": 0,
        "candidates_empty": 0,
        "candidate_correct": 0,
        "candidate_exec_errors": 0,
        "candidate_wrong_results": 0,
        "unique_candidates": 0,
        "unique_rejected": 0,
        "questions_with_rejected": 0,
        "questions_with_pairs": 0,
        "questions_capped_by_max_pairs": 0,
        "questions_processed": 0,
        "synthetic_rejected": 0,
    }

    # Step 1: 加载数据集
    print("=" * 60)
    print("Loading dataset...")
    raw_data = load_train_data(spider_path, split, include_others)

    if max_samples > 0:
        raw_data = raw_data[:max_samples]
        print(f"  Limited to {max_samples} samples")

    stats["total_loaded"] = len(raw_data)

    # Step 2: 加载 tables.json
    with open(spider_path / "tables.json") as f:
        tables_data = json.load(f)
    db_index = {db["db_id"]: db for db in tables_data}
    print(f"  Loaded {len(db_index)} database schemas from tables.json")

    # Step 3: 验证并过滤有效问题
    print("\nValidating questions...")
    valid_items = []
    for item in raw_data:
        db_id = item["db_id"]
        if db_id not in db_index:
            stats["no_schema"] += 1
            continue
        db_path = spider_path / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            stats["no_db"] += 1
            continue
        schema_str = load_schema(spider_path, db_id, "sft")
        item["_schema"] = schema_str
        item["_db_path"] = str(db_path)
        valid_items.append(item)

    print(f"  Valid questions: {len(valid_items)}")
    if stats["no_schema"] > 0:
        print(f"  Skipped (no schema): {stats['no_schema']}")
    if stats["no_db"] > 0:
        print(f"  Skipped (no db file): {stats['no_db']}")

    if not valid_items:
        print("ERROR: No valid questions to process!")
        return []

    # Step 4: 执行 gold SQL 验证
    print("\nExecuting gold SQL...")
    gold_valid = []
    for item in tqdm(valid_items, desc="Gold SQL"):
        success, result = execute_sql(item["_db_path"], item["query"])
        if not success:
            stats["gold_failed"] += 1
            continue
        item["_gold_result"] = result
        gold_valid.append(item)

    print(f"  Gold SQL valid: {len(gold_valid)}")
    if stats["gold_failed"] > 0:
        print(f"  Skipped (gold failed): {stats['gold_failed']}")

    if not gold_valid:
        print("ERROR: No questions with valid gold SQL!")
        return []

    # Step 5: 生成候选或加载预生成的候选
    if candidates_file:
        print(f"\nLoading pre-generated candidates from {candidates_file}...")
        with open(candidates_file) as f:
            cand_data = json.load(f)
        cand_lookup = {}
        for entry in cand_data:
            cand_lookup[(entry["db_id"], entry["question"])] = entry["candidates"]
        for item in gold_valid:
            key = (item["db_id"], item["question"])
            item["_candidates"] = cand_lookup.get(key, [])
        gold_valid = [item for item in gold_valid if item.get("_candidates")]
        print(f"  Matched {len(gold_valid)} items with candidates")
        if not gold_valid:
            print("ERROR: No items matched with pre-generated candidates!")
            return []
    else:
        print(f"\nLoading model for candidate generation...")
        llm, tokenizer = load_model(model_path)

    # Step 6: 分批采样并构造 preference pairs
    print(f"\nGenerating candidates and building preference pairs...")
    sampling_temperatures = [temperature] + (extra_temperatures or [])
    print(
        f"  n_candidates={n_candidates}, temperature={temperature}, top_p={top_p}, "
        f"rounds={candidate_rounds}, extra_temperatures={extra_temperatures or []}, "
        f"max_pairs_per_question={max_pairs_per_question}, "
        f"mutation_negatives={mutation_negatives}"
    )
    all_pairs = []

    for batch_start in range(0, len(gold_valid), batch_size):
        batch = gold_valid[batch_start:batch_start + batch_size]

        # 构建 prompts
        prompts = [
            build_prompt(item["_schema"], item["question"], "sft")
            for item in batch
        ]

        # 采样候选
        if candidates_file is None:
            from alignsql.models.inference import sample_candidates

            batch_candidates = [[] for _ in batch]
            pass_idx = 0
            for temp in sampling_temperatures:
                for round_idx in range(candidate_rounds):
                    sampled = sample_candidates(
                        llm, tokenizer, prompts,
                        n=n_candidates,
                        temperature=temp,
                        top_p=top_p,
                        max_tokens=max_new_tokens,
                        system_prompt=SYSTEM_PROMPTS["sft"],
                        extract_sql_fn=extract_sql,
                        strip_tokens=['<|think|>', '</think|>', '<|reserved_2066|>'],
                        seed=seed + pass_idx,
                    )
                    for i, cands in enumerate(sampled):
                        batch_candidates[i].extend(cands)
                    pass_idx += 1
        else:
            batch_candidates = [item["_candidates"] for item in batch]

        # 构造 preference pairs
        for item, cands in zip(batch, batch_candidates):
            pairs = create_pair(
                item, item["_schema"], cands,
                gold_result=item["_gold_result"],
                db_path=item["_db_path"],
                stats=stats,
                max_pairs=max_pairs_per_question,
                mutation_negatives=mutation_negatives,
            )
            all_pairs.extend(pairs)

        batch_num = batch_start // batch_size + 1
        total_batches = (len(gold_valid) + batch_size - 1) // batch_size
        if batch_num % 5 == 0 or batch_num == total_batches:
            avg_unique = stats["unique_candidates"] / stats["questions_processed"]
            print(f"  Batch {batch_num}/{total_batches} — "
                  f"{len(all_pairs)} pairs generated; "
                  f"{stats['questions_with_rejected']} questions with rejected; "
                  f"{stats['drop_no_rejected']} skipped no rejected; "
                  f"avg unique candidates/question={avg_unique:.2f}")

    # Step 7: 保存输出
    print(f"\nSaving {len(all_pairs)} pairs to {output_path}...")
    save_json(all_pairs, output_path)

    # Step 8: 打印统计
    print("\n" + "=" * 60)
    print("DPO Data Preparation Summary")
    print("=" * 60)

    pre_drop = stats["no_schema"] + stats["no_db"] + stats["gold_failed"]

    print(f"  Total source questions:            {stats['total_loaded']}")
    if pre_drop > 0:
        print(f"  Pre-filter dropped:                {pre_drop}")
        print(f"    - db_id not in tables.json:      {stats['no_schema']}")
        print(f"    - Database file missing:          {stats['no_db']}")
        print(f"    - Gold SQL execution failed:      {stats['gold_failed']}")
    print(f"  Questions with valid gold SQL:     {len(gold_valid)}")
    print(f"    Preference pairs generated:       {stats['pairs_created']}")
    print(f"    Skipped (all candidates correct): {stats['drop_no_rejected']}")
    print(f"    Questions with rejected:          {stats['questions_with_rejected']}")
    print(f"    Questions capped by max pairs:    {stats['questions_capped_by_max_pairs']}")
    print(f"    Synthetic rejected added:         {stats['synthetic_rejected']}")
    total_cands = stats["total_candidates"]
    empty_cands = stats["candidates_empty"]
    if total_cands > 0:
        print(f"\n  Candidates processed: {total_cands}"
              f" ({empty_cands} empty, {empty_cands/total_cands*100:.1f}%)")
        print(f"    Correct by execution:             {stats['candidate_correct']}"
              f" ({stats['candidate_correct']/total_cands*100:.1f}%)")
        print(f"    Wrong result:                     {stats['candidate_wrong_results']}"
              f" ({stats['candidate_wrong_results']/total_cands*100:.1f}%)")
        print(f"    Execution errors:                 {stats['candidate_exec_errors']}"
              f" ({stats['candidate_exec_errors']/total_cands*100:.1f}%)")
    if stats["questions_processed"] > 0:
        print(f"    Avg unique candidates/question:   "
              f"{stats['unique_candidates']/stats['questions_processed']:.2f}")
        print(f"    Avg unique rejected/question:     "
              f"{stats['unique_rejected']/stats['questions_processed']:.2f}")
    print(f"\n  Output: {output_path}")
    print()

    return all_pairs


# ── CLI ──────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs from Spider dataset"
    )
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to model (vLLM)")
    parser.add_argument("--spider_dir", type=str, default="dataset",
                        help="Path to Spider dataset directory")
    parser.add_argument("--output", type=str, default="data_processed/dpo_pairs.json",
                        help="Output path for DPO pairs")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "dev"],
                        help="Data split")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max questions to process (-1 = all)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Questions per generation batch")
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="Candidates per question")
    parser.add_argument("--temperature", type=float, default=1.2,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p")
    parser.add_argument("--candidate_rounds", type=int, default=1,
                        help="Sampling rounds per temperature")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base sampling seed")
    parser.add_argument("--extra_temperatures", type=str, default="",
                        help="Comma-separated extra temperatures, e.g. 0.8,1.6")
    parser.add_argument("--max_pairs_per_question", type=int, default=4,
                        help="Max preference pairs to keep per question")
    parser.add_argument("--mutation_negatives", type=int, default=0,
                        help="Verified gold-SQL mutations to add as rejected candidates per question")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens per candidate")
    parser.add_argument("--no_others", action="store_true",
                        help="Exclude train_others.json")
    parser.add_argument("--candidates_file", type=str, default=None,
                        help="Pre-generated candidates.json (skips model loading)")
    return parser.parse_args()


def parse_extra_temperatures(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main():
    args = parse_args()
    if not args.candidates_file and not args.model_path:
        print("ERROR: Either --model_path or --candidates_file is required")
        sys.exit(1)

    prepare_dpo(
        model_path=args.model_path,
        spider_dir=args.spider_dir,
        output_path=args.output,
        split=args.split,
        n_candidates=args.n_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        include_others=not args.no_others,
        candidates_file=args.candidates_file,
        max_pairs_per_question=args.max_pairs_per_question,
        candidate_rounds=args.candidate_rounds,
        seed=args.seed,
        extra_temperatures=parse_extra_temperatures(args.extra_temperatures),
        mutation_negatives=args.mutation_negatives,
    )


if __name__ == "__main__":
    main()
