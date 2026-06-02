#!/usr/bin/env python3
"""DPO 偏好数据准备 —— 采样候选 + 执行对比 + 生成 preference pairs.

Usage:
    python scripts/prepare_dpo.py --model_path /root/autodl-tmp/models/sft/merged \\
        --spider_dir dataset/spider --output data_processed/dpo_pairs_local.json

    # 利用已有的 SC 候选（跳过模型加载，只做执行对比）
    python scripts/prepare_dpo.py --candidates_file outputs/spider/sft/sc_n8/candidates.json \\
        --spider_dir dataset/spider --output data_processed/dpo_pairs_local.json

    # 快速测试
    python scripts/prepare_dpo.py --model_path <path> --max_samples 100

参数:
  --model_path      模型路径（与 --candidates_file 二选一）
  --candidates_file 复用 SC 候选（跳过采样，与 --model_path 二选一）
  --n_candidates    候选数（默认 8）
  --temperature     采样温度（默认 1.2）
  --max_samples     限制处理条数（调试用）
  --include_others  额外使用 train_others.json（默认只使用 SFT 训练过的 train_spider.json）
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, List, Tuple

os.environ["VLLM_USE_V1"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from alignsql.data.preprocessing import classify_difficulty
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

    "dpo": """I want you to act as a SQL terminal in front of an example database,
you need only to return the sql command to me.
Below is an instruction that describes a task, Write a response that appropriately completes the request.""",
}

SQL_KEYWORDS = ("SELECT", "WITH", "INSERT", "UPDATE", "DELETE")
TASK_INSTRUCTION = "Convert the following question to a SQL query based on the database schema."
NOISE_MARKERS = ("###REQUEST", "### REQUEST", "###INSTRUCTION", "### INSTRUCTION", "```")

# ── 辅助函数（与 evaluate_vllm.py 保持一致）──────────────────────


def prompt_format(stage: str) -> str:
    return "zeroshot" if stage == "zeroshot" else "sft"


def build_prompt(schema_str: str, question: str, stage: str) -> str:
    if prompt_format(stage) == "sft":
        return f"""{TASK_INSTRUCTION}

###Input:
{schema_str}

Question: {question}

###Response:"""

    base = f"""Database Schema:
{schema_str}

Question: {question}

"""
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

    if prompt_format(stage) == "sft":
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


def sql_text_quality(sql: str) -> tuple[bool, str]:
    """Reject sampled strings that look like web-corpus or prompt-injection noise."""
    text = (sql or "").strip()
    if not text:
        return False, "empty"
    upper = text.upper()
    if not any(upper.startswith(keyword) for keyword in SQL_KEYWORDS):
        return False, "not_sql_start"
    if any(marker in upper for marker in NOISE_MARKERS):
        return False, "noise_marker"

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii > 0 and non_ascii / max(len(text), 1) > 0.03:
        return False, "non_ascii_ratio"

    suspicious = sum(
        1 for ch in text
        if ord(ch) > 127 and not ch.isspace()
    )
    if suspicious >= 4:
        return False, "non_ascii_count"

    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\t\n\r")
    if control:
        return False, "control_char"

    return True, "ok"


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
    spider_dir: Path, split: str = "train", include_others: bool = False
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


LOCAL_ERROR_PRIORITY = {
    "TABLE_SELECTION": 0,
    "GROUP_BY": 1,
    "WHERE": 2,
    "DISTINCT": 3,
    "AGGREGATION": 4,
    "SET_OP": 5,
    "SUBQUERY": 6,
    "ORDER_BY_LIMIT": 7,
    "SYNTAX_ERROR": 8,
    "SCHEMA_ERROR": 9,
    "SELECT_COLUMNS": 10,
    "VALUE_OR_COLUMN": 11,
    "EXECUTION_ERROR": 12,
    "EMPTY": 99,
    "SURFACE_ONLY": 100,
}

DEFAULT_ERROR_MIX = {
    "TABLE_SELECTION": 0.30,
    "SYNTAX_ERROR": 0.15,
    "GROUP_BY": 0.15,
    "WHERE": 0.15,
    "DISTINCT": 0.075,
    "AGGREGATION": 0.075,
    "SET_OP": 0.025,
    "SUBQUERY": 0.025,
    "ORDER_BY_LIMIT": 0.05,
}


def normalize_sql_text(sql: str) -> str:
    return " ".join((sql or "").lower().strip().rstrip(";").split())


def token_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", a.lower()))
    b_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def sql_feature_flags(sql: str) -> dict[str, Any]:
    text = sql or ""
    lower = text.lower()
    agg_funcs = set(re.findall(r"\b(count|sum|avg|min|max)\s*\(", lower))
    return {
        "tables": extract_table_names(text),
        "where": bool(re.search(r"\bwhere\b", lower)),
        "group_by": bool(re.search(r"\bgroup\s+by\b", lower)),
        "having": bool(re.search(r"\bhaving\b", lower)),
        "order_by": bool(re.search(r"\border\s+by\b", lower)),
        "limit": bool(re.search(r"\blimit\b", lower)),
        "distinct": bool(re.search(r"\bdistinct\b", lower)),
        "set_op": bool(re.search(r"\b(union|intersect|except)\b", lower)),
        "subquery_count": max(0, len(re.findall(r"\bselect\b", lower)) - 1),
        "agg_funcs": agg_funcs,
        "select_count": estimate_select_count(text),
        "where_clause": extract_clause(lower, "where"),
        "group_clause": extract_clause(lower, "group by"),
    }


def extract_table_names(sql: str) -> set[str]:
    """Best-effort table extraction for local hard-negative bucketing."""
    tables = set()
    text = re.sub(r"\([^()]*\)", " ", sql or "")
    for match in re.finditer(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE):
        tables.add(match.group(1).lower())

    from_match = re.search(
        r"\bfrom\b\s+(.+?)(?=\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|\bunion\b|\bintersect\b|\bexcept\b|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if from_match:
        for part in from_match.group(1).split(","):
            token = part.strip().split()
            if token and token[0].lower() not in {"select", "join"}:
                tables.add(token[0].strip('"`[]').lower())
    return tables


def estimate_select_count(sql: str) -> int:
    match = re.search(r"\bselect\b\s+(.+?)\s+\bfrom\b", sql or "", re.IGNORECASE | re.DOTALL)
    if not match:
        return 0
    expr = match.group(1)
    depth = 0
    count = 1 if expr.strip() else 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            count += 1
    return count


def extract_clause(sql_lower: str, clause: str) -> str:
    clause_pattern = re.escape(clause).replace(r"\ ", r"\s+")
    start = re.search(rf"\b{clause_pattern}\b", sql_lower)
    if not start:
        return ""
    rest = sql_lower[start.end():]
    end = re.search(
        r"\b(group\s+by|having|order\s+by|limit|union|intersect|except)\b",
        rest,
        re.IGNORECASE,
    )
    return " ".join((rest[:end.start()] if end else rest).split())


def classify_local_error(gold_sql: str, candidate_sql: str, exec_success: bool, exec_result: Any) -> tuple[str, list[str]]:
    """Classify a rejected candidate using only train-side SQL and execution feedback."""
    if not candidate_sql or not candidate_sql.strip():
        return "EMPTY", ["empty candidate"]

    if normalize_sql_text(gold_sql) == normalize_sql_text(candidate_sql):
        return "SURFACE_ONLY", ["candidate text normalizes to gold"]

    if not exec_success:
        msg = str(exec_result).lower()
        if "no such table" in msg or "no such column" in msg:
            return "SCHEMA_ERROR", [str(exec_result)]
        if "syntax" in msg or "near" in msg or "incomplete input" in msg:
            return "SYNTAX_ERROR", [str(exec_result)]
        return "EXECUTION_ERROR", [str(exec_result)]

    gold = sql_feature_flags(gold_sql)
    cand = sql_feature_flags(candidate_sql)
    details = []

    if gold["tables"] != cand["tables"]:
        details.append(f"tables gold={sorted(gold['tables'])}, rejected={sorted(cand['tables'])}")
        return "TABLE_SELECTION", details
    if gold["group_by"] != cand["group_by"] or gold["group_clause"] != cand["group_clause"]:
        details.append("GROUP BY differs")
        return "GROUP_BY", details
    if gold["where"] != cand["where"] or gold["where_clause"] != cand["where_clause"]:
        details.append("WHERE differs")
        return "WHERE", details
    if gold["distinct"] != cand["distinct"]:
        details.append("DISTINCT differs")
        return "DISTINCT", details
    if gold["agg_funcs"] != cand["agg_funcs"]:
        details.append(f"aggregations gold={sorted(gold['agg_funcs'])}, rejected={sorted(cand['agg_funcs'])}")
        return "AGGREGATION", details
    if gold["set_op"] != cand["set_op"]:
        details.append("set operation differs")
        return "SET_OP", details
    if gold["subquery_count"] != cand["subquery_count"]:
        details.append(f"subqueries gold={gold['subquery_count']}, rejected={cand['subquery_count']}")
        return "SUBQUERY", details
    if gold["order_by"] != cand["order_by"] or gold["limit"] != cand["limit"]:
        details.append("ORDER BY/LIMIT differs")
        return "ORDER_BY_LIMIT", details
    if gold["select_count"] != cand["select_count"]:
        details.append(f"select columns gold={gold['select_count']}, rejected={cand['select_count']}")
        return "SELECT_COLUMNS", details
    return "VALUE_OR_COLUMN", ["same coarse structure, wrong execution result"]


def is_high_quality_correct_candidate(gold_sql: str, candidate_sql: str, result: Any) -> tuple[bool, str]:
    """Gate execution-correct sampled SQL before allowing it as a DPO chosen."""
    if not candidate_sql or not candidate_sql.strip():
        return False, "empty"
    if normalize_sql_text(gold_sql) == normalize_sql_text(candidate_sql):
        return False, "same_as_gold"
    if isinstance(result, list) and len(result) == 0:
        return False, "empty_result"

    similarity = token_similarity(gold_sql, candidate_sql)
    if similarity < 0.45:
        return False, "low_similarity"

    gold = sql_feature_flags(gold_sql)
    cand = sql_feature_flags(candidate_sql)
    # Correct candidates should preserve the broad query intent. This still
    # allows equivalent rewrites such as different join order or aliases.
    for key in ("set_op", "distinct"):
        if gold[key] != cand[key]:
            return False, f"{key}_differs"
    if abs(gold["subquery_count"] - cand["subquery_count"]) > 1:
        return False, "subquery_count_differs"
    if abs(gold["select_count"] - cand["select_count"]) > 1:
        return False, "select_count_differs"
    return True, "ok"


def parse_error_mix(value: str) -> dict[str, float]:
    if not value.strip():
        return DEFAULT_ERROR_MIX.copy()
    mix = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, weight = part.split(":", 1)
        mix[name.strip()] = float(weight)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("error mix weights must sum to > 0")
    return {k: v / total for k, v in mix.items()}


def balance_pairs_by_error_type(
    pairs: list[dict],
    error_mix: dict[str, float],
    target_pairs: int,
    seed: int,
) -> list[dict]:
    """Down/up-select pairs to keep the final DPO data focused on useful local errors."""
    if not pairs or target_pairs == 0:
        return pairs

    target_pairs = min(target_pairs, len(pairs)) if target_pairs > 0 else len(pairs)
    rng = random.Random(seed)
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair.get("error_type", "UNKNOWN")].append(pair)

    for group in groups.values():
        group.sort(
            key=lambda x: (
                LOCAL_ERROR_PRIORITY.get(x.get("error_type", "UNKNOWN"), 50),
                -x.get("similarity", 0.0),
                x.get("db_id", ""),
                x.get("question", ""),
            )
        )

    selected = []
    selected_ids = set()
    normalized_mix = parse_error_mix("") if not error_mix else error_mix
    for error_type, weight in normalized_mix.items():
        quota = int(round(target_pairs * weight))
        for pair in groups.get(error_type, [])[:quota]:
            selected.append(pair)
            selected_ids.add(id(pair))

    remaining = [
        pair for pair in pairs
        if id(pair) not in selected_ids
        and pair.get("error_type") not in {"EMPTY", "SURFACE_ONLY"}
    ]
    remaining.sort(
        key=lambda x: (
            LOCAL_ERROR_PRIORITY.get(x.get("error_type", "UNKNOWN"), 50),
            -x.get("similarity", 0.0),
        )
    )
    for pair in remaining:
        if len(selected) >= target_pairs:
            break
        selected.append(pair)
        selected_ids.add(id(pair))

    rng.shuffle(selected)
    return selected


def create_pair(
    item,
    schema_str,
    candidates,
    gold_result,
    db_path,
    stats,
    max_pairs=4,
    mutation_negatives=0,
    pair_strategy="local",
):
    """对一条问题构造 1～max_pairs 条 preference pairs。

    local 策略:
      1. chosen 固定为 gold SQL，避免把 execution-equivalent 但非 gold 的候选学成偏好。
      2. rejected 必须是训练集采样得到的真实错误候选，或训练集 gold 变异负样本。
      3. 对 rejected 打本地错误类型标签，并优先选择局部 hard negatives。
    """
    chosen = []
    correct_items = []
    rejected = []
    rejected_items = []
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
        quality_ok, quality_reason = sql_text_quality(sql)
        if not quality_ok:
            stats["candidate_text_filtered"] += 1
            stats["candidate_text_filter_reasons"][quality_reason] += 1
            continue
        success, result = execute_sql(db_path, sql)
        if success and compare_results(result, gold_result, order_sensitive):
            chosen.append(sql)
            ok, reason = is_high_quality_correct_candidate(gold_sql, sql, result)
            if ok:
                correct_items.append({
                    "sql": sql,
                    "similarity": token_similarity(gold_sql, sql),
                    "source": "sampled_correct",
                })
            else:
                stats["correct_candidate_reject_reasons"][reason] += 1
        else:
            rejected.append(sql)
            error_type, details = classify_local_error(gold_sql, sql, success, result)
            rejected_items.append({
                "sql": sql,
                "error_type": error_type,
                "error_details": details,
                "exec_success": success,
                "similarity": token_similarity(gold_sql, sql),
                "source": "sampled",
            })
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
            quality_ok, quality_reason = sql_text_quality(sql)
            if not quality_ok:
                stats["candidate_text_filtered"] += 1
                stats["candidate_text_filter_reasons"][quality_reason] += 1
                continue
            success, result = execute_sql(db_path, sql)
            if not success or not compare_results(result, gold_result, order_sensitive):
                rejected.append(sql)
                error_type, details = classify_local_error(gold_sql, sql, success, result)
                rejected_items.append({
                    "sql": sql,
                    "error_type": error_type,
                    "error_details": details,
                    "exec_success": success,
                    "similarity": token_similarity(gold_sql, sql),
                    "source": "mutation",
                })
                synthetic_rejected += 1
                existing.add(sql)

    rejected = list(dict.fromkeys(rejected))
    stats["unique_rejected"] += len(rejected)
    stats["synthetic_rejected"] += synthetic_rejected

    if not rejected:
        stats["drop_no_rejected"] += 1
        return []
    stats["questions_with_rejected"] += 1
    rejected_items = list({item["sql"]: item for item in rejected_items}.values())
    if pair_strategy == "local":
        rejected_items = [
            item for item in rejected_items
            if item["error_type"] not in {"EMPTY", "SURFACE_ONLY"}
        ]
        if not rejected_items:
            stats["drop_no_local_rejected"] += 1
            return []
        rejected_items.sort(
            key=lambda x: (
                LOCAL_ERROR_PRIORITY.get(x["error_type"], 50),
                -x["similarity"],
                x["sql"],
            )
        )
    else:
        rejected_items = [
            {
                "sql": sql,
                "error_type": "LEGACY",
                "error_details": [],
                "exec_success": None,
                "similarity": token_similarity(gold_sql, sql),
                "source": "legacy",
            }
            for sql in rejected
        ]

    pairs = [
        {"chosen": gold_sql, "chosen_source": "gold", "rejected": rejected_items[0]["sql"], **rejected_items[0]}
    ]
    correct_items = list({item["sql"]: item for item in correct_items}.values())
    correct_items.sort(key=lambda x: (-x["similarity"], x["sql"]))
    if max_pairs > 1 and correct_items and len(rejected_items) > 0:
        # Pair the best execution-correct sampled SQL with the strongest
        # rejected candidate not already used when possible.
        rejected_for_correct = rejected_items[1] if len(rejected_items) > 1 else rejected_items[0]
        pairs.append({
            "chosen": correct_items[0]["sql"],
            "chosen_source": "sampled_correct",
            "rejected": rejected_for_correct["sql"],
            **rejected_for_correct,
        })

    stats["pairs_created"] += len(pairs)
    stats["questions_with_pairs"] += 1
    if len(rejected_items) > max_pairs:
        stats["questions_capped_by_max_pairs"] += 1
    for pair in pairs:
        stats["error_type_counts"][pair["error_type"]] += 1
        stats["chosen_source_counts"][pair["chosen_source"]] += 1

    base_msg = [
        {"role": "system", "content": SYSTEM_PROMPTS["sft"]},
        {"role": "user", "content": build_prompt(schema_str, item["question"], "sft")},
    ]
    return [
        {
            "messages": base_msg,
            "chosen": p["chosen"],
            "chosen_source": p["chosen_source"],
            "rejected": p["rejected"],
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": gold_sql,
            "difficulty": item.get("difficulty") or classify_difficulty(gold_sql),
            "error_type": p["error_type"],
            "error_details": p["error_details"],
            "candidate_exec_success": p["exec_success"],
            "similarity": round(p["similarity"], 4),
            "negative_source": p["source"],
        }
        for p in pairs
    ]


def prepare_dpo(
    model_path: str = "",
    spider_dir: str = "dataset/spider",
    output_path: str = "data_processed/dpo_pairs.json",
    split: str = "train",
    n_candidates: int = 5,
    temperature: float = 1.2,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    max_samples: int = -1,
    batch_size: int = 100,
    include_others: bool = False,
    candidates_file: str | None = None,
    max_pairs_per_question: int = 4,
    candidate_rounds: int = 1,
    seed: int = 42,
    extra_temperatures: list[float] | None = None,
    mutation_negatives: int = 0,
    pair_strategy: str = "local",
    balance_error_types: bool = True,
    error_mix: dict[str, float] | None = None,
    target_pairs: int = 0,
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
        "drop_no_local_rejected": 0,
        "error_type_counts": Counter(),
        "chosen_source_counts": Counter(),
        "correct_candidate_reject_reasons": Counter(),
        "candidate_text_filtered": 0,
        "candidate_text_filter_reasons": Counter(),
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
        f"mutation_negatives={mutation_negatives}, "
        f"pair_strategy={pair_strategy}, balance_error_types={balance_error_types}"
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
                pair_strategy=pair_strategy,
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

    # Step 7: local hard-negative balance
    if pair_strategy == "local" and balance_error_types:
        before_balance = len(all_pairs)
        all_pairs = balance_pairs_by_error_type(
            all_pairs,
            error_mix or DEFAULT_ERROR_MIX,
            target_pairs=target_pairs or before_balance,
            seed=seed,
        )
        print(f"\nBalanced local pairs: {before_balance} -> {len(all_pairs)}")
        print("  Final error type distribution:")
        for error_type, count in Counter(p["error_type"] for p in all_pairs).most_common():
            print(f"    {error_type:<20} {count:>6}")
        print("  Final chosen source distribution:")
        for source, count in Counter(p.get("chosen_source", "unknown") for p in all_pairs).most_common():
            print(f"    {source:<20} {count:>6}")

    # Step 8: 保存输出
    print(f"\nSaving {len(all_pairs)} pairs to {output_path}...")
    save_json(all_pairs, output_path)

    # Step 9: 打印统计
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
    if pair_strategy == "local":
        print(f"    Skipped (no local hard negative): {stats['drop_no_local_rejected']}")
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
    if stats["candidate_text_filtered"]:
        print(f"\n  Candidate text filtered:            {stats['candidate_text_filtered']}")
        for reason, count in stats["candidate_text_filter_reasons"].most_common():
            print(f"    {reason:<20} {count:>6}")
    if stats["error_type_counts"]:
        print("\n  Local rejected error types before balance:")
        for error_type, count in stats["error_type_counts"].most_common():
            print(f"    {error_type:<20} {count:>6}")
    if stats["chosen_source_counts"]:
        print("\n  Chosen sources before balance:")
        for source, count in stats["chosen_source_counts"].most_common():
            print(f"    {source:<20} {count:>6}")
    if stats["correct_candidate_reject_reasons"]:
        print("\n  Execution-correct candidates not used as chosen:")
        for reason, count in stats["correct_candidate_reject_reasons"].most_common():
            print(f"    {reason:<20} {count:>6}")
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
    parser.add_argument("--spider_dir", type=str, default="dataset/spider",
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
    parser.add_argument("--max_pairs_per_question", type=int, default=2,
                        help="Max preference pairs to keep per question")
    parser.add_argument("--mutation_negatives", type=int, default=0,
                        help="Verified gold-SQL mutations to add as rejected candidates per question")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens per candidate")
    parser.add_argument("--include_others", action="store_true",
                        help="Include train_others.json in addition to train_spider.json")
    parser.add_argument("--no_others", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidates_file", type=str, default=None,
                        help="Pre-generated candidates.json (skips model loading)")
    parser.add_argument("--pair_strategy", type=str, default="local",
                        choices=["local", "legacy"],
                        help="Pair construction strategy: local hard negatives or legacy gold-vs-any-rejected")
    parser.add_argument("--no_balance_error_types", action="store_true",
                        help="Disable final error-type balancing for local strategy")
    parser.add_argument("--error_mix", type=str, default="",
                        help="Comma-separated target mix, e.g. TABLE_SELECTION:0.3,SYNTAX_ERROR:0.15")
    parser.add_argument("--target_pairs", type=int, default=0,
                        help="Target number of pairs after balancing (0 = keep as many as possible)")
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
        include_others=args.include_others and not args.no_others,
        candidates_file=args.candidates_file,
        max_pairs_per_question=args.max_pairs_per_question,
        candidate_rounds=args.candidate_rounds,
        seed=args.seed,
        extra_temperatures=parse_extra_temperatures(args.extra_temperatures),
        mutation_negatives=args.mutation_negatives,
        pair_strategy=args.pair_strategy,
        balance_error_types=not args.no_balance_error_types,
        error_mix=parse_error_mix(args.error_mix),
        target_pairs=args.target_pairs,
    )


if __name__ == "__main__":
    main()
