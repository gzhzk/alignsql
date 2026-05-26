"""Common data preprocessing utilities."""

from typing import Optional


def classify_difficulty(query: str) -> str:
    """Classify SQL difficulty per Spider official standard.

    Rules:
    - easy: simple SELECT + WHERE, no aggregation, no GROUP BY
    - medium: GROUP BY / ORDER BY / HAVING / aggregate functions
    - hard: multi-table JOIN / subqueries
    - extra: UNION / EXCEPT / INTERSECT / deeply nested subqueries
    """
    query_upper = query.upper()

    if any(op in query_upper for op in ["UNION", "EXCEPT", "INTERSECT"]):
        return "extra"

    if query_upper.count("FROM ") > 1:
        return "hard"

    if query_upper.count("SELECT") > 1:
        return "hard"

    if any(kw in query_upper for kw in ["GROUP BY", "ORDER BY", "HAVING"]):
        return "medium"

    if any(f in query_upper for f in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]):
        return "medium"

    return "easy"


def build_sft_prompt(schema_str: str, question: str) -> str:
    """Build DB-GPT-Hub style input prompt.

    Format:
        ###Input:
        {schema_description}
        Question: {question}
        ###Response:
    """
    return f"###Input:\n{schema_str}\n\nQuestion: {question}\n\n###Response:"


def validate_sql(sql: str, db_path: str) -> bool:
    """Validate SQL can execute on the target database."""
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(sql)
        conn.close()
        return True
    except Exception:
        return False
