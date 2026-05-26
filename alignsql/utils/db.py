"""Database execution utilities."""

import sqlite3
from pathlib import Path
from typing import Any, Optional


def execute_sql(sql: str, db_path: str | Path, timeout: float = 10.0) -> tuple[bool, Optional[list], Optional[str]]:
    """Execute SQL on a SQLite database.

    Returns:
        (success, result_rows_or_None, error_message_or_None)
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows, None
    except Exception as e:
        error_msg = str(e)
        # Categorize error type
        if "syntax" in error_msg.lower() or "near" in error_msg.lower():
            error_type = "syntax_error"
        elif "no such table" in error_msg.lower() or "no such column" in error_msg.lower():
            error_type = "schema_error"
        elif "timeout" in error_msg.lower():
            error_type = "timeout"
        else:
            error_type = "runtime_error"
        return False, None, error_type


def execute_many(sql_list: list[str], db_path: str | Path) -> list[dict]:
    """Execute multiple SQL statements and return results.

    Each result dict: {sql, success, result, error, exec_time}
    """
    import time

    results = []
    for sql in sql_list:
        start = time.time()
        success, rows, error = execute_sql(sql, db_path)
        elapsed = time.time() - start
        results.append({
            "sql": sql,
            "success": success,
            "result": rows if success else None,
            "error": error,
            "exec_time": elapsed,
        })
    return results
