"""Tests for utility modules."""

import tempfile
import sqlite3
from pathlib import Path

from alignsql.utils.db import execute_sql, execute_many


def test_execute_sql_valid():
    """Test executing a valid SQL query."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        # Create test database
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INT, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'hello')")
        conn.commit()
        conn.close()

        success, rows, error = execute_sql("SELECT name FROM test", db_path)
        assert success is True
        assert rows == [("hello",)]
        assert error is None
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_execute_sql_invalid():
    """Test executing an invalid SQL query."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INT)")
        conn.commit()
        conn.close()

        success, rows, error = execute_sql("SELECT wrong FROM test", db_path)
        assert success is False
        assert error is not None
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_execute_many():
    """Test executing multiple SQL queries."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (x INT)")
        conn.execute("INSERT INTO t VALUES (1), (2)")
        conn.commit()
        conn.close()

        results = execute_many(
            ["SELECT x FROM t WHERE x=1", "SELECT x FROM t WHERE x=999"],
            db_path,
        )
        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is True
        assert len(results[0]["result"]) == 1
        assert len(results[1]["result"]) == 0
    finally:
        Path(db_path).unlink(missing_ok=True)
