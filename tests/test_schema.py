"""Tests for schema processing modules."""

import json
import tempfile
from pathlib import Path

from alignsql.data.schema import SchemaSerializer, SchemaFilter


def _make_tables_json(tmpdir: Path, data: dict):
    path = tmpdir / "tables.json"
    with open(path, "w") as f:
        json.dump([data], f)
    return path


def test_schema_serializer_db_gpt_hub():
    tmpdir = Path(tempfile.mkdtemp())
    schema = {
        "db_id": "test_db",
        "table_names_original": ["users", "orders"],
        "column_names_original": [[0, "id"], [0, "name"], [1, "oid"], [1, "uid"]],
        "column_types": ["INT", "TEXT", "INT", "INT"],
        "primary_keys": [0, 2],
        "foreign_keys": [[3, 0]],
    }
    tables_path = _make_tables_json(tmpdir, schema)
    serializer = SchemaSerializer(tables_path, style="db_gpt_hub")
    result = serializer.serialize("test_db")
    assert result is not None
    assert "users" in result
    assert "orders" in result
    assert "primary key" in result.lower()


def test_schema_serializer_concise():
    tmpdir = Path(tempfile.mkdtemp())
    schema = {
        "db_id": "test_db",
        "table_names_original": ["users"],
        "column_names_original": [[0, "id"], [0, "name"]],
        "column_types": ["INT", "TEXT"],
        "primary_keys": [],
        "foreign_keys": [],
    }
    tables_path = _make_tables_json(tmpdir, schema)
    serializer = SchemaSerializer(tables_path, style="concise")
    result = serializer.serialize("test_db")
    assert result is not None
    assert "Database:" in result


def test_schema_filter():
    """Test schema filtering selects relevant tables."""
    schema = {
        "db_id": "test_db",
        "table_names_original": ["users", "orders", "logs"],
        "column_names_original": [
            [0, "id"], [0, "name"],
            [1, "oid"], [1, "user_id"],
            [2, "log_id"], [2, "message"],
        ],
        "column_types": ["INT", "TEXT", "INT", "INT", "INT", "TEXT"],
        "primary_keys": [0, 2, 4],
        "foreign_keys": [[3, 0]],
    }
    filterer = SchemaFilter(schema, top_k=2)
    filtered = filterer.filter("Find user orders")
    assert len(filtered["table_names_original"]) <= 2

    def test_unknown_db():
        serializer = SchemaSerializer(tmpdir / "tables.json")
        result = serializer.serialize("nonexistent")
        assert result is None
