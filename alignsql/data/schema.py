"""Schema serialization and filtering."""

import json
from pathlib import Path
from typing import Optional


class SchemaSerializer:
    """Serialize database schema into natural language descriptions.

    Supports multiple formats:
    - db_gpt_hub: Detailed NL description with PK/FK info (Spider default)
    - concise: Table + column names only, for long schemas
    - dail_sql: DAIL-SQL style format
    """

    STYLES = ("db_gpt_hub", "concise", "dail_sql")

    def __init__(self, tables_path: str | Path, style: str = "db_gpt_hub"):
        if style not in self.STYLES:
            raise ValueError(f"Unknown schema style: {style}. Options: {self.STYLES}")
        self.style = style
        with open(tables_path, "r", encoding="utf-8") as f:
            self.tables_data = json.load(f)

    def get_db_schema(self, db_id: str) -> Optional[dict]:
        """Look up database schema by db_id."""
        for db in self.tables_data:
            if db["db_id"] == db_id:
                return db
        return None

    def serialize(self, db_id: str) -> Optional[str]:
        """Serialize schema for a given database."""
        db_schema = self.get_db_schema(db_id)
        if db_schema is None:
            return None

        if self.style == "db_gpt_hub":
            return self._format_db_gpt_hub(db_schema)
        elif self.style == "concise":
            return self._format_concise(db_schema)
        elif self.style == "dail_sql":
            return self._format_dail_sql(db_schema)

    def _format_db_gpt_hub(self, db_schema: dict) -> str:
        """DB-GPT-Hub style: natural language with PK/FK details."""
        table_names = db_schema["table_names_original"]
        column_names = db_schema["column_names_original"]
        column_types = db_schema["column_types"]
        primary_keys = set(db_schema.get("primary_keys", []))
        foreign_keys = db_schema.get("foreign_keys", [])

        parts = [f"{db_schema['db_id']} contains tables such as {', '.join(table_names)}.\n"]

        for table_idx, table_name in enumerate(table_names):
            cols = []
            for idx, (tbl_idx, col_name) in enumerate(column_names):
                if tbl_idx == table_idx:
                    col_type = column_types[idx] if idx < len(column_types) else ""
                    cols.append(f"{col_name} ({col_type})")
            parts.append(f"Table {table_name} has columns such as {', '.join(cols)}.")

        schema_str = " ".join(parts)

        # Primary keys
        pk_parts = []
        for pk_idx in primary_keys:
            if pk_idx < len(column_names):
                tbl_idx, col_name = column_names[pk_idx]
                if tbl_idx < len(table_names):
                    pk_parts.append(f"{table_names[tbl_idx]}.{col_name} is the primary key")
        if pk_parts:
            schema_str += "\n" + ". ".join(pk_parts) + "."

        # Foreign keys
        fk_parts = []
        for src, tgt in foreign_keys:
            if src < len(column_names) and tgt < len(column_names):
                src_tbl, src_col = column_names[src]
                tgt_tbl, tgt_col = column_names[tgt]
                if src_tbl < len(table_names) and tgt_tbl < len(table_names):
                    fk_parts.append(
                        f"The {src_col} of {table_names[src_tbl]} "
                        f"is the foreign key of {tgt_col} of {table_names[tgt_tbl]}"
                    )
        if fk_parts:
            schema_str += "\n" + ". ".join(fk_parts) + "."

        return schema_str

    def _format_concise(self, db_schema: dict) -> str:
        """Concise format for long schemas (BIRD)."""
        table_names = db_schema["table_names_original"]
        column_names = db_schema["column_names_original"]

        lines = [f"Database: {db_schema['db_id']}"]
        for table_idx, table_name in enumerate(table_names):
            cols = []
            for tbl_idx, col_name in column_names:
                if tbl_idx == table_idx:
                    cols.append(col_name)
            lines.append(f"  {table_name}: {', '.join(cols)}")
        return "\n".join(lines)

    def _format_dail_sql(self, db_schema: dict) -> str:
        """DAIL-SQL style format."""
        table_names = db_schema["table_names_original"]
        column_names = db_schema["column_names_original"]
        column_types = db_schema["column_types"]

        lines = [f"Database: {db_schema['db_id']}"]
        for table_idx, table_name in enumerate(table_names):
            for idx, (tbl_idx, col_name) in enumerate(column_names):
                if tbl_idx == table_idx:
                    col_type = column_types[idx] if idx < len(column_types) else ""
                    lines.append(f"  {table_name}.{col_name} ({col_type})")
        return "\n".join(lines)


class SchemaFilter:
    """Filter database schema to only include tables relevant to a question.

    Used when schema is too long to fit in context (e.g., BIRD).
    Based on token overlap between question tokens and table/column names.
    """

    def __init__(self, schema: dict, top_k: int = 5):
        self.schema = schema
        self.top_k = top_k

    def filter(self, question: str) -> dict:
        """Select top-K tables most relevant to the question."""
        question_tokens = set(self._tokenize(question.lower()))

        scored_tables = []
        for table_idx, table_name in enumerate(self.schema["table_names_original"]):
            score = self._score_table(table_name.lower(), question_tokens)
            # Also consider columns
            for tbl_idx, col_name in self.schema["column_names_original"]:
                if tbl_idx == table_idx:
                    if col_name.lower() in question_tokens:
                        score += 1
            scored_tables.append((table_idx, table_name, score))

        scored_tables.sort(key=lambda x: x[2], reverse=True)
        selected = scored_tables[:self.top_k]
        selected_indices = {t[0] for t in selected}

        # Build filtered schema
        filtered = dict(self.schema)
        filtered["table_names_original"] = [t[1] for t in selected]
        filtered["column_names_original"] = [
            col for col in self.schema["column_names_original"]
            if col[0] in selected_indices
        ]
        filtered["column_types"] = [
            self.schema["column_types"][i]
            for i, col in enumerate(self.schema["column_names_original"])
            if col[0] in selected_indices
        ]
        # Keep only relevant foreign keys
        filtered["foreign_keys"] = [
            fk for fk in self.schema.get("foreign_keys", [])
            if self._get_table_idx(fk[0]) in selected_indices
            and self._get_table_idx(fk[1]) in selected_indices
        ]

        return filtered

    def _get_table_idx(self, col_idx: int) -> int:
        if col_idx < len(self.schema["column_names_original"]):
            return self.schema["column_names_original"][col_idx][0]
        return -1

    def _tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r"[a-zA-Z_]+", text)

    def _score_table(self, table_name: str, question_tokens: set) -> int:
        """Score table relevance based on token overlap."""
        # Direct match of table name in question
        table_tokens = set(self._tokenize(table_name))
        return len(table_tokens & question_tokens)
