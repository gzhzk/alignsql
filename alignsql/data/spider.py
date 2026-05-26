"""Spider dataset loader and preprocessor."""

import json
from pathlib import Path
from typing import Optional

from .schema import SchemaSerializer
from .preprocessing import classify_difficulty, build_sft_prompt

SYSTEM_PROMPT = (
    "I want you to act as a SQL terminal in front of an example database, "
    "you need only to return the sql command to me. "
    "Below is an instruction that describes a task, Write a response that appropriately completes the request."
)

TASK_INSTRUCTION = "Convert the following question to a SQL query based on the database schema."


class SpiderLoader:
    """Load Spider dataset and convert to training format."""

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.tables_path = self.dataset_dir / "tables.json"

        if not self.tables_path.exists():
            raise FileNotFoundError(f"tables.json not found at {self.tables_path}")

        self.schema_serializer = SchemaSerializer(self.tables_path, style="db_gpt_hub")

    def load_sft_data(
        self,
        split: str = "train",
        input_file: Optional[str] = None,
        max_samples: int = -1,
    ) -> list[dict]:
        """Load and convert Spider data to Alpaca format for SFT training.

        Args:
            split: "train" or "dev" or "test"
            input_file: explicit file path (optional, auto-resolves if None)
            max_samples: limit samples (-1 for all)

        Returns:
            list of dicts in Alpaca format: {db_id, difficulty, system, instruction, input, output}
        """
        if input_file:
            input_path = Path(input_file)
        else:
            # Auto-resolve based on split
            input_path = self._resolve_default_path(split)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix == ".parquet":
            return self._load_parquet(input_path, max_samples)
        else:
            return self._load_json(input_path, max_samples)

    def load_dev_data(self) -> list[dict]:
        """Load dev set for evaluation (Spider dev.json format)."""
        return self.load_sft_data(split="dev")

    def _resolve_default_path(self, split: str) -> Path:
        """Resolve default input path for a given split."""
        parquet = self.dataset_dir / f"{split}-00000-of-00001.parquet"
        if parquet.exists():
            return parquet
        # JSON fallbacks
        json_map = {
            "train": "train_spider.json",
            "dev": "dev.json",
            "test": "test.json",
        }
        return self.dataset_dir / json_map.get(split, f"{split}.json")

    def _load_parquet(self, path: Path, max_samples: int) -> list[dict]:
        """Load data from parquet format."""
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        total = table.num_rows if max_samples < 0 else min(max_samples, table.num_rows)

        results = []
        for i in range(total):
            db_id = table.column("db_id")[i].as_py()
            query = table.column("query")[i].as_py()
            question = table.column("question")[i].as_py()

            schema_str = self.schema_serializer.serialize(db_id)
            if schema_str is None:
                continue

            item = {
                "db_id": db_id,
                "difficulty": classify_difficulty(query),
                "system": SYSTEM_PROMPT,
                "instruction": TASK_INSTRUCTION,
                "input": build_sft_prompt(schema_str, question),
                "output": query,
                "question": question,
            }
            results.append(item)

        return results

    def _load_json(self, path: Path, max_samples: int) -> list[dict]:
        """Load data from JSON format (Spider default)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = len(data) if max_samples < 0 else min(max_samples, len(data))

        results = []
        for item in data[:total]:
            db_id = item["db_id"]
            query = item["query"]
            question = item["question"]

            schema_str = self.schema_serializer.serialize(db_id)
            if schema_str is None:
                continue

            result = {
                "db_id": db_id,
                "difficulty": classify_difficulty(query),
                "system": SYSTEM_PROMPT,
                "instruction": TASK_INSTRUCTION,
                "input": build_sft_prompt(schema_str, question),
                "output": query,
                "question": question,
            }
            results.append(result)

        return results

    @staticmethod
    def analyze_difficulty_distribution(data: list[dict]) -> dict:
        """Analyze difficulty distribution of a dataset."""
        counts = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}
        for item in data:
            diff = item.get("difficulty", "easy")
            counts[diff] = counts.get(diff, 0) + 1

        total = len(data)
        distribution = {}
        for diff, count in counts.items():
            distribution[diff] = {"count": count, "pct": round(100 * count / total, 1) if total else 0}
        distribution["total"] = total
        return distribution
