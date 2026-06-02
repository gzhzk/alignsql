#!/usr/bin/env python3
"""Compute candidate-pool oracle EX for saved SC candidates.

This is an offline analysis script. It does not generate SQL and does not need
a GPU. For each question, it checks whether any candidate SQL in the provided
candidate files executes to the same result as the gold SQL.

Examples:
  # DPO-only candidate oracle
  python scripts/oracle_analysis.py \
    --spider_dir dataset/spider \
    --candidate_file outputs/spider/dpo/sc_n12/candidates.json

  # Multi-policy candidate oracle: SFT + DPO
  python scripts/oracle_analysis.py \
    --spider_dir dataset/spider \
    --candidate_file outputs/spider/sft/sc_n12/candidates.json \
    --candidate_file outputs/spider/dpo/sc_n12/candidates.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "hard", "extra")


def normalize_sql(sql: str) -> str:
    return " ".join((sql or "").strip().rstrip(";").lower().split())


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql or "", re.IGNORECASE))


def execute_sql(db_path: Path, sql: str, max_ops: int) -> tuple[bool, Any]:
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        conn.text_factory = str

        def timeout() -> None:
            raise sqlite3.OperationalError("QUERY_TIMEOUT")

        conn.set_progress_handler(timeout, max_ops)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return True, rows
    except Exception as exc:
        try:
            conn.close()
        except Exception:
            pass
        return False, str(exc)


def compare_results(r1: list, r2: list, order_sensitive: bool = False) -> bool:
    if len(r1) != len(r2):
        return False
    if not r1:
        return True

    def norm(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, str):
            return value.strip()
        return value

    nr1 = [tuple(norm(x) for x in row) for row in r1]
    nr2 = [tuple(norm(x) for x in row) for row in r2]
    if order_sensitive:
        return nr1 == nr2
    return sorted(nr1, key=str) == sorted(nr2, key=str)


def load_hardness(reference_results: Path | None, total: int) -> list[str]:
    if not reference_results:
        return ["unknown"] * total
    with open(reference_results, encoding="utf-8") as f:
        results = json.load(f)["results"]
    return [item.get("hardness", "unknown") for item in results]


def db_path(spider_dir: Path, db_id: str) -> Path:
    return spider_dir / "database" / db_id / f"{db_id}.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute candidate oracle EX.")
    parser.add_argument("--spider_dir", type=Path, default=Path("dataset/spider"))
    parser.add_argument("--candidate_file", type=Path, action="append", required=True)
    parser.add_argument("--reference_results", type=Path, default=Path("outputs/spider/dpo/sc_n12/results.json"),
                        help="Optional results.json used only for hardness labels.")
    parser.add_argument("--max_ops", type=int, default=200_000,
                        help="SQLite progress handler step limit for candidate SQL.")
    parser.add_argument("--gold_max_ops", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    candidate_sets = []
    for path in args.candidate_file:
        with open(path, encoding="utf-8") as f:
            candidate_sets.append(json.load(f))

    base = candidate_sets[0]
    total = len(base)
    hardness = load_hardness(args.reference_results if args.reference_results.exists() else None, total)
    counts = Counter(hardness)
    hits = Counter()
    failures = Counter()
    per_item = []

    exec_cache: dict[tuple[str, str], tuple[bool, Any]] = {}

    def cached_execute(db_id: str, sql: str, max_ops: int) -> tuple[bool, Any]:
        key = (db_id, normalize_sql(sql))
        if key not in exec_cache:
            exec_cache[key] = execute_sql(db_path(args.spider_dir, db_id), sql, max_ops)
        return exec_cache[key]

    for idx, item in enumerate(base):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["gold_sql"]
        order_sensitive = has_order_by(gold_sql)

        gold_ok, gold_result = cached_execute(db_id, gold_sql, args.gold_max_ops)
        if not gold_ok:
            failures[f"gold:{str(gold_result)[:80]}"] += 1

        candidates = []
        seen = set()
        for cand_set in candidate_sets:
            current = cand_set[idx]
            if current["db_id"] != db_id or current["question"] != question:
                raise ValueError(f"candidate files are not aligned at index {idx}")
            for sql in current["candidates"]:
                norm = normalize_sql(sql)
                if norm and norm not in seen:
                    seen.add(norm)
                    candidates.append(sql)

        oracle = False
        correct_candidates = 0
        executable_candidates = 0
        for sql in candidates:
            ok, result = cached_execute(db_id, sql, args.max_ops)
            if not ok:
                failures[str(result)[:80]] += 1
                continue
            executable_candidates += 1
            if gold_ok and compare_results(result, gold_result, order_sensitive):
                oracle = True
                correct_candidates += 1

        diff = hardness[idx]
        hits[diff] += int(oracle)
        hits["all"] += int(oracle)
        per_item.append({
            "index": idx,
            "db_id": db_id,
            "question": question,
            "hardness": diff,
            "oracle_correct": oracle,
            "num_unique_candidates": len(candidates),
            "num_executable_candidates": executable_candidates,
            "num_correct_candidates": correct_candidates,
        })

    scores = {}
    for diff in (*DIFFICULTIES, "all"):
        count = total if diff == "all" else counts[diff]
        scores[diff] = {
            "count": count,
            "oracle_exec": hits[diff] / count if count else 0.0,
        }

    output = {
        "candidate_files": [str(path) for path in args.candidate_file],
        "scores": scores,
        "failures": dict(failures),
        "results": per_item,
    }

    print("Candidate files:")
    for path in args.candidate_file:
        print(f"  - {path}")
    print(f"Oracle EX: {scores['all']['oracle_exec'] * 100:.2f}% ({hits['all']}/{total})")
    for diff in DIFFICULTIES:
        score = scores[diff]
        if score["count"]:
            print(f"  {diff}: {score['oracle_exec'] * 100:.2f}% ({score['count']})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
