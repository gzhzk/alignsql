#!/usr/bin/env python3
"""Build candidate-level diagnostics for verifier/reranker analysis.

The script consumes saved self-consistency candidates from multiple policies
and produces:

* candidate_labels.jsonl: one row per unique SQL candidate per question.
* item_diagnostics.jsonl: one row per question with bucket and gap labels.
* summary.json: aggregate counts for error attribution.

It intentionally reuses the execution-comparison semantics from
scripts/oracle_analysis.py so oracle, diagnostics, and future verifier labels
share one evaluation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIFFICULTIES = ("easy", "medium", "hard", "extra", "unknown")
SQL_KEYWORDS = {
    "select", "from", "where", "join", "on", "as", "and", "or", "by",
    "group", "order", "having", "limit", "distinct", "count", "sum", "avg",
    "min", "max", "union", "except", "intersect", "in", "not", "like",
    "between", "asc", "desc", "is", "null", "inner", "left", "right",
}


def normalize_sql(sql: str) -> str:
    # Preserve literal case: SQLite string comparisons are case-sensitive, so
    # lowercasing the whole SQL can merge semantically different candidates.
    return " ".join((sql or "").strip().rstrip(";").split())


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql or "", re.IGNORECASE))


def db_path(spider_dir: Path, db_id: str) -> Path:
    return spider_dir / "database" / db_id / f"{db_id}.sqlite"


def execute_sql(db_file: Path, sql: str, max_ops: int) -> tuple[bool, Any]:
    conn = None
    try:
        conn = sqlite3.connect(str(db_file), timeout=10.0)
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
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        return False, str(exc)


def norm_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_rows(rows: list) -> list[tuple]:
    return [tuple(norm_cell(value) for value in row) for row in rows]


def compare_results(r1: list, r2: list, order_sensitive: bool = False) -> bool:
    if len(r1) != len(r2):
        return False
    if not r1:
        return True
    nr1 = normalize_rows(r1)
    nr2 = normalize_rows(r2)
    if order_sensitive:
        return nr1 == nr2
    return sorted(nr1, key=str) == sorted(nr2, key=str)


def result_digest(rows: list, order_sensitive: bool = False) -> str:
    normalized = normalize_rows(rows)
    if not order_sensitive:
        normalized = sorted(normalized, key=str)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def parse_mapping(values: list[str], kind: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must be in name=path format: {value}")
        name, path = value.split("=", 1)
        if not name:
            raise ValueError(f"{kind} name is empty: {value}")
        parsed[name] = Path(path)
    return parsed


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_hardness(results: dict[str, list[dict]], total: int) -> list[str]:
    if not results:
        return ["unknown"] * total
    first = next(iter(results.values()))
    return [item.get("hardness", "unknown") for item in first]


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", text or "")
        if len(token) > 1
    }


def sql_features(sql: str, question: str) -> dict[str, Any]:
    lowered = (sql or "").lower()
    sql_tokens = tokenize(sql)
    content_sql_tokens = {token for token in sql_tokens if token not in SQL_KEYWORDS}
    question_tokens = tokenize(question)
    overlap = question_tokens & content_sql_tokens
    table_like = re.findall(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", lowered)
    return {
        "sql_len": len(sql or ""),
        "sql_tokens": len(sql_tokens),
        "question_sql_token_overlap": len(overlap),
        "question_sql_token_overlap_ratio": len(overlap) / len(question_tokens) if question_tokens else 0.0,
        "num_select": len(re.findall(r"\bselect\b", lowered)),
        "num_join": len(re.findall(r"\bjoin\b", lowered)),
        "num_tables_mentioned": len(set(table_like)),
        "has_where": bool(re.search(r"\bwhere\b", lowered)),
        "has_group_by": bool(re.search(r"\bgroup\s+by\b", lowered)),
        "has_order_by": bool(re.search(r"\border\s+by\b", lowered)),
        "has_having": bool(re.search(r"\bhaving\b", lowered)),
        "has_limit": bool(re.search(r"\blimit\b", lowered)),
        "has_subquery": lowered.count("select") > 1,
        "has_set_op": bool(re.search(r"\b(union|except|intersect)\b", lowered)),
        "has_aggregate": bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", lowered)),
    }


def rank_clusters(
    clusters: dict[str, list[int]],
    candidates: list[dict[str, Any]],
    *,
    weight_field: str = "occurrence_count",
) -> list[tuple[str, list[int], int, float]]:
    ranked = []
    for key, indexes in clusters.items():
        weight = sum(candidates[idx][weight_field] for idx in indexes)
        avg_len = sum(len(candidates[idx]["sql"]) for idx in indexes) / len(indexes)
        ranked.append((key, indexes, weight, avg_len))
    return sorted(ranked, key=lambda item: (-item[2], item[3], item[0]))


def cluster_rank_map(ranked: list[tuple[str, list[int], int, float]]) -> dict[str, int]:
    return {key: rank for rank, (key, _indexes, _weight, _avg_len) in enumerate(ranked, start=1)}


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bucket_name(sft_ok: bool, dpo_ok: bool, oracle_ok: bool) -> str:
    if sft_ok and dpo_ok:
        return "A_sft_correct_dpo_correct"
    if sft_ok and not dpo_ok:
        return "B_sft_correct_dpo_wrong"
    if not sft_ok and dpo_ok:
        return "C_sft_wrong_dpo_correct"
    if oracle_ok:
        return "D_both_wrong_oracle_correct"
    return "E_both_wrong_oracle_wrong"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose SC candidate selection errors.")
    parser.add_argument("--spider_dir", type=Path, default=Path("dataset/spider"))
    parser.add_argument(
        "--candidate",
        action="append",
        default=[
            "sft=outputs/spider/sft/sc_n12/candidates.json",
            "dpo=outputs/spider/dpo/sc_n12/candidates.json",
        ],
        help="Policy candidate file in name=path format. Can be repeated.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[
            "sft=outputs/spider/sft/sc_n12/results.json",
            "dpo=outputs/spider/dpo/sc_n12/results.json",
        ],
        help="Policy SC result file in name=path format. Can be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/spider/diagnostics/sc_n12"))
    parser.add_argument("--max_ops", type=int, default=1_000_000)
    parser.add_argument("--gold_max_ops", type=int, default=1_000_000)
    args = parser.parse_args()

    candidate_paths = parse_mapping(args.candidate, "candidate")
    result_paths = parse_mapping(args.result, "result")

    candidate_sets = {name: load_json(path) for name, path in candidate_paths.items()}
    result_sets = {name: load_json(path)["results"] for name, path in result_paths.items()}

    policies = list(candidate_sets)
    base_policy = policies[0]
    base = candidate_sets[base_policy]
    total = len(base)
    hardness = load_hardness(result_sets, total)

    exec_cache: dict[tuple[str, str, int], tuple[bool, Any]] = {}

    def cached_execute(db_id: str, sql: str, max_ops: int) -> tuple[bool, Any]:
        key = (db_id, normalize_sql(sql), max_ops)
        if key not in exec_cache:
            exec_cache[key] = execute_sql(db_path(args.spider_dir, db_id), sql, max_ops)
        return exec_cache[key]

    candidate_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "total": total,
        "candidate_files": {name: str(path) for name, path in candidate_paths.items()},
        "result_files": {name: str(path) for name, path in result_paths.items()},
        "policy_accuracy": {},
        "bucket_counts": Counter(),
        "bucket_by_hardness": defaultdict(Counter),
        "oracle_by_hardness": Counter(),
        "counts_by_hardness": Counter(hardness),
        "merged_vote_accuracy": Counter(),
        "gap_dpo_wrong_oracle_correct": {
            "total": 0,
            "by_hardness": Counter(),
            "modes": Counter(),
            "correct_vote_rank": Counter(),
            "correct_shared_rank": Counter(),
        },
        "both_wrong_oracle_correct": {
            "total": 0,
            "by_hardness": Counter(),
            "modes": Counter(),
        },
        "policy_conflicts": {
            "sft_correct_dpo_wrong": Counter(),
            "sft_wrong_dpo_correct": Counter(),
        },
        "candidate_histograms": {
            "unique_candidates": Counter(),
            "executable_candidates": Counter(),
            "correct_candidates": Counter(),
        },
        "source_of_correct_candidate": Counter(),
        "candidate_feature_summary": {},
        "label_inconsistencies": Counter(),
    }

    for name, results in result_sets.items():
        correct = sum(bool(item.get("exec_correct")) for item in results)
        summary["policy_accuracy"][name] = {
            "correct": correct,
            "total": len(results),
            "exec": correct / len(results) if results else 0.0,
        }

    correct_feature_values: dict[str, list[float]] = defaultdict(list)
    wrong_feature_values: dict[str, list[float]] = defaultdict(list)
    executable_label_counts = Counter()

    for index, item in enumerate(base):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["gold_sql"]
        item_hardness = hardness[index] if index < len(hardness) else "unknown"
        order_sensitive = has_order_by(gold_sql)
        gold_ok, gold_result = cached_execute(db_id, gold_sql, args.gold_max_ops)

        by_norm: dict[str, dict[str, Any]] = {}
        ordered_candidates: list[dict[str, Any]] = []
        for policy in policies:
            current = candidate_sets[policy][index]
            if current["db_id"] != db_id or current["question"] != question:
                raise ValueError(f"candidate files are not aligned at index {index}")
            for rank, sql in enumerate(current["candidates"]):
                norm = normalize_sql(sql)
                if not norm:
                    continue
                if norm not in by_norm:
                    record = {
                        "candidate_local_id": len(ordered_candidates),
                        "sql": sql,
                        "normalized_sql": norm,
                        "occurrences": [],
                        "source_counts": Counter(),
                    }
                    by_norm[norm] = record
                    ordered_candidates.append(record)
                by_norm[norm]["occurrences"].append({"source": policy, "rank": rank})
                by_norm[norm]["source_counts"][policy] += 1

        shared_clusters: dict[str, list[int]] = defaultdict(list)
        vote_clusters: dict[str, list[int]] = defaultdict(list)
        executable_count = 0
        correct_count = 0

        for local_id, cand in enumerate(ordered_candidates):
            ok, result = cached_execute(db_id, cand["sql"], args.max_ops)
            cand["executable"] = ok
            cand["error"] = None if ok else str(result)[:160]
            cand["occurrence_count"] = len(cand["occurrences"])
            cand["source_counts"] = dict(cand["source_counts"])
            cand["sources"] = sorted(cand["source_counts"])
            cand["first_source"] = cand["occurrences"][0]["source"]
            cand["first_rank"] = cand["occurrences"][0]["rank"]
            cand["min_rank"] = min(occ["rank"] for occ in cand["occurrences"])
            cand["ranks_by_source"] = {
                policy: [occ["rank"] for occ in cand["occurrences"] if occ["source"] == policy]
                for policy in cand["sources"]
            }
            cand.update(sql_features(cand["sql"], question))

            if ok:
                executable_count += 1
                result_rows = result
                is_correct = bool(gold_ok and compare_results(result_rows, gold_result, order_sensitive))
                correct_count += int(is_correct)
                cand["is_exec_correct"] = is_correct
                cand["empty_result"] = len(result_rows) == 0
                cand["result_size"] = len(result_rows)
                shared_key = result_digest(result_rows, order_sensitive)
                vote_key = f"empty:{local_id}" if not result_rows else shared_key
                cand["shared_result_cluster_id"] = shared_key
                cand["vote_result_cluster_id"] = vote_key
                shared_clusters[shared_key].append(local_id)
                vote_clusters[vote_key].append(local_id)
            else:
                cand["is_exec_correct"] = False
                cand["empty_result"] = None
                cand["result_size"] = None
                cand["shared_result_cluster_id"] = None
                cand["vote_result_cluster_id"] = None

        shared_ranked = rank_clusters(shared_clusters, ordered_candidates)
        vote_ranked = rank_clusters(vote_clusters, ordered_candidates)
        shared_rank = cluster_rank_map(shared_ranked)
        vote_rank = cluster_rank_map(vote_ranked)
        shared_winner = shared_ranked[0][1][0] if shared_ranked else None
        vote_winner = vote_ranked[0][1][0] if vote_ranked else None

        for cand in ordered_candidates:
            shared_key = cand["shared_result_cluster_id"]
            vote_key = cand["vote_result_cluster_id"]
            if shared_key is not None:
                cand["shared_cluster_size_unique"] = len(shared_clusters[shared_key])
                cand["shared_cluster_size_occurrence"] = sum(
                    ordered_candidates[idx]["occurrence_count"] for idx in shared_clusters[shared_key]
                )
                cand["shared_cluster_rank"] = shared_rank[shared_key]
            else:
                cand["shared_cluster_size_unique"] = 0
                cand["shared_cluster_size_occurrence"] = 0
                cand["shared_cluster_rank"] = None
            if vote_key is not None:
                cand["vote_cluster_size_unique"] = len(vote_clusters[vote_key])
                cand["vote_cluster_size_occurrence"] = sum(
                    ordered_candidates[idx]["occurrence_count"] for idx in vote_clusters[vote_key]
                )
                cand["vote_cluster_rank"] = vote_rank[vote_key]
            else:
                cand["vote_cluster_size_unique"] = 0
                cand["vote_cluster_size_occurrence"] = 0
                cand["vote_cluster_rank"] = None
            cand["is_shared_cluster_winner"] = cand["candidate_local_id"] == shared_winner
            cand["is_vote_cluster_winner"] = cand["candidate_local_id"] == vote_winner

        policy_ok = {
            name: bool(result_sets[name][index].get("exec_correct", False))
            for name in result_sets
        }
        sft_ok = policy_ok.get("sft", False)
        dpo_ok = policy_ok.get("dpo", False)
        oracle_ok = correct_count > 0
        bucket = bucket_name(sft_ok, dpo_ok, oracle_ok)
        summary["bucket_counts"][bucket] += 1
        summary["bucket_by_hardness"][bucket][item_hardness] += 1
        summary["oracle_by_hardness"][item_hardness] += int(oracle_ok)

        if sft_ok and not dpo_ok:
            summary["policy_conflicts"]["sft_correct_dpo_wrong"][item_hardness] += 1
        if not sft_ok and dpo_ok:
            summary["policy_conflicts"]["sft_wrong_dpo_correct"][item_hardness] += 1

        vote_winner_correct = bool(vote_winner is not None and ordered_candidates[vote_winner]["is_exec_correct"])
        shared_winner_correct = bool(shared_winner is not None and ordered_candidates[shared_winner]["is_exec_correct"])
        summary["merged_vote_accuracy"]["vote_empty_unique_correct"] += int(vote_winner_correct)
        summary["merged_vote_accuracy"]["shared_empty_cluster_correct"] += int(shared_winner_correct)

        correct_vote_ranks = [
            cand["vote_cluster_rank"] for cand in ordered_candidates
            if cand["is_exec_correct"] and cand["vote_cluster_rank"] is not None
        ]
        correct_shared_ranks = [
            cand["shared_cluster_rank"] for cand in ordered_candidates
            if cand["is_exec_correct"] and cand["shared_cluster_rank"] is not None
        ]
        best_correct_vote_rank = min(correct_vote_ranks) if correct_vote_ranks else None
        best_correct_shared_rank = min(correct_shared_ranks) if correct_shared_ranks else None
        top_vote_weight = vote_ranked[0][2] if vote_ranked else 0
        best_correct_vote_weight = max(
            (cand["vote_cluster_size_occurrence"] for cand in ordered_candidates if cand["is_exec_correct"]),
            default=0,
        )

        if oracle_ok and not dpo_ok:
            gap = summary["gap_dpo_wrong_oracle_correct"]
            gap["total"] += 1
            gap["by_hardness"][item_hardness] += 1
            gap["correct_vote_rank"][str(best_correct_vote_rank or 0)] += 1
            gap["correct_shared_rank"][str(best_correct_shared_rank or 0)] += 1
            if vote_winner_correct:
                mode = "merged_vote_would_fix"
            elif best_correct_vote_weight == top_vote_weight:
                mode = "correct_tied_largest_but_lost_tiebreak"
            elif best_correct_vote_weight > 0:
                mode = "correct_minority"
            else:
                mode = "no_correct_executable_seen"
            gap["modes"][mode] += 1

        if bucket == "D_both_wrong_oracle_correct":
            both_gap = summary["both_wrong_oracle_correct"]
            both_gap["total"] += 1
            both_gap["by_hardness"][item_hardness] += 1
            if vote_winner_correct:
                mode = "merged_vote_would_fix"
            elif best_correct_vote_weight == top_vote_weight:
                mode = "correct_tied_largest_but_lost_tiebreak"
            elif best_correct_vote_weight > 0:
                mode = "correct_minority"
            else:
                mode = "no_correct_executable_seen"
            both_gap["modes"][mode] += 1

        summary["candidate_histograms"]["unique_candidates"][len(ordered_candidates)] += 1
        summary["candidate_histograms"]["executable_candidates"][executable_count] += 1
        summary["candidate_histograms"]["correct_candidates"][correct_count] += 1

        correct_sources = Counter()
        for cand in ordered_candidates:
            if cand["is_exec_correct"]:
                if set(cand["sources"]) == {"sft", "dpo"}:
                    correct_sources["both_sources"] += 1
                elif "sft" in cand["sources"]:
                    correct_sources["sft_only"] += 1
                elif "dpo" in cand["sources"]:
                    correct_sources["dpo_only"] += 1
        if correct_sources:
            summary["source_of_correct_candidate"].update(correct_sources)

        for cand in ordered_candidates:
            if cand["executable"]:
                label_key = "correct" if cand["is_exec_correct"] else "wrong"
                executable_label_counts[label_key] += 1
                target = correct_feature_values if cand["is_exec_correct"] else wrong_feature_values
                for feature in (
                    "occurrence_count",
                    "min_rank",
                    "sql_len",
                    "sql_tokens",
                    "question_sql_token_overlap_ratio",
                    "result_size",
                    "shared_cluster_size_occurrence",
                    "vote_cluster_size_occurrence",
                    "num_select",
                    "num_join",
                    "num_tables_mentioned",
                ):
                    value = cand.get(feature)
                    if isinstance(value, (int, float)):
                        target[feature].append(float(value))

            row = {
                "index": index,
                "db_id": db_id,
                "question": question,
                "hardness": item_hardness,
                "gold_sql": gold_sql,
                "bucket": bucket,
                "policy_exec_correct": policy_ok,
                **cand,
            }
            candidate_rows.append(row)

        item_rows.append({
            "index": index,
            "db_id": db_id,
            "question": question,
            "hardness": item_hardness,
            "bucket": bucket,
            "policy_exec_correct": policy_ok,
            "oracle_correct": oracle_ok,
            "num_unique_candidates": len(ordered_candidates),
            "num_executable_candidates": executable_count,
            "num_correct_candidates": correct_count,
            "vote_winner_correct": vote_winner_correct,
            "shared_winner_correct": shared_winner_correct,
            "best_correct_vote_rank": best_correct_vote_rank,
            "best_correct_shared_rank": best_correct_shared_rank,
            "best_correct_vote_weight": best_correct_vote_weight,
            "top_vote_weight": top_vote_weight,
        })

        for name, ok in policy_ok.items():
            if ok and not oracle_ok:
                summary["label_inconsistencies"][f"{name}_correct_but_no_oracle_candidate"] += 1

    for label, values in (("correct", correct_feature_values), ("wrong", wrong_feature_values)):
        summary["candidate_feature_summary"][label] = {
            "count": executable_label_counts[label],
            **{f"avg_{feature}": avg(vals) for feature, vals in values.items()},
        }

    # Convert Counters/defaultdicts into plain JSON-serializable values.
    def plain(value: Any) -> Any:
        if isinstance(value, Counter):
            return dict(value)
        if isinstance(value, defaultdict):
            return {key: plain(val) for key, val in value.items()}
        if isinstance(value, dict):
            return {key: plain(val) for key, val in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "candidate_labels.jsonl", "w", encoding="utf-8") as f:
        for row in candidate_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.output_dir / "item_diagnostics.jsonl", "w", encoding="utf-8") as f:
        for row in item_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(plain(summary), f, ensure_ascii=False, indent=2)

    print(f"Saved {args.output_dir / 'candidate_labels.jsonl'}")
    print(f"Saved {args.output_dir / 'item_diagnostics.jsonl'}")
    print(f"Saved {args.output_dir / 'summary.json'}")
    print("Bucket counts:")
    for key, value in summary["bucket_counts"].most_common():
        print(f"  {key}: {value}")
    print("Merged oracle:", sum(1 for row in item_rows if row["oracle_correct"]), "/", total)


if __name__ == "__main__":
    main()
