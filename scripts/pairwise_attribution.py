#!/usr/bin/env python3
"""Pairwise attribution for recoverable selector failures.

This script consumes candidate diagnostics and builds explicit pairs:

    wrong selected SQL  vs  oracle-correct candidate SQL

for cases where the current selector is wrong but the merged candidate pool
contains at least one correct candidate. It then performs a lightweight SQL
clause diff to identify what kind of signal a verifier would need.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLAUSE_ORDER = (
    "where",
    "group by",
    "having",
    "order by",
    "limit",
    "union",
    "intersect",
    "except",
)

SQL_KEYWORDS = {
    "select", "from", "where", "join", "on", "as", "and", "or", "by",
    "group", "order", "having", "limit", "distinct", "count", "sum", "avg",
    "min", "max", "union", "except", "intersect", "in", "not", "like",
    "between", "asc", "desc", "is", "null", "inner", "left", "right",
    "case", "when", "then", "else", "end", "exists", "all", "any",
}


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact(sql: str) -> str:
    return " ".join((sql or "").strip().rstrip(";").split())


def structural(sql: str) -> str:
    return compact(sql).lower()


def strip_literals(sql: str) -> str:
    text = re.sub(r"'(?:''|[^'])*'", "?", sql or "")
    text = re.sub(r'"(?:""|[^"])*"', "?", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "?", text)
    return text


def extract_literals(sql: str) -> list[str]:
    values = []
    values.extend(re.findall(r"'((?:''|[^'])*)'", sql or ""))
    values.extend(re.findall(r'"((?:""|[^"])*)"', sql or ""))
    values.extend(re.findall(r"\b\d+(?:\.\d+)?\b", sql or ""))
    return [value.strip() for value in values]


def find_clause_start(sql_lower: str, clause: str) -> re.Match[str] | None:
    pattern = re.escape(clause).replace(r"\ ", r"\s+")
    return re.search(rf"\b{pattern}\b", sql_lower)


def extract_clause(sql_lower: str, clause: str) -> str:
    start = find_clause_start(sql_lower, clause)
    if not start:
        return ""
    rest = sql_lower[start.end():]
    ends = [c for c in CLAUSE_ORDER if c != clause]
    end_match = re.search(
        r"\b(" + "|".join(re.escape(c).replace(r"\ ", r"\s+") for c in ends) + r")\b",
        rest,
        re.IGNORECASE,
    )
    return compact(rest[:end_match.start()] if end_match else rest)


def extract_select_clause(sql_lower: str) -> str:
    match = re.search(r"\bselect\b\s+(.+?)\s+\bfrom\b", sql_lower, re.DOTALL)
    return compact(match.group(1)) if match else ""


def extract_from_clause(sql_lower: str) -> str:
    match = re.search(
        r"\bfrom\b\s+(.+?)(?=\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|\bunion\b|\bintersect\b|\bexcept\b|$)",
        sql_lower,
        re.DOTALL,
    )
    return compact(match.group(1)) if match else ""


def split_top_level(expr: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    for idx, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            part = compact(expr[start:idx])
            if part:
                parts.append(part)
            start = idx + 1
    tail = compact(expr[start:])
    if tail:
        parts.append(tail)
    return parts


def extract_tables(sql_lower: str) -> set[str]:
    text = re.sub(r"\([^()]*\)", " ", sql_lower)
    tables = set()
    for match in re.finditer(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", text, re.IGNORECASE):
        tables.add(match.group(1).lower())
    from_clause = extract_from_clause(text)
    if from_clause:
        for part in from_clause.split(","):
            token = part.strip().split()
            if token and token[0] not in {"select", "join"}:
                tables.add(token[0].strip('"`[]').lower())
    return tables


def extract_join_conditions(sql_lower: str) -> list[str]:
    conditions = []
    pattern = (
        r"\bjoin\b\s+[a-z_][a-z0-9_]*(?:\s+(?:as\s+)?[a-z_][a-z0-9_]*)?"
        r"\s+\bon\b\s+(.+?)(?=\bjoin\b|\bwhere\b|\bgroup\s+by\b|\bhaving\b|\border\s+by\b|\blimit\b|$)"
    )
    for match in re.finditer(pattern, sql_lower, re.IGNORECASE | re.DOTALL):
        conditions.append(compact(match.group(1)))
    return sorted(conditions)


def sql_profile(sql: str) -> dict[str, Any]:
    lower = structural(sql)
    lower_no_literals = structural(strip_literals(sql))
    select_clause = extract_select_clause(lower)
    select_clause_no_literals = extract_select_clause(lower_no_literals)
    select_items = split_top_level(select_clause)
    select_items_no_literals = split_top_level(select_clause_no_literals)
    literals = extract_literals(sql)
    tokens = {
        token
        for token in re.findall(r"[a-z_][a-z0-9_]*", lower)
        if token not in SQL_KEYWORDS
    }
    return {
        "normalized": lower,
        "normalized_no_literals": lower_no_literals,
        "select_clause": select_clause,
        "select_clause_no_literals": select_clause_no_literals,
        "select_items": select_items,
        "select_items_no_literals": select_items_no_literals,
        "tables": sorted(extract_tables(lower)),
        "from_clause": extract_from_clause(lower),
        "join_conditions": extract_join_conditions(lower),
        "where_clause": extract_clause(lower, "where"),
        "where_clause_no_literals": extract_clause(lower_no_literals, "where"),
        "group_clause": extract_clause(lower, "group by"),
        "having_clause": extract_clause(lower, "having"),
        "order_clause": extract_clause(lower, "order by"),
        "limit_clause": extract_clause(lower, "limit"),
        "agg_funcs": sorted(set(re.findall(r"\b(count|sum|avg|min|max)\s*\(", lower))),
        "set_ops": sorted(set(re.findall(r"\b(union|intersect|except)\b", lower))),
        "distinct": bool(re.search(r"\bdistinct\b", lower)),
        "subquery_count": max(0, len(re.findall(r"\bselect\b", lower)) - 1),
        "literals": literals,
        "literals_lower": [value.lower() for value in literals],
        "tokens": sorted(tokens),
    }


def diff_sql(wrong_sql: str, correct_sql: str) -> dict[str, Any]:
    wrong = sql_profile(wrong_sql)
    correct = sql_profile(correct_sql)
    tags = []
    details = {}

    def add(tag: str, key: str, wrong_value: Any, correct_value: Any) -> None:
        tags.append(tag)
        details[key] = {"wrong": wrong_value, "correct": correct_value}

    if wrong["tables"] != correct["tables"]:
        add("TABLE_SELECTION", "tables", wrong["tables"], correct["tables"])
    if wrong["join_conditions"] != correct["join_conditions"]:
        add("JOIN_CONDITION", "join_conditions", wrong["join_conditions"], correct["join_conditions"])
    if wrong["select_clause_no_literals"] != correct["select_clause_no_literals"]:
        wrong_items = wrong["select_items_no_literals"]
        correct_items = correct["select_items_no_literals"]
        if sorted(wrong_items) == sorted(correct_items) and wrong_items != correct_items:
            add("COLUMN_ORDER", "select_items", wrong["select_items"], correct["select_items"])
        else:
            add("SELECT", "select_clause", wrong["select_clause"], correct["select_clause"])
    if wrong["where_clause_no_literals"] != correct["where_clause_no_literals"]:
        add("WHERE_CONDITION", "where_clause", wrong["where_clause"], correct["where_clause"])
    elif wrong["where_clause"] != correct["where_clause"]:
        add("FILTER_VALUE", "where_clause", wrong["where_clause"], correct["where_clause"])
    if wrong["group_clause"] != correct["group_clause"]:
        add("GROUP_BY", "group_clause", wrong["group_clause"], correct["group_clause"])
    if wrong["having_clause"] != correct["having_clause"]:
        add("HAVING", "having_clause", wrong["having_clause"], correct["having_clause"])
    if wrong["order_clause"] != correct["order_clause"]:
        add("ORDER_BY", "order_clause", wrong["order_clause"], correct["order_clause"])
        wrong_dir = "desc" if "desc" in wrong["order_clause"] else "asc" if "asc" in wrong["order_clause"] else ""
        correct_dir = "desc" if "desc" in correct["order_clause"] else "asc" if "asc" in correct["order_clause"] else ""
        if wrong_dir != correct_dir:
            tags.append("SORT_DIRECTION")
    if wrong["limit_clause"] != correct["limit_clause"]:
        add("LIMIT", "limit_clause", wrong["limit_clause"], correct["limit_clause"])
    if wrong["agg_funcs"] != correct["agg_funcs"]:
        add("AGGREGATION", "agg_funcs", wrong["agg_funcs"], correct["agg_funcs"])
    if wrong["set_ops"] != correct["set_ops"]:
        add("SET_OP", "set_ops", wrong["set_ops"], correct["set_ops"])
    if wrong["distinct"] != correct["distinct"]:
        add("DISTINCT", "distinct", wrong["distinct"], correct["distinct"])
    if wrong["subquery_count"] != correct["subquery_count"]:
        add("SUBQUERY", "subquery_count", wrong["subquery_count"], correct["subquery_count"])
    if wrong["literals"] != correct["literals"]:
        if wrong["literals_lower"] == correct["literals_lower"]:
            add("LITERAL_CASE", "literals", wrong["literals"], correct["literals"])
        else:
            add("LITERAL_VALUE", "literals", wrong["literals"], correct["literals"])
    if wrong["normalized_no_literals"] == correct["normalized_no_literals"] and wrong["literals"] != correct["literals"]:
        tags.append("VALUE_ONLY_DIFF")
    if not tags and wrong["normalized"] != correct["normalized"]:
        tags.append("SURFACE_OR_ALIAS")
        details["normalized"] = {"wrong": wrong["normalized"], "correct": correct["normalized"]}

    semantic_tags = semantic_error_tags(tags, wrong, correct)
    return {
        "diff_tags": sorted(set(tags)),
        "semantic_tags": semantic_tags,
        "details": details,
    }


def semantic_error_tags(diff_tags: list[str], wrong: dict[str, Any], correct: dict[str, Any]) -> list[str]:
    tags = set()
    diff = set(diff_tags)
    if {"ORDER_BY", "LIMIT"} & diff:
        tags.add("SORT_OR_EXTREME_VALUE")
    if "SORT_DIRECTION" in diff:
        tags.add("SORT_DIRECTION")
    if "TABLE_SELECTION" in diff:
        tags.add("TABLE_SELECTION")
    if "JOIN_CONDITION" in diff:
        tags.add("JOIN_KEY_OR_PATH")
    if {"WHERE_CONDITION", "FILTER_VALUE", "LITERAL_VALUE", "LITERAL_CASE"} & diff:
        tags.add("FILTER_OR_VALUE_GROUNDING")
    if {"GROUP_BY", "HAVING"} & diff:
        tags.add("GROUPING")
    if {"SELECT", "COLUMN_ORDER"} & diff:
        tags.add("PROJECTION")
    if "AGGREGATION" in diff:
        tags.add("AGGREGATION")
    if "SET_OP" in diff:
        tags.add("SET_OPERATION")
    if "SUBQUERY" in diff:
        tags.add("NESTED_QUERY")
    if len(wrong["tokens"]) > len(correct["tokens"]) + 4:
        tags.add("SPURIOUS_COMPLEXITY")
    return sorted(tags) or ["UNCLASSIFIED_SURFACE"]


def candidate_sort_key(row: dict[str, Any]) -> tuple:
    return (
        -int(row.get("vote_cluster_size_occurrence") or 0),
        int(row.get("min_rank") or 999),
        int(row.get("sql_len") or len(row.get("sql", ""))),
        row.get("sql", ""),
    )


def find_matching_candidate(candidates: list[dict[str, Any]], pred_sql: str) -> dict[str, Any] | None:
    pred_norm = compact(pred_sql)
    for candidate in candidates:
        if compact(candidate["sql"]) == pred_norm:
            return candidate
    pred_struct = structural(pred_sql)
    for candidate in candidates:
        if structural(candidate["sql"]) == pred_struct:
            return candidate
    return None


def plain_counter(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=lambda k: (-counter[k], str(k)))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wrong-vs-correct SQL attribution pairs.")
    parser.add_argument("--diagnostics_dir", type=Path, default=Path("outputs/spider/diagnostics/sc_n12"))
    parser.add_argument("--dpo_results", type=Path, default=Path("outputs/spider/dpo/sc_n12/results.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/spider/diagnostics/sc_n12"))
    args = parser.parse_args()

    candidate_rows = load_jsonl(args.diagnostics_dir / "candidate_labels.jsonl")
    item_rows = load_jsonl(args.diagnostics_dir / "item_diagnostics.jsonl")
    dpo_results = load_json(args.dpo_results)["results"]

    candidates_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_index[int(row["index"])].append(row)

    pairs = []
    summary = {
        "total_recoverable_dpo_wrong_oracle_correct": 0,
        "mode_counts": Counter(),
        "bucket_counts": Counter(),
        "by_hardness": Counter(),
        "diff_tag_counts": Counter(),
        "semantic_tag_counts": Counter(),
        "correct_source_counts": Counter(),
        "wrong_source_counts": Counter(),
        "correct_rank_counts": Counter(),
        "wrong_rank_counts": Counter(),
    }

    for item in item_rows:
        index = int(item["index"])
        policy_ok = item["policy_exec_correct"]
        if policy_ok.get("dpo", False) or not item["oracle_correct"]:
            continue

        candidates = candidates_by_index[index]
        correct_candidates = [row for row in candidates if row["is_exec_correct"]]
        if not correct_candidates:
            continue

        wrong_selected = find_matching_candidate(candidates, dpo_results[index]["pred_sql"])
        if wrong_selected is None:
            executable_wrong = [row for row in candidates if row["executable"] and not row["is_exec_correct"]]
            wrong_selected = sorted(executable_wrong, key=candidate_sort_key)[0] if executable_wrong else None
        if wrong_selected is None:
            continue

        best_correct = sorted(correct_candidates, key=candidate_sort_key)[0]
        attribution = diff_sql(wrong_selected["sql"], best_correct["sql"])

        if item["vote_winner_correct"]:
            mode = "merged_vote_would_fix"
        elif item["best_correct_vote_weight"] == item["top_vote_weight"]:
            mode = "correct_tied_largest_but_lost_tiebreak"
        else:
            mode = "correct_minority"

        pair = {
            "index": index,
            "db_id": item["db_id"],
            "question": item["question"],
            "hardness": item["hardness"],
            "bucket": item["bucket"],
            "mode": mode,
            "gold_sql": next((row["gold_sql"] for row in candidates), ""),
            "wrong_sql": wrong_selected["sql"],
            "correct_sql": best_correct["sql"],
            "wrong": {
                "sources": wrong_selected["sources"],
                "min_rank": wrong_selected["min_rank"],
                "occurrence_count": wrong_selected["occurrence_count"],
                "vote_cluster_size_occurrence": wrong_selected["vote_cluster_size_occurrence"],
                "vote_cluster_rank": wrong_selected["vote_cluster_rank"],
                "result_size": wrong_selected["result_size"],
                "sql_len": wrong_selected["sql_len"],
            },
            "correct": {
                "sources": best_correct["sources"],
                "min_rank": best_correct["min_rank"],
                "occurrence_count": best_correct["occurrence_count"],
                "vote_cluster_size_occurrence": best_correct["vote_cluster_size_occurrence"],
                "vote_cluster_rank": best_correct["vote_cluster_rank"],
                "result_size": best_correct["result_size"],
                "sql_len": best_correct["sql_len"],
            },
            **attribution,
        }
        pairs.append(pair)

        summary["total_recoverable_dpo_wrong_oracle_correct"] += 1
        summary["mode_counts"][mode] += 1
        summary["bucket_counts"][item["bucket"]] += 1
        summary["by_hardness"][item["hardness"]] += 1
        summary["correct_rank_counts"][str(best_correct["min_rank"])] += 1
        summary["wrong_rank_counts"][str(wrong_selected["min_rank"])] += 1
        for source in best_correct["sources"]:
            summary["correct_source_counts"][source] += 1
        for source in wrong_selected["sources"]:
            summary["wrong_source_counts"][source] += 1
        for tag in pair["diff_tags"]:
            summary["diff_tag_counts"][tag] += 1
        for tag in pair["semantic_tags"]:
            summary["semantic_tag_counts"][tag] += 1

    output_summary = {
        key: plain_counter(value) if isinstance(value, Counter) else value
        for key, value in summary.items()
    }

    save_jsonl(args.output_dir / "recoverable_pairs.jsonl", pairs)
    save_json(args.output_dir / "pair_attribution_summary.json", output_summary)

    print(f"Saved {args.output_dir / 'recoverable_pairs.jsonl'}")
    print(f"Saved {args.output_dir / 'pair_attribution_summary.json'}")
    print(f"Pairs: {len(pairs)}")
    print("Top semantic tags:")
    for tag, count in summary["semantic_tag_counts"].most_common(10):
        print(f"  {tag}: {count}")
    print("Top diff tags:")
    for tag, count in summary["diff_tag_counts"].most_common(10):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
