"""Evaluation metrics and scoring utilities for NL2SQL.

Usage
-----
>>> from alignsql.eval.metrics import compare_results, init_scores, print_scores
"""

from typing import Any, List


def compare_results(r1: list, r2: list) -> bool:
    """Compare two SQL result sets for semantic equality.

    Handles None normalization, float rounding, and row ordering.
    """
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

    rows1 = sorted([norm_row(r) for r in r1], key=safe_key)
    rows2 = sorted([norm_row(r) for r in r2], key=safe_key)
    return rows1 == rows2


def init_scores():
    """Create the score dict skeleton for Spider-style difficulty levels."""
    return {
        "easy": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "medium": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "hard": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "extra": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
        "all": {"count": 0, "exec": 0, "exact": 0, "partial": {}},
    }


PARTIAL_TYPES = [
    "select",
    "select(no AGG)",
    "where",
    "where(no OP)",
    "group(no Having)",
    "group",
    "order",
    "and/or",
    "IUEN",
    "keywords",
]

LEVELS = ["easy", "medium", "hard", "extra", "all"]


def print_scores(scores, etype: str = "all"):
    """Print formatted Spider evaluation scores to stdout."""
    print("\n" + "=" * 80)
    print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *LEVELS))

    counts = [scores[level]["count"] for level in LEVELS]
    print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))

    if etype in ("all", "exec"):
        print("=" * 80)
        print(" " * 20 + "EXECUTION ACCURACY")
        print("=" * 80)
        vals = [scores[level]["exec"] for level in LEVELS]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("execution", *vals))

    if etype in ("all", "match"):
        print("\n" + "=" * 80)
        print(" " * 18 + "EXACT MATCHING ACCURACY")
        print("=" * 80)
        vals = [scores[level]["exact"] for level in LEVELS]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exact match", *vals))

        print("\n" + "-" * 80)
        print("-" * 18 + "PARTIAL MATCHING ACCURACY")
        print("-" * 80)
        for t in PARTIAL_TYPES:
            vals = [scores[level]["partial"].get(t, {}).get("acc", 0) for level in LEVELS]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(t, *vals))

        print("\n" + "-" * 80)
        print("-" * 18 + "PARTIAL MATCHING RECALL")
        print("-" * 80)
        for t in PARTIAL_TYPES:
            vals = [scores[level]["partial"].get(t, {}).get("rec", 0) for level in LEVELS]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(t, *vals))

        print("\n" + "-" * 80)
        print("-" * 18 + "PARTIAL MATCHING F1")
        print("-" * 80)
        for t in PARTIAL_TYPES:
            vals = [scores[level]["partial"].get(t, {}).get("f1", 0) for level in LEVELS]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(t, *vals))
