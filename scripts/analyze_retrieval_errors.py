"""
DAY 5 DELIVERABLE (M2): Retrieval error analysis.

Continues "M2: Evaluate retrieval queries against known relevant
sources." (Day 5 plan). An aggregate Recall@5 hides where the retriever
is actually weak -- this breaks results down by regulator and by
language (the axis that matters most for this project's multilingual
story), and surfaces concrete failing examples so a human can see WHY
they failed, not just that they failed.

Usage:
    python3 scripts/analyze_retrieval_errors.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "data" / "evaluation" / "retrieval_eval_results_m2.csv"
EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"

K_VALUES = (1, 3, 5)
NUM_WORST_FAILURES = 5


def load_results():
    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_text_by_query_id():
    with open(EVAL_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["query_id"]: r["text"] for r in rows}


def _as_bool(value: str) -> bool:
    return value == "True"


def print_breakdown(title: str, rows: list[dict], group_key: str) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    print(f"\n{title}")
    print("-" * 50)
    header = f"{'Group':<10}" + "".join(f"{'Recall@' + str(k):>14}" for k in K_VALUES) + f"{'n':>6}"
    print(header)
    for group_name in sorted(groups):
        group_rows = groups[group_name]
        n = len(group_rows)
        line = f"{group_name:<10}"
        for k in K_VALUES:
            hits = sum(1 for r in group_rows if _as_bool(r[f"hit_at_{k}"]))
            recall = hits / n if n else 0.0
            line += f"{recall:>14.4f}"
        line += f"{n:>6}"
        print(line)


def print_worst_failures(rows: list[dict], text_by_query_id: dict, n: int) -> None:
    failures = [r for r in rows if not _as_bool(r["hit_at_5"])]

    print(f"\n{n} worst failures (hit_at_5 == False), showing first {n} of {len(failures)} total:")
    print("-" * 70)
    for row in failures[:n]:
        text = text_by_query_id.get(row["query_id"], "<text not found>")
        print(f"\nquery_id:            {row['query_id']}")
        print(f"text:                {text!r}")
        print(f"true_regulator:      {row['true_regulator']}")
        print(f"language:            {row['language']}")
        print(f"expected_source_id:  {row['expected_source_id']}")
        print(f"retrieved_source_ids: {row['retrieved_source_ids']}")


def main() -> None:
    rows = load_results()
    text_by_query_id = load_text_by_query_id()

    print_breakdown("Recall@K by regulator", rows, "true_regulator")
    print_breakdown("Recall@K by language (multilingual breakdown)", rows, "language")
    print_worst_failures(rows, text_by_query_id, NUM_WORST_FAILURES)


if __name__ == "__main__":
    main()
