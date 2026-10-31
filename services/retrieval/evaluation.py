"""
DAY 5 DELIVERABLE (M2): Retrieval evaluation harness.

"M2: Evaluate retrieval queries against known relevant sources." (Day 5 plan)
Gate: "EvidenceAgent returns selected chunks with source IDs."

Core metric logic, kept separate from scripts/evaluate_retrieval_recall_precision.py
(the CLI entry point) so it can be exercised directly against small, fake
retrievers in tests/unit/test_day5_retrieval_evaluation.py -- correctness of
the metric math must be proven before any real retrieval score from it is
trusted (see that test file).

Works against ANY object with a `.retrieve(query_text,
regulator_candidates=None, topic_hint=None, top_k=5) -> list[chunk]`
method, where each returned chunk has a `.source_id` attribute --
deliberately duck-typed so the same harness evaluates
services.retrieval.deterministic_retriever.DeterministicRetriever
(keyword-only, runs anywhere) and
services.retrieval.hybrid_retriever.HybridRetriever (keyword + vector,
needs a one-time embedding model download) without any code change.

"Known relevant sources" (the Day 5 task's own words) = the
`expected_source_id` column in data/evaluation/regulator_topic_eval_m2.csv
(M2's Day 4 deliverable) -- M1's verified mapping of an example to the
regulatory source that actually supports it.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_eval_rows(csv_path: str | Path) -> List[Dict[str, str]]:
    """
    Load data/evaluation/regulator_topic_eval_m2.csv and keep only rows
    with a non-blank expected_source_id -- the real evaluation subset
    (the Day 4 deliverable's docs: ~510/516 rows qualify).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("expected_source_id")]


def evaluate_retriever(
    retriever: Any,
    eval_rows: Sequence[Dict[str, str]],
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[int, Dict[str, Any]]:
    """
    Run `retriever` against every row in `eval_rows` and compute Recall@K
    and Precision@K for each K in k_values.

    Recall@K: fraction of rows where expected_source_id appears anywhere
    in the top-K retrieved source_ids.

    Precision@K: standard single-relevant-document convention (exactly one
    known-relevant source per query here, not a set) -- 1/K if the
    expected source was found within the top K, else 0, averaged across
    all rows. This deliberately differs from "found anywhere in top K
    scores 1.0" -- a hit at rank 1 scores the same as a hit at rank K only
    for Recall, not Precision, so Precision@K still reflects how early the
    correct result appeared relative to K.

    Calls retriever.retrieve() exactly ONCE per row (at top_k=max(k_values))
    and slices the result for each smaller K, rather than calling
    retrieve() once per K value.

    Returns: {k: {"recall": float, "precision": float,
                  "per_row_results": [...]}} for each k in k_values.
    per_row_results (same list object, shared across all k entries) has one
    dict per eval row: query_id, true_regulator, language,
    expected_source_id, retrieved_source_ids (top max(k_values)), and
    hit_at_<k> for every k in k_values.
    """
    if not k_values:
        raise ValueError("k_values must not be empty")

    max_k = max(k_values)
    per_row_results: List[Dict[str, Any]] = []

    for row in eval_rows:
        expected_source_id = row["expected_source_id"]
        true_regulator = row["true_regulator"]

        retrieved = retriever.retrieve(
            row["text"],
            regulator_candidates=[true_regulator],
            top_k=max_k,
        )
        retrieved_source_ids = [chunk.source_id for chunk in retrieved]

        row_result: Dict[str, Any] = {
            "query_id": row["query_id"],
            "true_regulator": true_regulator,
            "language": row.get("language", ""),
            "expected_source_id": expected_source_id,
            "retrieved_source_ids": retrieved_source_ids,
        }
        for k in k_values:
            top_k_ids = retrieved_source_ids[:k]
            hit = expected_source_id in top_k_ids
            row_result[f"hit_at_{k}"] = hit
        per_row_results.append(row_result)

    n = len(per_row_results)
    results: Dict[int, Dict[str, Any]] = {}
    for k in k_values:
        if n == 0:
            recall = 0.0
            precision = 0.0
        else:
            hits = sum(1 for r in per_row_results if r[f"hit_at_{k}"])
            recall = hits / n
            # Precision@k: 1/k when found within top-k, else 0, averaged.
            precision = sum(
                (1.0 / k) if r[f"hit_at_{k}"] else 0.0 for r in per_row_results
            ) / n
        results[k] = {
            "recall": recall,
            "precision": precision,
            "per_row_results": per_row_results,
        }
    return results
