"""
DAY 5 DELIVERABLE (M2): Evaluate retrieval queries against known relevant
sources.

"M2: Evaluate retrieval queries against known relevant sources." (Day 5 plan)
Gate: "EvidenceAgent returns selected chunks with source IDs."

Thin CLI wrapper around services.retrieval.evaluation -- the metric logic
itself lives there so it can be unit-tested against fake retrievers
independently of this script (see
tests/unit/test_day5_retrieval_evaluation.py).

By default evaluates DeterministicRetriever (keyword-only) against the
real 308-chunk regulatory KB -- this runs anywhere, no network needed.
Pass --hybrid to evaluate HybridRetriever (keyword + vector) instead; this
needs a one-time Hugging Face embedding-model download, so it only works
in an environment with normal internet access.

Usage:
    python3 scripts/evaluate_retrieval_recall_precision.py
    python3 scripts/evaluate_retrieval_recall_precision.py --hybrid
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.retrieval.evaluation import evaluate_retriever, load_eval_rows  # noqa: E402

EVAL_PATH = REPO_ROOT / "data" / "evaluation" / "regulator_topic_eval_m2.csv"
KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"
RESULTS_PATH = REPO_ROOT / "data" / "evaluation" / "retrieval_eval_results_m2.csv"

K_VALUES = (1, 3, 5)

RESULTS_FIELDNAMES = [
    "query_id",
    "true_regulator",
    "language",
    "expected_source_id",
    "retrieved_source_ids",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
]


def save_results_csv(per_row_results, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        for row in per_row_results:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "true_regulator": row["true_regulator"],
                    "language": row["language"],
                    "expected_source_id": row["expected_source_id"],
                    "retrieved_source_ids": ";".join(row["retrieved_source_ids"]),
                    "hit_at_1": row["hit_at_1"],
                    "hit_at_3": row["hit_at_3"],
                    "hit_at_5": row["hit_at_5"],
                }
            )


def build_retriever(use_hybrid: bool):
    from services.retrieval.deterministic_retriever import DeterministicRetriever

    if use_hybrid:
        from services.retrieval.hybrid_retriever import HybridRetriever
        from services.embeddings.vector_index import RegulatoryVectorIndex

        return HybridRetriever(
            keyword_retriever=DeterministicRetriever(corpus_path=KB_PATH),
            vector_index=RegulatoryVectorIndex(corpus_path=KB_PATH),
        )
    return DeterministicRetriever(corpus_path=KB_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Evaluate HybridRetriever (keyword+vector) instead of the "
        "default keyword-only DeterministicRetriever. Requires a "
        "one-time Hugging Face model download.",
    )
    args = parser.parse_args()

    retriever_name = "HybridRetriever" if args.hybrid else "DeterministicRetriever"
    print(f"Retriever: {retriever_name}")
    print(f"Eval set:  {EVAL_PATH}")
    print(f"KB:        {KB_PATH}\n")

    eval_rows = load_eval_rows(EVAL_PATH)
    print(f"Loaded {len(eval_rows)} eval rows with a verified expected_source_id.\n")

    retriever = build_retriever(args.hybrid)
    results = evaluate_retriever(retriever, eval_rows, k_values=K_VALUES)

    print("=" * 50)
    print(f"{'K':>3}  {'Recall@K':>10}  {'Precision@K':>12}")
    print("-" * 50)
    for k in K_VALUES:
        r = results[k]["recall"]
        p = results[k]["precision"]
        print(f"{k:>3}  {r:>10.4f}  {p:>12.4f}")
    print("=" * 50)

    per_row_results = results[K_VALUES[0]]["per_row_results"]  # same list for every k
    save_results_csv(per_row_results, RESULTS_PATH)
    print(f"\nSaved {len(per_row_results)} per-row results to {RESULTS_PATH}")

    total_failures_at_5 = sum(1 for r in per_row_results if not r["hit_at_5"])
    print(f"Rows with hit_at_5 == False (missed even at k=5): "
          f"{total_failures_at_5} / {len(per_row_results)}")


if __name__ == "__main__":
    main()
