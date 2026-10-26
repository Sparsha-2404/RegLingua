
"""Evaluate RegLingua hybrid retrieval on a small, manually labelled set."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.retrieval.hybrid_retriever import HybridRetriever

K = 3
OUTPUT_FILE = ROOT / "tests" / "evaluation" / "retrieval_results.json"

EVALUATION_CASES = [
    {
        "language": "English",
        "query": "guaranteed returns on investment products",
        "expected": "SEBI_IA_002",
    },
    {
        "language": "English",
        "query": "lenders must disclose interest rates fees and total loan cost",
        "expected": "RBI_LEND_001",
    },
    {
        "language": "English",
        "query": "instant unconditional loan approval claims by lending apps",
        "expected": "RBI_LEND_002",
    },
    {
        "language": "English",
        "query": "consent for unrelated purposes in a single checkbox",
        "expected": "DPDP_001",
    },
    {
        "language": "Code-mixed",
        "query": "loan app instant approval nahi milta",
        "expected": "RBI_LEND_002",
    },
    {
        "language": "Hindi",
        "query": "व्यक्तिगत डेटा की सहमति और चेकबॉक्स",
        "expected": "DPDP_HI_004",
    },
]


def main():
    retriever = HybridRetriever()
    results_report = []
    precision_total = 0.0
    recall_total = 0.0
    reciprocal_rank_total = 0.0

    print("\nRegLingua Retrieval Evaluation")
    print("=" * 65)

    for case in EVALUATION_CASES:
        retrieved = retriever.retrieve(query_text=case["query"], top_k=K)
        ranked_ids = [result.source_id for result in retrieved]
        expected = case["expected"]

        hit = expected in ranked_ids[:K]
        precision_at_k = sum(
            source_id == expected for source_id in ranked_ids[:K]
        ) / K
        recall_at_k = 1.0 if hit else 0.0
        reciprocal_rank = (
            1.0 / (ranked_ids.index(expected) + 1)
            if expected in ranked_ids
            else 0.0
        )

        precision_total += precision_at_k
        recall_total += recall_at_k
        reciprocal_rank_total += reciprocal_rank

        case_result = {
            "language": case["language"],
            "query": case["query"],
            "expected_source_id": expected,
            "ranked_source_ids": ranked_ids,
            "expected_found_in_top_k": hit,
            f"precision_at_{K}": round(precision_at_k, 4),
            f"recall_at_{K}": round(recall_at_k, 4),
            "reciprocal_rank": round(reciprocal_rank, 4),
        }
        results_report.append(case_result)

        print(f"\nLanguage: {case['language']}")
        print(f"Query: {case['query']}")
        print(f"Expected: {expected}")
        print(f"Top {K}: {ranked_ids}")
        print(
            f"P@{K}={precision_at_k:.3f} | "
            f"R@{K}={recall_at_k:.3f} | RR={reciprocal_rank:.3f}"
        )

    count = len(EVALUATION_CASES)
    summary = {
        "queries_evaluated": count,
        f"mean_precision_at_{K}": round(precision_total / count, 4),
        f"mean_recall_at_{K}": round(recall_total / count, 4),
        "mean_reciprocal_rank": round(reciprocal_rank_total / count, 4),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_note": "Small synthetic corpus; results are preliminary.",
        "top_k": K,
        "summary": summary,
        "cases": results_report,
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print(f"Queries evaluated: {count}")
    print(f"Mean Precision@{K}: {summary[f'mean_precision_at_{K}']:.3f}")
    print(f"Mean Recall@{K}:    {summary[f'mean_recall_at_{K}']:.3f}")
    print(f"MRR:                {summary['mean_reciprocal_rank']:.3f}")
    print(f"\nJSON report saved to: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
