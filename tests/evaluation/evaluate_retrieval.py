
"""Evaluate RegLingua hybrid retrieval on a small, manually labelled set."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.retrieval.hybrid_retriever import HybridRetriever
from agents.research_contracts import RegulatoryResearchInput, RiskPrediction
from agents.regulatory_research_agent import RegulatoryResearchAgent

K = 3
OUTPUT_FILE = ROOT / "tests" / "evaluation" / "retrieval_results.json"

NEGATIVE_QUERIES = [
    "What are the best restaurants for a weekend picnic?",
    "How do I grow tomatoes in a home garden?",
    "Recommend a science-fiction movie for tonight.",
]

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
    {
        "language": "English",
        "query": "investment adviser registration requirements for personalized advice",
        "expected": "SEBI_IA_001",
    },
    {
        "language": "English",
        "query": "mutual fund risk disclosures and past performance",
        "expected": "SEBI_MF_003",
    },
    {
        "language": "English",
        "query": "wallet balance and deposit insurance claims",
        "expected": "RBI_PAY_003",
    },
    {
        "language": "English",
        "query": "parental consent for children's data and targeted advertising",
        "expected": "DPDP_003",
    },
    {
        "language": "English",
        "query": "privacy notice in plain language explaining its purpose",
        "expected": "DPDP_002",
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


    agent = RegulatoryResearchAgent(retriever=retriever)
    negative_results = []

    for index, query in enumerate(NEGATIVE_QUERIES, start=1):
        request = RegulatoryResearchInput(
            case_id=f"NEGATIVE-{index:03d}",
            normalized_text=query,
            language="en",
            risk_prediction=RiskPrediction(
                label="LOW",
                confidence=0.9,
                factors=[],
            ),
            regulator_candidates=[],
            top_k=K,
        )
        output = agent.run(request)
        abstained = (
            output.status == "failure"
            and output.confidence == 0.0
            and not output.evidence_refs
            and not output.result.retrieval_results
            and output.next_action == "needs_review"
        )
        negative_results.append({
            "query": query,
            "status": output.status,
            "confidence": output.confidence,
            "evidence_refs": output.evidence_refs,
            "retrieved_source_ids": [
                chunk.source_id for chunk in output.result.retrieval_results
            ],
            "next_action": output.next_action,
            "abstained_as_expected": abstained,
        })
        print(
            f"Negative query: {query!r} | "
            f"status={output.status} | abstained={abstained}"
        )

    negative_summary = {
        "queries_evaluated": len(negative_results),
        "correct_abstentions": sum(
            item["abstained_as_expected"] for item in negative_results
        ),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_note": "Small synthetic corpus; results are preliminary.",
        "top_k": K,
        "summary": summary,
        "cases": results_report,
        "negative_query_summary": negative_summary,
        "negative_query_cases": negative_results,
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
    print(
        f"Correct abstentions: {negative_summary['correct_abstentions']}/"
        f"{negative_summary['queries_evaluated']}"
    )
    print(f"\nJSON report saved to: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
