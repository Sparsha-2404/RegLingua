
"""
M1 Day 10 â€” Explanation Grounding Evaluation

Uses the real M1 regulatory knowledge base for evidence retrieval.
Structural checks validate citation consistency and abstention behavior.
Substantive support still requires human review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.explainer_agent import ExplainerAgent
from core.schemas import EvidenceItem, ExplainerRequest, RiskLabel
from services.retrieval.deterministic_retriever import DeterministicRetriever


KB_PATH = REPO_ROOT / "data" / "regulatory" / "regulatory_kb_m1.json"
OUTPUT_PATH = (
    REPO_ROOT / "data" / "evaluation" / "explanation_grounding_m1.json"
)


def retrieve_evidence(
    retriever: DeterministicRetriever,
    query: str,
    regulator: str | None = None,
    top_k: int = 3,
) -> list[EvidenceItem]:
    """Retrieve real M1 corpus chunks and convert them to explainer evidence."""

    regulator_candidates = [regulator] if regulator else None

    chunks = retriever.retrieve(
        query_text=query,
        regulator_candidates=regulator_candidates,
        top_k=top_k,
    )

    return [
        EvidenceItem(
            clause_id=chunk.source_id,
            source_document=chunk.document_title,
            clause_text=chunk.text,
            similarity_score=float(chunk.score),
        )
        for chunk in chunks
    ]


def evaluate_case(
    case_id: str,
    flagged_text: str,
    risk_label: RiskLabel,
    evidence: list[EvidenceItem],
) -> dict:
    response = ExplainerAgent().explain(
        ExplainerRequest(
            case_id=case_id,
            flagged_text=flagged_text,
            risk_label=risk_label,
            evidence=evidence,
        )
    )

    evidence_clause_ids = {item.clause_id for item in evidence}
    has_evidence = bool(evidence)

    citation_matches_evidence = (
        response.clause_cited in evidence_clause_ids
    )


    is_compliant = risk_label == RiskLabel.COMPLIANT

    checks = {
        "grounded_flag_matches_evidence_presence": (
            response.grounded == has_evidence
        ),
        "citation_matches_supplied_evidence": (
            citation_matches_evidence
            if response.grounded
            else (
                response.clause_cited in ("UNSUPPORTED", "N/A")
                if is_compliant
                else response.clause_cited == "UNSUPPORTED"
            )
        ),
        "no_fabricated_regulatory_citation_without_evidence": (
            has_evidence
            or response.clause_cited in ("UNSUPPORTED", "N/A")
        ),
    }

    if has_evidence:
        evidence_status = "REQUIRES_MANUAL_SUPPORT_REVIEW"
    elif is_compliant:
        evidence_status = "UNVERIFIED"
    else:
        evidence_status = "ABSTAINED"


    evidence_details = [
        {
            "source_id": item.clause_id,
            "source_document": item.source_document,
            "clause_text": item.clause_text,
            "retrieval_score": item.similarity_score,
        }
        for item in evidence
    ]

    return {
        "case_id": case_id,
        "flagged_text": flagged_text,
        "risk_label": risk_label.value,
        "evidence_count": len(evidence),
        "evidence": evidence_details,
        "clause_cited": response.clause_cited,
        "grounded": response.grounded,
        "reason": response.reason,
        "checks": checks,
        "passed_structural_checks": all(checks.values()),
        "substantive_support_status": evidence_status,
    }


def main() -> None:
    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"M1 knowledge base not found: {KB_PATH}"
        )

    # Explicitly use the real M1 corpus, not sample_corpus.json.
    retriever = DeterministicRetriever(corpus_path=KB_PATH)

    # Use three contrasting cases:
    # 1. Risky text with retrieved evidence.
    # 2. Risky text with no retrieved evidence (tests abstention).
    # 3. Compliant text with retrieved evidence.
    risky_text = "We guarantee 20% returns."
    compliant_text = "Track your monthly budget."

    risky_evidence = retrieve_evidence(
        retriever,
        risky_text,
        regulator="SEBI",
    )

    # Deliberately use an empty evidence list to test abstention.
    no_evidence: list[EvidenceItem] = []

    compliant_evidence = retrieve_evidence(
        retriever,
        compliant_text,
        regulator="SEBI",
    )

    cases = [
        {
            "case_id": "M1-GROUND-001",
            "flagged_text": risky_text,
            "risk_label": RiskLabel.RISKY,
            "evidence": risky_evidence,
        },
        {
            "case_id": "M1-GROUND-002",
            "flagged_text": risky_text,
            "risk_label": RiskLabel.RISKY,
            "evidence": no_evidence,
        },
        {
            "case_id": "M1-GROUND-003",
            "flagged_text": compliant_text,
            "risk_label": RiskLabel.COMPLIANT,
            "evidence": compliant_evidence,
        },
    ]

    results = [
        evaluate_case(
            case_id=case["case_id"],
            flagged_text=case["flagged_text"],
            risk_label=case["risk_label"],
            evidence=case["evidence"],
        )
        for case in cases
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed = sum(
        result["passed_structural_checks"] for result in results
    )

    print(f"M1 knowledge base: {KB_PATH}")
    print(f"Cases evaluated: {len(results)}")
    print(f"Cases passing structural checks: {passed}/{len(results)}")
    print(f"Cases requiring manual evidence review: {len(results)}")
    print(f"Results saved to: {OUTPUT_PATH}")

    for result in results:
        status = (
            "PASS"
            if result["passed_structural_checks"]
            else "FAIL"
        )
        print(
            f"\n{result['case_id']}: {status} "
            f"(evidence={result['evidence_count']})"
        )
        print(f"  Citation: {result['clause_cited']}")
        print(f"  Grounded: {result['grounded']}")
        print("  Substantive support: MANUAL REVIEW REQUIRED")

        for name, check_passed in result["checks"].items():
            print(
                f"  {name}: "
                f"{'PASS' if check_passed else 'FAIL'}"
            )


if __name__ == "__main__":
    main()
