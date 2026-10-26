"""
Exercises the Day 1 contract + Day 2 deterministic retriever together.

Run directly for a readable demo:
    python -m tests.test_research_agent

Run as tests (if pytest is available):
    pytest tests/test_research_agent.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from agents.research_contracts import RegulatoryResearchInput, RiskPrediction
from agents.regulatory_research_agent import RegulatoryResearchAgent


def _run_case(agent, case_id, text, language, regulator_candidates, topic_hint=None):
    request = RegulatoryResearchInput(
        case_id=case_id,
        normalized_text=text,
        language=language,
        risk_prediction=RiskPrediction(label="HIGH", confidence=0.86, factors=["flagged_by_risk_agent"]),
        regulator_candidates=regulator_candidates,
        topic_hint=topic_hint,
        top_k=3,
    )
    output = agent.run(request)
    print(f"\n=== Case {case_id} ({language}) ===")
    print(f"query: {text!r}")
    print(f"status={output.status}  confidence={output.confidence}  next_action={output.next_action}")
    for chunk in output.result.retrieval_results:
        print(f"  [{chunk.score:.2f}] {chunk.source_id} ({chunk.regulator}) — {chunk.section}")
    return output


def test_sebi_guaranteed_returns_english():
    agent = RegulatoryResearchAgent()
    output = _run_case(
        agent,
        case_id="CASE-001",
        text="we guarantee assured returns on your investment with zero risk",
        language="en",
        regulator_candidates=["SEBI"],
    )
    assert output.status == "success"
    assert output.evidence_refs, "expected at least one evidence reference"
    assert output.result.retrieval_results[0].regulator == "SEBI"


def test_rbi_digital_lending_english():
    agent = RegulatoryResearchAgent()
    output = _run_case(
        agent,
        case_id="CASE-002",
        text="get instant unconditional loan approval in 2 minutes, no paperwork",
        language="en",
        regulator_candidates=["RBI"],
    )
    assert output.status in ("success", "partial")
    assert any(c.source_id == "RBI_LEND_002" for c in output.result.retrieval_results)


def test_dpdp_consent_hindi():
    agent = RegulatoryResearchAgent()
    output = _run_case(
        agent,
        case_id="CASE-003",
        text="इस ऐप का उपयोग करने के लिए सहमति दें और अपना डेटा साझा करें",
        language="hi",
        regulator_candidates=["DPDP"],
        topic_hint="consent",
    )
    assert output.status in ("success", "partial")
    assert any(c.regulator == "DPDP" for c in output.result.retrieval_results)


def test_no_regulator_filter_still_ranks_across_corpus():
    agent = RegulatoryResearchAgent()
    output = _run_case(
        agent,
        case_id="CASE-004",
        text="personalized investment recommendations for your portfolio",
        language="en",
        regulator_candidates=[],
    )
    assert output.result.retrieval_results, "expected results when no regulator filter is applied"


def test_abstains_when_no_keyword_overlap():
    agent = RegulatoryResearchAgent()
    output = _run_case(
        agent,
        case_id="CASE-005",
        text="our weekend picnic menu includes sandwiches and lemonade",
        language="en",
        regulator_candidates=[],
    )

    assert output.status == "failure"
    assert output.confidence == 0.0
    assert output.evidence_refs == []
    assert output.next_action == "needs_review"

if __name__ == "__main__":
    test_sebi_guaranteed_returns_english()
    test_rbi_digital_lending_english()
    test_dpdp_consent_hindi()
    test_no_regulator_filter_still_ranks_across_corpus()
    test_abstains_when_no_keyword_overlap()
    print("\nAll M3 Day 1/Day 2 checks passed.")

def test_abstains_on_unrelated_restaurant_query():
    agent = RegulatoryResearchAgent()
    output = _run_case(
        agent,
        case_id="CASE-006",
        text="What are the best restaurants for a weekend picnic?",
        language="en",
        regulator_candidates=[],
    )

    assert output.status == "failure"
    assert output.confidence == 0.0
    assert output.result.retrieval_results == []
    assert output.evidence_refs == []
    assert output.next_action == "needs_review"
