from agents.research_contracts import (
    RegulatoryResearchInput,
    RetrievedChunk,
    RiskPrediction,
)
from agents.regulatory_research_agent import RegulatoryResearchAgent


class FakeRetriever:
    def __init__(self, score):
        self.score = score

    def retrieve(
        self,
        query_text,
        regulator_candidates=None,
        topic_hint=None,
        top_k=5,
    ):
        return [
            RetrievedChunk(
                source_id="TEST_SOURCE",
                regulator="SEBI",
                document_title="Test regulation",
                section="Section 1",
                effective_date="2026-01-01",
                text="Test regulatory evidence",
                source_url="https://example.com/test",
                topic="test",
                score=self.score,
            )
        ]


def make_request():
    return RegulatoryResearchInput(
        case_id="THRESHOLD-TEST",
        normalized_text="test regulatory query",
        language="en",
        risk_prediction=RiskPrediction(
            label="HIGH",
            confidence=0.9,
            factors=["test"],
        ),
        regulator_candidates=["SEBI"],
        top_k=3,
    )


def test_score_below_abstention_threshold_fails():
    output = RegulatoryResearchAgent(
        retriever=FakeRetriever(0.0999)
    ).run(make_request())

    assert output.status == "failure"
    assert output.confidence == 0.0
    assert output.result.retrieval_results == []
    assert output.evidence_refs == []
    assert output.next_action == "needs_review"


def test_score_at_abstention_threshold_is_partial():
    output = RegulatoryResearchAgent(
        retriever=FakeRetriever(0.1)
    ).run(make_request())

    assert output.status == "partial"
    assert output.confidence == 0.1
    assert output.evidence_refs == ["TEST_SOURCE"]
    assert output.next_action == "needs_review"


def test_score_at_success_threshold_is_success():
    output = RegulatoryResearchAgent(
        retriever=FakeRetriever(0.5)
    ).run(make_request())

    assert output.status == "success"
    assert output.confidence == 0.5
    assert output.evidence_refs == ["TEST_SOURCE"]
    assert output.next_action == "evidence_selection"
