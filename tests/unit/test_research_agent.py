from agents.regulatory_research_agent import RegulatoryResearchAgent
from agents.research_contracts import (
    RegulatoryResearchInput,
    RiskPrediction,
)
from services.retrieval.hybrid_retriever import HybridRetriever


def test_research_agent_uses_hybrid_retriever():
    agent = RegulatoryResearchAgent()

    assert isinstance(agent.retriever, HybridRetriever)


def test_research_agent_returns_hybrid_retrieval_results():
    agent = RegulatoryResearchAgent()

    request = RegulatoryResearchInput(
        case_id="DAY6-001",
        normalized_text="guaranteed returns on investment",
        language="en",
        risk_prediction=RiskPrediction(
            label="HIGH",
            confidence=1.0,
            factors=["guaranteed returns"],
        ),
        regulator_candidates=["SEBI"],
        top_k=3,
    )

    output = agent.run(request)

    assert output.agent == "regulatory_researcher"
    assert output.status in ("success", "partial")
    assert output.result.retrieval_results
    assert len(output.result.retrieval_results) <= 3
    assert output.result.retrieval_results[0].source_id == "SEBI_IA_002"


def test_research_agent_preserves_retrieved_metadata():
    agent = RegulatoryResearchAgent()

    request = RegulatoryResearchInput(
        case_id="DAY6-002",
        normalized_text="guaranteed returns",
        language="en",
        risk_prediction=RiskPrediction(
            label="HIGH",
            confidence=1.0,
            factors=["guaranteed returns"],
        ),
        regulator_candidates=["SEBI"],
        top_k=2,
    )

    output = agent.run(request)

    result = output.result.retrieval_results[0]

    assert result.regulator == "SEBI"
    assert result.source_id == "SEBI_IA_002"
    assert result.document_title
    assert result.text
    assert result.source_url
    assert result.topic