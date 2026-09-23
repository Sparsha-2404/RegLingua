"""
RegLingua — Agent Factory
============================
Owned by: M4 (wiring), consuming M1/M2/M3 components.

Single place that assembles the concrete agent roster the Supervisor's
graph calls into. This is the direct replacement for the old
`agents.mock_agents.build_mock_agents()` factory: as of the Day 1-2
integration merge, RiskAgent, EvidenceAgent and the language detector are
real M2/M3 components (via adapters); Explainer, Rewriter and Verifier
remain mocks until Day 6-8 per the 14-day plan.

Nothing in `agents/supervisor.py` changed to make this swap — it already
only depends on the `agents.contracts` Protocols, which is exactly the
point of defining agent contracts on Day 1 before any real implementation
existed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.evidence_agent_adapter import EvidenceAgentAdapter
from agents.language_agent_adapter import LanguageAgentAdapter
from agents.mock_agents import MockExplainerAgent, MockRewriterAgent, MockVerifierAgent
from agents.regulatory_research_agent import RegulatoryResearchAgent
from agents.risk_agent_adapter import RiskAgentAdapter
from services.classifier.interface import MockRiskClassifier, RiskClassifier
from services.retrieval.deterministic_retriever import DeterministicRetriever

# M1's real, ingested regulatory knowledge base (308 chunks across
# SEBI/RBI/DPDP, extracted from RegLingua_M1_FINAL_v2.xlsx). Used in place
# of M3's original 10-entry demo corpus once you want the pipeline running
# against real regulatory text instead of the smoke-test fixture.
M1_REGULATORY_KB_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "regulatory" / "regulatory_kb_m1.json"
)


def build_agents(
    classifier: Optional[RiskClassifier] = None,
    corpus_path: Optional[Path] = None,
) -> dict[str, object]:
    """Build the full agent roster used by the Supervisor graph.

    Args:
        classifier: M2's risk classifier. Defaults to `MockRiskClassifier`
            (Day 1-2 dependency-free keyword mock). Pass a loaded
            `TfidfLogisticClassifier` here from Day 3 onward.
        corpus_path: regulatory corpus for M3's retriever. Defaults to
            M3's small demo corpus (`data/regulatory/sample_corpus.json`,
            used by M3's own unit tests). Pass `M1_REGULATORY_KB_PATH` to
            run against M1's real 308-chunk knowledge base instead.
    """
    risk_agent = RiskAgentAdapter(classifier=classifier or MockRiskClassifier())

    retriever = DeterministicRetriever(corpus_path=corpus_path) if corpus_path else DeterministicRetriever()
    research_agent = RegulatoryResearchAgent(retriever=retriever)
    evidence_agent = EvidenceAgentAdapter(research_agent=research_agent)

    return {
        "language_detector": LanguageAgentAdapter(),
        "risk_agent": risk_agent,
        "evidence_agent": evidence_agent,
        "explainer_agent": MockExplainerAgent(),
        "rewriter_agent": MockRewriterAgent(),
        "verifier_agent": MockVerifierAgent(risk_agent=risk_agent),
    }
