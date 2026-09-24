"""
EvidenceAgentAdapter — Day 1/2 integration glue between M3 and M4.
=====================================================================

M3's Regulatory Research Agent (agents/regulatory_research_agent.py, backed
by the Day 2 deterministic keyword retriever) speaks its own contract
(agents/research_contracts.py): `RegulatoryResearchInput` /
`RegulatoryResearchOutput` / `RetrievedChunk` — because M3 built it to be
testable standalone, independent of the Supervisor's design (Day 1 gate:
"all modules have interfaces even if implementations are mocks").

M4's Supervisor calls a case-scoped `EvidenceAgent.gather(EvidenceRequest)
-> EvidenceResponse` instead, using `core.schemas.EvidenceItem` as the
evidence type threaded through `CaseState`.

This module is the ONLY place that translates between the two, so M3's
retrieval internals (keyword search today, hybrid BM25 + vector search from
Day 5 onward) can keep changing without the Supervisor noticing.
"""

from __future__ import annotations

from typing import Optional

from agents.regulatory_research_agent import RegulatoryResearchAgent
from agents.research_contracts import RegulatoryResearchInput
from agents.research_contracts import RiskPrediction as ResearchRiskPrediction
from core.schemas import EvidenceItem, EvidenceRequest, EvidenceResponse, Regulator


class EvidenceAgentAdapter:
    """Implements `agents.contracts.EvidenceAgent` on top of M3's
    `RegulatoryResearchAgent`."""

    def __init__(self, research_agent: Optional[RegulatoryResearchAgent] = None) -> None:
        self.research_agent = research_agent or RegulatoryResearchAgent()

    def gather(self, request: EvidenceRequest) -> EvidenceResponse:
        regulator_candidates = (
            [request.regulator_hint.value]
            if request.regulator_hint and request.regulator_hint != Regulator.GENERAL
            else []
        )

        # M3's contract wants a `risk_prediction` to shape the query, but by
        # the time the Supervisor calls gather_evidence it has already
        # decided this case needs review — the actual risk_level value
        # doesn't change M3's retrieval behaviour (only regulator_candidates
        # and topic_hint do), so a fixed HIGH/1.0 placeholder is passed here
        # rather than re-deriving M2's four-way RiskLevel from M4's
        # three-way RiskLabel a second time.
        research_input = RegulatoryResearchInput(
            case_id=request.case_id,
            normalized_text=request.flagged_text,
            language=request.language.value,
            risk_prediction=ResearchRiskPrediction(label="HIGH", confidence=1.0, factors=[]),
            regulator_candidates=regulator_candidates,
            top_k=request.top_k,
        )

        output = self.research_agent.run(research_input)

        evidence = [
            EvidenceItem(
                clause_id=chunk.source_id,
                source_document=chunk.document_title,
                clause_text=chunk.text,
                similarity_score=chunk.score,
            )
            for chunk in output.result.retrieval_results
        ]

        return EvidenceResponse(
            case_id=request.case_id,
            evidence=evidence,
            retrieval_method="keyword",  # M3's Day 2 deterministic retriever; "hybrid" lands Day 5
        )
