"""
Agent 3 — Regulatory Researcher (M3, reduced scope: retrieval, embeddings,
hybrid search only).

Day 1 deliverable: a stable contract (core.contracts) that the Supervisor
and other agents call — this class IS that interface, currently backed by
a mock-free but still deterministic implementation.

Day 2 deliverable: the agent is now backed by an actual (deterministic,
keyword-based) retriever instead of a hardcoded mock, so the pipeline
gate — "one request travels through every stage using deterministic/mock
components" — is satisfiable for the research stage specifically.

Per the Supervisor's routing logic in the architecture doc:

    elif evidence_is_missing:
        route_to = "regulatory_researcher"

...this agent only runs once the Risk Analyst (Agent 2) has already
produced a risk_prediction on the case.
"""

from __future__ import annotations

from typing import Optional

from agents.research_contracts import (
    RegulatoryResearchInput,
    RegulatoryResearchOutput,
    RegulatoryResearchResult,
)
from services.retrieval.hybrid_retriever import HybridRetriever


class RegulatoryResearchAgent:
    AGENT_NAME = "regulatory_researcher"

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        # Dependency-injected so the retriever implementation can be replaced
        # without changing the agent contract or downstream callers.
        self.retriever = retriever or HybridRetriever()

    def run(self, request: RegulatoryResearchInput) -> RegulatoryResearchOutput:
        request.validate()

        chunks = self.retriever.retrieve(
            query_text=request.normalized_text,
            regulator_candidates=request.regulator_candidates or None,
            topic_hint=request.topic_hint,
            top_k=request.top_k,
        )

        top_score = chunks[0].score if chunks else 0.0

        # Abstain when retrieval returns no sufficiently relevant evidence.
        if not chunks or top_score < 0.1:
            return RegulatoryResearchOutput(
                agent=self.AGENT_NAME,
                status="failure",
                confidence=0.0,
                result=RegulatoryResearchResult(
                    retrieval_results=[],
                    regulator_candidates=request.regulator_candidates,
                ),
                evidence_refs=[],
                next_action="needs_review",
            )

        status = "success" if top_score >= 0.5 else "partial"

        return RegulatoryResearchOutput(
            agent=self.AGENT_NAME,
            status=status,
            confidence=top_score,
            result=RegulatoryResearchResult(
                retrieval_results=chunks,
                regulator_candidates=request.regulator_candidates,
            ),
            evidence_refs=[c.source_id for c in chunks],
            next_action="evidence_selection" if status == "success" else "needs_review",
        )
