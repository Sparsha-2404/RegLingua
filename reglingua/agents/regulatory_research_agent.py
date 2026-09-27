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

from reglingua.core.contracts import (
    RegulatoryResearchInput,
    RegulatoryResearchOutput,
    RegulatoryResearchResult,
)
from reglingua.services.retrieval.deterministic_retriever import DeterministicRetriever


class RegulatoryResearchAgent:
    AGENT_NAME = "regulatory_researcher"

    def __init__(self, retriever: Optional[DeterministicRetriever] = None):
        # Dependency-injected so the Day 5+ hybrid (BM25 + vector) retriever
        # can be swapped in later without touching this agent's contract
        # or anything that calls it.
        self.retriever = retriever or DeterministicRetriever()

    def run(self, request: RegulatoryResearchInput) -> RegulatoryResearchOutput:
        request.validate()

        chunks = self.retriever.retrieve(
            query_text=request.normalized_text,
            regulator_candidates=request.regulator_candidates or None,
            topic_hint=request.topic_hint,
            top_k=request.top_k,
        )

        if not chunks:
            # Abstain rather than invent a regulation — matches the
            # reliability section of the architecture doc: "If retrieval
            # finds no sufficiently relevant evidence, abstain."
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

        top_score = chunks[0].score
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
